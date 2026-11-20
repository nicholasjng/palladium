"""Index-generating and array-assembling primitives: iota, cumulative
reductions, concatenate, and dynamic_slice.

Each materializes its result with one loop nest per program instance; the
sources are read through `CVal.at`, so fused or scalar operands compose.
"""

from __future__ import annotations

from jax.extend.core import JaxprEqn

from palladium.emit.addressing import element_strides, flat_index
from palladium.emit.core import (
    REGISTER_BYTES,
    Cursor,
    CVal,
    Environment,
    declare,
    msl_type,
    rule,
    shaped,
)
from palladium.emit.numeric import extremum, extremum_identity
from palladium.emit.rules.control import store_or_declare
from palladium.emit.rules.elementwise import reads_by_flat_index
from palladium.errors import EmitError

_CUMULATIVE = {
    "cumsum": ("({a} + {b})", "0"),
    "cumprod": ("({a} * {b})", "1"),
    "cummax": ("max", None),
    "cummin": ("min", None),
}


@rule("iota")
def _rule_iota(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.broadcasted_iota` / `jnp.arange`: the index along one dimension.

    Read only by flat index, it is a lazy expression of that index.
    """
    dimension = int(eqn.params["dimension"])
    aval = shaped(eqn.outvars[0].aval)
    shape = tuple(int(d) for d in aval.shape)
    if shape and reads_by_flat_index(env, eqn.outvars[0]):
        stride = element_strides(shape)[dimension]
        coordinate = "$i" if stride == 1 else f"($i / {stride})"
        if dimension:
            coordinate = f"({coordinate} % {shape[dimension]})"
        ctype = msl_type(aval.dtype)
        env.bind(eqn.outvars[0], CVal("", shape, ctype, lazy=f"(({ctype}){coordinate})"))
        return
    dst = declare(env, cursor, eqn.outvars[0])
    strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_o") as idx:
        flat = flat_index(list(zip(idx, strides, strict=True)))
        cursor.emit(f"{dst.at(flat)} = ({dst.ctype}){idx[dimension]};")


def _rule_cumulative(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Serial prefix scan along one axis for every other index."""
    src = env.val(eqn.invars[0])
    dst = store_or_declare(env, cursor, eqn.outvars[0], min_bytes=REGISTER_BYTES)
    axis = int(eqn.params["axis"])
    reverse = bool(eqn.params.get("reverse", False))
    template, identity = _CUMULATIVE[eqn.primitive.name]
    if template in ("max", "min"):
        identity = extremum_identity(template, dst.ctype)
    strides = element_strides(dst.shape)
    outer_shape = tuple(size for d, size in enumerate(dst.shape) if d != axis)
    with cursor.loop_nest(outer_shape, "_o") as outer:
        acc = cursor.fresh("_acc")
        cursor.emit(f"{dst.ctype} {acc} = {identity};")
        with cursor.loop(dst.shape[axis], "_k", reverse=reverse) as k:
            idx = list(outer)
            idx.insert(axis, k)
            flat = flat_index(list(zip(idx, strides, strict=True)))
            value = src.at(flat)
            if src.lazy is not None:
                # A fused input is an expression: evaluate it once.
                value_name = cursor.fresh("_x")
                cursor.emit(f"{src.ctype} {value_name} = {value};")
                value = value_name
            if template in ("max", "min"):
                combined = extremum(template, dst.ctype, acc, value)
            else:
                combined = template.format(a=acc, b=value)
            cursor.emit(f"{acc} = {combined};")
            cursor.emit(f"{dst.at(flat)} = {acc};")


rule(*_CUMULATIVE)(_rule_cumulative)


@rule("concatenate")
def _rule_concatenate(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jnp.concatenate`: copy each operand at its offset along the dimension."""
    dst = declare(env, cursor, eqn.outvars[0])
    dimension = int(eqn.params["dimension"])
    dst_strides = element_strides(dst.shape)
    offset = 0
    for atom in eqn.invars:
        src = env.val(atom)
        if len(src.shape) != len(dst.shape):
            raise EmitError("concatenate operands must share the output rank")
        src_strides = element_strides(src.shape)
        with cursor.loop_nest(src.shape, "_c") as idx:
            src_flat = flat_index(list(zip(idx, src_strides, strict=True)))
            shifted = list(idx)
            if offset:
                shifted[dimension] = f"({idx[dimension]} + {offset})"
            dst_flat = flat_index(list(zip(shifted, dst_strides, strict=True)))
            cursor.emit(f"{dst.at(dst_flat)} = {src.at(src_flat)};")
        offset += src.shape[dimension]


@rule("dynamic_slice")
def _rule_dynamic_slice(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.dynamic_slice`: a window at runtime start indices, clamped so
    the whole window stays in bounds, as JAX defines it."""
    src = env.val(eqn.invars[0])
    starts = [env.val(atom) for atom in eqn.invars[1:]]
    dst = declare(env, cursor, eqn.outvars[0])
    sizes = tuple(int(s) for s in eqn.params["slice_sizes"])
    if len(starts) != len(src.shape) or sizes != dst.shape:
        raise EmitError("dynamic_slice needs one start index per operand dimension")
    if src.transposed:
        raise EmitError("dynamic_slice of a lazily transposed value is unsupported")
    origins = []
    for d, (start, size) in enumerate(zip(starts, sizes, strict=True)):
        origin = cursor.fresh("_start")
        cursor.emit(f"const int {origin} = clamp((int){start.expr}, 0, {src.shape[d] - size});")
        origins.append(origin)
    src_strides = element_strides(src.shape)
    dst_strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_s") as idx:
        src_flat = flat_index(
            [(f"({i} + {origin})", stride) for i, origin, stride in zip(idx, origins, src_strides)]
        )
        dst_flat = flat_index(list(zip(idx, dst_strides, strict=True)))
        cursor.emit(f"{dst.at(dst_flat)} = {src.at(src_flat)};")


@rule("rev")
def _rule_rev(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jnp.flip`: the operand read backwards along the given dimensions."""
    src = env.val(eqn.invars[0])
    dst = declare(env, cursor, eqn.outvars[0])
    reversed_dims = tuple(eqn.params["dimensions"])
    strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_r") as idx:
        mirrored = [
            f"({n - 1} - {i})" if d in reversed_dims else i
            for d, (i, n) in enumerate(zip(idx, dst.shape, strict=True))
        ]
        src_flat = flat_index(list(zip(mirrored, strides, strict=True)))
        dst_flat = flat_index(list(zip(idx, strides, strict=True)))
        cursor.emit(f"{dst.at(dst_flat)} = {src.at(src_flat)};")


@rule("pad")
def _rule_pad(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`lax.pad`: each output element is the operand's where its position maps
    onto one, and the padding value on the border and between interior-padded
    elements. Negative padding crops."""
    src, value = env.val(eqn.invars[0]), env.val(eqn.invars[1])
    dst = declare(env, cursor, eqn.outvars[0])
    src_strides = element_strides(src.shape)
    dst_strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_p") as idx:
        conditions, coords = [], []
        for i, n, (low, _, interior) in zip(idx, src.shape, eqn.params["padding_config"]):
            shifted, step = f"((int){i} - ({int(low)}))", int(interior) + 1
            conditions += [f"{shifted} >= 0", f"{shifted} <= {(n - 1) * step}"]
            if step > 1:
                conditions.append(f"{shifted} % {step} == 0")
            coords.append(f"({shifted} / {step})" if step > 1 else shifted)
        src_flat = flat_index(list(zip(coords, src_strides, strict=True)))
        dst_flat = flat_index(list(zip(idx, dst_strides, strict=True)))
        inside = " && ".join(conditions) or "true"
        cursor.emit(f"{dst.at(dst_flat)} = ({inside}) ? {src.at(src_flat)} : {value.at('0')};")


@rule("argmax", "argmin")
def _rule_argmax(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jnp.argmax` / `argmin` along one axis: the first index of the extremum,
    or of the first NaN, as JAX defines them."""
    src = env.val(eqn.invars[0])
    (axis,) = eqn.params["axes"]
    dst = declare(env, cursor, eqn.outvars[0])
    floating = src.ctype in ("float", "half", "bfloat")
    compare = ">" if eqn.primitive.name == "argmax" else "<"
    src_strides = element_strides(src.shape)
    dst_strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_a") as idx:
        best, at = cursor.fresh("best"), cursor.fresh("at")

        def element(k: str) -> str:
            coords = [*idx[:axis], k, *idx[axis:]]
            return src.at(flat_index(list(zip(coords, src_strides, strict=True))))

        cursor.emit(f"{src.ctype} {best} = {element('0')};")
        cursor.emit(f"{dst.ctype} {at} = 0;")
        with cursor.loop(src.shape[axis], "_k") as k:
            candidate = cursor.fresh("v")
            better = f"{candidate} {compare} {best}"
            if floating:
                better = f"({better} || (isnan({candidate}) && !isnan({best})))"
            cursor.emit(f"{src.ctype} {candidate} = {element(k)};")
            cursor.emit(f"if ({better}) {{ {best} = {candidate}; {at} = ({dst.ctype}){k}; }}")
        target = (
            dst.at(flat_index(list(zip(idx, dst_strides, strict=True)))) if dst.shape else dst.expr
        )
        cursor.emit(f"{target} = {at};")


@rule("slice")
def _rule_slice(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`x[a:b:s]` on a value: the elements at static starts and strides."""
    src = env.val(eqn.invars[0])
    dst = declare(env, cursor, eqn.outvars[0])
    starts = eqn.params["start_indices"]
    steps = eqn.params["strides"] or (1,) * len(starts)
    src_strides = element_strides(src.shape)
    dst_strides = element_strides(dst.shape)
    with cursor.loop_nest(dst.shape, "_s") as idx:
        coords = [
            f"({int(a)} + {i} * {int(t)})" for i, a, t in zip(idx, starts, steps, strict=True)
        ]
        src_flat = flat_index(list(zip(coords, src_strides, strict=True)))
        dst_flat = flat_index(list(zip(idx, dst_strides, strict=True)))
        cursor.emit(f"{dst.at(dst_flat)} = {src.at(src_flat)};")


@rule("copy", "copy_p", "stop_gradient")
def _rule_identity(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """Values are immutable once bound, so a copy is the value itself, and a
    kernel has no gradient to stop."""
    env.bind(eqn.outvars[0], env.val(eqn.invars[0]))
