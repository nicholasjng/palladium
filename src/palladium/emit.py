"""jaxpr -> MSL: palladium's core emitter.

Emission model: scalarized SSA over thread-local arrays. One Metal thread
runs one Pallas program instance; `program_id(k)` maps to a component of
`thread_position_in_grid`. Every jaxpr variable of shape S becomes a
thread-local C array of prod(S) elements (scalar for rank 0); every rule
emits plain element loops. Nothing is vectorized.

Blocks must be contiguous in the underlying array: block shape must equal
array shape on every dim but the leading one.

Per-primitive rules live in RULES, keyed by `eqn.primitive.name`,
registered with @rule(...).

`palladium.debug_msl` prints the emitted MSL; `metal_call(...).interpret`
is the CPU oracle.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import string
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Protocol

from jax.core import ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from palladium.trace import BlockInfo, KernelSpec

if TYPE_CHECKING:
    from typing_extensions import TypeIs

__all__ = ["RULES", "CVal", "EmitError", "EmitState", "emit_jaxpr", "emit_msl", "rule"]

# eqn.invars entries are Vars or inline Literals; eqn.outvars are always Vars.
Atom = Var | Literal


# limit compute primitive arity to a maximum of 6.
MAX_PRIMITIVE_ARITY = 6
PRIMITIVE_INVARS = string.ascii_lowercase[:MAX_PRIMITIVE_ARITY]


def _template_fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _unwrapped(expr: str) -> str:
    # The RHS of an assignment is syntactically complete, so one wrapping
    # paren pair is always redundant there; stripping it keeps the emitted
    # text readable. Safe only at assign sites: template parens elsewhere
    # defend operand atomicity (see the ELEMENTWISE comment).
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for i, ch in enumerate(expr):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(expr) - 1:
            # The leading paren closes early: shapes like (a) * (b).
            return expr
    return expr[1:-1]


def _shaped(aval: object) -> ShapedArray:
    # Invariant, not a hope: every non-Ref value in a Pallas kernel jaxpr
    # is shaped, and Refs never pass through declare()/val().
    assert isinstance(aval, ShapedArray), aval
    return aval


CTYPES = {
    "float32": "float",
    "float16": "half",
    "bfloat16": "bfloat",
    "int32": "int",
    "uint32": "uint",
    "bool": "bool",
}

_PID = ("_pid.x", "_pid.y", "_pid.z")


class EmitError(Exception):
    """The emitter cannot lower this kernel (unsupported or invalid input).

    Distinct from NotImplementedError, which marks a missing rule.
    """


@dataclasses.dataclass(frozen=True)
class CVal:
    """A jaxpr atom lowered to C.

    Attributes
    ----------
    expr : str
        C identifier, literal, or expression; any valid rvalue.
    shape : tuple of int
        Logical shape; `()` for scalars.
    ctype : str
        C type name, always a value of CTYPES (`"float"`, not `"float32"`).
    """

    expr: str
    shape: tuple[int, ...]
    ctype: str

    @property
    def size(self) -> int:
        """Element count; 1 for scalars (`math.prod(()) == 1`)."""
        return math.prod(self.shape)

    def at(self, index: str) -> str:
        """Element access that absorbs rank-0/rank-N mixing.

        Scalars ignore the index; arrays apply it. This is all the
        broadcasting the emitter does.

        Parameters
        ----------
        index : str
            C expression for the element index; ignored for scalars.

        Returns
        -------
        str
            `expr` for scalars, `expr[index]` for arrays.
        """
        return self.expr if not self.shape else f"{self.expr}[{index}]"


class EmitState:
    """Emission state: MSL lines and the jaxpr-var -> CVal environment.

    Attributes
    ----------
    lines : list of str
        Emitted MSL body lines, indented.
    env : dict
        Maps jaxpr Vars to their CVals.
    indent : int
        Current indentation depth, managed by `block`.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.env: dict[Var, CVal] = {}
        self._names = itertools.count()
        self.indent = 1

    def emit(self, line: str) -> None:
        """Append one MSL line at the current indentation."""
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        """Return a new unique C identifier; deterministic per process order."""
        return f"{prefix}{next(self._names)}"

    def val(self, atom: Atom) -> CVal:
        """Resolve a jaxpr atom to its CVal.

        Vars are looked up in the environment; Literals are formatted in
        place.

        Parameters
        ----------
        atom : Var or Literal
            An equation operand.

        Returns
        -------
        CVal
        """
        if isinstance(atom, Literal):
            ctype = CTYPES[str(_shaped(atom.aval).dtype)]
            v = atom.val
            if math.isinf(v):
                expr = "-INFINITY" if v < 0 else "INFINITY"
            elif math.isnan(v):
                expr = "NAN"
            else:
                expr = f"{float(v)!r}f" if ctype in ("float", "half") else str(int(v))
            return CVal(expr=expr, shape=(), ctype=ctype)
        return self.env[atom]

    def declare(self, var: Var) -> CVal:
        """Emit thread-local storage for `var` and bind it.

        Use for a rule producing a new value; use `bind` for aliasing.

        Parameters
        ----------
        var : Var
            A jaxpr output variable; its aval fixes shape and ctype.

        Returns
        -------
        CVal
            The declared storage, already bound in `env`.
        """
        aval = _shaped(var.aval)
        ctype = CTYPES[str(aval.dtype)]
        shape = tuple(int(d) for d in aval.shape)
        name = self.fresh()
        size = math.prod(shape)
        self.emit(f"{ctype} {name}[{size}];" if shape else f"{ctype} {name};")
        cval = CVal(expr=name, shape=shape, ctype=ctype)
        self.env[var] = cval
        return cval

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Bind `var` to an existing CVal: aliasing, no declaration.

        Returns
        -------
        CVal
            `cval`, unchanged, for chaining.
        """
        self.env[var] = cval
        return cval

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        """Emit a braced, indented block: `with state.block("for (...)"):`."""
        self.emit(header + " {")
        self.indent += 1
        yield
        self.indent -= 1
        self.emit("}")

    def copy(self, dst: CVal, src: CVal, count: int) -> None:
        """Emit `count` element assignments dst[i] = src[i] as a loop."""
        i = self.fresh("_i")
        with self.block(f"for (uint {i} = 0; {i} < {count}; ++{i})"):
            self.emit(f"{dst.at(i)} = {src.at(i)};")


RuleFn = Callable[[EmitState, JaxprEqn], None]

RULES: dict[str, RuleFn] = {}


def rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    """Register a lowering rule for one or more primitive names.

    Parameters
    ----------
    *names : str
        Primitive names (`eqn.primitive.name`) the rule handles.

    Returns
    -------
    callable
        Decorator returning the rule unchanged.
    """

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(state: EmitState, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr, dispatching each equation to RULES.

    Nested jaxprs (scan bodies, index maps) recurse through this same
    entry point.

    Parameters
    ----------
    state : EmitState
        Emission state; lines are appended in place.
    jaxpr : Jaxpr
        The (sub-)jaxpr to lower.
    in_vals : list of CVal
        Bindings for `jaxpr.invars`, in order.

    Returns
    -------
    list of CVal
        The values of `jaxpr.outvars`.

    Raises
    ------
    EmitError
        If the jaxpr captures arrays as constvars.
    NotImplementedError
        If an equation has no registered rule.
    """
    if jaxpr.constvars:
        raise EmitError(
            "kernel jaxpr has constvars (captured arrays); close over Python "
            "scalars only, or pass arrays as kernel operands"
        )
    for var, cval in zip(jaxpr.invars, in_vals, strict=True):
        state.env[var] = cval
    for eqn in jaxpr.eqns:
        impl = RULES.get(eqn.primitive.name)
        if impl is None:
            raise NotImplementedError(
                f"no MSL rule for primitive '{eqn.primitive.name}'; add one "
                "with @rule(...) in emit.py (planned coverage: ROADMAP)"
            )
        impl(state, eqn)
    return [state.val(v) for v in jaxpr.outvars]


def _check_contiguous(info: BlockInfo, what: str) -> None:
    # Any partial trailing dim makes the block strided in memory, which
    # the flat pointer-plus-offset model cannot express.
    full_block = _full_block_shape(info)
    if any(b != a for b, a in zip(full_block[1:], info.array_shape[1:], strict=True)):
        raise EmitError(
            f"{what}: block {info.block_shape} of array {info.array_shape} is "
            "not contiguous; only the leading block dim may be partial"
        )


def _constant_offset(info: BlockInfo) -> str | None:
    """Fold index maps with no inputs (gridless or constant) to an offset."""
    imj = info.index_map_jaxpr.jaxpr
    if imj.invars or imj.eqns:
        return None
    strides = _element_strides(info.array_shape)
    full_block = _full_block_shape(info)
    off = 0
    for o, b, s in zip(imj.outvars, full_block, strides, strict=True):
        # No invars and no eqns leaves only inline constants as outputs.
        assert isinstance(o, Literal), o
        off += int(o.val) * b * s
    return str(off)


def _element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[d + 1 :]) for d in range(len(shape)))


def _flat_index(terms: list[tuple[str, int]]) -> str:
    """C expression for `sum(var * stride for var, stride in terms)`,
    omitting the `* 1` for a unit stride and any zero-stride term."""
    parts = [var if stride == 1 else f"{var} * {stride}" for var, stride in terms if stride != 0]
    return " + ".join(parts) or "0"


def _full_block_shape(info: BlockInfo) -> tuple[int, ...]:
    """block_shape with squeezed dims restored as 1, rank-matched to array."""
    missing = len(info.array_shape) - len(info.block_shape)
    return (1,) * missing + info.block_shape


def emit_msl(spec: KernelSpec, kernel_name: str | None = None) -> str:
    """Assemble the full MSL source for a KernelSpec.

    Signature convention (relied on by `dispatch.bind`): operands in
    jaxpr order (inputs then outputs) bound to `[[buffer(k)]]`, then
    `uint3 _pid [[thread_position_in_grid]]`.

    Parameters
    ----------
    spec : KernelSpec
        Traced kernel, from `palladium.trace`.
    kernel_name : str, optional
        Overrides `spec.name` as the MSL function name.

    Returns
    -------
    str
        Complete, self-contained MSL source.

    Raises
    ------
    EmitError
        For grids over rank 3, non-contiguous blocks, or scratch operands.
    NotImplementedError
        If the kernel stages a primitive with no registered rule.
    """
    name = kernel_name or spec.name
    if len(spec.grid) > 3:
        raise EmitError(f"grid {spec.grid} has rank > 3; Metal grids are 3D")

    operands = list(spec.inputs) + list(spec.outputs)
    n_in = len(spec.inputs)
    params = []
    for k, info in enumerate(operands):
        _check_contiguous(info, f"operand {k}")
        qual = "device" if k >= n_in else "const device"
        ctype = CTYPES[info.dtype.name]
        params.append(f"{qual} {ctype}* arg{k}_base [[buffer({k})]]")
    params.append("uint3 _pid [[thread_position_in_grid]]")

    state = EmitState()
    ref_vals: list[CVal] = []
    for k, info in enumerate(operands):
        qual = "device" if k >= n_in else "const device"
        ctype = CTYPES[info.dtype.name]
        offset = _constant_offset(info)
        if offset is None:
            offset = _block_offset(state, spec, info)
        state.emit(f"{qual} {ctype}* arg{k} = arg{k}_base + {offset};")

        # access scalar refs (shape == ()) as axis-1 arrays, since all refs are pointers.
        ref_vals.append(
            CVal(expr=f"arg{k}", shape=info.block_shape or (1,), ctype=ctype)
        )

    n_refs = len(operands)
    if len(spec.jaxpr.invars) != n_refs:
        raise EmitError(
            f"kernel has {len(spec.jaxpr.invars)} refs but spec carries "
            f"{n_refs} operands; scratch_shapes are not supported yet"
        )
    emit_jaxpr(state, spec.jaxpr, ref_vals)

    head = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")\n{"
    return "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            head,
            *state.lines,
            "}",
            "",
        ]
    )


class _Slice(Protocol):
    """Structural stand-in for `jax._src.indexing.Slice`: a private class,
    not imported, same reasoning as `trace.py`'s `type(dim).__name__ ==
    "Squeezed"` check. `start` is a plain Python int when static, an atom
    (Var, possibly Literal) when dynamic (`pl.dslice`); `size`/`stride`
    are always static, verified against real jaxprs.
    """

    start: int | Atom
    size: int
    stride: int


class _NDIndexer(Protocol):
    """Structural stand-in for `jax._src.state.indexing.NDIndexer`, same
    reasoning as `_Slice`. One entry in `indices` per ref dimension."""

    shape: tuple[int, ...]
    indices: tuple[Atom | _Slice, ...]


def _is_slice(idx: Atom | _Slice) -> TypeIs[_Slice]:
    """Structural check, not `isinstance`: `_Slice` is a `Protocol` over a
    private class, and checking by class name (not import) is this file's
    established way of staying version-tolerant of JAX internals."""
    return type(idx).__name__ == "Slice"


def _ref_view(state: EmitState, ref: CVal, indexer: _NDIndexer) -> CVal:
    """Resolve an NDIndexer against a Ref's CVal: pointer offset + kept shape.

    A Slice index keeps that dim (contributes to the result's logical
    shape, in order); any other index (a scalar Var/Literal atom) squeezes
    it into the offset. Slice strides must be 1, and only the first kept
    dim may be partial: the same contiguous-trailing-block discipline
    BlockSpec blocks already follow (`_check_contiguous`). A fully-scalar
    indexer (no kept dims) resolves to a dereferenced single element.

    Raises
    ------
    EmitError
        For a non-unit Slice stride, or a partial Slice anywhere but the
        first kept dim (non-contiguous access; unimplemented).
    """
    strides = _element_strides(indexer.shape)
    terms: list[str] = []
    kept: list[tuple[int, int]] = []
    for d, (idx, stride) in enumerate(zip(indexer.indices, strides, strict=True)):
        if _is_slice(idx):
            if idx.stride != 1:
                raise EmitError(
                    f"ref access dim {d}: stride {idx.stride} != 1 is unimplemented"
                )
            kept.append((d, idx.size))
            start = idx.start
            expr = str(start) if isinstance(start, int) else state.val(start).expr
        else:
            expr = state.val(idx).expr
        if expr != "0":
            terms.append(f"{expr} * {stride}")

    for d, size in kept[1:]:
        if size != indexer.shape[d]:
            raise EmitError(
                f"ref access dim {d}: a partial Slice must be the first "
                "kept dim; non-contiguous access is unimplemented"
            )

    offset = " + ".join(terms) or "0"
    if not kept:
        return CVal(expr=f"{ref.expr}[{offset}]", shape=(), ctype=ref.ctype)
    expr = f"({ref.expr} + {offset})" if offset != "0" else ref.expr
    return CVal(expr=expr, shape=tuple(size for _, size in kept), ctype=ref.ctype)


@rule("get")
def _rule_get(state: EmitState, eqn: JaxprEqn) -> None:
    """Load from a Ref: `y = x_ref[...]` (full block) or an indexed access
    like `y = x_ref[i, :]` (non-Slice dims squeeze into the offset, Slice
    dims are kept in the loaded shape).
    """
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[1:])
    src = state.val(eqn.invars[0])
    if indexer_args:
        (indexer,) = indexer_args
        src = _ref_view(state, src, indexer)
    dst = state.declare(eqn.outvars[0])
    state.copy(dst, src, dst.size)


@rule("swap")
def _rule_swap(state: EmitState, eqn: JaxprEqn) -> None:
    """Store to a Ref: `o_ref[...] = y` (full block) or an indexed store
    like `o_ref[i, :] = y`.

    Deviation: the outvar binds to the (possibly indexed) ref view, not a
    pre-store snapshot, so reads of a swap result alias device memory.
    """
    ref, value = eqn.invars[0], eqn.invars[1]
    indexer_args = eqn.params["tree"].unflatten(eqn.invars[2:])
    dst_ref = state.val(ref)
    if indexer_args:
        (indexer,) = indexer_args
        dst_ref = _ref_view(state, dst_ref, indexer)
    stored = state.val(value)
    state.copy(dst_ref, stored, dst_ref.size)
    state.bind(eqn.outvars[0], dst_ref)


@rule("jit")
def _inline_jit(state: EmitState, eqn: JaxprEqn) -> None:
    """Inline a jit-wrapped call by walking its body jaxpr."""
    inner: Jaxpr = eqn.params["jaxpr"]
    body = inner.jaxpr
    if inner.consts:
        raise EmitError("jit with consts is unsupported")

    invals = [state.val(invar) for invar in eqn.invars]
    outvals = emit_jaxpr(state, body, invals)

    for outvar, val in zip(eqn.outvars, outvals, strict=True):
        state.bind(outvar, val)


@rule("random_wrap")
def _rule_random_wrap(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.wrap_key_data`: pure type-level wrap (uint32[2] array
    to a `key<fry>[]`-typed value), no MSL. Alias, like `program_id`."""
    state.bind(eqn.outvars[0], state.val(eqn.invars[0]))


@rule("random_unwrap")
def _rule_random_unwrap(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.key_data`: the inverse of `random_wrap`, equally pure
    aliasing (a `key<fry>[]`-typed value is a uint32[2] array underneath
    the whole time; nothing to convert)."""
    state.bind(eqn.outvars[0], state.val(eqn.invars[0]))


# Threefry-2x32-20, the algorithm `jax._src.random.threefry2x32` runs by default.
# Verified against `jax.random.bits(key, shape, 'uint32')` bit-for-bit:
# rotations, key schedule, counter construction (flat row-major index,
# hi word 0 for any shape under 2**32 elements).
_THREEFRY_ROT0 = (13, 15, 26, 6)
_THREEFRY_ROT1 = (17, 29, 16, 24)


def _emit_threefry2x32(
    state: EmitState, k1: str, k2: str, x1: str, x2: str
) -> tuple[str, str]:
    """Emit uint x0, x1 = threefry2x32(k1, k2, x1, x2); return their names."""
    ks2 = state.fresh("_ks2")
    state.emit(f"uint {ks2} = {k1} ^ {k2} ^ 0x1BD11BDAu;")
    ks = (k1, k2, ks2)

    x0 = state.fresh("_tx0")
    x1n = state.fresh("_tx1")
    state.emit(f"uint {x0} = {x1} + {ks[0]};")
    state.emit(f"uint {x1n} = {x2} + {ks[1]};")

    def apply_round(rot: int) -> None:
        state.emit(f"{x0} = {x0} + {x1n};")
        state.emit(f"{x1n} = ({x1n} << {rot}u) | ({x1n} >> {32 - rot}u);")
        state.emit(f"{x1n} = {x0} ^ {x1n};")

    # 5 groups of 4 rounds, alternating rotation sets; a, b index into ks
    # for the post-group key-schedule addition, n is that round's counter.
    schedule = (
        (_THREEFRY_ROT0, 1, 2, 1),
        (_THREEFRY_ROT1, 2, 0, 2),
        (_THREEFRY_ROT0, 0, 1, 3),
        (_THREEFRY_ROT1, 1, 2, 4),
        (_THREEFRY_ROT0, 2, 0, 5),
    )
    for rots, a, b, n in schedule:
        for r in rots:
            apply_round(r)
        state.emit(f"{x0} = {x0} + {ks[a]};")
        state.emit(f"{x1n} = {x1n} + {ks[b]} + {n}u;")

    return x0, x1n


@rule("random_bits")
def _rule_random_bits(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.bits`: Threefry-2x32-20 counter-based bits, one hash
    per output element. Counter is the element's flat row-major index
    (hi word 0, lo word the index); matches jax's own construction for
    any shape under 2**32 elements, which is every real kernel output.
    """
    if eqn.params["bit_width"] != 32:
        raise NotImplementedError("random_bits: only bit_width=32 is implemented")

    key = state.val(eqn.invars[0])
    dst = state.declare(eqn.outvars[0])
    k1, k2 = key.at("0"), key.at("1")

    idx = state.fresh("_i")
    with state.block(f"for (uint {idx} = 0; {idx} < {dst.size}; ++{idx})"):
        b0, b1 = _emit_threefry2x32(state, k1, k2, "0u", idx)
        state.emit(f"{dst.at(idx)} = {b0} ^ {b1};")


@rule("random_fold_in")
def _rule_random_fold_in(state: EmitState, eqn: JaxprEqn) -> None:
    """`jax.random.fold_in`: a fresh key from `(key, data)`, the same
    Threefry-2x32-20 hash seeded with (0, data) in place of a counter.
    Verified against `jax.random.fold_in`'s `key_data` output. This is
    the per-thread (fold in `program_id`) and per-step (fold in a loop
    index) key derivation an SDE kernel needs so lanes don't share a
    stream; `example 3`'s hand-written MSL notes the same requirement.
    """
    key = state.val(eqn.invars[0])
    data = state.val(eqn.invars[1])
    k1, k2 = key.at("0"), key.at("1")
    b0, b1 = _emit_threefry2x32(state, k1, k2, "0u", data.expr)

    name = state.fresh()
    state.emit(f"uint {name}[2] = {{{b0}, {b1}}};")
    state.bind(eqn.outvars[0], CVal(expr=name, shape=(2,), ctype="uint"))


@rule("reshape")
def _rule_reshape(state: EmitState, eqn: JaxprEqn) -> None:
    """Reshape: row-major reinterpretation of the same flat storage, no
    data movement, since `CVal.at()` already indexes flatly regardless of
    rank. Surfaced by `jax.random.uniform(key, ())`'s internal (1,)-to-()
    reshape (stretch 8), not previously needed.

    Raises
    ------
    NotImplementedError
        If `dimensions` (an axis permutation applied before reshaping)
        is set; that needs a real copy, not just a shape reinterpretation.
    """
    if eqn.params["dimensions"] is not None:
        raise NotImplementedError(
            "reshape with a dimensions permutation is unimplemented"
        )
    src = state.val(eqn.invars[0])
    state.bind(eqn.outvars[0], dataclasses.replace(src, shape=eqn.params["new_sizes"]))


# MSL templates for pure elementwise primitives; {a}/{b} are element expressions.
# The arity check in _rule_elementwise validates every template against its equation.
# fmin/fmax are float-only; integer min/max needs a ctype dispatch, unimplemented.
# The parens are deliberate: CVal.expr admits any rvalue, so a template
# must not assume its operands bind tighter than its own operator.
# Only the assign site may strip them.
ELEMENTWISE: dict[str, str] = {
    # binary
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "min": "fmin({a}, {b})",
    "max": "fmax({a}, {b})",
    "pow": "pow({a}, {b})",
    # unary
    "neg": "-{a}",
    "abs": "fabs({a})",
    "exp": "exp({a})",
    "log": "log({a})",
    "sin": "sin({a})",
    "cos": "cos({a})",
    "sqrt": "sqrt({a})",
    "tanh": "tanh({a})",
    # ternary
    "select_n": "({a} ? {c} : {b})",  # a: predicate (bool), c when true, b when false
    # logical
    "lt": "({a} < {b})",
    "le": "({a} <= {b})",
    "gt": "({b} < {a})",
    "ge": "({b} <= {a})",
    "eq": "({a} == {b})",
    "ne": "({a} != {b})",
    # bitwise, uint operands only.
    "or": "({a} | {b})",
    "shift_right_logical": "({a} >> {b})",
}


def _rule_elementwise(state: EmitState, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Three templates are derived rather than looked up: integer_pow expands
    to repeated multiplication (reciprocal for negative y); convert_element_type
    casts to the output's ctype; broadcast_in_dim is "{a}" (the element
    loop already broadcasts scalars).

    The arity check requires every template field to be filled and every
    operand consumed.
    """

    dst = state.declare(eqn.outvars[0])
    ops = [state.val(v) for v in eqn.invars]
    opname = eqn.primitive.name

    if opname == "integer_pow":
        exp: int = eqn.params["y"]
        body = " * ".join(["{a}"] * abs(exp))
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname == "convert_element_type":
        template = f"(({dst.ctype}){{a}})"
    elif opname == "bitcast_convert_type":
        # Reinterprets bits without a numeric conversion (unlike
        # convert_element_type's cast); Metal's equivalent is as_type<T>.
        template = f"as_type<{dst.ctype}>({{a}})"
    elif opname == "broadcast_in_dim":
        if dst.shape and any(op.shape for op in ops):
            raise EmitError("array-to-array broadcasts are unimplemented")
        template = "{a}"
    elif opname == "select_n":
        if (pred_type := ops[0].ctype) != "bool":
            raise EmitError(
                f"select_n requires predicate of type bool, got {pred_type}"
            )
        template = ELEMENTWISE[opname]
    else:
        template = ELEMENTWISE[opname]

    fields = _template_fields(template)
    expected = set(PRIMITIVE_INVARS[: len(ops)])
    if fields != expected:
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(ops)}")

    def assign(idx: str) -> str:
        inputs = {name: op.at(idx) for name, op in zip(PRIMITIVE_INVARS, ops)}
        return f"{dst.at(idx)} = {_unwrapped(template.format(**inputs))};"

    if dst.shape:
        idx = state.fresh("_i")
        with state.block(f"for (uint {idx} = 0; {idx} < {dst.size}; ++{idx})"):
            state.emit(assign(idx))
    else:
        state.emit(assign("0"))


for _name in [
    *ELEMENTWISE,
    "integer_pow",
    "convert_element_type",
    "bitcast_convert_type",
    "broadcast_in_dim",
]:
    RULES[_name] = _rule_elementwise


@rule("program_id")
def _rule_program_id(state: EmitState, eqn: JaxprEqn) -> None:
    """`pl.program_id(axis)` -> a component of _pid, as a rank-0 int.

    Pure aliasing, no storage or code. The (int) cast keeps index
    arithmetic signed.
    """
    axis: int = eqn.params["axis"]
    state.bind(eqn.outvars[0], CVal(f"(int){_PID[axis]}", (), "int"))


def _block_offset(state: EmitState, spec: KernelSpec, info: BlockInfo) -> str:
    """Element offset of this program instance's block, as a C expression.

    The index map is a jaxpr over grid indices (bound to _pid components),
    recursed through emit_jaxpr. Map outputs are block indices per array
    dim, converted to elements as

        offset = sum(idx[d] * full_block[d] * stride[d] for d in dims)

    Zero-literal terms are elided; falls back to "0" for an all-zero map.
    """
    pid_vals = [CVal(f"(int){_PID[k]}", (), "int") for k in range(len(spec.grid))]
    out_vals = emit_jaxpr(state, info.index_map_jaxpr.jaxpr, pid_vals)
    block_shape = _full_block_shape(info)
    strides = _element_strides(info.array_shape)

    offsets = []
    for val, shape, stride in zip(out_vals, block_shape, strides, strict=True):
        if val.expr == "0":
            continue
        offsets.append(f"{val.expr} * {shape * stride}")

    return " + ".join(offsets) or "0"


@rule("scan")
def _rule_scan(state: EmitState, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` / pure-carry `lax.scan` -> a C for-loop.

    fori_loop stages as scan with no xs/ys, usually with consts ordered
    before carries in both eqn.invars and body.invars. Consts pass through
    unchanged; carries get fresh mutable loop variables bound on the
    outvars.

    Copy-back runs in two phases: scan updates all carries simultaneously,
    so phase 1 snapshots reads that alias other carries into temps
    (skipping self-forward no-ops), then phase 2 overwrites the carries.
    """
    num_carry = len(eqn.outvars)
    num_consts = len(eqn.invars) - num_carry
    length: int = eqn.params["length"]
    body: Jaxpr = eqn.params["jaxpr"]

    if len(body.outvars) != num_carry:
        raise EmitError("lax.scan with stacked ys not supported, use lax.fori_loop")

    reverse: bool = eqn.params.get("reverse", False)
    if reverse:
        raise EmitError("no reverse-mode scan support, use lax.fori_loop")

    const_vals = [state.val(v) for v in eqn.invars[:num_consts]]

    carries = []
    for invar, outvar in zip(eqn.invars[num_consts:], eqn.outvars, strict=True):
        dst = state.declare(outvar)
        state.copy(dst, state.val(invar), dst.size)
        carries.append(dst)

    carry_exprs = {c.expr for c in carries}
    idx = state.fresh("_s")
    with state.block(f"for (uint {idx} = 0; {idx} < {length}; ++{idx})"):
        outs = emit_jaxpr(state, body, const_vals + carries)
        sources = []
        for out, carry in zip(outs, carries, strict=True):
            if out.expr == carry.expr:
                # Self-forward: the value is already in place.
                sources.append(None)
            elif out.expr in carry_exprs:
                # Aliases a different carry: snapshot before any overwrite.
                tmp = CVal(state.fresh(), out.shape, out.ctype)
                state.emit(
                    f"{tmp.ctype} {tmp.expr}[{tmp.size}];"
                    if tmp.shape
                    else f"{tmp.ctype} {tmp.expr};"
                )
                state.copy(tmp, out, out.size)
                sources.append(tmp)
            else:
                # Fresh SSA name: no carry write can clobber it.
                sources.append(out)

        for src, carry in zip(sources, carries, strict=True):
            if src is not None:
                state.copy(carry, src, carry.size)
