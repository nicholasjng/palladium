"""Core emitter machinery.

CVal, Cursor (MSL text position) and Environment (jaxpr Var bindings),
the rule registries, jaxpr walking, ref views, and MSL assembly.
Per-primitive lowering rules live in `rules` (one thread per program
instance).
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import string
from collections.abc import Callable, Iterator
from typing import cast

import jax.experimental.pallas as pl

# Pinned private import: NDIndexer sits in the same module as pl.Slice
# (jax._src.state.indexing) but, unlike Slice/ds/dslice, isn't re-exported
# through jax.experimental.pallas -- an apparent oversight, since it's the
# actual type `eqn.params["tree"].unflatten(...)` hands back for a get/swap
# indexer, not a new kind of API surface. PR filed upstream to close the
# gap (jax/experimental/pallas/__init__.py, alongside the existing `from
# jax._src.state.indexing import Slice as Slice` line); drop this import
# for `from jax.experimental.pallas import NDIndexer` once it lands.
from jax._src.state.indexing import NDIndexer
from jax.core import Atom, ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from palladium.errors import EmitError, UnsupportedPrimitiveError
from palladium.trace import BlockInfo, KernelSpec

__all__ = ["EmitError"]  # re-export; the class lives in palladium.errors


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
    space : str
        Metal address space of the storage behind `expr`: `"thread"` for
        emitter-declared locals, `"device"` for kernel operand refs and
        views of them. Pointer casts must be qualified with this; a
        mis-qualified cast is invalid MSL.
    readonly : bool
        True for `const device` input refs and views of them. Only
        readonly device storage may be aliased instead of copied by
        `get`: nothing can write through the ref, so a view has snapshot
        semantics.
    transposed : bool
        Rank-2 lazy transpose: `expr` is the untransposed source storage
        and `shape` the permuted logical shape, so element `(i, j)` lives
        at `expr[j * shape[0] + i]`. Set only when every consumer is a
        `dot_general` rhs, so no flat-indexing consumer can misread it.
    align : int
        Guaranteed element alignment of the storage behind `expr`: the
        offset from a 16-byte-aligned base is a known multiple of this
        many elements. 0 means exactly aligned (the gcd identity, and the
        de-facto assumption for declared thread locals); a vectorized
        path needing N-element loads requires `align % N == 0`. Composed
        through views as the gcd of the base alignment and every offset
        term's provable multiple.
    """

    expr: str
    shape: tuple[int, ...]
    ctype: str
    space: str = "thread"
    readonly: bool = False
    transposed: bool = False
    align: int = 0

    @property
    def size(self) -> int:
        """Element count; 1 for scalars (`math.prod(()) == 1`)."""
        return math.prod(self.shape)

    def at(self, index: str) -> str:
        """`expr` for scalars, `expr[index]` for arrays: the only
        rank-0/rank-N absorption the emitter does."""
        return self.expr if not self.shape else f"{self.expr}[{index}]"


class Cursor:
    """The write position into one kernel's growing MSL text.

    Owns the emitted lines, current indentation, and the unique-name
    counter; knows nothing about jaxpr Vars or bindings. Reusable on its
    own wherever a rule needs to emit text without touching the
    environment (e.g. `copy`, a pure element-loop codegen helper).
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 1
        self._names = itertools.count()

    def emit(self, line: str) -> None:
        """Append one MSL line at the current indentation."""
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        """Return a new unique C identifier; deterministic per process order."""
        return f"{prefix}{next(self._names)}"

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        """Emit a braced, indented block: `with cursor.block("for (...)"):`."""
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


class Environment:
    """The symbol table for one kernel: jaxpr Var bindings and def-use info.

    Owns no MSL text; knows nothing about indentation or emission order.

    Attributes
    ----------
    bindings : dict
        Maps jaxpr Vars to their lowered CVals.
    consumers, producers : dict
        Var -> consuming eqns (None marks a jaxpr outvar) and
        Var -> defining eqn, for rules that need lookahead.
    """

    def __init__(self) -> None:
        self.bindings: dict[Var, CVal] = {}
        # Var -> its consuming equations at that var's own jaxpr level
        # (None marks "is a jaxpr outvar", i.e. escapes the level).
        # Populated by emit_jaxpr before walking each (sub-)jaxpr; Vars
        # are unique objects per jaxpr, so levels never collide.
        self.consumers: dict[Var, list[JaxprEqn | None]] = {}
        # Var -> the equation that defines it, for rules that need to
        # inspect a not-yet-emitted producer.
        self.producers: dict[Var, JaxprEqn] = {}

    def val(self, atom: Atom) -> CVal:
        """Resolve a jaxpr atom: Vars from bindings, Literals formatted
        in place."""
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
        return self.bindings[atom]

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Bind `var` to an existing CVal: aliasing, no declaration."""
        self.bindings[var] = cval
        return cval


def declare(env: Environment, cursor: Cursor, var: Var) -> CVal:
    """Emit thread-local storage for `var` and bind it in `env`. Use for
    a rule producing a new value; use `env.bind` for aliasing.

    The one operation that genuinely needs both a Cursor (it emits the
    declaration) and an Environment (it binds the result) — everything
    else a rule does is purely one or the other.
    """
    aval = _shaped(var.aval)
    ctype = CTYPES[str(aval.dtype)]
    shape = tuple(int(d) for d in aval.shape)
    name = cursor.fresh()
    size = math.prod(shape)
    cursor.emit(f"{ctype} {name}[{size}];" if shape else f"{ctype} {name};")
    cval = CVal(expr=name, shape=shape, ctype=ctype)
    env.bind(var, cval)
    return cval


RuleFn = Callable[[Environment, Cursor, JaxprEqn], None]

RULES: dict[str, RuleFn] = {}


def rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    """Register a lowering rule for one or more primitive names."""

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def _emit_with_rules(
    env: Environment,
    cursor: Cursor,
    jaxpr: Jaxpr,
    in_vals: list[CVal],
    rules: dict[str, RuleFn],
    missing: Callable[[str], Exception],
) -> list[CVal]:
    """Walk a jaxpr, dispatching each equation through `rules`.

    Shared by both execution models: binds invars, records consumer and
    producer maps for the rules that need lookahead, then emits each
    equation. Nested jaxprs (scan bodies, index maps) recurse through the
    model-specific wrappers.

    Raises
    ------
    EmitError
        If the jaxpr captures arrays as constvars.
    """
    if jaxpr.constvars:
        raise EmitError(
            "kernel jaxpr has constvars (captured arrays); close over Python "
            "scalars only, or pass arrays as kernel operands"
        )
    for var, cval in zip(jaxpr.invars, in_vals, strict=True):
        env.bind(var, cval)
    for eqn in jaxpr.eqns:
        for iv in eqn.invars:
            if isinstance(iv, Var):
                env.consumers.setdefault(iv, []).append(eqn)
        for ov in eqn.outvars:
            env.producers[ov] = eqn
    for ov in jaxpr.outvars:
        if isinstance(ov, Var):
            env.consumers.setdefault(ov, []).append(None)
    for eqn in jaxpr.eqns:
        impl = rules.get(eqn.primitive.name)
        if impl is None:
            raise missing(eqn.primitive.name)
        impl(env, cursor, eqn)
    return [env.val(v) for v in jaxpr.outvars]


def emit_jaxpr(
    env: Environment, cursor: Cursor, jaxpr: Jaxpr, in_vals: list[CVal]
) -> list[CVal]:
    """Walk a jaxpr with the one-thread-per-instance rules (RULES).

    Parameters
    ----------
    env : Environment
        Var bindings and def-use info; updated in place.
    cursor : Cursor
        MSL text position; lines are appended in place.
    jaxpr : Jaxpr
        The (sub-)jaxpr to lower.
    in_vals : list of CVal
        Bindings for `jaxpr.invars`, in order.

    Returns
    -------
    list of CVal
        The values of `jaxpr.outvars`.
    """
    return _emit_with_rules(
        env,
        cursor,
        jaxpr,
        in_vals,
        RULES,
        lambda name: UnsupportedPrimitiveError(
            f"no MSL rule for primitive '{name}'; add one with @rule(...)"
        ),
    )


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
    parts = [
        var if stride == 1 else f"{var} * {stride}"
        for var, stride in terms
        if stride != 0
    ]
    return " + ".join(parts) or "0"


def _full_block_shape(info: BlockInfo) -> tuple[int, ...]:
    """block_shape with squeezed dims restored as 1, rank-matched to array."""
    missing = len(info.array_shape) - len(info.block_shape)
    return (1,) * missing + info.block_shape


def emit_msl(spec: KernelSpec, kernel_name: str | None = None) -> str:
    """Assemble the full MSL source for a KernelSpec.

    Signature convention (relied on by `dispatch.bind`): operands in
    jaxpr order (inputs then outputs) bound to `[[buffer(k)]]`, then
    `uint3 _pid [[thread_position_in_grid]]`, one thread per program
    instance.

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
    UnsupportedPrimitiveError
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

    env = Environment()
    cursor = Cursor()
    ref_vals: list[CVal] = []
    for k, info in enumerate(operands):
        qual = "device" if k >= n_in else "const device"
        ctype = CTYPES[info.dtype.name]
        offset = _constant_offset(info)
        if offset is None:
            offset = _block_offset(env, cursor, spec, info)
        cursor.emit(f"{qual} {ctype}* arg{k} = arg{k}_base + {offset};")

        # access scalar refs (shape == ()) as axis-1 arrays, since all refs are pointers.
        ref_vals.append(
            CVal(
                expr=f"arg{k}",
                shape=info.block_shape or (1,),
                ctype=ctype,
                space="device",
                readonly=k < n_in,
                align=_block_offset_align(spec, info),
            )
        )

    n_refs = len(operands)
    if len(spec.jaxpr.invars) != n_refs:
        raise EmitError(
            f"kernel has {len(spec.jaxpr.invars)} refs but spec carries "
            f"{n_refs} operands; scratch_shapes are not supported yet"
        )
    emit_jaxpr(env, cursor, spec.jaxpr, ref_vals)

    head = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")\n{"
    return "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            head,
            *cursor.lines,
            "}",
            "",
        ]
    )


def _ref_view(env: Environment, ref: CVal, indexer: NDIndexer) -> CVal:
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
    align = ref.align
    for d, (idx, stride) in enumerate(zip(indexer.indices, strides, strict=True)):
        if isinstance(idx, pl.Slice):
            if idx.stride != 1:
                raise EmitError(
                    f"ref access dim {d}: stride {idx.stride} != 1 is unimplemented"
                )
            # pl.Slice.size/.start are typed `int | Array` upstream (a
            # Slice can be user-constructed with a dynamic size), but a
            # kernel jaxpr equation never carries one: size is always
            # static (verified against real jaxprs) and a dynamic start
            # is always a jaxpr atom (Var/Literal), never a live Array.
            kept.append((d, cast(int, idx.size)))
            start = idx.start
            expr = (
                str(start)
                if isinstance(start, int)
                else env.val(cast(Atom, start)).expr
            )
        else:
            # NDIndexer.indices is typed to also admit fancy/advanced
            # (gather-style) integer-array indexers upstream, but a
            # kernel jaxpr's get/swap params never carry one: this
            # emitter's supported subset has no such rule, and a
            # non-Slice index here is always a jaxpr atom.
            expr = env.val(cast(Atom, idx)).expr
        if expr != "0":
            terms.append(f"{expr} * {stride}")
            # A dynamic index contributes its stride as the provable
            # multiple; a literal index contributes its exact offset.
            lit = expr.lstrip("-").isdigit()
            align = math.gcd(align, int(expr) * stride if lit else stride)

    for d, size in kept[1:]:
        if size != indexer.shape[d]:
            raise EmitError(
                f"ref access dim {d}: a partial Slice must be the first "
                "kept dim; non-contiguous access is unimplemented"
            )

    offset = " + ".join(terms) or "0"
    if not kept:
        return CVal(
            expr=f"{ref.expr}[{offset}]",
            shape=(),
            ctype=ref.ctype,
            space=ref.space,
            readonly=ref.readonly,
        )
    expr = f"({ref.expr} + {offset})" if offset != "0" else ref.expr
    return CVal(
        expr=expr,
        shape=tuple(size for _, size in kept),
        ctype=ref.ctype,
        space=ref.space,
        readonly=ref.readonly,
        align=align,
    )


def _transpose_is_dot_rhs_only(
    consumers: dict[Var, list[JaxprEqn | None]], eqn: JaxprEqn
) -> bool:
    """Whether this rank-2 `(1, 0)` transpose is consumed *only* as the
    rhs of dot_general equations (never as lhs, never escaping as a
    jaxpr outvar), i.e. safe to lower as a lazy `transposed` CVal that
    only dot_general knows how to index."""
    if tuple(eqn.params["permutation"]) != (1, 0):
        return False
    outvar = eqn.outvars[0]
    uses = consumers.get(outvar, [])
    return bool(uses) and all(
        use is not None
        and use.primitive.name == "dot_general"
        and use.invars[1] is outvar
        and use.invars[0] is not outvar
        for use in uses
    )


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
    "clamp": "clamp({b}, {a}, {c})",  # jaxpr order (min, x, max) -> metal (x, min, max)
    # logical
    "lt": "({a} < {b})",
    "le": "({a} <= {b})",
    "gt": "({b} < {a})",
    "ge": "({b} <= {a})",
    "eq": "({a} == {b})",
    "ne": "({a} != {b})",
    # bitwise, uint operands only.
    "and": "({a} & {b})",
    "or": "({a} | {b})",
    "xor": "({a} ^ {b})",
    "shift_right_logical": "({a} >> {b})",
}


def _block_offset(
    env: Environment, cursor: Cursor, spec: KernelSpec, info: BlockInfo
) -> str:
    """Element offset of this program instance's block, as a C expression.

    The index map is a jaxpr over grid indices (bound to _pid components),
    recursed through emit_jaxpr. Map outputs are block indices per array
    dim, converted to elements as

        offset = sum(idx[d] * full_block[d] * stride[d] for d in dims)

    Zero-literal terms are elided; falls back to "0" for an all-zero map.
    """
    pid_vals = [CVal(f"(int){_PID[k]}", (), "int") for k in range(len(spec.grid))]
    out_vals = emit_jaxpr(env, cursor, info.index_map_jaxpr.jaxpr, pid_vals)
    block_shape = _full_block_shape(info)
    strides = _element_strides(info.array_shape)

    offsets = []
    for val, shape, stride in zip(out_vals, block_shape, strides, strict=True):
        if val.expr == "0":
            continue
        offsets.append(f"{val.expr} * {shape * stride}")

    return " + ".join(offsets) or "0"


def _block_offset_align(spec: KernelSpec, info: BlockInfo) -> int:
    """Guaranteed element alignment of this operand's per-instance block
    offset, whatever the grid indices are: the gcd of every offset term's
    provable multiple (exact for constant index maps). Feeds `CVal.align`
    for the ref; 0 means exactly aligned."""
    off = _constant_offset(info)
    if off is not None:
        return int(off)
    block_shape = _full_block_shape(info)
    strides = _element_strides(info.array_shape)
    align = 0
    for b, s in zip(block_shape, strides, strict=True):
        align = math.gcd(align, b * s)
    return align
