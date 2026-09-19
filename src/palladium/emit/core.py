"""Core emitter machinery.

CVal, Cursor (MSL text position) and Environment (jaxpr Var bindings),
the rule registries, jaxpr walking, ref views, and MSL assembly.
Per-primitive lowering rules live in `rules` (one thread per program instance).
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import string
from collections.abc import Callable, Iterator
from typing import Literal as TLiteral, cast

import jax.experimental.pallas as pl
from jax._src.state.indexing import NDIndexer
from jax.core import Atom, ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from palladium.errors import EmitError, UnsupportedPrimitiveError
from palladium.trace import BlockInfo, KernelSpec

__all__ = ["EmitError"]


MAX_PRIMITIVE_ARITY = 6
PRIMITIVE_INVARS = string.ascii_lowercase[:MAX_PRIMITIVE_ARITY]


def _template_fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _unwrapped(expr: str) -> str:
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


def shaped(aval: object) -> ShapedArray:
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

# Size of each CTYPES value, for the per-thread stack estimate in
# `Cursor.account`. MSL bool is one byte.
CTYPE_BYTES = {
    "float": 4,
    "half": 2,
    "bfloat": 2,
    "int": 4,
    "uint": 4,
    "bool": 1,
}

_PID = ("_pid.x", "_pid.y", "_pid.z")
_TID = "_tid.x"
_TPT = "_tpt.x"


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
        emitter-declared locals, `"threadgroup"` for per-threadgroup
        locals, `"device"` for operand refs and views of them. Pointer
        casts must carry it; a mis-qualified cast is invalid MSL.
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
        Guaranteed element alignment: the offset from a 16-byte-aligned
        base is a known multiple of this many elements. 0 means exactly
        aligned (the gcd identity, assumed for declared thread locals);
        an N-element vectorized load requires `align % N == 0`. Composed
        through views as the gcd of the base alignment and every offset
        term's provable multiple.
    """

    expr: str
    shape: tuple[int, ...]
    ctype: str
    space: TLiteral["thread", "threadgroup", "device"] = "thread"
    readonly: bool = False
    transposed: bool = False
    align: int = 0

    @property
    def size(self) -> int:
        """Element count; 1 for scalars (`math.prod(()) == 1`)."""
        return math.prod(self.shape)

    def slot(self, index: str, shape: tuple[int, ...]) -> CVal:
        """A view of `shape` at slot `index`: element offset
        `index * prod(shape)` into this storage. Space, readonly, and
        ctype carry over; alignment composes as gcd with the slot size.
        """
        assert self.shape and not self.transposed, self
        size = math.prod(shape)
        return dataclasses.replace(
            self,
            expr=f"({self.expr} + {index} * {size})",
            shape=shape,
            align=math.gcd(self.align, size),
        )

    def at(self, index: str) -> str:
        """`expr` for scalars, `expr[index]` for arrays: the only
        rank-0/rank-N absorption the emitter does."""
        return self.expr if not self.shape else f"{self.expr}[{index}]"


@dataclasses.dataclass(frozen=True)
class EmitStats:
    """Per-program-instance storage one emitted kernel declares.

    Attributes
    ----------
    thread_bytes : int
        Bytes of `thread`-space storage: loaded blocks, intermediates,
        scratch. Bounded by the per-thread stack, for which Metal
        publishes no figure; shrink this when pipeline creation fails.
    threadgroup_bytes : int
        Bytes of `threadgroup`-space storage, shared per group. Budget:
        `metal_runtime.device_info()["max_threadgroup_memory_length"]`.

    Neither figure is liveness-aware: MSL declarations are function
    scoped, and Metal's own pipeline check is equally conservative.
    """

    thread_bytes: int
    threadgroup_bytes: int


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
        # Running total of thread-local bytes declared, and the separate
        # threadgroup-space total. Not liveness-aware: MSL declarations
        # are function-scoped and Metal's own pipeline check is equally
        # conservative, so this over-counts exactly where Metal does.
        self.thread_bytes = 0
        self.threadgroup_bytes = 0

    def account(self, ctype: str, count: int, space: str = "thread") -> None:
        """Record `count` elements of `ctype` declared in `space`."""
        nbytes = CTYPE_BYTES[ctype] * max(count, 1)
        if space == "threadgroup":
            self.threadgroup_bytes += nbytes
        else:
            self.thread_bytes += nbytes

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

    @contextlib.contextmanager
    def loop(
        self, count: int | str, prefix: str = "_i", reverse: bool = False
    ) -> Iterator[str]:
        """Emit a counted for-loop over `[0, count)`; yields the index name.

        Ascending by default: `for (uint idx = 0; idx < count; ++idx)`.
        With `reverse=True`, descending from `count - 1` to 0 with a
        *signed* index -- a uint would wrap past zero instead of failing
        `>= 0`, looping forever.

        `count` is inlined verbatim, so pass e.g. `f"{n}u"` where the
        call site needs an unsigned-literal suffix. An int `count` folds
        `count - 1` at emit time so the reverse header carries a literal
        bound rather than an expression.
        """
        idx = self.fresh(prefix)
        if reverse:
            first = count - 1 if isinstance(count, int) else f"{count} - 1"
            header = f"for (int {idx} = {first}; {idx} >= 0; --{idx})"
        else:
            header = f"for (uint {idx} = 0; {idx} < {count}; ++{idx})"
        with self.block(header):
            yield idx

    def copy(self, dst: CVal, src: CVal, count: int) -> None:
        """Emit `count` element assignments dst[i] = src[i] as a loop."""
        with self.loop(count) as i:
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

    def __init__(self, no_stream_refs: frozenset[Var] = frozenset()) -> None:
        self.bindings: dict[Var, CVal] = {}
        # Refs sharing a buffer with another ref (input_output_aliases):
        # never scan-ys streaming targets, since reads through the twin
        # var are invisible here.
        self.no_stream_refs = no_stream_refs
        # Per-emission scratch for rules that memoize per-eqn analyses,
        # conventionally keyed ("name", id(eqn)).
        self.rule_cache: dict[object, object] = {}
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
            ctype = CTYPES[str(shaped(atom.aval).dtype)]
            v = atom.val
            if math.isinf(v):
                expr = "-INFINITY" if v < 0 else "INFINITY"
            elif math.isnan(v):
                expr = "NAN"
            else:
                expr = (
                    f"{float(v)!r}f"
                    if ctype in ("float", "half", "bfloat")
                    else str(int(v))
                )
            if ctype == "bfloat":
                # MSL does not implicitly narrow a float expression to
                # bfloat. Keep literals typed so arithmetic stays bfloat.
                expr = f"bfloat({expr})"
            return CVal(expr=expr, shape=(), ctype=ctype)
        return self.bindings[atom]

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Bind `var` to an existing CVal: aliasing, no declaration."""
        self.bindings[var] = cval
        return cval

    def consumer_eqns(self, var: Var) -> list[JaxprEqn]:
        """Consuming equations at `var`'s own jaxpr level, without the
        None outvar marker (see `escapes`)."""
        return [e for e in self.consumers.get(var, []) if e is not None]

    def escapes(self, var: Var) -> bool:
        """Whether `var` is an outvar of its (sub-)jaxpr level."""
        return None in self.consumers.get(var, ())

    def sole_consumer(self, var: Var) -> JaxprEqn | None:
        """The single consuming equation, or None when `var` escapes,
        is unused, or has several consumers."""
        uses = self.consumers.get(var, [])
        if len(uses) == 1 and uses[0] is not None:
            return uses[0]
        return None


def declare(env: Environment, cursor: Cursor, var: Var) -> CVal:
    """Emit thread-local storage for `var` and bind it in `env`. Use for
    a rule producing a new value; use `env.bind` for aliasing.

    The one operation that genuinely needs both a Cursor (it emits the
    declaration) and an Environment (it binds the result) — everything
    else a rule does is purely one or the other.
    """
    aval = shaped(var.aval)
    ctype = CTYPES[str(aval.dtype)]
    shape = tuple(int(d) for d in aval.shape)
    name = cursor.fresh()
    size = math.prod(shape)
    cursor.account(ctype, size)
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
            f"no MSL rule for primitive '{name}'; add one with @rule(...)",
            primitive=name,
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


def emit_msl_stats(
    spec: KernelSpec, kernel_name: str | None = None
) -> tuple[str, EmitStats]:
    """Assemble the full MSL source for a KernelSpec, with its storage stats.

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
    tuple of (str, EmitStats)
        Complete, self-contained MSL source, and the per-instance
        storage the kernel declares.

    Raises
    ------
    EmitError
        For grids over rank 3, or non-contiguous blocks.
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
    if any(info.space == "threadgroup" for info in spec.scratch):
        params.append("uint3 _tid [[thread_position_in_threadgroup]]")
        params.append("uint3 _tpt [[threads_per_threadgroup]]")

    env = Environment(
        no_stream_refs=frozenset(
            spec.jaxpr.invars[k] for i, j in spec.aliases for k in (i, n_in + j)
        )
    )
    cursor = Cursor()
    ref_vals: list[CVal] = []
    # Aliased inputs share their buffer with an output; dropping readonly
    # forces gets to copy instead of binding a view that would observe
    # the in-place write.
    aliased_inputs = {i for i, _ in spec.aliases}
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
                readonly=k < n_in and k not in aliased_inputs,
                align=_block_offset_align(spec, info),
            )
        )

    for k, info in enumerate(spec.scratch):
        ctype = CTYPES[info.dtype.name]
        shape = info.shape
        size = math.prod(info.shape)
        # Both spaces are compile-time-sized local arrays; only the
        # qualifier differs. MSL requires threadgroup variables at kernel
        # scope, which is where these already land.
        cursor.account(ctype, size, info.space)
        scratch_op = f"{info.space} {ctype} scratch{k}"
        scratch_op += f"[{size}];" if shape else ";"
        cursor.emit(scratch_op)
        ref_vals.append(
            CVal(
                expr=f"scratch{k}",
                shape=shape,
                ctype=ctype,
                space=info.space,
                readonly=False,
                align=0,
            )
        )

    n_refs = len(operands) + len(spec.scratch)
    if len(spec.jaxpr.invars) != n_refs:
        raise EmitError(
            f"kernel has {len(spec.jaxpr.invars)} refs but spec carries {n_refs} operands"
        )
    emit_jaxpr(env, cursor, spec.jaxpr, ref_vals)

    head = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")\n{"
    source = "\n".join(
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
    return source, EmitStats(
        thread_bytes=cursor.thread_bytes,
        threadgroup_bytes=cursor.threadgroup_bytes,
    )


def emit_msl(spec: KernelSpec, kernel_name: str | None = None) -> str:
    """Assemble the full MSL source for a KernelSpec.

    Thin wrapper over `emit_msl_stats` for callers that only want the
    text. See that function for the full contract.
    """
    return emit_msl_stats(spec, kernel_name)[0]


def ref_view(env: Environment, ref: CVal, indexer: NDIndexer) -> CVal:
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
            # pl.Slice.size is always static, and a dynamic start
            # is always a jaxpr atom (Var/Literal), never a live Array.
            kept.append((d, cast(int, idx.size)))
            start = idx.start
            expr = (
                str(start)
                if isinstance(start, int)
                else env.val(cast(Atom, start)).expr
            )
        else:
            # A non-Slice index here is always a jaxpr atom.
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

    # Squeezed trailing dimensions can make even full slices strided:
    # x[:, 1] is a column, not a contiguous span starting at x[0, 1].
    kept_strides = _element_strides(tuple(size for _, size in kept))
    if any(
        size > 1 and strides[d] != expected
        for (d, size), expected in zip(kept, kept_strides, strict=True)
    ):
        raise EmitError("ref access is non-contiguous; strided views are unimplemented")

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


def _transpose_is_dot_rhs_only(env: Environment, eqn: JaxprEqn) -> bool:
    """Whether this rank-2 `(1, 0)` transpose is consumed *only* as the
    rhs of dot_general equations (never as lhs, never escaping as a
    jaxpr outvar), i.e. safe to lower as a lazy `transposed` CVal that
    only dot_general knows how to index."""
    if tuple(eqn.params["permutation"]) != (1, 0):
        return False
    outvar = eqn.outvars[0]
    uses = env.consumer_eqns(outvar)
    return (
        bool(uses)
        and not env.escapes(outvar)
        and all(
            use.primitive.name == "dot_general"
            and use.invars[1] is outvar
            and use.invars[0] is not outvar
            for use in uses
        )
    )


ELEMENTWISE: dict[str, str] = {
    # binary
    "add": "({a} + {b})",
    # AD cotangent accumulation; supported numeric arrays use ordinary addition.
    "add_any": "({a} + {b})",
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
