"""The lowering IR for one-thread-per-program kernels.

CVal (a lowered value), Cursor (the MSL text being written), Environment
(jaxpr Var bindings and def-use info), the rule registry, and the jaxpr
walk. Per-primitive rules live in `rules`; addressing in `addressing`;
kernel assembly in `kernel`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
from collections.abc import Callable, Iterator
from typing import Literal as TLiteral

import jax.numpy as jnp
import numpy as np
from jax.core import Atom, ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var
from numpy.typing import DTypeLike

from palladium.errors import EmitError, UnsupportedPrimitiveError


def shaped(aval: object) -> ShapedArray:
    # Every non-Ref value in a Pallas kernel jaxpr is shaped; Refs never
    # pass through declare() or val().
    assert isinstance(aval, ShapedArray), aval
    return aval


CTYPES: dict[np.dtype, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.float16): "half",
    np.dtype(jnp.bfloat16): "bfloat",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.bool_): "bool",
}


def msl_type(dtype: DTypeLike) -> str:
    """The MSL type name for a supported dtype; KeyError otherwise."""
    return CTYPES[np.dtype(dtype)]


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

PID = ("_pid.x", "_pid.y", "_pid.z")


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
        MSL type name, a value of CTYPES (`"float"`, not `"float32"`).
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
    # Ref-only addressing templates; $i is a flattened logical element index.
    # Such refs are materialized on reads, never passed to pointer optimizations.
    index_map: str | None = None
    valid: str | None = None
    # A ranked value of one element declared as a plain scalar variable, so
    # `at` never indexes it and no pointer to it may be formed.
    scalar_storage: bool = False
    # An unmaterialized elementwise value: `at(i)` substitutes the flat
    # element index for `$i` in this expression template instead of
    # reading storage. Only bound when exactly one same-shape elementwise
    # equation consumes the value, so each element is evaluated once.
    lazy: str | None = None

    @property
    def size(self) -> int:
        """Element count; 1 for scalars."""
        return math.prod(self.shape)

    def slot(self, index: str, shape: tuple[int, ...]) -> CVal:
        """A view of `shape` at element offset `index * prod(shape)`;
        alignment composes as gcd with the slot size.
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
        """`expr` for scalars, `expr[index]` for arrays; lazy and
        index-mapped values substitute `index` for `$i`."""
        if self.lazy is not None:
            return self.lazy.replace("$i", index if index.isidentifier() else f"({index})")
        if self.index_map is not None:
            return f"{self.expr}[{self.index_map.replace('$i', f'({index})')}]"
        if not self.shape or self.scalar_storage:
            return self.expr
        return f"{self.expr}[{index}]"

    def read(self, index: str) -> str:
        value = self.at(index)
        if self.valid is None:
            return value
        fill = {
            "float": "NAN",
            "half": "half(NAN)",
            "bfloat": "bfloat(NAN)",
            "int": "(-2147483647 - 1)",
            "uint": "0u",
            "bool": "false",
        }[self.ctype]
        return f"({self.valid.replace('$i', f'({index})')} ? {value} : {fill})"


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
    """The write position into one kernel's MSL text: emitted lines,
    indentation, the unique-name counter, helpers, and storage totals.
    Independent of jaxpr bindings.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 1
        self._names = itertools.count()
        self.thread_bytes = 0
        self.threadgroup_bytes = 0
        # MSL functions the body calls, by name, emitted once above the
        # kernel in first-use order.
        self.helpers: dict[str, str] = {}

    def require(self, name: str, source: str) -> str:
        """Register helper function `source` under `name`; returns `name`."""
        self.helpers.setdefault(name, source)
        return name

    def account(self, ctype: str, count: int, space: str = "thread") -> None:
        """Record `count` elements of `ctype` declared in `space`."""
        nbytes = CTYPE_BYTES[ctype] * max(count, 1)
        if space == "threadgroup":
            self.threadgroup_bytes += nbytes
        else:
            self.thread_bytes += nbytes

    def allocate(
        self,
        ctype: str,
        shape: tuple[int, ...],
        *,
        name: str | None = None,
        space: str = "thread",
        prefix: str = "t",
    ) -> CVal:
        """Declare typed local storage, account for it, and return its CVal."""
        if ctype not in CTYPE_BYTES:
            raise EmitError(f"cannot allocate unsupported C type {ctype!r}")
        if space not in ("thread", "threadgroup"):
            raise EmitError(f"cannot allocate local storage in {space!r} space")
        if any(size <= 0 for size in shape):
            raise EmitError("local storage dimensions must be positive")
        name = name or self.fresh(prefix)
        count = math.prod(shape)
        self.account(ctype, count, space)
        qualifier = "threadgroup " if space == "threadgroup" else ""
        # One-element values are declared as scalars so the compiler keeps
        # them in registers.
        scalar = bool(shape) and count == 1
        declaration = f"{qualifier}{ctype} {name}"
        declaration += f"[{count}]" if shape and not scalar else ""
        self.emit(declaration + ";")
        return CVal(name, shape, ctype, space=space, scalar_storage=scalar)

    def emit(self, line: str) -> None:
        """Append one MSL line at the current indentation."""
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        """Return a new unique C identifier, deterministic per Cursor."""
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
    def loop(self, count: int | str, prefix: str = "_i", reverse: bool = False) -> Iterator[str]:
        """Emit a counted for-loop over `[0, count)`; yields the index name.

        Reverse loops use a signed index, since a uint would wrap past
        zero. `count` is inlined verbatim; pass `f"{n}u"` where an
        unsigned literal is needed. A count of 1 emits no loop and
        yields "0".
        """
        if count == 1:
            yield "0"
            return
        idx = self.fresh(prefix)
        if reverse:
            first = count - 1 if isinstance(count, int) else f"{count} - 1"
            header = f"for (int {idx} = {first}; {idx} >= 0; --{idx})"
        else:
            header = f"for (uint {idx} = 0; {idx} < {count}; ++{idx})"
        with self.block(header):
            yield idx

    @contextlib.contextmanager
    def loop_nest(self, shape: tuple[int, ...], prefix: str = "_i") -> Iterator[list[str]]:
        """Emit nested counted loops over `shape`; yields one index per dim.

        Size-1 dimensions get the literal index "0" and no loop.
        """
        indices: list[str] = []
        with contextlib.ExitStack() as stack:
            for extent in shape:
                indices.append(stack.enter_context(self.loop(extent, prefix)))
            yield indices

    @contextlib.contextmanager
    def strided_loop(
        self,
        start: str,
        stop: str,
        step: str,
        *,
        name: str | None = None,
        prefix: str = "_i",
    ) -> Iterator[str]:
        """Emit `for (uint i = start; i < stop; i += step)` for cooperative work."""
        idx = name or self.fresh(prefix)
        with self.block(f"for (uint {idx} = {start}; {idx} < {stop}; {idx} += {step})"):
            yield idx

    def barrier(self) -> None:
        """Emit a barrier covering threadgroup memory."""
        self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

    def copy(self, dst: CVal, src: CVal, count: int) -> None:
        """Emit `count` element assignments dst[i] = src[i] as a loop."""
        with self.loop(count) as i:
            assignment = f"{dst.at(i)} = {src.read(i)};"
            if dst.valid is None:
                self.emit(assignment)
            else:
                with self.block(f"if ({dst.valid.replace('$i', f'({i})')})"):
                    self.emit(assignment)


class Environment:
    """The symbol table for one kernel: jaxpr Var bindings and def-use info.

    Attributes
    ----------
    bindings : dict
        Maps jaxpr Vars to their lowered CVals.
    consumers, producers : dict
        Var -> consuming eqns (None marks a jaxpr outvar) and
        Var -> defining eqn, for rules that need lookahead.
    """

    def __init__(
        self, no_stream_refs: frozenset[Var] = frozenset(), *, fuse_loads: bool = False
    ) -> None:
        self.bindings: dict[Var, CVal] = {}
        # Refs sharing a buffer with another ref (input_output_aliases):
        # never scan-ys streaming targets, since reads through the twin
        # var are invisible here.
        self.no_stream_refs = no_stream_refs
        # Read input blocks in place instead of copying them to thread
        # storage, where each element is read once.
        self.fuse_loads = fuse_loads
        # Per jaxpr level, rebuilt before each (sub-)jaxpr is walked: a
        # cached jit body is shared, Vars and all, by every call to it.
        self.consumers: dict[Var, list[JaxprEqn | None]] = {}
        self.producers: dict[Var, JaxprEqn] = {}
        # Outputs of the jit body being inlined, to the outer Vars they bind.
        self.jit_outputs: dict[Var, Var] = {}

    def val(self, atom: Atom) -> CVal:
        """Resolve a jaxpr atom: Vars from bindings, Literals formatted
        in place."""
        if isinstance(atom, Literal):
            return literal(atom)
        return self.bindings[atom]

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Bind `var` to an existing CVal: aliasing, no declaration."""
        self.bindings[var] = cval
        return cval

    def consumer_eqns(self, var: Var) -> list[JaxprEqn]:
        """Consuming equations at `var`'s jaxpr level, without the outvar marker."""
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


def literal(atom: Literal) -> CVal:
    """A jaxpr Literal as a typed C constant."""
    ctype = msl_type(shaped(atom.aval).dtype)
    v = atom.val
    if math.isinf(v):
        expr = "-INFINITY" if v < 0 else "INFINITY"
    elif math.isnan(v):
        expr = "NAN"
    else:
        # The shortest decimal that rounds back to the same float32.
        expr = f"{np.float32(v)!s}f" if ctype in ("float", "half", "bfloat") else str(int(v))
    if ctype == "bfloat":
        # MSL does not implicitly narrow float to bfloat.
        expr = f"bfloat({expr})"
    return CVal(expr=expr, shape=(), ctype=ctype)


def declare(env: Environment, cursor: Cursor, var: Var) -> CVal:
    """Emit thread-local storage for `var` and bind it in `env`; use
    `env.bind` for aliasing."""
    aval = shaped(var.aval)
    ctype = msl_type(aval.dtype)
    shape = tuple(int(d) for d in aval.shape)
    cval = cursor.allocate(ctype, shape)
    env.bind(var, cval)
    return cval


# Per-thread storage above which copied input blocks spill out of registers.
# Below it, copying a block first measured faster than reading it in place;
# above it, reading in place won by up to 2.9x (M2, blocked elementwise).
REGISTER_BYTES = 512


RuleFn = Callable[[Environment, Cursor, JaxprEqn], None]

RULES: dict[str, RuleFn] = {}


def rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    """Register a lowering rule for one or more primitive names."""

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(env: Environment, cursor: Cursor, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr, dispatching each equation to its rule in RULES.

    `in_vals` binds `jaxpr.invars` in order; returns the values of
    `jaxpr.outvars`. Consumer and producer maps are recorded before any
    rule runs, so rules can look ahead.

    Raises
    ------
    EmitError
        If the jaxpr captures arrays as constvars.
    UnsupportedPrimitiveError
        If an equation's primitive has no rule.
    """
    if jaxpr.constvars:
        raise EmitError(
            "kernel jaxpr has constvars (captured arrays); close over Python "
            "scalars only, or pass arrays as kernel operands"
        )
    for var, cval in zip(jaxpr.invars, in_vals, strict=True):
        env.bind(var, cval)
    for var in (*jaxpr.invars, *(ov for eqn in jaxpr.eqns for ov in eqn.outvars)):
        env.consumers[var] = []
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
        name = eqn.primitive.name
        impl = RULES.get(name)
        if impl is None:
            raise UnsupportedPrimitiveError(f"no MSL rule for primitive '{name}'", primitive=name)
        impl(env, cursor, eqn)
    return [env.val(v) for v in jaxpr.outvars]
