"""The lowering IR for one-thread-per-program kernels.

CVal (a lowered value), Cursor (the MSL text being written), Environment
(jaxpr Var bindings and def-use info), the rule registry, and the jaxpr
walk. Per-primitive rules live in `rules`; addressing in `addressing`;
kernel assembly in `kernel`.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import itertools
import math
from collections.abc import Callable, Generator
from typing import Literal as TLiteral

import jax.numpy as jnp
import numpy as np
from jax.core import Atom, ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var
from numpy.typing import DTypeLike

from palladium.errors import EmitError, UnsupportedPrimitiveError


def shaped(aval: object) -> ShapedArray:
    """Narrow an abstract value to the ShapedArray that every non-Ref kernel value has."""
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
    """The Metal scalar type of a supported dtype; any other raises KeyError."""
    return CTYPES[np.dtype(dtype)]


# Size of each Metal scalar type, for the storage estimate in `Cursor.account`.
CTYPE_BYTES = {ctype: dtype.itemsize for dtype, ctype in CTYPES.items()}

PID = ("_pid.x", "_pid.y", "_pid.z")


@dataclasses.dataclass(frozen=True)
class CVal:
    """Represent a lowered jaxpr value together with its storage and addressing facts.

    Attributes
    ----------
    expr : str
        MSL identifier, pointer, literal, or scalar expression.
    shape : tuple of int
        Logical dimensions; () is a scalar.
    ctype : str
        MSL scalar type name from CTYPES.
    space : {"thread", "threadgroup", "device"}, optional
        Storage address space, default "thread"; pointer casts must preserve it.
    readonly : bool, optional
        Whether backing storage is immutable, default False. Only immutable device reads
        may safely alias a Ref snapshot.
    transposed : bool, optional
        Lazy rank-2 transpose, default False. The expression retains source storage
        while shape records the logical transpose; only supported dot consumers may read
        it.
    align : int, optional
        Provable element-offset multiple relative to a 16-byte-aligned base. Zero
        denotes exact base alignment and is the gcd identity.
    index_map : str or None, optional
        Physical element-index template using $i for the logical flat index.
        Noncontiguous Ref reads must materialize it.
    valid : str or None, optional
        Bounds-predicate template using $i; guarded reads pad and guarded writes skip
        invalid lanes.
    scalar_storage : bool, optional
        Whether a ranked one-element value is stored in a scalar variable, default
        False; no pointer or subscript may be formed for it.
    lazy : str or None, optional
        Unmaterialized elementwise value: a scalar-expression template using $i in
        place of reading storage. Only bound when one same-shape consumer evaluates
        each element once.
    storage : str or None, optional
        The local variable `Cursor.allocate` declared for this value. Views and
        aliases keep it, so `Environment.storage_users` counts every binding that
        can reach the variable.

    Notes
    -----
    This record describes emitter invariants rather than validating them. In particular, a
    lazy transpose is not safe for arbitrary flat-index consumers and guarded device storage
    must retain its bounds predicate.
    """

    expr: str
    shape: tuple[int, ...]
    ctype: str
    space: TLiteral["thread", "threadgroup", "device"] = "thread"
    readonly: bool = False
    transposed: bool = False
    align: int = 0
    index_map: str | None = None
    valid: str | None = None
    scalar_storage: bool = False
    lazy: str | None = None
    storage: str | None = None

    @property
    def size(self) -> int:
        """Number of elements; one for a scalar."""
        return math.prod(self.shape)

    def slot(self, index: str, shape: tuple[int, ...]) -> CVal:
        """A view of slot `index` of stacked storage, each slot holding `shape`.

        Nothing is copied and no bounds check is emitted; the source must be
        ranked storage that is not lazily transposed.
        """
        assert self.shape and not self.transposed, self
        if self.scalar_storage:
            # One element in a scalar variable: its only slot is the variable.
            return dataclasses.replace(self, shape=shape)
        size = math.prod(shape)
        return dataclasses.replace(
            self,
            expr=f"({self.expr} + {index} * {size})",
            shape=shape,
            align=math.gcd(self.align, size),
        )

    def at(self, index: str) -> str:
        """The element at flat index `index`, as a load or a store target.

        Scalars and scalar-backed values ignore the index. No bounds predicate is
        applied: `read` is the guarded load.
        """
        if self.lazy is not None:
            return self.lazy.replace("$i", index if index.isidentifier() else f"({index})")
        if self.index_map is not None:
            return f"{self.expr}[{self.index_map.replace('$i', f'({index})')}]"
        if not self.shape or self.scalar_storage:
            return self.expr
        return f"{self.expr}[{index}]"

    def read(self, index: str) -> str:
        """`at(index)` as a load, guarded by `valid` when the value has one.

        Invalid lanes read the Pallas padding value: NaN for floats, INT_MIN for
        int, zero for uint and false for bool.
        """
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
    """Record the storage explicitly declared by one emitted kernel.

    Attributes
    ----------
    thread_bytes : int
        Primitive-emitter thread-local bytes per program, including input snapshots,
        temporaries, and scratch.
    threadgroup_bytes : int
        Shared bytes per cooperative group, compared with the device shared-memory
        limit.

    Notes
    -----
    Counts are not liveness-aware and do not include compiler-generated temporaries or MPP
    register use. Metal publishes no per-thread stack ceiling; reducing thread_bytes can
    help when pipeline creation fails.
    """

    thread_bytes: int
    threadgroup_bytes: int


class Cursor:
    """Accumulate MSL statements, helper functions, unique names, and storage estimates.

    Attributes
    ----------
    lines : list of str
        Statements with indentation applied as they are emitted.
    indent : int
        Indentation depth, initially one for a kernel body.
    thread_bytes : int
        Total declared thread-local bytes.
    threadgroup_bytes : int
        Total declared shared bytes.
    helpers : dict of str to str
        Helper source by function name, retained in first-use order.

    Notes
    -----
    Each kernel uses a fresh cursor so names are deterministic. This object emits source
    only; it has no JAX symbol bindings and performs no Metal compilation.
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
        """Register helper `source` under `name` once and return the name; the first source wins."""
        self.helpers.setdefault(name, source)
        return name

    def account(self, ctype: str, count: int, space: str = "thread") -> None:
        """Add `count` elements of `ctype`, at least one, to the declared-storage estimate."""
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
        """Declare uninitialized local storage and return its value.

        A fresh name with `prefix` is used unless `name` is given. Raises EmitError
        for an unknown scalar type, an address space other than thread or
        threadgroup, or a nonpositive dimension.
        """
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
        return CVal(name, shape, ctype, space=space, scalar_storage=scalar, storage=name)

    def emit(self, line: str) -> None:
        """Append one statement at the current indentation."""
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        """A new identifier `prefix<n>`; all prefixes share one counter."""
        return f"{prefix}{next(self._names)}"

    @contextlib.contextmanager
    def block(self, header: str) -> Generator[None]:
        """Open `header {`, indent what is emitted inside, and close the brace on exit."""
        self.emit(header + " {")
        self.indent += 1
        yield
        self.indent -= 1
        self.emit("}")

    @contextlib.contextmanager
    def loop(self, count: int | str, prefix: str = "_i", reverse: bool = False) -> Generator[str]:
        """Emit a counted loop and yield its index.

        A count of one emits no loop and yields "0". Forward indices are unsigned;
        a reverse loop runs from `count - 1` down to zero on a signed index.
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
    def loop_nest(self, shape: tuple[int, ...], prefix: str = "_i") -> Generator[list[str]]:
        """Nested `loop`s over `shape`, outermost first; yields one index per dimension."""
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
    ) -> Generator[str]:
        """Emit `for (uint i = start; i < stop; i += step)` and yield the index."""
        idx = name or self.fresh(prefix)
        with self.block(f"for (uint {idx} = {start}; {idx} < {stop}; {idx} += {step})"):
            yield idx

    def barrier(self) -> None:
        """Emit a threadgroup barrier; every thread of the group must reach it."""
        self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

    def copy(self, dst: CVal, src: CVal, count: int) -> None:
        """Emit `count` element assignments from `src` to `dst`.

        Invalid destination lanes are skipped and invalid source lanes read
        padding. The copy runs front to back: overlapping regions are not a memmove.
        """
        with self.loop(count) as i:
            assignment = f"{dst.at(i)} = {src.read(i)};"
            if dst.valid is None:
                self.emit(assignment)
            else:
                with self.block(f"if ({dst.valid.replace('$i', f'({i})')})"):
                    self.emit(assignment)


class Environment:
    """Track jaxpr value bindings and use information for safe local emission.

    Parameters
    ----------
    no_stream_refs : frozenset of Var, optional
        Refs sharing buffers through input/output aliases; these cannot be direct
        streaming destinations.
    fuse_loads : bool, optional
        Permit eligible single-read immutable input blocks to stay as device views,
        default False.

    Attributes
    ----------
    bindings : dict of Var to CVal
        Lowered values indexed by jaxpr variable.
    consumers : dict of Var to list
        Consuming equations at the current jaxpr level; None marks an escaping output.
    producers : dict of Var to JaxprEqn
        Defining equations used for lookahead.
    jit_outputs : dict of Var to Var
        Connections from inlined JIT outputs to their outer variables.
    no_stream_refs : frozenset of Var
        Refs excluded from early writes into output storage.
    fuse_loads : bool
        Whether eligible input snapshots may become immutable device views.

    Notes
    -----
    The jaxpr walker rebuilds use maps for each level because JIT bodies can reuse the same
    Var identities. Sharing an Environment between concurrent emissions is unsupported.
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
        # How many bound variables hold each declared storage, whole or in part.
        self.storage_users: collections.Counter[str | None] = collections.Counter()

    def val(self, atom: Atom) -> CVal:
        """The value bound to a variable, or a literal as a typed scalar constant."""
        if isinstance(atom, Literal):
            return literal(atom)
        return self.bindings[atom]

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Bind `var` to an existing value, without allocating, and return the value."""
        previous = self.bindings.get(var)
        if previous is not None:
            self.storage_users[previous.storage] -= 1
        self.storage_users[cval.storage] += 1
        self.bindings[var] = cval
        return cval

    def consumer_eqns(self, var: Var) -> list[JaxprEqn]:
        """The equations that use `var` at its own jaxpr level, in order."""
        return [e for e in self.consumers.get(var, []) if e is not None]

    def escapes(self, var: Var) -> bool:
        """Whether `var` is an output of its jaxpr level."""
        return None in self.consumers.get(var, ())

    def sole_consumer(self, var: Var) -> JaxprEqn | None:
        """The only equation using `var`; None when it is unused, escapes, or has more uses.

        Two appearances in one equation count as two uses, so a lazy expression
        is never duplicated.
        """
        uses = self.consumers.get(var, [])
        if len(uses) == 1 and uses[0] is not None:
            return uses[0]
        return None


def literal(atom: Literal) -> CVal:
    """A scalar jaxpr literal as a typed MSL constant.

    Floats print as the shortest decimal that rounds to the same float32, with
    INFINITY and NAN spelled out; bfloat gets an explicit constructor, since
    MSL does not narrow to it implicitly. Unsigned constants carry a `u`
    suffix. Captured array constants never reach this: the walker rejects them.
    """
    ctype = msl_type(shaped(atom.aval).dtype)
    v = atom.val
    if math.isinf(v):
        expr = "-INFINITY" if v < 0 else "INFINITY"
    elif math.isnan(v):
        expr = "NAN"
    elif ctype in ("float", "half", "bfloat"):
        # The shortest decimal that rounds back to the same float32.
        expr = f"{np.float32(v)!s}f"
    elif ctype == "uint":
        # Unsuffixed, a value above INT_MAX is a 64-bit literal, and arithmetic
        # on it no longer wraps at 32 bits.
        expr = f"{int(v)}u"
    elif int(v) == -(2**31):
        expr = "(-2147483647 - 1)"  # 2147483648 alone is 64-bit, too
    else:
        expr = str(int(v))
    if ctype == "bfloat":
        # MSL does not implicitly narrow float to bfloat.
        expr = f"bfloat({expr})"
    return CVal(expr=expr, shape=(), ctype=ctype)


def declare(env: Environment, cursor: Cursor, var: Var) -> CVal:
    """Allocate thread-local storage for `var` and bind it; `env.bind` reuses a value instead."""
    aval = shaped(var.aval)
    ctype = msl_type(aval.dtype)
    shape = tuple(int(d) for d in aval.shape)
    cval = cursor.allocate(ctype, shape)
    env.bind(var, cval)
    return cval


# Per-thread storage above which copied input blocks spill out of registers.
# Below it, copying a block first is faster than reading it in place; above
# it, reading in place is.
REGISTER_BYTES = 512


RuleFn = Callable[[Environment, Cursor, JaxprEqn], None]

RULES: dict[str, RuleFn] = {}


def rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    """Register the decorated function as the lowering of each named primitive."""

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(env: Environment, cursor: Cursor, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr, lowering each equation through its rule, and return the outputs.

    `in_vals` bind the jaxpr's inputs. Consumer and producer maps are rebuilt
    for this level before any rule runs. Raises UnsupportedPrimitiveError for
    a primitive without a rule and EmitError for captured array constants.
    Bindings and the cursor are mutated, so discard the cursor after an error.
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
