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
from collections.abc import Callable, Generator
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
    """Narrow an abstract value to the shaped values handled by ordinary emission.

    Parameters
    ----------
    aval : object
        Non-Ref JAX abstract value.

    Returns
    -------
    ShapedArray
        The same abstract value, with its shaped type established.

    Raises
    ------
    AssertionError
        The value is not a ShapedArray, for example a Ref abstract value.
    """
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
    """Map a supported NumPy or JAX dtype to its Metal scalar type.

    Parameters
    ----------
    dtype : dtype-like
        One of float32, float16, bfloat16, int32, uint32, or bool.

    Returns
    -------
    str
        MSL spelling of the scalar element type.

    Raises
    ------
    KeyError
        The normalized dtype is unsupported.
    """
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
        Unmaterialized scalar-expression template using $i, restricted to consumers that
        preserve flat-index semantics.

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
        """Return the logical number of elements represented by this value.

        Returns
        -------
        int
            Product of dimensions; a scalar has one element.
        """
        return math.prod(self.shape)

    def slot(self, index: str, shape: tuple[int, ...]) -> CVal:
        """Create a contiguous view of one stacked slot without copying storage.

        Parameters
        ----------
        index : str
            MSL slot index, multiplied by the number of elements in the slot.
        shape : tuple of int
            Logical shape of one slot; () describes one scalar element.

        Returns
        -------
        CVal
            View with a pointer offset and alignment reduced by the slot size.

        Raises
        ------
        AssertionError
            The source is scalar or marked as lazily transposed.

        Notes
        -----
        The caller must ensure the slot lies in storage and uses a pointer-compatible source.
        This helper emits no bounds check.
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
        """Render an element expression without applying a bounds predicate.

        Parameters
        ----------
        index : str
            MSL logical flat-index expression.

        Returns
        -------
        str
            Scalar expression, indexed storage access, or substituted lazy/index-map
            template.

        Notes
        -----
        Scalars and scalar-backed one-element arrays ignore the index. Use ``read`` for a
        guarded load; ``at`` also serves as a store destination.
        """
        if self.lazy is not None:
            return self.lazy.replace("$i", index if index.isidentifier() else f"({index})")
        if self.index_map is not None:
            return f"{self.expr}[{self.index_map.replace('$i', f'({index})')}]"
        if not self.shape or self.scalar_storage:
            return self.expr
        return f"{self.expr}[{index}]"

    def read(self, index: str) -> str:
        """Render a load expression with Pallas-compatible padding for invalid lanes.

        Parameters
        ----------
        index : str
            MSL logical flat-index expression.

        Returns
        -------
        str
            Element expression guarded by ``valid``, when present.

        Notes
        -----
        Invalid floating lanes use NaN, signed integers use INT_MIN, unsigned integers use zero,
        and booleans use false. Without ``valid`` this is identical to ``at``; no new bounds
        check is inferred.
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
        """Register a helper function once and return its call name.

        Parameters
        ----------
        name : str
            Helper function identifier.
        source : str
            Complete helper declaration and body.

        Returns
        -------
        str
            The supplied function name.

        Notes
        -----
        For repeated names the first source wins; conflicting definitions are not checked.
        """
        self.helpers.setdefault(name, source)
        return name

    def account(self, ctype: str, count: int, space: str = "thread") -> None:
        """Add a typed allocation to the declared-storage estimate.

        Parameters
        ----------
        ctype : str
            MSL scalar type name, such as "float", "half", "bfloat", "int", "uint", or
            "bool".
        count : int
            Element count; zero or negative counts are accounted as one element.
        space : str, optional
            "threadgroup" records shared bytes; any other value records thread bytes.

        Raises
        ------
        KeyError
            The scalar type has no known byte size.

        Notes
        -----
        This updates counters without emitting or validating an allocation. Prefer ``allocate``
        when declaring new storage.
        """
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
        """Declare local typed storage and return its lowered value.

        Parameters
        ----------
        ctype : str
            MSL scalar type name, such as "float", "half", "bfloat", "int", "uint", or
            "bool".
        shape : tuple of int
            Positive dimensions, or () for a scalar.
        name : str or None, optional
            Explicit identifier, or None for a fresh name.
        space : {"thread", "threadgroup"}, optional
            Allocation address space, default "thread".
        prefix : str, optional
            Fresh-name prefix when name is omitted, default "t".

        Returns
        -------
        CVal
            Uninitialized storage with accounting recorded. Ranked one-element values use
            scalar storage.

        Raises
        ------
        EmitError
            The scalar type or address space is unsupported, or any dimension is
            nonpositive.

        Notes
        -----
        Explicit names must be valid unique MSL identifiers. This emits a declaration only;
        reads before initialization remain invalid.
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
        return CVal(name, shape, ctype, space=space, scalar_storage=scalar)

    def emit(self, line: str) -> None:
        """Append an MSL statement at the current indentation depth.

        Parameters
        ----------
        line : str
            Statement text without indentation or a trailing newline.

        Notes
        -----
        The text is appended verbatim after indentation. Syntax and embedded newlines are not
        validated.
        """
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        """Generate a deterministic identifier unique to this cursor.

        Parameters
        ----------
        prefix : str, optional
            Identifier prefix, default "t"; callers must supply a valid MSL prefix.

        Returns
        -------
        str
            Prefix followed by the next cursor-wide integer, shared across all prefixes.
        """
        return f"{prefix}{next(self._names)}"

    @contextlib.contextmanager
    def block(self, header: str) -> Generator[None]:
        """Open a braced MSL block and indent statements emitted inside it.

        Parameters
        ----------
        header : str
            Block header such as ``for (...)`` or ``if (...)``, without an opening brace.

        Yields
        ------
        None
            Statements inside the context are written one indentation level deeper.

        Notes
        -----
        Closing syntax is emitted on normal context exit. Do not reuse a cursor after an
        exception interrupts a block.
        """
        self.emit(header + " {")
        self.indent += 1
        yield
        self.indent -= 1
        self.emit("}")

    @contextlib.contextmanager
    def loop(self, count: int | str, prefix: str = "_i", reverse: bool = False) -> Generator[str]:
        """Emit a counted loop and expose its index to nested emission.

        Parameters
        ----------
        count : int or str
            Exclusive upper bound; MSL expressions are inserted verbatim.
        prefix : str, optional
            Loop-index prefix, default "_i".
        reverse : bool, optional
            Traverse count - 1 down to zero with a signed index, default False.

        Yields
        ------
        str
            Loop index name, or literal "0" when the integer count equals one.

        Notes
        -----
        A count of one removes the loop entirely. A zero count emits a loop whose body does not
        execute. Forward indices are unsigned, so callers must use a nonnegative bound; reverse
        indices are signed to terminate at zero.
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
        """Emit nested loops over logical dimensions and expose their indices.

        Parameters
        ----------
        shape : tuple of int
            Static extents from outermost to innermost loop.
        prefix : str, optional
            Index-name prefix, default "_i".

        Yields
        ------
        list of str
            One index per dimension; unit extents use literal "0". An empty shape yields an
            empty list.
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
    ) -> Generator[str]:
        """Emit a loop that assigns strided work to cooperative threads.

        Parameters
        ----------
        start : str
            MSL starting index.
        stop : str
            MSL exclusive upper bound.
        step : str
            MSL positive increment.
        name : str or None, optional
            Explicit loop identifier, or None for a fresh name.
        prefix : str, optional
            Fresh-index prefix, default "_i".

        Yields
        ------
        str
            Unsigned loop-index identifier.

        Notes
        -----
        Expressions are inserted verbatim without validation; a zero step would produce a
        nonterminating loop.
        """
        idx = name or self.fresh(prefix)
        with self.block(f"for (uint {idx} = {start}; {idx} < {stop}; {idx} += {step})"):
            yield idx

    def barrier(self) -> None:
        """Emit a threadgroup barrier fencing shared-memory accesses.

        Notes
        -----
        All threads in the group must reach the barrier uniformly. This helper does not analyze
        surrounding control flow or fence device memory.
        """
        self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

    def copy(self, dst: CVal, src: CVal, count: int) -> None:
        """Emit element assignments while preserving guarded load and store behavior.

        Parameters
        ----------
        dst : CVal
            Writable destination value or view.
        src : CVal
            Source value or view; reads apply any padding predicate.
        count : int
            Number of logical elements to copy, within both values.

        Notes
        -----
        Invalid destination lanes are skipped; invalid source lanes use dtype-specific padding.
        The copy is sequential and does not implement overlapping-region memmove semantics.
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

    def val(self, atom: Atom) -> CVal:
        """Resolve a jaxpr variable or format an inline literal.

        Parameters
        ----------
        atom : Var or Literal
            Atom at the current jaxpr level.

        Returns
        -------
        CVal
            Bound value, or a typed scalar constant for a literal.

        Raises
        ------
        KeyError
            A variable has not been bound or a literal dtype is unsupported.
        """
        if isinstance(atom, Literal):
            return literal(atom)
        return self.bindings[atom]

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Associate a jaxpr variable with an existing lowered value.

        Parameters
        ----------
        var : Var
            Variable whose current binding is replaced, if present.
        cval : CVal
            Storage or expression to reuse without allocation.

        Returns
        -------
        CVal
            The supplied value, enabling binding and returning in one operation.
        """
        self.bindings[var] = cval
        return cval

    def consumer_eqns(self, var: Var) -> list[JaxprEqn]:
        """Return consuming equations at a variable's own jaxpr level.

        Parameters
        ----------
        var : Var
            Variable to query.

        Returns
        -------
        list of JaxprEqn
            Consumers in use order, excluding the escape marker; empty for unknown or unused
            variables.
        """
        return [e for e in self.consumers.get(var, []) if e is not None]

    def escapes(self, var: Var) -> bool:
        """Determine whether a variable is returned from its jaxpr level.

        Parameters
        ----------
        var : Var
            Variable to query.

        Returns
        -------
        bool
            True if its consumer list includes an output marker; False for unknown
            variables.
        """
        return None in self.consumers.get(var, ())

    def sole_consumer(self, var: Var) -> JaxprEqn | None:
        """Find the only use of a variable when it stays within its jaxpr level.

        Parameters
        ----------
        var : Var
            Variable to query.

        Returns
        -------
        JaxprEqn or None
            Single consuming equation; None for an unused, escaping, or multiply used value.

        Notes
        -----
        Repeated appearances in one equation count as multiple uses, preventing unsafe
        duplication of lazy expressions.
        """
        uses = self.consumers.get(var, [])
        if len(uses) == 1 and uses[0] is not None:
            return uses[0]
        return None


def literal(atom: Literal) -> CVal:
    """Format a scalar jaxpr literal with its Metal element type.

    Parameters
    ----------
    atom : Literal
        Scalar literal of a supported dtype.

    Returns
    -------
    CVal
        Rank-zero constant, including explicit INFINITY or NAN spellings when needed.

    Raises
    ------
    KeyError
        The literal dtype has no Metal scalar mapping.

    Notes
    -----
    Floating constants use a float32-roundtripping decimal; bfloat values receive an
    explicit narrowing constructor. Captured array constants are handled separately and
    rejected by the walker.
    """
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
    """Allocate thread-local storage for a variable and bind it in the environment.

    Parameters
    ----------
    env : Environment
        Bindings and def-use information for the current jaxpr.
    cursor : Cursor
        Destination for generated MSL statements and storage accounting.
    var : Var
        Shaped jaxpr variable whose dtype and shape determine the declaration.

    Returns
    -------
    CVal
        Fresh uninitialized storage already bound to the variable.

    Raises
    ------
    EmitError
        Local dimensions are nonpositive.
    KeyError
        The dtype is unsupported.

    Notes
    -----
    Use ``env.bind`` when reusing a value or view; this helper always allocates new storage.
    """
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
    """Register one primitive-lowering implementation under one or more names.

    Parameters
    ----------
    *names : str
        Primitive names to map to the decorated function.

    Returns
    -------
    callable
        Decorator preserving a rule with signature ``(env, cursor, eqn) -> None``.

    Notes
    -----
    Registration mutates the process-wide RULES dictionary. Existing names are replaced; an
    empty name list leaves the decorated function unregistered.
    """

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(env: Environment, cursor: Cursor, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr and dispatch each equation to its registered MSL rule.

    Parameters
    ----------
    env : Environment
        Bindings and def-use information for the current jaxpr.
    cursor : Cursor
        Destination for generated MSL statements and storage accounting.
    jaxpr : Jaxpr
        Kernel body or nested region to emit.
    in_vals : list of CVal
        Values matching jaxpr input variables in order.

    Returns
    -------
    list of CVal
        Lowered output atoms in jaxpr output order; empty for a state-only body.

    Raises
    ------
    EmitError
        Captured array constants or unsupported primitive cases are encountered.
    UnsupportedPrimitiveError
        An equation has no rule in RULES.
    ValueError
        Input arity differs from the jaxpr input count.

    Notes
    -----
    Consumer and producer maps are populated before any rule runs. Bindings and cursor state
    are mutated; emission failures may leave partial statements, so discard the cursor on
    error.
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
