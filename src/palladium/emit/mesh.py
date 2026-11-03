"""Mesh programs: a `pl.kernel` on a `plmtl.Mesh`, one threadgroup per program.

The body is plain Pallas over whole device arrays and threadgroup scratch,
and its block operations run cooperatively over the threadgroup:

- A copy between two refs is spread over all threads, four elements at a
  time where the rows allow it.
- `jnp.dot` of two threadgroup tiles runs on 8x8 SIMD-group matrix fragments,
  the SIMD groups arranged 2x2 over the output, each lane holding two
  adjacent elements of each fragment it owns (MLX's BaseMMAFrag layout).
  `acc + jnp.dot(...)` accumulates into `acc` in place, and `a @ b.T`
  reads `b` transposed.
- Elementwise operations on a dot's result apply to each lane's fragment
  elements; storing it writes each lane's elements.
- Vectors (`iota`, one-dimensional reads of device arrays, and elementwise
  functions of them) stay expressions of their index until they meet a
  block: broadcast along rows or columns, each lane evaluates them at its
  fragment coordinates, as causal masks and per-row statistics need.
- Scalars are uniform: every thread computes them. `fori_loop` becomes a
  loop with scalar and fragment carries, a `for` loop for static bounds and
  a `while` loop for traced ones.

Threadgroup tiles are stored with rows padded by four elements, which keeps
the fragment reads free of bank conflicts. A barrier goes before a
threadgroup read that follows a write, and before a write that follows a
read, since the last barrier.
"""

from __future__ import annotations

import dataclasses
import math
from contextlib import nullcontext

import jax.experimental.pallas as pl
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from palladium.emit.addressing import element_strides
from palladium.emit.core import Cursor, CVal, literal, msl_type, shaped
from palladium.emit.rules.control import _scan_shape
from palladium.emit.rules.elementwise import lowers_elementwise, scalar_expression
from palladium.emit.tensorops._shared import _kernel_source
from palladium.emit.tensorops.simdgroup import _MMA, emit_fragment_coordinates
from palladium.errors import EmitError
from palladium.trace import KernelSpec

_SIMD_WIDTH = 32
_WM, _WN = 2, 2  # SIMD groups along fragment rows and columns
_PAD = 4
_UNROLL = '_Pragma("clang loop unroll(full)")'
_NESTED = {"jit": "jaxpr", "pjit": "jaxpr", "custom_jvp_call": "call_jaxpr"}


@dataclasses.dataclass(frozen=True)
class _Scalar:
    expr: str
    ctype: str


@dataclasses.dataclass(frozen=True)
class _Ref:
    """A whole device array or a threadgroup tile; `strides` in elements."""

    expr: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    ctype: str
    space: str


@dataclasses.dataclass(frozen=True)
class _View:
    """A rectangular part of a ref at element `offset`, with the kept dimensions'
    extents and strides; `align` is a known divisor of the offset."""

    ref: _Ref
    offset: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    align: int


@dataclasses.dataclass(frozen=True)
class _Load:
    """A block read, placed only by its use: a copy or a dot operand."""

    view: _View


@dataclasses.dataclass(frozen=True)
class _Splat:
    expr: str
    ctype: str
    shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _Dot:
    """`a @ b`, or `a @ b.T` with `b` stored (n, k) when `transposed`."""

    a: _View
    b: _View
    transposed: bool = False


@dataclasses.dataclass(frozen=True)
class _Fragments:
    """A (rows, columns) value in fragments: lane `_lane` of SIMD group `_sg` holds
    `name[i][j]`, the elements at row `_sm + 16 * i` and columns
    `_sn + 16 * j + c` for c in 0, 1."""

    name: str
    shape: tuple[int, int]
    ctype: str


_INDEX = "$i"  # stands for the index in an _Indexed template


@dataclasses.dataclass(frozen=True)
class _Indexed:
    """A value varying along `axis` of `shape` only (other extents are 1 or
    broadcast): `template` with `_INDEX` replaced by the index along `axis`."""

    template: str
    ctype: str
    shape: tuple[int, ...]
    axis: int

    def at(self, index: str) -> str:
        return self.template.replace(_INDEX, f"({index})")


_Value = _Scalar | _Ref | _Load | _Splat | _Dot | _Fragments | _Indexed


def lower_mesh_program(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Emit a kernel running each mesh program as one threadgroup.

    Parameters
    ----------
    spec : KernelSpec
        Traced `pl.kernel` on a `plmtl.Mesh`.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``.

    Returns
    -------
    source : str
        Complete MSL declaring its threadgroup width.
    threadgroup_bytes : int
        Declared threadgroup storage per program.

    Raises
    ------
    EmitError
        The kernel uses scratch outside threadgroup memory, aliases, or an operation
        mesh programs do not lower: dot operands outside threadgroup memory, tiles
        that do not divide into fragments, loops over scanned inputs, or primitives
        other than reads, writes, elementwise operations, broadcasts of scalars,
        dots, vectors and loops.
    """
    if spec.aliases:
        raise EmitError("mesh programs do not support input_output_aliases yet")
    if any(info.space != "threadgroup" for info in spec.scratch):
        raise EmitError("mesh programs take plmtl.THREADGROUP scratch only")
    threads = (spec.simdgroups or 0) * _SIMD_WIDTH
    cursor = Cursor()
    cursor.require("simdgroup_mma", _MMA)
    infos = (*spec.inputs, *spec.outputs)
    params = tuple(
        f"{'const ' if index < len(spec.inputs) else ''}device {msl_type(info.dtype)}* "
        f"arg{index} [[buffer({index})]]"
        for index, info in enumerate(infos)
    ) + (
        "uint3 _pid [[threadgroup_position_in_grid]]",
        "uint _sg [[simdgroup_index_in_threadgroup]]",
        "uint _lane [[thread_index_in_simdgroup]]",
        "uint _tid [[thread_index_in_threadgroup]]",
    )
    scope: dict[Var, _Value] = {}
    for index, (var, info) in enumerate(zip(spec.jaxpr.invars, infos, strict=False)):
        shape = tuple(info.array_shape)
        scope[var] = _Ref(
            f"arg{index}", shape, element_strides(shape), msl_type(info.dtype), "device"
        )
    for var, info in zip(spec.jaxpr.invars[len(infos) :], spec.scratch, strict=True):
        if len(info.shape) != 2:
            raise EmitError("plmtl.THREADGROUP scratch is two-dimensional so far")
        rows, columns = info.shape
        name = cursor.fresh("_tg")
        ctype = msl_type(info.dtype)
        cursor.allocate(ctype, (rows * (columns + _PAD),), name=name, space="threadgroup")
        scope[var] = _Ref(name, info.shape, (columns + _PAD, 1), ctype, "threadgroup")
    emit_fragment_coordinates(cursor, "_lane")
    cursor.emit(f"const short _sm = 8 * (_sg / {_WN}) + _fm, _sn = 8 * (_sg % {_WN}) + _fn;")
    _Lowering(cursor, threads).walk(spec.jaxpr.eqns, scope, spec.jaxpr.outvars)
    source = _kernel_source(kernel_name or spec.name, params, cursor, threads)
    return source, cursor.threadgroup_bytes


def _last_uses(eqns, outvars) -> dict[Var, int]:
    """The index of each variable's last use; block outputs count as used after all."""
    last: dict[Var, int] = {}
    for index, eqn in enumerate(eqns):
        for atom in eqn.invars:
            if isinstance(atom, Var):
                last[atom] = index
    for atom in outvars:
        if isinstance(atom, Var):
            last[atom] = len(eqns)
    return last


class _Lowering:
    """Lowers a mesh program's jaxpr, tracking threadgroup reads and writes for barriers."""

    def __init__(self, cursor: Cursor, threads: int):
        self.cursor = cursor
        self.threads = threads
        self.written: set[str] = set()  # threadgroup tiles written since the last barrier
        self.read: set[str] = set()  # and read since it

    def walk(self, eqns, scope: dict[Var, _Value], outvars) -> None:
        last_use = _last_uses(eqns, outvars)
        for index, eqn in enumerate(eqns):
            name = eqn.primitive.name
            if not eqn.outvars:
                raise EmitError(f"mesh programs do not support primitive {name!r}")
            out = eqn.outvars[0]
            if name in _NESTED:
                self._inline(eqn, scope)
            elif name == "program_id":
                scope[out] = _Scalar(f"int(_pid.{'xyz'[eqn.params['axis']]})", "int")
            elif name == "get":
                self._get(eqn, scope)
            elif name == "swap":
                self._swap(eqn, scope)
            elif name == "dot_general":
                scope[out] = self._dot(eqn, scope)
            elif name == "broadcast_in_dim":
                scope[out] = self._broadcast(eqn, scope)
            elif name == "iota":
                if len(eqn.params["shape"]) != 1:
                    raise EmitError("mesh programs take one-dimensional iota only")
                ctype = msl_type(eqn.params["dtype"])
                scope[out] = _Indexed(f"{ctype}({_INDEX})", ctype, tuple(eqn.params["shape"]), 0)
            elif name == "scan":
                self._scan(eqn, scope, last_use, index)
            elif name == "while":
                self._while(eqn, scope, last_use, index)
            elif lowers_elementwise(eqn):
                scope[out] = self._elementwise(eqn, scope, last_use, index)
            else:
                raise EmitError(f"mesh programs do not support primitive {name!r}")

    def _value(self, atom, scope) -> _Value:
        if isinstance(atom, Literal):
            value = literal(atom)
            return _Scalar(value.expr, value.ctype)
        try:
            return scope[atom]
        except KeyError as error:
            raise EmitError(f"mesh program value {atom} is not defined") from error

    def _vector(self, value: _Value) -> _Value:
        """A one-dimensional read of a device array as an _Indexed value."""
        if not (isinstance(value, _Load) and len(value.view.shape) == 1):
            return value
        view = value.view
        if view.ref.space != "device":
            raise EmitError("mesh programs read vectors from device arrays only")
        element = f"{self._pointer(view)}[{_INDEX} * {view.strides[0]}]"
        return _Indexed(element, view.ref.ctype, view.shape, 0)

    def _broadcast(self, eqn: JaxprEqn, scope) -> _Value:
        operand = self._vector(self._value(eqn.invars[0], scope))
        shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
        if isinstance(operand, (_Scalar, _Splat)):
            return _Splat(operand.expr, operand.ctype, shape)
        if isinstance(operand, _Indexed):
            axis = eqn.params["broadcast_dimensions"][operand.axis]
            return _Indexed(operand.template, operand.ctype, shape, axis)
        raise EmitError("mesh programs broadcast scalars and vectors only")

    def _inline(self, eqn: JaxprEqn, scope) -> None:
        nested = eqn.params[_NESTED[eqn.primitive.name]]
        jaxpr = getattr(nested, "jaxpr", nested)
        if getattr(nested, "consts", ()) or jaxpr.constvars:
            raise EmitError("mesh programs cannot inline a nested jaxpr with constants")
        inner = {v: self._value(a, scope) for v, a in zip(jaxpr.invars, eqn.invars, strict=True)}
        self.walk(jaxpr.eqns, inner, jaxpr.outvars)
        for outvar, result in zip(eqn.outvars, jaxpr.outvars, strict=True):
            scope[outvar] = self._value(result, inner)

    # Barriers.

    def _barrier(self) -> None:
        self.cursor.barrier()
        self.written.clear()
        self.read.clear()

    def _reading(self, ref: _Ref) -> None:
        if ref.space == "threadgroup":
            if ref.expr in self.written:
                self._barrier()
            self.read.add(ref.expr)

    def _writing(self, ref: _Ref) -> None:
        if ref.space == "threadgroup":
            if ref.expr in self.read or ref.expr in self.written:
                self._barrier()
            self.written.add(ref.expr)

    def _threadgroup_effects(self, jaxpr: Jaxpr, env) -> tuple[set[str], set[str]]:
        """Threadgroup tiles read and written in `jaxpr`, nested loops included."""
        reads: set[str] = set()
        writes: set[str] = set()
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            ref = env.get(eqn.invars[0]) if eqn.invars and isinstance(eqn.invars[0], Var) else None
            if isinstance(ref, _Ref) and ref.space == "threadgroup":
                if name == "get":
                    reads.add(ref.expr)
                elif name == "swap":
                    writes.add(ref.expr)
            nested, atoms = None, eqn.invars
            if name == "scan":
                nested = getattr(eqn.params["jaxpr"], "jaxpr", eqn.params["jaxpr"])
            elif name == "while":
                nested = eqn.params["body_jaxpr"].jaxpr
                atoms = eqn.invars[eqn.params["cond_nconsts"] :]
            elif name in _NESTED:
                nested = getattr(eqn.params[_NESTED[name]], "jaxpr", eqn.params[_NESTED[name]])
            if nested is not None:
                inner = {
                    v: env.get(a)
                    for v, a in zip(nested.invars, atoms, strict=False)
                    if isinstance(a, Var)
                }
                r, w = self._threadgroup_effects(nested, inner)
                reads |= r
                writes |= w
        return reads, writes

    # Refs.

    def _view(self, ref: _Ref, eqn: JaxprEqn, first: int, scope) -> _View:
        transforms = eqn.params["tree"].unflatten(eqn.invars[first:])
        if not transforms:
            return _View(ref, "0", ref.shape, ref.strides, 0)
        if len(transforms) != 1 or not hasattr(transforms[0], "indices"):
            raise EmitError("mesh programs index refs with one indexer of slices and scalars")
        terms: list[str] = []
        shape: list[int] = []
        strides: list[int] = []
        align = 0
        for index, stride in zip(transforms[0].indices, ref.strides, strict=True):
            if isinstance(index, pl.Slice):
                if index.stride != 1:
                    raise EmitError("mesh programs slice refs with unit strides only")
                start = index.start
                shape.append(int(index.size))
                strides.append(stride)
            else:
                start = index
            if isinstance(start, int):
                expr, known = str(start), start * stride
            elif isinstance(start, Literal):
                expr, known = literal(start).expr, int(start.val) * stride
            else:
                value = self._value(start, scope)
                if not isinstance(value, _Scalar):
                    raise EmitError("mesh program ref indices are scalars")
                expr, known = value.expr, stride
            if expr != "0":
                terms.append(f"({expr}) * {stride}")
                align = math.gcd(align, known)
        offset = " + ".join(terms) or "0"
        return _View(ref, offset, tuple(shape), tuple(strides), align)

    def _get(self, eqn: JaxprEqn, scope) -> None:
        ref = self._value(eqn.invars[0], scope)
        if not isinstance(ref, _Ref):
            raise EmitError("mesh programs read refs passed to the kernel")
        view = self._view(ref, eqn, 1, scope)
        if view.shape:
            scope[eqn.outvars[0]] = _Load(view)
        else:
            self._reading(ref)
            scope[eqn.outvars[0]] = _Scalar(f"{ref.expr}[{view.offset}]", ref.ctype)

    def _swap(self, eqn: JaxprEqn, scope) -> None:
        ref = self._value(eqn.invars[0], scope)
        if not isinstance(ref, _Ref):
            raise EmitError("mesh programs write refs passed to the kernel")
        view = self._view(ref, eqn, 2, scope)
        value = self._value(eqn.invars[1], scope)
        if isinstance(value, _Load):
            self._copy(view, value.view)
        elif isinstance(value, (_Fragments, _Dot, _Splat)) and len(view.shape) == 2:
            self._store(view, self._fragments(value, view.shape))
        elif isinstance(value, _Scalar) and not view.shape:
            self._writing(ref)
            with self.cursor.block("if (_tid == 0)"):
                self.cursor.emit(f"{ref.expr}[{view.offset}] = {ref.ctype}({value.expr});")
        else:
            raise EmitError("mesh programs write a block read, a dot result or a scalar")
        for outvar in eqn.outvars:  # the old value, unused by plain stores
            scope[outvar] = _Scalar("0", ref.ctype)

    def _pointer(self, view: _View) -> str:
        return view.ref.expr if view.offset == "0" else f"({view.ref.expr} + {view.offset})"

    def _copy(self, dst: _View, src: _View) -> None:
        """Every thread copies its share, four elements at a time where rows allow."""
        if dst.shape != src.shape or len(dst.shape) != 2:
            raise EmitError("mesh programs copy equal two-dimensional blocks")
        self._reading(src.ref)
        self._writing(dst.ref)
        rows, columns = dst.shape
        ctype = dst.ref.ctype
        width = 4 if columns % 4 == 0 and ctype in ("float", "half") else 1
        units, per_row = rows * columns // width, columns // width
        copies = -(-units // self.threads)

        def address(view: _View, readonly: bool) -> str:
            element = f"{self._pointer(view)} + _r * {view.strides[0]} + _k"
            if width == 1:
                return f"({element})[0]"
            aligned = view.align % 4 == 0 and view.strides[0] % 4 == 0
            vector = f"{ctype}4" if aligned else f"packed_{ctype}4"
            space = f"{'const ' if readonly else ''}{view.ref.space}"
            return f"*({space} {vector}*)({element})"

        cursor = self.cursor
        with cursor.block(f"{_UNROLL} for (short _c = 0; _c < {copies}; ++_c)"):
            cursor.emit(f"const uint _e = _tid + _c * {self.threads};")
            guard = cursor.block(f"if (_e < {units})") if units % self.threads else nullcontext()
            with guard:
                cursor.emit(f"const uint _r = _e / {per_row}, _k = (_e % {per_row}) * {width};")
                cursor.emit(f"{address(dst, False)} = {address(src, True)};")

    # Fragments.

    def _fragment_counts(self, shape: tuple[int, ...]) -> tuple[int, int]:
        rows, columns = shape
        if rows % (8 * _WM) or columns % (8 * _WN):
            raise EmitError(
                f"mesh program fragments need rows divisible by {8 * _WM} and columns "
                f"by {8 * _WN}, got {shape}"
            )
        return rows // (8 * _WM), columns // (8 * _WN)

    def _new_fragments(self, shape: tuple[int, ...], ctype: str) -> _Fragments:
        fm, fn = self._fragment_counts(shape)
        name = self.cursor.fresh("_f")
        self.cursor.emit(f"vec<{ctype}, 2> {name}[{fm}][{fn}];")
        return _Fragments(name, (shape[0], shape[1]), ctype)

    def _each_element(self, fragments: _Fragments):
        fm, fn = self._fragment_counts(fragments.shape)
        u = _UNROLL
        return (
            self.cursor.block(f"{u} for (short i = 0; i < {fm}; ++i)"),
            self.cursor.block(f"{u} for (short j = 0; j < {fn}; ++j)"),
            self.cursor.block(f"{u} for (short c = 0; c < 2; ++c)"),
        )

    def _assign(self, target: _Fragments, value: object) -> None:
        """`target` = `value`, element by element."""
        a, b, c = self._each_element(target)
        with a, b, c:
            self.cursor.emit(f"{target.name}[i][j][c] = {self._element(value)};")

    def _element(self, value: object) -> str:
        """`value` at fragment element [i][j][c] of the enclosing loops."""
        if isinstance(value, _Fragments):
            return f"{value.name}[i][j][c]"
        if isinstance(value, (_Scalar, _Splat)):
            return value.expr
        if isinstance(value, _Indexed) and len(value.shape) == 2:
            row, column = f"_sm + i * {8 * _WM}", f"_sn + j * {8 * _WN} + c"
            return value.at(row if value.axis == 0 else column)
        raise EmitError("mesh programs combine blocks with blocks, vectors and scalars only")

    def _fragments(self, value: object, shape: tuple[int, ...]) -> _Fragments:
        """`value` as fragments of `shape`, computed if not already."""
        if isinstance(value, _Fragments):
            return value
        if isinstance(value, _Dot):
            fragments = self._new_fragments(shape, "float")
            self._assign(fragments, _Scalar("0.0f", "float"))
            self._mma(fragments, value)
            return fragments
        if isinstance(value, (_Splat, _Indexed)):
            fragments = self._new_fragments(shape, value.ctype)
            self._assign(fragments, value)
            return fragments
        raise EmitError("mesh programs keep blocks as fragments of a dot result or a scalar")

    def _dot(self, eqn: JaxprEqn, scope) -> _Dot:
        a, b = (self._value(atom, scope) for atom in eqn.invars)
        if not (isinstance(a, _Load) and isinstance(b, _Load)):
            raise EmitError("mesh programs take dot operands as reads of plmtl.THREADGROUP tiles")
        if a.view.ref.space != "threadgroup" or b.view.ref.space != "threadgroup":
            raise EmitError("stage dot operands in plmtl.THREADGROUP scratch first")
        (contract, batch) = eqn.params["dimension_numbers"]
        dims = (tuple(contract[0]), tuple(contract[1]))
        if dims not in (((1,), (0,)), ((1,), (1,))) or tuple(map(tuple, batch)) != ((), ()):
            raise EmitError("mesh programs lower (m, k) @ (k, n) and (m, k) @ (n, k).T dots only")
        transposed = dims[1] == (1,)
        if a.view.ref.ctype != "float" or b.view.ref.ctype != "float":
            raise EmitError("mesh program dots take float32 tiles so far")
        if a.view.shape[1] % 8 or a.view.strides[1] != 1 or b.view.strides[1] != 1:
            raise EmitError("mesh program dots need contiguous rows and k divisible by 8")
        self._fragment_counts((a.view.shape[0], b.view.shape[0 if transposed else 1]))
        return _Dot(a.view, b.view, transposed)

    def _mma(self, target: _Fragments, dot: _Dot) -> None:
        """target += a @ b, each SIMD group multiplying its fragments in serpentine order."""
        self._reading(dot.a.ref)
        self._reading(dot.b.ref)
        fm, fn = self._fragment_counts(target.shape)
        lda, ldb = dot.a.strides[0], dot.b.strides[0]
        cursor, u = self.cursor, _UNROLL
        a_ptr, b_ptr = cursor.fresh("_as"), cursor.fresh("_bs")
        cursor.emit(
            f"const threadgroup float* {a_ptr} = {self._pointer(dot.a)} + _sm * {lda} + _fn;"
        )
        # The lane's B elements are rows kk + _fm, columns _sn + 16 j + {0, 1};
        # transposed, those are columns kk + _fm of rows _sn + 16 j + {0, 1}.
        b_start = f"_sn * {ldb} + _fm" if dot.transposed else f"_fm * {ldb} + _sn"
        cursor.emit(f"const threadgroup float* {b_ptr} = {self._pointer(dot.b)} + {b_start};")
        with cursor.block(f"{u} for (short kk = 0; kk < {dot.a.shape[1]}; kk += 8)"):
            cursor.emit(f"simdgroup_frag _a[{fm}], _b[{fn}];")
            with cursor.block(f"{u} for (short i = 0; i < {fm}; ++i)"):
                row = f"i * {8 * _WM * lda} + kk"
                cursor.emit(f"_a[i] = simdgroup_frag({a_ptr}[{row}], {a_ptr}[{row} + 1]);")
            with cursor.block(f"{u} for (short j = 0; j < {fn}; ++j)"):
                if dot.transposed:
                    first, second = f"j * {8 * _WN * ldb} + kk", f"j * {8 * _WN * ldb} + {ldb} + kk"
                else:
                    first = f"kk * {ldb} + j * {8 * _WN}"
                    second = f"{first} + 1"
                cursor.emit(f"_b[j] = simdgroup_frag({b_ptr}[{first}], {b_ptr}[{second}]);")
            with (
                cursor.block(f"{u} for (short i = 0; i < {fm}; ++i)"),
                cursor.block(f"{u} for (short j = 0; j < {fn}; ++j)"),
            ):
                cursor.emit(f"const short n = (i % 2) ? ({fn} - 1 - j) : j;")
                cursor.emit(f"simdgroup_mma({target.name}[i][n], _a[i], _b[n]);")

    def _store(self, view: _View, fragments: _Fragments) -> None:
        self._writing(view.ref)
        fm, fn = self._fragment_counts(fragments.shape)
        cursor, u = self.cursor, _UNROLL
        pointer = self._pointer(view)
        with (
            cursor.block(f"{u} for (short i = 0; i < {fm}; ++i)"),
            cursor.block(f"{u} for (short j = 0; j < {fn}; ++j)"),
        ):
            cursor.emit(
                f"const uint _o = (_sm + i * {8 * _WM}) * {view.strides[0]} + _sn + j * {8 * _WN};"
            )
            for c in range(2):
                element = f"{fragments.name}[i][j][{c}]"
                cursor.emit(f"{pointer}[_o + {c}] = {view.ref.ctype}({element});")

    # Elementwise and loops.

    def _elementwise(self, eqn: JaxprEqn, scope, last_use, index) -> _Value:
        operands = [self._vector(self._value(atom, scope)) for atom in eqn.invars]
        ctype = msl_type(shaped(eqn.outvars[0].aval).dtype)
        shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
        if not shape:
            scalars = [op for op in operands if isinstance(op, _Scalar)]
            if len(scalars) != len(operands):
                raise EmitError("mesh program scalars combine with scalars only")
            expression = scalar_expression(
                self.cursor, eqn, [CVal(op.expr, (), op.ctype) for op in scalars]
            )
            local = self.cursor.fresh("_s")
            self.cursor.emit(f"const {ctype} {local} = {expression};")
            return _Scalar(local, ctype)
        dots = [op for op in operands if isinstance(op, _Dot)]
        accs = [(atom, op) for atom, op in zip(eqn.invars, operands) if isinstance(op, _Fragments)]
        if eqn.primitive.name == "add" and len(dots) == 1 and len(accs) == 1:
            # acc + dot: accumulate in place when acc is not needed afterwards.
            ((acc_atom, target),) = accs
            if not (isinstance(acc_atom, Var) and last_use.get(acc_atom) == index):
                acc, target = target, self._new_fragments(shape, ctype)
                self._assign(target, acc)
            self._mma(target, dots[0])
            return target
        values = [self._fragments(op, shape) if isinstance(op, _Dot) else op for op in operands]
        uniform = [op for op in values if isinstance(op, (_Scalar, _Splat))]
        if len(uniform) == len(values):
            expression = scalar_expression(
                self.cursor, eqn, [CVal(op.expr, (), op.ctype) for op in uniform]
            )
            return _Splat(expression, ctype, shape)
        axes = {op.axis for op in values if isinstance(op, _Indexed)}
        vectors = [op for op in values if isinstance(op, (_Scalar, _Splat, _Indexed))]
        if len(axes) == 1 and len(vectors) == len(values):
            # A vector stays an expression of its index: evaluated where it meets a block.
            template = scalar_expression(
                self.cursor,
                eqn,
                [
                    CVal(op.template, (), op.ctype)
                    if isinstance(op, _Indexed)
                    else CVal(op.expr, (), op.ctype)
                    for op in vectors
                ],
            )
            return _Indexed(template, ctype, shape, axes.pop())
        if len(shape) != 2:
            raise EmitError("mesh programs combine vectors along one axis only")
        result = self._new_fragments(shape, ctype)
        a, b, c = self._each_element(result)
        with a, b, c:
            expression = scalar_expression(
                self.cursor,
                eqn,
                [CVal(self._element(op), (), getattr(op, "ctype", ctype)) for op in values],
            )
            self.cursor.emit(f"{result.name}[i][j][c] = {expression};")
        return result

    def _scan(self, eqn: JaxprEqn, scope, last_use, index) -> None:
        params = eqn.params
        shape = _scan_shape(eqn)
        consts, carries = shape.num_consts, shape.num_carry
        body = params["jaxpr"]
        if isinstance(body, ClosedJaxpr):
            if body.consts:
                raise EmitError("mesh programs cannot lower a loop body with constants")
            body = body.jaxpr
        if len(eqn.invars) != consts + carries or len(body.outvars) != carries:
            raise EmitError("mesh programs lower loops without scanned inputs or outputs")
        if params["reverse"]:
            raise EmitError("mesh programs lower forward loops only")
        header = f"for (int _it = 0; _it < {params['length']}; ++_it)"
        self._loop(header, body, eqn.invars[:consts], eqn, scope, last_use, index)

    def _while(self, eqn: JaxprEqn, scope, last_use, index) -> None:
        """A loop testing a uniform scalar condition before each iteration."""
        params = eqn.params
        cond, body = params["cond_jaxpr"], params["body_jaxpr"]
        if cond.consts or body.consts:
            raise EmitError("mesh programs cannot lower a loop body with constants")
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        test = (cond.jaxpr, eqn.invars[:nc])
        consts = eqn.invars[nc : nc + nb]
        self._loop("while (true)", body.jaxpr, consts, eqn, scope, last_use, index, test)

    def _loop(self, header, body: Jaxpr, consts, eqn, scope, last_use, index, test=None) -> None:
        """Lower `body` under `header` with the carries of `eqn`, which follow
        `consts` among its operands; `test` is a condition jaxpr and its
        constants, evaluated on the carries before each iteration."""
        inits = eqn.invars[len(eqn.invars) - len(eqn.outvars) :]
        inner: dict[Var, _Value] = {
            v: self._value(a, scope) for v, a in zip(body.invars, consts, strict=False)
        }
        state: list[_Scalar | _Fragments] = []
        for v, atom in zip(body.invars[len(consts) :], inits, strict=True):
            init = self._value(atom, scope)
            carry: _Scalar | _Fragments
            if isinstance(init, _Scalar):
                name = self.cursor.fresh("_c")
                self.cursor.emit(f"{init.ctype} {name} = {init.expr};")
                carry = _Scalar(name, init.ctype)
            else:
                shape = tuple(int(d) for d in shaped(v.aval).shape)
                reusable = (
                    isinstance(init, _Fragments)
                    and isinstance(atom, Var)
                    and last_use.get(atom) == index
                )
                if reusable:
                    carry = init
                else:
                    carry = self._new_fragments(shape, msl_type(shaped(v.aval).dtype))
                    self._assign(
                        carry, init if isinstance(init, _Splat) else self._fragments(init, shape)
                    )
            inner[v] = carry
            state.append(carry)
        reads, writes = self._threadgroup_effects(body, inner)
        cursor = self.cursor
        with cursor.block(header):
            if reads & writes:  # the previous iteration's reads and writes are pending
                self.read |= reads
                self.written |= writes
            if test is not None:
                jaxpr, test_consts = test
                env = {v: self._value(a, scope) for v, a in zip(jaxpr.invars, test_consts)}
                env.update(zip(jaxpr.invars[len(test_consts) :], state, strict=True))
                self.walk(jaxpr.eqns, env, jaxpr.outvars)
                flag = self._value(jaxpr.outvars[0], env)
                if not isinstance(flag, _Scalar):
                    raise EmitError("mesh program loop conditions are scalars")
                cursor.emit(f"if (!({flag.expr})) break;")
            self.walk(body.eqns, inner, body.outvars)
            results = [self._value(atom, inner) for atom in body.outvars]
            # Scalars update in two phases: a carry may read another's old value.
            updates = []
            for carry, result in zip(state, results, strict=True):
                if isinstance(carry, _Scalar):
                    if not isinstance(result, _Scalar):
                        raise EmitError("mesh program loop carries keep their kind")
                    local = cursor.fresh("_n")
                    cursor.emit(f"const {carry.ctype} {local} = {result.expr};")
                    updates.append((carry.expr, local))
                elif not (isinstance(result, _Fragments) and result.name == carry.name):
                    self._assign(carry, self._fragments(result, carry.shape))
            for name, local in updates:
                cursor.emit(f"{name} = {local};")
        for outvar, carry in zip(eqn.outvars, state, strict=True):
            scope[outvar] = carry
