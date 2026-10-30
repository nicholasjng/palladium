"""Row programs: each Pallas program runs as one threadgroup over a block of rows.

Selected with `CompilerParams(program="threadgroup")`. The kernel stays plain
Pallas on a 1D grid of row tiles. Every value is classified by the logical
axes it varies over:

- *uniform* values vary at most by row: scalars, per-row reductions, row
  vectors. They live in registers, one per row, identical across the threads
  working on that row.
- *lane* values vary by column: blocks, column vectors such as weights, and
  everything computed from them. Each thread owns the columns
  `t, t + P, t + 2P, ...` of a row, where `P` threads share the row. A block is
  read from device memory once per row into a per-thread register array;
  each pass over the columns recomputes the lane values it needs from there.

A reduction over the columns is one pass, finished with a SIMD-group
reduction and, when a whole threadgroup shares the row, a combine through
threadgroup memory. A store of a lane value is one more pass. A row softmax
is therefore one read, three passes (max, sum of exp, store) and one write.

Slices, pads and concatenations along the columns remap the column at which
a value is evaluated; shifted columns of a block are read from device memory
(and its caches) instead of registers. Reductions over the rows accumulate
across the row loop; everything that uses them runs after the loop and is
stored to per-tile `(1, columns)` or `(1,)` outputs.

Rows up to 256 columns go to one SIMD group each, four rows at a time; wider
rows get the whole threadgroup, sized at about four columns per thread (as
MLX sizes its single-row kernels), up to 1024 threads.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Generator

from jax.extend.core import JaxprEqn, Literal, Var

from palladium.emit.addressing import block_offset
from palladium.emit.core import Cursor, CVal, Environment, msl_type, shaped
from palladium.emit.numeric import extremum, extremum_identity
from palladium.emit.rules.elementwise import lowers_elementwise, scalar_expression
from palladium.emit.tensorops._shared import SIMDGROUPS, _index_map_is, _kernel_source
from palladium.errors import EmitError
from palladium.trace import BlockInfo, KernelSpec

ROW, COLUMN = "row", "column"
_SIMD_WIDTH = 32
_NARROW_COLUMNS = 256  # at most this many columns: one SIMD group per row
_COLUMNS_PER_THREAD = 4
_MAX_THREADS = 1024
_MAX_CACHED = 16  # per-thread block elements kept in registers
_UNROLL = '_Pragma("clang loop unroll(full)")'
_REDUCTIONS = {"reduce_sum": "simd_sum", "reduce_max": "simd_max", "reduce_min": "simd_min"}
_SCANS = ("cumsum", "cumprod", "cummax", "cummin")
# Primitives whose nested jaxpr is inlined: `jnp.where` stages `jit`,
# `jax.nn.softmax` stages `custom_jvp_call`.
_NESTED = {"jit": "jaxpr", "pjit": "jaxpr", "custom_jvp_call": "call_jaxpr"}


@dataclasses.dataclass(frozen=True)
class _Value:
    """A jaxpr value inside the row loop or after it.

    `axes` names the logical axis of each dimension (None for extent 1), and
    `extent` is the number of columns of a lane value. Uniform values carry
    `expr`. A lane value is computed one of four ways: `read`, a device load
    at any column `$c`; `cache`, a register at the thread's own column slot
    `$k`; `eqn` and `operands`; or `pieces`, a column remap in which column `c`
    takes piece `j` at `c - start_j`, for the last `start_j <= c`. `per_row`
    values depend on the current row and do not exist after the row loop.
    """

    axes: tuple[str | None, ...]
    ctype: str
    expr: str | None = None
    read: str | None = None
    cache: str | None = None
    eqn: JaxprEqn | None = None
    operands: tuple[_Value, ...] = ()
    pieces: tuple[tuple[_Value, int], ...] = ()
    extent: int = 0
    per_row: bool = True

    @property
    def by_column(self) -> bool:
        """Whether the content differs per column; a broadcast uniform value has
        the column axis but not per-column content."""
        return self.expr is None


@dataclasses.dataclass(frozen=True)
class _Layout:
    """How threads share rows: `per_row` threads per row, `width` per group."""

    per_row: int
    width: int
    slots: int  # columns per thread, ceil(columns / per_row)

    @property
    def whole_group(self) -> bool:
        return self.per_row == self.width

    @property
    def thread(self) -> str:
        return "_tid" if self.whole_group else "_lane"


def _extent(axes: tuple[str | None, ...], shape: tuple[int, ...]) -> int:
    return shape[axes.index(COLUMN)] if COLUMN in axes else 0


def lower_row_program(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Emit a kernel running each Pallas program as one threadgroup of rows.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel on a 1D grid of row tiles.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``.

    Returns
    -------
    source : str
        Complete MSL using one threadgroup per Pallas program. It declares its
        group width with a ``// palladium: threadgroup width N`` line.
    threadgroup_bytes : int
        Declared shared storage per group.

    Raises
    ------
    EmitError
        The kernel is not a row program: its refs are not `(rows, columns)`
        blocks spanning all columns, `(rows,)` row vectors, `(columns,)`
        column vectors, or per-tile `(1, columns)` and `(1,)` outputs; it
        broadcasts across rows or shifts along them; or it uses a primitive
        other than full-block reads and stores, elementwise operations,
        broadcasts, reshapes, slices, pads and concatenations, reductions, and
        scans along the columns.
    """
    if len(spec.grid) != 1 or spec.scratch or spec.aliases or spec.jaxpr.constvars:
        raise EmitError("row programs need a 1D grid without scratch, aliases or constants")
    infos = (*spec.inputs, *spec.outputs)
    blocks = [info for info in infos if len(info.block_shape) == 2]
    if not blocks:
        raise EmitError("row programs need at least one (rows, columns) block")
    rows_per_tile = max(info.block_shape[0] for info in blocks)
    columns = blocks[0].block_shape[1]
    rows = next(i.array_shape[0] for i in blocks if i.block_shape[0] == rows_per_tile)
    (programs,) = spec.grid
    if programs != -(-rows // rows_per_tile):
        raise EmitError("row programs need one program per row tile")

    if columns <= _NARROW_COLUMNS:
        layout = _Layout(_SIMD_WIDTH, SIMDGROUPS * _SIMD_WIDTH, -(-columns // _SIMD_WIDTH))
    else:
        threads = -(-columns // _COLUMNS_PER_THREAD)
        per_row = min(_MAX_THREADS, -(-threads // _SIMD_WIDTH) * _SIMD_WIDTH)
        layout = _Layout(per_row, per_row, -(-columns // per_row))
    cursor = Cursor()
    env = Environment()
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
    cursor.emit(f"// palladium: threadgroup width {layout.width}")
    if layout.whole_group:  # SIMD-group partials of reductions and scans, as bits
        size = _SIMD_WIDTH * (min(layout.slots, _MAX_CACHED) + 1)
        cursor.allocate("uint", (size,), name="_partials", space="threadgroup")
    refs: dict[Var, tuple[str, str, BlockInfo]] = {}
    for index, (ref, info) in enumerate(zip(spec.jaxpr.invars, infos, strict=True)):
        block, array = info.block_shape, info.array_shape
        if block == (rows_per_tile, columns) == (rows_per_tile, array[1]) and _index_map_is(
            info, (0, None)
        ):
            kind = "tile"
        elif block == (rows_per_tile,) and _index_map_is(info, (0,)):
            kind = "rows"
        elif block == (columns,) == array and _index_map_is(info, (None,)):
            kind = "columns"
        elif (
            block == (1, columns)
            and array == (programs, columns)
            and _index_map_is(info, (0, None))
        ):
            kind = "tile columns"
        elif block == (1,) and array == (programs,) and _index_map_is(info, (0,)):
            kind = "tile scalar"
        else:
            raise EmitError(
                "row-program refs must be (rows, columns) tiles of one array shape, (rows,) "
                "row vectors, (columns,) column vectors, or per-tile (1, columns) and (1,) "
                "outputs"
            )
        if kind == "tile" and array[0] != rows:
            raise EmitError("row-program tiles must share one array shape")
        offset = block_offset(env, cursor, spec, info)
        pointer = f"arg{index}" if offset == "0" else f"(arg{index} + {offset})"
        refs[ref] = (pointer, kind, info)

    even_rows = rows % rows_per_tile == 0
    valid_rows = (
        str(rows_per_tile)
        if even_rows
        else f"min({rows_per_tile}, {rows} - (int)_pid.x * {rows_per_tile})"
    )
    body = _RowBody(cursor, env, refs, rows_per_tile, columns, layout, spec, even_rows)
    if layout.whole_group:
        header = f"for (uint _row = 0; _row < {valid_rows}; ++_row)"
    else:
        header = f"for (uint _row = _sg; _row < {valid_rows}; _row += {SIMDGROUPS})"
    body.loop_at = len(cursor.lines)
    with cursor.block(header):
        body.walk(spec.jaxpr.eqns, {})
    body.finish()
    source = _kernel_source(kernel_name or spec.name, params, cursor)
    return source, cursor.threadgroup_bytes


class _RowBody:
    """Lowers a jaxpr into the per-row loop and the epilogue after it."""

    def __init__(self, cursor, env, refs, rows_per_tile, columns, layout, spec, even_rows):
        self.cursor = cursor
        self.env = env
        self.refs = refs
        self.rows_per_tile = rows_per_tile
        self.columns = columns
        self.layout = layout
        self.even_rows = even_rows
        self.outputs = set(spec.jaxpr.invars[len(spec.inputs) :])
        self.loop_at = 0  # where declarations before the row loop go
        self.in_loop = True
        # Work that needs a reduction over all rows, emitted after the loop.
        self.deferred: list[Callable[[], object]] = []
        self.deferred_vars: set[Var] = set()
        # Row-reduction accumulators: name, type, combine, slots (0 for a scalar).
        self.accumulators: list[tuple[str, str, Callable[[str, str], str], int]] = []

    def walk(self, eqns, scope: dict[Var, _Value]) -> dict[Var, _Value]:
        for eqn in eqns:
            name = eqn.primitive.name
            if self.in_loop and self._defers(eqn):
                self.deferred.append(lambda eqn=eqn, scope=scope: self.walk([eqn], scope))
                self.deferred_vars.update(eqn.outvars)
            elif name in _NESTED:
                self._inline(eqn, scope)
            elif name == "get":
                scope[eqn.outvars[0]] = self._get(eqn)
            elif name == "swap":
                self._swap(eqn, scope)
            elif name in _REDUCTIONS:
                scope[eqn.outvars[0]] = self._reduce(eqn, scope)
            elif name in _SCANS:
                scope[eqn.outvars[0]] = self._scan(eqn, scope)
            elif name in ("broadcast_in_dim", "reshape", "squeeze", "expand_dims"):
                scope[eqn.outvars[0]] = self._reshape(eqn, scope)
            elif name in ("slice", "pad", "concatenate"):
                scope[eqn.outvars[0]] = self._remap(eqn, scope)
            elif name in ("stop_gradient", "copy"):  # the identity when lowering
                scope[eqn.outvars[0]] = self._value(eqn.invars[0], scope)
            elif lowers_elementwise(eqn):
                scope[eqn.outvars[0]] = self._elementwise(eqn, scope)
            else:
                raise EmitError(f"row programs do not support primitive {name!r}")
        return scope

    def _defers(self, eqn) -> bool:
        """Whether `eqn` needs a reduction over all rows, so runs after the loop."""
        per_tile = (
            eqn.primitive.name == "swap"
            and self.rows_per_tile > 1
            and self.refs[eqn.invars[0]][1].startswith("tile ")
        )
        return per_tile or any(isinstance(a, Var) and a in self.deferred_vars for a in eqn.invars)

    def finish(self) -> None:
        """Emit the epilogue: combine row reductions, then the deferred work."""
        self.in_loop = False
        if not self.deferred:
            return
        cursor = self.cursor
        if self.layout.whole_group:  # every thread saw every row
            for emit in self.deferred:
                emit()
            return
        # Each SIMD group saw every fourth row: combine into SIMD group 0.
        for index, (acc, ctype, combine, slots) in enumerate(self.accumulators):
            size = SIMDGROUPS * slots * _SIMD_WIDTH if slots else SIMDGROUPS
            scratch = f"_rows{index}"
            self._declare(f"threadgroup uint {scratch}[{size}];")
            cursor.account("uint", size, "threadgroup")
            if slots:
                with cursor.block(f"for (short _k = 0; _k < {slots}; ++_k)"):
                    slot = f"(_sg * {slots} + _k) * {_SIMD_WIDTH} + _lane"
                    cursor.emit(f"{scratch}[{slot}] = {_to_bits(f'{acc}[_k]', ctype)};")
            else:
                with cursor.block("if (_lane == 0)"):
                    cursor.emit(f"{scratch}[_sg] = {_to_bits(acc, ctype)};")
        cursor.barrier()
        with cursor.block("if (_sg == 0)"):
            for index, (acc, ctype, combine, slots) in enumerate(self.accumulators):
                scratch = f"_rows{index}"
                with cursor.block(f"for (uint g = 1; g < {SIMDGROUPS}; ++g)"):
                    if slots:
                        with cursor.block(f"for (short _k = 0; _k < {slots}; ++_k)"):
                            slot = f"(g * {slots} + _k) * {_SIMD_WIDTH} + _lane"
                            other = self._local(ctype, _from_bits(f"{scratch}[{slot}]", ctype))
                            cursor.emit(f"{acc}[_k] = {combine(f'{acc}[_k]', other)};")
                    else:
                        other = self._local(ctype, _from_bits(f"{scratch}[g]", ctype))
                        cursor.emit(f"{acc} = {combine(acc, other)};")
            for emit in self.deferred:
                emit()

    def _declare(self, line: str) -> None:
        """Put a declaration at kernel scope, before the row loop."""
        self.cursor.lines.insert(self.loop_at, "    " + line)
        self.loop_at += 1

    def _local(self, ctype: str, expr: str) -> str:
        """`expr` as an identifier; combines such as NaN-aware maxima repeat it."""
        if expr.isidentifier():
            return expr
        local = self.cursor.fresh("_e")
        self.cursor.emit(f"const {ctype} {local} = {expr};")
        return local

    def _value(self, atom, scope) -> _Value:
        if isinstance(atom, Literal):
            literal = self.env.val(atom)
            if literal.shape:
                raise EmitError("row programs support scalar literals only")
            return _Value((), literal.ctype, expr=literal.expr, per_row=False)
        try:
            value = scope[atom]
        except KeyError as error:
            raise EmitError(f"row program value {atom} is not defined") from error
        if not self.in_loop and value.per_row:
            raise EmitError(
                "after a reduction over the rows, only per-tile values remain: "
                "reduce single rows away first"
            )
        return value

    def _inline(self, eqn, scope):
        nested = eqn.params[_NESTED[eqn.primitive.name]]
        jaxpr = getattr(nested, "jaxpr", nested)
        if getattr(nested, "consts", ()) or jaxpr.constvars:
            raise EmitError("row programs cannot inline a nested jaxpr with constants")
        inner = dict(zip(jaxpr.invars, (self._value(a, scope) for a in eqn.invars), strict=True))
        self.walk(jaxpr.eqns, inner)
        for outvar, result in zip(eqn.outvars, jaxpr.outvars, strict=True):

            def bind(outvar=outvar, result=result):
                scope[outvar] = self._value(result, inner)

            if isinstance(result, Var) and result in self.deferred_vars:
                self.deferred.append(bind)
                self.deferred_vars.add(outvar)
            else:
                bind()

    def _get(self, eqn) -> _Value:
        ref = eqn.invars[0]
        if ref not in self.refs or eqn.params["tree"].flatten_up_to(()):
            raise EmitError("row programs read whole blocks of their refs")
        pointer, kind, info = self.refs[ref]
        ctype = msl_type(info.dtype)
        # Unit dimensions carry no axis, so reshapes can drop or add them.
        row = ROW if self.rows_per_tile > 1 else None
        if kind == "columns":
            return _Value(
                (COLUMN,), ctype, read=f"{pointer}[$c]", extent=self.columns, per_row=False
            )
        if kind == "rows":
            local = self.cursor.fresh("_r")
            self.cursor.emit(f"const {ctype} {local} = {pointer}[_row];")
            return _Value((row,), ctype, expr=local)
        if kind != "tile":
            raise EmitError("row programs read tiles, row vectors and column vectors")
        read = f"{pointer}[_row * {self.columns} + $c]"
        cache = None
        if self.layout.slots <= _MAX_CACHED:
            # Read this row of the block once; passes read the registers.
            cache = self.cursor.fresh("_x")
            self.cursor.emit(f"{ctype} {cache}[{self.layout.slots}];")
            with self._pass(self.columns) as (column, slot):
                self.cursor.emit(f"{cache}[{slot}] = {read.replace('$c', column)};")
            cache = f"{cache}[$k]"
        return _Value((row, COLUMN), ctype, read=read, cache=cache, extent=self.columns)

    def _swap(self, eqn, scope):
        ref, value_atom = eqn.invars[0], eqn.invars[1]
        if ref not in self.refs or eqn.params["tree"].flatten_up_to(()):
            raise EmitError("row programs write whole blocks of their refs")
        if ref not in self.outputs:
            raise EmitError("row programs write only to output refs")
        pointer, kind, info = self.refs[ref]
        value = self._value(value_atom, scope)
        ctype = msl_type(info.dtype)
        if kind in ("tile", "tile columns") and value.extent in (0, self.columns):
            row = f"_row * {self.columns} + " if kind == "tile" else ""
            with self._pass(self.columns) as (column, slot):
                element = self._lane_expr(value, column, slot, {})
                self.cursor.emit(f"{pointer}[{row}{column}] = {ctype}({element});")
        elif kind in ("rows", "tile scalar") and not value.by_column:
            index = "_row" if kind == "rows" else "0"
            with self.cursor.block(f"if ({self.layout.thread} == 0)"):
                self.cursor.emit(f"{pointer}[{index}] = {ctype}({value.expr});")
        else:
            raise EmitError("this value does not fit the shape of its output")
        for outvar in eqn.outvars:  # the old block value, unused by full stores
            scope[outvar] = _Value((), ctype, expr="0", per_row=False)

    def _reduce(self, eqn, scope) -> _Value:
        operand = self._value(eqn.invars[0], scope)
        axes = set(eqn.params["axes"])
        reduced = {operand.axes[d] for d in axes} - {None}
        kept = tuple(a for d, a in enumerate(operand.axes) if d not in axes)
        ctype = msl_type(shaped(eqn.outvars[0].aval).dtype)
        op = eqn.primitive.name
        if op == "reduce_sum":
            identity = f"{ctype}(0)"
        else:
            identity = extremum_identity(op.removeprefix("reduce_"), ctype)

        def combine(a: str, b: str) -> str:
            if op == "reduce_sum":
                return f"({a} + {b})"
            return extremum(op.removeprefix("reduce_"), ctype, a, b)

        if not reduced:  # over unit dimensions only
            return dataclasses.replace(operand, axes=kept)
        # A reduction over both axes reduces each row, then the rows.
        if COLUMN in reduced:
            operand = self._reduce_columns(operand, kept, ctype, op, identity, combine)
        if ROW in reduced:
            return self._reduce_rows(eqn, operand, kept, ctype, identity, combine)
        return operand

    def _reduce_columns(self, operand, kept, ctype, op, identity, combine) -> _Value:
        if not operand.by_column:  # the same value in every column
            total = f"({operand.expr} * {ctype}({operand.extent}))"
            return dataclasses.replace(
                operand,
                axes=kept,
                extent=0,
                expr=total if op == "reduce_sum" else operand.expr,
            )
        acc = self.cursor.fresh("_acc")
        self.cursor.emit(f"{ctype} {acc} = {identity};")
        with self._pass(operand.extent) as (column, slot):
            element = self._local(ctype, self._lane_expr(operand, column, slot, {}))
            self.cursor.emit(f"{acc} = {combine(acc, element)};")
        simd = _REDUCTIONS[op]
        self.cursor.emit(f"{acc} = {simd}({acc});")
        if self.layout.whole_group:
            # Combine the SIMD groups' partials; every group reads them all.
            groups = self.layout.width // _SIMD_WIDTH
            with self.cursor.block("if (_lane == 0)"):
                self.cursor.emit(f"_partials[_sg] = {_to_bits(acc, ctype)};")
            self.cursor.barrier()
            partial = _from_bits("_partials[_lane]", ctype)
            self.cursor.emit(f"{acc} = {simd}(_lane < {groups} ? {partial} : {identity});")
            self.cursor.barrier()
        # The accumulator is scoped to the row loop when declared in it.
        return _Value(kept, ctype, expr=acc, per_row=self.in_loop)

    def _reduce_rows(self, eqn, operand, kept, ctype, identity, combine) -> _Value:
        """Accumulate over the row loop; the result exists after it, per tile."""
        if not self.even_rows:
            raise EmitError("row programs reduce over the rows only when tiles divide them")
        if not self.in_loop:
            raise EmitError("row programs reduce over the rows once")
        acc = self.cursor.fresh("_rowacc")
        self.deferred_vars.add(eqn.outvars[0])
        if operand.by_column:
            if operand.extent != self.columns:
                raise EmitError("row programs reduce whole rows over the rows")
            slots = self.layout.slots
            self._declare(f"{ctype} {acc}[{slots}] = {{{', '.join([identity] * slots)}}};")
            with self._pass(self.columns) as (column, slot):
                element = self._local(ctype, self._lane_expr(operand, column, slot, {}))
                self.cursor.emit(f"{acc}[{slot}] = {combine(f'{acc}[{slot}]', element)};")
            self.accumulators.append((acc, ctype, combine, slots))
            return _Value(kept, ctype, cache=f"{acc}[$k]", extent=self.columns, per_row=False)
        self._declare(f"{ctype} {acc} = {identity};")
        self.cursor.emit(f"{acc} = {combine(acc, self._local(ctype, operand.expr or ''))};")
        self.accumulators.append((acc, ctype, combine, 0))
        return _Value(kept, ctype, expr=acc, per_row=False)

    def _scan(self, eqn, scope) -> _Value:
        """A cumulative op along the columns, kept in a per-thread register array.

        Slot `k` of the row's threads is the contiguous chunk of columns
        `[k * P, (k + 1) * P)`. Each chunk is scanned within SIMD groups by
        shuffles. With a whole threadgroup on the row, the first SIMD group then
        scans the SIMD-group totals of every chunk at once, so a row costs three
        barriers. Finally each thread adds the totals of earlier chunks.
        """
        operand = self._value(eqn.invars[0], scope)
        if operand.axes[eqn.params["axis"]] != COLUMN:
            raise EmitError("row programs scan along the columns only")
        layout = self.layout
        if operand.extent != self.columns:
            raise EmitError("row programs scan whole rows")
        if layout.slots > _MAX_CACHED:
            raise EmitError(
                f"row programs scan rows of at most {_MAX_CACHED * layout.per_row} columns"
            )
        ctype = msl_type(shaped(eqn.outvars[0].aval).dtype)
        op = eqn.primitive.name.removeprefix("cum")
        identity = {"sum": f"{ctype}(0)", "prod": f"{ctype}(1)"}.get(op) or extremum_identity(
            op, ctype
        )

        def combine(a: str, b: str) -> str:
            return {"sum": f"({a} + {b})", "prod": f"({a} * {b})"}.get(op) or extremum(
                op, ctype, a, b
            )

        reverse = eqn.params["reverse"]
        # A reverse scan runs from the last lane down; lane `total_lane` ends
        # up holding the SIMD group's whole chunk.
        shuffle, toward, total_lane, end_lane = (
            ("simd_shuffle_down", f"_lane + d < {_SIMD_WIDTH}", "0", f"{_SIMD_WIDTH - 1}")
            if reverse
            else ("simd_shuffle_up", "_lane >= d", f"{_SIMD_WIDTH - 1}", "0")
        )
        cursor = self.cursor

        def scan_lanes(value: str) -> None:
            with cursor.block(f"{_UNROLL} for (ushort d = 1; d < {_SIMD_WIDTH}; d <<= 1)"):
                cursor.emit(f"const {ctype} other = {shuffle}({value}, d);")
                with cursor.block(f"if ({toward})"):
                    cursor.emit(f"{value} = {combine('other', value)};")

        result, carry = cursor.fresh("_scan"), cursor.fresh("_carry")
        cursor.emit(f"{ctype} {result}[{layout.slots}];")
        with cursor.block(f"{_UNROLL} for (short _k = 0; _k < {layout.slots}; ++_k)"):
            cursor.emit(f"const uint _c = {layout.thread} + _k * {layout.per_row};")
            element = self._lane_expr(operand, "_c", "_k", {})
            value = cursor.fresh("_v")
            cursor.emit(f"{ctype} {value} = _c < {self.columns} ? {ctype}({element}) : {identity};")
            scan_lanes(value)
            cursor.emit(f"{result}[_k] = {value};")
            if layout.whole_group:
                with cursor.block(f"if (_lane == {total_lane})"):
                    slot = f"_k * {_SIMD_WIDTH} + _sg"
                    cursor.emit(f"_partials[{slot}] = {_to_bits(value, ctype)};")
        if layout.whole_group:
            # The first SIMD group turns each chunk's SIMD-group totals into
            # exclusive prefixes, and records the chunk's total after them.
            groups = layout.width // _SIMD_WIDTH
            cursor.barrier()
            with (
                cursor.block("if (_sg == 0)"),
                cursor.block(f"{_UNROLL} for (short _k = 0; _k < {layout.slots}; ++_k)"),
            ):
                partial = _from_bits(f"_partials[_k * {_SIMD_WIDTH} + _lane]", ctype)
                cursor.emit(f"{ctype} _p = _lane < {groups} ? {partial} : {identity};")
                scan_lanes("_p")
                cursor.emit(f"{ctype} _before = {shuffle}(_p, 1);")
                with cursor.block(f"if (_lane == {end_lane})"):
                    cursor.emit(f"_before = {identity};")
                before = _to_bits("_before", ctype)
                cursor.emit(f"_partials[_k * {_SIMD_WIDTH} + _lane] = {before};")
                with cursor.block(f"if (_lane == {total_lane})"):
                    total_slot = f"{layout.slots * _SIMD_WIDTH} + _k"
                    cursor.emit(f"_partials[{total_slot}] = {_to_bits('_p', ctype)};")
            cursor.barrier()
        cursor.emit(f"{ctype} {carry} = {identity};")
        slots = (
            f"for (short _k = {layout.slots - 1}; _k >= 0; --_k)"
            if reverse
            else f"for (short _k = 0; _k < {layout.slots}; ++_k)"
        )
        with cursor.block(f"{_UNROLL} {slots}"):
            if layout.whole_group:
                before = _from_bits(f"_partials[_k * {_SIMD_WIDTH} + _sg]", ctype)
                total = _from_bits(f"_partials[{layout.slots * _SIMD_WIDTH} + _k]", ctype)
                value = combine(before, f"{result}[_k]")
            else:
                total = f"simd_broadcast({result}[_k], {total_lane})"
                value = f"{result}[_k]"
            cursor.emit(f"const {ctype} _chunk_total = {total};")
            cursor.emit(f"{result}[_k] = {combine(carry, value)};")
            cursor.emit(f"{carry} = {combine(carry, '_chunk_total')};")
        if layout.whole_group:
            cursor.barrier()
        return dataclasses.replace(
            operand,
            cache=f"{result}[$k]",
            read=None,
            eqn=None,
            operands=(),
            pieces=(),
            per_row=self.in_loop,
        )

    def _reshape(self, eqn, scope) -> _Value:
        operand = self._value(eqn.invars[0], scope)
        out_shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
        if eqn.primitive.name == "broadcast_in_dim":
            axes: list[str | None] = [None] * len(out_shape)
            for source, target in zip(
                operand.axes, eqn.params["broadcast_dimensions"], strict=True
            ):
                axes[target] = source
            for target, extent in enumerate(out_shape):
                if axes[target] is not None or extent == 1:
                    continue
                # The axis the broadcast adds: rank-2 values are (rows, columns).
                if len(out_shape) == 2:
                    axes[target] = (ROW, COLUMN)[target]
                elif extent == self.columns != self.rows_per_tile:
                    axes[target] = COLUMN
                elif extent == self.rows_per_tile != self.columns:
                    axes[target] = ROW
                else:
                    raise EmitError("row programs cannot tell which axis this broadcast adds")
        else:  # reshape, squeeze, expand_dims keep the non-unit axes in order
            labelled = [a for a in operand.axes if a is not None]
            axes = [None if extent == 1 else labelled.pop(0) for extent in out_shape]
            if labelled:
                raise EmitError("row programs reshape only by adding or removing unit dimensions")
        named = [a for a in axes if a]
        if len(set(named)) != len(named):
            raise EmitError("row programs cannot repeat an axis")
        return dataclasses.replace(
            operand, axes=tuple(axes), extent=_extent(tuple(axes), out_shape)
        )

    def _remap(self, eqn, scope) -> _Value:
        """A slice, pad or concatenation along the columns, as a column remap."""
        name = eqn.primitive.name
        operands = [self._value(atom, scope) for atom in eqn.invars]
        axes = operands[0].axes
        out_shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
        in_shape = tuple(int(d) for d in shaped(eqn.invars[0].aval).shape)
        if COLUMN not in axes:  # only an identity is left for values without columns
            if name == "concatenate" or in_shape != out_shape:
                raise EmitError("row programs slice, pad and concatenate along the columns only")
            return operands[0]
        column_dim = axes.index(COLUMN)
        if name == "concatenate":
            if eqn.params["dimension"] != column_dim:
                raise EmitError("row programs concatenate along the columns only")
            pieces, start = [], 0
            for atom, value in zip(eqn.invars, operands, strict=True):
                pieces.append((value, start))
                start += int(shaped(atom.aval).shape[column_dim])
        else:
            if name == "slice":
                starts, limits = eqn.params["start_indices"], eqn.params["limit_indices"]
                if any(s not in (None, 1) for s in eqn.params["strides"] or ()):
                    raise EmitError("row programs slice with unit strides only")
                bounds = [
                    (int(s), in_shape[d] - int(l)) for d, (s, l) in enumerate(zip(starts, limits))
                ]
            else:  # pad
                if any(interior for _, _, interior in eqn.params["padding_config"]):
                    raise EmitError("row programs pad without interior padding")
                bounds = [(-int(lo), -int(hi)) for lo, hi, _ in eqn.params["padding_config"]]
            if any(bound != (0, 0) for d, bound in enumerate(bounds) if d != column_dim):
                raise EmitError("row programs slice and pad along the columns only")
            cut = bounds[column_dim][0]
            source = operands[0]
            if name == "slice":
                # Slicing a remap shifts its pieces; slicing anything else is one piece.
                pieces = [(p, s - cut) for p, s in source.pieces] or [(source, -cut)]
            else:
                # Fills only on padded sides; a negative pad crops.
                fill, (low, high) = operands[1], (-cut, -bounds[column_dim][1])
                pieces = [(fill, 0)] * (low > 0) + [(source, low)]
                pieces += [(fill, low + in_shape[column_dim])] * (high > 0)
        return _Value(
            axes,
            operands[0].ctype,
            pieces=tuple(pieces),
            extent=_extent(axes, out_shape),
            per_row=any(v.per_row for v in operands),
        )

    def _elementwise(self, eqn, scope) -> _Value:
        operands = tuple(self._value(atom, scope) for atom in eqn.invars)
        ctype = msl_type(shaped(eqn.outvars[0].aval).dtype)
        shape = tuple(int(d) for d in shaped(eqn.outvars[0].aval).shape)
        axes = next(
            (op.axes for op in operands if len(op.axes) == len(shape)), (None,) * len(shape)
        )
        per_row = any(op.per_row for op in operands)
        if any(op.by_column for op in operands):
            return _Value(
                axes,
                ctype,
                eqn=eqn,
                operands=operands,
                extent=_extent(axes, shape),
                per_row=per_row,
            )
        scalars = [CVal(op.expr or "", (), op.ctype) for op in operands]
        expression = scalar_expression(self.cursor, eqn, scalars)
        if not per_row:  # constant across rows: keep the expression, usable anywhere
            return _Value(axes, ctype, expr=expression, per_row=False)
        local = self.cursor.fresh("_u")
        self.cursor.emit(f"const {ctype} {local} = {expression};")
        return _Value(axes, ctype, expr=local)

    @contextlib.contextmanager
    def _pass(self, extent: int) -> Generator[tuple[str, str]]:
        """A fully unrolled loop over the columns this thread owns, out of
        `extent`; yields the column and the thread's slot for it."""
        layout = self.layout
        slots = -(-extent // layout.per_row)
        with self.cursor.block(f"{_UNROLL} for (short _k = 0; _k < {slots}; ++_k)"):
            self.cursor.emit(f"const uint _c = {layout.thread} + _k * {layout.per_row};")
            if extent % layout.per_row:
                with self.cursor.block(f"if (_c < {extent})"):
                    yield "_c", "_k"
            else:
                yield "_c", "_k"

    def _lane_expr(
        self, value: _Value, column: str, slot: str | None, memo: dict[tuple, str]
    ) -> str:
        """Emit what a thread needs for `value` at `column`; return its expression.

        `slot` names the thread's register slot when `column` is its own
        column of a whole row, and is None at a remapped column.
        """
        if not value.by_column:
            return value.expr or ""
        if value.pieces:
            choices = []
            for piece, start in value.pieces:
                if len(value.pieces) == 1:
                    index = f"({column} + {-start})" if start else column
                    inner_slot = slot if not start else None
                else:  # clamped, so a piece not chosen never reads out of bounds
                    index = f"uint(clamp(int({column}) - {start}, 0, {max(piece.extent, 1) - 1}))"
                    inner_slot = None
                choices.append((self._lane_expr(piece, index, inner_slot, memo), start))
            expression = choices[0][0]
            for element, start in choices[1:]:
                expression = f"(int({column}) >= {start} ? {element} : {expression})"
            return expression
        if slot is not None and value.cache is not None:
            return value.cache.replace("$k", slot)
        if value.read is not None:
            return value.read.replace("$c", f"({column})")
        if value.eqn is None:
            raise EmitError("row programs cannot shift a scan or a row reduction along the columns")
        key = (id(value.eqn), column, slot)
        if key not in memo:
            scalars = [
                CVal(self._lane_expr(op, column, slot, memo), (), op.ctype) for op in value.operands
            ]
            local = self.cursor.fresh("_l")
            expression = scalar_expression(self.cursor, value.eqn, scalars)
            self.cursor.emit(f"const {value.ctype} {local} = {expression};")
            memo[key] = local
        return memo[key]


def _bits_type(ctype: str) -> str:
    """The 32-bit type a value widens to exactly before it is stored as bits."""
    return {"half": "float", "bfloat": "float", "bool": "uint"}.get(ctype, ctype)


def _to_bits(expr: str, ctype: str) -> str:
    return f"as_type<uint>({_bits_type(ctype)}({expr}))"


def _from_bits(expr: str, ctype: str) -> str:
    return f"{ctype}(as_type<{_bits_type(ctype)}>({expr}))"
