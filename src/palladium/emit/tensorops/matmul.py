"""Recognition and emission for tiled TensorOps matrix products."""

from __future__ import annotations

import dataclasses

from jax.extend.core import JaxprEqn, Literal, Var

from palladium.emit.cooperative import CooperativeValue, emit_elementwise_store
from palladium.emit.core import Cursor, CVal, Environment
from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import (
    _index_map_is,
    _is_empty_get,
    _kernel_source,
    _shape,
    _TensorOpsMatmul,
    _TensorView,
)


@dataclasses.dataclass(frozen=True)
class _MatmulPlan:
    """Validated shapes, views, and epilogue for one tiled matmul."""

    spec: KernelSpec
    epilogue: JaxprEqn | None
    epilogue_input: Var | Literal | None
    epilogue_side_input: Var | Literal | None
    side_layout: str | None
    n: int
    k: int
    tm: int
    tn: int
    transpose_lhs: bool
    transpose_rhs: bool
    a_offset: str
    b_offset: str
    c_offset: str
    a_extents: tuple[int, int]
    a_strides: tuple[int, int]
    b_extents: tuple[int, int]
    b_strides: tuple[int, int]


def _check_zero_max(eqn: JaxprEqn, value: Var | Literal, shape: tuple[int, ...]) -> None:
    """Require a full-tile maximum against literal zero."""
    if (
        eqn.primitive.name != "max"
        or sum(atom is value for atom in eqn.invars) != 1
        or _shape(eqn.outvars[0]) != shape
    ):
        raise EmitError("dot_general='tensorops' max epilogue must consume the full dot tile")
    other = next(atom for atom in eqn.invars if atom is not value)
    if not isinstance(other, Literal) or float(other.val) != 0.0:
        raise EmitError("dot_general='tensorops' max epilogue currently requires scalar zero")


def _producer_map(eqns: list[JaxprEqn]) -> dict[Var, JaxprEqn]:
    return {var: eqn for eqn in eqns for var in eqn.outvars if isinstance(var, Var)}


def _producer(atom, producers: dict[Var, JaxprEqn], primitive: str) -> JaxprEqn:
    eqn = producers.get(atom) if isinstance(atom, Var) else None
    if eqn is None or eqn.primitive.name != primitive:
        raise EmitError(f"dot_general='tensorops' expected a {primitive} producer")
    return eqn


def _operand_from(eqn: JaxprEqn, producers: dict[Var, JaxprEqn], primitive: str):
    matches = [
        atom
        for atom in eqn.invars
        if isinstance(atom, Var)
        and atom in producers
        and producers[atom].primitive.name == primitive
    ]
    if len(matches) != 1:
        raise EmitError(f"dot_general='tensorops' expected one {primitive} operand")
    return matches[0]


def _mark(used: set[int], *eqns: JaxprEqn) -> None:
    used.update(map(id, eqns))


def _require_all_eqns(eqns: list[JaxprEqn], used: set[int]) -> None:
    if len(eqns) != len(used):
        raise EmitError("dot_general='tensorops' jaxpr contains unsupported extra operations")


def _matrix_read(atom, producers: dict[Var, JaxprEqn], ref: Var):
    """Resolve a rank-2 dot operand to its full-block read and transpose flag."""
    producer = producers.get(atom) if isinstance(atom, Var) else None
    transposed = producer is not None and producer.primitive.name == "transpose"
    if transposed:
        if tuple(producer.params["permutation"]) != (1, 0):
            raise EmitError("dot_general='tensorops' only supports matrix transposes")
        atom = producer.invars[0]
    get = _producer(atom, producers, "get")
    if not _is_empty_get(get, ref):
        raise EmitError("dot_general='tensorops' requires full-block input reads")
    return get, transposed


def _recognize_tensorops_matmul(spec: KernelSpec) -> _MatmulPlan:
    """Match a supported tiled matmul and return its checked layout plan."""
    eqns = spec.jaxpr.eqns
    if (
        len(spec.inputs) not in (2, 3)
        or len(spec.outputs) != 1
        or spec.scratch
        or spec.aliases
        or len(spec.grid) not in (2, 3)
        or len(spec.jaxpr.invars) != len(spec.inputs) + 1
    ):
        raise EmitError(
            "dot_general='tensorops' requires one full-block matmul whose result is "
            "stored directly to one output"
        )

    lhs_ref, rhs_ref = spec.jaxpr.invars[:2]
    out_ref = spec.jaxpr.invars[-1]
    epilogue = None
    epilogue_input = None
    epilogue_side_input = None
    side_layout = None
    expand_input = None
    transpose_lhs = transpose_rhs = False
    producers = _producer_map(eqns)
    stores = [
        eqn
        for eqn in eqns
        if eqn.primitive.name == "swap" and eqn.invars and eqn.invars[0] is out_ref
    ]
    if len(stores) != 1 or len(stores[0].invars) != 2:
        raise EmitError("dot_general='tensorops' requires one direct output store")
    store = stores[0]
    used: set[int] = set()
    _mark(used, store)
    if len(spec.grid) == 2:
        rank2_value = store.invars[1]
        final_producer = producers.get(rank2_value) if isinstance(rank2_value, Var) else None
        if final_producer is not None and final_producer.primitive.name == "max":
            epilogue = final_producer
            dot_value = _operand_from(epilogue, producers, "dot_general")
            dot = _producer(dot_value, producers, "dot_general")
            epilogue_input = dot.outvars[0]
            _check_zero_max(epilogue, epilogue_input, _shape(epilogue_input))
            stored_value = epilogue.outvars[0]
            _mark(used, epilogue, dot)
        elif final_producer is not None and final_producer.primitive.name == "add":
            epilogue = final_producer
            dot_operands = [
                atom
                for atom in epilogue.invars
                if isinstance(atom, Var)
                and atom in producers
                and producers[atom].primitive.name == "dot_general"
            ]
            if len(dot_operands) == 1:
                dot_value = dot_operands[0]
                dot = _producer(dot_value, producers, "dot_general")
                side_value = next(atom for atom in epilogue.invars if atom is not dot_value)
                side_producer = producers.get(side_value) if isinstance(side_value, Var) else None
                if side_producer is not None and side_producer.primitive.name == "get":
                    if len(spec.inputs) != 3 or not _is_empty_get(
                        side_producer, spec.jaxpr.invars[2]
                    ):
                        raise EmitError(
                            "dot_general='tensorops' residual must be a full-block read"
                        )
                    epilogue_input = dot_value
                    epilogue_side_input = side_value
                    side_layout = "matrix"
                    _mark(used, side_producer)
                elif (
                    side_producer is not None and side_producer.primitive.name == "broadcast_in_dim"
                ):
                    bias_broadcast = side_producer
                    bias_get = _producer(bias_broadcast.invars[0], producers, "get")
                    if len(spec.inputs) != 3 or not _is_empty_get(bias_get, spec.jaxpr.invars[2]):
                        raise EmitError(
                            "dot_general='tensorops' column bias must be a full-block read"
                        )
                    if (
                        tuple(bias_broadcast.params["broadcast_dimensions"]) != (1,)
                        or next(iter(bias_broadcast.params["shape"])) != 1
                    ):
                        raise EmitError(
                            "dot_general='tensorops' column bias must broadcast over matrix rows"
                        )
                    epilogue_input = dot_value
                    epilogue_side_input = side_value
                    side_layout = "column_bias"
                    _mark(used, bias_get, bias_broadcast)
                else:
                    raise EmitError("dot_general='tensorops' add requires a supported side input")
                if _shape(epilogue.outvars[0]) != _shape(dot_value):
                    raise EmitError("dot_general='tensorops' add must combine matching tiles")
                stored_value = epilogue.outvars[0]
                _mark(used, epilogue, dot)
            else:
                raise EmitError("dot_general='tensorops' add must consume one dot result")
        elif final_producer is not None and final_producer.primitive.name == "dot_general":
            dot = final_producer
            stored_value = rank2_value
            _mark(used, dot)
        else:
            raise EmitError("dot_general='tensorops' requires a supported matmul epilogue")
        if len(spec.grid) == 2:
            lhs_get, transpose_lhs = _matrix_read(dot.invars[0], producers, lhs_ref)
            rhs_get, transpose_rhs = _matrix_read(dot.invars[1], producers, rhs_ref)
            _mark(used, lhs_get, rhs_get)
            for atom in dot.invars:
                producer = producers.get(atom) if isinstance(atom, Var) else None
                if producer is not None and producer.primitive.name == "transpose":
                    _mark(used, producer)
        else:
            lhs_get = _producer(dot.invars[0], producers, "get")
            rhs_get = _producer(dot.invars[1], producers, "get")
            _mark(used, lhs_get, rhs_get)
            if not _is_empty_get(lhs_get, lhs_ref) or not _is_empty_get(rhs_get, rhs_ref):
                raise EmitError("dot_general='tensorops' requires full-block input reads")
            if dot.invars[0] is not lhs_get.outvars[0] or dot.invars[1] is not rhs_get.outvars[0]:
                raise EmitError("dot_general='tensorops' does not support transformed dot operands")
            transpose_lhs = transpose_rhs = False
        _require_all_eqns(eqns, used)
        if store.invars[0] is not out_ref or store.invars[1] is not stored_value:
            raise EmitError("dot_general='tensorops' requires a direct full-block output store")
    else:
        rank3_value = store.invars[1]
        final_producer = producers.get(rank3_value) if isinstance(rank3_value, Var) else None
        if final_producer is not None and final_producer.primitive.name == "max":
            epilogue = final_producer
            expanded_value = _operand_from(epilogue, producers, "broadcast_in_dim")
            expand = _producer(expanded_value, producers, "broadcast_in_dim")
            expand_input = expand.invars[0]
            epilogue_input = expand.outvars[0]
            dot = _producer(expand_input, producers, "dot_general")
            _check_zero_max(epilogue, epilogue_input, _shape(epilogue_input))
            stored_value = epilogue.outvars[0]
            _mark(used, epilogue, expand, dot)
        elif final_producer is not None and final_producer.primitive.name == "add":
            epilogue = final_producer
            side_values = [
                atom
                for atom in epilogue.invars
                if isinstance(atom, Var)
                and atom in producers
                and producers[atom].primitive.name == "get"
            ]
            if len(spec.inputs) != 3 or len(side_values) != 1:
                raise EmitError("dot_general='tensorops' batched residual requires one side input")
            epilogue_side_input = side_values[0]
            residual_get = _producer(epilogue_side_input, producers, "get")
            if not _is_empty_get(residual_get, spec.jaxpr.invars[2]):
                raise EmitError(
                    "dot_general='tensorops' batched residual must be a full-block read"
                )
            expanded_values = [atom for atom in epilogue.invars if atom is not epilogue_side_input]
            if len(expanded_values) != 1:
                raise EmitError("dot_general='tensorops' batched add must consume one matmul tile")
            expand = _producer(expanded_values[0], producers, "broadcast_in_dim")
            expand_input = expand.invars[0]
            dot = _producer(expand_input, producers, "dot_general")
            epilogue_input = expand.outvars[0]
            if _shape(epilogue.outvars[0]) != _shape(epilogue_input):
                raise EmitError("dot_general='tensorops' batched residual must match the dot tile")
            side_layout = "batched_matrix"
            stored_value = epilogue.outvars[0]
            _mark(used, epilogue, residual_get, expand, dot)
        elif final_producer is not None and final_producer.primitive.name == "broadcast_in_dim":
            expand = final_producer
            expand_input = expand.invars[0]
            upstream = producers.get(expand_input) if isinstance(expand_input, Var) else None
            if upstream is not None and upstream.primitive.name == "max":
                epilogue = upstream
                dot_value = _operand_from(epilogue, producers, "dot_general")
                dot = _producer(dot_value, producers, "dot_general")
                epilogue_input = dot.outvars[0]
                _check_zero_max(epilogue, epilogue_input, _shape(epilogue_input))
                _mark(used, epilogue)
            else:
                dot = _producer(expand_input, producers, "dot_general")
            stored_value = expand.outvars[0]
            _mark(used, expand, dot)
        else:
            raise EmitError(
                "dot_general='tensorops' batched form requires a supported matmul epilogue"
            )
        lhs_squeeze = _producer(dot.invars[0], producers, "squeeze")
        rhs_squeeze = _producer(dot.invars[1], producers, "squeeze")
        lhs_get = _producer(lhs_squeeze.invars[0], producers, "get")
        rhs_get = _producer(rhs_squeeze.invars[0], producers, "get")
        _mark(used, lhs_get, rhs_get, lhs_squeeze, rhs_squeeze)
        _require_all_eqns(eqns, used)
        if not _is_empty_get(lhs_get, lhs_ref) or not _is_empty_get(rhs_get, rhs_ref):
            raise EmitError("dot_general='tensorops' requires full-block input reads")
        if (
            tuple(lhs_squeeze.params["dimensions"]) != (0,)
            or tuple(rhs_squeeze.params["dimensions"]) != (0,)
            or dot.invars[0] is not lhs_squeeze.outvars[0]
            or dot.invars[1] is not rhs_squeeze.outvars[0]
        ):
            raise EmitError("dot_general='tensorops' requires one leading singleton batch tile")
        if (
            expand.invars[0] is not expand_input
            or tuple(expand.params["broadcast_dimensions"]) != (1, 2)
            or store.invars[0] is not out_ref
            or store.invars[1] is not stored_value
        ):
            raise EmitError("dot_general='tensorops' requires a direct batched output store")
    if side_layout is None and len(spec.inputs) != 2:
        raise EmitError("dot_general='tensorops' standalone matmul accepts exactly two inputs")
    if store.params["tree"].flatten_up_to(()) != []:
        raise EmitError("dot_general='tensorops' requires a full-block output store")
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dot.params["dimension_numbers"]

    a, b = spec.inputs[:2]
    c = spec.outputs[0]
    if any(info.dtype.name != "float32" for info in (*spec.inputs, c)):
        raise EmitError("dot_general='tensorops' currently supports float32 only")
    rank = len(a.array_shape)
    if rank == 2:
        if len(b.array_shape) != 2 or len(c.array_shape) != 2:
            raise EmitError("dot_general='tensorops' requires rank-2 matrices")
        if any(len(info.block_shape) != 2 for info in (a, b, c)):
            raise EmitError("dot_general='tensorops' requires rank-2 matrix blocks")
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("dot_general='tensorops' rank-2 form requires unbatched A @ B")
        if transpose_lhs:
            k, m = a.array_shape
        else:
            m, k = a.array_shape
        if transpose_rhs:
            n, kb = b.array_shape
        else:
            kb, n = b.array_shape
        if k != kb or c.array_shape != (m, n):
            raise EmitError("dot_general='tensorops' matrix dimensions do not match")
        tm, tn = c.block_shape
        ka = a.block_shape[0 if transpose_lhs else 1]
        if (
            k != ka
            or (not transpose_lhs and a.array_shape != (m, k))
            or (transpose_lhs and a.array_shape != (k, m))
            or (not transpose_rhs and b.array_shape != (k, n))
            or (transpose_rhs and b.array_shape != (n, k))
            or (not transpose_lhs and a.block_shape != (tm, k))
            or (transpose_lhs and a.block_shape != (k, tm))
            or (not transpose_rhs and b.block_shape != (k, tn))
            or (transpose_rhs and b.block_shape != (tn, k))
            or c.block_shape != (tm, tn)
            or spec.grid != (m // tm, n // tn)
        ):
            raise EmitError("dot_general='tensorops' requires full-K, row-major matrix tiles")
        maps_match = (
            _index_map_is(a, (None, 0) if transpose_lhs else (0, None))
            and _index_map_is(b, (1, None) if transpose_rhs else (None, 1))
            and _index_map_is(c, (0, 1))
        )
        a_offset = f"_pid.x * {tm}" if transpose_lhs else f"_pid.x * {tm * k}"
        b_offset = f"_pid.y * {tn * k}" if transpose_rhs else f"_pid.y * {tn}"
        c_offset = f"_pid.x * {tm * n} + _pid.y * {tn}"
        a_extents = (m, k) if transpose_lhs else (k, tm)
        a_strides = (1, m) if transpose_lhs else (1, k)
        b_extents = (k, tn) if transpose_rhs else (tn, k)
        b_strides = (1, k) if transpose_rhs else (1, n)
    elif rank == 3:
        if transpose_lhs or transpose_rhs:
            raise EmitError(
                "dot_general='tensorops' batched form does not support transposed operands"
            )
        if len(b.array_shape) != 3 or len(c.array_shape) != 3:
            raise EmitError("dot_general='tensorops' batched form requires rank-3 arrays")
        if any(len(info.block_shape) != 3 for info in (a, b, c)):
            raise EmitError("dot_general='tensorops' requires rank-3 batch blocks")
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("dot_general='tensorops' batched tiles require unbatched local matmuls")
        batch, m, k = a.array_shape
        batch_b, kb, n = b.array_shape
        if batch != batch_b or k != kb or c.array_shape != (batch, m, n):
            raise EmitError("dot_general='tensorops' batched matrix dimensions do not match")
        ba, tm, ka = a.block_shape
        bb, kb_tile, tn = b.block_shape
        bc, cm, cn = c.block_shape
        expected_blocks = ((1, tm, ka), (1, kb_tile, tn), (1, cm, cn))
        a_full = a.full_block_shape
        b_full = b.full_block_shape
        c_full = c.full_block_shape
        if a_full is None or b_full is None or c_full is None:
            raise EmitError("dot_general='tensorops' requires explicit full-K block mappings")
        if (
            (ba, bb, bc) != (1, 1, 1)
            or tuple(a_full) != expected_blocks[0]
            or tuple(b_full) != expected_blocks[1]
            or tuple(c_full) != expected_blocks[2]
            or ka != k
            or kb_tile != k
            or (cm, cn) != (tm, tn)
            or spec.grid != (batch, m // tm, n // tn)
        ):
            raise EmitError("dot_general='tensorops' requires full-K batch-local row-major tiles")
        maps_match = (
            _index_map_is(a, (0, 1, None))
            and _index_map_is(b, (0, None, 2))
            and _index_map_is(c, (0, 1, 2))
        )
        a_offset = f"_pid.x * {m * k} + _pid.y * {tm * k}"
        b_offset = f"_pid.x * {k * n} + _pid.z * {tn}"
        c_offset = f"_pid.x * {m * n} + _pid.y * {tm * n} + _pid.z * {tn}"
        a_extents, a_strides = (k, tm), (1, k)
        b_extents, b_strides = (tn, k), (1, n)
    else:
        raise EmitError("dot_general='tensorops' supports rank-2 and batched rank-3 arrays")

    if k != kb:
        raise EmitError("dot_general='tensorops' requires matching full-K blocks")
    if rank == 2 and any(info.full_block_shape != info.block_shape for info in (a, b, c)):
        raise EmitError("dot_general='tensorops' requires complete matrix blocks")
    if m % tm or n % tn:
        raise EmitError("dot_general='tensorops' requires evenly tiled M and N dimensions")
    expected_grid = (m // tm, n // tn) if rank == 2 else (batch, m // tm, n // tn)
    if spec.grid != expected_grid:
        raise EmitError("dot_general='tensorops' grid does not match its matrix tiles")
    if not maps_match:
        raise EmitError("dot_general='tensorops' requires standard row-major matrix grid maps")

    if side_layout is not None:
        if len(spec.inputs) != 3:
            raise EmitError("dot_general='tensorops' side-input fusion requires three inputs")
        residual = spec.inputs[2]
        if side_layout == "matrix":
            valid_side_map = (
                rank == 2
                and residual.array_shape == (m, n)
                and residual.block_shape == (tm, tn)
                and _index_map_is(residual, (0, 1))
            )
        elif side_layout == "column_bias":
            valid_side_map = (
                rank == 2
                and residual.array_shape == (n,)
                and residual.block_shape == (tn,)
                and _index_map_is(residual, (1,))
            )
        else:
            valid_side_map = (
                rank == 3
                and residual.array_shape == (batch, m, n)
                and residual.block_shape == (1, tm, tn)
                and _index_map_is(residual, (0, 1, 2))
            )
        if not valid_side_map or residual.full_block_shape != residual.block_shape:
            raise EmitError("dot_general='tensorops' side input must match a supported tile map")

    return _MatmulPlan(
        spec=spec,
        epilogue=epilogue,
        epilogue_input=epilogue_input,
        epilogue_side_input=epilogue_side_input,
        side_layout=side_layout,
        n=n,
        k=k,
        tm=tm,
        tn=tn,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
        a_offset=a_offset,
        b_offset=b_offset,
        c_offset=c_offset,
        a_extents=a_extents,
        a_strides=a_strides,
        b_extents=b_extents,
        b_strides=b_strides,
    )


def emit_tensorops_matmul(spec: KernelSpec, kernel_name: str | None = None) -> str:
    """Emit a group-cooperative MSL kernel for a tiled matrix product."""
    return lower_tensorops_matmul(spec, kernel_name)[0]


def lower_tensorops_matmul(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Emit a matmul kernel and return its threadgroup-memory requirement."""
    plan = _recognize_tensorops_matmul(spec)
    return _emit_tensorops_matmul(plan, kernel_name)


def _emit_tensorops_matmul(plan: _MatmulPlan, kernel_name: str | None) -> tuple[str, int]:
    """Lower a validated matmul plan to MSL."""
    spec = plan.spec
    epilogue = plan.epilogue
    epilogue_input = plan.epilogue_input
    epilogue_side_input = plan.epilogue_side_input
    side_layout = plan.side_layout
    n, k, tm, tn = plan.n, plan.k, plan.tm, plan.tn
    transpose_lhs, transpose_rhs = plan.transpose_lhs, plan.transpose_rhs
    a_offset, b_offset, c_offset = plan.a_offset, plan.b_offset, plan.c_offset
    a_extents, a_strides = plan.a_extents, plan.a_strides
    b_extents, b_strides = plan.b_extents, plan.b_strides
    name = kernel_name or spec.name
    output_arg = f"arg{len(spec.inputs)}"
    params = tuple(
        f"device float* arg{index} [[buffer({index})]]" for index in range(len(spec.inputs) + 1)
    ) + ("uint3 _pid [[threadgroup_position_in_grid]]",)
    cursor = Cursor()
    if epilogue is not None:
        cursor.emit(
            "const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;"
        )
        output_storage = cursor.allocate("float", (tm, tn), name="dot_result", space="threadgroup")
    else:
        output_storage = CVal(f"({output_arg} + {c_offset})", (tm, tn), "float", space="device")

    operation = _TensorOpsMatmul(
        name="op",
        descriptor="desc",
        m=tm,
        n=tn,
        k=k,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
        accumulate=False,
    )
    operation.emit_declaration(cursor)
    lhs = _TensorView(
        CVal(f"(arg0 + {a_offset})", (tm, k), "float", space="device", readonly=True),
        (tm, k),
        a_extents,
        a_strides,
    ).emit(cursor, "a")
    rhs = _TensorView(
        CVal(f"(arg1 + {b_offset})", (k, tn), "float", space="device", readonly=True),
        (k, tn),
        b_extents,
        b_strides,
    ).emit(cursor, "b")
    output_extents = (tn, tm)
    output_strides = (1, tn) if epilogue is not None else (1, n)
    output = _TensorView(output_storage, (tm, tn), output_extents, output_strides).emit(cursor, "c")
    operation.emit_run(cursor, lhs, rhs, output)

    if epilogue is not None:
        cursor.barrier()
        env = Environment()
        if epilogue_input is None:
            raise EmitError("TensorOps epilogue is missing its matrix input")
        residual_storage = None
        if side_layout == "matrix":
            residual_storage = CVal(
                f"(arg2 + {c_offset})", (tm, tn), "float", space="device", readonly=True
            )
        elif side_layout == "column_bias":
            residual_storage = CVal(
                f"(arg2 + _pid.y * {tn})", (tn,), "float", space="device", readonly=True
            )
        elif side_layout == "batched_matrix":
            residual_storage = CVal(
                f"(arg2 + {c_offset})", (1, tm, tn), "float", space="device", readonly=True
            )
        operands = tuple(
            CooperativeValue(output_storage, "tensorops")
            if atom is epilogue_input
            else residual_storage
            if atom is epilogue_side_input and residual_storage is not None
            else env.val(atom)
            for atom in epilogue.invars
        )
        output_tile = CVal(f"({output_arg} + {c_offset})", (tm, tn), "float", space="device")
        emit_elementwise_store(cursor, epilogue, operands, output_tile, thread_count="THREADS")

    if epilogue is not None:
        params += (
            "uint tid [[thread_index_in_threadgroup]]",
            "uint3 threads_per_group [[threads_per_threadgroup]]",
        )
    source = _kernel_source(name, params, cursor.lines)
    return source, cursor.threadgroup_bytes
