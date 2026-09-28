"""Compositional cooperative lowering for a tiled matrix product."""

from __future__ import annotations

import dataclasses
import string

from jax.extend.core import Literal, Var

from palladium.emit.core import CTYPES, ELEMENTWISE, Cursor, CVal, Environment, _block_offset
from palladium.emit.numeric import typed_expression
from palladium.errors import EmitError

from ._shared import (
    _index_map_is,
    _kernel_source,
    _TensorOpsMatmul,
    _TensorView,
)
from .ir import KernelIR
from .plan import Distribution, ProgramScope

_K_TILE = 128


def lower_matmul_ir(kernel: KernelIR, kernel_name: str | None = None) -> tuple[str, int]:
    """Lower a tiled rank-2 or batched dot and its epilogue from tensorops IR.

    The dot, epilogue, and output store are selected from the imported
    operation graph and validated against Pallas block maps.
    """
    plan = kernel.plan
    spec = plan.spec
    if plan.scope is not ProgramScope.THREADGROUP:
        raise EmitError("tensorops cooperative matmul requires threadgroup program scope")
    if (
        len(spec.inputs) < 2
        or len(spec.outputs) != 1
        or spec.scratch
        or spec.aliases
        or len(spec.grid) not in (2, 3)
        or len(spec.jaxpr.constvars) != 0
        or len(spec.jaxpr.invars) != len(spec.inputs) + 1
    ):
        raise EmitError("tensorops matmul requires a full-block matmul with one output")

    operations = kernel.body.operations
    dots = [op for op in operations if op.name == "dot_general"]
    stores = [op for op in operations if op.name == "swap"]
    if len(dots) != 1 or len(stores) != 1:
        raise EmitError("tensorops matmul requires one dot and one output store")
    dot_op, store_op = dots[0], stores[0]
    dot = dot_op.equation
    store = store_op.equation
    if len(store.invars) != 2 or store.params["tree"].flatten_up_to(()) != []:
        raise EmitError("tensorops matmul currently supports only a full-block output store")
    if store.invars[0] is not spec.jaxpr.invars[-1]:
        raise EmitError("tensorops matmul store must target its output ref")
    if (
        dot_op.results[0].layout is None
        or dot_op.results[0].layout.distribution is not Distribution.TENSOROPS
    ):
        raise EmitError("tensorops layout assignment did not give the dot TensorOps ownership")

    producers = {
        variable: eqn
        for eqn in spec.jaxpr.eqns
        for variable in eqn.outvars
        if isinstance(variable, Var)
    }
    lhs_get, transpose_lhs, lhs_squeeze = _matrix_get(
        dot.invars[0], producers, spec.jaxpr.invars[0]
    )
    rhs_get, transpose_rhs, rhs_squeeze = _matrix_get(
        dot.invars[1], producers, spec.jaxpr.invars[1]
    )

    lhs_info, rhs_info = spec.inputs[:2]
    output_info = spec.outputs[0]
    matmul_dtype = lhs_info.dtype.name
    if matmul_dtype not in ("float32", "float16", "bfloat16") or any(
        info.dtype.name != matmul_dtype for info in (*spec.inputs, output_info)
    ):
        raise EmitError("TensorOps matmul requires matching float32, float16, or bfloat16 types")
    rank = len(output_info.array_shape)
    if rank not in (2, 3) or any(
        len(info.array_shape) != rank or len(info.block_shape) != rank
        for info in (lhs_info, rhs_info, output_info)
    ):
        raise EmitError("TensorOps matmul supports rank-2 matrices and rank-3 batches")
    if any(info.full_block_shape != info.block_shape for info in (lhs_info, rhs_info, output_info)):
        raise EmitError("TensorOps matmul requires complete matrix blocks")

    if rank == 2:
        m, k = lhs_info.array_shape[::-1] if transpose_lhs else lhs_info.array_shape
        kb, n = rhs_info.array_shape[::-1] if transpose_rhs else rhs_info.array_shape
        tm, tn = output_info.block_shape
        batch = 1
    else:
        if transpose_lhs or transpose_rhs:
            raise EmitError(
                "tensorops batched TensorOps matmul does not support transposed operands"
            )
        batch, m, k = lhs_info.array_shape
        batch_rhs, kb, n = rhs_info.array_shape
        if batch != batch_rhs:
            raise EmitError("tensorops batched TensorOps operands must have matching batch sizes")
        _, tm, tn = output_info.block_shape
    if (
        k != kb
        or output_info.array_shape != ((m, n) if rank == 2 else (batch, m, n))
        or lhs_info.block_shape[-2:] != ((k, tm) if transpose_lhs else (tm, k))
        or rhs_info.block_shape[-2:] != ((tn, kb) if transpose_rhs else (kb, tn))
        or (
            rank == 3
            and (lhs_info.block_shape[0], rhs_info.block_shape[0], output_info.block_shape[0])
            != (1, 1, 1)
        )
        or spec.grid
        != (
            ((m + tm - 1) // tm, (n + tn - 1) // tn)
            if rank == 2
            else (batch, (m + tm - 1) // tm, (n + tn - 1) // tn)
        )
        or k <= 0
        or m <= 0
        or n <= 0
        or tm % 16
        or tn % 16
    ):
        raise EmitError("TensorOps matmul requires full-K, evenly tiled matrix operands")
    if not (
        _index_map_is(
            lhs_info, (None, 0) if transpose_lhs else ((0, None) if rank == 2 else (0, 1, None))
        )
        and _index_map_is(
            rhs_info, (1, None) if transpose_rhs else ((None, 1) if rank == 2 else (0, None, 2))
        )
        and _index_map_is(output_info, (0, 1) if rank == 2 else (0, 1, 2))
    ):
        raise EmitError("TensorOps matmul requires standard row-major block maps")

    if (
        _shape(lhs_get.outvars[0]) != lhs_info.block_shape
        or _shape(rhs_get.outvars[0]) != rhs_info.block_shape
    ):
        raise EmitError("tensorops dot operands must be full matrix tiles")

    store_value = store.invars[1]
    result_broadcast = None
    if rank == 3:
        broadcast = producers.get(store_value) if isinstance(store_value, Var) else None
        if broadcast is not None and broadcast.primitive.name == "broadcast_in_dim":
            if tuple(broadcast.params["broadcast_dimensions"]) != (1, 2):
                raise EmitError(
                    "tensorops batched matmul store requires a leading singleton broadcast"
                )
            result_broadcast = broadcast
            store_value = broadcast.invars[0]
    elementwise_eqns, used = _epilogue_path(store_value, dot.outvars[0], producers)
    if not elementwise_eqns and store_value is not dot.outvars[0]:
        raise EmitError("tensorops matmul store value must be the dot result or its epilogue")
    used.update((id(dot), id(store), id(lhs_get), id(rhs_get)))
    if result_broadcast is not None:
        used.add(id(result_broadcast))
    used.update(id(eqn) for eqn in (lhs_squeeze, rhs_squeeze) if eqn is not None)
    for atom, transposed in ((dot.invars[0], transpose_lhs), (dot.invars[1], transpose_rhs)):
        if transposed:
            if not isinstance(atom, Var) or atom not in producers:
                raise EmitError("tensorops matmul transpose producer was not imported")
            used.add(id(producers[atom]))
    all_gets = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "get"]
    if not set(map(id, all_gets)).issubset(used) or len(used) != len(spec.jaxpr.eqns):
        raise EmitError("tensorops matmul jaxpr has operations outside the dot epilogue path")

    cursor = Cursor()
    params = tuple(
        f"device {CTYPES[info.dtype.name]}* arg{index} [[buffer({index})]]"
        for index, info in enumerate((*spec.inputs, output_info))
    ) + ("uint3 _pid [[threadgroup_position_in_grid]]",)
    has_epilogue = bool(elementwise_eqns)
    has_edge_tiles = m % tm != 0 or n % tn != 0
    available_epilogue_values = {dot.outvars[0]}
    cooperative_epilogue = has_epilogue
    for eqn in elementwise_eqns:
        if any(
            not isinstance(atom, Literal) and atom not in available_epilogue_values
            for atom in eqn.invars
        ):
            cooperative_epilogue = False
        available_epilogue_values.update(eqn.outvars)
    threadgroup_epilogue = has_epilogue and not cooperative_epilogue
    needs_lane_loop = threadgroup_epilogue or has_edge_tiles
    if needs_lane_loop:
        params += (
            "uint tid [[thread_index_in_threadgroup]]",
            "uint3 threads_per_group [[threads_per_threadgroup]]",
        )
        cursor.emit(
            "const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;"
        )

    env = Environment()
    infos = (*spec.inputs, output_info)
    ref_values: dict[Var, CVal] = {}
    offsets = tuple(_block_offset(env, cursor, spec, info) for info in infos)
    for index, (ref, info, offset) in enumerate(
        zip(spec.jaxpr.invars, infos, offsets, strict=True)
    ):
        pointer = f"arg{index}" if offset == "0" else f"(arg{index} + {offset})"
        ref_values[ref] = CVal(
            pointer,
            info.block_shape,
            CTYPES[info.dtype.name],
            space="device",
            readonly=index < len(spec.inputs),
        )

    values: dict[Var, CVal] = {}
    for eqn in all_gets:
        if len(eqn.invars) != 1 or eqn.invars[0] not in ref_values:
            raise EmitError("tensorops matmul reads must be full-block ref gets")
        if tuple(eqn.params["tree"].flatten_up_to(())):
            raise EmitError("tensorops matmul does not support indexed ref gets yet")
        ref_value = ref_values[eqn.invars[0]]
        if _shape(eqn.outvars[0]) != ref_value.shape:
            raise EmitError("tensorops matmul get shape does not match its input block")
        ref_index = spec.jaxpr.invars.index(eqn.invars[0])
        if ref_index >= len(spec.inputs):
            raise EmitError("tensorops matmul cannot read from its output ref")
        expected_side = (tm, tn) if rank == 2 else (1, tm, tn)
        side_map = (0, 1) if rank == 2 else (0, 1, 2)
        column_bias = (
            rank == 2 and ref_value.shape == (tn,) and _index_map_is(spec.inputs[ref_index], (1,))
        )
        if ref_index >= 2 and (
            not column_bias
            and (
                ref_value.shape != expected_side
                or not _index_map_is(spec.inputs[ref_index], side_map)
            )
        ):
            raise EmitError("tensorops matmul epilogue reads must match the output tile layout")
        if ref_index >= 2 and not column_bias:
            full_row_stride = spec.inputs[ref_index].array_shape[-1]
            ref_value = dataclasses.replace(
                ref_value,
                shape=(tm, tn),
                index_map=f"(($i / {tn}) * {full_row_stride} + ($i % {tn}))",
            )
        values[eqn.outvars[0]] = ref_value

    output_ctype = CTYPES[output_info.dtype.name]
    output_storage = (
        cursor.allocate(output_ctype, (tm, tn), name="dot_result", space="threadgroup")
        if threadgroup_epilogue
        else ref_values[spec.jaxpr.invars[-1]]
    )
    k_tile = min(_K_TILE, ((k + 15) // 16) * 16)
    operation = _TensorOpsMatmul.from_eqn(dot, producers, name="op", accumulate=True)
    operation = _TensorOpsMatmul(
        name=operation.name,
        descriptor="desc",
        m=operation.m,
        n=operation.n,
        k=k_tile,
        transpose_lhs=operation.transpose_lhs,
        transpose_rhs=operation.transpose_rhs,
        accumulate=True,
    )
    operation.emit_declaration(cursor)

    a_offset, b_offset, c_offset = offsets[:2] + (offsets[-1],)
    m_pid, n_pid = ("x", "y") if rank == 2 else ("y", "z")
    valid_m = str(tm) if m % tm == 0 else f"min({tm}, {m} - (int)_pid.{m_pid} * {tm})"
    valid_n = str(tn) if n % tn == 0 else f"min({tn}, {n} - (int)_pid.{n_pid} * {tn})"
    lhs_tensor = _TensorView(
        CVal(
            f"(arg0 + {a_offset})",
            (tm, k),
            CTYPES[lhs_info.dtype.name],
            space="device",
            readonly=True,
        ),
        (tm, k),
        (m, valid_m) if transpose_lhs else (k, valid_m),
        (1, m) if transpose_lhs else (1, k),
    ).emit(cursor, "a")
    rhs_tensor = _TensorView(
        CVal(
            f"(arg1 + {b_offset})",
            (k, tn),
            CTYPES[rhs_info.dtype.name],
            space="device",
            readonly=True,
        ),
        (k, tn),
        (k, valid_n) if transpose_rhs else (valid_n, k),
        (1, k) if transpose_rhs else (1, n),
    ).emit(cursor, "b")
    output_extents = (tn, tm) if threadgroup_epilogue else (valid_n, valid_m)
    output_strides = (1, tn) if threadgroup_epilogue else (1, n)
    output_tensor = _TensorView(
        output_storage,
        (tm, tn),
        output_extents,
        output_strides,
    ).emit(cursor, "c")
    edge_storage = None
    edge_tensor = None
    if has_edge_tiles and not threadgroup_epilogue:
        edge_storage = cursor.allocate(
            output_ctype, (tm, tn), name="edge_result", space="threadgroup"
        )
        edge_tensor = _TensorView(edge_storage, (tm, tn), (tn, tm), (1, tn)).emit(cursor, "c_edge")
    cooperative_result = operation.emit_cooperative_destination(
        cursor, lhs_tensor, rhs_tensor, element_type=output_ctype
    )
    with cursor.loop("cTc.get_capacity()", "init") as index:
        cursor.emit(f"cTc[{index}] = {output_ctype}(0.0f);")
    with cursor.block(f"for (int k_start = 0; k_start < {k}; k_start += {k_tile})"):
        chunk_k = f"min({k_tile}, {k} - k_start)"
        lhs_chunk = _TensorView(
            CVal(
                f"(arg0 + {a_offset} + k_start * {m if transpose_lhs else 1})",
                (tm, k_tile),
                CTYPES[lhs_info.dtype.name],
                space="device",
                readonly=True,
            ),
            (tm, k_tile),
            (m, valid_m) if transpose_lhs else (chunk_k, valid_m),
            (1, m) if transpose_lhs else (1, k),
        ).emit(cursor, "a_k")
        rhs_chunk = _TensorView(
            CVal(
                f"(arg1 + {b_offset} + k_start * {1 if transpose_rhs else n})",
                (k_tile, tn),
                CTYPES[rhs_info.dtype.name],
                space="device",
                readonly=True,
            ),
            (k_tile, tn),
            (chunk_k, valid_n) if transpose_rhs else (valid_n, chunk_k),
            (1, k) if transpose_rhs else (1, n),
        ).emit(cursor, "b_k")
        operation.emit_run(cursor, lhs_chunk, rhs_chunk, cooperative_result)
    values[dot.outvars[0]] = CVal("cTc", (tm, tn), output_ctype, space="thread")

    output_ref = CVal(
        f"(arg{len(spec.inputs)} + {c_offset})", (tm, tn), output_ctype, space="device"
    )
    if cooperative_epilogue:
        with cursor.loop("cTc.get_capacity()", "element") as index:
            scalar_values = {dot.outvars[0]: f"cTc[{index}]"}
            for eqn in elementwise_eqns:
                operands = tuple(
                    _scalar_value(atom, scalar_values, values, env, index, (tm, tn))
                    for atom in eqn.invars
                )
                result_type = CTYPES[eqn.outvars[0].aval.dtype.name]
                expression = _elementwise_expression(eqn, result_type, operands)
                result_name = cursor.fresh("tensorops_epilogue")
                cursor.emit(f"{result_type} {result_name} = {expression};")
                scalar_values[eqn.outvars[0]] = result_name
            try:
                expression = scalar_values[store_value]
            except KeyError as error:
                raise EmitError("tensorops matmul store value was not lowered") from error
            cursor.emit(f"cTc[{index}] = {expression};")
    if threadgroup_epilogue:
        cursor.emit(f"{cooperative_result.expr}.store({output_tensor.expr});")
        cursor.barrier()
        with cursor.strided_loop("tid", str(output_ref.size), "THREADS", name="element") as index:
            cursor.emit(f"const uint row = {index} / {tn};")
            cursor.emit(f"const uint column = {index} % {tn};")
            with cursor.block(f"if (row < {valid_m} && column < {valid_n})"):
                scalar_values = {dot.outvars[0]: output_storage.at(index)}
                for eqn in elementwise_eqns:
                    if not _is_tile_shape(_shape(eqn.outvars[0]), rank, tm, tn) and not (
                        eqn.primitive.name == "broadcast_in_dim"
                        and _shape(eqn.outvars[0]) == (1, tn)
                    ):
                        raise EmitError(
                            "tensorops matmul epilogue operations must preserve the output tile shape"
                        )
                    operands = tuple(
                        _scalar_value(atom, scalar_values, values, env, index, (tm, tn))
                        for atom in eqn.invars
                    )
                    result_type = CTYPES[eqn.outvars[0].aval.dtype.name]
                    expression = _elementwise_expression(eqn, result_type, operands)
                    result_name = cursor.fresh("tensorops_epilogue")
                    cursor.emit(f"{result_type} {result_name} = {expression};")
                    scalar_values[eqn.outvars[0]] = result_name
                try:
                    expression = scalar_values[store_value]
                except KeyError as error:
                    raise EmitError("tensorops matmul store value was not lowered") from error
                cursor.emit(f"{output_ref.expr}[row * {n} + column] = {expression};")
    else:
        if has_edge_tiles:
            if edge_tensor is None or edge_storage is None:
                raise EmitError("tensorops edge tile storage was not allocated")
            with cursor.block(f"if ({valid_m} == {tm} && {valid_n} == {tn})"):
                cursor.emit(f"{cooperative_result.expr}.store({output_tensor.expr});")
            cursor.emit("else {")
            cursor.indent += 1
            cursor.emit(f"{cooperative_result.expr}.store({edge_tensor.expr});")
            cursor.barrier()
            with cursor.strided_loop("tid", str(tm * tn), "THREADS", name="edge") as index:
                cursor.emit(f"const uint row = {index} / {tn};")
                cursor.emit(f"const uint column = {index} % {tn};")
                with cursor.block(f"if (row < {valid_m} && column < {valid_n})"):
                    cursor.emit(
                        f"{output_ref.expr}[row * {n} + column] = {edge_storage.at(index)};"
                    )
            cursor.indent -= 1
            cursor.emit("}")
        else:
            cursor.emit(f"{cooperative_result.expr}.store({output_tensor.expr});")

    source = _kernel_source(kernel_name or spec.name, params, cursor.lines)
    return source, cursor.threadgroup_bytes


def _shape(atom) -> tuple[int, ...]:
    return tuple(int(size) for size in getattr(atom.aval, "shape", ()))


def _full_get(eqn, ref: Var):
    if (
        eqn is None
        or eqn.primitive.name != "get"
        or len(eqn.invars) != 1
        or eqn.invars[0] is not ref
    ):
        raise EmitError("tensorops dot operands must be full-block reads of the first two refs")
    return eqn


def _matrix_get(atom, producers, ref: Var):
    """Resolve a matrix operand read, retaining a lazy 2D transpose."""
    eqn = producers.get(atom) if isinstance(atom, Var) else None
    squeezed = None
    if eqn is not None and eqn.primitive.name == "squeeze":
        if tuple(eqn.params["dimensions"]) != (0,):
            raise EmitError("tensorops batched matmul requires a leading singleton squeeze")
        squeezed = eqn
        atom = eqn.invars[0]
        eqn = producers.get(atom) if isinstance(atom, Var) else None
    transpose = eqn is not None and eqn.primitive.name == "transpose"
    if transpose:
        if tuple(eqn.params["permutation"]) != (1, 0):
            raise EmitError("TensorOps matmul supports only matrix transposes")
        atom = eqn.invars[0]
        eqn = producers.get(atom) if isinstance(atom, Var) else None
    return _full_get(eqn, ref), transpose, squeezed


def _epilogue_path(value, dot_value, producers) -> tuple[list, set[int]]:
    """Collect the supported elementwise producer chain feeding one store."""
    ordered = []
    used: set[int] = set()
    allowed = set(ELEMENTWISE)
    allowed.update(("max", "broadcast_in_dim"))

    def visit(atom):
        if atom is dot_value or isinstance(atom, Literal):
            return
        producer = producers.get(atom) if isinstance(atom, Var) else None
        if producer is None:
            raise EmitError("tensorops matmul epilogue has an unbound operand")
        if id(producer) in used:
            return
        if producer.primitive.name == "get":
            used.add(id(producer))
            return
        if producer.primitive.name not in allowed and not typed_expression(
            producer.primitive.name, "float"
        ):
            raise EmitError(
                f"tensorops matmul epilogue primitive {producer.primitive.name!r} is unsupported"
            )
        for operand in producer.invars:
            visit(operand)
        used.add(id(producer))
        ordered.append(producer)

    visit(value)
    return ordered, used


def _scalar_value(atom, scalar_values, values, env, index: str, shape: tuple[int, ...]) -> str:
    """Resolve one epilogue operand for the current output element."""
    if isinstance(atom, Var) and atom in scalar_values:
        return scalar_values[atom]
    if isinstance(atom, Literal):
        literal = env.val(atom)
        if literal.shape:
            raise EmitError("tensorops matmul epilogue literals must be scalar")
        return literal.expr
    try:
        value = values[atom]
    except KeyError as error:
        raise EmitError(f"tensorops matmul epilogue operand {atom} has no lowered value") from error
    if value.shape == (shape[-1],) and len(shape) == 2:
        return value.at(f"({index} % {shape[-1]})")
    if value.shape not in ((), shape):
        raise EmitError(
            "tensorops matmul epilogue operands must be scalar or match the output tile"
        )
    return value.at(index)


def _elementwise_expression(eqn, ctype: str, operands: tuple[str, ...]) -> str:
    """Format one supported scalar equation for the current output element."""
    if eqn.primitive.name == "broadcast_in_dim":
        if len(operands) != 1 or tuple(eqn.params["broadcast_dimensions"]) not in ((1,), (1, 2)):
            raise EmitError("tensorops matmul only supports column and singleton-batch broadcasts")
        return operands[0]
    template = typed_expression(eqn.primitive.name, ctype) or ELEMENTWISE.get(eqn.primitive.name)
    if template is None:
        raise EmitError(
            f"tensorops matmul epilogue primitive {eqn.primitive.name!r} is unsupported"
        )
    names = string.ascii_lowercase[: len(operands)]
    fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
    if len(names) != len(eqn.invars) or fields != set(names):
        raise EmitError(f"{eqn.primitive.name} epilogue has unsupported arity")
    return template.format(**dict(zip(names, operands, strict=True)))


def _is_tile_shape(shape: tuple[int, ...], rank: int, tm: int, tn: int) -> bool:
    return shape == (tm, tn) or (rank == 3 and shape == (1, tm, tn))
