"""Cooperative lowering for standalone row reductions."""

from __future__ import annotations

from palladium.emit.core import Cursor, Environment, _block_offset
from palladium.errors import EmitError

from ._shared import _index_map_is, _kernel_source, _shape
from .ir import KernelIR
from .plan import ProgramScope


def lower_row_reduction_ir(kernel: KernelIR, kernel_name: str | None = None) -> tuple[str, int]:
    """Lower a float32 axis-1 reduction across SIMD-group lanes.

    Each SIMD group cooperates on one row: lanes accumulate strided columns,
    then use a hardware SIMD reduction. SIMD groups stride across tile rows.
    """

    plan = kernel.plan
    spec = plan.spec
    if plan.scope is not ProgramScope.THREADGROUP:
        raise EmitError("tensorops cooperative row reduction requires threadgroup program scope")
    if (
        len(spec.inputs) != 1
        or len(spec.outputs) != 1
        or spec.scratch
        or spec.aliases
        or len(spec.grid) != 1
        or len(spec.jaxpr.constvars) != 0
        or len(spec.jaxpr.invars) != 2
    ):
        raise EmitError("tensorops row reduction requires one input, one output, and a 1D grid")

    input_info, output_info = spec.inputs[0], spec.outputs[0]
    if (
        input_info.dtype.name != "float32"
        or output_info.dtype.name != "float32"
        or len(input_info.array_shape) != 2
        or len(input_info.block_shape) != 2
        or len(output_info.array_shape) != 1
        or len(output_info.block_shape) != 1
    ):
        raise EmitError(
            "tensorops row reduction currently requires float32 matrix-to-vector buffers"
        )
    rows, columns = input_info.array_shape
    tile_rows, tile_columns = input_info.block_shape
    if (
        output_info.array_shape != (rows,)
        or output_info.block_shape != (tile_rows,)
        or tile_columns != columns
        or input_info.full_block_shape != input_info.block_shape
        or output_info.full_block_shape != output_info.block_shape
        or spec.grid != ((rows + tile_rows - 1) // tile_rows,)
        or not _index_map_is(input_info, (0, None))
        or not _index_map_is(output_info, (0,))
        or rows <= 0
        or columns <= 0
        or tile_rows <= 0
    ):
        raise EmitError(
            "tensorops row reduction requires row tiles spanning the full reduction axis"
        )

    operations = spec.jaxpr.eqns
    gets = [eqn for eqn in operations if eqn.primitive.name == "get"]
    reductions = [eqn for eqn in operations if eqn.primitive.name in ("reduce_sum", "reduce_max")]
    stores = [eqn for eqn in operations if eqn.primitive.name == "swap"]
    if len(operations) != 3 or len(gets) != 1 or len(reductions) != 1 or len(stores) != 1:
        raise EmitError(
            "tensorops row reduction requires one full-block read, reduction, and store"
        )
    get, reduction, store = gets[0], reductions[0], stores[0]
    input_ref, output_ref = spec.jaxpr.invars
    if (
        len(get.invars) != 1
        or get.invars[0] is not input_ref
        or bool(get.params["tree"].flatten_up_to(()))
        or _shape(get.outvars[0]) != input_info.block_shape
        or len(reduction.invars) != 1
        or reduction.invars[0] is not get.outvars[0]
        or tuple(reduction.params.get("axes", ())) != (1,)
        or _shape(reduction.invars[0]) != input_info.block_shape
        or _shape(reduction.outvars[0]) != output_info.block_shape
        or len(store.invars) != 2
        or store.invars[0] is not output_ref
        or store.invars[1] is not reduction.outvars[0]
        or store.params["tree"].flatten_up_to(()) != []
    ):
        raise EmitError(
            "tensorops row reduction jaxpr does not match a full-block axis-1 reduction"
        )

    cursor = Cursor()
    env = Environment()
    input_offset = _block_offset(env, cursor, spec, input_info)
    output_offset = _block_offset(env, cursor, spec, output_info)
    name = kernel_name or spec.name
    params = (
        "const device float* arg0 [[buffer(0)]]",
        "device float* arg1 [[buffer(1)]]",
        "uint3 _pid [[threadgroup_position_in_grid]]",
        "uint3 threads_per_group [[threads_per_threadgroup]]",
        "uint lane [[thread_index_in_simdgroup]]",
        "uint simdgroup [[simdgroup_index_in_threadgroup]]",
        "uint simdgroups [[simdgroups_per_threadgroup]]",
    )
    valid_rows = (
        str(tile_rows)
        if rows % tile_rows == 0
        else f"min({tile_rows}, {rows} - (int)_pid.x * {tile_rows})"
    )
    with (
        cursor.strided_loop("simdgroup", str(tile_rows), "simdgroups", name="row") as row,
        cursor.block(f"if ({row} < {valid_rows})"),
    ):
        if reduction.primitive.name == "reduce_sum":
            initial, combine = "0.0f", "simd_sum"
        else:
            initial, combine = "-INFINITY", "simd_max"
        accumulator = cursor.fresh("reduce")
        cursor.emit(f"float {accumulator} = {initial};")
        with cursor.strided_loop(
            "lane",
            str(columns),
            "(threads_per_group.x / simdgroups)",
            name="column",
        ) as column:
            value = f"arg0[{input_offset} + {row} * {columns} + {column}]"
            if combine == "simd_sum":
                cursor.emit(f"{accumulator} += {value};")
            else:
                cursor.emit(f"{accumulator} = max({accumulator}, {value});")
        cursor.emit(f"{accumulator} = {combine}({accumulator});")
        with cursor.block("if (lane == 0)"):
            cursor.emit(f"arg1[{output_offset} + {row}] = {accumulator};")

    source = _kernel_source(name, params, cursor.lines)
    return source, cursor.threadgroup_bytes
