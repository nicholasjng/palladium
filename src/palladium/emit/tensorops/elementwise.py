"""Cooperative, lane-strided lowering for standalone pointwise kernels."""

from __future__ import annotations

from jax.extend.core import Literal, Var

from palladium.emit.cooperative import elementwise_expression
from palladium.emit.core import ELEMENTWISE, Cursor, Environment, _block_offset
from palladium.emit.numeric import typed_expression
from palladium.errors import EmitError

from ._shared import _dtype_name, _index_map_is, _kernel_source, _shape
from .ir import KernelIR
from .plan import Distribution, ProgramScope


def _tile_coordinates(index: str, tile_shape: tuple[int, ...]) -> tuple[str, ...]:
    coordinates = []
    for dimension, tile_size in enumerate(tile_shape):
        tile_stride = 1
        for trailing_tile in tile_shape[dimension + 1 :]:
            tile_stride *= trailing_tile
        coordinate = index if tile_stride == 1 else f"({index} / {tile_stride})"
        if tile_size != 1:
            coordinate = f"({coordinate} % {tile_size})"
        coordinates.append(coordinate)
    return tuple(coordinates)


def _tile_element_offset(index: str, tile_shape: tuple[int, ...], shape: tuple[int, ...]) -> str:
    """Map a flattened tile index to its strided location in the full array."""

    coordinates = _tile_coordinates(index, tile_shape)
    terms = []
    for dimension, coordinate in enumerate(coordinates):
        array_stride = 1
        for trailing_array in shape[dimension + 1 :]:
            array_stride *= trailing_array
        terms.append(coordinate if array_stride == 1 else f"({coordinate} * {array_stride})")
    return " + ".join(terms)


def _broadcasted_tile_offset(
    index: str,
    output_tile: tuple[int, ...],
    input_tile: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> str:
    """Map one output lane to a scalar, row-vector, or full-tile operand."""

    if not input_tile:
        return "0"
    output_coordinates = _tile_coordinates(index, output_tile)
    leading_dims = len(output_tile) - len(input_tile)
    terms = []
    for dimension, extent in enumerate(input_tile):
        coordinate = "0" if extent == 1 else output_coordinates[leading_dims + dimension]
        stride = 1
        for trailing in input_shape[dimension + 1 :]:
            stride *= trailing
        terms.append(coordinate if stride == 1 else f"({coordinate} * {stride})")
    return " + ".join(terms)


def lower_elementwise_ir(kernel: KernelIR, kernel_name: str | None = None) -> tuple[str, int]:
    """Lower same-shape float32 pointwise operations across threadgroup lanes.

    Each lane handles independent flattened elements from one Pallas tile.
    The initial scope is intentionally restricted to rank-1 to rank-3 buffers,
    one output, and a straight-line scalar elementwise graph.
    """

    plan = kernel.plan
    spec = plan.spec
    if plan.scope is not ProgramScope.THREADGROUP:
        raise EmitError("tensorops cooperative elementwise lowering requires threadgroup scope")
    if (
        not spec.inputs
        or len(spec.outputs) != 1
        or spec.scratch
        or spec.aliases
        or len(spec.jaxpr.constvars) != 0
        or len(spec.jaxpr.invars) != len(spec.inputs) + 1
    ):
        raise EmitError("tensorops elementwise lowering requires inputs and one non-aliased output")

    infos = (*spec.inputs, spec.outputs[0])
    output = spec.outputs[0]
    shape = output.array_shape
    tile_shape = output.block_shape
    rank = len(shape)
    if (
        output.dtype.name != "float32"
        or rank not in (1, 2, 3)
        or len(tile_shape) != rank
        or output.full_block_shape != tile_shape
        or any(size <= 0 for size in tile_shape)
        or any(size <= 0 for size in shape)
        or spec.grid
        != tuple((size + tile - 1) // tile for size, tile in zip(shape, tile_shape, strict=True))
    ):
        raise EmitError(
            "tensorops elementwise currently requires tiled rank-1 to rank-3 float32 output"
        )
    if output.full_block_shape != tile_shape or not _index_map_is(output, tuple(range(rank))):
        raise EmitError("tensorops elementwise output must use the standard row-major tile map")
    for info in spec.inputs:
        if info.dtype.name != "float32" or info.full_block_shape != info.block_shape:
            raise EmitError(
                "tensorops elementwise broadcast inputs must be full-block float32 refs"
            )
        if info.array_shape == () and info.block_shape == ():
            map_axes = ()
        elif info.array_shape == shape and info.block_shape == tile_shape:
            map_axes = tuple(range(rank))
        elif (
            rank >= 2 and info.array_shape == (shape[-1],) and info.block_shape == (tile_shape[-1],)
        ):
            map_axes = (rank - 1,)
        else:
            raise EmitError(
                "tensorops elementwise inputs must match the output tile, be scalar, or be a row vector"
            )
        if not _index_map_is(info, map_axes):
            raise EmitError("tensorops elementwise broadcast input has an unsupported block map")

    input_refs = spec.jaxpr.invars[:-1]
    output_ref = spec.jaxpr.invars[-1]
    operations = spec.jaxpr.eqns
    gets = [eqn for eqn in operations if eqn.primitive.name == "get"]
    stores = [eqn for eqn in operations if eqn.primitive.name == "swap"]
    if len(gets) != len(spec.inputs) or len(stores) != 1:
        raise EmitError(
            "tensorops elementwise requires one full-block read per input and one output store"
        )

    get_values: dict[Var, int] = {}
    for eqn in gets:
        if (
            len(eqn.invars) != 1
            or eqn.invars[0] not in input_refs
            or bool(eqn.params["tree"].flatten_up_to(()))
            or _shape(eqn.outvars[0]) != spec.inputs[input_refs.index(eqn.invars[0])].block_shape
        ):
            raise EmitError("tensorops elementwise reads must be full-block input gets")
        get_values[eqn.outvars[0]] = input_refs.index(eqn.invars[0])
    if set(get_values.values()) != set(range(len(spec.inputs))):
        raise EmitError("tensorops elementwise must read every input exactly once")

    store = stores[0]
    if (
        len(store.invars) != 2
        or store.invars[0] is not output_ref
        or store.params["tree"].flatten_up_to(()) != []
        or _shape(store.invars[1]) != tile_shape
    ):
        raise EmitError("tensorops elementwise output must be a full-block store matching the tile")

    non_memory_ids = {id(eqn) for eqn in gets} | {id(store)}
    elementwise_eqns = [eqn for eqn in operations if id(eqn) not in non_memory_ids]
    if len(gets) + len(elementwise_eqns) + 1 != len(operations):
        raise EmitError(
            "tensorops elementwise jaxpr contains unsupported control or memory operations"
        )

    cursor = Cursor()
    env = Environment()
    imported_operations = {
        id(operation.equation): operation for operation in kernel.body.operations
    }
    offsets = tuple(_block_offset(env, cursor, spec, info) for info in infos)
    name = kernel_name or spec.name
    params = tuple(
        f"{'device' if index == len(spec.inputs) else 'const device'} float* arg{index} [[buffer({index})]]"
        for index in range(len(infos))
    ) + (
        "uint3 _pid [[threadgroup_position_in_grid]]",
        "uint tid [[thread_index_in_threadgroup]]",
        "uint3 threads_per_group [[threads_per_threadgroup]]",
    )
    cursor.emit(
        "const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;"
    )
    element_count = 1
    for size in tile_shape:
        element_count *= size

    with cursor.strided_loop("tid", str(element_count), "THREADS", name="element") as index:
        coordinates = _tile_coordinates(index, tile_shape)
        local_offset = _tile_element_offset(index, tile_shape, shape)
        valid = " && ".join(
            f"((int)_pid.{axis} * {tile} + {coordinate}) < {extent}"
            for axis, tile, coordinate, extent in zip(
                "xyz"[:rank], tile_shape, coordinates, shape, strict=True
            )
        )
        with cursor.block(f"if ({valid})"):
            values: dict[Var, str] = {}
            for value, input_index in get_values.items():
                info = spec.inputs[input_index]
                if not info.block_shape:
                    values[value] = f"arg{input_index}[{offsets[input_index]}]"
                    continue
                local = cursor.fresh("tensorops_input")
                operand_offset = _broadcasted_tile_offset(
                    index, tile_shape, info.block_shape, info.array_shape
                )
                cursor.emit(
                    f"float {local} = arg{input_index}[{offsets[input_index]} + {operand_offset}];"
                )
                values[value] = local

            for eqn in elementwise_eqns:
                if eqn.primitive.name == "broadcast_in_dim":
                    if len(eqn.invars) != 1 or len(eqn.outvars) != 1:
                        raise EmitError(
                            "tensorops elementwise broadcast must have one input and output"
                        )
                    source = eqn.invars[0]
                    if source not in values:
                        raise EmitError("tensorops elementwise broadcast source was not lowered")
                    source_shape = _shape(source)
                    result_shape = _shape(eqn.outvars[0])
                    broadcast_dimensions = tuple(eqn.params.get("broadcast_dimensions", ()))
                    if (
                        len(result_shape) > rank
                        or len(broadcast_dimensions) != len(source_shape)
                        or broadcast_dimensions != tuple(sorted(set(broadcast_dimensions)))
                        or any(
                            axis < 0 or axis >= len(result_shape) for axis in broadcast_dimensions
                        )
                        or any(
                            source_shape[axis] != result_shape[broadcast_axis]
                            for axis, broadcast_axis in enumerate(broadcast_dimensions)
                        )
                        or any(
                            dim not in (1, tile_shape[rank - len(result_shape) + axis])
                            for axis, dim in enumerate(result_shape)
                        )
                    ):
                        raise EmitError(
                            "tensorops elementwise supports trailing row-vector broadcasts only"
                        )
                    imported = imported_operations.get(id(eqn))
                    if (
                        imported is None
                        or not imported.results
                        or imported.results[0].layout is None
                        or imported.results[0].layout.distribution is not Distribution.FLAT_STRIDED
                    ):
                        raise EmitError(
                            "tensorops elementwise layout assignment did not produce flat ownership"
                        )
                    values[eqn.outvars[0]] = values[source]
                    continue
                if (
                    eqn.primitive.name not in ELEMENTWISE
                    and typed_expression(eqn.primitive.name, "float") is None
                ):
                    raise EmitError(
                        f"tensorops cooperative elementwise primitive {eqn.primitive.name!r} is unsupported"
                    )
                result = eqn.outvars[0]
                if _dtype_name(result) != "float32" or _shape(result) != tile_shape:
                    raise EmitError(
                        "tensorops elementwise intermediates must match the output tile shape"
                    )
                imported = imported_operations.get(id(eqn))
                if (
                    imported is None
                    or not imported.results
                    or imported.results[0].layout is None
                    or imported.results[0].layout.distribution is not Distribution.FLAT_STRIDED
                ):
                    raise EmitError(
                        "tensorops elementwise layout assignment did not produce flat ownership"
                    )
                bindings = []
                for atom in eqn.invars:
                    if isinstance(atom, Literal):
                        if getattr(atom.aval, "shape", ()):
                            raise EmitError("tensorops elementwise only supports scalar literals")
                        bindings.append((atom, env.val(atom).expr))
                    elif atom in values:
                        bindings.append((atom, values[atom]))
                    else:
                        raise EmitError(
                            "tensorops elementwise has an unbound scalar operation operand"
                        )
                expression = elementwise_expression(eqn, "float", tuple(bindings))
                local = cursor.fresh("tensorops_value")
                cursor.emit(f"float {local} = {expression};")
                values[result] = local

            store_value = store.invars[1]
            if not isinstance(store_value, Var) or store_value not in values:
                raise EmitError("tensorops elementwise store value was not lowered")
            output_value = values[store_value]
            cursor.emit(f"arg{len(spec.inputs)}[{offsets[-1]} + {local_offset}] = {output_value};")

    return _kernel_source(name, params, cursor.lines), cursor.threadgroup_bytes
