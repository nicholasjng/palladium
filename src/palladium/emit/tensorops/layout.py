"""Ownership assignment for values in the tensorops Pallas IR."""

from __future__ import annotations

import dataclasses

from jax.extend.core import Literal, Var

from .ir import IROperation, IRRegion, IRValue, KernelIR
from .plan import AddressSpace, Distribution, ProgramScope, ValueLayout


def assign_layouts(kernel: KernelIR) -> KernelIR:
    """Assign storage and ownership layouts to jaxpr values.

    Dots receive TensorOps ownership, nonscalar row reductions row-strided
    ownership, and other threadgroup values inherit a distributed operand
    layout or default to flat striding.
    """

    body, _ = _assign_region(kernel.body, kernel.plan.scope)
    return dataclasses.replace(kernel, body=body)


def _assign_region(
    region: IRRegion,
    scope: ProgramScope,
    incoming: dict[Var, ValueLayout] | None = None,
) -> tuple[IRRegion, dict[Var, ValueLayout]]:
    layouts: dict[Var, ValueLayout] = dict(incoming or {})
    arguments = tuple(
        _with_layout(value, layouts.get(value.atom, value.layout)) for value in region.arguments
    )
    for value in arguments:
        if isinstance(value.atom, Var) and value.layout is not None:
            layouts[value.atom] = value.layout

    operations: list[IROperation] = []
    for operation in region.operations:
        operands = tuple(_with_layout(v, _current_layout(v, layouts)) for v in operation.operands)
        child_regions = []
        for child in operation.regions:
            child_incoming = {
                arg.atom: op.layout
                for arg, op in zip(child.arguments, operands, strict=False)
                if isinstance(arg.atom, Var) and op.layout is not None
            }
            assigned_child, _ = _assign_region(child, scope, child_incoming)
            child_regions.append(assigned_child)

        results = tuple(
            _with_layout(value, _result_layout(operation, operands, scope, index))
            for index, value in enumerate(operation.results)
        )
        for result in results:
            if isinstance(result.atom, Var) and result.layout is not None:
                layouts[result.atom] = result.layout
        operations.append(
            dataclasses.replace(
                operation,
                operands=operands,
                results=results,
                regions=tuple(child_regions),
            )
        )

    results = tuple(
        _with_layout(value, _current_layout(value, layouts)) for value in region.results
    )
    return dataclasses.replace(
        region, arguments=arguments, operations=tuple(operations), results=results
    ), layouts


def _current_layout(value: IRValue, layouts: dict[Var, ValueLayout]) -> ValueLayout | None:
    if isinstance(value.atom, Literal):
        return value.layout
    return layouts.get(value.atom, value.layout)


def _with_layout(value: IRValue, layout: ValueLayout | None) -> IRValue:
    return dataclasses.replace(value, layout=layout)


def _result_layout(
    operation: IROperation,
    operands: tuple[IRValue, ...],
    scope: ProgramScope,
    index: int,
) -> ValueLayout | None:
    result = operation.results[index]
    shape = result.layout.shape if result.layout is not None else ()
    dtype = result.layout.dtype if result.layout is not None else "float32"
    if scope is ProgramScope.THREAD or not shape:
        if result.layout is not None:
            return result.layout
        return ValueLayout(shape, dtype, AddressSpace.THREAD)

    if operation.name == "dot_general":
        return ValueLayout(shape, dtype, AddressSpace.THREADGROUP, Distribution.TENSOROPS)

    if operation.name in ("reduce_max", "reduce_sum") and shape:
        return ValueLayout(shape, dtype, AddressSpace.THREADGROUP, Distribution.ROW_STRIDED)

    if operation.name == "scan":
        num_consts = int(operation.equation.params.get("num_consts", 0))
        num_carry = int(operation.equation.params.get("num_carry", len(operation.results)))
        if index < num_carry:
            input_index = num_consts + index
            if input_index < len(operands) and operands[input_index].layout is not None:
                return operands[input_index].layout

    matching = [
        operand.layout
        for operand in operands
        if operand.layout is not None and operand.layout.shape == shape and shape
    ]
    distributed = next(
        (
            layout
            for layout in matching
            if layout.address_space is AddressSpace.THREADGROUP
            and layout.distribution is not Distribution.UNASSIGNED
        ),
        None,
    )
    if distributed is not None:
        return ValueLayout(shape, dtype, AddressSpace.THREADGROUP, distributed.distribution)
    return ValueLayout(shape, dtype, AddressSpace.THREADGROUP, Distribution.FLAT_STRIDED)
