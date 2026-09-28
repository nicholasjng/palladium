"""TensorOps planning and compositional MSL lowerings."""

from __future__ import annotations

import dataclasses

from palladium.emit.core import emit_msl_stats
from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import SIMDGROUPS, uses_tensorops
from .attention import lower_attention_ir
from .elementwise import lower_elementwise_ir
from .ir import IRRegion, KernelIR, import_kernel
from .layout import assign_layouts
from .matmul import lower_matmul_ir
from .plan import Distribution, KernelPlan, ProgramScope, plan_kernel
from .reduction import lower_row_reduction_ir


@dataclasses.dataclass(frozen=True)
class Compilation:
    """TensorOps plan and analyzed IR alongside emitted MSL source."""

    plan: KernelPlan
    ir: KernelIR
    source: str
    threadgroup_bytes: int


def compile_kernel(
    spec: KernelSpec,
    kernel_name: str | None = None,
    *,
    scope: ProgramScope | None = None,
    simdgroups: int = 4,
    dot_general: str = "auto",
) -> Compilation:
    """Run tensorops planning and layout analysis, then lower the supported kernel.

    Thread programs use the existing primitive emitter. Threadgroup attention
    and matmuls are selected from imported tensorops IR and emitted through shared
    TensorOps and cooperative MSL primitives.
    """

    if dot_general not in ("auto", "default", "tensorops"):
        raise ValueError("dot_general must be 'auto', 'default', or 'tensorops'")
    if scope is None and uses_tensorops(spec, dot_general):
        scope = ProgramScope.THREADGROUP
    plan = plan_kernel(spec, scope=scope, simdgroups=simdgroups)
    ir = assign_layouts(import_kernel(plan))
    operations = tuple(_walk_operations(ir.body))
    if plan.scope is ProgramScope.THREAD:
        if uses_tensorops(spec, dot_general):
            raise EmitError("TensorOps dots require threadgroup program scope")
        source, stats = emit_msl_stats(
            spec, kernel_name, dot_general="default" if dot_general == "tensorops" else dot_general
        )
        return Compilation(plan, ir, source, stats.threadgroup_bytes)
    if plan.simdgroups != SIMDGROUPS:
        raise EmitError(f"TensorOps lowering currently requires {SIMDGROUPS} simdgroups")

    dots = tuple(operation for operation in operations if operation.name == "dot_general")
    scans = tuple(operation for operation in operations if operation.name == "scan")
    if dots and any(
        result.layout is None or result.layout.distribution is not Distribution.TENSOROPS
        for operation in dots
        for result in operation.results
    ):
        raise EmitError(
            "tensorops layout assignment did not give every cooperative dot TensorOps ownership"
        )
    reductions = tuple(
        operation
        for operation in operations
        if operation.name in ("reduce_max", "reduce_sum")
        and operation.results
        and operation.results[0].layout is not None
        and operation.results[0].layout.shape
    )
    if any(
        operation.results[0].layout.distribution is not Distribution.ROW_STRIDED
        for operation in reductions
    ):
        raise EmitError(
            "tensorops layout assignment did not give cooperative row reductions row ownership"
        )

    if scans:
        lower = lower_attention_ir
    elif dots:
        lower = lower_matmul_ir
    elif reductions:
        lower = lower_row_reduction_ir
    else:
        lower = lower_elementwise_ir
    source, threadgroup_bytes = lower(ir, kernel_name)
    return Compilation(plan, ir, source, threadgroup_bytes)


def _walk_operations(region: IRRegion):
    for operation in region.operations:
        yield operation
        for child in operation.regions:
            yield from _walk_operations(child)
