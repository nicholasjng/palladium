"""Pick the cooperative TensorOps lowering for a kernel with a dot."""

from __future__ import annotations

from palladium.trace import KernelSpec

from ._shared import _contains
from .attention import lower_attention
from .matmul import lower_matmul


def compile_kernel(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Choose a cooperative lowering and emit Metal TensorOps source.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``. The name must be a valid
        MSL identifier.

    Returns
    -------
    source : str
        Complete MSL using one threadgroup per Pallas program.
    threadgroup_bytes : int
        Explicitly declared shared storage per group.

    Raises
    ------
    EmitError
        The kernel dataflow, types, layouts, tiles, or device shared-memory budget do
        not match the supported cooperative form.

    Notes
    -----
    A scan anywhere in the kernel selects the attention recognizer; otherwise matmul is
    attempted. Recognition errors propagate here; the outer auto policy is responsible for
    fallback.
    """
    lower = lower_attention if _contains(spec.jaxpr, "scan") else lower_matmul
    return lower(spec, kernel_name)
