"""Pick the cooperative TensorOps lowering for a kernel with a dot."""

from __future__ import annotations

from palladium.trace import KernelSpec

from ._shared import _contains
from .attention import lower_attention
from .matmul import lower_matmul


def compile_kernel(
    spec: KernelSpec, kernel_name: str | None = None, *, prefer_simdgroup: bool = False
) -> tuple[str, int]:
    """Choose a cooperative lowering and emit Metal TensorOps source.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``. The name must be a valid
        MSL identifier.
    prefer_simdgroup : bool, optional
        Lower supported float32 matmuls and attention with `simdgroup_matrix`
        instead of `matmul2d`.

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
    if _contains(spec.jaxpr, "scan"):
        return lower_attention(spec, kernel_name, prefer_simdgroup=prefer_simdgroup)
    return lower_matmul(spec, kernel_name, prefer_simdgroup=prefer_simdgroup)
