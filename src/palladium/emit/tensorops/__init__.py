"""Cooperative lowerings: Pallas kernels built around a dot, emitted as one
threadgroup per program with Metal 4 TensorOps or `simdgroup_matrix`."""

from palladium.trace import KernelSpec

from ._shared import (
    SIMDGROUPS,
    _contains,
    declared_threadgroup_width,
    uses_tensorops,
)
from .attention import lower_attention
from .matmul import lower_matmul

__all__ = [
    "SIMDGROUPS",
    "compile_kernel",
    "declared_threadgroup_width",
    "uses_tensorops",
]


def compile_kernel(
    spec: KernelSpec, kernel_name: str | None = None, *, prefer_simdgroup: bool = False
) -> tuple[str, int]:
    """Choose a cooperative lowering and emit its Metal source.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``. The name must be a valid
        MSL identifier.
    prefer_simdgroup : bool, optional
        Lower supported matmuls and attention with `simdgroup_matrix` instead of
        `matmul2d`.

    Returns
    -------
    source : str
        Complete MSL using one threadgroup per Pallas program.
    threadgroup_bytes : int
        Explicitly declared shared storage per group.

    Raises
    ------
    EmitError
        The kernel is neither of the recognized forms. Under the "auto" policy the
        caller falls back to one thread per program.

    Notes
    -----
    A scan anywhere in the kernel selects the attention recognizer; otherwise the
    matmul recognizer.
    """
    if _contains(spec.jaxpr, "scan"):
        return lower_attention(spec, kernel_name, prefer_simdgroup=prefer_simdgroup)
    return lower_matmul(spec, kernel_name, prefer_simdgroup=prefer_simdgroup)
