"""Pick the cooperative TensorOps lowering for a kernel with a dot."""

from __future__ import annotations

from palladium.trace import KernelSpec

from ._shared import _contains
from .attention import lower_attention
from .matmul import lower_matmul


def compile_kernel(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """MSL source and threadgroup bytes: attention for a scan, matmul otherwise."""
    lower = lower_attention if _contains(spec.jaxpr, "scan") else lower_matmul
    return lower(spec, kernel_name)
