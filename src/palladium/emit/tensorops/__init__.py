"""Pallas-to-MSL lowering built around Metal 4 TensorOps."""

from ._shared import (
    SIMDGROUPS,
    declared_threadgroup_width,
    uses_tensorops,
)
from .compile import compile_kernel

__all__ = [
    "SIMDGROUPS",
    "compile_kernel",
    "declared_threadgroup_width",
    "uses_tensorops",
]
