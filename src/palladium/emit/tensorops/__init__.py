"""Opt-in Metal TensorOps lowerings for matrix products and attention."""

from __future__ import annotations

from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import SIMDGROUPS, has_dot_general, uses_tensorops
from .attention import emit_tensorops_attention
from .matmul import (
    _emit_tensorops_matmul,
    _recognize_tensorops_matmul,
    emit_tensorops_matmul,
)


def emit_tensorops(spec: KernelSpec, kernel_name: str | None = None) -> tuple[str, int]:
    """Dispatch to the supported standalone matmul or fused attention lowering."""
    if any(eqn.primitive.name == "dot_general" for eqn in spec.jaxpr.eqns):
        return _emit_tensorops_matmul(_recognize_tensorops_matmul(spec), kernel_name)
    if any(eqn.primitive.name == "scan" for eqn in spec.jaxpr.eqns):
        return emit_tensorops_attention(spec, kernel_name)
    raise EmitError("TensorOps lowering could not find a supported matmul or attention pattern")


__all__ = [
    "SIMDGROUPS",
    "emit_tensorops",
    "emit_tensorops_attention",
    "emit_tensorops_matmul",
    "has_dot_general",
    "uses_tensorops",
]
