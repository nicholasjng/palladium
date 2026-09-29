"""Palladium as the Pallas backend for the ``mps`` platform.

Importing palladium installs a lowering for the ``pallas_call`` primitive
that, when a program is lowered for ``mps``, emits a ``palladium.dispatch``
custom call carrying the kernel. A plain
``pl.pallas_call`` under ``jax.jit`` on a jax-mps device therefore runs
Palladium's Metal kernel; every other platform keeps JAX's own lowering.

Metal-side options travel as ``compiler_params``::

    pl.pallas_call(kernel, out_shape=..., compiler_params=palladium.CompilerParams(
        dot_general="tensorops", threadgroup=128))
"""

from __future__ import annotations

import dataclasses
from typing import Any

from jax._src import effects as jax_effects
from jax._src.interpreters import mlir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.pallas_call import pallas_call_p
from metal_runtime import MathMode

from palladium._callable import DOT_GENERAL_POLICIES
from palladium.diagnostics import check_threadgroup, normalize_threadgroup
from palladium.emit import emit_msl
from palladium.mps import MpsDispatchDescriptor, lower_dispatch
from palladium.trace import spec_from_params

__all__ = ["CompilerParams", "install"]

_FAST_ORDINAL = 2


@dataclasses.dataclass(frozen=True)
class CompilerParams(pallas_core.CompilerParams):
    """Metal-side options for a ``pl.pallas_call`` lowered by Palladium.

    Attributes
    ----------
    dot_general : str
        "auto" (TensorOps for tiled matmuls and attention, the default),
        "tensorops" to require it, or "default" for the primitive path.
    threadgroup : int, tuple, or None
        Explicit threadgroup size; required by cooperative kernels.
    math_mode : metal_runtime.MathMode
        Only FAST is available on the jax-mps path today.
    """

    BACKEND: str = "palladium"
    dot_general: str = "auto"
    threadgroup: int | tuple[int, ...] | None = None
    math_mode: MathMode = MathMode.FAST

    def __post_init__(self) -> None:
        if self.dot_general not in DOT_GENERAL_POLICIES:
            raise ValueError("dot_general must be 'auto', 'default', or 'tensorops'")


def _palladium_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, interpret: Any, **params):
    """Lower one pallas_call as a Palladium Metal kernel for jax-mps."""
    options = params.get("compiler_params")
    if options is None:
        options = CompilerParams()
    elif not isinstance(options, CompilerParams):
        raise TypeError(
            f"pallas_call on mps lowers through Palladium, which takes "
            f"palladium.CompilerParams, not {type(options).__name__}"
        )
    if options.math_mode != MathMode.FAST:
        raise ValueError(
            "mps lowering supports math_mode=FAST only; jax-mps's metal_kernel "
            "API does not expose Palladium's SAFE/RELAXED modes"
        )
    spec = spec_from_params(params)
    if spec.aliases:
        raise ValueError(
            "input_output_aliases are not supported on the mps path: jax-mps's "
            "MLX custom-kernel path allocates functional outputs"
        )
    threadgroup = normalize_threadgroup(options.threadgroup)
    check_threadgroup(spec, threadgroup)
    msl = emit_msl(spec, dot_general=options.dot_general)
    descriptor = MpsDispatchDescriptor.from_spec(
        spec, msl, threadgroup=threadgroup, math_mode=_FAST_ORDINAL
    )
    return lower_dispatch(ctx, *in_nodes, descriptor=descriptor)


_installed = False


def install() -> None:
    """Route pallas_call lowering for the ``mps`` platform through Palladium.

    Registered as the primitive's common rule wrapping JAX's own, so it
    applies whether or not the jax-mps plugin was discovered before this
    import: JAX only accepts platform-specific registrations for platforms
    it already knows. Other platforms and ``interpret=True`` are delegated
    unchanged.
    """
    global _installed
    if _installed:
        return
    original = mlir._lowerings[pallas_call_p].rule

    def lowering(ctx, *in_nodes, interpret, **params):
        if interpret:
            return original(ctx, *in_nodes, interpret=interpret, **params)
        return mlir.lower_per_platform(
            ctx,
            "pallas_call",
            {"mps": _palladium_lowering},
            original,
            jax_effects.no_effects,
            *in_nodes,
            interpret=interpret,
            **params,
        )

    mlir.register_lowering(pallas_call_p, lowering)
    _installed = True
