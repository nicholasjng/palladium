"""palladium: Pallas kernels on Apple GPUs.

trace (Pallas -> KernelSpec) and emit (KernelSpec -> MSL), dispatched two
ways: a plain `pl.pallas_call` on the jax-mps `mps` platform, or
`metal_call` from the CPU platform through a jax.ffi target. `debug_msl`
returns the intermediate MSL.
"""

from __future__ import annotations

import importlib.metadata as _metadata
from collections.abc import Callable

import palladium.mps  # registers palladium as the Pallas backend for mps
from palladium.diagnostics import KernelDiagnostics
from palladium.emit import emit_msl
from palladium.errors import (
    DispatchError,
    EmitError,
    PalladiumError,
    TraceError,
    UnsupportedPrimitiveError,
)
from palladium.ffi import metal_call
from palladium.launch import CompilerParams, MathMode
from palladium.trace import BlockInfo, KernelSpec, ScratchInfo, trace
from palladium.vjp import with_vjp

__all__ = [
    "BlockInfo",
    "CompilerParams",
    "DispatchError",
    "EmitError",
    "KernelDiagnostics",
    "KernelSpec",
    "MathMode",
    "PalladiumError",
    "ScratchInfo",
    "TraceError",
    "UnsupportedPrimitiveError",
    "debug_msl",
    "emit_msl",
    "metal_call",
    "trace",
    "with_vjp",
]

try:
    __version__ = _metadata.version("palladium")
except _metadata.PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"


def debug_msl(kernel: Callable, *example_args, **pallas_kwargs) -> str:
    """Return the Metal source for a Pallas kernel specialized to example inputs.

    Parameters
    ----------
    kernel : callable
        Pallas kernel operating on input and output Refs.
    *example_args : array-like or jax.ShapeDtypeStruct
        Input shapes and dtypes used for tracing; array contents are not read.
    **pallas_kwargs : dict
        Keywords for ``pl.pallas_call``, including required ``out_shape`` and optional
        grid, BlockSpecs, scratch, aliases, and ``CompilerParams``.

    Returns
    -------
    str
        Complete MSL source; emission does not compile or dispatch the kernel.

    Raises
    ------
    TraceError
        The Pallas call has unsupported structure, effects, aliases, or provably
        colliding writes.
    EmitError
        The kernel uses unsupported shapes, addressing, effects, or primitive
        parameters.
    UnsupportedPrimitiveError
        A staged primitive has no registered lowering rule.

    Notes
    -----
    This helper traces and emits only. It does not validate backend-specific launch options
    or compile Metal source. Captured arrays must be passed as kernel operands.

    See Also
    --------
    trace : Extract a reusable kernel specification.
    emit_msl : Emit source from an existing specification.
    """
    import jax.experimental.pallas as pl

    params = pallas_kwargs.get("compiler_params") or CompilerParams()
    spec = trace(pl.pallas_call(kernel, **pallas_kwargs), *example_args)
    return emit_msl(spec, dot_general=params.dot_general, program=params.program)
