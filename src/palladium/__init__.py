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
from palladium.launch import CompilerParams
from palladium.trace import BlockInfo, KernelSpec, ScratchInfo, trace
from palladium.vjp import with_vjp

__all__ = [
    "BlockInfo",
    "CompilerParams",
    "DispatchError",
    "EmitError",
    "KernelDiagnostics",
    "KernelSpec",
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
    """Trace `kernel` through pallas_call and return the emitted MSL.

    Parameters
    ----------
    kernel : callable
        A Pallas kernel function (operates on Refs).
    *example_args
        Arrays or `jax.ShapeDtypeStruct`s fixing input shapes; no data is
        read and nothing is compiled or dispatched.
    **pallas_kwargs
        The usual `pl.pallas_call` keywords (out_shape, grid, ...), with
        options in `compiler_params=palladium.CompilerParams(...)`.

    Returns
    -------
    str
        The MSL source Palladium would compile for these shapes.
    """
    import jax.experimental.pallas as pl

    params = pallas_kwargs.get("compiler_params") or CompilerParams()
    spec = trace(pl.pallas_call(kernel, **pallas_kwargs), *example_args)
    return emit_msl(spec, dot_general=params.dot_general)
