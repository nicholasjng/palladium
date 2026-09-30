"""palladium: Pallas kernels on Apple GPUs.

trace (Pallas -> KernelSpec) and emit (KernelSpec -> MSL), dispatched two
ways: a plain `pl.pallas_call` on the jax-mps `mps` platform, or
`metal_call` from the CPU platform through a jax.ffi target. `debug_msl`
returns the intermediate MSL.
"""

from __future__ import annotations

import importlib.metadata as _metadata
from collections.abc import Callable

from palladium.diagnostics import KernelDiagnostics
from palladium.emit import emit_jaxpr, emit_msl, rule
from palladium.errors import (
    DispatchError,
    EmitError,
    PalladiumError,
    TraceError,
    UnsupportedPrimitiveError,
)
from palladium.ffi import FfiCallable, metal_call
from palladium.mps import MPS_CUSTOM_CALL_TARGET, MpsDispatchDescriptor
from palladium.pallas_backend import CompilerParams, install as _install_pallas_backend
from palladium.threadgroup import (
    barrier,
    thread_index,
    threadgroup_memory,
    threads_per_threadgroup,
)
from palladium.trace import BlockInfo, KernelSpec, ScratchInfo, trace
from palladium.verify import VerificationError
from palladium.vjp import with_auxiliary_vjp, with_reference_vjp, with_vjp

__all__ = [
    "MPS_CUSTOM_CALL_TARGET",
    "BlockInfo",
    "CompilerParams",
    "DispatchError",
    "EmitError",
    "FfiCallable",
    "KernelDiagnostics",
    "KernelSpec",
    "MpsDispatchDescriptor",
    "PalladiumError",
    "ScratchInfo",
    "TraceError",
    "UnsupportedPrimitiveError",
    "VerificationError",
    "barrier",
    "debug_msl",
    "emit_jaxpr",
    "emit_msl",
    "metal_call",
    "rule",
    "thread_index",
    "threadgroup_memory",
    "threads_per_threadgroup",
    "trace",
    "with_auxiliary_vjp",
    "with_reference_vjp",
    "with_vjp",
]

try:
    __version__ = _metadata.version("palladium")
except _metadata.PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

# Plain pl.pallas_call lowered for the mps platform runs through Palladium.
_install_pallas_backend()


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
        The usual `pl.pallas_call` keywords (out_shape, grid, ...).

    Returns
    -------
    str
        The MSL source Palladium would compile for these shapes.
    """
    import jax.experimental.pallas as pl

    dot_general = pallas_kwargs.pop("dot_general", "auto")
    spec = trace(pl.pallas_call(kernel, **pallas_kwargs), *example_args)
    return emit_msl(spec, dot_general=dot_general)
