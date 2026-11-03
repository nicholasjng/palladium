"""The `metal` PJRT plugin (`native/pjrt`), registered with jax through the
`jax_plugins` entry point, which calls `initialize` on backend discovery.

The plugin compiles through a callback into Python: `palladium.stablehlo`
turns the program into buffer slots and kernel launches, which `_compile`
describes to the plugin through its pp_* builder functions. Execution stays
native.

It registers only when JAX_PLATFORMS names `metal` (e.g.
`JAX_PLATFORMS=metal,cpu`): most programs do not compile yet, and otherwise
every jax process would bring it up and warn that it is experimental.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.resources
import traceback
from pathlib import Path

import ml_dtypes
import numpy as np
from jax._src import config, xla_bridge
from jax._src.interpreters import mlir

_LIBRARY_NAME = "libpalladium_pjrt.dylib"

# PJRT_Buffer_Type by dtype.
_PJRT_TYPES = {
    np.dtype(np.bool_): 1,
    np.dtype(np.int8): 2,
    np.dtype(np.int16): 3,
    np.dtype(np.int32): 4,
    np.dtype(np.int64): 5,
    np.dtype(np.uint8): 6,
    np.dtype(np.uint16): 7,
    np.dtype(np.uint32): 8,
    np.dtype(np.uint64): 9,
    np.dtype(np.float16): 10,
    np.dtype(np.float32): 11,
    np.dtype(np.float64): 12,
    np.dtype(ml_dtypes.bfloat16): 13,
}
_SLOT_KINDS = {"parameter": 0, "fresh": 1, "constant": 2}
_UNIMPLEMENTED, _INTERNAL = 12, 13


class _Stats(ctypes.Structure):
    _fields_ = [
        ("executions", ctypes.c_uint64),
        ("command_buffers", ctypes.c_uint64),
        ("launches", ctypes.c_uint64),
        ("encode_seconds", ctypes.c_double),
        ("throttle_seconds", ctypes.c_double),
        ("gpu_busy_seconds", ctypes.c_double),
        ("gpu_sum_seconds", ctypes.c_double),
        ("pool_hits", ctypes.c_uint64),
        ("pool_misses", ctypes.c_uint64),
        ("allocated_bytes", ctypes.c_uint64),
        ("pool_dropped", ctypes.c_uint64),
        ("allocation_seconds", ctypes.c_double),
    ]


_COMPILER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
_INT64S = ctypes.POINTER(ctypes.c_int64)
_SIGNATURES = {
    "palladium_pjrt_set_compiler": (None, [_COMPILER]),
    "pp_error": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]),
    "pp_slot": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            _INT64S,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ],
    ),
    "pp_kernel": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ],
    ),
    "pp_output": (None, [ctypes.c_void_p, ctypes.c_int]),
    "pp_alias": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    "palladium_pjrt_stats": (None, [ctypes.POINTER(_Stats), ctypes.c_int]),
}


def library_path() -> Path:
    """The plugin library in the installed package; reject a missing build."""
    candidate = importlib.resources.files("palladium").joinpath(_LIBRARY_NAME)
    if not candidate.is_file():
        raise FileNotFoundError(f"palladium's PJRT plugin ({_LIBRARY_NAME}) isn't built")
    return Path(str(candidate))


@functools.cache
def library() -> ctypes.CDLL:
    """The plugin, loaded once with the builder signatures declared; the same
    image jax loads, so the compiler set here is the one it calls."""
    lib = ctypes.CDLL(str(library_path()))
    for name, (restype, argtypes) in _SIGNATURES.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


def stats(reset: bool = False) -> dict[str, float]:
    """Execution counters of the plugin since the last reset.

    Parameters
    ----------
    reset : bool
        Zero the counters after reading them.

    Returns
    -------
    dict
        ``executions``, ``command_buffers`` and ``launches`` encoded;
        ``encode_seconds`` and ``throttle_seconds`` of host time inside
        Execute; ``gpu_busy_seconds`` (union of command-buffer GPU intervals)
        and ``gpu_sum_seconds``; the buffer pool's ``pool_hits``, new
        MTLBuffers (``pool_misses``, ``allocated_bytes``,
        ``allocation_seconds``) and releases it dropped when full
        (``pool_dropped``). GPU time is recorded by completion handlers, so
        synchronize on the last outputs before reading.
    """
    out = _Stats()
    library().palladium_pjrt_stats(ctypes.byref(out), int(reset))
    return {field[0]: getattr(out, field[0]) for field in _Stats._fields_}


class _Recorded(Exception):
    """A builder call failed; the plugin holds its error."""


def _describe(lib: ctypes.CDLL, builder: int, code: bytes) -> None:
    from palladium.stablehlo import compile_program

    program = compile_program(code)
    for slot in program.slots:
        dims = (ctypes.c_int64 * len(slot.shape))(*slot.shape)
        data = slot.data or b""
        if slot.dtype not in _PJRT_TYPES:
            raise ValueError(f"metal: no PJRT type for {slot.dtype}")
        status = lib.pp_slot(
            builder,
            _SLOT_KINDS[slot.kind],
            _PJRT_TYPES[slot.dtype],
            dims,
            len(slot.shape),
            data,
            len(data),
        )
        if status < 0:
            raise _Recorded
    for launch in program.launches:
        slots = (ctypes.c_int * len(launch.slots))(*launch.slots)
        # An int is the x extent; a zero threadgroup lets the pipeline choose.
        grid, group = (
            (ctypes.c_size_t * 3)(*((n, 1, 1) if isinstance(n, int) else n))
            for n in (launch.grid, launch.threadgroup)
        )
        status = lib.pp_kernel(
            builder,
            launch.source.encode(),
            launch.function.encode(),
            launch.math_mode,
            slots,
            len(launch.slots),
            grid,
            group,
            (ctypes.c_size_t * len(launch.offsets))(*launch.offsets) if launch.offsets else None,
        )
        if status < 0:
            raise _Recorded
    for slot in program.outputs:
        lib.pp_output(builder, slot)
    for parameter, output in program.aliases:
        lib.pp_alias(builder, parameter, output)


@_COMPILER
def _compile(code: int, size: int, builder: int) -> int:
    from palladium.stablehlo import UnsupportedProgram

    lib = library()
    try:
        _describe(lib, builder, ctypes.string_at(code, size))
    except _Recorded:
        return 1
    except UnsupportedProgram as e:
        lib.pp_error(builder, _UNIMPLEMENTED, str(e).encode())
        return 1
    except Exception:  # noqa: BLE001 - an exception must not unwind into the plugin
        lib.pp_error(builder, _INTERNAL, f"metal: {traceback.format_exc()}".encode())
        return 1
    return 0


def initialize() -> None:
    if "metal" in (config.jax_platforms.value or "").split(","):
        xla_bridge.register_plugin("metal", library_path=str(library_path()))
        library().palladium_pjrt_set_compiler(_compile)
        if "metal" not in mlir._platforms_with_donation:
            mlir._platforms_with_donation.append("metal")
