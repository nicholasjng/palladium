"""The `metal` PJRT plugin (`native/pjrt`), registered with jax through the
`jax_plugins` entry point, which calls `initialize` on backend discovery.

The plugin compiles through a callback into Python: `palladium.stablehlo`
turns the program into buffer slots and steps, which `_compile` describes to
the plugin through its pp_* builder functions. Execution stays native, except
for host calls: Python callbacks and ops without a kernel, which the plugin
runs through `_host` once the GPU has computed their inputs.

It registers only when JAX_PLATFORMS names `metal` (e.g.
`JAX_PLATFORMS=metal,cpu`): otherwise every jax process would bring it up
and warn that it is experimental, and unsupported programs fail to compile.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import ml_dtypes
import numpy as np
from jax._src import xla_bridge  # register_plugin has no public name
from jax._src.interpreters import mlir

from palladium._runtime import installed_library, load_library

if TYPE_CHECKING:
    from palladium.stablehlo.program import HostCall

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
_HOST = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
)
_INT64S = ctypes.POINTER(ctypes.c_int64)
_SIGNATURES = {
    "palladium_pjrt_set_compiler": (None, [_COMPILER]),
    "palladium_pjrt_set_host": (None, [_HOST]),
    "palladium_pjrt_set_stablehlo_version": (None, [ctypes.c_int64] * 3),
    "palladium_pjrt_released": (ctypes.c_size_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]),
    "pp_host": (
        None,
        [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ],
    ),
    "pp_branch": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]),
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
    "palladium_pjrt_profile": (ctypes.c_size_t, [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]),
}


def library_path() -> Path:
    """The plugin library in the installed package; reject a missing build."""
    return installed_library(_LIBRARY_NAME, "PJRT plugin")


@functools.cache
def library() -> ctypes.CDLL:
    """The plugin, loaded once with the builder signatures declared; the same
    image jax loads, so the compiler set here is the one it calls."""
    return load_library(library_path(), _SIGNATURES)


def stats(reset: bool = False) -> dict[str, float]:
    """Execution counters of the plugin since the last reset.

    Parameters
    ----------
    reset : bool
        Zero the counters after reading them.

    Returns
    -------
    dict
        ``executions``, ``command_buffers`` and ``launches`` encoded.
        ``encode_seconds`` and ``throttle_seconds`` of host time inside
        Execute. ``gpu_busy_seconds``, the union of the command buffers' GPU
        intervals, and ``gpu_sum_seconds``, their sum. Of the buffer pool:
        ``pool_hits``; ``pool_misses``, ``allocated_bytes`` and
        ``allocation_seconds`` for new MTLBuffers; and ``pool_dropped``,
        the buffers it freed.

    Notes
    -----
    Completion handlers record GPU time, so synchronize on the last outputs
    before reading. The pool evicts its oldest buffers to stay within a
    quarter of the device's recommended working set, or within
    ``PALLADIUM_POOL_BYTES``.
    """
    out = _Stats()
    library().palladium_pjrt_stats(ctypes.byref(out), int(reset))
    return {field[0]: getattr(out, field[0]) for field in _Stats._fields_}


def profile() -> list[float]:
    """GPU seconds of each step of the last execution, in program order.

    Returns
    -------
    list of float
        One entry per step of the executed program (zero for host steps and
        branches); empty unless the process runs with
        ``PALLADIUM_PJRT_PROFILE`` set.

    Notes
    -----
    Profiling runs every launch in a command buffer of its own and waits for
    it, so an execution takes longer than the sum, and launches that would
    overlap do not.
    """
    lib = library()
    count = lib.palladium_pjrt_profile(None, 0)
    seconds = (ctypes.c_double * count)()
    lib.palladium_pjrt_profile(seconds, count)
    return list(seconds)


class _Recorded(Exception):
    """A builder call failed; the plugin holds its error."""


# Host calls of live programs by id. The plugin reports the ids of destroyed
# programs.
_host_calls: dict[int, HostCall] = {}
_host_ids = itertools.count()
_host_lock = threading.Lock()


def _forget_released(lib: ctypes.CDLL) -> None:
    ids = (ctypes.c_int * 256)()
    while count := lib.palladium_pjrt_released(ids, len(ids)):
        with _host_lock:
            for k in ids[:count]:
                _host_calls.pop(k, None)


def _view(address: int | None, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """The buffer at `address` as an array, without a copy."""
    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if not size or not address:
        return np.empty(shape, dtype)
    return np.frombuffer((ctypes.c_char * size).from_address(address), dtype).reshape(shape)


@_HOST
def _host(key: int, data, count: int, message: int, capacity: int) -> int:
    try:
        with _host_lock:
            call = _host_calls[key]
        arrays = [_view(data[k], dtype, shape) for k, (dtype, shape) in enumerate(call.types)]
        inputs, outputs = arrays[: call.inputs], arrays[call.inputs :]
        # Inputs are copies: the function may keep them, and the buffers
        # return to the pool.
        results = call.function(*(a.copy() for a in inputs))
        if len(results) != len(outputs):
            raise RuntimeError(f"{call.name} returned {len(results)} results, not {len(outputs)}")
        for k, (result, output) in enumerate(zip(results, outputs, strict=True)):
            result = np.asarray(result)
            if result.shape != output.shape or result.dtype != output.dtype:
                raise RuntimeError(
                    f"{call.name} returned {result.dtype}{list(result.shape)} for result {k}, "
                    f"not {output.dtype}{list(output.shape)}"
                )
            output[...] = result
    except BaseException as e:  # noqa: BLE001 - an exception must not unwind into the plugin
        text = f"metal: host call failed: {type(e).__name__}: {e}".encode()[:capacity]
        ctypes.memmove(message, text + b"\0", len(text) + 1)
        return 1
    return 0


def _describe(lib: ctypes.CDLL, builder: int, code: bytes) -> None:
    from palladium.stablehlo import compile_program
    from palladium.stablehlo.program import Branch, HostCall

    _forget_released(lib)
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
    for launch in program.steps:
        if isinstance(launch, Branch):
            lib.pp_branch(builder, -1 if launch.slot is None else launch.slot, launch.skip)
            continue
        slots = (ctypes.c_int * len(launch.slots))(*launch.slots)
        if isinstance(launch, HostCall):
            with _host_lock:
                key = next(_host_ids)
                _host_calls[key] = launch
            offsets = launch.offsets or (0,) * len(slots)
            lib.pp_host(builder, key, slots, len(slots), (ctypes.c_size_t * len(slots))(*offsets))
            continue
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
    if "metal" in (jax.config.jax_platforms or "").split(","):
        xla_bridge.register_plugin("metal", library_path=str(library_path()))
        library().palladium_pjrt_set_compiler(_compile)
        library().palladium_pjrt_set_host(_host)
        # jaxlib serializes each program for the StableHLO version the plugin
        # reports; the compiler reads it with jaxlib's own bindings.
        from jaxlib.mlir.dialects import stablehlo

        version = stablehlo.get_current_version().split(".")
        library().palladium_pjrt_set_stablehlo_version(*(int(v) for v in version))
        if "metal" not in mlir._platforms_with_donation:
            mlir._platforms_with_donation.append("metal")
        # JAX accepts lowering rules for `metal` only after it is registered.
        from jax.experimental import pallas as pl

        from palladium import host
        from palladium.ffi import pallas_lowering

        mlir.register_lowering(pl.pallas_call_p, pallas_lowering, platform="metal")
        host.register()
