"""ctypes declarations of the Metal runtime's C API (`native/runtime/c_api.h`).

`libpalladium_ffi` links the runtime statically and exports its `mr_*`
functions: the jax.ffi handler calls them from C++, and Python reaches them
here. `Kernel` and `run` dispatch hand-written MSL directly, for baselines,
benchmarks and tests; they are not palladium's API. `PALLADIUM_FFI_LIBRARY`
overrides the library path for an out-of-tree build.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.resources
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_LIBRARY_NAME = "libpalladium_ffi.dylib"

# MRStatus.
OK = 0
ERROR_NO_DEVICE = 1
ERROR_COMPILE = 2
ERROR_FUNCTION_NOT_FOUND = 3
ERROR_DISPATCH = 4
ERROR_INVALID_ARGUMENT = 5
ERROR_UNKNOWN = 6
ERROR_ALLOCATION = 7

# MRMathMode ordinals by MathMode value.
MATH_MODES = {"safe": 0, "relaxed": 1, "fast": 2}


class LaunchDesc(ctypes.Structure):
    _fields_ = [
        ("pipeline", ctypes.c_void_p),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("buffer_offsets", ctypes.POINTER(ctypes.c_size_t)),
        ("buffer_count", ctypes.c_size_t),
        ("scalars", ctypes.POINTER(ctypes.c_void_p)),
        ("scalar_sizes", ctypes.POINTER(ctypes.c_size_t)),
        ("scalar_count", ctypes.c_size_t),
        ("threadgroup_memory", ctypes.POINTER(ctypes.c_size_t)),
        ("threadgroup_memory_count", ctypes.c_size_t),
        ("grid_x", ctypes.c_size_t),
        ("grid_y", ctypes.c_size_t),
        ("grid_z", ctypes.c_size_t),
        ("threadgroup_x", ctypes.c_size_t),
        ("threadgroup_y", ctypes.c_size_t),
        ("threadgroup_z", ctypes.c_size_t),
    ]


class DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("max_threads_per_threadgroup", ctypes.c_size_t),
        ("max_threadgroup_memory_length", ctypes.c_size_t),
        ("max_buffer_length", ctypes.c_size_t),
        ("recommended_max_working_set_size", ctypes.c_size_t),
        ("has_unified_memory", ctypes.c_int),
        ("supports_non_uniform_threadgroups", ctypes.c_int),
        ("apple_gpu_family", ctypes.c_int),
        ("name", ctypes.c_char * 128),
    ]


class PipelineInfo(ctypes.Structure):
    _fields_ = [
        ("thread_execution_width", ctypes.c_size_t),
        ("max_threads_per_threadgroup", ctypes.c_size_t),
        ("static_threadgroup_memory_length", ctypes.c_size_t),
    ]


_ERR = ctypes.POINTER(ctypes.c_char_p)
_OUT = ctypes.POINTER(ctypes.c_void_p)
_HANDLE = ctypes.c_void_p
_SIGNATURES = {
    "mr_free_error_message": (None, [ctypes.c_char_p]),
    "mr_compile_library": (
        ctypes.c_int,
        [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, _OUT, _ERR],
    ),
    "mr_release_library": (None, [_HANDLE]),
    "mr_get_pipeline": (ctypes.c_int, [_HANDLE, ctypes.c_char_p, _OUT, _ERR]),
    "mr_release_pipeline": (None, [_HANDLE]),
    "mr_device_info": (ctypes.c_int, [ctypes.POINTER(DeviceInfo), _ERR]),
    "mr_pipeline_info": (ctypes.c_int, [_HANDLE, ctypes.POINTER(PipelineInfo), _ERR]),
    "mr_wrap_buffer": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_size_t, _OUT, _ERR]),
    "mr_buffer_flush_to": (ctypes.c_int, [_HANDLE, _ERR]),
    "mr_release_buffer": (None, [_HANDLE]),
    "mr_clear_wrap_cache": (None, []),
    "mr_dispatch": (ctypes.c_int, [ctypes.POINTER(LaunchDesc), _ERR]),
    "mr_batch_create": (ctypes.c_int, [_OUT, _ERR]),
    "mr_batch_add": (ctypes.c_int, [_HANDLE, ctypes.POINTER(LaunchDesc), _ERR]),
    "mr_batch_commit": (ctypes.c_int, [_HANDLE, _ERR]),
    "mr_batch_wait": (ctypes.c_int, [_HANDLE, _ERR]),
    "mr_batch_gpu_time": (ctypes.c_int, [_HANDLE, ctypes.POINTER(ctypes.c_double), _ERR]),
    "mr_release_batch": (None, [_HANDLE]),
}


def library_path() -> Path:
    """Resolve the native library or its environment override; reject a missing build."""
    override = os.environ.get("PALLADIUM_FFI_LIBRARY")
    if override:
        return Path(override)
    # An editable install spans two roots (source tree and CMake build
    # dir); joinpath resolves to whichever root has the file.
    candidate = importlib.resources.files("palladium").joinpath(_LIBRARY_NAME)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"palladium's native library ({_LIBRARY_NAME}) isn't built or "
            "installed. `uv sync`/`pip install` builds it automatically via "
            "CMakeLists.txt; if this is a from-source checkout, re-run that. "
            "Otherwise set PALLADIUM_FFI_LIBRARY to point at an existing build."
        )
    return Path(str(candidate))


@functools.cache
def library() -> ctypes.CDLL:
    """Load the native library once, with the C API's signatures declared."""
    lib = ctypes.CDLL(str(library_path()))
    for name, (restype, argtypes) in _SIGNATURES.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


class MetalError(RuntimeError):
    """A C API call failed; the message is the runtime's."""


def _check(status: int, err: ctypes.c_char_p) -> None:
    if status == OK:
        return
    message = err.value.decode(errors="replace") if err.value else f"MRStatus {status}"
    if err.value:
        library().mr_free_error_message(err)
    raise MetalError(message)


class Kernel:
    """An MSL function compiled for direct dispatch with `run`.

    Parameters
    ----------
    source : str
        Complete MSL source.
    function_name : str
        Entry point.
    math_mode : str, optional
        "fast" (default), "relaxed" or "safe"; a `palladium.MathMode` works too.

    Raises
    ------
    MetalError
        The source does not compile, or has no such function.
    """

    def __init__(self, source: str, function_name: str, math_mode: str = "fast"):
        self._library, self._pipeline = ctypes.c_void_p(), ctypes.c_void_p()
        self.math_mode = math_mode
        lib, err = library(), ctypes.c_char_p()
        encoded = source.encode()
        status = lib.mr_compile_library(
            encoded,
            len(encoded),
            MATH_MODES[math_mode],
            ctypes.byref(self._library),
            ctypes.byref(err),
        )
        _check(status, err)
        name = function_name.encode()
        _check(
            lib.mr_get_pipeline(
                self._library, name, ctypes.byref(self._pipeline), ctypes.byref(err)
            ),
            err,
        )
        info = PipelineInfo()
        _check(lib.mr_pipeline_info(self._pipeline, ctypes.byref(info), ctypes.byref(err)), err)
        self.thread_execution_width = info.thread_execution_width

    def __del__(self) -> None:
        if self._pipeline:
            library().mr_release_pipeline(self._pipeline)
        if self._library:
            library().mr_release_library(self._library)


def run(
    kernel: Kernel,
    buffers: Sequence[np.ndarray],
    grid: int | Sequence[int],
    threadgroup: int | Sequence[int] | None = None,
    scalars: Sequence[np.generic | np.ndarray] = (),
) -> float:
    """Dispatch `kernel` once and return its GPU execution seconds.

    Parameters
    ----------
    kernel : Kernel
        Compiled function.
    buffers : sequence of numpy.ndarray
        Writable C-contiguous arrays bound at buffer indices 0, 1, ...; they are
        wrapped in place, so the kernel's writes land in them.
    grid : int or sequence of int
        Thread counts in one to three dimensions.
    threadgroup : int, sequence of int, or None, optional
        Threads per group; None lets the runtime choose.
    scalars : sequence of numpy scalars or arrays, optional
        Values bound by value after the buffers.

    Returns
    -------
    float
        Device-side seconds of the dispatch.

    Raises
    ------
    MetalError
        Wrapping, validation or execution failed.
    """
    if not all(a.flags.c_contiguous and a.flags.writeable for a in buffers):
        raise ValueError("run: buffers must be writable C-contiguous arrays")
    values = [np.ascontiguousarray(value) for value in scalars]
    grid3 = (*np.atleast_1d(grid).tolist(), 1, 1)[:3]
    group3 = (0, 1, 1) if threadgroup is None else (*np.atleast_1d(threadgroup).tolist(), 1, 1)[:3]
    lib, err = library(), ctypes.c_char_p()
    handles: list[ctypes.c_void_p] = []
    batch = ctypes.c_void_p()
    try:
        for array in buffers:
            handles.append(ctypes.c_void_p())
            status = lib.mr_wrap_buffer(
                array.ctypes.data, array.nbytes, ctypes.byref(handles[-1]), ctypes.byref(err)
            )
            _check(status, err)
        desc = LaunchDesc(
            pipeline=kernel._pipeline,
            buffers=(ctypes.c_void_p * len(handles))(*(h.value for h in handles)),
            buffer_offsets=(ctypes.c_size_t * len(handles))(),
            buffer_count=len(handles),
            scalars=(ctypes.c_void_p * len(values))(*(v.ctypes.data for v in values)),
            scalar_sizes=(ctypes.c_size_t * len(values))(*(v.nbytes for v in values)),
            scalar_count=len(values),
            grid_x=grid3[0],
            grid_y=grid3[1],
            grid_z=grid3[2],
            threadgroup_x=group3[0],
            threadgroup_y=group3[1],
            threadgroup_z=group3[2],
        )
        _check(lib.mr_batch_create(ctypes.byref(batch), ctypes.byref(err)), err)
        _check(lib.mr_batch_add(batch, ctypes.byref(desc), ctypes.byref(err)), err)
        _check(lib.mr_batch_wait(batch, ctypes.byref(err)), err)
        seconds = ctypes.c_double()
        _check(lib.mr_batch_gpu_time(batch, ctypes.byref(seconds), ctypes.byref(err)), err)
        for handle in handles:  # a no-op unless the wrap had to copy
            _check(lib.mr_buffer_flush_to(handle, ctypes.byref(err)), err)
        return seconds.value
    finally:
        if batch:
            lib.mr_release_batch(batch)
        for handle in handles:
            if handle:
                lib.mr_release_buffer(handle)
