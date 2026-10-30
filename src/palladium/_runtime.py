"""ctypes declarations of the Metal runtime's C API (`native/runtime/c_api.h`).

`libpalladium_ffi` links the runtime statically and exports its `mr_*`
functions: the jax.ffi handler calls them from C++, and Python reaches them
here. `PALLADIUM_FFI_LIBRARY` overrides the library path for an out-of-tree
build.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.resources
import os
from pathlib import Path

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
