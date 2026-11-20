"""Queries of the Metal device palladium compiles for."""

from __future__ import annotations

import ctypes
import functools
from typing import Any

from palladium import _runtime

__all__ = ["device_limits", "prefers_simdgroup_matrix"]


def device_limits() -> dict[str, Any]:
    """Query the Metal device capabilities used for launch validation.

    Returns
    -------
    dict of str to object
        ``name``, ``unified_memory``, ``recommended_max_working_set_size``,
        ``max_threads_per_threadgroup``, ``max_threadgroup_memory_length``,
        ``max_buffer_length``, ``supports_non_uniform_threadgroups`` and
        ``apple_gpu_family`` (the highest N with MTLGPUFamilyAppleN support), or
        an empty dictionary if the native library or the device is unavailable.

    Notes
    -----
    Missing keys mean unknown capabilities. Callers must not interpret an absent limit as an
    unlimited resource budget.
    """
    return dict(_query_device())


@functools.cache
def _query_device() -> dict[str, Any]:
    try:
        library = _runtime.library()
    except (FileNotFoundError, OSError):  # pragma: no cover - native library not built
        return {}
    info = _runtime.DeviceInfo()
    err = ctypes.c_char_p()
    if library.mr_device_info(ctypes.byref(info), ctypes.byref(err)) != _runtime.OK:
        library.mr_free_error_message(err)  # pragma: no cover - no Metal device
        return {}
    return {
        "name": info.name.decode(),
        "unified_memory": bool(info.has_unified_memory),
        "recommended_max_working_set_size": info.recommended_max_working_set_size,
        "max_threads_per_threadgroup": info.max_threads_per_threadgroup,
        "max_threadgroup_memory_length": info.max_threadgroup_memory_length,
        "max_buffer_length": info.max_buffer_length,
        "supports_non_uniform_threadgroups": bool(info.supports_non_uniform_threadgroups),
        "apple_gpu_family": info.apple_gpu_family,
    }


def prefers_simdgroup_matrix() -> bool:
    """Whether products should use `simdgroup_matrix` rather than `matmul2d`.

    True on Apple GPU families 7 to 9, where the `simdgroup_matrix` kernel is at
    least as fast for every supported float type. Family 10 GPUs have matrix
    accelerators that Metal 4's `matmul2d` uses, so they keep it.
    """
    return device_limits().get("apple_gpu_family") in (7, 8, 9)
