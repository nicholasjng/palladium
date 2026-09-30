"""Queries of the Metal device palladium compiles for."""

from __future__ import annotations

from typing import Any

__all__ = ["device_limits", "simdgroup_width"]


def device_limits() -> dict[str, Any]:
    """`metal_runtime.device_info()`, or an empty dict with no device.
    Callers treat a missing device as unknown limits, not an error."""
    try:
        import metal_runtime as mr
    except ImportError:  # pragma: no cover - metal_runtime is a hard dep
        return {}
    try:
        return dict(mr.device_info())
    except mr.DeviceError:  # pragma: no cover - no Metal device
        return {}


def simdgroup_width() -> int:
    """Threads per SIMD group. 32 on every Apple GPU family to date, and
    the fallback when no device is present."""
    return int(device_limits().get("simdgroup_width", 32))
