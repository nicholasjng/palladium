"""Queries of the Metal device palladium compiles for."""

from __future__ import annotations

from typing import Any

__all__ = ["device_limits", "simdgroup_width"]


def device_limits() -> dict[str, Any]:
    """Query the Metal device capabilities used for launch validation.

    Returns
    -------
    dict of str to object
        Copy of ``metal_runtime.device_info()``, or an empty dictionary if the
        dependency or device is unavailable.

    Notes
    -----
    Missing keys mean unknown capabilities. Callers must not interpret an absent limit as an
    unlimited resource budget.
    """
    try:
        import metal_runtime as mr
    except ImportError:  # pragma: no cover - metal_runtime is a hard dep
        return {}
    try:
        return dict(mr.device_info())
    except mr.DeviceError:  # pragma: no cover - no Metal device
        return {}


def simdgroup_width() -> int:
    """Return the device SIMD-group width used to size cooperative launches.

    Returns
    -------
    int
        Reported SIMD-group width, falling back to 32 when the device or field is
        unavailable.
    """
    return int(device_limits().get("simdgroup_width", 32))
