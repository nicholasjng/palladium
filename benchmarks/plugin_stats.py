"""The metal plugin's counters per benchmarked call; nothing on other backends."""

from __future__ import annotations

import os

import mew


def on_metal() -> bool:
    """Whether this process's default JAX device is the metal plugin."""
    if os.environ.get("BENCH_BACKEND") == "mlx":
        return False
    try:
        import jax
    except ImportError:
        return False
    return jax.devices()[0].platform == "metal"


def reset() -> None:
    """Zero the counters before the timed calls."""
    if on_metal():
        from palladium import _pjrt

        _pjrt.stats(reset=True)


def report(state: mew.State, calls: int, wall: float | None = None) -> None:
    """Set the counters of `calls` finished calls on `state`.

    Per call: ``gpu_us`` (GPU busy time), ``launches``, ``encode_us`` and
    ``throttle_us`` (host time in Execute, encoding and waiting for the
    in-flight bound) and ``pool_misses`` (new MTLBuffers). With `wall`, the
    seconds the calls took, also ``gpu_fraction``, GPU busy time over wall
    time. Wall time well above ``gpu_us`` is host or allocation cost.
    """
    if not calls or not on_metal():
        return
    from palladium import _pjrt

    s = _pjrt.stats(reset=True)
    state.set_counter("gpu_us", s["gpu_busy_seconds"] / calls * 1e6)
    state.set_counter("launches", s["launches"] / calls)
    state.set_counter("encode_us", s["encode_seconds"] / calls * 1e6)
    state.set_counter("throttle_us", s["throttle_seconds"] / calls * 1e6)
    state.set_counter("pool_misses", s["pool_misses"] / calls)
    if wall:
        state.set_counter("gpu_fraction", s["gpu_busy_seconds"] / wall)
