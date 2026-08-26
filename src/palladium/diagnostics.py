"""Kernel diagnostics: launch geometry and MSL size for a traced kernel.

`MetalCallable.explain` / `FfiCallable.explain` return a
KernelDiagnostics; setting PALLADIUM_EXPLAIN=1 prints one stderr line
per newly compiled kernel.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

from palladium.emit import emit_msl_stats
from palladium.errors import EmitError
from palladium.trace import KernelSpec

__all__ = [
    "KernelDiagnostics",
    "check_threadgroup",
    "device_limits",
    "explain_spec",
    "normalize_threadgroup",
    "simdgroup_width",
]


@dataclasses.dataclass(frozen=True)
class KernelDiagnostics:
    """How one traced kernel will execute.

    Attributes
    ----------
    name : str
        MSL function name (`spec.name`).
    grid : tuple of int
        Threads dispatched per grid axis.
    threadgroup : tuple of int or None
        Fixed threadgroup size; None lets the runtime choose.
    msl_lines : int
        Line count of the emitted source.
    thread_bytes : int
        Per-instance `thread`-space storage; the figure to shrink (via
        the grid and BlockSpecs) when pipeline creation fails for stack
        space. Metal publishes no ceiling for it.
    threadgroup_bytes : int
        Per-group `threadgroup`-space storage, 0 unless the kernel uses
        `palladium.threadgroup_memory`.
    threadgroup_limit : int or None
        The device's `max_threadgroup_memory_length`, when a device is
        present. `threadgroup_bytes` must fit under it.
    """

    name: str
    grid: tuple[int, ...]
    threadgroup: tuple[int, ...] | None
    msl_lines: int
    thread_bytes: int = 0
    threadgroup_bytes: int = 0
    threadgroup_limit: int | None = None

    def __str__(self) -> str:
        parts = [f"palladium kernel {self.name}: grid={self.grid}"]
        if self.threadgroup is not None:
            parts.append(f"threadgroup={self.threadgroup}")
        parts.append(f"stack~{_human(self.thread_bytes)}/thread")
        if self.threadgroup_bytes:
            shared = f"shared={_human(self.threadgroup_bytes)}"
            if self.threadgroup_limit:
                shared += f"/{_human(self.threadgroup_limit)}"
            parts.append(shared)
        parts.append(f"msl_lines={self.msl_lines}")
        return " ".join(parts)


def _human(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes}B"
    return f"{nbytes / 1024:.1f}KB"


def device_limits() -> dict[str, Any]:
    """`metal_runtime.device_info()`, or an empty dict with no device.

    Diagnostics work without a GPU, as the emitter and tracer do, so
    consumers treat a missing device as unknown limits, not an error.
    """
    try:
        import metal_runtime as mr
    except ImportError:  # pragma: no cover - metal_runtime is a hard dep
        return {}
    try:
        return dict(mr.device_info())
    except mr.DeviceError:  # pragma: no cover - no Metal device
        return {}


def normalize_threadgroup(
    threadgroup: int | tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """The one place `threadgroup=` becomes a tuple of ints.

    The knob is `int | tuple[int, ...] | None` at every entry point
    (`metal_call`, `metal_call_jit`, `bind`, `explain`); None means "let
    the runtime choose". Shared so the eager and jax.ffi paths cannot
    normalize it differently.

    Also accepts `"simdgroup"`, resolving to the device's SIMD width: a
    reduction over exactly one SIMD group is the common case and pays no
    cross-simdgroup latency.
    """
    if threadgroup is None:
        return None
    if threadgroup == "simdgroup":
        return (simdgroup_width(),)
    if isinstance(threadgroup, int):
        return (int(threadgroup),)
    return tuple(int(t) for t in threadgroup)


def simdgroup_width() -> int:
    """Threads per SIMD group. 32 on every Apple GPU family to date, and
    the fallback when no device is present."""
    return int(device_limits().get("simdgroup_width", 32))


def check_threadgroup(spec: KernelSpec, threadgroup: tuple[int, ...] | None) -> None:
    """Validate a cooperative kernel's launch geometry against the device.

    Checks threadgroup-space storage against the device budget (Metal
    otherwise rejects the pipeline with a vaguer message) and the group
    size against the device maximum. Kernels with no threadgroup storage
    are unaffected.
    """
    if not spec.uses_threadgroup:
        return
    limits = device_limits()
    if not limits:  # pragma: no cover - no device to validate against
        return

    _, stats = emit_msl_stats(spec)
    budget = limits.get("max_threadgroup_memory_length")
    if budget and stats.threadgroup_bytes > budget:
        raise EmitError(
            f"kernel {spec.name!r} declares {stats.threadgroup_bytes} bytes of "
            f"threadgroup_memory, over this device's "
            f"max_threadgroup_memory_length of {budget}. Shrink the "
            "threadgroup_memory request, or split the reduction across "
            "more, smaller threadgroups."
        )

    max_threads = limits.get("max_threads_per_threadgroup")
    if threadgroup and max_threads:
        total = 1
        for t in threadgroup:
            total *= t
        if total > max_threads:
            raise EmitError(
                f"threadgroup={threadgroup} is {total} threads, over this "
                f"device's max_threads_per_threadgroup of {max_threads}"
            )

    if not limits.get("supports_non_uniform_threadgroups", True):
        raise EmitError(  # pragma: no cover - all Apple silicon supports this
            f"kernel {spec.name!r} uses threads_per_threadgroup() to bound a "
            "cooperative loop, which is only correct when the device "
            "dispatches non-uniform threadgroups; this device reports it does "
            "not, so the final partial group would read unwritten slots."
        )


def explain_spec(
    spec: KernelSpec, threadgroup: int | tuple[int, ...] | None = None
) -> KernelDiagnostics:
    """Diagnostics for a traced spec: emits MSL, compiles nothing."""
    msl, stats = emit_msl_stats(spec)
    grid = tuple(int(g) for g in spec.grid)
    tg = normalize_threadgroup(threadgroup)
    limits = device_limits()
    return KernelDiagnostics(
        name=spec.name,
        grid=grid,
        threadgroup=tg,
        msl_lines=len(msl.splitlines()),
        thread_bytes=stats.thread_bytes,
        threadgroup_bytes=stats.threadgroup_bytes,
        threadgroup_limit=limits.get("max_threadgroup_memory_length"),
    )


def _explain_enabled() -> bool:
    return os.environ.get("PALLADIUM_EXPLAIN", "") not in ("", "0")


def log_compile(
    spec: KernelSpec, threadgroup: int | tuple[int, ...] | None = None
) -> None:
    """One stderr line per compiled kernel when PALLADIUM_EXPLAIN is set.

    Called on the cache-miss path, so cached shapes stay silent.
    """
    if _explain_enabled():
        print(explain_spec(spec, threadgroup), file=sys.stderr)
