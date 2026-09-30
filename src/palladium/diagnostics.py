"""Kernel diagnostics: launch geometry and MSL size for a traced kernel.

`FfiCallable.explain` returns a KernelDiagnostics; setting
PALLADIUM_EXPLAIN=1 prints one stderr line per newly compiled kernel.
"""

from __future__ import annotations

import dataclasses
import os
import sys

from palladium.device import device_limits
from palladium.emit import emit_msl_stats
from palladium.emit.tensorops import emits_cooperative
from palladium.launch import check_threadgroup, launch_geometry
from palladium.trace import KernelSpec

__all__ = ["KernelDiagnostics", "explain_spec", "log_compile"]


@dataclasses.dataclass(frozen=True)
class KernelDiagnostics:
    """How one traced kernel will execute.

    Attributes
    ----------
    name : str
        MSL function name (`spec.name`).
    grid : tuple of int
        Threads dispatched per Metal grid axis (x, y, z).
    threadgroup : tuple of int or None
        Fixed threadgroup size; None lets the runtime choose.
    msl_lines : int
        Line count of the emitted source.
    thread_bytes : int
        Per-instance `thread`-space storage; the figure to shrink (via
        the grid and BlockSpecs) when pipeline creation fails for stack
        space. Metal publishes no ceiling for it.
    threadgroup_bytes : int
        Per-group `threadgroup`-space storage, 0 unless the kernel lowers
        through TensorOps.
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
    cooperative: bool = False

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


def explain_spec(
    spec: KernelSpec,
    threadgroup: tuple[int, ...] | None = None,
    *,
    dot_general: str = "auto",
) -> KernelDiagnostics:
    """Diagnostics for a traced spec: emits MSL, compiles nothing."""
    msl, stats = emit_msl_stats(spec, dot_general=dot_general)
    grid, group = launch_geometry(spec, msl, threadgroup)
    check_threadgroup(spec, group)
    return KernelDiagnostics(
        name=spec.name,
        grid=grid,
        threadgroup=group,
        msl_lines=len(msl.splitlines()),
        thread_bytes=stats.thread_bytes,
        threadgroup_bytes=stats.threadgroup_bytes,
        threadgroup_limit=device_limits().get("max_threadgroup_memory_length"),
        cooperative=emits_cooperative(msl),
    )


def _explain_enabled() -> bool:
    return os.environ.get("PALLADIUM_EXPLAIN", "") not in ("", "0")


def log_compile(
    spec: KernelSpec,
    threadgroup: tuple[int, ...] | None = None,
    dot_general: str = "auto",
) -> None:
    """One stderr line per compiled kernel when PALLADIUM_EXPLAIN is set;
    called on the cache-miss path only."""
    if _explain_enabled():
        print(explain_spec(spec, threadgroup, dot_general=dot_general), file=sys.stderr)
