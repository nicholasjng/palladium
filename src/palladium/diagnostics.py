"""Kernel diagnostics: launch geometry and MSL size for a traced kernel.

`MetalCallable.explain` / `FfiCallable.explain` return a
KernelDiagnostics; setting PALLADIUM_EXPLAIN=1 prints one stderr line
per newly compiled kernel.
"""

from __future__ import annotations

import dataclasses
import os
import sys

from palladium.emit import emit_msl
from palladium.trace import KernelSpec

__all__ = ["KernelDiagnostics", "explain_spec"]


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
    """

    name: str
    grid: tuple[int, ...]
    threadgroup: tuple[int, ...] | None
    msl_lines: int

    def __str__(self) -> str:
        parts = [f"palladium kernel {self.name}: grid={self.grid}"]
        if self.threadgroup is not None:
            parts.append(f"threadgroup={self.threadgroup}")
        parts.append(f"msl_lines={self.msl_lines}")
        return " ".join(parts)


def explain_spec(
    spec: KernelSpec, threadgroup: int | tuple[int, ...] | None = None
) -> KernelDiagnostics:
    """Diagnostics for a traced spec: emits MSL, compiles nothing."""
    msl = emit_msl(spec)
    grid = tuple(int(g) for g in spec.grid)
    if threadgroup is None:
        tg = None
    elif isinstance(threadgroup, int):
        tg = (threadgroup,)
    else:
        tg = tuple(int(t) for t in threadgroup)
    return KernelDiagnostics(
        name=spec.name,
        grid=grid,
        threadgroup=tg,
        msl_lines=len(msl.splitlines()),
    )


def _explain_enabled() -> bool:
    return os.environ.get("PALLADIUM_EXPLAIN", "") not in ("", "0")


def log_compile(
    spec: KernelSpec, threadgroup: int | tuple[int, ...] | None = None
) -> None:
    """One stderr line per compiled kernel when PALLADIUM_EXPLAIN is set.

    Called on the cache-miss path of MetalCallable and FfiCallable, so
    repeated calls on cached shapes stay silent.
    """
    if _explain_enabled():
        print(explain_spec(spec, threadgroup), file=sys.stderr)
