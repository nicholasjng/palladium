"""Kernel diagnostics: which execution model a kernel gets, and why.

The cooperative model is a large performance cliff (dot-heavy kernels
run several times slower on the classic model), and the decision is
silent by default. `MetalCallable.explain` / `FfiCallable.explain`
return a KernelDiagnostics; setting PALLADIUM_EXPLAIN=1 prints one
stderr line per newly compiled kernel.
"""

from __future__ import annotations

import dataclasses
import os
import sys

from palladium.emit import emit_msl
from palladium.emit.coop import coop_probe
from palladium.emit.core import SIMDGROUP_WIDTH
from palladium.emit.gemm import gemm_groups
from palladium.trace import KernelSpec

__all__ = ["KernelDiagnostics", "explain_spec"]


@dataclasses.dataclass(frozen=True)
class KernelDiagnostics:
    """How one traced kernel will execute.

    Attributes
    ----------
    name : str
        MSL function name (`spec.name`).
    model : str
        `"cooperative"` (one 32-thread SIMD-group per program instance)
        or `"classic"` (one thread per instance).
    rows : int or None
        Query rows R per instance in the cooperative model; None on the
        classic model.
    reason : str or None
        Why the cooperative model was rejected; None when it was taken.
    grid : tuple of int
        Threads dispatched per grid axis (already scaled by the
        SIMD-group width for cooperative kernels).
    threadgroup : tuple of int or None
        Fixed threadgroup size; None lets the runtime choose.
    msl_lines : int
        Line count of the emitted source.
    """

    name: str
    model: str
    rows: int | None
    reason: str | None
    grid: tuple[int, ...]
    threadgroup: tuple[int, ...] | None
    msl_lines: int

    def __str__(self) -> str:
        parts = [f"palladium kernel {self.name}: model={self.model}"]
        if self.rows is not None:
            parts.append(f"rows={self.rows}")
        parts.append(f"grid={self.grid}")
        if self.threadgroup is not None:
            parts.append(f"threadgroup={self.threadgroup}")
        parts.append(f"msl_lines={self.msl_lines}")
        if self.reason is not None:
            parts.append(f'reason="{self.reason}"')
        return " ".join(parts)


def explain_spec(
    spec: KernelSpec, threadgroup: int | tuple[int, ...] | None = None
) -> KernelDiagnostics:
    """Diagnostics for a traced spec: emits MSL, compiles nothing.

    `threadgroup` mirrors the `metal_call` keyword: an explicit size
    opts the kernel out of the cooperative model.
    """
    rows, reason = coop_probe(spec)
    cooperative = threadgroup is None and rows is not None
    if threadgroup is not None and rows is not None:
        rows, reason = None, "explicit threadgroup opts out of the cooperative model"
    msl = emit_msl(spec, cooperative=cooperative)
    grid = tuple(int(g) for g in spec.grid)
    if cooperative:
        groups = gemm_groups(spec)
        grid = (grid[0] * SIMDGROUP_WIDTH, *grid[1:])
        tg: tuple[int, ...] | None = (SIMDGROUP_WIDTH * groups, 1, 1)
    elif threadgroup is None:
        tg = None
    elif isinstance(threadgroup, int):
        tg = (threadgroup,)
    else:
        tg = tuple(int(t) for t in threadgroup)
    return KernelDiagnostics(
        name=spec.name,
        model="cooperative" if cooperative else "classic",
        rows=rows if cooperative else None,
        reason=None if cooperative else reason,
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
