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
from palladium.emit.tensorops import declared_threadgroup_width
from palladium.launch import check_threadgroup, launch_geometry
from palladium.trace import KernelSpec

__all__ = ["KernelDiagnostics", "explain_spec", "log_compile"]


@dataclasses.dataclass(frozen=True)
class KernelDiagnostics:
    """Describe the launch and declared storage of an emitted kernel.

    Attributes
    ----------
    name : str
        MSL function name.
    grid : tuple of int
        Metal thread grid padded to x, y, z; cooperative grids include group scaling.
    threadgroup : tuple of int or None
        Fixed group dimensions, or None for runtime selection.
    msl_lines : int
        Line count of the complete emitted source.
    thread_bytes : int, optional
        Declared thread-local bytes per program, default zero. Metal does not publish a
        corresponding stack ceiling.
    threadgroup_bytes : int, optional
        Declared shared bytes per cooperative group, default zero.
    threadgroup_limit : int or None, optional
        Device shared-memory limit, or None when unavailable.
    cooperative : bool, optional
        Whether one threadgroup executes each program, default False.

    Notes
    -----
    Byte counts describe emitted declarations and are not liveness-aware. Compiler register
    allocation and temporary storage may differ; a missing device limit does not imply
    unlimited resources.
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
        """Format a compact launch and storage report for logs.

        Returns
        -------
        str
            Single-line report; unknown limits and automatic group dimensions are omitted.
        """
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
    """Format bytes as B or one-decimal KB; values below 1024 stay in bytes."""
    if nbytes < 1024:
        return f"{nbytes}B"
    return f"{nbytes / 1024:.1f}KB"


def explain_spec(
    spec: KernelSpec,
    threadgroup: tuple[int, ...] | None = None,
    *,
    dot_general: str = "auto",
    program: str = "thread",
) -> KernelDiagnostics:
    """Emit a traced kernel and report its geometry and declared storage.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    threadgroup : tuple of int or None, optional
        One to three positive dimensions, or None for runtime selection. Cooperative
        kernels require their fixed group size.
    dot_general : {"auto", "default", "tensorops", "simdgroup"}, optional
        Use cooperative TensorOps when recognized, force primitive emission, require
        TensorOps, or require them with `simdgroup_matrix` float32 matmuls,
        respectively. The default is "auto".

    Returns
    -------
    KernelDiagnostics
        Emission statistics and available device limits; no compilation or dispatch
        occurs.

    Raises
    ------
    EmitError
        The kernel uses unsupported shapes, addressing, effects, or primitive
        parameters.
    UnsupportedPrimitiveError
        A staged primitive has no registered lowering rule.
    ValueError
        The policy is unknown or an explicit cooperative threadgroup does not match.

    Notes
    -----
    An unavailable Metal device leaves the shared-memory limit unknown. Resource counts
    describe declarations rather than compiler-measured stack or register use.
    """
    msl, stats = emit_msl_stats(spec, dot_general=dot_general, program=program)
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
        cooperative=declared_threadgroup_width(msl) is not None,
    )


def log_compile(
    spec: KernelSpec,
    threadgroup: tuple[int, ...] | None = None,
    dot_general: str = "auto",
    program: str = "thread",
) -> None:
    """Print kernel diagnostics when the explain environment flag is enabled.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    threadgroup : tuple of int or None, optional
        One to three positive dimensions, or None for runtime selection. Cooperative
        kernels require their fixed group size.
    dot_general : {"auto", "default", "tensorops", "simdgroup"}, optional
        Use cooperative TensorOps when recognized, force primitive emission, require
        TensorOps, or require them with `simdgroup_matrix` float32 matmuls,
        respectively. The default is "auto".

    Notes
    -----
    ``PALLADIUM_EXPLAIN`` enables logging for any value except the empty string and "0".
    This function emits source again when enabled; it does not deduplicate logs. The caller
    invokes it on cache misses to obtain one line per specialization.
    """
    if os.environ.get("PALLADIUM_EXPLAIN", "") not in ("", "0"):
        print(
            explain_spec(spec, threadgroup, dot_general=dot_general, program=program),
            file=sys.stderr,
        )
