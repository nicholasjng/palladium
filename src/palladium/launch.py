"""Launch options and geometry shared by `metal_call` and `pallas_call` on mps."""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import Any, Literal

from jax._src.pallas import core as pallas_core
from metal_runtime import MathMode

from palladium.emit.core import DOT_GENERAL_POLICIES
from palladium.errors import EmitError
from palladium.trace import KernelSpec

__all__ = [
    "CompilerParams",
    "check_threadgroup",
    "device_limits",
    "launch_geometry",
    "normalize_threadgroup",
    "simdgroup_width",
]


@dataclasses.dataclass(frozen=True)
class CompilerParams(pallas_core.CompilerParams):
    """Metal-side options for a ``pl.pallas_call`` lowered by Palladium.

    Attributes
    ----------
    dot_general : str
        "auto" (TensorOps for tiled matmuls and attention, the default),
        "tensorops" to require it, or "default" for the primitive path.
    threadgroup : int, tuple, or None
        Explicit threadgroup size; required by cooperative kernels.
    math_mode : metal_runtime.MathMode
        FAST (the default), RELAXED, or SAFE; SAFE keeps IEEE ordering, as
        compensated arithmetic needs. The mps path supports FAST only.
    """

    BACKEND: str = "palladium"
    dot_general: str = "auto"
    threadgroup: int | tuple[int, ...] | str | None = None
    math_mode: MathMode = MathMode.FAST

    def __post_init__(self) -> None:
        if self.dot_general not in DOT_GENERAL_POLICIES:
            raise ValueError("dot_general must be 'auto', 'default', or 'tensorops'")
        object.__setattr__(self, "threadgroup", normalize_threadgroup(self.threadgroup))


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


def normalize_threadgroup(
    threadgroup: int | tuple[int, ...] | Literal["simdgroup", "threadgroup"] | None,
) -> tuple[int, ...] | None:
    """Normalize a `threadgroup=` argument to a tuple of ints, shared by every
    entry point. None means the runtime chooses; `"simdgroup"` resolves to
    the device's SIMD width (a reduction over one SIMD group pays no
    cross-simdgroup latency)."""
    if threadgroup is None:
        return None
    if threadgroup == "simdgroup":
        return (simdgroup_width(),)
    if isinstance(threadgroup, str):
        raise ValueError("unknown threadgroup policy; expected 'simdgroup'")  # noqa: TRY004
    try:
        values = (threadgroup,) if isinstance(threadgroup, int) else tuple(threadgroup)
        result = tuple(operator.index(t) for t in values)
    except TypeError as exc:
        raise ValueError("threadgroup dimensions must be integers") from exc
    if not 1 <= len(result) <= 3 or any(t <= 0 for t in result):
        raise ValueError("threadgroup must have 1 to 3 positive dimensions")
    return result


def simdgroup_width() -> int:
    """Threads per SIMD group. 32 on every Apple GPU family to date, and
    the fallback when no device is present."""
    return int(device_limits().get("simdgroup_width", 32))


def check_threadgroup(spec: KernelSpec, threadgroup: tuple[int, ...] | None) -> None:
    """Validate launch geometry: cooperative requirements and lane-indexed
    scratch bounds always, device resource limits when a device is present.
    Explicit group sizes are checked for independent kernels too."""
    if spec.uses_threadgroup and threadgroup is None:
        raise EmitError(
            f"kernel {spec.name!r} uses cooperative instructions or shared scratch; "
            "pass an explicit threadgroup= size"
        )
    limits = device_limits()
    shared_bytes = sum(
        math.prod(s.shape) * s.dtype.itemsize for s in spec.scratch if s.space == "threadgroup"
    )
    budget = limits.get("max_threadgroup_memory_length")
    if budget and shared_bytes > budget:
        raise EmitError(
            f"kernel {spec.name!r} declares {shared_bytes} bytes of "
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

    if threadgroup:
        actual = math.prod(
            min(g, t)
            for g, t in zip(
                spec.grid + (1,) * (3 - len(spec.grid)), threadgroup + (1,) * (3 - len(threadgroup))
            )
        )
        if any(actual > extent for extent in spec.lane_scratch_extents):
            raise EmitError(
                f"threadgroup has up to {actual} lanes but a directly lane-indexed "
                f"scratch dimension is smaller: {spec.lane_scratch_extents}"
            )
    if spec.uses_threadgroup and not limits.get("supports_non_uniform_threadgroups", True):
        raise EmitError(  # pragma: no cover - all Apple silicon supports this
            f"kernel {spec.name!r} uses threads_per_threadgroup() to bound a "
            "cooperative loop, which is only correct when the device "
            "dispatches non-uniform threadgroups; this device reports it does "
            "not, so the final partial group would read unwritten slots."
        )


def _pad3(dims: tuple[int, ...]) -> tuple[int, int, int]:
    padded = tuple(int(d) for d in dims) + (1, 1, 1)
    return padded[0], padded[1], padded[2]


def launch_geometry(
    spec: KernelSpec, msl_source: str, threadgroup: tuple[int, ...] | None
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """The Metal (grid, threadgroup) in threads for a kernel's emitted MSL.

    A cooperative TensorOps kernel runs one threadgroup per program, so its
    grid scales by the required threadgroup, and any explicit threadgroup
    must match it. Otherwise the grid is the Pallas grid and a None
    threadgroup lets the runtime choose.
    """
    from palladium.emit.tensorops import cooperative_launch, emits_cooperative

    grid = _pad3(spec.grid)
    group = None if threadgroup is None else _pad3(threadgroup)
    if emits_cooperative(msl_source):
        required, grid = cooperative_launch(spec.grid, simdgroup_width())
        if group is not None and group != required:
            raise ValueError(f"cooperative kernel requires threadgroup={required}, got {group}")
        group = required
    return grid, group
