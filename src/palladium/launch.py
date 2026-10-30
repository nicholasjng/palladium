"""Launch options and geometry shared by `metal_call` and `pallas_call` on mps."""

from __future__ import annotations

import dataclasses
import enum
import math
import operator

from jax._src.pallas import core as pallas_core

from palladium.device import device_limits
from palladium.emit.kernel import DOT_GENERAL_POLICIES, PROGRAM_SCOPES
from palladium.emit.tensorops import declared_threadgroup_width
from palladium.errors import EmitError
from palladium.trace import KernelSpec

__all__ = [
    "CompilerParams",
    "MathMode",
    "check_threadgroup",
    "launch_geometry",
    "normalize_threadgroup",
]


class MathMode(enum.StrEnum):
    """How much freedom the Metal compiler has to rewrite floating-point arithmetic."""

    #: IEEE semantics; compensated arithmetic needs it.
    SAFE = "safe"
    #: Reassociates like FAST, but keeps infinities and NaNs well-defined.
    RELAXED = "relaxed"
    #: Metal's default: permits reassociation and flushes denormals.
    FAST = "fast"


@dataclasses.dataclass(frozen=True)
class CompilerParams(pallas_core.CompilerParams):
    """Configure Metal emission and launch options for either Palladium entry point.

    Attributes
    ----------
    dot_general : {"auto", "default", "tensorops", "simdgroup"}, optional
        Default "auto" attempts cooperative lowering for dots on a 2D or 3D grid and
        falls back if recognition fails. "default" forces primitive emission.
        "tensorops" requires cooperative lowering for kernels containing dots.
        "simdgroup" also requires it, and emits float32 tiled matmuls with
        `simdgroup_matrix` instead of Metal 4 `matmul2d` where the tile shape allows;
        that measured about 20% faster on an M2.
    program : {"thread", "threadgroup"}, optional
        "thread" (default) runs each Pallas program as one Metal thread, or
        one threadgroup when a dot lowers cooperatively. "threadgroup" runs
        each program as one threadgroup of rows (`emit/threadgroup.py`);
        kernels it cannot lower raise EmitError.
    threadgroup : int, tuple of int, or None, optional
        One to three positive dimensions, or None for runtime selection. Cooperative
        kernels fix their own size and reject mismatches.
    math_mode : MathMode, optional
        FAST by default; RELAXED and SAFE are available through CPU FFI. The mps path
        accepts FAST only.
    BACKEND : str
        Pallas backend identifier, "palladium"; leave this at its default.

    Raises
    ------
    ValueError
        The dot policy is unknown or threadgroup dimensions are not one to three
        positive integers.

    Notes
    -----
    Device thread limits and cooperative group compatibility are checked later. FAST permits
    floating-point reassociation and approximations; compensated arithmetic requires SAFE.
    Requesting "tensorops" on a kernel without a dot still uses primitive emission.
    """

    BACKEND: str = "palladium"
    dot_general: str = "auto"
    program: str = "thread"
    threadgroup: int | tuple[int, ...] | None = None
    math_mode: MathMode = MathMode.FAST

    def __post_init__(self) -> None:
        if self.dot_general not in DOT_GENERAL_POLICIES:
            raise ValueError("dot_general must be 'auto', 'default', 'tensorops', or 'simdgroup'")
        if self.program not in PROGRAM_SCOPES:
            raise ValueError("program must be 'thread' or 'threadgroup'")
        normalize_threadgroup(self.threadgroup)

    @property
    def threadgroup_dims(self) -> tuple[int, ...] | None:
        """Return the normalized threadgroup dimensions without changing the stored option.

        Returns
        -------
        tuple of int or None
            An integer option becomes a one-element tuple; None leaves launch selection to
            the runtime.
        """
        return normalize_threadgroup(self.threadgroup)


def normalize_threadgroup(threadgroup: int | tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Normalize an explicit Metal threadgroup size and validate its dimensions.

    Parameters
    ----------
    threadgroup : int, tuple of int, or None
        Integer or sequence of one to three positive integer dimensions. None requests
        automatic sizing.

    Returns
    -------
    tuple of int or None
        Normalized dimensions, preserving None.

    Raises
    ------
    ValueError
        Dimensions are nonintegral, nonpositive, empty, or more than three in number.

    Notes
    -----
    This checks representation only; device limits and cooperative kernel requirements are
    checked separately.
    """
    if threadgroup is None:
        return None
    try:
        values = (threadgroup,) if isinstance(threadgroup, int) else tuple(threadgroup)
        result = tuple(operator.index(t) for t in values)
    except TypeError as exc:
        raise ValueError("threadgroup dimensions must be integers") from exc
    if not 1 <= len(result) <= 3 or any(t <= 0 for t in result):
        raise ValueError("threadgroup must have 1 to 3 positive dimensions")
    return result


def check_threadgroup(spec: KernelSpec, threadgroup: tuple[int, ...] | None) -> None:
    """Validate an explicit threadgroup against the available device thread limit.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    threadgroup : tuple of int or None, optional
        One to three positive dimensions, or None for runtime selection. Cooperative
        kernels require their fixed group size.

    Raises
    ------
    EmitError
        The product of dimensions exceeds ``max_threads_per_threadgroup``.

    Notes
    -----
    None or an unavailable device limit skips the check. The specification is accepted for a
    shared validation interface; this check does not inspect its body or validate shared-
    memory use.
    """
    max_threads = device_limits().get("max_threads_per_threadgroup")
    if threadgroup and max_threads and math.prod(threadgroup) > max_threads:
        raise EmitError(
            f"threadgroup={threadgroup} is {math.prod(threadgroup)} threads, over this "
            f"device's max_threads_per_threadgroup of {max_threads}"
        )


def _pad3(dims: tuple[int, ...]) -> tuple[int, int, int]:
    """Pad dimensions with ones to rank three; callers must reject higher-rank grids first."""
    padded = tuple(int(d) for d in dims) + (1, 1, 1)
    return padded[0], padded[1], padded[2]


def launch_geometry(
    spec: KernelSpec, msl_source: str, threadgroup: tuple[int, ...] | None
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """Translate a program grid into Metal thread and threadgroup dimensions.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    msl_source : str
        Emitted source; a declared threadgroup width marks cooperative execution.
    threadgroup : tuple of int or None, optional
        One to three positive dimensions, or None for runtime selection. Cooperative
        kernels require their fixed group size.

    Returns
    -------
    grid : tuple of int
        Three-dimensional thread grid, padded with ones. Cooperative kernels scale x
        by their declared group width.
    threadgroup : tuple of int or None
        Three-dimensional group size; None lets the runtime choose for primitive
        kernels.

    Raises
    ------
    ValueError
        An explicit group differs from the size required by a cooperative kernel.

    Notes
    -----
    Geometry follows the emitted source after any TensorOps fallback; callers must
    supply a grid of at most three dimensions.
    """
    grid = _pad3(spec.grid)
    group = None if threadgroup is None else _pad3(threadgroup)
    width = declared_threadgroup_width(msl_source)
    if width is not None:
        required = (width, 1, 1)
        if group is not None and group != required:
            raise ValueError(f"cooperative kernel requires threadgroup={required}, got {group}")
        group = required
        grid = (grid[0] * width, grid[1], grid[2])
    return grid, group
