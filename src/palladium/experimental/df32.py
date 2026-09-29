"""float64 emulated as float32 pairs (df32) for hand-written Metal kernels.

`PRELUDE` is the MSL library (two_sum, two_prod, the df32 struct and its
arithmetic); `kernel` compiles a source using it with math mode pinned to
SAFE, which the prelude requires; `split` and `join` convert between
float64 arrays and (hi, lo) float32 pairs on the host. Pallas has no df32
dtype, so these kernels are written in MSL directly.
"""

import importlib.resources

import metal_runtime as mr
import numpy as np
from numpy.typing import NDArray

PRELUDE = (
    importlib.resources.files("palladium.experimental")
    .joinpath("df32.metal")
    .read_text(encoding="utf-8")
)

FRAGMENT = mr.Fragment("df32", PRELUDE)


def kernel(source: str, function_name: str, **kwargs) -> mr.Kernel:
    """Build an `mr.Kernel` from `source` with `PRELUDE` prepended and
    `math_mode` defaulting to SAFE; FAST reassociates away the compensation
    terms the prelude depends on.

    Parameters
    ----------
    source: str
        Kernel source, assembled after `PRELUDE`.
    function_name: str
        Kernel function to dispatch.
    **kwargs
        Forwarded to `mr.Kernel`. An explicit `math_mode` other than SAFE
        fails to compile.
    """
    kwargs.setdefault("math_mode", mr.MathMode.SAFE)
    assembled = mr.assemble(FRAGMENT, mr.Fragment(function_name, source))
    return mr.Kernel(assembled, function_name, **kwargs)


def split(x: NDArray[np.float64]) -> NDArray[np.float32]:
    """Split a float64 array into (hi, lo) float32 pairs, shape `(*x.shape, 2)`.

    `hi` is `x` rounded to float32, `lo` the residual `x - hi` computed in
    float64. The residual carries up to 53 - 24 = 29 significant bits and
    `lo` holds 24, so the split loses 5 bits against float64. Raises
    ValueError for finite values outside float32 range.
    """
    f32max = np.finfo(np.float32).max
    finite = np.isfinite(x)
    if np.any(finite & (np.abs(x) > f32max)):
        raise ValueError("input array contains values out of range for np.float32")
    hi = x.astype(np.float32)
    hi64 = hi.astype(np.float64)
    lo = np.subtract(x, hi64, out=np.zeros_like(x), where=finite)
    lo = np.where(lo == 0, np.copysign(lo, x), lo)
    return np.stack([hi, lo], axis=-1, dtype=np.float32)


def join(pairs: NDArray[np.float32]) -> NDArray[np.float64]:
    """Join `(hi, lo)` float32 pairs (last dimension 2) into a float64 array
    of shape `pairs.shape[:-1]`."""
    if pairs.shape[-1] != 2:
        raise ValueError("join(): input must be an array of shape (..., 2)")
    return pairs[..., 0].astype(np.float64) + pairs[..., 1].astype(np.float64)
