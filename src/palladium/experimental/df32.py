"""float64 emulated as float32 pairs (df32) for hand-written Metal kernels.

`PRELUDE` is the MSL library (two_sum, two_prod, the df32 struct and its
arithmetic); `kernel` compiles a source that uses it with math mode pinned
to SAFE, which the prelude requires; `split` and `join` move between
NumPy float64 arrays and (hi, lo) float32 pairs on the host.

The Pallas frontend has no df32 dtype to trace through, so these kernels
are written in MSL directly; see examples/df32_precision.py and
examples/symplectic_longrun.py.
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
    """
    Build an `mr.Kernel` from `source` with `PRELUDE` prepended and
    `math_mode` pinned to `mr.MathMode.SAFE`.

    `mr.Kernel` defaults to `MathMode.FAST`, which silently reassociates away
    the compensation terms this prelude depends on.
    This helper exists so that's not something a caller has to remember.

    Parameters
    ----------
    source: str
        The kernel source to append to `PRELUDE`.
    function_name: str
        The name of the kernel function to dispatch.
    **kwargs
        Forwarded to `mr.Kernel`. Passing `math_mode` explicitly is allowed,
        but anything other than `SAFE` will fail to compile.

    Returns
    -------
        A compiled `mr.Kernel` with `math_mode=mr.MathMode.SAFE` by default.
    """
    kwargs.setdefault("math_mode", mr.MathMode.SAFE)
    assembled = mr.assemble(FRAGMENT, mr.Fragment(function_name, source))
    return mr.Kernel(assembled, function_name, **kwargs)


def split(x: NDArray[np.float64]) -> NDArray[np.float32]:
    """
    Split an array of double-precision floats into (hi, lo) pairs of float32s.

    `hi` is a 32-bit rounding of the input, and `lo` is the `x - hi` residual
    computed in 64-bit precision. As the residual contains up to 53 - 24 = 29
    significant bits, while `lo` has only 24, the resulting split loses 5 bits
    of precision as compared to `np.float64`.

    Parameters
    ----------
    x: NDArray[np.float64]
        The input array to split.

    Returns
    -------
        An array of shape `(*x.shape, 2)` consisting of `(hi, lo)` pairs of
        `np.float32`s that make up `x`, up to arithmetic precision.
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
    """
    Join an array of `(hi, lo)` pairs of `np.float32`s into an array of `np.float64`
    numbers.

    The input `pairs` array must have a last dimension equal to 2.

    Parameters
    ----------
    pairs: NDArray[np.float64]
        The input array to create 64-bit floating point numbers from.

    Returns
    -------
        An array of shape `(*x.shape[:-1])` consisting of `np.float64`s
        with the low and high parts equal to the pair constituents.
    """
    if pairs.shape[-1] != 2:
        raise ValueError("join(): input must be an array of shape (..., 2)")
    return pairs[..., 0].astype(np.float64) + pairs[..., 1].astype(np.float64)
