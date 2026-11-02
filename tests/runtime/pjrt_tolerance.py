"""The M1 agreement rule between the Metal and CPU backends.

Floating arrays must agree in normwise relative error,
``max|actual - desired| <= rtol * max|desired|``, per array. Componentwise
relative error is unattainable for values that cancel to near zero, and a
fixed absolute tolerance is vacuous for arrays whose scale is below it (such
as Adam's second moments). Integers and booleans match exactly; nonfinite
values must match in position and value.
"""

import numpy as np

# rtol by itemsize: float32 and the 16-bit floats (f16, bf16).
RTOL = {4: 2e-5, 2: 1e-2}


def assert_normwise_close(actual, desired, rtol: float | None = None, err_msg: str = "") -> None:
    a, d = np.asarray(actual), np.asarray(desired)
    assert a.shape == d.shape, f"{err_msg}: shape {a.shape} versus {d.shape}"
    if d.dtype.kind in "biu":
        np.testing.assert_array_equal(a, d, err_msg=err_msg)
        return
    if rtol is None:
        rtol = RTOL[d.dtype.itemsize]
    a, d = a.astype(np.float64), d.astype(np.float64)
    finite = np.isfinite(d)
    # NaNs compare equal here; infinities must have the same sign.
    np.testing.assert_array_equal(a[~finite], d[~finite], err_msg=f"{err_msg}: nonfinite values")
    assert np.isfinite(a[finite]).all(), f"{err_msg}: nonfinite where the reference is finite"
    if not finite.any():
        return
    scale = float(np.abs(d[finite]).max())
    error = float(np.abs(a[finite] - d[finite]).max())
    assert error <= rtol * scale, (
        f"{err_msg}: normwise relative error {error / scale if scale else np.inf:.3e} > {rtol:g}"
        f" (max abs error {error:.3e}, max |reference| {scale:.3e})"
    )
