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
    scale = np.abs(d[np.isfinite(d)]).max(initial=0)
    # An absolute tolerance of rtol * scale; NaNs and infinities must match.
    np.testing.assert_allclose(
        a,
        d,
        rtol=0,
        atol=rtol * scale,
        err_msg=f"normwise rtol {rtol:g}, scale {scale:.3e} {err_msg}",
    )


def assert_trees_close(actual, desired, rtol: float | None = None, err_msg: str = "") -> None:
    """`assert_normwise_close` for each leaf of two pytrees of the same structure."""
    import jax

    for a, d in zip(jax.tree.leaves(actual), jax.tree.leaves(desired), strict=True):
        assert_normwise_close(a, d, rtol, err_msg)


def assert_matches_cpu(
    fn, *args, exact: bool = False, rtol: float | None = None, err_msg: str = ""
):
    """Run `fn` jitted on metal and on CPU, compare the outputs, and return metal's.

    `exact` requires bitwise-equal values (NaNs equal), as index maps produce.
    """
    import jax

    run = jax.jit(fn)
    got, want = (run(*jax.device_put(args, jax.devices(name)[0])) for name in ("metal", "cpu"))
    if exact:
        for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_array_equal(np.asarray(g), np.asarray(w), err_msg=err_msg)
    else:
        assert_trees_close(got, want, rtol, err_msg)
    return got
