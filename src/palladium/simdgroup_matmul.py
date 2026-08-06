"""Standalone tiled-simdgroup matmul: Metal 3 `simdgroup_matrix` hardware
matrix units, one 8x8 tile per Metal threadgroup (one simdgroup, 32
threads, cooperating on that tile via `simdgroup_load`/
`simdgroup_multiply_accumulate`/`simdgroup_store`).

Deliberately not integrated with the trace/emit/bind pipeline. Inlining
this into `_rule_dot_general` so it fires inside an arbitrary fused
kernel (e.g. the flash-attention example) would need 32 cooperating
threads to coexist with a kernel that's otherwise one-thread-per-Pallas-
program-instance, which needs its own execution-model design (see
docs/simdgroup-matmul-design.md, "what has to change" 1-4). This covers
only the standalone case: a kernel that *is* a single matmul, called
directly, not staged through `pallas_call`.

Verified against `docs/simdgroup-matmul-design.md`'s isolated benchmark:
the tile-grid kernel there is exactly what this module compiles and
dispatches (same MSL shape, same grid/threadgroup mapping). Measured,
with a GPU-clock warm-up soak and both sides on equal footing, to reach
parity with `jax.jit` CPU matmul at 512x512x512 (not a win -- an earlier
pass claimed 1.61x here, which turned out to be an artifact of measuring
mid-clock-ramp, see that doc's methodology note) and close most of the
gap by 256; still dispatch-floor-bound at 64/128, which happen to be the
sizes this project's actual kernels use. This module's own `__call__`
also pays real per-call `mr.Buffer` upload cost the isolated benchmark
didn't include, so end-to-end timings through this API run slower still
than the numbers above, which measure dispatch only.
"""

from __future__ import annotations

import dataclasses

import metal_runtime as mr
import numpy as np

from palladium.emit import EmitError

__all__ = ["SimdgroupMatmul", "simdgroup_matmul"]

# ctype -> Metal simdgroup_<T>8x8 spelling. Only float32 has been run
# against a NumPy oracle here; half is a plausible Metal extension
# (simdgroup_half8x8 exists) but untested, so it's excluded until it is.
_SIMDGROUP_CTYPES = {"float32": "float"}

_MSL_TEMPLATE = """\
#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    device const {ctype}* A [[buffer(0)]],
    device const {ctype}* B [[buffer(1)]],
    device {ctype}* C [[buffer(2)]],
    uint2 tile [[threadgroup_position_in_grid]])
{{
    uint ti = tile.y, tj = tile.x;
    simdgroup_{ctype}8x8 acc = simdgroup_{ctype}8x8(0);
    for (uint tk = 0; tk < {k}u / 8u; ++tk) {{
        simdgroup_{ctype}8x8 a, b;
        simdgroup_load(a, A + ti * 8 * {k}u + tk * 8u, {k}u);
        simdgroup_load(b, B + tk * 8u * {n}u + tj * 8u, {n}u);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }}
    simdgroup_store(acc, C + ti * 8u * {n}u + tj * 8u, {n}u);
}}
"""


@dataclasses.dataclass(frozen=True)
class SimdgroupMatmul:
    """A compiled tiled-simdgroup matmul for one fixed `(m, k, n, dtype)`.

    One simdgroup (32 threads) per 8x8 output tile, parallel across the
    `(m/8, n/8)` tile grid; no threadgroup-memory staging or reuse across
    tiles sharing a row/column of `a`/`b` (every tile re-reads its inputs
    straight from `device` memory via `simdgroup_load`) -- the ROADMAP
    stretch-12 endgame is a further step up from this.

    Attributes
    ----------
    m, k, n : int
        Fixed operand shapes this kernel was compiled for.
    dtype : str
        NumPy dtype name (`"float32"`).
    kernel : metal_runtime.Kernel
        The compiled pipeline.
    """

    m: int
    k: int
    n: int
    dtype: str
    kernel: mr.Kernel

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """`a @ b`. `a` is `(m, k)`, `b` is `(k, n)`, both row-major.

        Raises
        ------
        ValueError
            `a`/`b` don't match the shapes this kernel was compiled for.
        """
        a = np.asarray(a, dtype=self.dtype)
        b = np.asarray(b, dtype=self.dtype)
        if a.shape != (self.m, self.k) or b.shape != (self.k, self.n):
            raise ValueError(
                f"expected shapes {(self.m, self.k)}, {(self.k, self.n)}; "
                f"got {a.shape}, {b.shape}"
            )
        a_buf = mr.Buffer(np.ascontiguousarray(a))
        b_buf = mr.Buffer(np.ascontiguousarray(b))
        c_buf = mr.Buffer.empty([self.m, self.n], dtype=self.dtype)
        mr.run(
            self.kernel,
            grid=((self.n // 8) * 32, self.m // 8, 1),
            threadgroup=(32, 1, 1),
            buffers=[a_buf, b_buf, c_buf],
        )
        return c_buf.to_numpy()


def simdgroup_matmul(m: int, k: int, n: int, dtype: str = "float32") -> SimdgroupMatmul:
    """Compile a standalone tiled-simdgroup matmul for fixed `(m, k, n)`.

    Parameters
    ----------
    m, k, n : int
        Operand shapes: `a` is `(m, k)`, `b` is `(k, n)`. Each must be a
        multiple of 8 (`simdgroup_float8x8`'s tile size); no padding or
        remainder path is implemented, matching how `_rule_dot_general`
        rejects rather than silently handles shapes it doesn't support.
    dtype : str, optional
        `"float32"` (default); the only dtype verified against a NumPy
        oracle so far.

    Returns
    -------
    SimdgroupMatmul
        Compiled, NumPy-in/NumPy-out callable for exactly this shape.

    Raises
    ------
    EmitError
        `m`, `k`, or `n` isn't a multiple of 8, or `dtype` has no
        verified `simdgroup_<T>8x8` mapping.
    """
    if m % 8 or k % 8 or n % 8:
        raise EmitError(
            f"simdgroup_matmul: m={m}, k={k}, n={n} must each be a "
            "multiple of 8 (simdgroup_float8x8 tile size); no padding "
            "path is implemented"
        )
    ctype = _SIMDGROUP_CTYPES.get(dtype)
    if ctype is None:
        raise EmitError(
            f"simdgroup_matmul: dtype {dtype!r} has no verified Metal "
            f"simdgroup_<T>8x8 mapping (supported: {sorted(_SIMDGROUP_CTYPES)})"
        )
    name = "palladium_simdgroup_matmul"
    src = _MSL_TEMPLATE.format(name=name, ctype=ctype, k=k, n=n)
    kernel = mr.Kernel(src, name)
    return SimdgroupMatmul(m=m, k=k, n=n, dtype=dtype, kernel=kernel)
