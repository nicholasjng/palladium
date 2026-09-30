"""float64 emulated as float32 pairs (df32) for hand-written Metal kernels.

`df32.metal` is the MSL library: two_sum, two_prod, the df32 struct, and
its arithmetic. Pallas has no df32 dtype, so kernels using it are written
in MSL directly; `symplectic_longrun.py` uses it for a compensated
integrator. A df32 buffer is a float32 array of shape (..., 2) holding
(hi, lo) pairs.
"""

from pathlib import Path

import metal_runtime as mr

PRELUDE = Path(__file__).with_name("df32.metal").read_text(encoding="utf-8")

FRAGMENT = mr.Fragment("df32", PRELUDE)


def kernel(source: str, function_name: str, **kwargs) -> mr.Kernel:
    """Build an `mr.Kernel` from `source` with `PRELUDE` prepended.

    `math_mode` defaults to SAFE, which the prelude requires: FAST
    reassociates away its compensation terms, and any other mode fails to
    compile. Other keywords are forwarded to `mr.Kernel`.
    """
    kwargs.setdefault("math_mode", mr.MathMode.SAFE)
    assembled = mr.assemble(FRAGMENT, mr.Fragment(function_name, source))
    return mr.Kernel(assembled, function_name, **kwargs)
