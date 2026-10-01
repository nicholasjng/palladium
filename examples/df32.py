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
    """Compile handwritten MSL with the compensated float32-pair arithmetic prelude.

    Parameters
    ----------
    source : str
        MSL kernel source referencing df32 arithmetic.
    function_name : str
        Kernel entry point.
    **kwargs : dict
        Keywords forwarded to metal_runtime.Kernel; math_mode defaults to SAFE.

    Returns
    -------
    metal_runtime.Kernel
        Compiled kernel with the df32 fragment assembled before the supplied source.

    Raises
    ------
    metal_runtime.CompileError
        The source fails compilation, including when a non-SAFE math mode violates the
        prelude requirement.

    Notes
    -----
    Buffers store high/low float32 pairs in a trailing dimension of size two. This helper
    compiles handwritten Metal; Pallas does not expose a df32 dtype. SAFE is required
    because reassociation removes compensation terms.
    """
    kwargs.setdefault("math_mode", mr.MathMode.SAFE)
    assembled = mr.assemble(FRAGMENT, mr.Fragment(function_name, source))
    return mr.Kernel(assembled, function_name, **kwargs)
