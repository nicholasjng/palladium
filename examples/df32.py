"""float64 emulated as float32 pairs (df32) for hand-written Metal kernels.

`df32.metal` is the MSL library: two_sum, two_prod, the df32 struct, and
its arithmetic. Pallas has no df32 dtype, so kernels using it are written
in MSL directly; `symplectic_longrun.py` uses it for a compensated
integrator. A df32 buffer is a float32 array of shape (..., 2) holding
(hi, lo) pairs.
"""

from pathlib import Path

from palladium import _runtime

PRELUDE = Path(__file__).with_name("df32.metal").read_text(encoding="utf-8")


def kernel(source: str, function_name: str, math_mode: str = "safe") -> _runtime.Kernel:
    """Compile handwritten MSL with the compensated float32-pair arithmetic prelude.

    Parameters
    ----------
    source : str
        MSL kernel source referencing df32 arithmetic.
    function_name : str
        Kernel entry point.
    math_mode : str, optional
        "safe" by default; the prelude rejects anything else.

    Returns
    -------
    palladium._runtime.Kernel
        Compiled kernel with the prelude before the supplied source; `#line`
        markers keep compile errors pointing at the right part.

    Raises
    ------
    palladium._runtime.MetalError
        The source fails compilation, including when a non-SAFE math mode violates the
        prelude requirement.

    Notes
    -----
    Buffers store high/low float32 pairs in a trailing dimension of size two. This helper
    compiles handwritten Metal; Pallas does not expose a df32 dtype. SAFE is required
    because reassociation removes compensation terms.
    """
    assembled = f'#line 1 "df32.metal"\n{PRELUDE}\n#line 1 "{function_name}"\n{source}\n'
    return _runtime.Kernel(assembled, function_name, math_mode)
