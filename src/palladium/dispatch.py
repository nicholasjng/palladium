"""Step 3 of the pipeline: hand emitted MSL to metal-runtime and run it.

Deliberately nothing but glue around Kernel/Buffer/run, plus the
debugging hooks: `PALLADIUM_DUMP_MSL=1` prints every kernel's source
before it compiles (a directory path writes `<name>_<hash>.metal` files
instead; the `MOSAIC_GPU_DUMP_PTX` idiom one layer up), and a Metal
compile failure re-raises with the line-numbered source attached so
`program_source:LINE:COL` diagnostics resolve by eye.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import metal_runtime as mr
import numpy as np

from palladium.trace import KernelSpec

__all__ = ["BoundKernel", "bind"]


def _dump_msl(name: str, msl_source: str) -> None:
    dump = os.environ.get("PALLADIUM_DUMP_MSL")
    if not dump:
        return
    if dump == "1" or dump.lower() == "stdout":
        print(f"// palladium kernel: {name}\n{msl_source}")
        return
    directory = Path(dump)
    directory.mkdir(parents=True, exist_ok=True)
    # Hash suffix: one shape-specialized kernel per file, no overwrites.
    digest = hashlib.sha256(msl_source.encode()).hexdigest()[:12]
    (directory / f"{name}_{digest}.metal").write_text(msl_source)


def _numbered(msl_source: str) -> str:
    return "\n".join(
        f"{n:4d} | {line}" for n, line in enumerate(msl_source.splitlines(), 1)
    )


@dataclasses.dataclass(frozen=True)
class BoundKernel:
    """A compiled Metal kernel behind a NumPy-in/NumPy-out call.

    Attributes
    ----------
    spec : KernelSpec
        The traced kernel this binary was emitted from.
    kernel : metal_runtime.Kernel
        The compiled pipeline.
    msl_source : str
        The exact source that compiled; kept because reading the emitted
        text beats re-deriving it when a kernel misbehaves.
    threadgroup : int or tuple of int, optional
        Explicit threadgroup size; None lets the runtime choose.
    """

    spec: KernelSpec
    kernel: mr.Kernel
    msl_source: str
    threadgroup: int | tuple[int, ...] | None = None

    def __call__(self, *arrays: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Dispatch over `spec.grid` threads and return the outputs.

        Parameters
        ----------
        *arrays : numpy.ndarray
            One array per kernel input, matching `spec.inputs` shapes;
            copied into fresh device buffers each call.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            One array per kernel output; a bare array for single-output
            kernels.
        """
        spec = self.spec
        if len(arrays) != len(spec.inputs):
            raise TypeError(
                f"kernel takes {len(spec.inputs)} arrays, got {len(arrays)}"
            )
        in_bufs = []
        for a, info in zip(arrays, spec.inputs, strict=True):
            arr = np.ascontiguousarray(np.asarray(a, dtype=info.dtype))
            if arr.shape != info.array_shape:
                raise TypeError(f"expected shape {info.array_shape}, got {arr.shape}")
            in_bufs.append(mr.Buffer(arr))
        out_bufs = [
            mr.Buffer.zeros(list(info.array_shape), dtype=info.dtype.name)
            for info in spec.outputs
        ]
        grid = tuple(int(g) for g in spec.grid)
        mr.run(
            self.kernel,
            grid=grid if len(grid) > 1 else grid[0],
            threadgroup=self.threadgroup,
            buffers=[*in_bufs, *out_bufs],
        )
        outs = tuple(b.to_numpy() for b in out_bufs)
        return outs[0] if len(outs) == 1 else outs


def bind(
    spec: KernelSpec,
    msl_source: str,
    *,
    math_mode: mr.MathMode = mr.MathMode.FAST,
    threadgroup: int | tuple[int, ...] | None = None,
) -> BoundKernel:
    """Compile emitted MSL into a dispatchable kernel.

    Parameters
    ----------
    spec : KernelSpec
        The traced kernel; fixes the function name, grid, and operand
        layout the source was emitted for.
    msl_source : str
        MSL text from `emit_msl`.
    math_mode : metal_runtime.MathMode, optional
        FAST by default. Use SAFE for kernels using the df32 prelude:
        FAST deletes compensated arithmetic (measured, see metal-runtime
        notes/float32x2.md).
    threadgroup : int or tuple of int, optional
        Explicit threadgroup size; None lets the runtime choose.

    Returns
    -------
    BoundKernel

    Raises
    ------
    metal_runtime.CompileError
        On MSL compile failure, with the line-numbered source attached.
    """
    _dump_msl(spec.name, msl_source)
    try:
        kernel = mr.Kernel(msl_source, spec.name, math_mode=math_mode)
    except mr.CompileError as e:
        raise mr.CompileError(
            f"{e}\n\npalladium-emitted source:\n{_numbered(msl_source)}"
        ) from None
    return BoundKernel(
        spec=spec, kernel=kernel, msl_source=msl_source, threadgroup=threadgroup
    )
