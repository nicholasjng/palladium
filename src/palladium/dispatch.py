"""Step 3 of the pipeline: hand emitted MSL to metal-runtime and run it.

Provided (not an exercise): the runtime contract is metal-runtime's, and
this module is deliberately nothing but glue around Kernel/Buffer/run.
"""

from __future__ import annotations

import dataclasses

import metal_runtime as mr
import numpy as np

from palladium.trace import KernelSpec

__all__ = ["BoundKernel", "bind"]


@dataclasses.dataclass(frozen=True)
class BoundKernel:
    """A compiled Metal kernel behind a NumPy-in/NumPy-out call.

    Buffer binding order matches emit_msl's signature convention: inputs
    then outputs, in jaxpr order. `msl_source` is kept for inspection —
    when a kernel misbehaves, read the text before the Python.
    """

    spec: KernelSpec
    kernel: mr.Kernel
    msl_source: str
    threadgroup: int | tuple[int, ...] | None = None

    def __call__(self, *arrays: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        spec = self.spec
        if len(arrays) != len(spec.inputs):
            raise TypeError(f"kernel takes {len(spec.inputs)} arrays, got {len(arrays)}")
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
    """Compile `msl_source` and return a NumPy-in/NumPy-out callable.

    Pass `math_mode=mr.MathMode.SAFE` for kernels using the df32 prelude —
    FAST deletes compensated arithmetic (measured, see metal-runtime
    notes/float32x2.md).
    """
    kernel = mr.Kernel(msl_source, spec.name, math_mode=math_mode)
    return BoundKernel(
        spec=spec, kernel=kernel, msl_source=msl_source, threadgroup=threadgroup
    )
