"""Step 3 of the pipeline: compile emitted MSL via metal-runtime and run it.

Glue around Kernel/Buffer/run, plus debugging hooks. `PALLADIUM_DUMP_MSL=1`
prints each kernel's source before compiling; a directory path instead
writes `<name>_<hash>.metal` files. Metal compile failures re-raise with
the line-numbered source attached.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import metal_runtime as mr
import numpy as np

from palladium.emit import is_simdgroup_cooperative
from palladium.emit.core import SIMDGROUP_WIDTH
from palladium.trace import KernelSpec

__all__ = ["BoundKernel", "PendingResult", "bind"]


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


@dataclasses.dataclass
class PendingResult:
    """A launched kernel whose command buffer may still be executing.

    Returned by `BoundKernel.launch`, committed but not waited on.

    Attributes
    ----------
    batch : metal_runtime.Batch
        Already committed; `wait()` blocks on it if unfinished.
    out_bufs : list of metal_runtime.Buffer
        This launch's outputs, read back on `wait()`.
    """

    batch: mr.Batch
    out_bufs: list[mr.Buffer]

    def wait(self) -> np.ndarray | tuple[np.ndarray, ...]:
        """Block until the GPU is done, then return the outputs."""
        self.batch.wait()
        outs = tuple(b.to_numpy() for b in self.out_bufs)
        return outs[0] if len(outs) == 1 else outs


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
        Exact source that compiled.
    threadgroup : int or tuple of int, optional
        Explicit threadgroup size; None lets the runtime choose.
    """

    spec: KernelSpec
    kernel: mr.Kernel
    msl_source: str
    threadgroup: int | tuple[int, ...] | None = None
    # Simdgroup-cooperative kernels (emit.is_simdgroup_cooperative) index
    # program instances by threadgroup_position_in_grid with one 32-thread
    # SIMD-group per instance, so their launch geometry is fixed: 32x the
    # grid's x threads, threadgroup (32, 1, 1). Must match what emit_msl
    # was told when the source was generated.
    cooperative: bool = False
    # Safe to reuse across calls: a BoundKernel is cached per input
    # shape/dtype, so shape never changes call-to-call. Excluded from
    # compare/repr since it's cache state, not part of a BoundKernel's identity.
    _in_bufs: list[mr.Buffer] = dataclasses.field(
        default_factory=list, compare=False, repr=False
    )
    # copy_from() into a reused _in_bufs entry races an in-flight GPU read of
    # that same shared-storage buffer unless the prior launch is known done;
    # waited on (a no-op if the caller already did) before the next reuse.
    _last_pending: PendingResult | None = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def launch(self, *arrays: np.ndarray) -> PendingResult:
        """Encode and commit one dispatch without blocking on the result.

        `__call__` is `launch` followed by `wait()`.

        Parameters
        ----------
        *arrays : numpy.ndarray
            One array per kernel input, matching `spec.inputs` shapes;
            copied into fresh or reused device buffers (see `_in_bufs`).

        Returns
        -------
        PendingResult
            Committed, not yet waited on.
        """
        spec = self.spec
        if len(arrays) != len(spec.inputs):
            raise TypeError(
                f"kernel takes {len(spec.inputs)} arrays, got {len(arrays)}"
            )
        if self._last_pending is not None:
            self._last_pending.batch.wait()  # no-op if the caller already waited
        first_call = len(self._in_bufs) < len(spec.inputs)
        in_bufs = []
        for i, (a, info) in enumerate(zip(arrays, spec.inputs, strict=True)):
            arr = np.ascontiguousarray(np.asarray(a, dtype=info.dtype))
            if arr.shape != info.array_shape:
                raise TypeError(f"expected shape {info.array_shape}, got {arr.shape}")
            if first_call:
                self._in_bufs.append(mr.Buffer(arr))
            else:
                self._in_bufs[i].copy_from(arr)
            in_bufs.append(self._in_bufs[i])
        # Fresh per call, unlike inputs: to_numpy() is a live view, so reusing
        # this buffer would mutate an array a caller might still be holding
        # from an earlier, not-yet-waited-on PendingResult.
        out_bufs = [
            mr.Buffer.empty(list(info.array_shape), dtype=info.dtype.name)
            for info in spec.outputs
        ]
        return self._dispatch(in_bufs, out_bufs)

    def __call__(self, *arrays: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Dispatch over `spec.grid` threads and return the outputs.

        Parameters
        ----------
        *arrays : numpy.ndarray
            One array per kernel input, matching `spec.inputs` shapes;
            copied into fresh or reused device buffers (see `_in_bufs`).

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            One array per kernel output; a bare array for single-output
            kernels.
        """
        return self.launch(*arrays).wait()

    def _dispatch(
        self, in_bufs: list[mr.Buffer], out_bufs: list[mr.Buffer]
    ) -> PendingResult:
        """Encode, commit, and track one dispatch on prepared buffers."""
        grid = tuple(int(g) for g in self.spec.grid)
        if self.cooperative:
            # One SIMD-group (one threadgroup) per instance.
            grid = (grid[0] * SIMDGROUP_WIDTH, *grid[1:])
            threadgroup: int | tuple[int, ...] | None = (SIMDGROUP_WIDTH, 1, 1)
        else:
            threadgroup = self.threadgroup
        batch = mr.Batch()
        batch.add(
            self.kernel,
            grid=grid if len(grid) > 1 else grid[0],
            threadgroup=threadgroup,
            buffers=[*in_bufs, *out_bufs],
        )
        batch.commit()
        pending = PendingResult(batch, out_bufs)
        object.__setattr__(self, "_last_pending", pending)
        return pending

    def pinned(
        self, *arrays: np.ndarray
    ) -> Callable[[], np.ndarray | tuple[np.ndarray, ...]]:
        """Upload `arrays` once; return a zero-argument callable that
        re-dispatches on the pinned device buffers.

        Skips the per-call input upload of `__call__` for repeated calls
        on unchanging inputs. Later mutation of the passed arrays is not
        observed (data is copied at pin time). Outputs stay fresh per
        call, same reasoning as `launch`.
        """
        pending = self.launch(*arrays)
        pending.wait()

        def call() -> np.ndarray | tuple[np.ndarray, ...]:
            if self._last_pending is not None:
                self._last_pending.batch.wait()
            out_bufs = [
                mr.Buffer.empty(list(info.array_shape), dtype=info.dtype.name)
                for info in self.spec.outputs
            ]
            return self._dispatch(self._in_bufs, out_bufs).wait()

        return call


def bind(
    spec: KernelSpec,
    msl_source: str,
    *,
    math_mode: mr.MathMode = mr.MathMode.FAST,
    threadgroup: int | tuple[int, ...] | None = None,
    cooperative: bool | None = None,
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
        FAST by default. Use SAFE for df32-prelude kernels; FAST drops
        compensated arithmetic.
    threadgroup : int or tuple of int, optional
        Explicit threadgroup size; None lets the runtime choose. An
        explicit size opts the kernel out of the cooperative model (its
        launch geometry is fixed at (32, 1, 1)) unless `cooperative`
        overrides.
    cooperative : bool, optional
        Launch with the simdgroup-cooperative geometry. None (default)
        mirrors `emit_msl`'s own default, `emit.is_simdgroup_cooperative
        (spec)`, unless an explicit `threadgroup` was given. Pass the
        same value that was passed to `emit_msl` if you overrode it
        there; geometry and codegen must agree.

    Returns
    -------
    BoundKernel

    Raises
    ------
    metal_runtime.CompileError
        On MSL compile failure, with the line-numbered source attached.
    """
    if cooperative is None:
        cooperative = threadgroup is None and is_simdgroup_cooperative(spec)
    _dump_msl(spec.name, msl_source)
    try:
        kernel = mr.Kernel(msl_source, spec.name, math_mode=math_mode)
    except mr.CompileError as e:
        raise mr.CompileError(
            f"{e}\n\npalladium-emitted source:\n{_numbered(msl_source)}"
        ) from None
    return BoundKernel(
        spec=spec,
        kernel=kernel,
        msl_source=msl_source,
        threadgroup=threadgroup,
        cooperative=cooperative,
    )
