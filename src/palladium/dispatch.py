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
import threading
from collections.abc import Callable
from pathlib import Path

import metal_runtime as mr
import numpy as np

from palladium.diagnostics import check_threadgroup, normalize_threadgroup
from palladium.emit import emit_msl_stats
from palladium.errors import DispatchError, EmitError, StackOverflowError
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


# NumPy refuses to export ml_dtypes extension dtypes over DLPack, so a
# bfloat16 array cannot cross into mr.Buffer as-is. The same bytes ship
# as uint16 and the buffer is relabeled; a pure reinterpretation, so the
# round trip is lossless and costs one O(1) view.
def _to_native(arr: np.ndarray) -> tuple[np.ndarray, str | None]:
    if arr.dtype.name == "bfloat16":
        return arr.view(np.uint16), "bfloat16"
    return arr, None


def _read_buffer(buf: mr.Buffer) -> np.ndarray:
    if buf.dtype == "bfloat16":
        import ml_dtypes

        return buf.to_numpy(dtype="uint16").view(ml_dtypes.bfloat16)
    return buf.to_numpy()


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
    done : bool
        True once `wait()` has returned; lets `BoundKernel` reuse the
        input-buffer slot without re-blocking on finished work.
    """

    batch: mr.Batch
    out_bufs: list[mr.Buffer]
    done: bool = False
    # Output indices that alias an input buffer: read back as a copy, not
    # a live view, since the buffer is rewritten by the next launch.
    copy_out: frozenset[int] = frozenset()
    _results: tuple[np.ndarray, ...] | None = None

    def wait(self) -> np.ndarray | tuple[np.ndarray, ...]:
        """Block until the GPU is done, then return the outputs.

        Results are materialized once and cached: the slot ring calls
        this before rewriting an aliased output's buffer, so the arrays
        a later caller-side `wait()` returns stay intact.
        """
        if self._results is None:
            self.batch.wait()
            self.done = True
            self._results = tuple(
                np.array(arr) if i in self.copy_out else arr
                for i, arr in enumerate(_read_buffer(b) for b in self.out_bufs)
            )
        outs = self._results
        return outs[0] if len(outs) == 1 else outs


@dataclasses.dataclass
class _Slot:
    """One ring entry: private input buffers plus the launch that last
    read them; overwritable only once that launch is known finished."""

    in_bufs: list[mr.Buffer]
    pending: PendingResult | None = None


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
    pipeline_depth : int
        Maximum launches in flight at once; see the field comment.
    """

    spec: KernelSpec
    kernel: mr.Kernel
    msl_source: str
    threadgroup: int | tuple[int, ...] | None = None
    # Maximum number of launches in flight at once.
    # Each concurrent launch needs its own input buffers, so the ceiling is also a memory bound:
    # depth x input bytes, though the ring only grows when launches actually overlap.
    pipeline_depth: int = 8
    # Ring of input-buffer slots, LRU-ordered, grown on demand up to
    # pipeline_depth. Buffer reuse is safe across calls: a BoundKernel is
    # cached per input shape/dtype, so shape never changes call-to-call.
    _slots: list[_Slot] = dataclasses.field(
        default_factory=list, compare=False, repr=False
    )
    # Index of the least-recently-launched slot, the next reuse candidate.
    _next: int = dataclasses.field(default=0, compare=False, repr=False)
    # Serializes slot acquisition and the upload-and-commit phase,
    # to avoid races with the GPU.
    _launch_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    def launch(self, *arrays: np.ndarray) -> PendingResult:
        """Encode and commit one dispatch without blocking on the result.

        `__call__` is `launch` followed by `wait()`. Thread-safe: uploads
        are serialized on a per-kernel lock, waits are not.

        Launches without intervening waits overlap on the GPU queue, up
        to `pipeline_depth` in flight, amortizing the fixed per-dispatch
        queue latency; past that, `launch` blocks on the oldest.

        Parameters
        ----------
        *arrays : numpy.ndarray
            One array per kernel input, matching `spec.inputs` shapes;
            copied into fresh or reused device buffers (see `_slots`).

        Returns
        -------
        PendingResult
            Committed, not yet waited on.

        Raises
        ------
        DispatchError
            On an argument count, shape, or dtype mismatch against the
            traced spec. Dtypes are checked strictly, never cast: a
            silent f64 -> f32 cast would hide a jax_enable_x64 mixup.
        """
        spec = self.spec
        if len(arrays) != len(spec.inputs):
            raise DispatchError(
                f"kernel takes {len(spec.inputs)} arrays, got {len(arrays)}"
            )
        # Validated and made contiguous before any slot is acquired, so an
        # argument error never blocks on (or claims) in-flight work.
        natives = []
        for i, (a, info) in enumerate(zip(arrays, spec.inputs, strict=True)):
            arr = np.asarray(a)
            if arr.dtype != info.dtype:
                raise DispatchError(
                    f"argument {i}: dtype {arr.dtype} does not match the "
                    f"traced {info.dtype}; cast explicitly"
                )
            if arr.shape != info.array_shape:
                raise DispatchError(
                    f"argument {i}: expected shape {info.array_shape}, got {arr.shape}"
                )
            # Non-contiguous inputs are copied, not rejected: upload
            # copies into the device buffer anyway. Contiguity must
            # come first: _to_native's view needs a contiguous array.
            arr = np.ascontiguousarray(arr)
            natives.append(_to_native(arr))
        with self._launch_lock:
            slot = self._acquire_slot()
            if not slot.in_bufs:
                slot.in_bufs.extend(
                    mr.Buffer(native, dtype=relabel) for native, relabel in natives
                )
            else:
                for buf, (native, relabel) in zip(slot.in_bufs, natives, strict=True):
                    buf.copy_from(native, dtype=relabel)
            # Fresh per call, unlike inputs: to_numpy() is a live view, so
            # reusing this buffer would mutate an array a caller might still
            # be holding from an earlier, not-yet-waited-on PendingResult.
            # Aliased outputs instead share the slot's input buffer (the
            # in-place contract) and are copied out on wait().
            alias_of = {j: i for i, j in spec.aliases}
            out_bufs = [
                slot.in_bufs[alias_of[j]]
                if j in alias_of
                else mr.Buffer.empty(list(info.array_shape), dtype=info.dtype.name)
                for j, info in enumerate(spec.outputs)
            ]
            pending = self._dispatch(slot.in_bufs, out_bufs)
            slot.pending = pending
            return pending

    def _acquire_slot(self) -> _Slot:
        """Pick the slot for the next launch; call under `_launch_lock`.

        Reuses the least-recently-launched slot if its work is finished
        (a launch-then-wait caller stays at one slot), grows the ring up
        to `pipeline_depth` while launches overlap, then blocks on the
        oldest in-flight batch: the queue is FIFO, so it finishes first.
        """
        if self._slots:
            slot = self._slots[self._next]
            if slot.pending is None or slot.pending.done:
                self._advance()
                return slot
        if len(self._slots) < self.pipeline_depth:
            slot = _Slot(in_bufs=[])
            # Inserted at the cursor, so ring order stays launch order and
            # _next keeps pointing at the least-recently-launched slot.
            self._slots.insert(self._next, slot)
            self._advance()
            return slot
        slot = self._slots[self._next]
        assert slot.pending is not None  # full ring: every slot launched
        # wait(), not batch.wait(): reusing the slot rewrites its buffers,
        # so an aliased output must be materialized for its caller first.
        slot.pending.wait()
        self._advance()
        return slot

    def _advance(self) -> None:
        object.__setattr__(self, "_next", (self._next + 1) % len(self._slots))

    def __call__(self, *arrays: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Dispatch over `spec.grid` threads and return the outputs.

        Parameters
        ----------
        *arrays : numpy.ndarray
            One array per kernel input, matching `spec.inputs` shapes;
            copied into fresh or reused device buffers (see `_slots`).

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
        batch = mr.Batch()
        batch.add(
            self.kernel,
            grid=grid if len(grid) > 1 else grid[0],
            threadgroup=self.threadgroup,
            buffers=[*in_bufs, *out_bufs],
        )
        batch.commit()
        copy_out = frozenset(j for _, j in self.spec.aliases)
        return PendingResult(batch, out_bufs, copy_out=copy_out)

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
        if self.spec.aliases:
            raise DispatchError(
                "pinned() is unsupported for kernels with "
                "input_output_aliases: the kernel writes its pinned input "
                "in place, so repeated calls would not see the original data"
            )
        pending = self.launch(*arrays)
        pending.wait()
        # Detach the slot that served the upload: its buffers become
        # private to this callable, so later launch() calls can't
        # overwrite the pinned data (they allocate a replacement slot).
        with self._launch_lock:
            index = next(i for i, s in enumerate(self._slots) if s.pending is pending)
            in_bufs = self._slots.pop(index).in_bufs
            object.__setattr__(
                self, "_next", self._next % len(self._slots) if self._slots else 0
            )

        def call() -> np.ndarray | tuple[np.ndarray, ...]:
            # No lock and no wait-before-dispatch: the pinned inputs are
            # never rewritten and outputs are fresh per call, so there is
            # no buffer to race.
            out_bufs = [
                mr.Buffer.empty(list(info.array_shape), dtype=info.dtype.name)
                for info in self.spec.outputs
            ]
            return self._dispatch(in_bufs, out_bufs).wait()

        return call


def bind(
    spec: KernelSpec,
    msl_source: str,
    *,
    math_mode: mr.MathMode = mr.MathMode.FAST,
    threadgroup: int | tuple[int, ...] | None = None,
    pipeline_depth: int = 8,
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
        Explicit threadgroup size; None lets the runtime choose.
    pipeline_depth : int, optional
        Maximum launches in flight at once for `BoundKernel.launch`
        callers, 8 by default; 1 restores strictly serial dispatch.

    Returns
    -------
    BoundKernel

    Raises
    ------
    ValueError
        `pipeline_depth` is not a positive integer.
    metal_runtime.CompileError
        On MSL compile failure, with the line-numbered source attached.
        Fragment-assembled source (`metal_runtime.build_source`) is
        re-raised as-is instead: Metal's diagnostic
        already names the fragment and line, which the flat dump's line
        numbers would only obscure.
    """
    if pipeline_depth < 1:
        raise ValueError(f"pipeline_depth must be >= 1, got {pipeline_depth}")
    # Resolve sentinels ('simdgroup') and int shorthand once, here: what
    # BoundKernel stores goes straight to mr.Batch.add, which takes only
    # ints and sequences.
    threadgroup = normalize_threadgroup(threadgroup)
    check_threadgroup(spec, threadgroup)
    if spec.uses_threadgroup and threadgroup is None:
        raise EmitError(
            f"kernel {spec.name!r} declares threadgroup_memory scratch, so it "
            "must be dispatched with an explicit threadgroup= size. Leaving it "
            "None lets the runtime pick a size (commonly far larger than the "
            "declared extent), and a thread_index() past that extent writes "
            "out of bounds with no error. Pass threadgroup=N with N no larger "
            "than the leading extent of every threadgroup_memory request."
        )
    _dump_msl(spec.name, msl_source)
    try:
        kernel = mr.Kernel(msl_source, spec.name, math_mode=math_mode)
    except mr.PipelineBuildError as e:
        if "stack space" in str(e):
            # Metal's pipeline creation rejects kernels whose thread-local
            # arrays overflow the per-thread stack; translate the opaque
            # message into the actual fix. Re-emitting to recover the byte
            # count is wasted work only on this already-failing path, and it
            # turns "too large" into a number to aim at.
            measured = emit_msl_stats(spec)[1].thread_bytes
            raise StackOverflowError(
                f"{e}\n\nEvery loaded block and intermediate value lives in "
                f"thread-local memory, and this kernel declares about "
                f"{measured} bytes of it per program instance -- too much for "
                "the per-thread stack. Shrink it by adding or refining the "
                "grid and BlockSpecs so each program instance touches a "
                "smaller block; palladium.metal_call(...).explain(*args) "
                "reports the figure without compiling.",
                stack_bytes=measured,
            ) from None
        raise
    except mr.CompileError as e:
        if msl_source.startswith("#line ") or "\n#line " in msl_source:
            raise
        raise mr.CompileError(
            f"{e}\n\npalladium-emitted source:\n{_numbered(msl_source)}"
        ) from None
    return BoundKernel(
        spec=spec,
        kernel=kernel,
        msl_source=msl_source,
        threadgroup=threadgroup,
        pipeline_depth=pipeline_depth,
    )
