"""Step 1 of the pipeline: extract the kernel jaxpr from Pallas.

Traces the wrapped `pl.pallas_call` with `jax.make_jaxpr` and repackages
the resulting `pallas_call` equation into a KernelSpec.

Pinned to JAX 0.11: `grid_mapping`/`block_mapping` dataclass fields are
version-sensitive (see `_block_infos`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Literal as TLiteral

import jax
import jax.experimental.pallas as pl
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Var

from palladium import effects
from palladium.errors import TraceError

__all__ = ["BlockInfo", "KernelSpec", "ScratchInfo", "trace"]


@dataclasses.dataclass(frozen=True)
class ScratchInfo:
    """An extra scratch buffer in a kernel without a backing caller array.

    Attributes
    ----------
    shape: tuple of int
        Buffer shape.
    dtype: numpy.dtype
        Buffer element type.
    space: str
        Metal address space: `"thread"` (private to one program
        instance, the `pl.MemorySpace` default) or `"threadgroup"`
        (shared across the threadgroup, requested via
        `palladium.threadgroup_memory`). Decides the qualifier
        `emit_msl` declares the storage with, and nothing else -- both
        are compile-time-sized local arrays.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    space: TLiteral["thread", "threadgroup"] = "thread"


@dataclasses.dataclass(frozen=True)
class BlockInfo:
    """One kernel operand: the block a single program instance sees.

    Attributes
    ----------
    block_shape : tuple of int
        Kernel-visible block shape, squeezed dims removed.
    array_shape : tuple of int
        Full array shape.
    dtype : numpy.dtype
        Element type, shared by block and array.
    index_map_jaxpr : ClosedJaxpr
        Staged BlockSpec index map: grid indices to block index, in
        units of blocks.
    """

    block_shape: tuple[int, ...]
    array_shape: tuple[int, ...]
    dtype: np.dtype
    index_map_jaxpr: ClosedJaxpr


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """Everything the emitter needs, and nothing it doesn't.

    Attributes
    ----------
    name : str
        Kernel name, used as the MSL function name.
    jaxpr : Jaxpr
        The kernel body: a stateful jaxpr over Refs.
    grid : tuple of int
        Pallas grid; `(1,)` for gridless calls.
    inputs, outputs : tuple of BlockInfo
        Operand descriptions in jaxpr order.
    scratch: tuple of ScratchInfo
        Scratch buffer descriptions, in jaxpr order after the operands.
    raw_params : dict
        Full, unprocessed pallas_call params.
    aliases : tuple of (int, int)
        Validated `input_output_aliases` pairs (input index, output index):
        the two refs share one buffer, so the kernel updates in place.
    """

    name: str
    jaxpr: Jaxpr
    grid: tuple[int, ...]
    inputs: tuple[BlockInfo, ...]
    outputs: tuple[BlockInfo, ...]
    scratch: tuple[ScratchInfo, ...]
    raw_params: dict[str, Any]
    aliases: tuple[tuple[int, int], ...] = ()

    @property
    def uses_threadgroup(self) -> bool:
        """Whether any scratch entry lives in `threadgroup` space.

        True means the kernel communicates across threads, so the
        dispatch threadgroup size stops being a free tuning knob and
        becomes part of the kernel's contract (`palladium.bind`).
        """
        return any(info.space == "threadgroup" for info in self.scratch)

    @property
    def num_programs(self) -> int:
        """Total number of program instances, one Metal thread each."""
        n = 1
        for g in self.grid:
            n *= g
        return n


def _block_dim(dim: Any) -> int | None:
    # JAX 0.11 stages BlockSpec shapes as BlockDim objects rather than
    # plain ints; squeezed dims vanish from the visible block. pl.Element,
    # pl.Indirect, and pl.BoundedSlice dims also carry `block_size` but
    # mean different indexing semantics, and accepting them here would
    # lower wrong offsets silently.
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, pl.Squeezed):
        return None
    if isinstance(dim, pl.Blocked):
        return int(dim.block_size)
    raise TraceError(
        f"unsupported block dim type {type(dim).__name__}: only int, "
        "pl.Blocked, and pl.Squeezed dims are lowered (pl.Element, "
        "pl.Indirect, and pl.BoundedSlice indexing is unimplemented)"
    )


def _block_infos(block_mappings: list[Any]) -> tuple[BlockInfo, ...]:
    infos = []
    for bm in block_mappings:
        dims = [_block_dim(d) for d in bm.block_shape]
        infos.append(
            BlockInfo(
                block_shape=tuple(d for d in dims if d is not None),
                array_shape=tuple(bm.array_aval.shape),
                dtype=np.dtype(bm.array_aval.dtype),
                index_map_jaxpr=bm.index_map_jaxpr,
            )
        )
    return tuple(infos)


def _validate_aliases(
    jaxpr: Jaxpr,
    aliases: tuple[tuple[int, int], ...],
    inputs: tuple[BlockInfo, ...],
    outputs: tuple[BlockInfo, ...],
) -> None:
    """One buffer behind both refs is only transparent when (a) both
    sides slice it identically, and (b) every read of the input executes
    before any write of the output; Pallas semantics keep the input's
    pre-call values visible throughout. (b) is checked on effect order
    over the kernel eqns, which covers sub-jaxprs.
    """
    for i, j in aliases:
        a, b = inputs[i], outputs[j]
        if (
            a.array_shape != b.array_shape
            or a.dtype != b.dtype
            or a.block_shape != b.block_shape
            or str(a.index_map_jaxpr) != str(b.index_map_jaxpr)
        ):
            raise TraceError(
                f"input_output_aliases pair ({i}, {j}): input and output "
                "must have identical array shape, dtype, block shape, and "
                "BlockSpec index map to share one buffer"
            )
        x_var = jaxpr.invars[i]
        o_var = jaxpr.invars[len(inputs) + j]
        last_read = max(
            (n for n, e in enumerate(jaxpr.eqns) if effects.eqn_reads_ref(e, x_var)),
            default=-1,
        )
        first_write = min(
            (n for n, e in enumerate(jaxpr.eqns) if effects.eqn_writes_ref(e, o_var)),
            default=len(jaxpr.eqns),
        )
        if last_read >= first_write:
            raise TraceError(
                f"input_output_aliases pair ({i}, {j}): input {i} is read "
                f"after (or while) output {j} is written. The refs share "
                "one buffer on the GPU, so that read would observe the "
                "in-place write; reorder the kernel to read all its input "
                "before the first write to the aliased output"
            )


def _depends_on(
    jaxpr: Jaxpr, seeds: set[Var], targets: Sequence[Var | Literal]
) -> bool:
    """Forward data-dependence over top-level eqns: does any of `targets`
    depend on a var in `seeds`? Coarse across sub-jaxpr eqns: any
    tainted invar taints all outvars."""
    tainted = set(seeds)
    for eqn in jaxpr.eqns:
        if any(isinstance(v, Var) and v in tainted for v in eqn.invars):
            tainted.update(eqn.outvars)
    return any(isinstance(v, Var) and v in tainted for v in targets)


def _map_uses_axis(index_map: ClosedJaxpr, axis: int) -> bool:
    inner = index_map.jaxpr
    if axis >= len(inner.invars):
        return False
    seed = inner.invars[axis]
    return seed in inner.outvars or _depends_on(inner, {seed}, list(inner.outvars))


def _validate_parallel_writes(
    jaxpr: Jaxpr,
    outputs: tuple[BlockInfo, ...],
    grid: tuple[int, ...],
    n_in: int,
) -> None:
    """Program instances run as parallel threads, so instances writing
    the same output element race. Rejects the provable case: a grid axis
    that neither the output's index map nor any top-level write index
    depends on. Writes inside sub-jaxprs and non-injective maps that use
    the axis pass unchecked (best-effort by design).
    """
    for j, info in enumerate(outputs):
        o_var = jaxpr.invars[n_in + j]
        writes = [
            e for e in jaxpr.eqns if e.primitive.name == "swap" and e.invars[0] is o_var
        ]
        if not writes:
            continue  # no top-level writes to analyze; sub-jaxprs stay unchecked
        for axis, extent in enumerate(grid):
            if extent <= 1 or _map_uses_axis(info.index_map_jaxpr, axis):
                continue
            pid_vars = {
                out
                for e in jaxpr.eqns
                if e.primitive.name == "program_id" and e.params["axis"] == axis
                for out in e.outvars
            }
            for eqn in writes:
                indexer_args = list(eqn.invars[2:])
                if not (pid_vars and _depends_on(jaxpr, pid_vars, indexer_args)):
                    raise TraceError(
                        f"output {j} is written identically by all "
                        f"{extent} program instances along grid axis "
                        f"{axis}: neither its BlockSpec index map nor the "
                        "write index depends on that axis, and instances "
                        "run as parallel threads, so the writes race. Make "
                        f"the output BlockSpec index map use grid axis "
                        f"{axis}, or index the write with pl.program_id({axis})"
                    )


def _scratch_infos(scratch_avals: Any) -> list[ScratchInfo]:
    # `memory_space` is typed `Any` upstream, so palladium's THREADGROUP
    # sentinel rides through Pallas tracing on it untouched. Anything
    # else (the pl.MemorySpace members) is thread-private storage.
    from palladium.threadgroup import THREADGROUP

    infos = []
    for aval in scratch_avals:
        space: TLiteral["thread", "threadgroup"] = (
            "threadgroup"
            if getattr(aval, "memory_space", None) is THREADGROUP
            else "thread"
        )
        infos.append(
            ScratchInfo(
                shape=tuple(int(d) for d in aval.shape),
                dtype=np.dtype(aval.dtype),
                space=space,
            )
        )
    return infos


def trace(pallas_fn: Callable, *example_args) -> KernelSpec:
    """Extract a KernelSpec from a function that calls `pl.pallas_call`.

    Parameters
    ----------
    pallas_fn : callable
        The wrapped callable returned by `pl.pallas_call(...)`, or any
        function that invokes exactly one pallas_call.
    *example_args
        Arrays or `jax.ShapeDtypeStruct`s fixing shapes and dtypes; no
        data is read.

    Returns
    -------
    KernelSpec

    Raises
    ------
    TraceError
        If tracing finds zero or more than one pallas_call equation, for
        PrefetchScalarGridSpec kernels, or when the kernel body carries
        effects palladium cannot perform on the GPU (debug prints, callbacks).
    """
    closed = jax.make_jaxpr(pallas_fn)(*example_args)
    eqns = [e for e in closed.jaxpr.eqns if e.primitive.name == "pallas_call"]
    if not eqns:
        raise TraceError(
            "no pallas_call equation found; pass the callable returned by "
            "pl.pallas_call, or a function that invokes one"
        )
    if len(eqns) > 1:
        raise TraceError(
            f"found {len(eqns)} pallas_call equations; palladium handles one "
            "kernel at a time, trace them separately"
        )
    eqn = eqns[0]
    params = dict(eqn.params)
    grid_mapping = params["grid_mapping"]

    kernel_jaxpr = params["jaxpr"]
    if hasattr(kernel_jaxpr, "jaxpr"):
        # ClosedJaxpr on some JAX versions, bare Jaxpr on others.
        kernel_jaxpr = kernel_jaxpr.jaxpr

    foreign = effects.foreign_effects(kernel_jaxpr)
    if foreign:
        names = ", ".join(sorted({type(e).__name__ for e in foreign}))
        raise TraceError(
            f"kernel body has effects palladium cannot perform on the GPU: "
            f"{names}. Only Ref reads and writes are supported; remove "
            "debug prints, callbacks, and other host-side effects from the "
            "kernel."
        )

    if grid_mapping.num_index_operands:
        raise TraceError("PrefetchScalarGridSpec is not supported")

    n_in, n_out = grid_mapping.num_inputs, grid_mapping.num_outputs
    scratch_bufs = grid_mapping.scratch_avals
    mappings = list(grid_mapping.block_mappings)

    grid = tuple(int(g) for g in grid_mapping.grid)
    if not grid:
        # Gridless pallas_call: a single program instance.
        grid = (1,)

    inputs = _block_infos(mappings[:n_in])
    outputs = _block_infos(mappings[n_in : n_in + n_out])
    aliases = tuple(
        (int(i), int(j)) for i, j in params.get("input_output_aliases") or ()
    )
    if aliases:
        _validate_aliases(kernel_jaxpr, aliases, inputs, outputs)
    _validate_parallel_writes(kernel_jaxpr, outputs, grid, n_in)

    return KernelSpec(
        name=params.get("name") or "palladium_kernel",
        jaxpr=kernel_jaxpr,
        grid=grid,
        inputs=inputs,
        outputs=outputs,
        scratch=tuple(_scratch_infos(scratch_bufs)),
        raw_params=params,
        aliases=aliases,
    )
