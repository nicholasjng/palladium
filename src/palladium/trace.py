"""Extract the kernel jaxpr from Pallas: trace the wrapped `pl.pallas_call`
with `jax.make_jaxpr` and repackage the `pallas_call` equation into a
KernelSpec. Pinned to JAX 0.11: `grid_mapping`/`block_mapping` fields are
version-sensitive.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.experimental.pallas as pl
import numpy as np
from jax.extend.core import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
    jaxprs_in_params,
    subjaxprs,
)

from palladium import effects
from palladium.errors import TraceError

__all__ = [
    "BlockInfo",
    "KernelSpec",
    "ScratchInfo",
    "spec_from_params",
    "trace",
]


@dataclasses.dataclass(frozen=True)
class ScratchInfo:
    """A kernel scratch buffer with no backing caller array, private to one
    program instance.

    Attributes
    ----------
    shape : tuple of int
        Buffer shape.
    dtype : numpy.dtype
        Buffer element type.
    """

    shape: tuple[int, ...]
    dtype: np.dtype


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
    # Unsqueezed layout, preserving the original axis positions.
    full_block_shape: tuple[int | None, ...]


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """A traced pallas_call, reduced to what MSL emission needs.

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
    scratch : tuple of ScratchInfo
        Scratch buffer descriptions, in jaxpr order after the operands.
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
    aliases: tuple[tuple[int, int], ...] = ()


def _block_dim(dim: Any) -> int | None:
    # Squeezed dims vanish from the visible block. pl.Element, pl.Indirect,
    # and pl.BoundedSlice also carry `block_size` but index differently, so
    # accepting them would silently lower wrong offsets.
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
                full_block_shape=tuple(dims),
            )
        )
    return tuple(infos)


def _validate_aliases(
    jaxpr: Jaxpr,
    aliases: tuple[tuple[int, int], ...],
    inputs: tuple[BlockInfo, ...],
    outputs: tuple[BlockInfo, ...],
) -> None:
    """Sharing one buffer is transparent only when both refs slice it
    identically and every read of the input precedes any write of the
    output (Pallas keeps the input's pre-call values visible throughout).
    Order is checked on equation effects, which cover sub-jaxprs.
    """
    for i, j in aliases:
        a, b = inputs[i], outputs[j]
        if (
            a.array_shape != b.array_shape
            or a.dtype != b.dtype
            or a.block_shape != b.block_shape
            or a.full_block_shape != b.full_block_shape
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


def _depends_on(jaxpr: Jaxpr, seeds: set[Var], targets: Sequence[Var | Literal]) -> bool:
    """Whether any of `targets` depends on a var in `seeds` over top-level
    eqns. Coarse across sub-jaxprs: any tainted invar taints all outvars."""
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
    """Reject the provable write race: a grid axis that neither the output's
    index map nor any write index depends on. A write inside control flow
    counts as indexed by the axis when the control-flow equation takes an
    operand derived from `program_id(axis)` or calls it in its body.
    Non-injective maps that use the axis pass unchecked.
    """
    for j, info in enumerate(outputs):
        o_var = jaxpr.invars[n_in + j]
        writes = [e for e in jaxpr.eqns if effects.eqn_writes_ref(e, o_var)]
        for axis, extent in enumerate(grid):
            if not writes or extent <= 1 or _map_uses_axis(info.index_map_jaxpr, axis):
                continue
            pid_vars = {
                out
                for e in jaxpr.eqns
                if e.primitive.name == "program_id" and e.params["axis"] == axis
                for out in e.outvars
            }
            for eqn in writes:
                if eqn.primitive.name == "swap":
                    indexed = _depends_on(jaxpr, pid_vars, list(eqn.invars[2:]))
                else:
                    indexed = _depends_on(jaxpr, pid_vars, list(eqn.invars)) or any(
                        _calls_program_id(child, axis) for child in jaxprs_in_params(eqn.params)
                    )
                if not indexed:
                    raise TraceError(
                        f"output {j} is written identically by all "
                        f"{extent} program instances along grid axis "
                        f"{axis}: neither its BlockSpec index map nor the "
                        "write index depends on that axis, and instances "
                        "run as parallel threads, so the writes race. Make "
                        f"the output BlockSpec index map use grid axis "
                        f"{axis}, or index the write with pl.program_id({axis})"
                    )


def _calls_program_id(jaxpr: Jaxpr, axis: int) -> bool:
    return any(
        eqn.primitive.name == "program_id" and eqn.params["axis"] == axis for eqn in jaxpr.eqns
    ) or any(_calls_program_id(child, axis) for child in subjaxprs(jaxpr))


def _scratch_infos(scratch_avals: Any) -> list[ScratchInfo]:
    return [
        ScratchInfo(shape=tuple(int(d) for d in aval.shape), dtype=np.dtype(aval.dtype))
        for aval in scratch_avals
    ]


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
    eqns = _pallas_call_eqns(closed.jaxpr)
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
    return spec_from_params(eqns[0].params)


def spec_from_params(pallas_params: Mapping[str, Any]) -> KernelSpec:
    """Build a KernelSpec from a pallas_call equation's parameters, as found
    by `trace` or received by a lowering rule."""
    params = dict(pallas_params)
    grid_mapping = params["grid_mapping"]

    kernel_jaxpr = params["jaxpr"]

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
        grid = (1,)

    inputs = _block_infos(mappings[:n_in])
    outputs = _block_infos(mappings[n_in : n_in + n_out])
    aliases = tuple((int(i), int(j)) for i, j in params.get("input_output_aliases") or ())
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
        aliases=aliases,
    )


def _pallas_call_eqns(jaxpr: Jaxpr) -> list[JaxprEqn]:
    """pallas_call equations in `jaxpr`, looking through jit wrappers."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            found.append(eqn)
        elif eqn.primitive.name == "jit":
            found.extend(_pallas_call_eqns(eqn.params["jaxpr"].jaxpr))
    return found
