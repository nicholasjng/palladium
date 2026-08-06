"""Step 1 of the pipeline: get the kernel jaxpr out of Pallas.

`pl.pallas_call` does not run the kernel; it stages it into a single
`pallas_call` equation whose params carry the kernel body (a stateful jaxpr
over `Ref`s) and the grid/block structure. We trace the *wrapped* call with
`jax.make_jaxpr`, find that equation, and repackage the parts the emitter
needs into a `KernelSpec`.

This module is provided (not an exercise): it is JAX-version-sensitive
plumbing, and the interesting compiler work lives downstream in `emit.py`.
Verified against JAX 0.11.0; the version-sensitive bits are the
`grid_mapping` / `block_mapping` dataclass fields, see `_block_infos`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr

__all__ = ["BlockInfo", "KernelSpec", "trace"]


@dataclasses.dataclass(frozen=True)
class BlockInfo:
    """One kernel operand: the block a single program instance sees."""

    block_shape: tuple[int, ...]
    array_shape: tuple[int, ...]
    dtype: np.dtype
    # Jaxpr mapping grid indices -> block index (in units of blocks), i.e.
    # the staged form of the BlockSpec index_map.
    index_map_jaxpr: ClosedJaxpr


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """Everything the emitter needs, and nothing it doesn't."""

    name: str
    jaxpr: Jaxpr  # the kernel body: a stateful Jaxpr over Refs
    grid: tuple[int, ...]
    inputs: tuple[BlockInfo, ...]
    outputs: tuple[BlockInfo, ...]
    # The full, unprocessed pallas_call params. Later exercises (scratch
    # shapes, dimension semantics) reach in here rather than growing this
    # dataclass speculatively.
    raw_params: dict[str, Any]

    @property
    def num_programs(self) -> int:
        n = 1
        for g in self.grid:
            n *= g
        return n


def _block_dim(dim: Any) -> int | None:
    """Normalize one block_shape entry across JAX's dim-semantics types.

    JAX 0.11 stages BlockSpec shapes as Blocked(block_size=N) / Element /
    Squeezed objects rather than plain ints. Squeezed dims vanish from the
    kernel-visible block; everything else carries a block size.
    """
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if hasattr(dim, "block_size"):
        return int(dim.block_size)
    if type(dim).__name__ == "Squeezed":
        return None
    raise NotImplementedError(f"unhandled block dim type: {dim!r}")


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


def trace(pallas_fn: Callable, *example_args) -> KernelSpec:
    """Extract a KernelSpec from a function that calls `pl.pallas_call`.

    `pallas_fn` is the *wrapped* callable returned by `pl.pallas_call(...)`
    (or any function that invokes exactly one pallas_call); `example_args`
    are arrays (or ShapeDtypeStructs) fixing shapes and dtypes.
    """
    closed = jax.make_jaxpr(pallas_fn)(*example_args)
    eqns = [e for e in closed.jaxpr.eqns if e.primitive.name == "pallas_call"]
    if not eqns:
        raise ValueError(
            "no pallas_call equation found; pass the callable returned by "
            "pl.pallas_call, or a function that invokes one"
        )
    if len(eqns) > 1:
        raise ValueError(
            f"found {len(eqns)} pallas_call equations; palladium handles one "
            "kernel at a time — trace them separately"
        )
    eqn = eqns[0]
    params = dict(eqn.params)
    grid_mapping = params["grid_mapping"]

    kernel_jaxpr = params["jaxpr"]
    if hasattr(kernel_jaxpr, "jaxpr"):  # ClosedJaxpr on some versions
        kernel_jaxpr = kernel_jaxpr.jaxpr

    if grid_mapping.num_index_operands:
        raise NotImplementedError("PrefetchScalarGridSpec is not supported")

    n_in, n_out = grid_mapping.num_inputs, grid_mapping.num_outputs
    mappings = list(grid_mapping.block_mappings)

    grid = tuple(int(g) for g in grid_mapping.grid)
    if not grid:
        grid = (1,)  # gridless pallas_call: a single program instance

    return KernelSpec(
        name=params.get("name") or "palladium_kernel",
        jaxpr=kernel_jaxpr,
        grid=grid,
        inputs=_block_infos(mappings[:n_in]),
        outputs=_block_infos(mappings[n_in : n_in + n_out]),
        raw_params=params,
    )
