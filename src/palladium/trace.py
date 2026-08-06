"""Step 1 of the pipeline: extract the kernel jaxpr from Pallas.

Traces the wrapped `pl.pallas_call` with `jax.make_jaxpr` and repackages
the resulting `pallas_call` equation into a KernelSpec.

Pinned to JAX 0.11: `grid_mapping`/`block_mapping` dataclass fields are
version-sensitive (see `_block_infos`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr

from palladium.errors import TraceError

__all__ = ["BlockInfo", "KernelSpec", "trace"]


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
    raw_params : dict
        Full, unprocessed pallas_call params.
    """

    name: str
    jaxpr: Jaxpr
    grid: tuple[int, ...]
    inputs: tuple[BlockInfo, ...]
    outputs: tuple[BlockInfo, ...]
    raw_params: dict[str, Any]

    @property
    def num_programs(self) -> int:
        """Total number of program instances, one Metal thread each."""
        n = 1
        for g in self.grid:
            n *= g
        return n


def _block_dim(dim: Any) -> int | None:
    # JAX 0.11 stages BlockSpec shapes as BlockDim objects rather than
    # plain ints; squeezed dims vanish from the visible block. Matched by
    # class name, not attribute: Element, Indirect, and BoundedSlice dims
    # also carry `block_size` but mean different indexing semantics, and
    # accepting them here would lower wrong offsets silently.
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    kind = type(dim).__name__
    if kind == "Squeezed":
        return None
    if kind == "Blocked":
        return int(dim.block_size)
    raise TraceError(
        f"unsupported block dim type {kind}: only int, pl.Blocked, and "
        "pl.Squeezed dims are lowered (pl.Element, pl.Indirect, and "
        "pl.BoundedSlice indexing is unimplemented)"
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
        If tracing finds zero or more than one pallas_call equation, or
        for PrefetchScalarGridSpec kernels.
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

    if grid_mapping.num_index_operands:
        raise TraceError("PrefetchScalarGridSpec is not supported")

    n_in, n_out = grid_mapping.num_inputs, grid_mapping.num_outputs
    mappings = list(grid_mapping.block_mappings)

    grid = tuple(int(g) for g in grid_mapping.grid)
    if not grid:
        # Gridless pallas_call: a single program instance.
        grid = (1,)

    return KernelSpec(
        name=params.get("name") or "palladium_kernel",
        jaxpr=kernel_jaxpr,
        grid=grid,
        inputs=_block_infos(mappings[:n_in]),
        outputs=_block_infos(mappings[n_in : n_in + n_out]),
        raw_params=params,
    )
