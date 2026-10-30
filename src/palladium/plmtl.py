"""Metal kernels with explicit threadgroup memory, written as plain Pallas.

`kernel` mirrors `jax.experimental.pallas.kernel` on a `Mesh` of
threadgroups. The body sees whole device arrays and `THREADGROUP` scratch, and
stages tiles with ordinary ref reads and writes; `jnp.dot` of two threadgroup
tiles runs on SIMD-group matrix fragments, and palladium places the barriers.
Use `pl.program_id` for the mesh coordinates: the Pallas interpreter, which
`.interpret` runs, does not bind mesh axis names for `lax.axis_index`.

Example::

    def matmul(a_ref, b_ref, o_ref, a_tile, b_tile):
        i, j = pl.program_id(0), pl.program_id(1)

        def step(kb, acc):
            a_tile[...] = a_ref[pl.ds(i * 64, 64), pl.ds(kb * 16, 16)]
            b_tile[...] = b_ref[pl.ds(kb * 16, 16), pl.ds(j * 64, 64)]
            return acc + jnp.dot(a_tile[...], b_tile[...])

        acc = lax.fori_loop(0, k // 16, step, jnp.zeros((64, 64), jnp.float32))
        o_ref[pl.ds(i * 64, 64), pl.ds(j * 64, 64)] = acc

    call = plmtl.kernel(
        matmul,
        out_type=jax.ShapeDtypeStruct((m, n), jnp.float32),
        mesh=plmtl.Mesh((m // 64, n // 64)),
        scratch_types=[plmtl.THREADGROUP((64, 16), jnp.float32),
                       plmtl.THREADGROUP((16, 64), jnp.float32)],
    )
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
from collections.abc import Callable, Generator, Sequence
from typing import Any

import jax.numpy as jnp
from jax._src.pallas import core as pallas_core
from jax.core import ShapedArray
from jax.experimental import pallas as pl

from palladium.ffi import FfiCallable
from palladium.launch import CompilerParams

__all__ = ["THREADGROUP", "MemorySpace", "Mesh", "kernel"]


class MemorySpace(enum.Enum):
    """Where a scratch ref lives; whole device arrays use Pallas's default space."""

    #: Shared by the threads of one threadgroup, as staged tiles.
    THREADGROUP = "threadgroup"

    def __str__(self) -> str:
        return self.value

    def __call__(self, shape: Sequence[int], dtype: Any) -> pallas_core.MemoryRef:
        return pallas_core.MemoryRef(ShapedArray(tuple(shape), jnp.dtype(dtype)), self)


THREADGROUP = MemorySpace.THREADGROUP


@dataclasses.dataclass(frozen=True)
class Mesh:
    """A grid of threadgroups, each `simdgroups` SIMD groups of 32 threads.

    Parameters
    ----------
    grid : tuple of int
        Threadgroups per axis, at most three axes; `pl.program_id(axis)` reads
        a program's coordinate.
    simdgroups : int, optional
        SIMD groups per threadgroup; 4 by default, the only count supported
        so far.
    """

    grid: tuple[int, ...]
    simdgroups: int = 4

    @property
    def shape(self) -> collections.OrderedDict[object, int]:
        return collections.OrderedDict(zip(("x", "y", "z"), self.grid, strict=False))

    @property
    def default_memory_space(self) -> pallas_core.MemorySpace:
        return pallas_core.MemorySpace.ANY

    @property
    def core_type(self) -> str:
        return "metal"

    def discharges_effect(self, effect: Any) -> bool:
        return False

    def check_is_compatible_with(self, other_mesh: Any) -> None:
        if other_mesh != self:
            raise ValueError(f"Mesh {self} is not compatible with {other_mesh}")

    @property
    def supported_memory_spaces(self) -> Sequence[Any]:
        return [pallas_core.MemorySpace.ANY, *MemorySpace]

    @contextlib.contextmanager
    def tracing_context(self) -> Generator[None]:
        with pallas_core.tracing_grid_env(tuple(self.grid), mapped_dims=()):
            yield


def kernel(
    body: Callable,
    out_type: Any,
    *,
    mesh: Mesh,
    scratch_types: Any = (),
    compiler_params: CompilerParams | None = None,
    name: str | None = None,
) -> FfiCallable:
    """Create a Metal kernel over `mesh`, dispatched from the JAX CPU backend.

    Parameters
    ----------
    body : callable
        Kernel body on input, output and scratch refs, as for `pl.kernel`.
    out_type : jax.ShapeDtypeStruct or pytree of them
        Output shapes and dtypes.
    mesh : Mesh
        Threadgroup grid.
    scratch_types : pytree of MemoryRef, optional
        Scratch refs, typically `THREADGROUP(shape, dtype)` tiles.
    compiler_params : CompilerParams or None, optional
        Math mode; the other options do not apply to mesh kernels.
    name : str or None, optional
        Kernel name in diagnostics and the MSL source.

    Returns
    -------
    FfiCallable
        Callable on NumPy or JAX arrays, like `palladium.metal_call`; its
        `interpret` attribute runs the same kernel in the Pallas interpreter.
    """

    def build(interpret: bool) -> Callable:
        return pl.kernel(
            body,
            out_type=out_type,
            mesh=mesh,
            scratch_types=scratch_types,
            interpret=interpret,
            name=name,
        )

    return FfiCallable(build(False), build(True), compiler_params)
