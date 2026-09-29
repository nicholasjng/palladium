"""Threadgroup-shared scratch and the cooperative primitives around it.

Ordinary `scratch_shapes` entries are `thread`-space, private to one Metal
thread. `threadgroup_memory` requests storage in Metal's `threadgroup`
address space, visible to every thread in the group; `barrier`,
`thread_index`, and `threads_per_threadgroup` are the primitives for using
it. Barriers are placed by the kernel author, never inferred.

Pallas's `MemorySpace` enum has no "threadgroup" member, so a sentinel
rides through tracing on `MemoryRef.memory_space` (typed `Any` upstream).

Under `interpret=True`, `thread_index()` is 0, `threads_per_threadgroup()`
is 1, `barrier()` is a no-op, and every instance is a threadgroup of one.
A kernel that reduces across threads computes something else there;
validate it against a NumPy reference instead.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax._src import core as jax_core, effects
from jax._src.interpreters import mlir
from jax.experimental import pallas as pl

from palladium.effects import GpuNativeEffect

__all__ = [
    "barrier",
    "thread_index",
    "threadgroup_memory",
    "threads_per_threadgroup",
]


class _ThreadgroupSpace:
    """Sentinel marking a scratch request as `threadgroup`-space; carried on
    `MemoryRef.memory_space` and recovered by `palladium.trace`."""

    def __repr__(self) -> str:
        return "threadgroup"


THREADGROUP = _ThreadgroupSpace()


def threadgroup_memory(shape: tuple[int, ...], dtype: Any) -> pl.MemoryRef:
    """Request a `threadgroup`-space scratch Ref, shared by the whole group.

    Pass the result in `scratch_shapes=`; the kernel receives it as a
    trailing Ref argument. The allocation is compile-time sized and shared,
    not per-thread: a `(64,)` request is 64 elements for the whole group.
    Indexing it by `thread_index()` is safe only when the threadgroup size
    does not exceed the leading extent, so `bind` requires an explicit
    `threadgroup=` for these kernels.

    Parameters
    ----------
    shape : tuple of int
        Shared array shape.
    dtype : dtype-like
        Element type.
    """
    aval = jax_core.ShapedArray(tuple(int(d) for d in shape), np.dtype(dtype))
    return pl.MemoryRef(aval, THREADGROUP)


class _ThreadgroupEffect(GpuNativeEffect):
    """Marks the cooperative primitives as effectful; without it JAX's DCE
    drops the zero-output `barrier()` before palladium sees it."""


_EFFECT = _ThreadgroupEffect()

# Allow the effect through control flow so Pallas interpret can carry it
# through its while loop.
for _set in (
    effects.control_flow_allowed_effects,
    effects.lowerable_effects,
    effects.partial_eval_kept_effects,
):
    _set.add_type(_ThreadgroupEffect)


barrier_p = jax_core.Primitive("palladium_barrier")
barrier_p.multiple_results = True
barrier_p.def_effectful_abstract_eval(lambda **_: ([], {_EFFECT}))
barrier_p.def_impl(lambda **_: [])
mlir.register_lowering(barrier_p, lambda ctx, **_: [])


def barrier() -> None:
    """Synchronize every thread in the threadgroup; lowers to
    `threadgroup_barrier(mem_flags::mem_threadgroup)`. Place one between a
    write to `threadgroup_memory` and any read of another thread's slot.
    A no-op under `interpret=True`."""
    barrier_p.bind()


thread_index_p = jax_core.Primitive("palladium_thread_index")
thread_index_p.def_effectful_abstract_eval(
    lambda **_: (jax_core.ShapedArray((), np.dtype(np.int32)), {_EFFECT})
)
thread_index_p.def_impl(lambda **_: np.int32(0))
mlir.register_lowering(thread_index_p, mlir.lower_fun(lambda: np.int32(0), multiple_results=False))


def thread_index() -> jax.Array:
    """This thread's linear index within its threadgroup, as int32:
    `[[thread_position_in_threadgroup]]` linearized x-fastest over the
    actual group dimensions, including partial groups. Distinct from
    `pl.program_id`, the position in the whole grid. Always 0 under
    `interpret=True`."""
    return thread_index_p.bind()


threads_per_threadgroup_p = jax_core.Primitive("palladium_threads_per_threadgroup")
threads_per_threadgroup_p.def_effectful_abstract_eval(
    lambda **_: (jax_core.ShapedArray((), np.dtype(np.int32)), {_EFFECT})
)
threads_per_threadgroup_p.def_impl(lambda **_: np.int32(1))
mlir.register_lowering(
    threads_per_threadgroup_p,
    mlir.lower_fun(lambda: np.int32(1), multiple_results=False),
)


def threads_per_threadgroup() -> jax.Array:
    """Threads in this threadgroup, as int32: the product of
    `[[threads_per_threadgroup]]`. Use it as a cooperative reduction's
    bound rather than a constant: Metal dispatches non-uniform
    threadgroups, so a grid that is not a multiple of the group size ends
    in a smaller group, and a constant would fold in slots no thread
    wrote. Always 1 under `interpret=True`."""
    return threads_per_threadgroup_p.bind()
