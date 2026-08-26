"""Threadgroup-shared scratch and the cooperative primitives around it.

Ordinary `scratch_shapes` entries are `thread`-space: private to one
Metal thread. This module adds storage in Metal's `threadgroup` address
space, visible to every thread in the same threadgroup, plus the barrier
and thread-position primitives needed to use it safely.

Pallas's `MemorySpace` enum has no "threadgroup" member, so a
palladium-specific sentinel rides through tracing on `MemoryRef`'s
`memory_space` field (typed `Any` upstream) and comes back out on
`grid_mapping.scratch_avals`.

Barriers are placed by the kernel author, never inferred.

Interpret-mode caveat
---------------------
`interpret=True` runs program instances sequentially with no notion of a
threadgroup: `thread_index()` is 0, `threads_per_threadgroup()` is 1,
`barrier()` is a no-op, and every instance is modelled as a threadgroup
of one. A kernel that reduces across threads therefore computes
something *different* under interpret, so the oracle is not a
correctness check for it; validate against a NumPy reference instead.
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
    """Sentinel marking a scratch request as `threadgroup`-space.

    Carried on `MemoryRef.memory_space` and recovered by
    `palladium.trace`. Not a `pl.MemorySpace` member: a Metal fact, not
    a Pallas one.
    """

    def __repr__(self) -> str:
        return "threadgroup"


THREADGROUP = _ThreadgroupSpace()


def threadgroup_memory(shape: tuple[int, ...], dtype: Any) -> pl.MemoryRef:
    """Request a `threadgroup`-space scratch Ref, shared by the whole group.

    Pass the result in `scratch_shapes=`, like a `pl.MemorySpace.ANY(...)`
    request; the kernel receives it as a trailing Ref argument.

    The allocation is sized at compile time and shared, not per-thread: a
    `(64,)` request is 64 elements for the whole threadgroup. Indexing it
    by `thread_index()` is therefore safe only when the dispatch
    threadgroup size does not exceed the leading extent, which is why
    `bind` requires an explicit `threadgroup=` for these kernels.

    Parameters
    ----------
    shape : tuple of int
        Shared array shape.
    dtype : dtype-like
        Element type.

    Examples
    --------
    >>> scratch_shapes=[palladium.threadgroup_memory((32,), jnp.float32)]  # doctest: +SKIP
    """
    aval = jax_core.ShapedArray(tuple(int(d) for d in shape), np.dtype(dtype))
    return pl.MemoryRef(aval, THREADGROUP)


class _ThreadgroupEffect(GpuNativeEffect):
    """Marks the cooperative primitives as effectful.

    Load-bearing: a zero-output primitive with no declared effect is dead
    code, and JAX's DCE would drop `barrier()` from the jaxpr before
    palladium sees it.
    """


_EFFECT = _ThreadgroupEffect()

# Interpret mode runs the grid as a `lax.while_loop` and lowers the body
# to MLIR for CPU: without the control_flow registration it raises
# "Effects not supported in `while`", without the no-op lowering below,
# "MLIR translation rule not found for platform cpu".
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
    """Synchronize every thread in the threadgroup.

    Lowers to `threadgroup_barrier(mem_flags::mem_threadgroup)`. Place one
    between a write to `threadgroup_memory` and any read of another
    thread's slot.

    Under `interpret=True` this is a no-op (see the module docstring).
    """
    barrier_p.bind()


thread_index_p = jax_core.Primitive("palladium_thread_index")
thread_index_p.def_effectful_abstract_eval(
    lambda **_: (jax_core.ShapedArray((), np.dtype(np.int32)), {_EFFECT})
)
thread_index_p.def_impl(lambda **_: np.int32(0))
mlir.register_lowering(
    thread_index_p, mlir.lower_fun(lambda: np.int32(0), multiple_results=False)
)


def thread_index() -> jax.Array:
    """This thread's linear index within its threadgroup, as int32.

    Lowers to `[[thread_position_in_threadgroup]]`. Distinct from
    `pl.program_id`, which is the position in the whole grid.

    Under `interpret=True` this is always 0 (see the module docstring).
    """
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
    """How many threads are in *this* threadgroup, as int32.

    Lowers to `[[threads_per_threadgroup]]`. Use it as a cooperative
    reduction's bound rather than a compile-time constant: Metal
    dispatches non-uniform threadgroups, so a grid that is not a multiple
    of the threadgroup size ends in a smaller group, and a baked-in
    constant would fold in slots no thread wrote.

    Under `interpret=True` this is always 1 (see the module docstring).
    """
    return threads_per_threadgroup_p.bind()
