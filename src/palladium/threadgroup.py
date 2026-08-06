"""Threadgroup-shared scratch and the cooperative primitives around it.

Part A's `scratch_shapes` entries are `thread`-space: private to one
Metal thread, one program instance. This module adds the shared tier --
storage in Metal's `threadgroup` address space, visible to every thread
in the same threadgroup, plus the barrier and thread-position primitives
a cooperative kernel needs to use it safely.

Pallas's own `MemorySpace` enum has no "threadgroup" member, and
hijacking one of its members to mean something Metal-specific would be
misleading on every other backend. `MemoryRef`'s `memory_space` field is
typed `Any`, though, so a palladium-specific sentinel rides through
tracing untouched and comes back out on `grid_mapping.scratch_avals`.

Barriers are placed by the kernel author, never inferred. Auto-inserting
them from a write-then-read hazard analysis is real analysis work and
easy to get subtly wrong; this is the same division of responsibility as
hand-written Metal or CUDA.

Interpret-mode caveat
---------------------
`interpret=True` runs program instances sequentially with no notion of a
threadgroup, so it cannot simulate cross-thread communication. Under
interpret, `thread_index()` is 0, `threads_per_threadgroup()` is 1, and
`barrier()` is a no-op: every instance is modelled as a threadgroup of
one. A kernel that genuinely reduces across threads therefore computes
something *different* under interpret, and the interpret oracle is not a
correctness check for it. Validate cooperative kernels against a NumPy
reference instead -- `tests/test_19_threadgroup.py` does exactly that.
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

    Carried through Pallas tracing on `MemoryRef.memory_space` (typed
    `Any` upstream) and recovered by `palladium.trace`. Deliberately not
    a `pl.MemorySpace` member: this is a Metal fact, not a Pallas one.
    """

    def __repr__(self) -> str:
        return "threadgroup"


THREADGROUP = _ThreadgroupSpace()


def threadgroup_memory(shape: tuple[int, ...], dtype: Any) -> pl.MemoryRef:
    """Request a `threadgroup`-space scratch Ref, shared by the whole group.

    Pass the result in `scratch_shapes=`, exactly like an ordinary
    `pl.MemorySpace.ANY(...)` request; the kernel receives it as a
    trailing Ref argument.

    The allocation is sized at compile time from `shape` and is shared,
    not per-thread: a `(64,)` request is 64 elements for the entire
    threadgroup, however many threads that is. Indexing it by
    `thread_index()` is therefore only safe when the dispatch threadgroup
    size does not exceed the leading extent -- see `palladium.bind`,
    which requires an explicit `threadgroup=` for kernels that use this.

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

    Load-bearing, not decorative: a zero-output primitive with no
    declared effect is dead code, and JAX's DCE removes `barrier()` from
    the kernel jaxpr entirely before palladium ever sees it. A silently
    vanishing barrier is a data race, so the effect is what keeps it
    alive.
    """


_EFFECT = _ThreadgroupEffect()

# Pallas's interpret mode runs the grid as a `lax.while_loop` and lowers
# the kernel body to MLIR for CPU, so the effect has to be permitted in
# control flow and the primitives need a (no-op) CPU lowering. Without
# the control_flow registration interpret raises "Effects not supported
# in `while`"; without the lowering it raises "MLIR translation rule not
# found for platform cpu".
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

    Lowers verbatim to `threadgroup_barrier(mem_flags::mem_threadgroup)`.
    Place one between a write to `threadgroup_memory` and any read of
    another thread's slot, the same as in hand-written Metal.

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

    Lowers to `[[threads_per_threadgroup]]`. Use it as the bound of a
    cooperative reduction loop rather than a compile-time constant:
    Metal dispatches non-uniform threadgroups, so when the grid is not a
    multiple of the threadgroup size the final group is smaller and this
    reports its true size. A baked-in constant would fold in slots no
    thread ever wrote.

    Under `interpret=True` this is always 1 (see the module docstring).
    """
    return threads_per_threadgroup_p.bind()
