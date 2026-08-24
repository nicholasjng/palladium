"""BoundKernel.launch()/PendingResult: the non-blocking dispatch path.

Builds a KernelSpec by hand (no jax tracing needed -- launch()/__call__ only
read spec.grid/inputs/outputs) so this runs independently of the emitter.
"""

from typing import Any, cast

import numpy as np

from palladium.dispatch import bind
from palladium.trace import BlockInfo, KernelSpec

_INC_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void inc(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                uint tid [[thread_position_in_grid]]) {
    o[tid] = x[tid] + 1.0f;
}
"""


def _make_bound(n: int, **bind_kwargs):
    # BoundKernel never reads jaxpr/index_map_jaxpr, only grid/inputs/outputs;
    # cast rather than build real jaxprs this test has no use for.
    no_jaxpr = cast(Any, None)
    info_in = BlockInfo((n,), (n,), np.dtype(np.float32), no_jaxpr)
    info_out = BlockInfo((n,), (n,), np.dtype(np.float32), no_jaxpr)
    spec = KernelSpec(
        name="inc",
        jaxpr=no_jaxpr,
        grid=(n,),
        inputs=(info_in,),
        outputs=(info_out,),
        raw_params={},
    )
    return bind(spec, _INC_SOURCE, **bind_kwargs)


def test_launch_wait_matches_call(rng):
    n = 256
    bound = _make_bound(n)
    x = rng.standard_normal(n).astype(np.float32)

    via_call = bound(x)
    via_launch = bound.launch(x).wait()

    np.testing.assert_allclose(via_launch, via_call)
    np.testing.assert_allclose(via_call, x + 1.0)


def test_stepping_loop_via_launch_matches_repeated_call(rng):
    """The launch()-then-wait()-then-launch() pattern the docstring
    describes must produce the same trajectory as plain repeated calls."""
    n = 64
    bound_a = _make_bound(n)
    bound_b = _make_bound(n)
    x0 = rng.standard_normal(n).astype(np.float32)

    x = x0.copy()
    for _ in range(5):
        x = bound_a(x)

    pending = bound_b.launch(x0)
    for _ in range(4):
        x_step = pending.wait()
        pending = bound_b.launch(x_step)
    y = pending.wait()

    np.testing.assert_allclose(y, x)


def test_second_launch_before_wait_does_not_corrupt_the_first(rng):
    """copy_from() reusing an input buffer must not race a still-in-flight
    read from an earlier, not-yet-waited-on launch."""
    n = 1 << 16  # large enough that the kernel is still running when we re-launch
    bound = _make_bound(n)
    x0 = rng.standard_normal(n).astype(np.float32)
    x1 = rng.standard_normal(n).astype(np.float32)

    first = bound.launch(x0)
    second = bound.launch(x1)  # must not corrupt `first`'s in-flight read of x0

    np.testing.assert_allclose(first.wait(), x0 + 1.0)
    np.testing.assert_allclose(second.wait(), x1 + 1.0)


def test_launch_stream_deeper_than_the_ring_stays_correct(rng):
    """Many launches with all waits deferred: slot reuse under backpressure
    must keep every launch's inputs and outputs paired correctly."""
    n = 1 << 14
    depth = 4
    launches = 4 * depth + 3  # several full ring wraps, ending mid-ring
    bound = _make_bound(n, pipeline_depth=depth)
    xs = [rng.standard_normal(n).astype(np.float32) for _ in range(launches)]

    pendings = [bound.launch(x) for x in xs]

    for x, pending in zip(xs, pendings, strict=True):
        np.testing.assert_allclose(pending.wait(), x + 1.0)


def test_synchronous_callers_stay_at_one_slot(rng):
    """launch-then-wait callers must not pay pipeline_depth times the
    input memory: a finished slot is reused, not grown past."""
    n = 256
    bound = _make_bound(n)
    for _ in range(2 * bound.pipeline_depth):
        bound(rng.standard_normal(n).astype(np.float32))
    assert len(bound._slots) == 1


def test_ring_grows_only_while_launches_overlap(rng):
    n = 1 << 16
    bound = _make_bound(n, pipeline_depth=3)
    xs = [rng.standard_normal(n).astype(np.float32) for _ in range(5)]

    pendings = [bound.launch(x) for x in xs]
    assert len(bound._slots) <= 3
    for x, pending in zip(xs, pendings, strict=True):
        np.testing.assert_allclose(pending.wait(), x + 1.0)


def test_pinned_inputs_survive_later_launches(rng):
    """pinned() detaches its input buffers from the slot ring, so later
    launch() traffic on the same BoundKernel can't overwrite them."""
    n = 256
    bound = _make_bound(n)
    x_pinned = rng.standard_normal(n).astype(np.float32)
    call = bound.pinned(x_pinned)

    np.testing.assert_allclose(call(), x_pinned + 1.0)
    for _ in range(3):
        other = rng.standard_normal(n).astype(np.float32)
        np.testing.assert_allclose(bound(other), other + 1.0)
    np.testing.assert_allclose(call(), x_pinned + 1.0)
