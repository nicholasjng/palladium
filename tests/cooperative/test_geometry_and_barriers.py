"""Threadgroup geometry, lane indices, and convergent barriers."""

import jax
import metal_runtime as mr
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium
from palladium.diagnostics import normalize_threadgroup


@pytest.fixture
def metal_device():
    try:
        mr.device_name()
    except mr.DeviceError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("factory", [palladium.metal_call, palladium.metal_call_jit])
def test_thread_indices_without_scratch_2d(metal_device, factory):
    def kernel(out, sizes):
        out[0, 0] = palladium.thread_index()
        sizes[0, 0] = palladium.threads_per_threadgroup()

    shape = (7, 5)
    spec = pl.BlockSpec((1, 1), lambda i, j: (i, j))
    out = jax.ShapeDtypeStruct(shape, np.int32)
    call = factory(
        kernel, grid=shape, threadgroup=(4, 3), out_specs=(spec, spec), out_shape=(out, out)
    )
    actual, sizes = call()
    expected = np.empty(shape, np.int32)
    expected_sizes = np.empty(shape, np.int32)
    for x in range(7):
        for y in range(5):
            nx, ny = min(4, 7 - x // 4 * 4), min(3, 5 - y // 3 * 3)
            expected[x, y] = x % 4 + nx * (y % 3)
            expected_sizes[x, y] = nx * ny
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(sizes, expected_sizes)


@pytest.mark.parametrize("value", [0, -1, (), (2, 0), (1, 1, 1, 1), (1.5,), "bad"])
def test_invalid_geometry(value):
    with pytest.raises(ValueError):
        normalize_threadgroup(value)


def test_nested_lane_dependent_barrier_rejected():
    def kernel(out):
        @jax.jit
        def sync():
            jax.lax.cond(palladium.thread_index() == 0, lambda: palladium.barrier(), lambda: None)

        sync()
        out[0] = 1

    call = palladium.metal_call(
        kernel, threadgroup=4, out_shape=jax.ShapeDtypeStruct((1,), np.int32)
    )
    with pytest.raises(palladium.TraceError, match="vary across"):
        call.explain()


def test_barrier_requires_explicit_geometry_without_scratch():
    def kernel(out):
        palladium.barrier()
        out[0] = 1

    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((1,), np.int32))
    with pytest.raises(palladium.EmitError, match="explicit threadgroup"):
        call.explain()


def test_lane_scratch_extent_checked_without_device():
    def kernel(out, scratch):
        scratch[palladium.thread_index()] = 1
        out[0] = 1

    spec = pl.BlockSpec((1,), lambda i: (i,))
    call = palladium.metal_call(
        kernel,
        grid=(32,),
        threadgroup=32,
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((32,), np.int32),
        scratch_shapes=[palladium.threadgroup_memory((16,), np.int32)],
    )
    with pytest.raises(palladium.EmitError, match="scratch dimension"):
        call.explain()


def test_uniform_barriers_in_static_loop(metal_device):
    def kernel(out, scratch):
        t = palladium.thread_index()
        scratch[t] = 0

        def body(_, carry):
            scratch[t] = scratch[t] + 1
            palladium.barrier()
            return carry

        jax.lax.fori_loop(0, 4, body, 0)
        out[0] = scratch[(t + 1) % palladium.threads_per_threadgroup()]

    call = palladium.metal_call(
        kernel,
        grid=(35,),
        threadgroup=8,
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((35,), np.int32),
        scratch_shapes=[palladium.threadgroup_memory((8,), np.int32)],
    )
    np.testing.assert_array_equal(call(), np.full(35, 4, np.int32))


def test_jitted_lane_predicate_barrier_rejected():
    @jax.jit
    def lane_predicate():
        return palladium.thread_index() == 0

    def kernel(out):
        jax.lax.cond(lane_predicate(), lambda: palladium.barrier(), lambda: None)
        out[0] = 1

    call = palladium.metal_call(
        kernel, threadgroup=4, out_shape=jax.ShapeDtypeStruct((1,), np.int32)
    )
    with pytest.raises(palladium.TraceError, match="vary across"):
        call.explain()


def test_thread_indices_3d(metal_device):
    def kernel(out):
        out[0, 0, 0] = palladium.thread_index()

    shape, group = (5, 3, 3), (4, 2, 2)
    call = palladium.metal_call(
        kernel,
        grid=shape,
        threadgroup=group,
        out_specs=pl.BlockSpec((1, 1, 1), lambda i, j, k: (i, j, k)),
        out_shape=jax.ShapeDtypeStruct(shape, np.int32),
    )
    expected = np.empty(shape, np.int32)
    for start in np.ndindex(2, 2, 2):
        origin = tuple(s * g for s, g in zip(start, group))
        extent = tuple(min(g, n - o) for g, n, o in zip(group, shape, origin))
        slices = tuple(slice(o, o + e) for o, e in zip(origin, extent))
        expected[slices] = np.arange(np.prod(extent)).reshape(extent, order="F")
    np.testing.assert_array_equal(call(), expected)
