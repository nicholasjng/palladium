"""Block layouts, indexed views, edge masks, and strided memory access."""

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium


@pytest.fixture
def metal_device():
    try:
        mr.device_name()
    except mr.DeviceError as exc:
        pytest.skip(str(exc))


def copy_kernel(x, out):
    out[...] = x[...] * 2


@pytest.mark.parametrize(
    "shape,block",
    [((3,), (8,)), ((17,), (8,)), ((9, 13), (4, 8)), ((3, 5), (8, 8)), ((8, 12), (2, 3))],
)
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("factory", [palladium.metal_call, palladium.metal_call_jit])
def test_edge_and_strided_tiles(metal_device, shape, block, dtype, factory):
    spec = pl.BlockSpec(block, lambda *ids: ids)
    call = factory(
        copy_kernel,
        grid=tuple((a + b - 1) // b for a, b in zip(shape, block)),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    got = call(x) if factory is palladium.metal_call else jax.jit(call)(jnp.asarray(x))
    np.testing.assert_array_equal(got, x * 2)
    np.testing.assert_array_equal(got, call.interpret(x))


@pytest.mark.parametrize("block", [(2, None, 3), (2, 7, None), (None, 3, 2)])
def test_squeezed_axes(metal_device, block):
    shape = (5, 7, 3)
    extents = tuple(1 if b is None else b for b in block)
    spec = pl.BlockSpec(block, lambda i, j, k: (i, j, k))
    call = palladium.metal_call(
        copy_kernel,
        grid=tuple((a + b - 1) // b for a, b in zip(shape, extents)),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(shape, np.float32),
    )
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(call(x), x * 2)
    np.testing.assert_array_equal(call(x), call.interpret(x))


def test_column_read_write_snapshot(metal_device):
    def kernel(x, out):
        out[...] = x[...]
        saved = out[:, 1]
        out[:, 1] = x[:, 2]
        out[:, 2] = saved

    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    call = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))
    expected = x.copy()
    expected[:, [1, 2]] = x[:, [2, 1]]
    np.testing.assert_array_equal(call(x), expected)
    np.testing.assert_array_equal(call(x), call.interpret(x))


def test_edge_indexed_writes_preserve_alias(metal_device):
    def kernel(x, out):
        value = x[:, 1]
        out[:, 1] = value + 10

    shape, block = (5, 7), (3, 4)
    spec = pl.BlockSpec(block, lambda i, j: (i, j))
    call = palladium.metal_call(
        kernel,
        grid=(2, 2),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(shape, np.float32),
        input_output_aliases={0: 0},
    )
    x = np.arange(35, dtype=np.float32).reshape(shape)
    expected = x.copy()
    expected[:, [1, 5]] += 10
    np.testing.assert_array_equal(call(x), expected)


def test_edge_source_guards_memory_access():
    spec = pl.BlockSpec((8,), lambda i: (i,))
    source = palladium.debug_msl(
        copy_kernel,
        jax.ShapeDtypeStruct((9,), np.float32),
        grid=(2,),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((9,), np.float32),
    )
    assert " ? arg0[" in source
    assert "if (" in source and "arg1[" in source


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16, np.int32, np.uint32])
def test_padding_matches_interpreter(metal_device, dtype):
    def kernel(x, out):
        out[0] = x[7]

    call = palladium.metal_call(
        kernel,
        grid=(2,),
        math_mode=mr.MathMode.SAFE,
        in_specs=[pl.BlockSpec((8,), lambda i: (i,))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((2,), dtype),
    )
    x = np.arange(9, dtype=np.float32).astype(dtype)
    np.testing.assert_array_equal(
        np.asarray(call(x), np.float64), np.asarray(call.interpret(x), np.float64)
    )


def test_strided_tile_read_and_write(metal_device):
    def kernel(x, out):
        out[...] = x[...]
        out[::2, 1::2] = x[::2, 1::2] + 10

    shape = (7, 9)
    spec = pl.BlockSpec((4, 6), lambda i, j: (i, j))
    call = palladium.metal_call(
        kernel,
        grid=(2, 2),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(shape, np.float32),
    )
    x = np.arange(63, dtype=np.float32).reshape(shape)
    expected = x.copy()
    expected[::2, 1::2] += 10
    np.testing.assert_array_equal(call(x), expected)
    np.testing.assert_array_equal(call(x), call.interpret(x))


def test_gridless_strided_block(metal_device):
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    call = palladium.metal_call(
        copy_kernel,
        in_specs=[pl.BlockSpec((4, 1), lambda: (0, 2))],
        out_shape=jax.ShapeDtypeStruct((4, 1), np.float32),
    )
    np.testing.assert_array_equal(call(x), x[:, 2:3] * 2)
    np.testing.assert_array_equal(call(x), call.interpret(x))
