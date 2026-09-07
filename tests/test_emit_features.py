"""Permutation reshapes and variadic selects: source and Metal checks."""

import re

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest

import palladium


@pytest.fixture
def metal_device():
    try:
        mr.device_name()
    except mr.DeviceError as exc:
        pytest.skip(str(exc))


RESHAPES = [
    ((2, 3), (1, 0), (6,)),
    ((2, 3, 4), (2, 0, 1), (4, 6)),
    ((2, 1, 3), (1, 2, 0), (3, 2)),
    ((2, 3), (0, 1), (6,)),
    ((1, 1), (1, 0), ()),
    ((), (), (1,)),
]


def _reshape_kernel(perm, new_shape):
    def kernel(x, o):
        o[...] = jax.lax.reshape(x[...], new_shape, dimensions=perm)

    return kernel


@pytest.mark.parametrize("shape,perm,new_shape", RESHAPES)
def test_permuted_reshape_emits(shape, perm, new_shape):
    source = palladium.debug_msl(
        _reshape_kernel(perm, new_shape),
        jax.ShapeDtypeStruct(shape, jnp.float32),
        out_shape=jax.ShapeDtypeStruct(new_shape, jnp.float32),
    )
    # One input load and at most one destination; no transpose intermediate.
    assert len(re.findall(r"^    float t\d+(?:\[\d+\])?;", source, re.MULTILINE)) <= 2
    assert "arg1[" in source


@pytest.mark.parametrize("shape,perm,new_shape", RESHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_permuted_reshape_on_metal(metal_device, shape, perm, new_shape, dtype):
    call = palladium.metal_call(
        _reshape_kernel(perm, new_shape),
        out_shape=jax.ShapeDtypeStruct(new_shape, dtype),
        math_mode=mr.MathMode.SAFE,
    )
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    expected = np.transpose(x, perm).reshape(new_shape)
    np.testing.assert_array_equal(call(x), expected)
    np.testing.assert_array_equal(call.interpret(x), expected)


def _select_kernel(count):
    def kernel(which, *refs):
        *cases, out = refs
        out[...] = jax.lax.select_n(which[...], *(case[...] for case in cases))

    return kernel


@pytest.mark.parametrize("count", [1, 2, 3, 8])
@pytest.mark.parametrize("index_dtype", [np.int32, np.uint32])
def test_multiway_select_emits_all_cases(count, index_dtype):
    shapes = [jax.ShapeDtypeStruct((2, 4), np.float32)] * count
    source = palladium.debug_msl(
        _select_kernel(count),
        jax.ShapeDtypeStruct((), index_dtype),
        *shapes,
        out_shape=shapes[0],
    )
    # Eight cases exceed the generic template's six-operand limit.
    assert source.count(" ? ") == count - 1
    for i in range(1, count + 1):
        assert f"= arg{i}[" in source


@pytest.mark.parametrize("count", [1, 2, 3, 8])
@pytest.mark.parametrize("index_dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("scalar_index", [False, True])
def test_multiway_select_on_metal(metal_device, count, index_dtype, scalar_index):
    shape = (2, 4)
    cases = [
        np.arange(8, dtype=np.float32).reshape(shape) + i * 10 for i in range(count)
    ]
    which = (
        np.asarray(count - 1, dtype=index_dtype)
        if scalar_index
        else (np.arange(8) % count).astype(index_dtype).reshape(shape)
    )
    call = palladium.metal_call(
        _select_kernel(count), out_shape=jax.ShapeDtypeStruct(shape, np.float32)
    )
    np.testing.assert_array_equal(call(which, *cases), call.interpret(which, *cases))


def test_scalar_select_and_permuted_reshape_inside_scan(metal_device):
    def kernel(x, o):
        def body(i, carry):
            flat = jax.lax.reshape(carry, (6,), dimensions=(1, 0))
            return jax.lax.reshape(flat, (2, 3)) + jax.lax.select_n(
                i, jnp.float32(1), jnp.float32(3), jnp.float32(5)
            )

        o[...] = jax.lax.fori_loop(0, 3, body, x[...])

    call = palladium.metal_call(
        kernel, out_shape=jax.ShapeDtypeStruct((2, 3), np.float32)
    )
    x = np.arange(6, dtype=np.float32).reshape((2, 3))
    np.testing.assert_array_equal(call(x), call.interpret(x))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int32, jnp.bool_])
def test_select_case_dtypes(metal_device, dtype):
    which = np.asarray([0, 1, 2, 1], dtype=np.int32)
    cases = [
        np.asarray(values, dtype=dtype)
        for values in ([0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1])
    ]
    call = palladium.metal_call(
        _select_kernel(3), out_shape=jax.ShapeDtypeStruct((4,), dtype)
    )
    np.testing.assert_array_equal(call(which, *cases), call.interpret(which, *cases))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_select_out_of_range_uses_endpoints(metal_device, dtype):
    # This is palladium's policy; JAX leaves these indices implementation-defined.
    limits = np.iinfo(dtype)
    which = np.asarray([limits.min, limits.max, 1], dtype=dtype)
    cases = [np.full((3,), i * 10, dtype=np.float32) for i in range(3)]
    call = palladium.metal_call(
        _select_kernel(3), out_shape=jax.ShapeDtypeStruct((3,), np.float32)
    )
    np.testing.assert_array_equal(call(which, *cases), [0, 20, 10])
