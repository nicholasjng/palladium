"""Numerical boundaries for typed lowering and JAX 0.11.2 AD primitives."""

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


def call_for(op, *args, mode=mr.MathMode.SAFE):
    def kernel(*refs):
        *inputs, out = refs
        out[...] = op(*(ref[...] for ref in inputs))

    shape = jax.eval_shape(op, *args)
    return palladium.metal_call(kernel, out_shape=shape, math_mode=mode)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("op", [jax.lax.min, jax.lax.max, jax.lax.div, jax.lax.rem])
def test_integer_binary_edges(metal_device, dtype, op):
    limits = np.iinfo(dtype)
    x = np.array([limits.min, limits.max, 2**24 + 1, 2**24 + 3, 0, 1, 7], dtype)
    y = np.array([limits.max, 0, 2**24, 2**24 + 2, 0, 0, 3], dtype)
    call = call_for(op, x, y)
    np.testing.assert_array_equal(call(x, y), call.interpret(x, y))


@pytest.mark.parametrize("scalar", [False, True])
def test_signed_overflow_division_abs(metal_device, scalar):
    x = np.array(-2147483648 if scalar else [-2147483648, -7, -1, 0, 1], np.int32)
    for op in (jnp.abs, lambda x: jax.lax.div(x, np.int32(-1))):
        call = call_for(op, x)
        np.testing.assert_array_equal(call(x), call.interpret(x))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize(
    "op", [jax.lax.shift_right_logical, jax.lax.shift_right_arithmetic, jax.lax.shift_left]
)
def test_shift_boundaries_and_broadcast(metal_device, dtype, op):
    x = np.array([0, 1, np.iinfo(dtype).max, np.iinfo(dtype).min], dtype)[:, None]
    counts = [0, 1, 30, 31, 32, 33, np.iinfo(dtype).max]
    if dtype == np.int32:
        counts.append(-1)
        x = np.concatenate((x, np.array([[-1], [-2]], dtype)))
    y = np.array(counts, dtype)[None, :]
    call = call_for(op, x, y)
    np.testing.assert_array_equal(call(x, y), call.interpret(x, y))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float16, jnp.bfloat16, np.float32])
@pytest.mark.parametrize("axis", [None, 0, 1, ()])
@pytest.mark.parametrize("op", [jnp.min, jnp.max])
def test_typed_reduction(metal_device, dtype, axis, op):
    if np.dtype(dtype).kind in "iu":
        limits = np.iinfo(dtype)
        x = np.array([[limits.min, limits.max, 2**24 + 1], [2**24 + 3, 0, 7]], dtype)
    else:
        x = np.array([[-np.inf, -2.5, 0], [np.inf, 0.5, np.nan]], dtype)
    fn = lambda x: op(x, axis=axis)
    call = call_for(fn, x)
    np.testing.assert_array_equal(
        np.asarray(call(x), np.float64), np.asarray(call.interpret(x), np.float64)
    )


@pytest.mark.parametrize("op", [jnp.minimum, jnp.maximum])
def test_float_extrema_nan_and_zero(metal_device, op):
    x = np.array([np.nan, 1, -0.0, 0.0, -np.inf, np.inf], np.float32)
    y = np.array([1, np.nan, 0.0, -0.0, np.inf, -np.inf], np.float32)
    call = call_for(op, x, y)
    got, want = call(x, y), np.asarray(call.interpret(x, y))
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(np.signbit(got[2:4]), np.signbit(want[2:4]))


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("mode", [mr.MathMode.SAFE, mr.MathMode.FAST])
@pytest.mark.parametrize("op", [jax.lax.one_minus_square, jnp.log2])
def test_new_float_primitives(metal_device, dtype, mode, op):
    eps = float(jnp.finfo(dtype).eps)
    values = [-1 - eps, -1, -1 + eps / 2, -eps / 4, 0, eps / 4, 1 - eps / 2, 1, 1 + eps, 2]
    if op is jnp.log2:
        values = [eps, 0.125, 0.5, 1, 1 + eps, 2, 8, 16]
    x = np.array(values, dtype)
    call = call_for(op, x, mode=mode)
    np.testing.assert_allclose(
        np.asarray(call(x), np.float32),
        np.asarray(call.interpret(x), np.float32),
        rtol=2 * eps,
        atol=eps**2,
    )


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_integer_one_minus_square(metal_device, dtype):
    x = np.array([np.iinfo(dtype).min, np.iinfo(dtype).max, 0, 1, 2, 65537], dtype)
    call = call_for(jax.lax.one_minus_square, x)
    np.testing.assert_array_equal(call(x), call.interpret(x))


def test_tanh_ad_in_control_flow(metal_device):
    # Differentiate twice, then stage through both scan and cond.
    derivative = jax.grad(lambda x: jnp.tanh(x))
    second = jax.grad(derivative)

    def op(x):
        return jax.lax.fori_loop(0, 2, lambda i, y: jax.lax.cond(i == 0, derivative, second, y), x)

    x = np.array(0.3, np.float32)
    call = call_for(op, x)
    np.testing.assert_allclose(call(x), call.interpret(x), rtol=2e-6)


@pytest.mark.parametrize("op", [jax.lax.min, jax.lax.max])
def test_integer_scalar_literal(metal_device, op):
    x = np.array([2**24 + 1, 2**24 + 3], np.int32)
    call = call_for(lambda x: op(x, np.int32(2**24)), x)
    np.testing.assert_array_equal(call(x), call.interpret(x))


@pytest.mark.parametrize("op", [jax.lax.one_minus_square, jnp.log2])
def test_new_float_special_values(metal_device, op):
    x = np.array([-np.inf, -2, -1, -0.0, 0.0, 1, 2, np.inf, np.nan], np.float32)
    call = call_for(op, x)
    np.testing.assert_array_equal(call(x), call.interpret(x))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32])
def test_minmax_pair(metal_device, dtype):
    def kernel(x, lo, hi):
        lo[...], hi[...] = jnp.minmax(x[...])

    x = np.array([1, 2**24 + 1, 7, 2**24 + 3], dtype)
    scalar = jax.ShapeDtypeStruct((), dtype)
    call = palladium.metal_call(kernel, out_shape=(scalar, scalar), math_mode=mr.MathMode.SAFE)
    np.testing.assert_array_equal(call(x), call.interpret(x))


@pytest.mark.parametrize("op", [jax.lax.min, jax.lax.max])
def test_boolean_extrema(metal_device, op):
    x, y = np.array([False, False, True, True]), np.array([False, True, False, True])
    call = call_for(op, x, y)
    np.testing.assert_array_equal(call(x, y), call.interpret(x, y))


def test_new_primitives_trace_without_device():
    def kernel(x, out):
        out[...] = jnp.log2(jax.lax.one_minus_square(x[...]))

    arg = jax.ShapeDtypeStruct((4,), np.float32)
    source = palladium.debug_msl(kernel, arg, out_shape=arg)
    assert "precise::log2" in source


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_integer_extrema_do_not_lower_via_float(dtype):
    def kernel(x, out):
        out[...] = jnp.max(jnp.minimum(x[...], dtype(7)))

    source = palladium.debug_msl(
        kernel, jax.ShapeDtypeStruct((4,), dtype), out_shape=jax.ShapeDtypeStruct((), dtype)
    )
    assert "fmin(" not in source and "fmax(" not in source
    assert "INFINITY" not in source
