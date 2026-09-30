"""Counter-based RNG (`jax.random` inside a kernel).

`jax.random.uniform` stages as `random_wrap`, `random_bits`, then
`shift_right_logical`, `or`, `bitcast_convert_type` and ordinary arithmetic;
`random_fold_in` is the same hash seeded with (0, data). Threefry-2x32-20 is
a well-defined algorithm, so every rule is checked bit-for-bit against JAX's
own output.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium

U32 = jnp.uint32
F32 = jnp.float32


def test_random_bits_matches_jax_exactly(rng):
    """Raw `random_bits` output, bit-for-bit, not just the float conversion."""

    def kernel(k_ref, o_ref):
        key = jax.random.wrap_key_data(k_ref[...], impl="threefry2x32")
        o_ref[...] = jax.random.bits(key, (16,), "uint32")

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((16,), U32))
    key = jax.random.key(int(rng.integers(0, 2**31)))
    kd = np.asarray(jax.random.key_data(key))
    got = f(kd)
    want = np.asarray(jax.random.bits(key, (16,), "uint32"))
    np.testing.assert_array_equal(got, want)


def test_uniform_matches_jax_exactly():
    """The full uniform() pipeline: random_bits, shift, or, bitcast, sub/mul/add."""

    def kernel(k_ref, o_ref):
        key = jax.random.wrap_key_data(k_ref[...], impl="threefry2x32")
        o_ref[...] = jax.random.uniform(key, (64,))

    f = palladium.metal_call(kernel, out_shape=jax.ShapeDtypeStruct((64,), F32))
    key = jax.random.key(1234)
    kd = np.asarray(jax.random.key_data(key))
    got = f(kd)
    want = np.asarray(jax.random.uniform(key, (64,)))
    np.testing.assert_array_equal(got, want)


def test_fold_in_matches_jax_exactly():
    """fold_in's derived key data matches JAX bit-for-bit."""

    def kernel(k_ref, d_ref, ko_ref):
        key = jax.random.wrap_key_data(k_ref[...], impl="threefry2x32")
        folded = jax.random.fold_in(key, d_ref[0])
        ko_ref[...] = jax.random.key_data(folded)

    f = palladium.metal_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2,), U32),
    )
    key = jax.random.key(99)
    kd = np.asarray(jax.random.key_data(key))
    data = np.array([17], dtype=np.uint32)
    got = f(kd, data)
    want = np.asarray(jax.random.key_data(jax.random.fold_in(key, 17)))
    np.testing.assert_array_equal(got, want)


def test_per_thread_independent_streams():
    """Folding program_id into the key gives each lane a distinct stream rather than a copy of lane 0's."""

    def kernel(k_ref, o_ref):
        base = jax.random.wrap_key_data(k_ref[...], impl="threefry2x32")
        key = jax.random.fold_in(base, pl.program_id(0))
        o_ref[...] = jax.random.uniform(key, (4,))

    f = palladium.metal_call(
        kernel,
        grid=(8,),
        in_specs=[pl.BlockSpec((2,), lambda i: (0,))],
        out_specs=pl.BlockSpec((4,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((32,), F32),
    )
    key = jax.random.key(7)
    kd = np.asarray(jax.random.key_data(key))
    got = np.asarray(f(kd))
    assert not isinstance(got, tuple)
    want = np.stack(
        [np.asarray(jax.random.uniform(jax.random.fold_in(key, i), (4,))) for i in range(8)]
    )
    got = got.reshape(8, 4)
    np.testing.assert_array_equal(got, want)
    # Every lane's stream is distinct.
    assert len({tuple(row) for row in got}) == 8
