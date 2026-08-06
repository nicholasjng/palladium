"""Stretch 8: counter-based RNG (`jax.random` inside a kernel).

`jax.random.uniform(key, shape)` inside a kernel stages as `random_wrap`
(pure type wrap of a uint32[2] array, no MSL), `random_bits` (the real
generator), then ordinary ops the emitter already had except three:
`shift_right_logical`, `or`, `bitcast_convert_type`. `random_fold_in`
(per-thread/per-step key derivation) is the same hash again, seeded with
(0, data) instead of a counter. All verified against real jaxprs before
implementing, and every rule here is checked bit-for-bit against JAX's
own output (`jax.random.bits`, `jax.random.uniform`, `jax.random.fold_in`
key data), not just "close enough": Threefry-2x32-20 is a specific,
well-defined algorithm, and this reproduces it exactly rather than
substituting a different hash validated only statistically. See NEXT.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

pytestmark = pytest.mark.exercise

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
    """`fold_in` derives a fresh key; check the derived key's own bits,
    not just a downstream float that could hide a wrong key."""

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
    """The point: folding `program_id` into the key per thread gives each
    lane a distinct stream, not a copy of lane 0's. `example 3`'s
    hand-written MSL flags the same requirement for its own RNG."""

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
    got = f(kd)
    assert isinstance(got, np.ndarray)
    want = np.stack(
        [
            np.asarray(jax.random.uniform(jax.random.fold_in(key, i), (4,)))
            for i in range(8)
        ]
    )
    got = got.reshape(8, 4)
    np.testing.assert_array_equal(got, want)
    # Every lane's stream is genuinely different, not silently identical.
    assert len({tuple(row) for row in got}) == 8
