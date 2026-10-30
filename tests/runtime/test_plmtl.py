"""`plmtl.kernel`: plain Pallas on a mesh of threadgroups with threadgroup scratch,
checked against the Pallas interpreter and NumPy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl

import palladium
from palladium import plmtl
from palladium.errors import EmitError


def _matmul(m, n, k, tm, tn, kt, epilogue=lambda acc: acc):
    def body(a_ref, b_ref, o_ref, a_tile, b_tile):
        i, j = pl.program_id(0), pl.program_id(1)

        def step(kb, acc):
            a_tile[...] = a_ref[pl.ds(i * tm, tm), pl.ds(kb * kt, kt)]
            b_tile[...] = b_ref[pl.ds(kb * kt, kt), pl.ds(j * tn, tn)]
            return acc + jnp.dot(a_tile[...], b_tile[...])

        acc = lax.fori_loop(0, k // kt, step, jnp.zeros((tm, tn), jnp.float32))
        o_ref[pl.ds(i * tm, tm), pl.ds(j * tn, tn)] = epilogue(acc)

    return plmtl.kernel(
        body,
        jax.ShapeDtypeStruct((m, n), jnp.float32),
        mesh=plmtl.Mesh((m // tm, n // tn)),
        scratch_types=[
            plmtl.THREADGROUP((tm, kt), jnp.float32),
            plmtl.THREADGROUP((kt, tn), jnp.float32),
        ],
    )


def _operands(rng, m, n, k):
    return rng.standard_normal((m, k), np.float32), rng.standard_normal((k, n), np.float32)


def _check(call, *args, want):
    got = np.asarray(call(*args))
    np.testing.assert_allclose(got, np.asarray(call.interpret(*args)), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize(
    ("m", "n", "k", "tm", "tn", "kt"),
    [
        (128, 192, 64, 64, 64, 16),  # the MLX-style tiling, several K steps
        (64, 64, 128, 32, 32, 32),  # one fragment per SIMD group
        (256, 128, 48, 128, 64, 16),  # four fragment rows per SIMD group
    ],
)
def test_matmul_matches_interpreter(metal_device, rng, m, n, k, tm, tn, kt):
    """Tiles staged in threadgroup memory multiply on SIMD-group fragments."""
    a, b = _operands(rng, m, n, k)
    _check(_matmul(m, n, k, tm, tn, kt), a, b, want=a @ b)


def test_epilogue_with_a_comparison(metal_device, rng):
    """Elementwise epilogues run per fragment element, comparisons included."""
    a, b = _operands(rng, 64, 64, 32)
    call = _matmul(64, 64, 32, 64, 64, 16, lambda acc: jnp.where(acc > 0, acc, 0.1 * acc) + 1.0)
    p = a @ b
    _check(call, a, b, want=np.where(p > 0, p, 0.1 * p) + 1.0)


def test_single_dot_without_a_loop(metal_device, rng):
    """A dot outside a loop starts from zeroed fragments."""

    def body(a_ref, b_ref, o_ref, a_tile, b_tile):
        a_tile[...] = a_ref[...]
        b_tile[...] = b_ref[...]
        o_ref[...] = jnp.dot(a_tile[...], b_tile[...]) * 2.0

    a, b = _operands(rng, 32, 32, 16)
    call = plmtl.kernel(
        body,
        jax.ShapeDtypeStruct((32, 32), jnp.float32),
        mesh=plmtl.Mesh((1,)),
        scratch_types=[
            plmtl.THREADGROUP((32, 16), jnp.float32),
            plmtl.THREADGROUP((16, 32), jnp.float32),
        ],
    )
    _check(call, a, b, want=2 * a @ b)


def test_copy_through_threadgroup_memory(metal_device, rng):
    """Reading rows of a tile that other threads wrote needs a barrier between the copies:
    the read starts one row in, so no thread reads back only its own writes."""

    def body(x_ref, o_ref, tile):
        i = pl.program_id(0)
        tile[...] = x_ref[pl.ds(i * 8, 8), :]
        o_ref[pl.ds(i * 7, 7), :] = tile[pl.ds(1, 7), :]

    x = rng.standard_normal((32, 36)).astype(np.float32)
    call = plmtl.kernel(
        body,
        jax.ShapeDtypeStruct((28, 36), jnp.float32),
        mesh=plmtl.Mesh((4,)),
        scratch_types=[plmtl.THREADGROUP((8, 36), jnp.float32)],
    )
    _, msl, _ = call._spec_and_msl((jax.ShapeDtypeStruct(x.shape, jnp.float32),))
    assert msl.count("threadgroup_barrier") == 1
    np.testing.assert_array_equal(call(x), x.reshape(4, 8, 36)[:, 1:].reshape(28, 36))


def test_matmul_source_has_two_barriers_per_step_and_accumulates_in_place():
    """The emitter places one barrier before staging (the previous step's reads) and one
    before the multiply, and accumulates into the loop carry."""
    shapes = (
        jax.ShapeDtypeStruct((128, 64), jnp.float32),
        jax.ShapeDtypeStruct((64, 128), jnp.float32),
    )
    call = _matmul(128, 128, 64, 64, 64, 16)
    spec, msl, _ = call._spec_and_msl(shapes)
    assert spec.simdgroups == 4 and {s.space for s in spec.scratch} == {"threadgroup"}
    assert "// palladium: threadgroup width 128" in msl
    loop = msl[msl.index("for (int _it") :]
    assert loop.count("threadgroup_barrier") == 2
    assert msl.count("vec<float, 2> _f") == 1  # the carry is the only accumulator


def test_rejects_dots_of_device_reads():
    def body(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    call = plmtl.kernel(body, jax.ShapeDtypeStruct((32, 32), jnp.float32), mesh=plmtl.Mesh((1,)))
    shapes = (jax.ShapeDtypeStruct((32, 32), jnp.float32),) * 2
    with pytest.raises(EmitError, match="stage dot operands"):
        call._spec_and_msl(shapes)


def test_rejects_tiles_that_do_not_divide_into_fragments():
    call = _matmul(48, 48, 16, 24, 24, 16)
    shapes = (
        jax.ShapeDtypeStruct((48, 16), jnp.float32),
        jax.ShapeDtypeStruct((16, 48), jnp.float32),
    )
    with pytest.raises(EmitError, match="fragments need rows divisible by 16"):
        call._spec_and_msl(shapes)


def test_rejects_meshes_other_than_four_simdgroups():
    def body(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    call = plmtl.kernel(
        body, jax.ShapeDtypeStruct((8,), jnp.float32), mesh=plmtl.Mesh((1,), simdgroups=2)
    )
    with pytest.raises(palladium.TraceError, match="4 SIMD groups"):
        call._spec_and_msl((jax.ShapeDtypeStruct((8,), jnp.float32),))
