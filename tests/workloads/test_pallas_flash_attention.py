import jax
import jax.numpy as jnp
import numpy as np
import pytest

from palladium.trace import trace
from palladium.workloads.pallas_flash_attention import (
    make_pallas_flash_attention,
    pallas_flash_attention,
    reference_attention,
)

SHAPE = (1, 32, 2, 64)


@pytest.mark.parametrize("causal", [False, True])
def test_pallas_online_softmax_matches_reference(causal):
    rng = np.random.default_rng(41)
    q, k, v = (rng.standard_normal(SHAPE, dtype=np.float32) for _ in range(3))
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        actual = pallas_flash_attention(
            q, k, v, tile_q=16, tile_k=16, causal=causal, interpret=True
        )

    expected = reference_attention(q, k, v, causal)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_pallas_online_softmax_uses_the_runtime_head_dimension():
    shape = (1, 32, 2, 32)
    rng = np.random.default_rng(19)
    q, k, v = (rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        actual = pallas_flash_attention(q, k, v, tile_q=16, tile_k=16, interpret=True)

    np.testing.assert_allclose(actual, reference_attention(q, k, v, False), rtol=3e-5, atol=3e-5)


def test_pallas_cross_attention_supports_distinct_query_and_key_lengths():
    q_shape = (1, 16, 2, 16)
    kv_shape = (1, 32, 2, 16)
    rng = np.random.default_rng(47)
    q = rng.standard_normal(q_shape, dtype=np.float32)
    k, v = (rng.standard_normal(kv_shape, dtype=np.float32) for _ in range(2))
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        actual = pallas_flash_attention(q, k, v, tile_q=16, tile_k=16, interpret=True)

    expected = reference_attention(q, k, v, False)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_pallas_cross_attention_supports_more_queries_than_keys():
    q_shape = (1, 32, 2, 16)
    kv_shape = (1, 16, 2, 16)
    rng = np.random.default_rng(53)
    q = rng.standard_normal(q_shape, dtype=np.float32)
    k, v = (rng.standard_normal(kv_shape, dtype=np.float32) for _ in range(2))
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        actual = pallas_flash_attention(q, k, v, tile_q=16, tile_k=16, interpret=True)

    np.testing.assert_allclose(actual, reference_attention(q, k, v, False), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("tile_q,tile_k", [(16, 16), (16, 32), (16, 64), (32, 32), (64, 32)])
def test_pallas_attention_tile_sweep_matches_reference(tile_q, tile_k):
    shape = (1, 64, 1, 16)
    rng = np.random.default_rng(31)
    q, k, v = (rng.standard_normal(shape, dtype=np.float32) for _ in range(3))
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        actual = pallas_flash_attention(
            q, k, v, tile_q=tile_q, tile_k=tile_k, causal=False, interpret=True
        )

    np.testing.assert_allclose(actual, reference_attention(q, k, v, False), rtol=3e-5, atol=3e-5)


def test_attention_jaxpr_keeps_both_dots_inside_the_kv_loop():
    call = make_pallas_flash_attention(SHAPE, tile_q=16, tile_k=16)
    args = [jax.ShapeDtypeStruct(SHAPE, jnp.float32)] * 3
    spec = trace(call, *args)
    kv_loop = next(eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "scan")
    body = kv_loop.params["jaxpr"]

    assert body is not None
    assert [eqn.primitive.name for eqn in body.eqns].count("dot_general") == 2
    assert spec.grid == (1, 2, 2)
    assert [info.block_shape for info in spec.inputs] == [
        (1, 16, 1, 64),
        (1, 32, 1, 64),
        (1, 32, 1, 64),
    ]


def test_cross_attention_jaxpr_tracks_query_and_key_lengths_separately():
    q_shape = (1, 16, 2, 64)
    kv_shape = (1, 32, 2, 64)
    call = make_pallas_flash_attention(q_shape, tile_q=16, tile_k=16, key_length=kv_shape[1])
    args = [jax.ShapeDtypeStruct(q_shape, jnp.float32)] + [
        jax.ShapeDtypeStruct(kv_shape, jnp.float32)
    ] * 2
    spec = trace(call, *args)

    assert spec.grid == (1, 2, 1)
    assert [info.array_shape for info in spec.inputs] == [q_shape, kv_shape, kv_shape]
    assert [info.block_shape for info in spec.inputs] == [
        (1, 16, 1, 64),
        (1, 32, 1, 64),
        (1, 32, 1, 64),
    ]
