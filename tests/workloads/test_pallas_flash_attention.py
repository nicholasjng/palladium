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
