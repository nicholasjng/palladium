"""Kernels that resemble online-softmax attention without being it.

The cooperative emitters write the algorithm themselves, so the recognizer
must reject every kernel that computes something else: "auto" then emits the
kernel as written, and "tensorops" raises.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flash_attention import attention_specs
from jax.experimental import pallas as pl

import palladium
from palladium.errors import EmitError

SHAPE = (1, 64, 2, 32)
TILE = 32


def _attention(q_ref, k_ref, v_ref, out_ref, *, variant, causal=False):
    """`examples/flash_attention.py`'s kernel with one deviation per `variant`."""
    head_dim = SHAPE[-1]
    q = q_ref[0, :, 0, :]
    q_indices = pl.program_id(2) * TILE + jnp.arange(TILE)
    scale = jnp.asarray(head_dim**-0.5, dtype=jnp.float32)

    def update(key_block, state):
        running_max, running_sum, running_output = state
        k_start = key_block * TILE
        k = k_ref[0, pl.dslice(k_start, TILE), 0, :]
        v_start = k_ref.shape[1] - TILE - k_start if variant == "values reversed" else k_start
        v = v_ref[0, pl.dslice(v_start, TILE), 0, :]
        scores = (q @ k.T) * scale
        if variant == "score bias":
            scores = scores + 1.0 * key_block
        if causal:
            key_indices = k_start + jnp.arange(TILE)
            if variant == "mask off by one":
                key_indices = key_indices + 1
            active = key_indices[None, :] <= q_indices[:, None]
            scores = jnp.where(active, scores, -jnp.inf)
        block_max = jnp.max(scores, axis=1)
        new_max = jnp.maximum(running_max, block_max)
        decay = jnp.exp(running_max - new_max)
        if variant == "decay squared":
            decay = decay * decay
        old_scale = jnp.where(running_sum > 0.0, decay, 0.0)
        probabilities = jnp.exp(scores - new_max[:, None])
        weights = jnp.tanh(probabilities) if variant == "weights squashed" else probabilities
        new_sum = old_scale * running_sum + jnp.sum(probabilities, axis=1)
        new_output = old_scale[:, None] * running_output + weights @ v
        return new_max, new_sum, new_output

    initial = (
        jnp.full((TILE,), -jnp.inf, dtype=jnp.float32),
        jnp.full((TILE,), 1.0 if variant == "sum starts at one" else 0.0, dtype=jnp.float32),
        jnp.zeros((TILE, head_dim), dtype=jnp.float32),
    )
    _, running_sum, running_output = jax.lax.fori_loop(0, k_ref.shape[1] // TILE, update, initial)
    result = running_output / running_sum[:, None]
    if variant == "output squashed":
        result = jnp.tanh(result * 3.0)
    out_ref[0, :, 0, :] = result


def _kwargs(variant, causal, dot_general):
    grid, in_specs, out_specs = attention_specs(SHAPE[0], SHAPE[1], SHAPE[2], TILE, SHAPE[3])
    return {
        "kernel": functools.partial(_attention, variant=variant, causal=causal),
        "grid": grid,
        "in_specs": in_specs,
        "out_specs": out_specs,
        "out_shape": jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        "compiler_params": palladium.CompilerParams(dot_general=dot_general),
    }


_NEAR_MISSES = [
    ("output squashed", False),
    ("weights squashed", False),
    ("values reversed", False),
    ("score bias", False),
    ("decay squared", False),
    ("sum starts at one", False),
    ("mask off by one", True),
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dot_general", ["tensorops", "simdgroup"])
def test_attention_written_without_where_is_recognized(dot_general, causal):
    """The plain spelling, without `jnp.where` around the probabilities, is the same
    algorithm and lowers cooperatively."""
    kwargs = _kwargs(None, causal, dot_general)
    msl = palladium.debug_msl(kwargs.pop("kernel"), *(jnp.zeros(SHAPE),) * 3, **kwargs)
    assert "threadgroup width" in msl


@pytest.mark.parametrize("variant, causal", _NEAR_MISSES)
def test_near_miss_is_not_lowered_as_attention(variant, causal):
    """Every deviation from the algorithm is rejected by the cooperative lowering."""
    kwargs = _kwargs(variant, causal, "tensorops")
    with pytest.raises(EmitError, match="tensorops attention|tensorops softmax|tensorops causal"):
        palladium.debug_msl(kwargs.pop("kernel"), *(jnp.zeros(SHAPE),) * 3, **kwargs)


@pytest.mark.parametrize("variant, causal", [(None, False), (None, True), *_NEAR_MISSES])
def test_auto_matches_the_interpreter(rng, metal_device, variant, causal):
    """Under "auto" a near miss runs as written, and matches the Pallas interpreter."""
    kwargs = _kwargs(variant, causal, "auto")
    call = palladium.metal_call(kwargs.pop("kernel"), **kwargs)
    q, k, v = (rng.standard_normal(SHAPE, dtype=np.float32) for _ in range(3))
    np.testing.assert_allclose(call(q, k, v), call.interpret(q, k, v), rtol=2e-4, atol=2e-5)
