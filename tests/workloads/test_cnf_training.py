"""CNF density convention and discrete adjoint checks, with the kernels on Metal."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cnf_density import (
    init_parameters,
    initial_state,
    make_flow,
    make_log_prob,
    make_training_step,
    mixture_data,
    mixture_log_prob,
    parameter_count,
    reference_flow,
)
from jax.experimental import pallas as pl

import palladium

# The kernels run on Metal: through pallas_call on mps, metal_call elsewhere.
CALL = pl.pallas_call if jax.default_backend() == "mps" else palladium.metal_call
make_flow = functools.partial(make_flow, call=CALL)
make_log_prob = functools.partial(make_log_prob, call=CALL)
make_training_step = functools.partial(make_training_step, call=CALL)


def test_density_matches_change_of_variables():
    """Independent check of the divergence integral and reverse-density sign.

    The integrated continuous logdet approximates the discrete RK4 map logdet;
    a sufficiently fine solve makes their discrepancy small.
    """
    n, width = 8, 2
    weights = init_parameters(width)
    points = jnp.asarray(mixture_data(n))
    actual = jax.jit(make_log_prob(n, width=width))(weights, points)
    with jax.default_device(jax.devices("cpu")[0]):
        p = jnp.asarray(np.asarray(weights))
        data = jnp.asarray(np.asarray(points))

        def inverse(z):
            augmented = jnp.concatenate((z, jnp.zeros(1)))[None, :]
            return reference_flow(augmented, p[None, :])[0, :2]

        z = jax.vmap(inverse)(data)
        jacobian = jax.vmap(jax.jacfwd(inverse))(data)
        sign, logdet = jnp.linalg.slogdet(jacobian)
        assert np.all(np.asarray(sign) > 0)
        expected = -math.log(2 * math.pi) - 0.5 * jnp.sum(z * z, axis=1) + logdet
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_cnf_fits_heldout_mixture_and_refines():
    n, width = 256, 4
    train = jnp.asarray(mixture_data(n, 17))
    heldout = jnp.asarray(mixture_data(n, 29))
    state = initial_state(width)
    update, loss = make_training_step(n, width=width)
    initial = float(loss(state[0], heldout))
    for _ in range(500):
        state, _ = update(state, train)
    final = float(loss(state[0], heldout))
    params = np.asarray(state[0])
    with jax.default_device(jax.devices("cpu")[0]):
        cpu_points, cpu_params = jnp.asarray(np.asarray(heldout)), jnp.asarray(params)
        _, cpu_loss = make_training_step(n, width=width, variant="jax")
        _, refined_loss = make_training_step(n, width=width, steps=32, variant="jax")
        expected = float(cpu_loss(cpu_params, cpu_points))
        refined = float(refined_loss(cpu_params, cpu_points))
        target = -float(jnp.mean(mixture_log_prob(cpu_points)))
        expected_grad = np.asarray(jax.grad(cpu_loss)(cpu_params, cpu_points))
    actual_grad = jax.jit(jax.grad(loss))(state[0], heldout)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=8e-4, atol=5e-6)
    assert final < initial - 0.7
    assert abs(final - target) < 0.2
    assert abs(final - expected) < 2e-5
    assert abs(final - refined) < 2e-3


@pytest.mark.parametrize("steps,interval,width", [(1, 1, 2), (5, 2, 2), (5, 9, 4), (16, 4, 4)])
def test_checkpointed_cnf_vjp(steps, interval, width):
    n = 8
    rng = np.random.default_rng(steps + width)
    states = jnp.asarray(rng.normal(size=(n, 3)).astype(np.float32))
    weights = jnp.asarray(rng.normal(0, 0.3, (n, parameter_count(width))).astype(np.float32))
    cots = jnp.asarray(rng.normal(size=(n, 3)).astype(np.float32))
    # An independent CPU transform of the complete scan is the oracle.
    with jax.default_device(jax.devices("cpu")[0]):
        cpu_args = jax.device_put((states, weights), jax.devices("cpu")[0])
        cpu_cots = jax.device_put(cots, jax.devices("cpu")[0])
        reference = functools.partial(reference_flow, steps=steps)
        value, pb = jax.vjp(reference, *cpu_args)
        expected = jax.device_get((value, pb(cpu_cots)))
    for variant, spacing in [
        ("reverse", interval),
        ("reverse", 1),
        ("reference", interval),
    ]:
        solve = make_flow(n, steps=steps, interval=spacing, width=width, variant=variant)

        @jax.jit
        def evaluate(z, p, cot, solve=solve):
            result, pullback = jax.vjp(solve, z, p)
            return result, pullback(cot)

        actual = evaluate(states, weights, cots)
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-4, atol=3e-6)


def test_translation_log_density():
    n, width = 8, 2
    params = jnp.zeros(parameter_count(width)).at[-2:].set(jnp.array([0.4, -0.7]))
    points = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2) / 10
    log_prob = jax.jit(make_log_prob(n, width=width, steps=5, interval=2))
    expected = -math.log(2 * math.pi) - 0.5 * jnp.sum(
        (points - jnp.array([0.4, -0.7])) ** 2, axis=1
    )
    np.testing.assert_allclose(log_prob(params, points), expected, rtol=1e-5)
