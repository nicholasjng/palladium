"""Differential validation of all solver inputs and actual parameter recovery."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from palladium.workloads.ode_training import (
    initial_state,
    make_solver,
    make_training_step,
    recovery_problem,
    reference_solve,
)


@pytest.mark.parametrize("steps,interval", [(1, 1), (7, 1), (7, 3), (7, 10), (23, 5), (100, 10)])
@pytest.mark.parametrize("seed", [17, 29])
def test_all_six_vjp_inputs(steps, interval, seed):
    n = 8
    rng = np.random.default_rng(seed)
    args = tuple(
        jnp.asarray(rng.uniform(lo, hi, n), dtype=jnp.float32)
        for lo, hi in [
            (0.3, 2),
            (0.4, 1.8),
            (0.7, 1.4),
            (0.2, 0.6),
            (0.05, 0.2),
            (0.2, 0.6),
        ]
    )
    cotangents = tuple(jnp.asarray(rng.normal(size=n), dtype=jnp.float32) for _ in range(2))
    reference = functools.partial(reference_solve, steps=steps, dt=0.03)
    with jax.default_device(jax.devices("cpu")[0]):
        cpu_args = jax.device_put(args, jax.devices("cpu")[0])
        cpu_cotangents = jax.device_put(cotangents, jax.devices("cpu")[0])
        expected, pullback = jax.vjp(reference, *cpu_args)
        expected_grad = tuple(np.asarray(v) for v in pullback(cpu_cotangents))
        expected = tuple(np.asarray(v) for v in expected)
    for variant, spacing in [
        ("reverse", interval),
        ("reverse", 1),
        ("tangent", interval),
        ("reference", interval),
    ]:
        solve = make_solver(n, steps=steps, dt=0.03, interval=spacing, variant=variant)

        @jax.jit
        def evaluate(inputs, cots, solve=solve):
            value, pb = jax.vjp(solve, *inputs)
            return value, pb(cots)

        actual, actual_grad = evaluate(args, cotangents)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=3e-4, atol=3e-6)


def test_forward_and_configuration():
    args = (jnp.ones(8),) * 6
    solve = make_solver(8, steps=7, interval=3, variant="forward")
    np.testing.assert_allclose(jax.jit(solve)(*args), reference_solve(*args, steps=7), rtol=2e-5)
    for kwargs in ({"steps": 0}, {"interval": 0}, {"dt": 0}, {"variant": "bad"}):
        with pytest.raises(ValueError):
            make_solver(8, **kwargs)


def test_recovery_and_optimizer_agreement():
    data = recovery_problem(64)
    update, loss = make_training_step(make_solver(64))
    ref_update, _ = make_training_step(make_solver(64, variant="jax"))
    state = initial_state()
    actual = expected = state
    for _ in range(5):
        actual, actual_loss = update(actual, data)
        expected, expected_loss = ref_update(expected, data)
        np.testing.assert_allclose(actual_loss, expected_loss, rtol=3e-4, atol=3e-6)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-6)
    initial_loss = float(loss(state[0], data))
    for _ in range(500):
        state, _ = update(state, data)
    final_loss = float(loss(state[0], data))
    assert final_loss < 1e-6
    assert final_loss < initial_loss * 1e-4
    np.testing.assert_allclose(state[0], [1.1, 0.4, 0.1, 0.4], atol=0.005, rtol=0)
