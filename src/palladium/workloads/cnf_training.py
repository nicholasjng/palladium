"""Experimental 2D continuous normalizing flow with a fused discrete adjoint.

A time-dependent, one-hidden-layer tanh MLP supplies the vector field.
The augmented ODE integrates its exact divergence. JAX differentiates each
pure RK4 step when tracing the backward Pallas kernel, including derivatives
of the divergence with respect to state and weights.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium


def parameter_count(width):
    return 6 * width + 2


def init_parameters(width=4, seed=17):
    if width < 1:
        raise ValueError("width must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.3, parameter_count(width)).astype(np.float32)
    weights[-2:] = 0
    return jnp.asarray(weights)


def field_and_divergence(state, parameters, time):
    """Scalar tuple implementation shared by JAX and Pallas.

    Each hidden unit has weights (x, y, t), bias, and two output weights.
    The final two parameters are output biases.
    """
    x, y = state[:2]
    vx, vy = parameters[-2], parameters[-1]
    divergence = x * 0
    for i in range((len(parameters) - 2) // 6):
        wx, wy, wt, bias, ux, uy = parameters[6 * i : 6 * i + 6]
        h = jnp.tanh(wx * x + wy * y + wt * time + bias)
        vx, vy = vx + ux * h, vy + uy * h
        divergence = divergence + (ux * wx + uy * wy) * (1 - h * h)
    return vx, vy, divergence


def rk4_step(state, parameters, time, dt):
    def rhs(value, t):
        vx, vy, divergence = field_and_divergence(value, parameters, t)
        return vx, vy, -divergence

    def plus(value, slope, scale):
        return tuple(x + scale * k for x, k in zip(value, slope, strict=True))

    k1 = rhs(state, time)
    k2 = rhs(plus(state, k1, 0.5 * dt), time + 0.5 * dt)
    k3 = rhs(plus(state, k2, 0.5 * dt), time + 0.5 * dt)
    k4 = rhs(plus(state, k3, dt), time + dt)
    return tuple(
        x + dt / 6 * (a + 2 * b + 2 * c + d)
        for x, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )


def reference_flow(states, parameters, *, steps=16, reverse=True):
    """Vectorized JAX baseline, independent of the scalar Pallas step code.

    Per-example augmented states (n,3) and weights (n,p). Array operations
    evaluate all hidden units together rather than expanding scalar arithmetic
    into many JAX operations.
    """
    dt = (-1.0 if reverse else 1.0) / steps
    start = 1.0 if reverse else 0.0

    hidden = parameters[:, :-2].reshape((parameters.shape[0], -1, 6))

    def rhs(z, t):
        activation = jnp.tanh(
            jnp.sum(z[:, None, :2] * hidden[:, :, :2], axis=-1)
            + hidden[:, :, 2] * t
            + hidden[:, :, 3]
        )
        velocity = jnp.sum(activation[:, :, None] * hidden[:, :, 4:], axis=1)
        velocity = velocity + parameters[:, -2:]
        divergence = jnp.sum(
            jnp.sum(hidden[:, :, :2] * hidden[:, :, 4:], axis=-1) * (1 - activation**2),
            axis=1,
        )
        return jnp.concatenate((velocity, -divergence[:, None]), axis=1)

    def step(i, state):
        t = start + i * dt
        k1 = rhs(state, t)
        k2 = rhs(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(state + dt * k3, t + dt)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return jax.lax.fori_loop(0, steps, step, states)


def make_flow(n, *, width=4, steps=16, interval=4, variant="reverse", reverse=True):
    """Integrate data to base (reverse=True) or base to data.

    Returns (n,3) augmented states. Parameters have shape (n,6*width+2);
    broadcast shared weights before calling. The checkpointed VJP differentiates
    the discrete RK4 scheme. Higher-order AD of the custom call is unsupported.
    """
    if min(n, width, steps, interval) < 1:
        raise ValueError("n, width, steps and interval must be positive")
    if variant not in ("jax", "reverse", "reference", "forward"):
        raise ValueError(f"unknown variant {variant!r}")
    reference = functools.partial(reference_flow, steps=steps, reverse=reverse)
    if variant == "jax":
        return reference
    count = parameter_count(width)
    checkpoints = (steps + interval - 1) // interval
    dt = (-1.0 if reverse else 1.0) / steps
    start = 1.0 if reverse else 0.0

    def forward_kernel(state_ref, params_ref, out_ref, *checkpoint_refs):
        params = tuple(params_ref[0, i] for i in range(count))
        initial = tuple(state_ref[0, i] for i in range(3))

        def segment(chunk, value):
            if variant == "reverse":
                for j in range(3):
                    checkpoint_refs[0][0, 3 * chunk + j] = value[j]

            def advance(local, state):
                index = chunk * interval + local
                return jax.lax.cond(
                    index < steps,
                    lambda z: rk4_step(z, params, start + index * dt, dt),
                    lambda z: z,
                    state,
                )

            return jax.lax.fori_loop(0, interval, advance, value)

        result = jax.lax.fori_loop(0, checkpoints, segment, initial)
        for j in range(3):
            out_ref[0, j] = result[j]

    def backward_kernel(
        state_ref, params_ref, checkpoint_ref, cot_ref, state_grad_ref, params_grad_ref
    ):
        del state_ref
        params = tuple(params_ref[0, i] for i in range(count))
        initial_cot = tuple(cot_ref[0, i] for i in range(3))
        zero_grads = tuple(jnp.float32(0) for _ in range(count))

        def reverse_step(index, carry):
            cot, grads = carry
            step_index = steps - 1 - index
            chunk = step_index // interval
            local_step = step_index % interval
            state = tuple(checkpoint_ref[0, 3 * chunk + j] for j in range(3))

            def replay(local, z):
                return rk4_step(z, params, start + (chunk * interval + local) * dt, dt)

            state = jax.lax.fori_loop(0, local_step, replay, state)
            # This VJP is staged as scalar arithmetic into the Metal kernel.
            _, pullback = jax.vjp(
                lambda z, p: rk4_step(z, p, start + step_index * dt, dt), state, params
            )
            cot, local_grads = pullback(cot)
            grads = tuple(a + b for a, b in zip(grads, local_grads, strict=True))
            return cot, grads

        cot, grads = jax.lax.fori_loop(0, steps, reverse_step, (initial_cot, zero_grads))
        for j in range(3):
            state_grad_ref[0, j] = cot[j]
        for j in range(count):
            params_grad_ref[0, j] = grads[j]

    state_spec = pl.BlockSpec((1, 3), lambda i: (i, 0))
    params_spec = pl.BlockSpec((1, count), lambda i: (i, 0))
    checkpoint_spec = pl.BlockSpec((1, 3 * checkpoints), lambda i: (i, 0))
    shape = jax.ShapeDtypeStruct((n, 3), jnp.float32)
    saves = variant == "reverse"
    forward = palladium.mps_call_jit(
        forward_kernel,
        grid=(n,),
        in_specs=(state_spec, params_spec),
        out_specs=(state_spec, checkpoint_spec) if saves else state_spec,
        out_shape=(shape, jax.ShapeDtypeStruct((n, 3 * checkpoints), jnp.float32))
        if saves
        else shape,
    )
    if variant == "forward":
        return forward
    if variant == "reference":
        return forward.with_reference_vjp(reference)
    backward = palladium.mps_call_jit(
        backward_kernel,
        grid=(n,),
        in_specs=(state_spec, params_spec, checkpoint_spec, state_spec),
        out_specs=(state_spec, params_spec),
        out_shape=(shape, jax.ShapeDtypeStruct((n, count), jnp.float32)),
    )
    return forward.with_auxiliary_vjp(backward, output_count=1)


def make_log_prob(n, **kwargs):
    """Gaussian base density minus the reverse-time log-density increment."""
    flow = make_flow(n, **kwargs)

    def log_prob(parameters, points):
        states = jnp.concatenate((points, jnp.zeros((n, 1), points.dtype)), axis=1)
        weights = jnp.broadcast_to(parameters, (n, parameters.shape[0]))
        result = flow(states, weights)
        base = -math.log(2 * math.pi) - 0.5 * jnp.sum(result[:, :2] ** 2, axis=1)
        return base - result[:, 2]

    return log_prob


def make_training_step(n, *, learning_rate=0.01, **kwargs):
    log_prob = make_log_prob(n, **kwargs)

    def loss(parameters, points):
        return -jnp.mean(log_prob(parameters, points))

    @jax.jit
    def update(state, points):
        parameters, moment, variance, count = state
        value, grad = jax.value_and_grad(loss)(parameters, points)
        count = count + 1
        moment = 0.9 * moment + 0.1 * grad
        variance = 0.999 * variance + 0.001 * grad**2
        parameters = parameters - learning_rate * (moment / (1 - 0.9**count)) / (
            jnp.sqrt(variance / (1 - 0.999**count)) + 1e-8
        )
        return (parameters, moment, variance, count), value

    return update, jax.jit(loss)


def initial_state(width=4, seed=17):
    parameters = init_parameters(width, seed)
    return (
        parameters,
        jnp.zeros_like(parameters),
        jnp.zeros_like(parameters),
        jnp.int32(0),
    )


def mixture_data(n, seed=17):
    """Equal mixture of two known axis-aligned Gaussians."""
    rng = np.random.default_rng(seed)
    means = np.array([[-1.3, -0.4], [1.3, 0.4]])
    samples = means[rng.integers(2, size=n)] + rng.normal(size=(n, 2)) * [0.45, 0.65]
    return samples.astype(np.float32)


def mixture_log_prob(points):
    means = jnp.array([[-1.3, -0.4], [1.3, 0.4]], points.dtype)
    scales = jnp.array([0.45, 0.65], points.dtype)
    component = (
        -math.log(2 * math.pi)
        - jnp.sum(jnp.log(scales))
        - 0.5 * jnp.sum(((points[:, None, :] - means) / scales) ** 2, axis=-1)
    )
    return jax.scipy.special.logsumexp(component, axis=1) - math.log(2)
