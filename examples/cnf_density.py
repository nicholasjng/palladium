"""Fit a 2D Gaussian mixture with a continuous normalizing flow.

A time-dependent, one-hidden-layer tanh MLP supplies the vector field, and
the augmented ODE integrates its exact divergence. The forward kernel
checkpoints every `interval` RK4 steps; the backward kernel replays each
segment and differentiates the discrete scheme, which JAX traces from the
same scalar step code. The base is a standard normal; density evaluation
integrates from t=1 to 0. The CNF benchmark and tests import from here.

    JAX_PLATFORMS=mps,cpu uv run python examples/cnf_density.py
    uv run python examples/cnf_density.py --device cpu   # metal_call, no jax-mps
"""

import argparse
import functools
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import palladium


def parameter_count(width):
    """Return the flattened parameter count of the time-dependent two-output MLP.

    Parameters
    ----------
    width : int
        Hidden-unit count.

    Returns
    -------
    int
        Six parameters per hidden unit plus two output biases.

    Notes
    -----
    This arithmetic helper does not validate width; construction helpers require it to be
    positive.
    """
    return 6 * width + 2


def init_parameters(width=4, seed=17):
    """Initialize reproducible float32 CNF weights with zero output biases.

    Parameters
    ----------
    width : int, optional
        Positive hidden-layer width, default 4.
    seed : int, optional
        Deterministic random seed, default 17.

    Returns
    -------
    jax.Array
        Flat parameter vector of length 6 * width + 2.

    Raises
    ------
    ValueError
        The hidden width is less than one.
    """
    if width < 1:
        raise ValueError("width must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.3, parameter_count(width)).astype(np.float32)
    weights[-2:] = 0
    return jnp.asarray(weights)


def field_and_divergence(state, parameters, time):
    """Evaluate a time-dependent vector field and its analytic spatial divergence.

    Parameters
    ----------
    state : tuple
        Scalar augmented state (x, y, log-density increment); only x and y enter the
        field.
    parameters : sequence
        Flat weights: each unit stores (wx, wy, wt, bias, ux, uy), followed by two
        output biases.
    time : scalar
        Current integration time.

    Returns
    -------
    tuple
        Velocity components vx, vy, and positive divergence.

    Notes
    -----
    Tuple/scalar arithmetic is shared by JAX and Pallas. The parameter length must be 6 *
    width + 2; this helper does not validate malformed vectors.
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
    """Advance the augmented CNF state with one classical RK4 step.

    Parameters
    ----------
    state : tuple
        Three scalar state components (x, y, log-density increment).
    parameters : sequence
        Flat vector-field weights.
    time : scalar
        Time at the start of the step.
    dt : scalar
        Signed step size; density evaluation uses a negative step.

    Returns
    -------
    tuple
        Updated state, integrating negative divergence in the third component.
    """

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


def reference_flow(states, parameters, *, steps=16):
    """Integrate a batch of augmented states from time one to zero using vectorized JAX.

    Parameters
    ----------
    states : jax.Array
        Augmented states of shape (n, 3).
    parameters : jax.Array
        Per-example weights of shape (n, 6 * width + 2); broadcast shared weights before
        calling.
    steps : int, optional
        Positive number of fixed RK4 steps from time one to zero, default 16.

    Returns
    -------
    jax.Array
        Final augmented states of shape (n, 3).

    Notes
    -----
    This independent vectorized RK4 implementation serves as the numerical and gradient
    reference. It expects positive steps and compatible parameter shapes; unlike make_flow
    it does not validate those arguments.
    """
    dt = -1.0 / steps
    start = 1.0

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


def _with_reference_vjp(forward, reference):
    """`forward` with the VJP JAX derives from `reference`, for timing the
    forward kernel against JAX's own backward pass."""

    @jax.custom_vjp
    def differentiated(*args):
        return forward(*args)

    def fwd(*args):
        return forward(*args), args

    def bwd(primals, cotangents):
        return jax.vjp(reference, *primals)[1](cotangents)

    differentiated.defvjp(fwd, bwd)
    return differentiated


def make_flow(n, *, width=4, steps=16, interval=4, variant="reverse", call=pl.pallas_call):
    """Build a fixed-size reverse-time CNF integrator with a selected gradient implementation.

    Parameters
    ----------
    n : int
        Number of ensemble members or examples.
    width : int, optional
        Positive hidden-layer width, default 4.
    steps : int, optional
        Positive number of fixed RK4 steps from time one to zero, default 16.
    interval : int, optional
        Positive checkpoint spacing in RK4 steps, default 4.
    variant : {"reverse", "reference", "jax"}, optional
        Checkpointed Pallas adjoint, Pallas forward with the reference VJP, or ordinary
        JAX only. Default "reverse".
    call : callable, optional
        Pallas kernel factory, default pl.pallas_call; use palladium.metal_call on the
        CPU backend.

    Returns
    -------
    callable
        Function (states, parameters) -> (n, 3) final states. Inputs are (n, 3) states
        and (n, 6 * width + 2) per-example weights.

    Raises
    ------
    ValueError
        Any size/step/interval argument is nonpositive or variant is unknown.

    Notes
    -----
    Shared weights must be broadcast by the caller. A final partial checkpoint segment is
    guarded, and intervals larger than the total step count are supported. The reverse
    variant hides checkpoint outputs and differentiates the discrete RK4 map through
    explicit replay.
    """
    if min(n, width, steps, interval) < 1:
        raise ValueError("n, width, steps and interval must be positive")
    if variant not in ("jax", "reverse", "reference"):
        raise ValueError(f"unknown variant {variant!r}")
    reference = functools.partial(reference_flow, steps=steps)
    if variant == "jax":
        return reference
    count = parameter_count(width)
    checkpoints = (steps + interval - 1) // interval
    dt = -1.0 / steps
    start = 1.0

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

    forward = call(
        forward_kernel,
        grid=(n,),
        in_specs=(state_spec, params_spec),
        out_specs=(state_spec, checkpoint_spec) if saves else state_spec,
        out_shape=(shape, jax.ShapeDtypeStruct((n, 3 * checkpoints), jnp.float32))
        if saves
        else shape,
    )
    if variant == "reference":
        return _with_reference_vjp(forward, reference)
    backward = call(
        backward_kernel,
        grid=(n,),
        in_specs=(state_spec, params_spec, checkpoint_spec, state_spec),
        out_specs=(state_spec, params_spec),
        out_shape=(shape, jax.ShapeDtypeStruct((n, count), jnp.float32)),
    )
    return palladium.with_vjp(forward, backward, residuals=1)


def make_log_prob(n, **kwargs):
    """Build a CNF log-density callable for a shared parameter vector.

    Parameters
    ----------
    n : int
        Number of ensemble members or examples.
    **kwargs : dict
        Options forwarded to ``make_flow``, such as width, steps, checkpoint interval,
        variant, and call factory.

    Returns
    -------
    callable
        Function (parameters, points) -> (n,) log densities for points of shape (n, 2).

    Notes
    -----
    Density evaluation appends a zero divergence state, broadcasts weights, and subtracts
    the reverse-time log-density increment from a standard-normal base log density.
    """
    flow = make_flow(n, **kwargs)

    def log_prob(parameters, points):
        states = jnp.concatenate((points, jnp.zeros((n, 1), points.dtype)), axis=1)
        weights = jnp.broadcast_to(parameters, (n, parameters.shape[0]))
        result = flow(states, weights)
        base = -math.log(2 * math.pi) - 0.5 * jnp.sum(result[:, :2] ** 2, axis=1)
        return base - result[:, 2]

    return log_prob


def make_training_step(n, *, learning_rate=0.01, **kwargs):
    """Build a jitted Adam update and mean negative-log-likelihood evaluation.

    Parameters
    ----------
    n : int
        Number of ensemble members or examples.
    learning_rate : float, optional
        Adam update scale, default 0.01.
    **kwargs : dict
        Options forwarded to ``make_flow``, such as width, steps, checkpoint interval,
        variant, and call factory.

    Returns
    -------
    update : callable
        Function (optimizer_state, points) -> (new_state, pre_update_loss).
    loss : callable
        Jitted function (parameters, points) -> scalar mean negative log density.

    Notes
    -----
    Use initial_state to create the parameter, first-moment, second-moment, and step-count
    tuple. Data shape is fixed to (n, 2).
    """
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
    """Create the parameter and zero-moment state for the example Adam optimizer.

    Parameters
    ----------
    width : int, optional
        Positive hidden-layer width, default 4.
    seed : int, optional
        Deterministic random seed, default 17.

    Returns
    -------
    tuple
        Parameters, first moment, second moment, and int32 step count, in update-call
        order.
    """
    parameters = init_parameters(width, seed)
    return (
        parameters,
        jnp.zeros_like(parameters),
        jnp.zeros_like(parameters),
        jnp.int32(0),
    )


def mixture_data(n, seed=17):
    """Sample the fixed two-component Gaussian mixture used for CNF training.

    Parameters
    ----------
    n : int
        Number of ensemble members or examples.
    seed : int, optional
        Deterministic random seed, default 17.

    Returns
    -------
    numpy.ndarray
        Float32 points of shape (n, 2), with equally likely components and fixed axis
        scales.
    """
    rng = np.random.default_rng(seed)
    means = np.array([[-1.3, -0.4], [1.3, 0.4]])
    samples = means[rng.integers(2, size=n)] + rng.normal(size=(n, 2)) * [0.45, 0.65]
    return samples.astype(np.float32)


def mixture_log_prob(points):
    """Evaluate the exact log density of the training mixture with a stable log-sum-exp.

    Parameters
    ----------
    points : jax.Array
        Points of shape (n, 2), with floating dtype.

    Returns
    -------
    jax.Array
        One log density per point, shape (n,).
    """
    means = jnp.array([[-1.3, -0.4], [1.3, 0.4]], points.dtype)
    scales = jnp.array([0.45, 0.65], points.dtype)
    component = (
        -math.log(2 * math.pi)
        - jnp.sum(jnp.log(scales))
        - 0.5 * jnp.sum(((points[:, None, :] - means) / scales) ** 2, axis=-1)
    )
    return jax.scipy.special.logsumexp(component, axis=1) - math.log(2)


def main():
    """Train the example CNF from command-line options and print held-out and refinement
    diagnostics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--variant", choices=("reverse", "reference", "jax"), default="reverse")
    args = parser.parse_args()
    with jax.default_device(jax.devices(args.device)[0]):
        train = jnp.asarray(mixture_data(args.n, 17))
        heldout = jnp.asarray(mixture_data(args.n, 29))
        state = initial_state(args.width)
        update, loss = make_training_step(
            args.n,
            width=args.width,
            steps=args.steps,
            interval=args.interval,
            variant=args.variant,
            call=palladium.metal_call if args.device == "cpu" else pl.pallas_call,
            learning_rate=args.learning_rate,
        )
        start = time.perf_counter()
        jax.block_until_ready(update(state, train))
        first_ms = (time.perf_counter() - start) * 1000
        initial_nll = float(loss(state[0], heldout))
        curve = []
        start = time.perf_counter()
        for index in range(args.iterations):
            state, value = update(state, train)
            if (index + 1) % 100 == 0:
                curve.append({"iteration": index + 1, "pre_update_train_nll": float(value)})
        jax.block_until_ready(state)
        train_seconds = time.perf_counter() - start
        final_nll = float(loss(state[0], heldout))
        params = np.asarray(state[0])
    # Evaluate the trained weights with an independent backend and finer time grid.
    with jax.default_device(jax.devices("cpu")[0]):
        _, reference_loss = make_training_step(
            args.n, width=args.width, steps=args.steps, variant="jax"
        )
        _, refined_loss = make_training_step(
            args.n, width=args.width, steps=2 * args.steps, variant="jax"
        )
        cpu_data, cpu_params = (
            jnp.asarray(mixture_data(args.n, 29)),
            jnp.asarray(params),
        )
        reference_nll = float(reference_loss(cpu_params, cpu_data))
        refined_nll = float(refined_loss(cpu_params, cpu_data))
        target_nll = -float(jnp.mean(mixture_log_prob(cpu_data)))
    print(
        json.dumps(
            {
                "config": vars(args),
                "first_training_call_ms": first_ms,
                "training_seconds": train_seconds,
                "initial_heldout_nll": initial_nll,
                "final_heldout_nll": final_nll,
                "cpu_reference_nll": reference_nll,
                "cpu_double_steps_nll": refined_nll,
                "true_distribution_nll": target_nll,
                "training_curve": curve,
                "parameters": params.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
