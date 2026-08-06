"""Golden-MSL snapshots: pin the emitted text, not just its behavior.

The oracle tests prove kernels compute the right values; these prove the
emitter produces the *same text* it did when the snapshot was blessed.
Unintended codegen drift (a qualifier change, a lost offset factor, an
extra copy) shows up as a reviewable diff instead of silence. EmitState's name
counter is deterministic, so snapshots are stable across runs.

Regenerate after an *intended* emitter change:

    PALLADIUM_REGEN_GOLDEN=1 uv run pytest tests/test_msl_snapshots.py

then review the diff in jj like any other code change. These tests need no
GPU (pure text comparison) and run on CI runners without a Metal device;
the canary job failing here means a JAX upgrade changed kernel staging.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

import palladium

GOLDEN = Path(__file__).parent / "golden"

DT, STEPS = 0.01, 500


def _copy_2d():
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((16, 32), jnp.float32))
    return palladium.trace(f, np.zeros((16, 32), np.float32))


def _blocked_saxpy():
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = 2.5 * x_ref[...] + y_ref[...]

    spec_8 = pl.BlockSpec((8,), lambda i: (i,))
    f = pl.pallas_call(
        kernel,
        grid=(32,),
        in_specs=[spec_8, spec_8],
        out_specs=spec_8,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
    )
    x = np.zeros(256, np.float32)
    return palladium.trace(f, x, x)


def _rk4_lotka_volterra():
    def kernel(x_ref, y_ref, a_ref, b_ref, c_ref, d_ref, xo_ref, yo_ref):
        a, b, c, d = a_ref[...], b_ref[...], c_ref[...], d_ref[...]

        def rhs(x, y):
            return a * x - b * x * y, c * x * y - d * y

        def step(_, carry):
            x, y = carry
            k1x, k1y = rhs(x, y)
            k2x, k2y = rhs(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y)
            k3x, k3y = rhs(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y)
            k4x, k4y = rhs(x + DT * k3x, y + DT * k3y)
            return (
                x + DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
                y + DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            )

        x, y = jax.lax.fori_loop(0, STEPS, step, (x_ref[...], y_ref[...]))
        xo_ref[...] = x
        yo_ref[...] = y

    n = 64
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    f = pl.pallas_call(
        kernel,
        grid=(n,),
        in_specs=[spec_1] * 6,
        out_specs=(spec_1, spec_1),
        out_shape=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.float32),
        ),
    )
    args = [np.zeros(n, np.float32)] * 6
    return palladium.trace(f, *args)


def _conditional_loop():
    # Pins the adaptive-controller vocabulary in one kernel: a comparison, select_n
    # reached through jnp.where's jit wrapper (inlined by _inline_jit),
    # and both inside a scan with a const. The jit staging is a JAX
    # implementation detail; if an upgrade changes it, this golden turns
    # the canary red before any GPU sees the difference.
    def kernel(y0_ref, r_ref, o_ref):
        r = r_ref[...]

        def step(_, y):
            grown = y + r
            return jnp.where(grown <= 1.0, grown, y)

        o_ref[...] = jax.lax.fori_loop(0, 20, step, y0_ref[...])

    f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32))
    x = np.zeros(64, np.float32)
    return palladium.trace(f, x, x)


SNAPSHOTS = {
    "copy_2d": _copy_2d,
    "blocked_saxpy": _blocked_saxpy,
    "rk4_lotka_volterra": _rk4_lotka_volterra,
    "conditional_loop": _conditional_loop,
}


@pytest.mark.parametrize("name", SNAPSHOTS)
def test_emitted_msl_matches_golden(name):
    msl = palladium.emit_msl(SNAPSHOTS[name]())
    path = GOLDEN / f"{name}.metal"
    if os.environ.get("PALLADIUM_REGEN_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(msl)
        pytest.skip(f"regenerated {path.name}")
    assert path.exists(), (
        f"missing snapshot {path.name}; bless it with "
        "PALLADIUM_REGEN_GOLDEN=1 uv run pytest tests/test_msl_snapshots.py"
    )
    assert msl == path.read_text(), (
        f"emitted MSL for '{name}' drifted from its snapshot; if the change "
        "is intended, regenerate with PALLADIUM_REGEN_GOLDEN=1 and review "
        "the diff"
    )
