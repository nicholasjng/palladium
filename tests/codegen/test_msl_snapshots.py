"""Golden-MSL snapshots: pin the emitted text, not only its behavior.

Cursor's name counter is deterministic, so snapshots are stable across runs.
Regenerate after an intended emitter change:

    PALLADIUM_REGEN_GOLDEN=1 uv run pytest tests/codegen/test_msl_snapshots.py

No GPU is needed; a failure on CI without a Metal device means a JAX upgrade
changed kernel staging.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from kernels import lv_kernel

import palladium

GOLDEN = Path(__file__).parent / "golden"


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
    n = 64
    spec_1 = pl.BlockSpec((1,), lambda i: (i,))
    f = pl.pallas_call(
        lv_kernel,
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
    # A comparison and a select_n reached through jnp.where's jit wrapper,
    # inside a loop with a const; the jit staging is a JAX implementation
    # detail that an upgrade may change.
    def kernel(y0_ref, r_ref, o_ref):
        r = r_ref[...]

        def step(_, y):
            grown = y + r
            return jnp.where(grown <= 1.0, grown, y)

        o_ref[...] = jax.lax.fori_loop(0, 20, step, y0_ref[...])

    f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((64,), jnp.float32))
    x = np.zeros(64, np.float32)
    return palladium.trace(f, x, x)


def _dense_output_scan():
    # A scanned xs sliced per step and a stacked ys streaming straight to the
    # output ref, with the consuming swap degenerating to a no-op.
    def kernel(y0_ref, ts_ref, o_ref):
        def step(y, t):
            y_next = y + 0.1 * t * y
            return y_next, y_next

        _, ys = jax.lax.scan(step, y0_ref[...], ts_ref[...])
        o_ref[...] = ys

    f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((16, 4), jnp.float32))
    return palladium.trace(f, np.zeros(4, np.float32), np.zeros(16, np.float32))


SNAPSHOTS = {
    "copy_2d": _copy_2d,
    "blocked_saxpy": _blocked_saxpy,
    "rk4_lotka_volterra": _rk4_lotka_volterra,
    "conditional_loop": _conditional_loop,
    "dense_output_scan": _dense_output_scan,
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
        "PALLADIUM_REGEN_GOLDEN=1 uv run pytest tests/codegen/test_msl_snapshots.py"
    )
    assert msl == path.read_text(), (
        f"emitted MSL for '{name}' drifted from its snapshot; if the change "
        "is intended, regenerate with PALLADIUM_REGEN_GOLDEN=1 and review "
        "the diff"
    )
