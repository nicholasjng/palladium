"""Provided, green from day one: tracing Pallas into KernelSpecs.

Also your map of the territory: run with -s and read the printed jaxprs —
every exercise rule consumes exactly these structures.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from palladium import trace


def _mad_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0 + y_ref[...]


def test_gridless_spec(rng):
    f = pl.pallas_call(
        _mad_kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
    x = rng.standard_normal((8, 128), dtype=np.float32)
    spec = trace(f, x, x)

    assert spec.grid == (1,)
    assert [i.block_shape for i in spec.inputs] == [(8, 128), (8, 128)]
    assert spec.outputs[0].array_shape == (8, 128)
    assert spec.outputs[0].dtype == np.float32
    names = [e.primitive.name for e in spec.jaxpr.eqns]
    assert names == ["get", "mul", "get", "add", "swap"]
    print("\n", spec.jaxpr)


def test_gridded_spec_and_loop_staging(rng):
    def kernel(x_ref, o_ref):
        y = x_ref[...]
        y = jax.lax.fori_loop(0, 10, lambda i, c: c + 0.1 * c, y)
        o_ref[...] = y

    f = pl.pallas_call(
        kernel,
        grid=(16,),
        in_specs=[pl.BlockSpec((8,), lambda i: (i,))],
        out_specs=pl.BlockSpec((8,), lambda i: (i,)),
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
    )
    x = rng.standard_normal(128, dtype=np.float32)
    spec = trace(f, x)

    assert spec.grid == (16,)
    assert spec.num_programs == 16
    assert spec.inputs[0].block_shape == (8,)
    assert spec.inputs[0].array_shape == (128,)
    names = [e.primitive.name for e in spec.jaxpr.eqns]
    # fori_loop stages as a pure-carry scan: this is what exercise 4 lowers.
    assert "scan" in names
    scan_eqn = next(e for e in spec.jaxpr.eqns if e.primitive.name == "scan")
    assert scan_eqn.params["length"] == 10
    assert len(scan_eqn.invars) == len(scan_eqn.outvars)
    # The BlockSpec index map is itself a jaxpr; exercise 3 evaluates it.
    imj = spec.inputs[0].index_map_jaxpr.jaxpr
    assert len(imj.invars) == 1 and not imj.eqns
    print("\n", spec.jaxpr)


def test_interpret_mode_is_the_oracle(rng):
    """interpret=True runs the kernel on CPU — the reference for every
    exercise. (Without it, pallas_call refuses to run on CPU at all.)"""
    f = pl.pallas_call(
        _mad_kernel,
        out_shape=jax.ShapeDtypeStruct((32,), jnp.float32),
        interpret=True,
    )
    x = rng.standard_normal(32, dtype=np.float32)
    y = rng.standard_normal(32, dtype=np.float32)
    np.testing.assert_allclose(f(x, y), 2.0 * x + y, rtol=1e-6)


def test_trace_rejects_multiple_pallas_calls(rng):
    f = pl.pallas_call(_mad_kernel, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32))

    def two_calls(x, y):
        return f(x, y) + f(y, x)

    x = rng.standard_normal(4, dtype=np.float32)
    with pytest.raises(ValueError, match="2 pallas_call"):
        trace(two_calls, x, x)
