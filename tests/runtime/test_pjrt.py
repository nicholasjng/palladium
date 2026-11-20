"""The `metal` PJRT plugin through jax. jax picks its platforms once per process,
so each test runs a script in a subprocess with JAX_PLATFORMS set."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

# Scripts may import `pjrt_tolerance`, the agreement rule, from this
# directory, and the workloads in examples/.
_DIRECTORIES = [Path(__file__).resolve().parent, Path(__file__).resolve().parents[2] / "examples"]
_PATH = f"import sys; sys.path[:0] = {[str(d) for d in _DIRECTORIES]!r}\n"


def _run(script: str, platforms: str | None = "metal,cpu") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    if platforms:
        env["JAX_PLATFORMS"] = platforms
    return subprocess.run(
        [sys.executable, "-c", _PATH + textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _check(script: str, **kwargs) -> str:
    result = _run(script, **kwargs)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_device_and_attributes():
    out = _check("""
        import jax
        from palladium.device import device_limits
        [d] = jax.devices("metal")
        assert d.platform == "metal" and d.id == 0 and d.process_index == 0
        assert d.device_kind == device_limits()["name"], d.device_kind
        assert d.apple_gpu_family == device_limits()["apple_gpu_family"]
        assert d.default_memory().kind == "device"
        stats = d.memory_stats()
        assert stats["bytes_limit"] > 0 and stats["bytes_in_use"] >= 0
        print(jax.default_backend())
    """)
    assert out.strip() == "metal"


def test_buffers_round_trip():
    _check("""
        import jax, ml_dtypes, numpy as np
        [d] = jax.devices("metal")
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        arrays = [
            x, x.transpose(2, 0, 1), x[:, ::2, 1:], np.float32(3), np.zeros((0, 5), np.float16),
            np.array([True, False]), np.arange(6, dtype=np.uint8),
            np.arange(4, dtype=np.int32), np.arange(5).astype(ml_dtypes.bfloat16),
            np.arange(4, dtype=np.complex64),
        ]
        for a in arrays:
            y = jax.device_put(a, d)
            assert y.devices() == {d}
            back = np.asarray(y)
            assert back.dtype == a.dtype and back.shape == a.shape, (a, back)
            np.testing.assert_array_equal(back, a)
        y = jax.device_put(x, d)
        np.testing.assert_array_equal(jax.device_put(y, jax.devices("cpu")[0]), x)
        y.delete()
        assert y.is_deleted()
    """)


def test_programs_match_cpu():
    """Fused elementwise and index-mapping programs, against the CPU backend."""
    out = _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from pjrt_tolerance import assert_matches_cpu
        metal, cpu = jax.devices("metal")[0], jax.devices("cpu")[0]
        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 4), np.float32)
        v = rng.standard_normal(4, np.float32)
        i = rng.integers(-50, 50, (3, 4), np.int32)
        u = rng.integers(0, 1 << 31, (5,), np.uint32)
        cases = {
            "unary": (lambda x: jnp.exp(x) + jnp.tanh(x) - jnp.log1p(jnp.abs(x)), x),
            "broadcast": (lambda x, v: x * v - jnp.maximum(x, v) / 3 + x[:, :1], x, v),
            "where": (lambda x: jnp.where(x > 0, x, 0.1 * x), x),
            "index maps": (lambda x: x.T.reshape(2, 6)[::-1, 1::2] + jnp.arange(3.0), x),
            "int": (lambda i: (i * 3 - 7) // 2 + (i << 1) - i % 5, i),
            "uint": (lambda u: (u >> 3) ^ (u << 5), u),
            "bool": (lambda x: (x > 0) & ~(x > 1), x),
            "convert": (lambda x: (x * 10).astype(jnp.int32), x),
            "half": (lambda h: h * 2 + 1, x.astype(np.float16)),
            "bfloat16": (lambda b: b * 2 + 1, x.astype(ml_dtypes.bfloat16)),
            "erf composite": (jax.scipy.special.erf, x),
            "gelu": (jax.nn.gelu, x),
            "outputs": (lambda x, v: (x, v * 2, x + jnp.array([1.0, 2.0, 3.0, 4.0])), x, v),
            "no arguments": (lambda: jnp.arange(5) * 2,),
            "zero size": (lambda x: x * 2, np.zeros((0, 3), np.float32)),
            "scalar predicate": (lambda p, x: jax.lax.select(p, x, -x), np.bool_(True), x),
            "double negation": (lambda x: jax.lax.neg(jax.lax.neg(x)), x),
            "negated negative constants": (
                lambda x: x + jax.lax.neg(jax.lax.full((3, 4), -1.5))
                - jax.lax.neg(jax.lax.full((3, 4), -np.inf)), x),
            "cbrt": (jnp.cbrt, np.array([-27, -1e-3, -0.0, 0.0, 0.5, 1e6, np.inf, np.nan], np.float32)),
            "cbrt bfloat16": (jnp.cbrt, (x * 9).astype(ml_dtypes.bfloat16)),
            "scalar bounds": (lambda a, x, b: jax.lax.clamp(a, x, b),
                              np.float32(-0.5), x, np.float32(0.5)),
        }
        for name, (fn, *args) in cases.items():
            got = assert_matches_cpu(fn, *args, err_msg=name)
            assert all(g.devices() == {metal} for g in jax.tree.leaves(got)), name
        print(len(cases))
    """)
    assert out.strip() == "21"


def test_executions_run_ahead_of_the_gpu():
    """Chained calls queue without waiting; an input deleted while its call is in
    flight, and the uploads reusing pooled memory after it, leave the result intact."""
    _check("""
        import jax, numpy as np
        metal = jax.devices("metal")[0]
        step = jax.jit(lambda y: y * 0.5 + 1)
        y, want = jax.device_put(np.zeros(1 << 20, np.float32), metal), np.zeros(1 << 20, np.float32)
        for _ in range(64):
            y, want = step(y), want * np.float32(0.5) + 1
        np.testing.assert_array_equal(np.asarray(y), want)

        x = np.arange(1 << 20, dtype=np.float32)
        results = []
        for k in range(32):
            a = jax.device_put(x + k, metal)
            results.append(step(a))
            a.delete()
            jax.device_put(np.full_like(x, -1), metal)  # same size: pooled memory
        for k, r in enumerate(results):
            np.testing.assert_array_equal(np.asarray(r), (x + k) * 0.5 + 1)
    """)


def test_guarded_index_maps_match_cpu():
    """Guarded operands may themselves contain reads, reshapes and guards."""
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        y = np.arange(8, dtype=np.float32).reshape(2, 4) + 20
        cases = [
            (lambda x, y: jnp.concatenate([x * 2, y + 1, x], 0), x, y),
            (lambda x: jnp.concatenate([x[:, :0], x, x[:, :1], x[:, :0]], 1), x),
            (lambda x, p: lax.pad(x * 2, p, ((2, 1, 1), (1, 3, 2))), x, np.float32(-7)),
            (lambda x, p: lax.pad(x, p, ((-1, 2, 1), (2, -3, 2))), x, np.float32(9)),
            (lambda x, p: lax.pad(x, p, ((2, 3, 0), (1, 1, 0))),
             np.zeros((0, 4), np.float32), np.float32(5)),
            (lambda x: jnp.concatenate([x, x], 0), np.zeros((0, 4), np.float32)),
            (lambda x: lax.pad(x, np.float32(0), ((-3, 0, 0), (0, 0, 0))), x),
            (lambda x, y: jnp.concatenate([
                lax.pad(x.T.reshape(2, 6), np.float32(-1), ((1, 0, 0), (0, 0, 0))),
                y.T.reshape(1, 8)[:, :6], x.reshape(2, 6)], 0), x, y),
            # A branch's reshape shortcut must not affect later uses of x.
            (lambda x: (jnp.concatenate([x.T.reshape(2, 6), x.reshape(2, 6)], 0),
                        x.T.reshape(2, 6) + x.reshape(2, 6)), x),
        ]
        for dtype in (np.int32, np.uint32):
            for starts in ((0, 0), (1, 2), (100, 100), ((-100, -1) if dtype == np.int32
                                                     else (2**32 - 1, 2**32 - 1))):
                i, j = map(dtype, starts)
                cases += [
                    (lambda x, i, j: lax.dynamic_slice(x * 2, (i, j), (2, 2)), x, i, j),
                    (lambda x, y, i, j: lax.dynamic_update_slice(x + 1, y * 3, (i, j)),
                     x, y[:, :2], i, j),
                    (lambda x, i, j: lax.dynamic_slice(x, (i, j), (0, 2)), x, i, j),
                    (lambda x, y, i, j: lax.dynamic_update_slice(x, y, (i, j)),
                     x, y[:0], i, j),
                ]
        cases += [
            (lambda x, y: lax.dynamic_update_slice(x, y, ()), np.float32(2), np.float32(3)),
            (lambda x: lax.dynamic_slice(x, (), ()), np.float32(2)),
            (lambda x: lax.pad(x, np.float32(-1), ()), np.float32(2)),
            (lambda x, i: lax.dynamic_slice(x, (i,), (2,), allow_negative_indices=False),
             np.arange(300, dtype=np.float32), np.int32(-100)),
            (lambda x, i: lax.dynamic_slice(x, (i,), (2,)),
             np.arange(300, dtype=np.float32), np.uint32(250)),
            (lambda x, y, i: lax.dynamic_update_slice(x, y, (i,), allow_negative_indices=False),
             np.arange(300, dtype=np.float32), np.ones(2, np.float32), np.int32(-100)),
            # A value hoisted out of the inner branch reads a scalar that hoists further.
            (lambda v, s, b, c: jnp.concatenate([jnp.concatenate([
                jnp.broadcast_to((v * s)[:, None], b.shape), b], 1), c * s], 0),
             x[:, 0], np.float32(2), x, np.concatenate([y, y], 1)),
        ]
        for dtype in (np.int32, np.bool_, np.float16):
            cases += [
                (lambda x, y: jnp.concatenate([x, y, x], 0), x.astype(dtype), y.astype(dtype)),
                (lambda x, p: lax.pad(x, p, ((1, 2, 1), (2, -1, 0))),
                 x.astype(dtype), dtype(1)),
            ]
        for fn, *args in cases:
            assert_matches_cpu(fn, *args, exact=True)
    """)


def test_random_key_can_be_created_on_metal():
    _check("""
        import jax, numpy as np
        key = jax.random.key(17)
        assert key.devices() == {jax.devices('metal')[0]}
        np.testing.assert_array_equal(np.asarray(jax.random.key_data(key)), [0, 17])
    """)


def test_sharding_constraints_and_optimization_barriers_are_identities():
    # Eager jax.random.normal on a key wraps it in sdy.sharding_constraint;
    # jax.checkpoint emits stablehlo.optimization_barrier.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        key = jax.random.key(3)
        np.testing.assert_allclose(
            np.asarray(jax.random.normal(key, (64, 32))),
            np.asarray(jax.random.normal(jax.device_put(key, cpu), (64, 32))),
            rtol=2e-5, atol=1e-6)
        f = jax.jit(jax.grad(jax.checkpoint(lambda x: jnp.sum(jnp.sin(x) ** 2))))
        x = np.linspace(-2, 2, 48, dtype=np.float32)
        assert 'optimization_barrier' in f.lower(x).as_text()
        np.testing.assert_allclose(np.asarray(f(jax.device_put(x, metal))),
                                   np.asarray(f(jax.device_put(x, cpu))), rtol=1e-5, atol=1e-6)
    """)


def test_reductions_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        rng = np.random.default_rng(5)
        cases = []
        for shape in ((3, 5, 37), (2, 10001), (10001,), (3, 0, 5), (0, 5), (2, 1)):
            x = rng.uniform(-1, 1, shape).astype(np.float32)
            for axes in [(i,) for i in range(len(shape))] + [tuple(range(len(shape)))]:
                cases += [(lambda x, axes=axes: (x * 0.5 + 1).sum(axes), x)]
                if all(shape[i] for i in axes):
                    cases += [(lambda x, axes=axes: (x.max(axes), x.min(axes)), x)]
        x = rng.uniform(0.1, 1, (7, 129)).astype(np.float32)
        cases += [
            (lambda x: jax.nn.softmax(x, -1), x),
            (lambda x: (x - x.mean(-1, keepdims=True)) /
                       jnp.sqrt(x.var(-1, keepdims=True) + 1e-5), x),
            (lambda x: (x.sum(0).sum(), x.sum(-1).sum()), x),
            (lambda x: jnp.sum(jnp.pad(x, ((1, 2), (3, 0))) ** 2, 0), x),
            (lambda x: lax.reduce(x, np.float32(5), lax.max, (1,)), x),
            (lambda x: lax.reduce(x, np.float32(5), lax.add, (0,)), np.empty(0, np.float32)),
            (lambda x: x.sum((0, 2)), np.arange(231, dtype=np.float32).reshape(3, 7, 11)),
            (lambda x: x.T.sum(0), x),
        ]
        for dtype in (np.int32, np.uint32, np.bool_, np.float16, ml_dtypes.bfloat16):
            x = rng.integers(0, 8, (5, 257)).astype(dtype)
            cases += [(lambda x: (x.sum(0), x.max(1), x.min(0)), x)]
            if dtype in (np.int32, np.uint32, np.bool_):
                cases += [(lambda x: (jnp.bitwise_and.reduce(x, 0),
                                      jnp.bitwise_or.reduce(x, 1)), x)]
        for fn, x in cases:
            assert_matches_cpu(fn, x)
    """)


def test_reduction_intermediates_survive_queued_executions():
    _check("""
        import jax, numpy as np
        metal = jax.devices('metal')[0]
        run = jax.jit(lambda x: (x - x.mean(-1, keepdims=True)).sum(-1))
        outputs = []
        for i in range(40):
            x = jax.device_put(np.full((3, 10001), i, np.float32), metal)
            outputs.append(run(x))
            x.delete()
        for out in outputs:
            np.testing.assert_allclose(np.asarray(out), 0, atol=0.1)
    """)


def test_reduction_special_values_and_integer_extrema():
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        for n in (4, 37, 10001):  # 4: folded into the consumer
            x = np.zeros((5, n), np.float32)
            x[0, 1] = np.nan
            x[1, 0], x[1, -1] = -np.inf, np.inf
            x[2, :] = -0.0
            x[3, 1] = -0.0
            x[4, :] = np.inf
            fn = jax.jit(lambda x: (x.min(-1), x.max(-1), x.sum(-1)))
            got = fn(jax.device_put(x, metal))
            # Use NumPy for special values: XLA:CPU can discard NaNs in extrema.
            with np.errstate(invalid='ignore'):
                want = x.min(-1), x.max(-1), x.sum(-1)
            for g, w in zip(got, want, strict=True):
                np.testing.assert_allclose(np.asarray(g), np.asarray(w), equal_nan=True)
            # Check signed-zero ties explicitly; XLA may choose different zero signs.
            minimum, maximum = map(np.asarray, got[:2])
            assert np.signbit(minimum[2]) and np.signbit(maximum[2])
            assert np.signbit(minimum[3]) and not np.signbit(maximum[3])
        # Folded extrema of values computed in the same kernel: NaNs from
        # infinities and zero signs survive (fast math would lose both).
        y = np.array([[1, -1, 2, 3], [1, 2, 3, 4], [0.5, 0.25, -2, 8]], np.float32)
        for fn in (lambda y: jnp.log(y).max(-1), lambda y: jnp.log(y).min(-1),
                   lambda y: (y * jnp.inf - jnp.inf).max(-1), lambda y: (-0.0 * y).max(-1)):
            got = np.asarray(jax.jit(fn)(jax.device_put(y, metal)))
            want = np.asarray(jax.jit(fn)(jax.device_put(y, cpu)))
            np.testing.assert_array_equal(got, want)
            numbers = ~np.isnan(want)  # a NaN's sign is unspecified
            assert np.array_equal(np.signbit(got[numbers]), np.signbit(want[numbers]))
        for dtype in (np.int32, np.uint32):
            info = np.iinfo(dtype)
            x = np.array([info.min, info.max, info.max - 1, 2**25 + 1], dtype=dtype)
            fn = jax.jit(lambda x: (x.min(), x.max(), x.sum()))
            got, want = fn(jax.device_put(x, metal)), fn(jax.device_put(x, cpu))
            for g, w in zip(got, want, strict=True):
                np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
        # Empty reductions return their initializer; zero-size outputs do not dispatch.
        for fn, x in [
            (lambda x: lax.reduce(x, np.float32(-np.inf), lax.max, (1,)),
             np.empty((3, 0), np.float32)),
            (lambda x: lax.reduce(x, np.float32(np.inf), lax.min, (1,)),
             np.empty((3, 0), np.float32)),
        ]:
            run = jax.jit(fn)
            np.testing.assert_array_equal(np.asarray(run(jax.device_put(x, metal))),
                                          np.asarray(run(jax.device_put(x, cpu))))
    """)


def test_static_loops_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        x = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
        cases = []
        for length in (0, 1, 2, 3, 8, 200):
            cases += [
                (lambda x, length=length: lax.fori_loop(0, length, lambda i, y: y * 0.9 + i, x), x),
                (lambda x, length=length: lax.scan(lambda y, _: (y * 0.9 + 1, y * 2),
                                                  x, None, length=length), x),
                # Three-way rotation exposes carry/output aliasing with two banks.
                (lambda x, length=length: lax.fori_loop(0, length,
                    lambda i, s: (s[1], s[2], s[0] + i), (x, x + 1, x + 2)), x),
            ]
        cases += [
            (lambda x: lax.while_loop(lambda s: s[0] < 5,
                lambda s: (s[0] + 1, s[1] + s[2], s[2]), (-3, x, x * 2)), x),
            (lambda x: lax.fori_loop(0, 3, lambda i, y:
                lax.fori_loop(0, 4, lambda j, z: z * 0.5 + j, y) + i, x), x),
            (lambda x: lax.fori_loop(0, 4, lambda i, y:
                y - y.mean(-1, keepdims=True) + i, x), x),
            # A scan with xs exercises dynamic slices and the trajectory update.
            (lambda x: lax.scan(lambda c, row: (c + row, c * 2), jnp.zeros(4), x), x),
            (lambda x: lax.scan(lambda c, row: (c + row, c * 2), jnp.zeros(4), x,
                               reverse=True), x),
            (lambda x: lax.fori_loop(0, 4, lambda i, y: y + 1, x), np.zeros((0, 3), np.float32)),
            # A carry also read besides its update must not be updated in place.
            (lambda x: lax.fori_loop(1, 3, lambda i, t: lax.dynamic_update_slice(
                t, lax.dynamic_slice(t, (i - 1, 0), (1, 4)) * 2 + 1, (i, 0)), x), x),
            # An inner trajectory is re-initialized on every outer iteration.
            (lambda x: lax.fori_loop(0, 3, lambda i, a: a + lax.scan(
                lambda c, _: (c * 0.5 + i, c), a[0], None, length=3)[1], x), x),
            # Induction variables: non-unit steps and int32 wrap-around.
            (lambda x: lax.while_loop(lambda s: s[0] < 6,
                lambda s: (s[0] + 1, s[1] + 3, s[2] + s[1]), (0, 5, x))[1:], x),
            (lambda x: lax.while_loop(lambda s: s[0] < 5,
                lambda s: (s[0] + 1, s[1] + np.int32(1), s[2] + (s[1] % 7)),
                (0, np.int32(2**31 - 2), x)), x),
            (lambda x: lax.while_loop(lambda s: s[0] < np.uint32(2**32 - 1),
                lambda s: (s[0] + np.uint32(1), s[1] + 1),
                (np.uint32(2**32 - 3), x)), x),
        ]
        for fn, x in cases:
            assert_matches_cpu(fn, x)
    """)


def test_static_loop_buffers_survive_queued_calls():
    _check("""
        import jax, numpy as np
        from jax import lax
        device = jax.devices('metal')[0]
        run = jax.jit(lambda x: lax.fori_loop(0, 17, lambda i, y: y + 1, x))
        results = []
        for i in range(40):
            x = jax.device_put(np.full((4096,), i, np.float32), device)
            results.append(run(x))
            x.delete()
        for i, y in enumerate(results):
            np.testing.assert_array_equal(np.asarray(y), np.full((4096,), i + 17, np.float32))
    """)


def test_sbi_trajectory_and_summary_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from training_steps import sbi_trajectory
        from pjrt_tolerance import assert_normwise_close
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        theta = np.random.default_rng(17).uniform(0.5, 1.5, (512, 4)).astype(np.float32)
        def simulate(theta):
            traj = sbi_trajectory(theta)
            return traj, jnp.concatenate([traj.mean(0), traj.std(0), traj[-1]], -1)
        run = jax.jit(simulate)
        got, want = run(jax.device_put(theta, metal)), run(jax.device_put(theta, cpu))
        for g, w in zip(got, want, strict=True):
            assert_normwise_close(g, w)
    """)


def test_dot_general_matches_cpu():
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        rng = np.random.default_rng(71)
        def array(shape):
            return rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        cases = []
        for m, n, k in ((32, 48, 64), (17, 19, 7), (1, 1, 13), (512, 128, 6),
                        (128, 5, 128), (3, 5, 0), (0, 5, 3), (3, 0, 5)):
            a, b = array((m,k)), array((k,n))
            cases += [(lambda a,b: a @ b, a,b), (lambda a,b: a.T @ b.T, a.T,b.T)]
        cases += [
            (lambda a,b: a @ b, array((3,32,16)), array((3,16,48))),
            (lambda a,b: a @ b, array((2,3,7,5)), array((2,3,5,9))),
            (lambda a,b: a @ b, array((2,1,7,5)), array((3,5,9))),
            (lambda a,b: a @ b, array((9,)), array((9,))),
            (lambda a,b: a @ b, array((3,9)), array((9,))),
            (lambda a,b: lax.dot_general(a,b,(((),()),((),()))), array((3,)), array((5,))),
            (lambda a,b: a @ b, array((0,3,7)), array((0,7,5))),
            (lambda a,b: lax.dot_general(a,b, (((2,0),(1,3)), ((1,),(0,)))),
             array((3,2,4,5)), array((2,4,7,3))),
            (lambda a,b: lax.dot_general(a,b, (((2,),(0,)), ((1,0),(2,1)))),
             array((2,3,4,5)), array((4,2,3,6))),
            (lambda a,b: (jax.nn.softmax((a * 2) @ b, -1), (a @ b).sum()),
             array((17,7)), array((7,19))),
            (lambda a,b: lax.fori_loop(0,3,lambda i,x: x @ b, a),
             array((16,16)), array((16,16))),
            (lambda a,b: jax.value_and_grad(lambda x: jnp.sum((x @ b)**2))(a),
             array((17,7)), array((7,19))),
        ]
        for dtype in (np.float16, ml_dtypes.bfloat16):
            a,b = array((17,13)).astype(dtype), array((13,19)).astype(dtype)
            cases += [(lambda a,b:a@b, a,b),
                      (lambda a,b:lax.dot_general(a,b,(((1,),(0,)),((),())),
                                                 preferred_element_type=jnp.float32), a,b)]
        for index, (fn, a,b) in enumerate(cases):
            assert_matches_cpu(fn, a, b, err_msg=f'case {index}')
    """)


def test_sbi_dense_training_matches_cpu_over_ten_steps():
    _check("""
        import jax, numpy as np
        from training_steps import adamw, sbi_loss
        from pjrt_tolerance import assert_trees_close
        rng = np.random.default_rng(17)
        shapes = {'w1': (6,128), 'w2': (128,128), 'wa': (128,5),
                   'wm': (128,20), 'ws': (128,20)}
        params = {k: (rng.standard_normal(s) * 0.02).astype(np.float32) for k,s in shapes.items()}
        opt = (jax.tree.map(np.zeros_like, params), jax.tree.map(np.zeros_like, params), np.int32(0))
        x = rng.standard_normal((512,6)).astype(np.float32)
        theta = rng.uniform(0.5,1.5,(512,4)).astype(np.float32)
        @jax.jit
        def step(params,opt,x,theta):
            loss,grads = jax.value_and_grad(sbi_loss)(params,x,theta)
            return adamw(params,grads,opt),loss
        state = {name: jax.device_put((params,opt,x,theta),jax.devices(name)[0])
                 for name in ('metal','cpu')}
        for _ in range(10):
            losses = {}
            for name,(p,o,a,t) in state.items():
                (p,o),losses[name] = step(p,o,a,t)
                state[name] = (p,o,a,t)
            # Normwise per leaf: near-zero entries cannot match componentwise.
            assert_trees_close((state['metal'][:2],losses['metal']),
                               (state['cpu'][:2],losses['cpu']),rtol=1e-4)
    """)


def test_wide_integer_operations_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        jax.config.update('jax_enable_x64', True)
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]
        for dtype in (np.int64,np.uint64):
            info = np.iinfo(dtype)
            a = np.array([info.min,info.max,2**32+1,2**53+1,0,1,17],dtype)
            b = np.array([1,2,3,5,0,63,64],dtype)
            shifts = np.array([0,1,31,32,63,64,info.max],dtype)
            cases = [
                (lambda a,b: (a+b,a-b,a*b,-a,jnp.abs(a)),a,b),
                (lambda a,b: (lax.div(a,b),lax.rem(a,b)),a,b),
                (lambda a,b: (a&b,a|b,a^b,~a,jnp.minimum(a,b),jnp.maximum(a,b)),a,b),
                (lambda a,b: (a<b,jnp.where(a<b,a,b),jnp.sign(a)),a,b),
                (lambda a,b: (lax.shift_left(a,b),lax.shift_right_logical(a,b),
                              lax.shift_right_arithmetic(a,b)),a,shifts),
                (lambda a,b: (lax.population_count(a),lax.clz(a)),a,b),
                (lambda a,b: (a.astype(jnp.uint32),a.astype(jnp.float32),
                              lax.bitcast_convert_type(a,jnp.uint64)),a,b),
                (lambda a,b: a+jnp.array([info.max,info.min,1,0,2**40,2**53,7],dtype),a,b),
                (lambda a,b: (a+dtype(info.max),a+dtype(info.min)),a,b),
            ]
            if dtype == np.int64:
                cases.append((lambda a,b:(lax.div(a,b),lax.rem(a,b)),a,np.full_like(a,-1)))
            for fn,a,b in cases:
                assert_matches_cpu(fn, a, b, exact=True)
    """)


def test_threefry_random_matches_cpu():
    _check("""
        import jax, numpy as np
        from pjrt_tolerance import assert_normwise_close
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]
        with jax.default_device(cpu):
            key=jax.random.key(17)
        for shape in ((),(0,),(1,),(7,),(512,6)):
            for fn in (lambda k:jax.random.uniform(k,shape),lambda k:jax.random.normal(k,shape),
                       lambda k:jax.random.bits(k,shape)):
                run=jax.jit(fn)
                got=run(jax.device_put(key,metal))
                want=run(jax.device_put(key,cpu))
                assert_normwise_close(got,want)
        # Key splitting and folding exercise the same counters with different layouts.
        for fn in (lambda k:jax.random.key_data(jax.random.split(k,7)),
                   lambda k:jax.random.key_data(jax.random.fold_in(k,23))):
            run=jax.jit(fn)
            np.testing.assert_array_equal(np.asarray(run(jax.device_put(key,metal))),
                                          np.asarray(run(jax.device_put(key,cpu))))
        with jax.enable_x64():
            run = jax.jit(lambda k:jax.random.bits(k,(7,),dtype=np.uint64))
            np.testing.assert_array_equal(np.asarray(run(jax.device_put(key,metal))),
                                          np.asarray(run(jax.device_put(key,cpu))))
    """)


def test_full_sbi_training_matches_cpu_over_ten_steps():
    _check("""
        import jax, numpy as np
        from training_steps import workload
        from pjrt_tolerance import assert_trees_close
        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            fn,params,opt,key = workload('sbi')
            keys = [jax.random.fold_in(key,i) for i in range(10)]
        state = {name:jax.device_put((params,opt),jax.devices(name)[0])
                 for name in ('metal','cpu')}
        run = jax.jit(fn)
        for key in keys:
            losses = {}
            for name,(p,o) in state.items():
                state[name],losses[name] = run(p,o,jax.device_put(key,jax.devices(name)[0]))
            # Every parameter and optimizer-state leaf, and the loss, per array.
            got = (state['metal'],losses['metal'])
            assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(got))
            assert_trees_close(got,(state['cpu'],losses['cpu']),rtol=1e-4)
    """)


def test_llm_training_step_matches_cpu():
    _check("""
        import jax,numpy as np
        from training_steps import workload
        from pjrt_tolerance import assert_matches_cpu
        with jax.default_device(jax.devices('cpu')[0]):
            fn,params,opt,_=workload('llm')
        tokens=np.arange(4*129,dtype=np.int32).reshape(4,129)%128
        got = assert_matches_cpu(fn, params, opt, tokens, rtol=1e-4)
        assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(got))
    """)


def test_cond_and_switch_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]
        x = np.arange(12,dtype=np.float32).reshape(3,4) / 10 + 0.1
        cases = [
            (lambda p,x:lax.cond(p>0,lambda y:y*2,lambda y:y-3,x),x),
            (lambda p,x:lax.switch(p,(lambda y:y,lambda y:y.T.reshape(y.shape),
                                     lambda y:y[::-1]+1),x),x),
            (lambda p,x:lax.cond(p>0,lambda y:(y,y.sum(),y[:,0]),
                                     lambda y:(y*3,y.max(),y[:,-1]),x),x),
            (lambda p,x:lax.cond(p>0,lambda y:lax.cond(p>1,lambda z:z+1,lambda z:z-1,y),
                                     lambda y:y*2,x),x),
            (lambda p,x:lax.cond(p>0,lambda y:y@ y.T,lambda y:y[:, :3]+1,x),x),
            (lambda p,x:lax.cond(p>0,lambda y:lax.fori_loop(0,3,lambda i,z:z+1,y),
                                     lambda y:y*2,x),x),
            (lambda p,x:lax.fori_loop(0,5,lambda i,y:lax.cond(i<p,lambda z:z*2,
                                                            lambda z:z+1,y),x),x),
            (lambda p,x:lax.cond(p>0,lambda y:jnp.log(y),lambda y:jnp.log(-y),x),x),
            (lambda p,x:lax.cond(p>0,lambda y:y.sum(),lambda y:jnp.log(-y).sum(),x),x),
            (lambda p,x:jax.grad(lambda y:lax.cond(p>0,lambda z:(z*z).sum(),
                                                   lambda z:(z*3).sum(),y))(x),x),
            (lambda p,x:lax.cond(x.sum()>p,lambda y:y*2,lambda y:y-2,x),x),
            (lambda p,x:lax.cond(p>0,lambda y:y+1,lambda y:y-1,x),np.zeros((0,3),np.float32)),
        ]
        for dtype in (np.int32,np.uint32,np.bool_,np.float16,ml_dtypes.bfloat16):
            cases.append((lambda p,x:lax.cond(p>0,lambda y:y,lambda y:y[::-1],x),x.astype(dtype)))
        for fn,x in cases:
            for p in (-100,0,1,2,100):
                assert_matches_cpu(fn, np.int32(p), x)
    """)


def test_raw_case_out_of_range_indices_select_last_branch():
    """JAX's switch clamps before lowering; raw StableHLO sends any index outside
    the branches to the last one. Runs the generated kernel directly: no jax."""
    from jaxlib.mlir.dialects import stablehlo

    from palladium._runtime import Kernel, run
    from palladium.stablehlo import compile_program

    module = """module {
      func.func @main(%index: tensor<i32>) -> tensor<i32> {
        %result = "stablehlo.case"(%index) ({
          %c = stablehlo.constant dense<10> : tensor<i32>
          stablehlo.return %c : tensor<i32>
        }, {
          %c = stablehlo.constant dense<20> : tensor<i32>
          stablehlo.return %c : tensor<i32>
        }, {
          %c = stablehlo.constant dense<30> : tensor<i32>
          stablehlo.return %c : tensor<i32>
        }) : (tensor<i32>) -> tensor<i32>
        return %result : tensor<i32>
      }
    }"""
    program = compile_program(stablehlo.serialize_portable_artifact_str(module, "1.0.0"))
    [launch] = program.launches
    kernel = Kernel(launch.source, launch.function)
    for index, expected in ((-100, 30), (-1, 30), (0, 10), (1, 20), (2, 30), (3, 30), (100, 30)):
        buffers = [np.array(index, np.int32), np.array(0, np.int32)]
        run(kernel, [buffers[s] for s in launch.slots], launch.grid)
        assert buffers[program.outputs[0]].item() == expected


def test_execution_stats_count_launches_and_gpu_time():
    _check("""
        import jax, numpy as np
        from jax import lax
        from palladium import _pjrt
        device = jax.devices('metal')[0]
        x = jax.device_put(np.ones(1 << 20, np.float32), device)
        loop = jax.jit(lambda x: lax.fori_loop(0, 5, lambda i, y: y * 2 + 1, x))
        passthrough = jax.jit(lambda x: x)
        jax.block_until_ready((loop(x), passthrough(x)))
        _pjrt.stats(reset=True)
        for _ in range(10):
            y = loop(x)
        jax.block_until_ready(y)
        # The same launches each call, at least one per replayed iteration;
        # the blit-only call has a command buffer but no launch.
        jax.block_until_ready(passthrough(x))
        s = _pjrt.stats(reset=True)
        assert s['executions'] == 11 and s['command_buffers'] == 11, s
        assert s['launches'] >= 50 and s['launches'] % 10 == 0, s
        assert 0 < s['gpu_busy_seconds'] <= s['gpu_sum_seconds'] + 1e-12, s
        assert s['encode_seconds'] > 0 and s['throttle_seconds'] >= 0, s
        assert _pjrt.stats()['executions'] == 0
    """)


def test_pool_evicts_stale_sizes_for_new_ones():
    # Buffers of sizes no longer requested must not keep new sizes out.
    _check("""
        import os
        os.environ['PALLADIUM_POOL_BYTES'] = str(64 << 20)
        import jax, numpy as np
        from palladium import _pjrt
        device = jax.devices('metal')[0]
        double = jax.jit(lambda x: x * 2)
        _pjrt.stats(reset=True)
        for n in range(8):  # 256 MiB of 16 MiB inputs and outputs, a new size each
            jax.block_until_ready(double(jax.device_put(np.ones((4 << 20) + n, np.float32), device)))
        assert _pjrt.stats()['pool_dropped'] > 0  # stale sizes were evicted
        x = jax.device_put(np.ones(3 << 22, np.float32), device)  # 12 MiB
        jax.block_until_ready(double(x))
        _pjrt.stats(reset=True)
        for _ in range(4):
            jax.block_until_ready(double(x))
        s = _pjrt.stats(reset=True)
        assert s['pool_misses'] == 0 and s['pool_hits'] == 4, s
        big = jax.device_put(np.ones(80 << 18, np.float32), device)  # larger than the pool
        del big
        assert _pjrt.stats()['pool_dropped'] == 1
    """)


def test_attention_gradient_inside_jit_matches_cpu():
    # The forward kernel, rowsum(dO * O) and both backward kernels in one program.
    _check("""
        import jax, numpy as np
        from flash_attention import differentiable_attention
        shape = (1, 128, 2, 64)
        rng = np.random.default_rng(0)
        q, k, v = (rng.standard_normal(shape, np.float32) for _ in range(3))
        for causal in (False, True):
            f = differentiable_attention(shape, causal=causal, tile_q=32)
            grad = jax.jit(jax.grad(lambda *a: (f(*a) ** 2).sum(), argnums=(0, 1, 2)))
            got = grad(*jax.device_put((q, k, v), jax.devices('metal')[0]))
            with jax.default_device(jax.devices('cpu')[0]):
                attention = lambda *a: jax.nn.dot_product_attention(*a, is_causal=causal)
                want = jax.grad(lambda *a: (attention(*a) ** 2).sum(), argnums=(0, 1, 2))(q, k, v)
            for g, w in zip(got, want):
                g, w = np.asarray(g, np.float64), np.asarray(w, np.float64)
                assert np.abs(g - w).max() <= 1e-5 * np.abs(w).max(), (causal, np.abs(g - w).max())
    """)


def test_compiled_analysis_api():
    _check("""
        import jax,numpy as np
        x=jax.device_put(np.arange(32,dtype=np.float32),jax.devices('metal')[0])
        compiled=jax.jit(lambda x:x*2+1).lower(x).compile()
        assert compiled.cost_analysis() == {}
        # PJRT reports UNIMPLEMENTED; JAX 0.12 translates it to None here.
        assert compiled.memory_analysis() is None
        np.testing.assert_array_equal(np.asarray(compiled(x)),np.arange(32,dtype=np.float32)*2+1)
    """)


def test_executable_and_inputs_can_be_released_after_dispatch():
    _check("""
        import gc
        import jax,numpy as np
        device=jax.devices('metal')[0]
        fn=jax.jit(lambda x:x*2+1)
        x=jax.device_put(np.zeros(1<<20,np.float32),device)
        compiled=fn.lower(x).compile()
        x.delete()
        outputs=[]
        for i in range(32):
            x=jax.device_put(np.full(1<<20,i,np.float32),device)
            outputs.append(compiled(x))
            x.delete()
        del compiled,fn,x
        jax.clear_caches()
        gc.collect()
        for i,y in enumerate(outputs):
            np.testing.assert_array_equal(np.asarray(y),np.full(1<<20,i*2+1,np.float32))
    """)


def test_donation_reuses_storage_after_reads_complete():
    _check("""
        import jax,jax.numpy as jnp,numpy as np
        from jax import lax
        metal=jax.devices('metal')[0]
        host=np.arange(4096,dtype=np.float32)
        cases=[lambda x:x+1, lambda x:x[::-1], lambda x:x,
               lambda x:jnp.zeros_like(x),
               lambda x:lax.fori_loop(0,3,lambda _,a:a+1,x)]
        for case,fn in enumerate(cases):
            for data in (host,np.empty(0,np.float32)):
                x=jax.device_put(data,metal)
                pointer=x.unsafe_buffer_pointer()
                before=jax.jit(lambda x:x*2)(x)
                y=jax.jit(fn,donate_argnums=(0,),keep_unused=True)(x)
                assert x.is_deleted(), (case,data.shape)
                assert y.unsafe_buffer_pointer()==pointer
                np.testing.assert_array_equal(np.asarray(before),data*2)
                expected=jax.jit(fn)(jax.device_put(data,jax.devices('cpu')[0]))
                np.testing.assert_array_equal(np.asarray(y),np.asarray(expected))
        x,y=(jax.device_put(a,metal) for a in (host,host+100))
        px,py=x.unsafe_buffer_pointer(),y.unsafe_buffer_pointer()
        a,b,c=jax.jit(lambda x,y:(y,x,y),donate_argnums=(0,1))(x,y)
        assert x.is_deleted() and y.is_deleted()
        assert a.unsafe_buffer_pointer()==px and b.unsafe_buffer_pointer()==py
        for got,want in ((a,host+100),(b,host),(c,host+100)):
            np.testing.assert_array_equal(np.asarray(got),want)
        # Repeated arguments are marked non-donatable by the PJRT caller.
        x=jax.device_put(host,metal)
        result=jax.jit(lambda x,y:x+y,donate_argnums=(0,))(x,x)
        np.testing.assert_array_equal(np.asarray(result),host*2)
        np.testing.assert_array_equal(np.asarray(x),host)
        # Keep chaining donated outputs without synchronizing each update.
        step=jax.jit(lambda x:x+1,donate_argnums=(0,))
        for _ in range(50):
            x=step(x)
        np.testing.assert_array_equal(np.asarray(x),host+50)
    """)


def test_concurrent_dispatch_retains_shared_inputs():
    _check("""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        import gc,jax,numpy as np
        metal=jax.devices('metal')[0]
        host=np.arange(65536,dtype=np.float32)
        shared=jax.device_put(host,metal)
        run=jax.jit(lambda x,y:(x+y,x,y,x+y)).lower(shared,shared).compile()
        step=jax.jit(lambda x:x+1).lower(shared).compile()
        barrier=Barrier(4)
        def worker(k):
            local=jax.device_put(np.full_like(host,k),metal)
            barrier.wait()
            for _ in range(40):
                result=run(shared,local)
                # Chain consumers of both computed and passthrough outputs.
                result=tuple(step(a) for a in result)
            return result
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(worker,range(4)))
        shared.delete()
        del run,step
        gc.collect()
        # Exercise pool reuse after releasing handles with work still queued.
        noise=[jax.device_put(np.full_like(host,-11),metal) for _ in range(24)]
        for k,result in enumerate(results):
            expected=(host+k+1,host+1,np.full_like(host,k+1),host+k+1)
            for got,want in zip(result,expected,strict=True):
                np.testing.assert_array_equal(np.asarray(got),want)
    """)


def test_gather_indexing_matches_cpu():
    _check("""
        import jax,jax.numpy as jnp,ml_dtypes,numpy as np
        from jax import lax
        metal,cpu=jax.devices('metal')[0],jax.devices('cpu')[0]
        x=np.arange(30,dtype=np.float32).reshape(5,6)
        indices=np.array([-8,-1,0,2,2,4,8],np.int32)
        cases=[
            (lambda x,i:x[i],x,indices),
            (lambda x,i:x[:,i],x,indices),
            (lambda x,i:jnp.take(x,i,axis=0,mode='fill',fill_value=-7),x,indices),
            (lambda x,i:jnp.take(x,i,axis=1,mode='clip'),x,indices),
            (lambda x,i:(x*2+1)[i],x,indices.reshape(1,7)),
            (lambda x,i:x[i,i],x,np.array([0,1,4],np.int32)),
            (lambda x,i:jnp.take_along_axis(x,i,axis=1),x,np.array([[0,5]]*5,np.int32)),
            (lambda x,i:jax.vmap(lambda a,b:a[b])(x,i),x,np.array([[0,5],[1,2],[2,3],[3,4],[4,0]],np.int32)),
            (lambda x,i:lax.gather(x,i,lax.GatherDimensionNumbers(offset_dims=(1,2),
               collapsed_slice_dims=(),start_index_map=(0,1)),(2,3)),
             x,np.array([[-2,1],[4,5],[1,2]],np.int32)),
            (lambda x,i:x[i],x,np.zeros(0,np.int32)),
            (lambda x,i:x[i],x,np.int32(2)),
        ]
        for dtype in (np.int32,np.uint32,np.bool_,np.float16,ml_dtypes.bfloat16):
            cases.append((lambda x,i:x[i],x.astype(dtype),indices))
        for fn,x,i in cases:
            run=jax.jit(fn)
            got=run(jax.device_put(x,metal),jax.device_put(i,metal))
            want=run(jax.device_put(x,cpu),jax.device_put(i,cpu))
            np.testing.assert_array_equal(np.asarray(got),np.asarray(want))
    """)


def test_scatter_add_and_embedding_gradients_match_cpu():
    _check("""
        import jax,jax.numpy as jnp,ml_dtypes,numpy as np
        from jax import lax
        metal,cpu=jax.devices('metal')[0],jax.devices('cpu')[0]
        x=np.arange(30,dtype=np.float32).reshape(5,6)
        i=np.array([-8,-1,0,2,2,4,8],np.int32)
        v=np.arange(42,dtype=np.float32).reshape(7,6)
        cases=[
            (lambda x,i,v:x.at[i].add(v),x,i,v),
            (lambda x,i,v:(x+1).at[i].add(v*2),x,i,v),
            (lambda x,i,v:x.at[i].add(v,mode='clip'),x,i,v),
            (lambda x,i,v:x.at[:,i].add(v.T),x,i,np.arange(35,dtype=np.float32).reshape(7,5)),
            (lambda x,i,v:x.at[i,i].add(v),x,np.array([0,2,2,4],np.int32),np.arange(4,dtype=np.float32)),
            (lambda x,i,v:jax.vmap(lambda a,b,c:a.at[b].add(c))(x,i,v),x,
             np.array([[0,5]]*5,np.int32),np.ones((5,2),np.float32)),
            (lambda x,i,v:x.at[i].add(v),x,np.zeros(0,np.int32),np.zeros((0,6),np.float32)),
            (lambda x,i,v:jax.grad(lambda a:jnp.sum(a[i]**2))(x),x,i,v),
            (lambda x,i,v:lax.fori_loop(0,3,lambda _,a:a.at[i].add(v),x),x,i,v),
        ]
        for dtype in (np.int32,np.uint32,np.float16,ml_dtypes.bfloat16):
            cases.append((lambda x,i,v:x.at[i].add(v),x.astype(dtype),i,v.astype(dtype)))
        # Overlapping windows exercise the update-window coordinate mapping.
        dims=lax.ScatterDimensionNumbers(update_window_dims=(1,),
                inserted_window_dims=(),scatter_dims_to_operand_dims=(0,))
        for dtype in (np.float32,np.float16,ml_dtypes.bfloat16):
            cases.append((lambda x,i,v:lax.scatter_add(x,i,v,dims,mode='drop'),
                          np.zeros(7,dtype),np.array([[0],[3],[2]],np.int32),
                          np.ones((3,4),dtype)))
            # Odd-sized storage and heavy contention, including narrow rounding.
            cases.append((lambda x,i,v:x.at[i].add(v),np.zeros(5,dtype),
                          np.zeros(10001,np.int32),np.ones(10001,dtype)))
        # Thousands of colliding updates catch lost writes, not just index mapping.
        cases.append((lambda x,i,v:x.at[i].add(v),np.zeros(4,np.float32),
                      np.arange(10001,dtype=np.int32)%4,np.ones(10001,np.float32)))
        for fn,x,i,v in cases:
            run=jax.jit(fn)
            got=run(*(jax.device_put(a,metal) for a in (x,i,v)))
            want=run(*(jax.device_put(a,cpu) for a in (x,i,v)))
            np.testing.assert_array_equal(np.asarray(got),np.asarray(want))
        # StableHLO bounds-checks each result index. XLA CPU drops whole windows
        # crossing an edge, so use the specification's scalar reference here.
        for dtype in (np.float32,np.float16,ml_dtypes.bfloat16):
            x=np.zeros(7,dtype)
            i=np.array([[-1],[5],[2]],np.int32)
            v=np.ones((3,4),dtype)
            got=jax.jit(lambda x,i,v:lax.scatter_add(x,i,v,dims,mode='drop'))(
                *(jax.device_put(a,metal) for a in (x,i,v)))
            want=x.copy()
            for row,start in enumerate(i[:,0]):
                for offset in range(v.shape[1]):
                    if 0<=start+offset<len(want):
                        want[start+offset]+=v[row,offset]
            np.testing.assert_array_equal(np.asarray(got),want)
    """)


def test_argmax_argmin_and_variadic_reducers_match_cpu():
    _check("""
        import jax,jax.numpy as jnp,ml_dtypes,numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        metal,cpu=jax.devices('metal')[0],jax.devices('cpu')[0]
        rng=np.random.default_rng(4)
        cases=[]
        for dtype in (np.float32,np.float16,ml_dtypes.bfloat16,np.int32,np.uint32,np.bool_):
            x=rng.integers(0,7,(3,5,37)).astype(dtype)
            for axis in (0,1,2,None):
                cases.append((lambda x,axis=axis:(x.argmax(axis),x.argmin(axis)),x))
        for n in (1,33,10001):
            x=np.zeros((5,n),np.float32)
            x[0,:]=np.nan
            x[1,:]=np.inf
            x[2,:]=-np.inf
            x[3,-1]=np.nan
            x[4,:]=-0.0
            x[4,-1]=0.0
            cases.append((lambda x:(x.argmax(-1),x.argmin(-1)),x))
        x=np.array([[2**30+1,2**30+2,2**30+2],[-2**31,2**31-1,0]],np.int32)
        cases.append((lambda x:(x.argmax(-1),x.argmin(-1)),x))
        cases.append((lambda x:lax.reduce((x,x*2),(np.float32(0),np.float32(0)),
                      lambda a,b:(a[0]+b[0],a[1]+b[1]),(0,2)),
                      np.arange(3*5*7,dtype=np.float32).reshape(3,5,7)))
        cases.append((lambda x:lax.reduce((x,x),(np.float32(3),np.float32(4)),
                      lambda a,b:(a[0]+b[0],a[1]+b[1]),(1,)),np.empty((3,0),np.float32)))
        for fn,x in cases:
            assert_matches_cpu(fn, x, exact=True)
    """)


def test_pallas_kernels_inside_jit_match_cpu():
    """pallas_call and metal_call run as launches of the program, on every path;
    the reference is the same kernel in Pallas's interpreter on CPU."""
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from jax.experimental import pallas as pl
        import palladium
        from pjrt_tolerance import assert_trees_close
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]

        def pair(x_ref,y_ref,o_ref,p_ref):
            o_ref[...] = x_ref[...]*2
            p_ref[...] = x_ref[...]+y_ref[...]
        def scale(x_ref,o_ref):
            o_ref[...] = x_ref[...]*0.5+1
        def first_half(x_ref,o_ref):
            o_ref[...] = x_ref[...]*10
        spec = pl.BlockSpec((4,),lambda i:(i,))
        kwargs = {
            'pair':dict(grid=(2,),in_specs=[spec,spec],out_specs=[spec,spec],
                        out_shape=[jax.ShapeDtypeStruct((8,),jnp.float32)]*2),
            'scale':dict(grid=(2,),in_specs=[spec],out_specs=spec,
                         out_shape=jax.ShapeDtypeStruct((8,),jnp.float32),
                         compiler_params=palladium.CompilerParams(math_mode=palladium.MathMode.SAFE)),
            'half':dict(grid=(1,),in_specs=[pl.BlockSpec((4,),lambda i:(0,))],
                        out_specs=pl.BlockSpec((4,),lambda i:(0,)),
                        out_shape=jax.ShapeDtypeStruct((8,),jnp.float32),
                        input_output_aliases={0:0}),
        }
        kernels = {'pair':pair,'scale':scale,'half':first_half}
        def programs(make):
            k = {name:make(kernels[name],**kw) for name,kw in kwargs.items()}
            return [
                lambda x,y:[3*o for o in k['pair'](jnp.sin(x),y)],
                lambda x,y:lax.fori_loop(0,4,lambda i,z:k['scale'](z)+i,x),
                lambda x,y:k['half'](x+y),
                lambda x,y:jax.vmap(k['pair'],in_axes=(0,None))(jnp.stack([x,y,x*y]),y),
            ]
        x = np.linspace(-1,1,8,dtype=np.float32); y = np.arange(8,dtype=np.float32)
        references = programs(lambda f,**kw:pl.pallas_call(f,**kw,interpret=True))
        for make in (pl.pallas_call,palladium.metal_call):
            for fn,ref in zip(programs(make),references,strict=True):
                got = jax.jit(fn)(jax.device_put(x,metal),jax.device_put(y,metal))
                with jax.default_device(cpu):
                    want = jax.jit(ref)(x,y)
                assert_trees_close(got, want)
    """)


def test_plmtl_threadgroup_kernels_run_inside_jit():
    """A tiled matmul with threadgroup scratch: its launch fixes the threadgroup."""
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from jax.experimental import pallas as pl
        from palladium import plmtl
        from pjrt_tolerance import assert_normwise_close
        def matmul(a_ref,b_ref,o_ref,a_tile,b_tile):
            i,j = pl.program_id(0),pl.program_id(1)
            def step(kb,acc):
                a_tile[...] = a_ref[pl.ds(i*64,64),pl.ds(kb*16,16)]
                b_tile[...] = b_ref[pl.ds(kb*16,16),pl.ds(j*64,64)]
                return acc+jnp.dot(a_tile[...],b_tile[...])
            acc = lax.fori_loop(0,8,step,jnp.zeros((64,64),jnp.float32))
            o_ref[pl.ds(i*64,64),pl.ds(j*64,64)] = acc
        call = plmtl.kernel(matmul,out_type=jax.ShapeDtypeStruct((128,128),jnp.float32),
                            mesh=plmtl.Mesh((2,2)),
                            scratch_types=[plmtl.THREADGROUP((64,16),jnp.float32),
                                           plmtl.THREADGROUP((16,64),jnp.float32)])
        rng = np.random.default_rng(0)
        a,b = (rng.standard_normal((128,128),np.float32) for _ in range(2))
        metal = jax.devices('metal')[0]
        got = jax.jit(lambda a,b:call(a,b)+1)(jax.device_put(a,metal),jax.device_put(b,metal))
        assert_normwise_close(got,(a.astype(np.float64)@b+1).astype(np.float32))
    """)


def test_cnf_training_matches_cpu_over_ten_steps():
    """The CNF update with Palladium's adjoint kernels, and in plain JAX, whose
    backward loop needs fusions split to stay within 31 buffers per kernel."""
    _check("""
        import jax, numpy as np
        import cnf_density as cnf
        from pjrt_tolerance import assert_trees_close
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]
        data = cnf.mixture_data(256)
        def train(device, variant):
            with jax.default_device(device):
                update,_ = cnf.make_training_step(256,variant=variant)
                state = jax.device_put(cnf.initial_state(),device)
                history = []
                for _ in range(10):
                    state,loss = update(state,jax.device_put(data,device))
                    history.append((state,loss))
                return history
        want = train(cpu,'jax')
        for variant in ('reverse','jax'):
            for got,ref in zip(train(metal,variant),want,strict=True):
                assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(got))
                assert_trees_close(got, ref, rtol=1e-4)
    """)


def test_programs_over_the_buffer_limit_match_cpu():
    _check("""
        import jax, numpy as np
        from pjrt_tolerance import assert_matches_cpu
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]
        xs = [np.full(5,k,np.float32)+np.arange(5,dtype=np.float32) for k in range(40)]
        for fn in (lambda *xs:tuple(x+1 for x in xs[:20]),
                   lambda *xs:sum(x*k for k,x in enumerate(xs))):
            assert_matches_cpu(fn, *xs)
    """)


def test_unsupported_ops_report_unimplemented():
    """A CPU compile first, so the metal one hashes its topology for the cache key."""
    result = _run("""
        import jax, jax.numpy as jnp, numpy as np
        assert float(jax.device_put(np.ones(3, np.float32), jax.devices("cpu")[0]).sum()) == 3
        jax.jit(lambda x: jnp.fft.fft(x).real)(jnp.ones(3, device=jax.devices("metal")[0]))
    """)
    assert result.returncode != 0
    assert "UNIMPLEMENTED: metal: element type complex<f32> is not supported" in result.stderr


def test_registers_only_when_jax_platforms_names_metal():
    out = _check("import jax; print(sorted({d.platform for d in jax.devices()}))", platforms=None)
    assert "metal" not in out


def test_registers_without_taking_the_default_when_listed_second():
    out = _check(
        "import jax; print(jax.default_backend(), len(jax.devices('metal')))", platforms="cpu,metal"
    )
    assert out.split() == ["cpu", "1"]


def test_convolutions_and_their_gradients_match_cpu():
    # Strides, padding (negative too), both dilations, other layouts and ranks,
    # and transposed convolutions, which take the phase decomposition when the
    # window and output divide by the input dilation.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        r = lambda *s: rng.standard_normal(s).astype(np.float32)
        nhwc = ('NHWC', 'HWIO', 'NHWC')
        cases = [
            ((2, 16, 16, 3), (4, 4, 3, 8), dict(window_strides=(4, 4), padding='VALID')),
            ((1, 20, 20, 3), (7, 7, 3, 6), dict(window_strides=(4, 4), padding=((3, 3), (3, 3)))),
            ((2, 12, 12, 4), (3, 3, 4, 4), dict(window_strides=(1, 1), padding='SAME', rhs_dilation=(2, 2))),
            ((2, 6, 6, 4), (3, 3, 4, 2), dict(window_strides=(1, 1), padding=((2, 2), (1, 2)), lhs_dilation=(2, 2))),
            ((1, 10, 10, 2), (3, 3, 2, 3), dict(window_strides=(1, 1), padding=((-1, 0), (0, -2)))),
            ((2, 5, 6, 3), (4, 3, 3, 2), dict(window_strides=(1, 1), padding=((1, 2), (0, 0)), lhs_dilation=(2, 1))),
        ]
        for xs, ws, kw in cases:
            f = lambda x, w, kw=kw: lax.conv_general_dilated(x, w, dimension_numbers=nhwc, **kw)
            grads = jax.grad(lambda x, w, f=f: jnp.sum(jnp.sin(f(x, w))), (0, 1))
            x, w = r(*xs), r(*ws)
            assert_matches_cpu(f, x, w, rtol=2e-5, err_msg=str(kw))
            assert_matches_cpu(grads, x, w, rtol=2e-5, err_msg=str(kw))
        for k, s, pad in [(2, 2, 'SAME'), (4, 2, 'SAME'), (4, 2, 'VALID'), (6, 3, 'SAME'), (3, 3, ((1, 1), (2, 0)))]:
            t = lambda x, w, k=k, s=s, pad=pad: lax.conv_transpose(x, w, (s, s), pad, dimension_numbers=nhwc)
            grads = jax.grad(lambda x, w, t=t: jnp.sum(jnp.tanh(t(x, w))), (0, 1))
            x, w = r(2, 7, 6, 3), r(k, k, 3, 5)
            assert_matches_cpu(t, x, w, rtol=2e-5, err_msg=f'transpose {k} {s} {pad}')
            assert_matches_cpu(grads, x, w, rtol=2e-5, err_msg=f'transpose {k} {s} {pad}')
        assert_matches_cpu(lambda x, w: lax.conv(x, w, (2, 1), 'SAME'), r(2, 3, 10, 12), r(4, 3, 3, 3), rtol=2e-5)
        assert_matches_cpu(lambda x, w: lax.conv_general_dilated(x, w, (2,), 'SAME', dimension_numbers=('NWC', 'WIO', 'NWC')), r(2, 17, 3), r(5, 3, 4), rtol=2e-5)
        assert_matches_cpu(lambda x, w: lax.conv_general_dilated(x, w, (1, 2, 1), 'SAME', dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')), r(1, 5, 6, 4, 2), r(3, 3, 2, 2, 3), rtol=2e-5)
        bf16 = lambda a: a.astype(jnp.bfloat16)
        assert_matches_cpu(lambda x, w: lax.conv_general_dilated(x, w, (1, 1), 'SAME', dimension_numbers=nhwc), bf16(r(2, 8, 8, 16)), bf16(r(3, 3, 16, 16)), rtol=1e-2)
    """)


def test_pooling_and_its_gradient_match_cpu():
    # Max pooling's gradient is select_and_scatter: the first maximum of each
    # window takes the gradient, and padding is never selected.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        x = np.random.default_rng(1).standard_normal((2, 9, 10, 3)).astype(np.float32)
        pools = [
            lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID'),
            lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME'),
            lambda x: lax.reduce_window(x, 0.0, lax.add, (1, 3, 3, 1), (1, 1, 1, 1), 'SAME') / 9,
        ]
        for pool in pools:
            assert_matches_cpu(pool, x, exact=True)
            assert_matches_cpu(jax.grad(lambda x, pool=pool: jnp.sum(jnp.cos(pool(x)))), x, rtol=1e-6)
        # Windows entirely on dilation holes give the initial value, inf.
        dilated = lambda x: lax.reduce_window(x, jnp.inf, lax.min, (1, 2, 2, 1), (1, 1, 1, 1), 'VALID',
                                              window_dilation=(1, 2, 2, 1), base_dilation=(1, 2, 1, 1))
        assert np.isinf(assert_matches_cpu(dilated, x, exact=True)).any()
        ties = np.ones((1, 4, 4, 1), np.float32)
        weights = jnp.arange(4.0).reshape(1, 2, 2, 1)
        got = assert_matches_cpu(jax.grad(lambda x: jnp.sum(pools[0](x) * weights)), ties, exact=True)
        assert np.count_nonzero(got) == 3  # one per window, the zero weight aside
    """)


def test_sam_training_step_matches_cpu_over_three_steps():
    """Each of three steps, taken on both backends from the CPU's state.

    The states are not left to run apart: max-pooling picks a winner, so two
    trajectories a few 1e-6 apart can disagree on one and then differ by
    1e-3 in a gradient, with either backend stepping correctly. From one
    state the backends agree to rounding at every step. Drift over long runs
    is what examples/pjrt_long_run.py measures, against the CPU's own spread.
    """
    _check("""
        import jax, numpy as np
        from training_steps import workload
        from pjrt_tolerance import assert_trees_close
        cpu, metal = jax.devices('cpu')[0], jax.devices('metal')[0]
        with jax.default_device(cpu):
            fn, params, opt, images, masks = workload('sam')
            images = jax.random.uniform(jax.random.key(1), images.shape)
            masks = (jax.random.uniform(jax.random.key(2), masks.shape) > 0.5).astype(np.float32)
        run = jax.jit(fn)
        state = (params, opt, images, masks)
        for _ in range(3):
            (p, o), loss = run(*state)
            got = run(*jax.device_put(state, metal))
            assert_trees_close(got, ((p, o), loss), rtol=1e-4)
            state = (p, o, images, masks)
    """)


def test_long_contractions_with_few_outputs_split_k():
    # One 64x64 tile accumulating K = 8192 in one sequence would err several
    # times more than CPU; in 16 chunks it matches CPU's error.
    _check("""
        import jax, numpy as np
        rng = np.random.default_rng(0)
        a = rng.uniform(-1, 1, (64, 8192)).astype(np.float32)
        b = rng.uniform(-1, 1, (8192, 64)).astype(np.float32)
        exact = a.astype(np.float64) @ b.astype(np.float64)
        f = jax.jit(lambda a, b: a @ b)
        err = {d: np.abs(np.asarray(f(*jax.device_put((a, b), jax.devices(d)[0]))) - exact).max()
               for d in ('metal', 'cpu')}
        assert err['metal'] <= 1.5 * err['cpu'], err
        batched = jax.jit(lambda a, b: jax.numpy.einsum('bmk,bkn->bmn', a, b))
        a3, b3 = a.reshape(2, 32, 8192), np.stack([b[:, :32], b[:, 32:]])
        got = np.asarray(batched(*jax.device_put((a3, b3), jax.devices('metal')[0])))
        np.testing.assert_allclose(got, np.einsum('bmk,bkn->bmn', a3, b3), rtol=1e-4, atol=1e-3)
    """)


def test_nans_and_infinities_match_cpu():
    # Kernels must not assume finite operands: isnan(x), lowered as x != x, is
    # true for NaN, and x - x and x * 0 are NaN for NaN and infinite x.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from pjrt_tolerance import assert_matches_cpu
        x = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.5, -2.5, 3.0], np.float32)
        cases = {
            "isnan": jnp.isnan,
            "self comparisons": lambda x: (x != x, x == x),
            "nan_to_num": jnp.nan_to_num,
            "where": lambda x: jnp.where(jnp.isnan(x), 0.0, x),
            "x - x": lambda x: x - x,
            "x * 0": lambda x: x * 0.0,
            "nanmax": jnp.nanmax,
            "isfinite": lambda x: (jnp.isfinite(x), jnp.isinf(x)),
        }
        for name, fn in cases.items():
            for dtype in (np.float32, np.float16):
                assert_matches_cpu(fn, x.astype(dtype), exact=True, err_msg=name)
    """)


def test_deep_fusions_match_cpu():
    # 2,000 chained ops fuse into one kernel, deeper than Python's default
    # recursion limit.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from pjrt_tolerance import assert_matches_cpu
        def fn(x):
            for k in range(1000):
                x = jnp.sin(x) * 0.5 + 0.1 * (k % 3)  # a contraction: errors do not grow
            return x
        assert_matches_cpu(fn, np.linspace(-1, 1, 64, dtype=np.float32))
    """)


def test_loop_carries_computed_by_their_own_launches_match_cpu():
    # Products, reductions and scatters as carries write the loop's output
    # buffers directly: permuted, repeated and read again in the same iteration.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        w = rng.standard_normal((32, 32)).astype(np.float32) / 6
        x = rng.standard_normal((32, 32)).astype(np.float32)
        i = np.array([1, 3, 3, 30], np.int32)
        def repeated(c, w):
            y = c[1] @ w
            return y, y, c[0].sum()
        cases = {
            "product": (lambda w, x: lax.fori_loop(0, 10, lambda i, h: h @ w, x), w, x),
            "swapped": (lambda w, x: lax.fori_loop(
                0, 7, lambda i, c: (c[1] @ w, c[0] @ w), (x, x * 2)), w, x),
            "repeated": (lambda w, x: lax.fori_loop(
                0, 5, lambda i, c: repeated(c, w), (x, x * 2, 0.0)), w, x),
            "read again": (lambda w, x: lax.fori_loop(
                0, 6, lambda i, c: (c[0] @ w, c[1] + (c[0] @ w).sum(0) + c[0][0]), (x, x[0])),
                w, x),
            "reduction": (lambda x: lax.fori_loop(
                0, 6, lambda i, c: (jnp.tanh(c[1] + c[0] / 1e3).sum(), c[1] * 0.9 + c[0] / 1e4),
                (np.float32(1), x)), np.tile(x, (4, 2))),
            "argmax": (lambda x: lax.fori_loop(
                0, 6, lambda i, c: (jnp.argmax(c[1] + c[0], 1), c[1] * 0.9 + c[0][:, None]),
                (jnp.zeros(32, jnp.int32), x)), x),
            "scatter": (lambda x, i: lax.fori_loop(
                0, 6, lambda k, c: c.at[i].add(c[0] * 0.1 + 1), x), x[0], i),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, rtol=1e-4, err_msg=name)
    """)


def test_narrow_integer_operations_match_cpu():
    # int8, uint8, int16 and uint16 compute in 32 bits and wrap on the way
    # back; extrema cover overflow, division, and shift counts past the width.
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        for dt in (np.int8, np.uint8, np.int16, np.uint16):
            info = np.iinfo(dt)
            edge = np.array([info.min, info.min + 1, -1 if info.min else 1, 0, 1, 2,
                             info.max - 1, info.max], dt)
            a = np.concatenate([edge, rng.integers(info.min, info.max, 56).astype(dt)])
            b = np.concatenate([edge[::-1], rng.integers(info.min, info.max, 56).astype(dt)])
            x = a.reshape(8, 8)
            other = np.dtype(dt.__name__[1:] if info.min == 0 else "u" + dt.__name__)
            cases = {
                "arithmetic": (lambda a, b: (a + b, a - b, a * b, -a, jnp.abs(a), jnp.sign(a),
                                             jnp.maximum(a, b), jnp.minimum(a, b)), a, b),
                "division": (lambda a, b: (lax.div(a, b), lax.rem(a, b), a // b, a % b), a, b),
                "bits": (lambda a, b: (a & b, a | b, a ^ b, ~a, lax.population_count(a),
                                       lax.clz(a)), a, b),
                "shifts": (lambda a, s: (lax.shift_left(a, s), lax.shift_right_logical(a, s),
                                         lax.shift_right_arithmetic(a, s)),
                           a, (np.arange(64) % 20).astype(dt)),
                "shifts by any count": (lambda a, s: (lax.shift_left(a, s),
                                                      lax.shift_right_logical(a, s),
                                                      lax.shift_right_arithmetic(a, s)), a, b),
                "comparisons": (lambda a, b: (a < b, a <= b, a == b, a != b,
                                              jnp.where(a > b, a, b)), a, b),
                "conversions": (lambda a: (a.astype(np.float32), a.astype(np.int32),
                                           a.astype(np.uint32), a.astype(np.bool_),
                                           a.astype(np.int8), a.astype(np.uint16),
                                           lax.bitcast_convert_type(a, other)), a),
                "conversions to": (lambda f, i, m: (f.astype(dt), i.astype(dt), m.astype(dt)),
                                   rng.uniform(max(info.min, -100), 100, 64).astype(np.float32),
                                   rng.integers(-2**31, 2**31, 64).astype(np.int32),
                                   rng.random(64) > 0.5),
                "index maps": (lambda x: (x.T, x[::-1, 1::2], x.reshape(4, 16),
                                          jnp.concatenate([x, x + dt(1)], 1),
                                          jnp.pad(x, 2, constant_values=7),
                                          jnp.arange(8, dtype=dt) + x,
                                          x[jnp.array([3, 0, 7])]), x),
                "constants": (lambda x: (x + jnp.asarray(edge), jnp.asarray(edge)), edge),
                "reductions": (lambda x: (x.sum(), x.max(), x.min(), x.reshape(-1, 8).sum(1),
                                          x.reshape(64, -1).min(0),
                                          jnp.argmax(x.reshape(64, -1), 1)), np.tile(a, 200)),
                "loop": (lambda x: lax.fori_loop(0, 5, lambda i, y: y * dt(3) + x, x), a),
                "as indices": (lambda t, i: jnp.take(t, i, mode="clip"),
                               np.arange(300, dtype=np.float32), a),
                "image": (lambda x: (x.astype(np.float32) / 255 - 0.5) * 2, a),
            }
            for name, (fn, *args) in cases.items():
                assert_matches_cpu(fn, *args, exact=name != "image",
                                   err_msg=f"{dt.__name__} {name}")
    """)


def test_dynamic_loops_match_cpu():
    """Loops whose condition the host reads: data-dependent trip counts, zero
    trips, permuted carries, a GEMM in the body, nesting in counted loops and
    conditionals, and batched loops."""
    out = _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 5), np.float32)
        a = rng.standard_normal((4, 4), np.float32)
        a = a @ a.T + 4 * np.eye(4, dtype=np.float32)

        def newton(x):
            def cond(s): return jnp.any(jnp.abs(s[0] * s[0] - jnp.abs(x)) > 1e-5) & (s[1] < 50)
            def body(s): return (0.5 * (s[0] + jnp.abs(x) / s[0]), s[1] + 1)
            return lax.while_loop(cond, body, (jnp.ones_like(x), 0))

        def swap(x):
            def body(s): return (s[1], s[0] + 1, s[2] + 1)
            return lax.while_loop(lambda s: s[2] < 5, body, (x, x * 2, 0.0))

        cases = {
            "newton": (newton, x),
            "traced bound": (lambda x, n: lax.fori_loop(0, n, lambda j, y: y * 1.01 + j, x), x, np.int32(7)),
            "zero trips": (lambda x: lax.while_loop(lambda s: s[1] < 0, lambda s: (s[0] + 1, s[1] + 1), (x, 0)), x),
            "odd and even trips": (lambda x, n: lax.while_loop(lambda s: s[1] < n, lambda s: (s[0] * 2, s[1] + 1), (x, 0)), x, np.int32(4)),
            "permuted carries": (swap, x),
            "gemm": (lambda a: lax.while_loop(lambda s: jnp.abs(s).max() < 1e6, lambda s: s @ a, a), a),
            "in a counted loop": (lambda x: lax.fori_loop(0, 3, lambda j, y: newton(y)[0] + 1.0, x), x),
            "in a conditional": (lambda x: lax.cond(x[0, 0] > 100, lambda: x, lambda: newton(x)[0]), x),
            "batched": (lambda x: jax.vmap(lambda r: lax.while_loop(
                lambda s: s[1] < jnp.int32(3) + (r[0] > 0), lambda s: (s[0] * 2, s[1] + 1), (r, 0))[0])(x), x),
            "scalar": (lambda x: lax.while_loop(lambda s: s < 100.0, lambda s: s * 2 + 1, x[0, 0] ** 2 + 1), x),
            "many trips": (lambda x, n: lax.while_loop(lambda s: s[1] < n, lambda s: (s[0] * 1.001 + 1e-3, s[1] + 1), (x, 0)), x, np.int32(1003)),
        }
        # Every trip count around the groups of iterations between two reads.
        counted = jax.jit(lambda x, n: lax.while_loop(lambda s: s[1] < n, lambda s: (s[0] + 1, s[1] + 1), (x, 0)))
        for n in range(20):
            total, trips = counted(jax.device_put(x, jax.devices("metal")[0]), np.int32(n))
            np.testing.assert_allclose(total, x + n, rtol=1e-6)
            assert int(trips) == n
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, err_msg=name)
        assert_matches_cpu(lambda x, n: lax.while_loop(lambda s: s[1] < n, lambda s: (s[0] * 2, s[1] + 1), (x, 0)), x, np.int32(5))
        print(len(cases))
    """)
    assert out.strip() == "11"


def test_host_callbacks_run_between_launches():
    out = _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from jax.experimental import checkify, io_callback
        from pjrt_tolerance import assert_matches_cpu
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        assert_matches_cpu(lambda x: jax.pure_callback(np.sin, spec, x * 2) + 1, x)
        assert_matches_cpu(lambda x: lax.fori_loop(0, 3, lambda j, y: jax.pure_callback(np.cos, spec, y), x), x)
        # A loop that calls the host reads its condition once per iteration.
        assert_matches_cpu(lambda x: lax.while_loop(
            lambda s: s[0].sum() < 2.5, lambda s: (jax.pure_callback(np.cos, spec, s[0]), s[1] + 1), (x, 0)), x)

        metal = jax.devices("metal")[0]
        x = jax.device_put(x, metal)
        seen = []
        def record(a):
            seen.append(float(a))
            return np.float32(len(seen))
        fn = jax.jit(lambda x: io_callback(record, jax.ShapeDtypeStruct((), jnp.float32), x.sum(), ordered=True)
                     + io_callback(record, jax.ShapeDtypeStruct((), jnp.float32), x.max(), ordered=True))
        assert float(fn(x)) == 3.0 and seen == [15.0, 5.0], seen

        @jax.jit
        def printing(x):
            jax.debug.print("sum {}", x.sum())
            return lax.fori_loop(0, 3, lambda j, y: (jax.debug.print("step {} {}", j, y[0, 0]), y + 1)[1], x)
        np.testing.assert_array_equal(printing(x), np.asarray(x) + 3)

        checked = jax.jit(checkify.checkify(lambda y: (checkify.check(y.sum() > 0, "negative"), y * 2)[1]))
        error, _ = checked(x)
        error.throw()
        error, _ = checked(-x)
        assert "negative" in str(error.get())

        def fails(a):
            raise ValueError("no luck")
        # A callback in a conditional branch runs only if the branch is taken,
        # as does a loop the host drives.
        guarded = jax.jit(lambda p, x: lax.cond(
            p, lambda: jax.pure_callback(fails, spec, x),
            lambda: lax.while_loop(lambda y: y.sum() < 100, lambda y: y * 2 + 1, x)))
        assert float(guarded(False, x).sum()) >= 100
        switched = jax.jit(lambda k, x: lax.switch(k, [
            lambda: x, lambda: jax.pure_callback(fails, spec, x), lambda: jax.pure_callback(np.sin, spec, x)]))
        np.testing.assert_array_equal(switched(0, x), x)
        np.testing.assert_allclose(switched(2, x), np.sin(np.asarray(x)), rtol=1e-6)
        try:
            np.asarray(jax.jit(lambda x: jax.pure_callback(fails, spec, x))(x))
        except Exception as e:
            assert "no luck" in str(e), e
        else:
            raise AssertionError("the callback's error was lost")
        # A wrong result shape is an error, not memory written out of bounds.
        try:
            np.asarray(jax.jit(lambda x: jax.pure_callback(lambda a: a[0], spec, x))(x))
        except Exception as e:
            assert "returned float32[3]" in str(e), e
        else:
            raise AssertionError("a result of the wrong shape was accepted")
    """)
    assert out.splitlines() == ["sum 15.0", "step 0 0.0", "step 1 1.0", "step 2 2.0"]


def test_linear_algebra_matches_cpu():
    """Cholesky, triangular solves, LU, QR, eigh and SVD run on LAPACK between
    launches; solve, inv, determinants and their gradients build on them."""
    out = _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 5), np.float32)
        a = rng.standard_normal((4, 4), np.float32)
        a = a @ a.T + 4 * np.eye(4, dtype=np.float32)
        batch = np.stack([a, 2 * a, a + np.eye(4, dtype=np.float32)])
        solve = lax.linalg.triangular_solve
        cases = {
            "cholesky": (jnp.linalg.cholesky, a),
            "cholesky batch": (jnp.linalg.cholesky, batch),
            "cholesky indefinite": (jnp.linalg.cholesky, np.stack([a, -a])),
            "left lower": (lambda a, b: solve(jnp.tril(a), b[:4], left_side=True, lower=True), a, x),
            "left upper transposed": (lambda a, b: solve(jnp.triu(a), b[:4], left_side=True, lower=False, transpose_a=True), a, x),
            "right upper": (lambda a, b: solve(jnp.triu(a), b[:, :4], left_side=False, lower=False), a, x),
            "right lower transposed": (lambda a, b: solve(jnp.tril(a), b[:, :4], left_side=False, lower=True, transpose_a=True), a, x),
            "unit diagonal": (lambda a, b: solve(jnp.tril(a), b[:4], left_side=True, lower=True, unit_diagonal=True), a, x),
            "solve": (lambda a, b: jnp.linalg.solve(a, b[:4]), a, x),
            "solve batch": (lambda a, b: jax.vmap(lambda m: jnp.linalg.solve(m, b[:4, 0]))(a), batch, x),
            "inv": (jnp.linalg.inv, a),
            "slogdet": (jnp.linalg.slogdet, a),
            "lu rectangular": (jax.scipy.linalg.lu, x),
            "qr": (jnp.linalg.qr, x),
            "qr complete": (lambda x: jnp.linalg.qr(x, mode="complete"), x),
            "eigh": (jnp.linalg.eigh, a),
            "svd": (lambda x: jnp.linalg.svd(x, full_matrices=False), x),
            "singular values": (lambda x: jnp.linalg.svd(x, compute_uv=False), x),
            "pinv": (jnp.linalg.pinv, x),
            "normal log density": (lambda a, x: jax.scipy.stats.multivariate_normal.logpdf(x[:, :4], jnp.zeros(4), a), a, x),
            "householder qr": (lambda x: lax.linalg.householder_product(*jax._src.lax.linalg.geqrf(x)), x),
            "hessenberg": (lambda g: jax.scipy.linalg.hessenberg(g, calc_q=True), x[:5]),
            "tridiagonal": (lambda a: lax.linalg.tridiagonal(a, lower=True), a),
            "schur": (jax.scipy.linalg.schur, x[:5]),
            "schur batch": (lax.linalg.schur, np.stack([x[:5], x[1:], x[:5].T])),
            "cholesky gradient": (jax.grad(lambda m: jnp.linalg.cholesky(m).sum()), a),
            "eigh gradient": (jax.grad(lambda m: jnp.linalg.eigh(m)[0].sum()), a),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, rtol=2e-4, err_msg=name)
        print(len(cases))
    """)
    assert out.strip() == "27"


def test_sorts_match_cpu():
    out = _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 5), np.float32)
        v = rng.standard_normal(3000, np.float32)
        i = rng.integers(-50, 50, (6, 5), np.int32)
        special = np.array([np.nan, 1, -0.0, 0.0, -np.inf, np.inf, -1, np.nan, 2], np.float32)
        cases = {
            "long": (jnp.sort, v),
            "each axis": (lambda x: (jnp.sort(x, axis=0), jnp.sort(x, axis=1)), x),
            "descending": (lambda v: jnp.sort(v, descending=True), v),
            "nonfinite and zeros": (jnp.sort, special),
            "argsort is stable": (lambda i: jnp.argsort(jnp.tile(i.ravel() % 4, 9), stable=True), i),
            "integers": (lambda i: jnp.sort(i, axis=1), i),
            "by key": (lambda i, x: lax.sort((i % 3, x), dimension=1, num_keys=1), i, x),
            "two keys": (lambda i, x: lax.sort((i % 3, i % 2, x), dimension=0, num_keys=2), i, x),
            "one element": (lambda x: jnp.sort(x[:, :1], axis=1), x),
            "half": (jnp.sort, v.astype(np.float16)),
            "top k": (lambda v: lax.top_k(v, 5), v),
            "median": (lambda x: jnp.median(x, axis=1), x),
            "searchsorted": (lambda v, x: jnp.searchsorted(jnp.sort(v), x.ravel()), v, x),
            "unique": (lambda i: jnp.unique(i, size=20, fill_value=0), i),
            "permutation": (lambda: jax.random.permutation(jax.random.key(0), 50),),
            "gradient": (jax.grad(lambda y: jnp.sort(y)[:10].sum()), v),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, exact=name != "median", err_msg=name)
        print(len(cases))
    """)
    assert out.strip() == "16"


def test_cumulative_and_window_reductions_match_cpu():
    """Cumulative reductions in blocks along a long axis, both directions and
    with reducer bodies; long windows; and the reducers beyond sum and extrema."""
    out = _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 5), np.float32)
        big = rng.standard_normal((3, 1000), np.float32)
        i = rng.integers(-50, 50, (6, 5), np.int32)
        cases = {
            "cumsum": (lambda b: jnp.cumsum(b, axis=1), big),
            "cumsum leading axis": (lambda b: jnp.cumsum(b.T, axis=0), big),
            "cumsum reversed": (lambda b: lax.cumsum(b, axis=1, reverse=True), big),
            "cumsum of a fused value": (lambda b: jnp.cumsum(jnp.exp(b * 0.1) + 1, axis=1), big),
            "cumsum int": (lambda i: jnp.cumsum(jnp.tile(i.ravel(), 20)), i),
            "cumsum half": (lambda b: jnp.cumsum(b[0, :200].astype(jnp.float16)), big),
            "cumprod": (lambda b: jnp.cumprod(1 + 0.01 * b, axis=1), big),
            "cummax": (lambda b: lax.cummax(b, axis=1), big),
            "cummin reversed": (lambda b: lax.cummin(b, axis=1, reverse=True), big),
            "cumlogsumexp": (lambda b: lax.cumlogsumexp(b, axis=1), big),
            "cumsum gradient": (jax.grad(lambda y: (jnp.cumsum(y) ** 2).sum()), big[0]),
            "long window": (lambda b: lax.reduce_window(b, 0.0, lax.add, (1, 100), (1, 10), "VALID"), big),
            "long padded window": (lambda b: lax.reduce_window(b, -np.inf, lax.max, (1, 70), (1, 7), "SAME"), big),
            "product": (lambda x: (jnp.prod(x, axis=1), jnp.prod(x)), x),
            "long product": (lambda b: jnp.prod(1 + 0.001 * b), big),
            "product int": (lambda i: jnp.prod(i % 3 + 1, axis=0), i),
            "product half": (lambda x: jnp.prod(x.astype(jnp.float16), axis=0), x),
            "reducer body": (lambda b: lax.reduce(b, -np.inf, jnp.logaddexp, (1,)), big),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, rtol=1e-4 if "half" not in name else None, err_msg=name)
        print(len(cases))
    """)
    assert out.strip() == "18"


def test_scatters_integer_dots_and_bitcasts_match_cpu():
    out = _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 5), np.float32)
        v = rng.standard_normal(300, np.float32)
        i = rng.integers(-50, 50, (6, 5), np.int32)
        rows = jnp.array([1, 3, 1])
        cases = {
            "set": (lambda x: x.at[jnp.array([1, 3]), 2].set(7.0), x),
            "set slice": (lambda x: x.at[1:3].set(0.0), x),
            "set half": (lambda x: x.astype(jnp.float16).at[jnp.array([1, 3])].set(2.0), x),
            "set bool": (lambda x: (x > 0).at[jnp.array([0, 1])].set(True), x),
            "multiply": (lambda x: x.at[rows].multiply(2.0), x),
            "multiply int8": (lambda i: i.astype(jnp.int8).at[rows].multiply(2), i),
            "max": (lambda x: x.at[rows, 0].max(0.5), x),
            "min": (lambda x: x.at[rows, 0].min(0.5), x),
            "max half": (lambda x: x.astype(jnp.float16).at[rows, 0].max(0.5), x),
            "add int16": (lambda i: i.astype(jnp.int16).at[rows].add(3), i),
            "segment max": (lambda v: jax.ops.segment_max(v, jnp.arange(300) % 7, 7), v),
            "integer matmul": (lambda i: i @ i.T, i),
            "integer matvec": (lambda i: jnp.dot(i, i[0]), i),
            "boolean matmul": (lambda x: (x > 0) @ (x.T > 0), x),
            "narrowing bitcast": (lambda v: lax.bitcast_convert_type(v, jnp.int16), v),
            "widening bitcast": (lambda v: lax.bitcast_convert_type(
                lax.bitcast_convert_type(v, jnp.uint8), jnp.float32), v),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, err_msg=name)
        print(len(cases))
    """)
    assert out.strip() == "16"


def test_grouped_convolutions_and_their_gradients_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 8, 7, 6), np.float32)

        def conv(x, w, groups, strides=(1, 1), padding="SAME"):
            return lax.conv_general_dilated(x, w, strides, padding, feature_group_count=groups)

        # Depthwise (one input feature per group), a channel multiplier, and
        # groups large enough for the GEMM, each with both gradients.
        for groups, per_group, multiplier in [(8, 1, 1), (8, 1, 3), (2, 4, 2), (4, 2, 1)]:
            w = rng.standard_normal((groups * multiplier, per_group, 3, 3), np.float32)
            for strides, padding in [((1, 1), "SAME"), ((2, 1), "VALID")]:
                name = f"{groups} groups of {per_group}, x{multiplier}, {padding}"
                assert_matches_cpu(lambda x, w: conv(x, w, groups, strides, padding), x, w, err_msg=name)
                loss = lambda x, w: (conv(x, w, groups, strides, padding) ** 2).sum()
                assert_matches_cpu(jax.grad(loss, (0, 1)), x, w, rtol=1e-4, err_msg=name)
        w = rng.standard_normal((5, 1, 6), np.float32)
        nwc = lambda x, w: lax.conv_general_dilated(
            x, w, (2,), "SAME", dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=6)
        assert_matches_cpu(nwc, x[0], w)
    """)


def test_newer_stablehlo_ops_and_the_bit_generator():
    """The plugin reports jaxlib's StableHLO version, so jax may emit `fma`,
    which the special functions and samplers use."""
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        v = rng.standard_normal(300, np.float32)
        key = np.asarray(jax.random.key_data(jax.random.key(0)))
        wrap = jax.random.wrap_key_data
        special = jax.scipy.special
        cases = {
            "erfinv": (lambda v: special.erfinv(jnp.tanh(v)), v),
            "gammaln": (lambda v: special.gammaln(jnp.abs(v) + 0.5), v),
            "digamma": (lambda v: special.digamma(jnp.abs(v) + 0.5), v),
            "gammainc": (lambda v: special.gammainc(2.0, jnp.abs(v)), v),
            "betainc": (lambda v: special.betainc(2.0, 3.0, jax.nn.sigmoid(v)), v),
            "i0e": (special.i0e, v),
            "expm": (lambda v: jax.scipy.linalg.expm(v[:16].reshape(4, 4) * 0.1), v),
            "histogram": (lambda v: jnp.histogram(v, bins=10, range=(-3, 3))[0], v),
            "gamma": (lambda k: jax.random.gamma(wrap(k), 2.0, (50,)), key),
            "poisson": (lambda k: jax.random.poisson(wrap(k), 3.0, (50,)), key),
            "beta": (lambda k: jax.random.beta(wrap(k), 2.0, 3.0, (20,)), key),
            "truncated normal": (lambda k: jax.random.truncated_normal(wrap(k), -1.0, 1.0, (20,)), key),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, rtol=2e-4, err_msg=name)

        # StableHLO leaves the generator to the backend: check its statistics.
        metal = jax.devices("metal")[0]
        with jax.default_device(metal):
            sample = jax.jit(lambda seed: jax.random.split(jax.random.key(seed, impl="rbg")))
            keys = sample(0)
            a = np.asarray(jax.jit(lambda k: jax.random.normal(k, (1 << 16,)))(keys[0]))
            b = np.asarray(jax.jit(lambda k: jax.random.normal(k, (1 << 16,)))(keys[1]))
        assert abs(a.mean()) < 0.02 and abs(a.std() - 1) < 0.02, (a.mean(), a.std())
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.02 and abs(np.corrcoef(a[1:], a[:-1])[0, 1]) < 0.02
        assert len(np.unique(a)) > 60000
    """)


def test_diffrax_solves_match_cpu():
    """diffrax drives its steps with a `while_loop` and reports errors through
    callbacks in conditional branches."""
    _check("""
        import diffrax, jax, jax.numpy as jnp, numpy as np
        from pjrt_tolerance import assert_matches_cpu

        def field(t, y, a):
            x, z = y[..., 0], y[..., 1]
            return jnp.stack([a[0] * x - a[1] * x * z, a[2] * x * z - a[3] * z], -1)

        def solve(y0, controller):
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(field), diffrax.Tsit5(), 0.0, 0.5, 0.01, y0,
                args=jnp.array([1.5, 1.0, 1.0, 3.0]), stepsize_controller=controller,
                saveat=diffrax.SaveAt(t1=True), max_steps=200)
            return solution.ys

        y0 = np.ones((16, 2), np.float32)
        assert_matches_cpu(lambda y: solve(y, diffrax.ConstantStepSize()), y0, rtol=1e-4)
        assert_matches_cpu(lambda y: solve(y, diffrax.PIDController(rtol=1e-5, atol=1e-6)), y0, rtol=1e-4)
    """)


def test_llm_lora_step_matches_cpu():
    """The Qwen2.5 port of examples/lora (grouped-query attention with RoPE,
    SwiGLU, tied head) in a tiny configuration with random weights: three
    LoRA training steps agree with the CPU backend, in float32 and with
    bfloat16 weights."""
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from pjrt_tolerance import assert_trees_close
        from lora.model import QwenConfig, init_lora, init_params
        from lora.train import adam_init, make_train_step
        config = QwenConfig(
            vocab_size=97, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=64, lora_num_layers=2, lora_rank=2)
        rng = np.random.default_rng(0)
        tokens = rng.integers(1, config.vocab_size, size=(2, 17)).astype(np.int32)
        tokens[1, 9:] = 0
        lengths = np.array([[0, 17], [0, 9]], np.int32)
        for dtype, rtol in ((jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)):
            lora = init_lora(config)
            results = {}
            for name in ("metal", "cpu"):
                with jax.default_device(jax.devices(name)[0]):
                    params = init_params(config, seed=0, dtype=dtype)
                    step = jax.jit(make_train_step(config, learning_rate=1e-3))
                    state, losses = (lora, adam_init(lora)), []
                    for _ in range(3):
                        *state, loss = step(params, *state, tokens, lengths)
                        losses.append(loss)
                    results[name] = losses
            assert_trees_close(results["metal"], results["cpu"], rtol=rtol)
            assert float(results["cpu"][2]) < float(results["cpu"][0])
    """)


def test_sam2_lora_steps_match_cpu():
    """The SAM2 port of examples/sam2 (Hiera trunk, neck, prompt encoder and
    mask decoder) with random weights: the logits and three LoRA training
    steps agree with the CPU backend, as written and with its global
    attention on the flash-attention kernels, forward and backward."""
    _check("""
        import dataclasses
        import jax, numpy as np
        from pjrt_tolerance import assert_trees_close
        from sam2.model import Sam2Config, block_specs, forward, init_lora, init_params
        from sam2.train import adamw_init, make_train_step
        plain = Sam2Config(image_size=128)
        fused = dataclasses.replace(plain, fused_attention=True)
        assert sum(spec.fused for spec in block_specs(fused)) == 3
        rng = np.random.default_rng(0)
        images = rng.normal(size=(1, 128, 128, 3)).astype(np.float32)
        boxes = np.array([[16, 20, 100, 88]], np.float32)
        masks = np.zeros((1, 128, 128), np.float32)
        masks[0, 20:88, 16:100] = 1.0
        lora = init_lora(plain, rank=4)

        def run(device, config):
            with jax.default_device(jax.devices(device)[0]):
                params = init_params(config, seed=0)
                logits = jax.jit(lambda i, b: forward(params, config, i, b))(images, boxes)
                step = jax.jit(make_train_step(params, config, (128, 128)))
                state, losses = (lora, adamw_init(lora)), []
                for _ in range(3):
                    *state, loss = step(*state, images, boxes, masks)
                    losses.append(loss)
                return logits, losses

        # The adapters are not compared: this Adam has no bias correction, so
        # an element whose gradient is at rounding level still moves by the
        # learning rate, in either direction.
        want = run("cpu", plain)
        assert float(want[1][2]) < float(want[1][0])
        for config in (plain, fused):
            assert_trees_close(run("metal", config), want, rtol=1e-4)
    """)


def test_recognized_attention_and_normalizations_match_cpu():
    """Plain attention on the fused kernel, in the forms the recognizer takes,
    and normalizations whose operand is stored for its readers."""
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        normal = lambda *shape: rng.standard_normal(shape, np.float32)

        def attention(q, k, v, scale=1.0):
            return jax.nn.softmax((q @ jnp.swapaxes(k, -1, -2)) * scale, axis=-1) @ v

        def layout(q, k, v):  # (batch, length, heads, head_dim), as models store it
            q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
            return attention(q * 0.125, k, v).transpose(0, 2, 1, 3)

        cases = {
            "heads": (attention, normal(2, 4, 256, 64), normal(2, 4, 256, 64), normal(2, 4, 256, 64)),
            "scaled": (lambda q, k, v: attention(q, k, v, 0.125), *(normal(3, 256, 64) for _ in range(3))),
            "more keys than queries": (attention, normal(2, 128, 96), normal(2, 512, 96), normal(2, 512, 96)),
            "32-row query tiles": (attention, normal(4, 96, 32), normal(4, 688, 32), normal(4, 688, 32)),
            "transposed layout": (layout, *(normal(1, 256, 4, 64) for _ in range(3))),
            "large scores": (lambda q, k, v: attention(q, k, v, 50.0), *(normal(2, 256, 32) for _ in range(3))),
            "fused operands": (lambda q, k, v: attention(jnp.tanh(q) + 1, k * 2, v - k), *(normal(2, 256, 32) for _ in range(3))),
            "gradient": (jax.grad(lambda q, k, v: (attention(q, k, v) ** 2).sum(), (0, 1, 2)), *(normal(2, 256, 32) for _ in range(3))),
        }
        for name, (fn, *args) in cases.items():
            assert_matches_cpu(fn, *args, rtol=1e-4, err_msg=name)

        def layer_norm(x, w, b):
            mean = x.mean(-1, keepdims=True)
            var = ((x - mean) ** 2).mean(-1, keepdims=True)
            return (x - mean) * jax.lax.rsqrt(var + 1e-6) * w + b

        x = normal(64, 64, 96)
        assert_matches_cpu(lambda x, y, w, b: layer_norm(jax.nn.gelu(x) + y, w, b), x, normal(64, 64, 96), normal(96), normal(96))
        assert_matches_cpu(lambda x: jax.nn.log_softmax(x * 3, axis=-1), x)
        assert_matches_cpu(lambda x: (x.max(-1), x.min(1), jnp.where(x > 2, jnp.nan, x).max(-1)), x)
        assert_matches_cpu(lambda x: x.astype(jnp.bfloat16).sum(-1), x, rtol=1e-2)
        assert_matches_cpu(lambda x: (x * 100).astype(jnp.int32).sum(-1), x)
    """)


def test_floating_remainder_is_exact():
    """Metal's fmod rounds the quotient: fmod(x, x) can return x, and a large
    quotient loses every digit. The remainder is exact, as on the CPU, also
    in a Pallas kernel."""
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
        from jax.experimental import pallas as pl
        from pjrt_tolerance import assert_matches_cpu
        rng = np.random.default_rng(0)
        a = (rng.standard_normal(2048) * 10.0 ** rng.integers(-6, 12, 2048)).astype(np.float32)
        b = (rng.standard_normal(2048) * 10.0 ** rng.integers(-6, 6, 2048)).astype(np.float32)
        a[:6] = [7.5, -7.5, 1e10, 5.0, -0.0, np.inf]
        b[:6] = [2.5, 2.5, 3.0, 0.1, 1.0, 2.0]
        assert_matches_cpu(lax.rem, a, b, exact=True)
        assert_matches_cpu(lambda a: lax.rem(a, a), a, exact=True)
        assert_matches_cpu(jnp.mod, a, b, exact=True)
        assert_matches_cpu(lambda a, b: lax.rem(a, jnp.zeros_like(b)), a, b, exact=True)
        for dtype in (np.float16, ml_dtypes.bfloat16):
            assert_matches_cpu(lax.rem, a.astype(dtype), (jnp.abs(b) + 1).astype(dtype), exact=True)

        def kernel(a_ref, b_ref, out_ref):
            out_ref[...] = lax.rem(a_ref[...], b_ref[...])

        block = pl.BlockSpec((16, 64), lambda i: (i, 0))
        call = pl.pallas_call(kernel, grid=(2,), in_specs=[block, block], out_specs=block,
                              out_shape=jax.ShapeDtypeStruct((32, 64), jnp.float32))
        got = jax.jit(call)(a.reshape(32, 64), b.reshape(32, 64))
        np.testing.assert_array_equal(got, np.fmod(a, b).reshape(32, 64))
    """)


def test_a_mosaic_matmul_kernel_runs_unchanged():
    """JAX's bundled TPU matmul: a scalar-prefetch grid spec, Mosaic's compiler
    parameters, and an accumulator in scratch along the K axis, which runs as
    a loop inside each thread."""
    _check("""
        import jax, numpy as np
        from jax.experimental.pallas.ops.tpu import matmul
        rng = np.random.default_rng(0)
        x, y = rng.standard_normal((2, 128, 128), np.float32)
        got = jax.jit(lambda x, y: matmul.matmul(x, y, block_shape=(32, 32), block_k=32))(x, y)
        np.testing.assert_allclose(got, x @ y, rtol=1e-4, atol=1e-4)
    """)
