"""The `metal` PJRT plugin through jax. jax picks its platforms once per process,
so each test runs a script in a subprocess with JAX_PLATFORMS set."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run(script: str, platforms: str | None = "metal,cpu") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    if platforms:
        env["JAX_PLATFORMS"] = platforms
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
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
        }
        for name, (fn, *args) in cases.items():
            got = jax.jit(fn)(*(jax.device_put(a, metal) for a in args))
            want = jax.jit(fn)(*(jax.device_put(a, cpu) for a in args))
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
                assert g.devices() == {metal}, name
                tol = 1e-2 if g.dtype.itemsize == 2 else 2e-5
                np.testing.assert_allclose(
                    np.asarray(g, np.float64), np.asarray(w, np.float64), rtol=tol, atol=1e-5,
                    err_msg=name,
                )
        print(len(cases))
    """)
    assert out.strip() == "15"


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
        ]
        for dtype in (np.int32, np.bool_, np.float16):
            cases += [
                (lambda x, y: jnp.concatenate([x, y, x], 0), x.astype(dtype), y.astype(dtype)),
                (lambda x, p: lax.pad(x, p, ((1, 2, 1), (2, -1, 0))),
                 x.astype(dtype), dtype(1)),
            ]
        for fn, *args in cases:
            run = jax.jit(fn)
            got = run(*(jax.device_put(a, metal) for a in args))
            want = run(*(jax.device_put(a, cpu) for a in args))
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
                np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    """)


def test_random_key_can_be_created_on_metal():
    _check("""
        import jax, numpy as np
        key = jax.random.key(17)
        assert key.devices() == {jax.devices('metal')[0]}
        np.testing.assert_array_equal(np.asarray(jax.random.key_data(key)), [0, 17])
    """)


def test_reductions_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
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
            run = jax.jit(fn)
            got, want = run(jax.device_put(x, metal)), run(jax.device_put(x, cpu))
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
                tol = 1e-2 if g.dtype.itemsize == 2 else 2e-5
                np.testing.assert_allclose(np.asarray(g, np.float64), np.asarray(w, np.float64),
                                           rtol=tol, atol=1e-5)
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
        for n in (37, 10001):
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
            (lambda x: lax.while_loop(lambda s: s[0] < np.uint32(2**32 - 1),
                lambda s: (s[0] + np.uint32(1), s[1] + 1),
                (np.uint32(2**32 - 3), x)), x),
        ]
        for fn, x in cases:
            run = jax.jit(fn)
            got, want = run(jax.device_put(x, metal)), run(jax.device_put(x, cpu))
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
                np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=2e-5, atol=1e-5)
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


def test_dynamic_loop_reports_unimplemented():
    result = _run("""
        import jax, numpy as np
        from jax import lax
        device = jax.devices('metal')[0]
        run = jax.jit(lambda n: lax.while_loop(lambda i: i < n, lambda i: i + 1, 0))
        run(jax.device_put(np.int32(3), device))
    """)
    assert result.returncode != 0
    assert "UNIMPLEMENTED: metal: stablehlo.while requires" in result.stderr


def test_sbi_trajectory_and_summary_match_cpu():
    _check(f"""
        import sys
        import jax, jax.numpy as jnp, numpy as np
        sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / "examples")!r})
        from training_steps import sbi_trajectory
        metal, cpu = jax.devices('metal')[0], jax.devices('cpu')[0]
        theta = np.random.default_rng(17).uniform(0.5, 1.5, (512, 4)).astype(np.float32)
        def simulate(theta):
            traj = sbi_trajectory(theta)
            return traj, jnp.concatenate([traj.mean(0), traj.std(0), traj[-1]], -1)
        run = jax.jit(simulate)
        got, want = run(jax.device_put(theta, metal)), run(jax.device_put(theta, cpu))
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=2e-5, atol=1e-5)
    """)


def test_dot_general_matches_cpu():
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
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
            run = jax.jit(fn)
            got = run(jax.device_put(a,metal),jax.device_put(b,metal))
            want = run(jax.device_put(a,cpu),jax.device_put(b,cpu))
            for g,w in zip(jax.tree.leaves(got),jax.tree.leaves(want),strict=True):
                tol = 1e-2 if g.dtype.itemsize == 2 else 2e-5
                np.testing.assert_allclose(np.asarray(g,np.float64),np.asarray(w,np.float64),
                                           rtol=tol,atol=tol,err_msg=f'case {index}')
    """)


def test_sbi_dense_training_matches_cpu_over_ten_steps():
    _check(f"""
        import sys
        import jax, numpy as np
        sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / "examples")!r})
        from training_steps import adamw, sbi_loss
        rng = np.random.default_rng(17)
        shapes = {{'w1': (6,128), 'w2': (128,128), 'wa': (128,5),
                   'wm': (128,20), 'ws': (128,20)}}
        params = {{k: (rng.standard_normal(s) * 0.02).astype(np.float32) for k,s in shapes.items()}}
        opt = (jax.tree.map(np.zeros_like, params), jax.tree.map(np.zeros_like, params), np.int32(0))
        x = rng.standard_normal((512,6)).astype(np.float32)
        theta = rng.uniform(0.5,1.5,(512,4)).astype(np.float32)
        @jax.jit
        def step(params,opt,x,theta):
            loss,grads = jax.value_and_grad(sbi_loss)(params,x,theta)
            return adamw(params,grads,opt),loss
        state = {{name: jax.device_put((params,opt,x,theta),jax.devices(name)[0])
                 for name in ('metal','cpu')}}
        for _ in range(10):
            losses = {{}}
            for name,(p,o,a,t) in state.items():
                (p,o),losses[name] = step(p,o,a,t)
                state[name] = (p,o,a,t)
            for g,w in zip(jax.tree.leaves((state['metal'][:2],losses['metal'])),
                           jax.tree.leaves((state['cpu'][:2],losses['cpu'])),strict=True):
                # Near-zero gradients amplify float32 reduction differences under Adam.
                np.testing.assert_allclose(np.asarray(g),np.asarray(w),rtol=1e-4,atol=2e-6)
    """)


def test_wide_integer_operations_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
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
                run=jax.jit(fn)
                got=run(jax.device_put(a,metal),jax.device_put(b,metal))
                want=run(jax.device_put(a,cpu),jax.device_put(b,cpu))
                for g,w in zip(jax.tree.leaves(got),jax.tree.leaves(want),strict=True):
                    np.testing.assert_array_equal(np.asarray(g),np.asarray(w))
    """)


def test_threefry_random_matches_cpu():
    _check("""
        import jax, numpy as np
        metal,cpu = jax.devices('metal')[0],jax.devices('cpu')[0]
        with jax.default_device(cpu):
            key=jax.random.key(17)
        for shape in ((),(0,),(1,),(7,),(512,6)):
            for fn in (lambda k:jax.random.uniform(k,shape),lambda k:jax.random.normal(k,shape),
                       lambda k:jax.random.bits(k,shape)):
                run=jax.jit(fn)
                got=run(jax.device_put(key,metal))
                want=run(jax.device_put(key,cpu))
                if got.dtype.kind == 'u':
                    np.testing.assert_array_equal(np.asarray(got),np.asarray(want))
                else:
                    np.testing.assert_allclose(np.asarray(got),np.asarray(want),rtol=2e-5,atol=1e-6)
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
    _check(f"""
        import sys
        import jax, numpy as np
        sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / "examples")!r})
        from training_steps import steps
        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            fn,params,opt,key = steps()['sbi']
            keys = [jax.random.fold_in(key,i) for i in range(10)]
        state = {{name:jax.device_put((params,opt),jax.devices(name)[0])
                 for name in ('metal','cpu')}}
        run = jax.jit(fn)
        for key in keys:
            losses = {{}}
            for name,(p,o) in state.items():
                state[name],losses[name] = run(p,o,jax.device_put(key,jax.devices(name)[0]))
            for g,w in zip(jax.tree.leaves((state['metal'],losses['metal'])),
                           jax.tree.leaves((state['cpu'],losses['cpu'])),strict=True):
                np.testing.assert_allclose(np.asarray(g),np.asarray(w),rtol=1e-4,atol=2e-6,
                                           equal_nan=False)
    """)


def test_cond_and_switch_match_cpu():
    _check("""
        import jax, jax.numpy as jnp, ml_dtypes, numpy as np
        from jax import lax
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
            run=jax.jit(fn)
            for p in (-100,0,1,2,100):
                got=run(jax.device_put(np.int32(p),metal),jax.device_put(x,metal))
                want=run(jax.device_put(np.int32(p),cpu),jax.device_put(x,cpu))
                for g,w in zip(jax.tree.leaves(got),jax.tree.leaves(want),strict=True):
                    np.testing.assert_allclose(np.asarray(g,np.float64),np.asarray(w,np.float64),
                                               rtol=2e-5,atol=1e-5,equal_nan=True)
    """)


def test_cond_predicates_and_inputs_survive_queued_calls():
    _check("""
        import jax,numpy as np
        from jax import lax
        metal=jax.devices('metal')[0]
        run=jax.jit(lambda p,x:lax.cond(p,lambda y:y+1,lambda y:y-1,x))
        results=[]
        for i in range(40):
            p=jax.device_put(np.bool_(i%2),metal)
            x=jax.device_put(np.full(4096,i,np.float32),metal)
            results.append(run(p,x))
            p.delete()
            x.delete()
        for i,y in enumerate(results):
            np.testing.assert_array_equal(np.asarray(y),np.full(4096,i+(1 if i%2 else -1),np.float32))
    """)


def test_raw_case_out_of_range_indices_select_last_branch():
    # JAX switch clamps before lowering; exercise raw StableHLO's different
    # negative-index behavior through its generated kernel as well.
    _check('''
        import numpy as np
        from jaxlib.mlir.dialects import stablehlo
        from palladium.stablehlo import compile_program
        from palladium._runtime import Kernel, run
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
        program = compile_program(stablehlo.serialize_portable_artifact_str(module, '1.0.0'))
        [launch] = program.launches
        kernel = Kernel(launch.source, launch.function)
        for index, expected in ((-100,30),(-1,30),(0,10),(1,20),(2,30),(3,30),(100,30)):
            buffers = [np.array(index,np.int32), np.array(0,np.int32)]
            run(kernel,[buffers[s] for s in launch.slots],launch.grid)
            assert buffers[program.outputs[0]].item() == expected
    ''')


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


def test_unsupported_ops_report_unimplemented():
    """A CPU compile first, so the metal one hashes its topology for the cache key."""
    result = _run("""
        import jax, jax.numpy as jnp, numpy as np
        assert float(jax.device_put(np.ones(3, np.float32), jax.devices("cpu")[0]).sum()) == 3
        jax.jit(lambda x: x.prod())(jnp.ones(3, device=jax.devices("metal")[0]))
    """)
    assert result.returncode != 0
    assert "UNIMPLEMENTED: metal: stablehlo.reduce has an unsupported reducer body" in result.stderr


def test_registers_only_when_jax_platforms_names_metal():
    out = _check("import jax; print(sorted({d.platform for d in jax.devices()}))", platforms=None)
    assert "metal" not in out


def test_registers_without_taking_the_default_when_listed_second():
    out = _check(
        "import jax; print(jax.default_backend(), len(jax.devices('metal')))", platforms="cpu,metal"
    )
    assert out.split() == ["cpu", "1"]
