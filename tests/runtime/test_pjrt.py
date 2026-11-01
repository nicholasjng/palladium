"""The `metal` PJRT plugin through jax. jax picks its platforms once per process,
so each test runs a script in a subprocess with JAX_PLATFORMS set."""

import os
import subprocess
import sys
import textwrap


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


def test_unsupported_ops_report_unimplemented():
    """A CPU compile first, so the metal one hashes its topology for the cache key."""
    result = _run("""
        import jax, jax.numpy as jnp, numpy as np
        assert float(jax.device_put(np.ones(3, np.float32), jax.devices("cpu")[0]).sum()) == 3
        jax.jit(lambda x: x.sum())(jnp.ones(3, device=jax.devices("metal")[0]))
    """)
    assert result.returncode != 0
    assert "UNIMPLEMENTED: metal: stablehlo.reduce is not supported yet" in result.stderr


def test_registers_only_when_jax_platforms_names_metal():
    out = _check("import jax; print(sorted({d.platform for d in jax.devices()}))", platforms=None)
    assert "metal" not in out


def test_registers_without_taking_the_default_when_listed_second():
    out = _check(
        "import jax; print(jax.default_backend(), len(jax.devices('metal')))", platforms="cpu,metal"
    )
    assert out.split() == ["cpu", "1"]
