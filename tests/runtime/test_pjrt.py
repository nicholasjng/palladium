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


def test_compilation_is_not_implemented_yet():
    """A CPU compile first, so the metal one hashes its topology for the cache key."""
    result = _run("""
        import jax, jax.numpy as jnp, numpy as np
        assert float(jax.device_put(np.ones(3, np.float32), jax.devices("cpu")[0]).sum()) == 3
        jnp.ones(3, device=jax.devices("metal")[0])
    """)
    assert result.returncode != 0
    assert "metal: compilation is not implemented yet" in result.stderr


def test_registers_only_when_jax_platforms_names_metal():
    out = _check("import jax; print(sorted({d.platform for d in jax.devices()}))", platforms=None)
    assert "metal" not in out


def test_registers_without_taking_the_default_when_listed_second():
    out = _check(
        "import jax; print(jax.default_backend(), len(jax.devices('metal')))", platforms="cpu,metal"
    )
    assert out.split() == ["cpu", "1"]
