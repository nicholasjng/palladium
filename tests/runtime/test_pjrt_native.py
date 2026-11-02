"""The native PJRT dependency-error test, built in this interpreter's build tree.

The target is excluded from the default build so the wheel never ships it;
`cmake --build` compiles it from the current sources in the directory that
scikit-build-core configured (`build/{wheel_tag}`).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TARGET = "palladium_pjrt_dependencies_test"


def _build_dir() -> Path | None:
    # Wheel tags start with the interpreter tag, e.g. cp314t-cp314t-macosx_...
    prefix = f"cp{sys.version_info.major}{sys.version_info.minor}{sys.abiflags}-"
    caches = sorted(
        (_ROOT / "build").glob(f"{prefix}*/CMakeCache.txt"), key=lambda p: p.stat().st_mtime
    )
    return caches[-1].parent if caches else None


def test_dependency_failures_propagate_without_host_waits():
    build = _build_dir()
    cmake = shutil.which("cmake")
    if build is None or cmake is None:
        pytest.skip("needs cmake and a configured build directory (run `uv sync`)")
    built = subprocess.run(
        [cmake, "--build", str(build), "--target", _TARGET],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run(
        [str(build / _TARGET)], check=False, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PJRT dependency tests passed" in result.stdout
