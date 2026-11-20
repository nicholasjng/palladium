"""Context and import paths for every mew run.

mew imports this before discovering benchmarks (``[tool.mew] setup``), in
every environment the suite runs from, so it needs nothing beyond mew. It
records the commit, the JAX platforms or MLX build, package versions, and
what a timing depends on besides the code: the macOS version, the power
source and low-power mode, and the Metal device's limits. It also puts
``examples/`` on the import path for the workloads the benchmarks share.
"""

import importlib.metadata
import os
import platform
import subprocess
import sys

import mew

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))


def _pmset(*args: str) -> str:
    try:
        return subprocess.run(
            ["pmset", "-g", *args], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _low_power_mode() -> bool | None:
    for line in _pmset().splitlines():
        if line.split()[:1] == ["lowpowermode"]:
            return line.split()[1] == "1"
    return None


mew.update_context(mew.vcs_context())
mew.set_context("jax_platforms", os.environ.get("JAX_PLATFORMS"))
mew.set_context("bench_backend", os.environ.get("BENCH_BACKEND"))
mew.set_context("macos", platform.mac_ver()[0] or None)
mew.set_context("power.source", "AC" if "AC Power" in _pmset("batt") else "battery")
mew.set_context("power.low_power_mode", _low_power_mode())
for package in ("jax", "jaxlib", "jax-mps", "mlx", "palladium"):
    try:
        mew.set_context(f"versions.{package}", importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        pass
try:
    from palladium.device import device_limits
except ImportError:  # the jax-mps and MLX environments have no palladium
    pass
else:
    mew.set_context("metal_device", device_limits())
