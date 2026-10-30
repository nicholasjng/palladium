"""The `metal` PJRT plugin (`native/pjrt`), registered with jax through the
`jax_plugins` entry point, which calls `initialize` on backend discovery.

The plugin has a device and buffers but no compiler yet, so it registers only
when JAX_PLATFORMS names `metal` (e.g. `JAX_PLATFORMS=metal,cpu`); otherwise
every jax process would bring it up and warn that it is experimental.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from jax._src import config, xla_bridge

_LIBRARY_NAME = "libpalladium_pjrt.dylib"


def library_path() -> Path:
    """The plugin library in the installed package; reject a missing build."""
    candidate = importlib.resources.files("palladium").joinpath(_LIBRARY_NAME)
    if not candidate.is_file():
        raise FileNotFoundError(f"palladium's PJRT plugin ({_LIBRARY_NAME}) isn't built")
    return Path(str(candidate))


def initialize() -> None:
    if "metal" in (config.jax_platforms.value or "").split(","):
        xla_bridge.register_plugin("metal", library_path=str(library_path()))
