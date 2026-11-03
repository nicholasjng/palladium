"""Context for every mew run: the commit, the JAX platforms or MLX build, and versions.

mew imports this before discovering benchmarks (``[tool.mew] setup``), in
every environment the suite runs from, so it needs nothing beyond mew.
"""

import importlib.metadata
import os

import mew

mew.update_context(mew.vcs_context())
mew.set_context("jax_platforms", os.environ.get("JAX_PLATFORMS"))
for package in ("jax", "jaxlib", "jax-mps", "mlx", "palladium"):
    try:
        mew.set_context(f"versions.{package}", importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        pass
