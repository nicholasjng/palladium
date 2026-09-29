"""Experimental APIs: usable, but their interfaces may change between
releases without a deprecation period, like ``jax.experimental``.

- ``df32``: float64 emulated as float32 pairs in hand-written MSL, for
  kernels whose accumulated round-off is the limiting error.
"""
