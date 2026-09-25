"""Load and register the MSL lowering rules.

Implementations are grouped by responsibility; importing this package keeps
the historical side effect of populating the public ``RULES`` registry.
"""

from . import (
    array as array,
    control as control,
    dot as dot,
    elementwise as elementwise,
    memory as memory,
    random as random,
)
