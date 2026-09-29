"""MSL lowering rules grouped by responsibility; importing this package
populates the ``RULES`` registry.
"""

from . import (
    array as array,
    control as control,
    dot as dot,
    elementwise as elementwise,
    memory as memory,
    random as random,
    structural as structural,
)
