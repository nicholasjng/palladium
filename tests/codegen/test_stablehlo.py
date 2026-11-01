"""`palladium.stablehlo`: StableHLO programs to fused Metal kernels, checked on
the generated program; the metal PJRT plugin runs them in tests/runtime/test_pjrt.py."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxlib.mlir.dialects import stablehlo

from palladium.stablehlo import UnsupportedProgram, compile_program


def _compile(fn, *shapes, dtype=np.float32):
    lowered = jax.jit(fn).lower(*(jax.ShapeDtypeStruct(s, dtype) for s in shapes))
    text = str(lowered.compiler_ir("stablehlo"))
    return compile_program(stablehlo.serialize_portable_artifact_str(text, "1.0.0"))


def test_one_kernel_per_output_shape():
    program = _compile(lambda x: (x + 1, x * 2, x.T), (3, 4))
    assert [s.kind for s in program.slots] == ["parameter", "fresh", "fresh", "fresh"]
    assert program.outputs == [1, 2, 3]
    assert [(launch.grid, launch.slots) for launch in program.launches] == [
        (12, (0, 1, 2)),
        (12, (0, 3)),
    ]


def test_index_maps_fuse_without_copies():
    """Transposes, broadcasts and reshapes become index arithmetic on one read."""
    program = _compile(lambda x, v: x.T.reshape(2, 6) * v.reshape(1, 6), (3, 4), (6,))
    [launch] = program.launches
    body = launch.source.split("{", 1)[1]
    assert body.count("b0[") == 1 and body.count("b1[") == 1
    assert not re.search(r"\w+ v\d+ = v\d+;", body), body  # no copies between values


def test_reshape_of_a_parameter_reads_at_the_flat_index():
    [launch] = _compile(lambda x: x.reshape(2, 6) + 1, (3, 4)).launches
    assert "b0[i]" in launch.source


def test_parameters_and_constant_arrays_pass_through_as_slots():
    program = _compile(lambda x: (x, jnp.array([1.0, 2.0], jnp.float32)), (2,))
    assert [s.kind for s in program.slots] == ["parameter", "constant"]
    assert program.outputs == [0, 1] and not program.launches
    np.testing.assert_array_equal(np.frombuffer(program.slots[1].data, np.float32), [1, 2])


def test_constant_arrays_read_by_a_kernel_become_slots():
    program = _compile(lambda x: x + jnp.array([1.0, 2.0], jnp.float32), (2,))
    assert [s.kind for s in program.slots] == ["parameter", "fresh", "constant"]
    assert program.launches[0].slots == (0, 2, 1)


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (lambda x: x.sum(), "stablehlo.reduce"),
        (lambda x: x @ x, "stablehlo.dot_general"),
    ],
)
def test_rejects_unsupported_ops(fn, message):
    with pytest.raises(UnsupportedProgram, match=message):
        _compile(fn, (4, 4))
