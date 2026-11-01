"""`palladium.stablehlo`: StableHLO programs to fused Metal kernels, checked on
the generated program; the metal PJRT plugin runs them in tests/runtime/test_pjrt.py."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jaxlib.mlir.dialects import stablehlo

from palladium.stablehlo import UnsupportedProgram, compile_program


def _compile(fn, *shapes, dtype=np.float32, version="1.0.0"):
    lowered = jax.jit(fn).lower(*(jax.ShapeDtypeStruct(s, dtype) for s in shapes))
    text = str(lowered.compiler_ir("stablehlo"))
    return compile_program(stablehlo.serialize_portable_artifact_str(text, version))


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


def test_padding_empty_operand_never_binds_it():
    # A padding-only result must not read an empty input, even speculatively.
    program = _compile(lambda x, p: lax.pad(x, p, ((2, 3, 0), (1, 1, 0))), (0, 4), ())
    [launch] = program.launches
    assert 0 not in launch.slots
    assert program.slots[program.outputs[0]].shape == (5, 6)


def test_guarded_index_maps_still_fuse():
    program = _compile(
        lambda x, y: jnp.concatenate([lax.pad(x * 2, np.float32(-1), ((1, 0, 1),)), y]),
        (4,),
        (3,),
    )
    assert len(program.launches) == 1
    assert [s.kind for s in program.slots] == ["parameter", "parameter", "fresh"]


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (lambda x: x.prod(), "stablehlo.reduce"),
        (lambda x: x.argmax(), "stablehlo.reduce"),
        (lambda x: x.astype(jnp.int32) @ x.astype(jnp.int32), "stablehlo.dot_general"),
    ],
)
def test_rejects_unsupported_ops(fn, message):
    with pytest.raises(UnsupportedProgram, match=message):
        _compile(fn, (4, 4))


def test_reductions_materialize_once_before_consumers():
    def fn(x):
        s = (x * x).sum(-1)
        return s, s, x + s[:, None], s.sum()

    program = _compile(fn, (3, 17))
    assert len(program.launches) == 3
    assert program.outputs[0] == program.outputs[1]
    # No materialization of x*x; consumers share the reduction's slot.
    first, second, consumer = program.launches
    assert first.slots == (0, program.outputs[0])
    assert second.slots == (program.outputs[0], program.outputs[-1])
    assert program.outputs[0] in consumer.slots[:-1]
    assert first.threadgroup == second.threadgroup == 128


def test_full_reduction_splits_into_partials():
    program = _compile(lambda x: x.sum(), (10001,))
    assert len(program.launches) == 2
    first, second = program.launches
    partial = first.slots[-1]
    assert program.slots[partial].shape == (3,)
    assert partial == second.slots[0]
    assert first.grid == 3 * 128 and second.grid == 128


def test_counted_loop_reuses_sources_and_bounded_storage():
    def compile_scan(length):
        return _compile(
            lambda x: lax.scan(lambda c, _: (c + 1, None), x, None, length=length)[0], (4,)
        )

    short, long = compile_scan(3), compile_scan(200)
    assert len(short.slots) == len(long.slots)
    assert len(long.launches) > len(short.launches)
    assert (
        len({launch.source for launch in long.launches}) == 3
    )  # counter init and two body kernels
    # The result alternates output banks according to parity.
    assert short.outputs != long.outputs


def test_zero_trip_loop_returns_initial_value():
    program = _compile(
        lambda x: lax.while_loop(lambda s: s[0] < 0, lambda s: (s[0] + 1, s[1] * 2), (0, x))[1],
        (4,),
    )
    assert program.outputs == [0]
    assert not any(launch.function.startswith("fused") for launch in program.launches)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: lax.while_loop(lambda s: s[0] < x[0], lambda s: (s[0] + 1, s[1] + 2), (0, x))[1],
        lambda x: lax.while_loop(lambda s: s[0] < 3, lambda s: (s[0] + 2, s[1] + 2), (0, x))[1],
        lambda x: lax.while_loop(lambda s: s[0] <= 3, lambda s: (s[0] + 1, s[1] + 2), (0, x))[1],
    ],
)
def test_unproven_loop_is_rejected(fn):
    with pytest.raises(UnsupportedProgram, match="stablehlo.while requires"):
        _compile(fn, (4,))


def test_loop_replay_budget_is_checked():
    with pytest.raises(UnsupportedProgram, match="replay limit"):
        _compile(lambda x: lax.fori_loop(0, 100001, lambda i, y: y + 1, x), (4,))


def test_aligned_dot_reuses_simdgroup_emitter_without_input_copies():
    program = _compile(lambda a, b: a @ b, (32, 64), (64, 48))
    [launch] = program.launches
    assert launch.slots == (0, 1, program.outputs[0])
    assert "simdgroup_multiply_accumulate" in launch.source
    assert launch.grid == (128, 1, 3) and launch.threadgroup == 128


def test_dot_tail_uses_existing_tensorops_emitter():
    program = _compile(lambda a, b: a @ b, (3, 7), (7, 5))
    [launch] = program.launches
    assert "matmul2d" in launch.source


def test_dot_dependencies_materialize_before_consumers():
    program = _compile(lambda a, b: (a * 2) @ b.T + 1, (32, 16), (48, 16))
    assert len(program.launches) == 4  # two packed operands, GEMM, epilogue
    gemm = program.launches[2]
    assert gemm.function == "gemm"
    assert gemm.slots[:2] == tuple(l.slots[-1] for l in program.launches[:2])
    assert gemm.slots[-1] in program.launches[-1].slots[:-1]


def test_empty_dot_never_reads_inputs():
    program = _compile(lambda a, b: a @ b, (3, 0), (0, 5))
    [launch] = program.launches
    assert launch.slots == (program.outputs[0],)


def test_dot_explicit_algorithm_is_rejected():
    with pytest.raises(UnsupportedProgram, match="explicit algorithms"):
        _compile(
            lambda a, b: lax.dot_general(
                a, b, (((1,), (0,)), ((), ())), precision=lax.DotAlgorithmPreset.F32_F32_F32
            ),
            (16, 16),
            (16, 16),
            version=stablehlo.get_current_version(),
        )


def test_wide_integer_types_stay_local_to_pjrt():
    from palladium.emit.core import CTYPES

    assert np.dtype(np.uint64) not in CTYPES and np.dtype(np.int64) not in CTYPES
    with jax.enable_x64():
        program = _compile(
            lambda x: (x + np.uint64(2**63 + 1)) >> np.uint64(32), (4,), dtype=np.uint64
        )
    assert "ulong" in program.launches[0].source
    assert "9223372036854775809ul" in program.launches[0].source


def test_unsupported_wide_operations_fail_explicitly():
    with jax.enable_x64():
        with pytest.raises(UnsupportedProgram, match="reduce on 64-bit"):
            _compile(lambda x: x.sum(), (4,), dtype=np.int64)
        with pytest.raises(UnsupportedProgram, match="shape-changing"):
            _compile(lambda x: lax.bitcast_convert_type(x, jnp.int64), (2, 2), dtype=np.int32)


def test_elementwise_cond_fuses_with_guarded_branch_reads():
    program = _compile(lambda x: lax.cond(x[0] > 0, lambda y: y * 2, lambda y: y + 3, x), (8,))
    [launch] = program.launches
    assert "if (" in launch.source and "else {" in launch.source
    assert [s.kind for s in program.slots] == ["parameter", "fresh"]


def test_cond_materializes_reductions_before_selection():
    program = _compile(lambda x: lax.cond(x[0] > 0, lambda y: y.sum(), lambda y: y.max(), x), (8,))
    assert len(program.launches) == 3
    assert all(l.function.startswith("reduce") for l in program.launches[:2])
    assert {l.slots[-1] for l in program.launches[:2]} <= set(program.launches[-1].slots[:-1])


def test_unsupported_op_in_case_is_not_silently_discarded():
    with pytest.raises(UnsupportedProgram, match="reduce has an unsupported reducer"):
        _compile(lambda x: lax.cond(x[0] > 0, lambda y: y.sum(), lambda y: y.prod(), x), (8,))


def test_gather_fuses_producers_without_intermediate_buffers():
    program = _compile(lambda x, i: (x * 2)[i.astype(jnp.int32)] + 1, (7, 3), (4,))
    assert len(program.launches) == 1
    assert [s.kind for s in program.slots] == ["parameter", "parameter", "fresh"]
    assert program.slots[program.outputs[0]].shape == (4, 3)


def test_scatter_initialization_precedes_atomic_updates():
    program = _compile(lambda x, i, v: x.at[i.astype(jnp.int32)].add(v), (7, 3), (4,), (4, 3))
    assert len(program.launches) == 2
    init, update = program.launches
    assert init.grid == 21 and update.grid == 12
    assert init.slots[-1] == update.slots[-1] == program.outputs[0]
    assert "atomic_fetch_add_explicit" in update.source


def test_non_add_scatter_is_rejected():
    with pytest.raises(UnsupportedProgram, match="single add update"):
        _compile(lambda x, i: x.at[i.astype(jnp.int32)].set(1), (7, 3), (4,))
