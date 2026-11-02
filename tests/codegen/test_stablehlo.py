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


def test_one_kernel_per_output_element_count():
    """Outputs of equal size share a launch, even with different shapes."""
    program = _compile(lambda x: (x + 1, x * 2, x.T, x[:2]), (3, 4))
    assert [s.kind for s in program.slots] == ["parameter", *["fresh"] * 4]
    assert program.outputs == [1, 2, 3, 4]
    assert [(launch.grid, launch.slots) for launch in program.launches] == [
        (12, (0, 1, 2, 3)),
        (8, (0, 4)),
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

    program = _compile(fn, (17, 17))  # both reductions are long: not folded
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
    # One launch per iteration: the counter is a constant, not a kernel.
    assert len(short.launches) == 3 and len(long.launches) == 200
    assert len({launch.source for launch in long.launches}) == 1
    # The result alternates output banks according to parity.
    assert short.outputs != long.outputs


def test_loop_counter_binds_one_constant_array_at_offsets():
    def loop(x):
        return lax.while_loop(lambda s: s[0] < 7, lambda s: (s[0] + 1, s[1] + s[0]), (3, x))[1]

    program = _compile(loop, (4,))
    [array] = [k for k, s in enumerate(program.slots) if s.kind == "constant" and s.shape == (4,)]
    assert np.frombuffer(program.slots[array].data, np.int32).tolist() == [3, 4, 5, 6]
    offsets = [launch.offsets[launch.slots.index(array)] for launch in program.launches]
    assert offsets == [0, 4, 8, 12]


def test_scan_outputs_update_in_place_with_one_launch_per_iteration():
    """Stacking ys writes one row per iteration, never the whole trajectory."""
    program = _compile(
        lambda x: lax.scan(lambda c, _: (c * 0.5 + 1, c), x, None, length=50), (3, 4)
    )
    trajectory = [k for k, s in enumerate(program.slots) if s.shape == (50, 3, 4)]
    body = [launch for launch in program.launches if launch.grid == 12]
    assert len(body) == 50, [launch.grid for launch in program.launches]
    assert all(launch.grid <= 50 * 12 for launch in program.launches)
    # Every iteration writes the same trajectory buffer.
    assert len({s for launch in body for s in launch.slots if s in trajectory}) == 1


def test_carry_read_besides_its_update_is_not_updated_in_place():
    def body(i, t):
        previous = lax.dynamic_slice(t, (i - 1, 0), (1, 4))
        return lax.dynamic_update_slice(t, previous * 2 + 1, (i, 0))

    program = _compile(lambda t: lax.fori_loop(1, 6, body, t), (6, 4))
    # Each iteration rewrites the whole carry into a fresh bank. fori_loop's
    # own index is an induction variable too: no per-iteration scalar kernel.
    assert [launch.grid for launch in program.launches] == [24] * 5


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
    program = _compile(lambda x: lax.cond(x[0] > 0, lambda y: y.sum(), lambda y: y.max(), x), (32,))
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


def test_argmax_uses_one_variadic_reduction():
    program = _compile(lambda x: x.argmax(1), (3, 17))
    [launch] = program.launches
    assert launch.function.startswith("variadic_reduce")
    assert launch.math_mode == 0
    assert len(launch.slots) == 3  # input, value result, index result


def test_concatenate_siblings_share_values_independent_of_the_guarded_dimension():
    """A value both operands need is computed once, before the branch chain."""

    def fn(x):
        shared = x * 3 + 1  # not an atom: it gets a temporary
        return jnp.concatenate([shared[:, None], (shared * 2)[:, None]], 1)

    [launch] = _compile(fn, (5,)).launches
    body = launch.source.split("{", 1)[1]
    assert body.count("* 3.0f") == 1, body
    assert body.index("* 3.0f") < body.index("if ("), body


def test_cond_branches_stay_lazy():
    """Selection evaluates only the taken branch: nothing moves before it."""
    [launch] = _compile(
        lambda p, x: lax.cond(p > 0, lambda y: jnp.exp(y), lambda y: jnp.exp(y) + 1, x),
        (),
        (5,),
    ).launches
    body = launch.source.split("{", 1)[1]
    assert body.count("exp(") == 2 and body.index("if (") < body.index("exp("), body


def test_pad_reads_stay_inside_their_guard():
    """The operand is smaller than the output: its reads need the guard."""

    def fn(v):
        return lax.pad(jnp.broadcast_to(v[:, None], (3, 4)), np.float32(0), ((1, 1, 0), (0, 0, 0)))

    [launch] = _compile(fn, (3,)).launches
    body = launch.source.split("{", 1)[1]
    assert body.index("if (") < body.index("b0["), body


def test_rk4_simulation_kernel_computes_each_value_once():
    """The SBI trajectory's loop body: stacked RK4 stages share their components."""
    from training_steps import sbi_trajectory  # examples/ is on the path (conftest)

    program = _compile(sbi_trajectory, (512, 4))
    source = max((launch.source for launch in program.launches), key=len)
    # The StableHLO body has 46 arithmetic ops; recomputation once reached 4,032.
    assert len(re.findall(r"\n\s+float v\d+ =", source)) <= 80, source
    assert len(re.findall(r"\bb\d+\[", source)) <= 10, source


def _double_and_sum(x_ref, y_ref, o_ref, p_ref):
    o_ref[...] = x_ref[...] * 2
    p_ref[...] = x_ref[...] + y_ref[...]


def test_palladium_dispatch_binds_inputs_then_results():
    """`metal_call` (and `pallas_call` lowered for metal) is one launch of its MSL."""
    from jax.experimental import pallas as pl

    import palladium

    spec = pl.BlockSpec((4,), lambda i: (i,))
    call = palladium.metal_call(
        _double_and_sum,
        grid=(2,),
        in_specs=[spec, spec],
        out_specs=[spec, spec],
        out_shape=[jax.ShapeDtypeStruct((8,), jnp.float32)] * 2,
        compiler_params=palladium.CompilerParams(math_mode=palladium.MathMode.SAFE),
    )
    program = _compile(lambda x, y: call(x + 1, y), (8,), (8,))
    fused, dispatch = program.launches
    assert fused.slots == (0, 2)  # x + 1, materialized for the kernel
    assert dispatch.slots == (2, 1, *program.outputs)
    assert dispatch.function == "palladium_kernel" and "kernel void" in dispatch.source
    assert dispatch.grid == (2, 1, 1) and dispatch.threadgroup == 0
    assert dispatch.math_mode == 0 and dispatch.offsets == ()


def test_aliased_dispatch_result_starts_as_a_copy_of_its_input():
    """An aliased result keeps the input's values wherever the kernel does not write."""
    from jax.experimental import pallas as pl

    import palladium

    def first_half(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 10

    call = palladium.metal_call(
        first_half,
        grid=(1,),
        in_specs=[pl.BlockSpec((4,), lambda i: (0,))],
        out_specs=pl.BlockSpec((4,), lambda i: (0,)),
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        input_output_aliases={0: 0},
    )
    program = _compile(call, (8,))
    copy, dispatch = program.launches
    [output] = program.outputs
    assert copy.slots == (0, output) and dispatch.slots == (0, output)


def test_vmapped_dispatch_launches_once_per_batch_element_at_offsets():
    from jax.experimental import pallas as pl

    import palladium

    spec = pl.BlockSpec((4,), lambda i: (i,))
    call = palladium.metal_call(
        _double_and_sum,
        grid=(2,),
        in_specs=[spec, spec],
        out_specs=[spec, spec],
        out_shape=[jax.ShapeDtypeStruct((8,), jnp.float32)] * 2,
    )
    # y is shared by the batch: stride 0.
    program = _compile(jax.vmap(call, in_axes=(0, None)), (3, 8), (8,))
    assert [launch.offsets for launch in program.launches] == [
        (),
        (32, 0, 32, 32),
        (64, 0, 64, 64),
    ]


def test_unknown_custom_calls_are_unsupported():
    def fn(x):
        return jax.ffi.ffi_call("somebody_elses_kernel", jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    with pytest.raises(UnsupportedProgram, match="somebody_elses_kernel"):
        _compile(fn, (4,))


def test_kernels_bind_at_most_31_buffers():
    """Metal's limit: wide fusions split by output, deep ones by operand."""
    wide = _compile(lambda *xs: tuple(x + 1 for x in xs), *[(4,)] * 20)
    deep = _compile(lambda *xs: sum(x * k for k, x in enumerate(xs)), *[(4,)] * 40)
    for program in (wide, deep):
        assert all(len(launch.slots) <= 31 for launch in program.launches)
    assert len(wide.launches) == 2
    # The sum is split once: one stored partial, then the rest.
    assert len(deep.launches) == 2


def test_short_reductions_fold_into_their_consumers():
    """No launch of 128 threads per output to add four numbers."""

    def fn(x):
        p = jax.nn.softmax(x, axis=-1)  # max and sum over 4 elements
        return p * x.sum(0)  # and over 3

    [launch] = _compile(fn, (3, 4)).launches
    assert launch.grid == 12 and launch.threadgroup == 0
    assert launch.math_mode == 1  # RELAXED: a float maximum keeps its NaN rules
    [launch] = _compile(lambda x: (x * 2).sum(-1) + 1, (5, 16)).launches
    assert launch.grid == 5 and launch.math_mode == 2
    assert launch.source.count("b0[") == 16
    assert [launch.function[:6] for launch in _compile(lambda x: x.sum(-1), (5, 17)).launches] == [
        "reduce"
    ]


def test_short_reductions_read_in_branches_are_stored_once():
    """Branch coordinates get fresh names, so folds there would be regenerated."""
    program = _compile(lambda x: jnp.concatenate([x[:, 0], x.sum(-1)]), (4, 3))
    stored, fused = program.launches
    assert stored.grid == 4 and stored.slots[-1] in fused.slots[:-1]
    # At the body's coordinates, the reduction folds into the one kernel.
    assert len(_compile(lambda x: x.sum(-1) * 2, (4, 3)).launches) == 1


def test_plain_jax_cnf_kernels_stay_small():
    """Folding at branch coordinates once made one of these kernels 9 MB of MSL."""
    import cnf_density  # examples/ is on the path (conftest)

    update, _ = cnf_density.make_training_step(256, variant="jax")
    exported = jax.export.export(update, platforms=["metal"])(
        cnf_density.initial_state(), cnf_density.mixture_data(256)
    )
    program = compile_program(exported.mlir_module_serialized)
    assert max(len(launch.source) for launch in program.launches) < 100_000
