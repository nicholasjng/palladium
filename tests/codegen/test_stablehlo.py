"""`palladium.stablehlo`: StableHLO programs to fused Metal kernels, checked on
the generated program and on the compiler's parts; the metal PJRT plugin runs
programs in tests/runtime/test_pjrt.py."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jaxlib.mlir.dialects import stablehlo

from palladium.device import prefers_simdgroup_matrix
from palladium.stablehlo import UnsupportedProgram, compile_program
from palladium.stablehlo.graph import Op, Parameter
from palladium.stablehlo.program import RELAXED, SAFE, Launch
from palladium.stablehlo.schedule import Product, Scheduler, induction_values, rebound
from palladium.stablehlo.term import Term, delinear, linear


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


def test_guarded_index_maps_fuse():
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


def test_aligned_dot_reuses_the_pallas_gemm_without_input_copies():
    """SIMD-group matrices, or `matmul2d` on GPUs with matrix accelerators."""
    program = _compile(lambda a, b: a @ b, (32, 64), (64, 48))
    [launch] = program.launches
    assert launch.slots == (0, 1, program.outputs[0])
    kernel = "simdgroup_multiply_accumulate" if prefers_simdgroup_matrix() else "matmul2d"
    assert kernel in launch.source
    assert launch.grid == (128, 3, 1) and launch.threadgroup == 128


def test_dot_tail_uses_the_tensorops_emitter():
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
    assert launch.math_mode == SAFE
    assert len(launch.slots) == 3  # input, value result, index result


def test_concatenate_siblings_share_values_independent_of_the_guarded_dimension():
    """A value both operands need is computed once, before the branch chain."""

    def fn(x):
        shared = x * 3 + 1
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
    # The StableHLO body has 46 arithmetic ops.
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
    assert dispatch.grid == (2, 1, 1) and dispatch.threadgroup == (0, 0, 0)
    assert dispatch.math_mode == SAFE and dispatch.offsets == ()


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
    assert launch.math_mode == RELAXED  # a float maximum keeps its NaN rules
    [launch] = _compile(lambda x: (x * 2).sum(-1) + 1, (5, 16)).launches
    assert launch.grid == 5 and launch.math_mode == RELAXED
    assert launch.source.count("b0[") == 16
    assert [launch.function[:6] for launch in _compile(lambda x: x.sum(-1), (5, 17)).launches] == [
        "reduce"
    ]


def test_short_reductions_read_in_branches_are_stored_once():
    """Every branch reading a fold would regenerate it: it gets a buffer instead."""
    program = _compile(lambda x: jnp.concatenate([x[:, 0], x.sum(-1)]), (4, 3))
    stored, fused = program.launches
    assert stored.grid == 4 and stored.slots[-1] in fused.slots[:-1]
    # At the body's coordinates, the reduction folds into the one kernel.
    assert len(_compile(lambda x: x.sum(-1) * 2, (4, 3)).launches) == 1


def test_plain_jax_cnf_kernels_stay_small():
    """Short reductions read inside branches are stored, not regenerated per
    branch, which keeps these kernels far below a megabyte of MSL."""
    import cnf_density  # examples/ is on the path (conftest)

    update, _ = cnf_density.make_training_step(256, variant="jax")
    exported = jax.export.export(update, platforms=["metal"])(
        cnf_density.initial_state(), cnf_density.mixture_data(256)
    )
    program = compile_program(exported.mlir_module_serialized)
    assert max(len(launch.source) for launch in program.launches) < 100_000


def test_transposed_convolutions_skip_the_dilation_zeros():
    """A 4x4 stride-4 transposed convolution packs one tap per output phase,
    not a 16x larger im2col matrix of mostly zeros."""
    x = jax.ShapeDtypeStruct((2, 32, 32, 64), jnp.float32)
    w = jax.ShapeDtypeStruct((4, 4, 64, 8), jnp.float32)
    up = jax.jit(
        lambda x, w: jax.lax.conv_transpose(
            x, w, (4, 4), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
    )
    program = compile_program(
        jax.export.export(up, platforms=["metal"])(x, w).mlir_module_serialized
    )
    largest = max(int(np.prod(slot.shape)) for slot in program.slots)
    assert largest == 2 * 128 * 128 * 8  # the output; patches are the input's size


def test_long_contractions_with_few_tiles_split_k():
    a = jax.ShapeDtypeStruct((64, 8192), jnp.float32)
    program = compile_program(
        jax.export.export(jax.jit(lambda a, b: a @ b), platforms=["metal"])(
            a, a.update(shape=(8192, 64))
        ).mlir_module_serialized
    )
    names = [launch.function for launch in program.launches]
    assert names.count("gemm") == 1 and any(n.startswith("split_k_sum") for n in names)
    [gemm] = [launch for launch in program.launches if launch.function == "gemm"]
    assert program.slots[gemm.slots[2]].shape == (1, 16, 64, 64)


def test_elementwise_kernels_keep_nans():
    """Fused kernels run RELAXED, which keeps `x - x` a NaN for NaN and
    infinite x, and a float compared with itself (how JAX spells isnan) is
    `isnan`, which no math mode folds away."""
    [launch] = _compile(lambda x: (jnp.isnan(x), lax.eq(x, x), x - x), (4,)).launches
    assert launch.math_mode == RELAXED
    assert launch.source.count("isnan(") == 2
    # Comparing two different values stays a comparison.
    [launch] = _compile(lambda x, y: x != y, (4,), (4,)).launches
    assert "isnan" not in launch.source


def test_values_hoist_out_of_nested_branches_to_their_own_level():
    """A value hoisted out of an inner branch can read one that hoists further:
    each is declared in the block it is memoized for, here the kernel body for
    the scalar, which a sibling of the outer branch reads too."""

    def fn(v, s, b, c):
        row = jnp.broadcast_to((v * s)[:, None], b.shape)  # depends on the outer branch only
        return jnp.concatenate([jnp.concatenate([row, b], 1), c * s], 0)

    [launch] = _compile(fn, (4,), (), (4, 3), (2, 6)).launches
    body = launch.source.split("{", 1)[1]
    [scalar] = re.findall(r"float (v\d+) = b1\[0u\];", body)
    assert body.index(f"float {scalar}") < body.index("if (")
    assert body.count(scalar) == 3  # declared once, used in both outer branches


def test_deep_programs_compile():
    """Neither a long fusion nor a long chain of launches is bounded by
    Python's default recursion limit."""

    def fused(x):
        for _ in range(1000):
            x = jnp.sin(x) * 2.0
        return x

    [launch] = _compile(fused, (8,)).launches
    assert launch.source.count("sin(") == 1000

    def launches(x, w):  # no fusion across products: depth is in the schedule
        for _ in range(1200):
            x = x @ w
        return x

    assert len(_compile(launches, (16, 16), (16, 16)).launches) == 1200


def test_scalar_predicates_select_whole_tensors():
    """StableHLO allows a rank-0 predicate, which applies to every element."""
    program = _compile(lambda p, x, y: lax.select(p > 0, x, y), (), (3, 4), (3, 4))
    [launch] = program.launches
    assert launch.slots == (0, 1, 2, 3) and launch.grid == 12


def test_cbrt_uses_functions_msl_has():
    """MSL has no cbrt."""
    [launch] = _compile(jnp.cbrt, (4,)).launches
    assert "cbrt" not in launch.source


def test_total_order_comparisons_are_rejected():
    """A kernel comparison orders NaNs and signed zeros as MSL does, which is
    not the total order."""
    text = """
    func.func public @main(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xi1> {
      %0 = stablehlo.compare LT, %x, %y, TOTALORDER : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      return %0 : tensor<4xi1>
    }"""
    with pytest.raises(UnsupportedProgram, match="total order"):
        compile_program(stablehlo.serialize_portable_artifact_str(text, "1.0.0"))
    compile_program(
        stablehlo.serialize_portable_artifact_str(text.replace("TOTALORDER", "FLOAT"), "1.0.0")
    )


def test_reshape_chains_stay_small():
    """A reshape names its flat index, so a chain of them grows by a line
    each, not by a factor of the rank each."""

    def windows(x):  # partition into 8x8 windows and back, as windowed attention does
        for _ in range(3):
            x = x.reshape(2, 4, 8, 4, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 64, 3)
            x = x.reshape(2, 4, 4, 8, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(2, 32, 32, 3)
        return x * 2

    [launch] = _compile(windows, (2, 32, 32, 3)).launches
    assert len(launch.source) < 4_000, len(launch.source)
    assert launch.source.count("b0[") == 1


def test_sam_kernels_stay_small():
    """Window partitions chain reshapes and transposes; the SAM-like step's
    kernels stay kilobytes of MSL."""
    from training_steps import workload  # examples/ is on the path (conftest)

    fn, *args = workload("sam")
    exported = jax.export.export(jax.jit(fn), platforms=["metal"])(*args)
    program = compile_program(exported.mlir_module_serialized)
    assert max(len(launch.source) for launch in program.launches) < 100_000


def test_loop_carries_with_their_own_buffer_are_not_copied():
    """A carry computed by a product is written by the product, without a
    launch copying it into the loop's output buffer."""
    program = _compile(lambda w, x: lax.fori_loop(0, 10, lambda i, h: h @ w, x), (64, 64), (64, 64))
    assert [launch.function for launch in program.launches] == ["gemm"] * 10
    # Iterations alternate between two output buffers; none writes its input.
    outputs = [launch.slots[2] for launch in program.launches]
    assert outputs == outputs[:2] * 5 and outputs[0] != outputs[1]
    assert all(launch.slots[0] != launch.slots[2] for launch in program.launches)
    assert program.outputs == [outputs[-1]]


def test_narrow_integers_compute_in_32_bits_and_wrap():
    """8- and 16-bit integers bind as their own MSL types; scatter on them
    is not supported."""
    [launch] = _compile(lambda x: x.astype(jnp.float32) / 255, (4,), dtype=np.uint8).launches
    assert "device const uchar* b0" in launch.source
    [launch] = _compile(lambda x, y: x * y, (4,), (4,), dtype=np.int8).launches
    assert "device char* o0" in launch.source
    with pytest.raises(UnsupportedProgram, match="scatter"):
        _compile(lambda x: x.at[jnp.array([1, 1])].add(1), (4,), dtype=np.int16)


def test_index_terms_fold_constants_and_know_their_names():
    i, k = Term.name("i"), Term.name("k")
    assert (i * 1 + 0) is i and (i - 0) is i and (i // 1) is i
    assert (Term.constant(7) // 2 % 3).value == 0
    assert (Term.constant(2) - 5).value == 2**32 - 3  # unsigned, as in the kernel
    assert (i % 1).value == 0 and (i * 0).value == 0
    term = (i // 6 + k) * 2
    assert term.value is None and term.names == {"i", "k"} and not term.simple
    assert i.simple and Term.constant(3).simple
    assert Term.call("uint(clamp(int(v3), 0, 4))", Term.name("v3")).names == {"v3"}


def test_index_terms_parenthesize_only_where_binding_requires():
    i, k = Term.name("i"), Term.name("k")
    assert (i * 4 + 3).text == "i * 4u + 3u" and str(i * 4 + 3) == "(i * 4u + 3u)"
    assert ((i + 1) * 2).text == "(i + 1u) * 2u"
    assert (5 - (i - k)).text == "5u - (i - k)" and (i - k * 2).text == "i - k * 2u"
    assert (i // 6 % 5 * 2).text == "i / 6u % 5u * 2u"  # left to right, as C parses it
    assert str(i) == "i" and str(Term.constant(3)) == "3u"


@pytest.mark.parametrize("shape", [(3, 4), (2, 1, 5), (1, 1), (7,), ()])
def test_delinear_inverts_linear(shape):
    for flat in range(int(np.prod(shape))):
        coords = delinear(Term.constant(flat), shape)
        assert tuple(c.value for c in coords) == np.unravel_index(flat, shape)
        assert linear(coords, shape).value == flat
    named = delinear(Term.name("i"), shape)
    assert all(c.value == 0 for c, n in zip(named, shape, strict=True) if n == 1)


def test_negated_values_and_negative_literals_never_spell_a_decrement():
    """`-{a}` of `-x` or of `-1.5f` would be `--`, which C reads as a decrement."""

    def fn(x):
        return lax.neg(lax.neg(x)) + lax.neg(lax.full((4,), -1.5)) - lax.neg(lax.full((4,), -3))

    [launch] = _compile(fn, (4,)).launches
    assert "--" not in launch.source


_F32 = np.dtype(np.float32)


def test_product_layout_follows_the_dimension_numbers():
    """Batch dimensions pair in the order given; free ones keep theirs."""
    lhs, rhs = Parameter(_F32, (5, 2, 3, 7)), Parameter(_F32, (7, 4, 2, 3))
    attributes = {
        "lhs_batch": (1,),
        "rhs_batch": (2,),
        "lhs_contract": (3, 2),
        "rhs_contract": (0, 3),
    }
    product = Product.of(Op("dot_general", [lhs, rhs], _F32, (2, 5, 4), attributes))
    assert (product.batch, product.m, product.n, product.k) == (2, 5, 4, 21)
    assert product.left == (1, 0, 3, 2) and product.right == (2, 0, 3, 1)
    with pytest.raises(UnsupportedProgram, match="matching floating operands"):
        Product.of(Op("dot_general", [lhs, Parameter(np.dtype(np.int32), (7,))], _F32, (), {}))


def test_induction_values_wrap_like_the_carry_type():
    values = induction_values(250, 3, np.arange(4), np.dtype(np.uint8))
    assert values.dtype == np.uint8 and values.tolist() == [250, 253, 0, 3]
    assert induction_values(-2, 1, np.array(5), np.dtype(np.int32)).tolist() == 3


def test_rebound_launches_map_slots_and_offset_induction_arrays():
    launch = Launch("source", "name", (0, 2, 3), 4, offsets=(0, 8, 0))
    replayed = rebound(launch, {0: 10, 2: 12, 3: 13}, {0: 24})
    assert replayed.slots == (10, 12, 13) and replayed.offsets == (24, 8, 0)
    assert (replayed.source, replayed.function, replayed.grid) == ("source", "name", 4)


def test_scheduler_compiles_a_graph_built_by_hand():
    """The scheduler works on nodes alone: no MLIR module behind them."""
    x = Parameter(_F32, (3, 4))
    doubled = Op("add", [x, x], _F32, (3, 4), {})
    flipped = Op("transpose", [doubled], _F32, (4, 3), {"perm": (1, 0)})
    program = Scheduler([x]).compile([doubled, flipped, x])
    [launch] = program.launches
    assert program.outputs == [1, 2, 0] and launch.slots == (0, 1, 2) and launch.grid == 12
    assert launch.source.count("b0[") == 2  # at the thread's index and at the transposed one
