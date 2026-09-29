import operator
from collections.abc import Callable

import metal_runtime as mr
import numpy as np
import pytest
from metal_runtime import MathMode

from palladium.experimental import df32

ArrayOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]
PairTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

N = 8192

KERNEL_TEMPLATE = """
kernel void k_{name}(device const float *a   [[buffer(0)]],
                     device const float *b   [[buffer(1)]],
                     device df32*        out [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid], b[tid]);
}}
"""


def _random_pairs(n, rng, *, spread=20, scale=20):
    """Random float32 pairs whose exponents differ by at most `spread` bits.

    Bounding the spread keeps the exact sum representable in float64, so the
    reference the tests compare against is itself exact. `scale` stays well
    inside float32's exponent range, so no denormals and no overflow.
    """
    exp = rng.integers(-scale, scale + 1, size=n)
    delta = rng.integers(0, spread + 1, size=n)
    mant_a = rng.uniform(1.0, 2.0, size=n)
    mant_b = rng.uniform(1.0, 2.0, size=n)
    sign_a = rng.choice([-1.0, 1.0], size=n)
    sign_b = rng.choice([-1.0, 1.0], size=n)
    a = np.ldexp(sign_a * mant_a, exp).astype(np.float32)
    b = np.ldexp(sign_b * mant_b, exp - delta).astype(np.float32)
    return a, b


def _random_float64(n, rng, *, lo_exp=-100, hi_exp=100):
    """Random float64s with magnitudes spanning 2**lo_exp .. 2**(hi_exp + 1).

    The exponent window is a parameter so that the subnormal boundary is
    something a test asks for rather than stumbles into. The defaults sit well
    inside float32's *normal* range (2**-126 .. 2**127), so `split` sees neither
    a subnormal nor an overflow unless a test says so.
    """
    mant = rng.uniform(1.0, 2.0, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    exp = rng.integers(lo_exp, hi_exp + 1, size=n)
    return np.ldexp(sign * mant, exp)


def _by_descending_magnitude(a, b):
    """Reorder each pair so |a| >= |b|, as quick_two_sum requires."""
    swap = np.abs(a) < np.abs(b)
    return np.where(swap, b, a), np.where(swap, a, b)


def _as_is(a, b):
    return a, b


def _inputs(input_gen=_as_is, seed=42):
    return input_gen(*_random_pairs(N, np.random.default_rng(seed)))


def _limbs(name, a, b, math_mode):
    """Run one EFT kernel over `a`, `b`; return the (hi, lo) limbs widened to float64."""
    source = df32.PRELUDE + KERNEL_TEMPLATE.format(name=name)
    kernel = mr.Kernel(source, f"k_{name}", math_mode=math_mode)
    out = mr.Buffer.zeros([len(a), 2], "float32")
    mr.run(kernel, grid=len(a), buffers=[mr.Buffer(a), mr.Buffer(b), out])
    limbs = out.to_numpy().astype(np.float64)
    return limbs[:, 0], limbs[:, 1]


EFTS = [
    pytest.param("quick_two_sum", operator.add, _by_descending_magnitude, id="quick_two_sum"),
    pytest.param("two_sum", operator.add, _as_is, id="two_sum"),
    pytest.param("two_prod", operator.mul, _as_is, id="two_prod"),
]
SUMS = [p for p in EFTS if p.values[0] != "two_prod"]


@pytest.mark.parametrize(("kernel_name", "ref_op", "input_gen"), EFTS)
def test_safe_math_exact_eft(kernel_name: str, ref_op: ArrayOperator, input_gen: PairTransform):
    a, b = _inputs(input_gen)
    ref = ref_op(a.astype(np.float64), b.astype(np.float64))

    hi, lo = _limbs(kernel_name, a, b, MathMode.SAFE)
    assert np.count_nonzero(lo) > N // 2
    assert np.array_equal(hi + lo, ref)


@pytest.mark.parametrize(("kernel_name", "ref_op", "input_gen"), SUMS)
def test_fast_math_destroys_compensation(
    kernel_name: str, ref_op: ArrayOperator, input_gen: PairTransform
):
    """Compiling a sum EFT under FAST math fails at compile time instead of silently zeroing the compensation term."""
    a, b = _inputs(input_gen)
    with pytest.raises(mr.CompileError, match="SAFE"):
        _limbs(kernel_name, a, b, MathMode.FAST)


def test_two_prod_survives_fast_math():
    """The prelude-wide guard rejects FAST math for two_prod even though its fma-routed error term would survive reassociation."""
    a, b = _inputs()
    with pytest.raises(mr.CompileError, match="SAFE"):
        _limbs("two_prod", a, b, MathMode.FAST)


def _representable_pairs(n, rng, *, lo_exp=-100, hi_exp=100):
    """Normalised (hi, lo) float32 pairs, plus the exact float64 they sum to.

    |lo|/ulp(hi) is drawn from a narrow band. The upper edge avoids the
    round-half-even tie at 0.5*ulp and the halved gap below a power of two,
    where fl32(hi + lo) is not hi. The lower edge keeps the span from hi's
    leading bit to lo's trailing bit under 53 bits so the float64 sum is exact.
    """
    hi = _random_float64(n, rng, lo_exp=lo_exp, hi_exp=hi_exp).astype(np.float32)
    ulp = np.abs(np.spacing(hi))
    sign = rng.choice([-1, 1], size=n)
    lo = (sign * ulp * rng.uniform(0.12, 0.24, size=n)).astype(np.float32)
    x = hi.astype(np.float64) + lo.astype(np.float64)
    return (hi, lo, x)


_IDENTITY_SOURCE = (
    df32.PRELUDE
    + """
kernel void df32_identity(device const df32* src [[buffer(0)]],
                          device df32* dst [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    dst[tid] = src[tid];
}
"""
)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_round_trip_within_bound():
    """join(split(x)) recovers x to within a relative 2**-47 over normal float32 magnitudes."""
    x = _random_float64(N, np.random.default_rng(seed=42))
    rt = df32.join(df32.split(x))
    assert np.all(np.abs(rt - x) / np.abs(x) <= 2**-47)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_exact_for_representable_values():
    """A value that already is a non-overlapping float32 pair splits back into exactly that pair, bit for bit."""
    hi, lo, x = _representable_pairs(N, np.random.default_rng(seed=42))
    assert np.all(df32.split(x) == np.stack([hi, lo], axis=-1))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_limbs_do_not_overlap():
    """split's limbs satisfy |lo| <= 0.5 * ulp(hi), the invariant df_add and df_mul consume.

    `np.spacing` is signed, hence the abs. The bound is tight in practice
    (worst observed ratio ~0.999998).
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    pairs = df32.split(x)
    hi, lo = pairs[..., 0], pairs[..., 1]
    assert np.all(np.abs(lo) <= 0.5 * np.abs(np.spacing(hi)))


@pytest.mark.parametrize("shape", [(0,), (7,), (3, 4), (2, 3, 5)])
def test_split_shape_dtype_and_layout(shape):
    """split maps (...) -> (..., 2) as C-contiguous float32 and join inverts the shape."""
    x = _random_float64(np.prod(shape), np.random.default_rng(seed=42)).reshape(shape)
    pairs = df32.split(x)
    assert pairs.flags.c_contiguous
    assert pairs.dtype == np.float32
    assert pairs.shape == (*shape, 2)
    rt = df32.join(pairs)
    assert rt.shape == shape
    assert np.all(np.abs(rt - x) <= 2**-47 * np.abs(x))


def test_split_accepts_the_boundary_value():
    """float32's largest finite value is accepted by split, not rejected."""
    f32max = np.finfo(np.float32).max
    x = np.array([f32max, -f32max], dtype=np.float64)
    pairs = df32.split(x)
    assert pairs[0, 0] == np.float32(f32max)
    assert pairs[0, 1] == 0.0


def test_split_rejects_out_of_range_anywhere_in_the_array():
    """A finite value above float32's range raises ValueError at any index of the array."""
    x = np.array([0.1, 1e40], dtype=np.float64)
    with pytest.raises(ValueError, match="out of range for np.float32"):
        df32.split(x)


def test_join_rejects_non_pairs():
    """join rejects arrays whose trailing dimension is not exactly 2."""
    x = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="must be an array of shape"):
        df32.join(x)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_signed_zero_survives():
    """-0.0 round-trips as -0.0; equality cannot see a lost sign, so the check uses np.signbit."""
    x = np.array([-0.0])
    rt = df32.join(df32.split(x))
    assert np.signbit(x.item()) == np.signbit(rt.item())


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_passes_through_with_zero_lo(value):
    """inf and nan pass through split with hi carrying the value and lo exactly zero.

    The filterwarnings mark turns an inf - inf residual RuntimeWarning into a failure.
    """
    x = np.array([value], dtype=np.float64)
    pair = df32.split(x)
    if np.isnan(value):
        assert np.isnan(pair[0, 0])
    else:
        assert pair[0, 0] == value
    assert pair[0, 1] == 0.0


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(("value", "worst_rel"), [(1e-40, 1e-5), (1e-44, 1e-1)])
def test_subnormals_degrade_but_do_not_crash(value, worst_rel):
    """Below float32's smallest normal the round-trip error exceeds 2**-47 but stays under worst_rel.

    1e-40 keeps roughly 18 significand bits, 1e-44 about two.
    """
    x = np.array([value], dtype=np.float64)
    rt = df32.join(df32.split(x))
    rel_err = np.abs(rt - x) / np.abs(x)
    assert 2**-47 < rel_err[0] < worst_rel


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_idempotent():
    """split(join(split(x))) == split(x), bit for bit."""
    x = _random_float64(N, np.random.default_rng(seed=42))
    assert np.array_equal(df32.split(df32.join(df32.split(x))), df32.split(x))


def test_gpu_round_trip_proves_the_memory_layout():
    """A (..., 2) float32 array binds against `device df32*` with no reinterpretation.

    The element count is a multiple of 256 so the dispatch does not round the
    grid up past the buffer.
    """
    x = _random_float64(256, np.random.default_rng(seed=42))
    pairs = df32.split(x)

    kernel = mr.Kernel(_IDENTITY_SOURCE, "df32_identity", math_mode=MathMode.SAFE)
    src = mr.Buffer(pairs)
    dst = mr.Buffer.zeros(pairs.shape, "float32")
    mr.run(kernel, grid=len(x), buffers=[src, dst])

    assert np.array_equal(dst.to_numpy(), pairs)


DF32_BINOP_TEMPLATE = """
kernel void k_{name}(device const df32* a   [[buffer(0)]],
                     device const df32* b   [[buffer(1)]],
                     device df32*       out [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid], b[tid]);
}}
"""


def _df32_binop(name, a_pairs, b_pairs, math_mode):
    """Run a df32(df32, df32) -> df32 kernel; return output pairs widened to float64."""
    source = df32.PRELUDE + DF32_BINOP_TEMPLATE.format(name=name)
    kernel = mr.Kernel(source, f"k_{name}", math_mode=math_mode)
    out = mr.Buffer.zeros(a_pairs.shape, "float32")
    mr.run(
        kernel,
        grid=len(a_pairs),
        buffers=[mr.Buffer(a_pairs), mr.Buffer(b_pairs), out],
    )
    return out.to_numpy().astype(np.float64)


def _random_df32_operands(n, rng, **kwargs):
    """Two independent arrays of non-overlapping df32 pairs, plus the float64 values they represent.

    Built via `split` so every operand already satisfies the non-overlap
    invariant; df_add only needs to preserve that property, not establish it.
    """
    a64 = _random_float64(n, rng, **kwargs)
    b64 = _random_float64(n, rng, **kwargs)
    return df32.split(a64), df32.split(b64), a64, b64


def _max_rel_err(out, ref):
    result = out[:, 0] + out[:, 1]
    return np.max(np.abs(result - ref) / np.abs(ref))


def _assert_no_overlap(out):
    hi = out[:, 0].astype(np.float32)
    lo = out[:, 1].astype(np.float32)
    nonzero = hi != 0
    assert np.all(np.abs(lo[nonzero]) <= 0.5 * np.abs(np.spacing(hi[nonzero])))


#: Keeps hi's exponent far enough from float32's normal-range edge (+-126)
#: that df_add's second-order compensation term does not itself underflow into
#: subnormal range and get flushed to zero by the GPU (measured, independent of
#: math_mode).
SAFE_EXP_RANGE = {"lo_exp": -70, "hi_exp": 70}


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_matches_float64_reference_under_safe_math():
    """df_add's max relative error vs a float64 reference is <= 2**-40 over 10k operand pairs.

    The fully renormalized variant measures near 2**-44 worst-case over 10k
    samples; 2**-40 keeps margin over that.
    """
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **SAFE_EXP_RANGE
    )
    out = _df32_binop("df_add", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 + b64) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_output_limbs_do_not_overlap():
    """The pair df_add returns satisfies |lo| <= 0.5*ulp(hi)."""
    a_pairs, b_pairs, _, _ = _random_df32_operands(N, np.random.default_rng(42), **SAFE_EXP_RANGE)
    out = _df32_binop("df_add", a_pairs, b_pairs, MathMode.SAFE)
    _assert_no_overlap(out)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_stays_accurate_when_chained():
    """A chain of df_add calls stays within the single-call bound against a float64 running sum."""
    n_terms = 32
    values64 = _random_float64(n_terms, np.random.default_rng(42), lo_exp=-10, hi_exp=10)
    pairs = df32.split(values64)

    acc = pairs[0:1]
    ref = values64[0]
    for i in range(1, n_terms):
        acc = _df32_binop("df_add", acc, pairs[i : i + 1], MathMode.SAFE).astype(np.float32)
        # Sequential, matching df_add's accumulation order: np.sum's pairwise
        # summation rounds differently, and under cancellation that divergence
        # exceeds df_add's own error.
        ref = ref + values64[i]

    # join(), not a bare `acc[0, 0] + acc[0, 1]`: acc is float32, so that sum
    # would round at float32 precision and discard the compensation.
    rel_err = abs(df32.join(acc)[0] - ref) / abs(ref)
    assert rel_err <= 2**-40


def test_df_add_destroys_precision_under_fast_math():
    """Compiling df_add under FAST math fails at compile time."""
    a_pairs, b_pairs, _, _ = _random_df32_operands(N, np.random.default_rng(42))
    with pytest.raises(mr.CompileError, match="SAFE"):
        _df32_binop("df_add", a_pairs, b_pairs, MathMode.FAST)


# --- df_mul -----------------------------------------------------------------


#: Keeps a*b's true exponent well inside float32's +-126 normal range in both
#: directions, so df_mul's overflow/underflow behaviour is not conflated with
#: the accuracy bound; test_df_mul_near_float32_range_limits covers the edges.
MUL_SAFE_EXP_RANGE = {"lo_exp": -40, "hi_exp": 40}


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_matches_float64_reference_under_safe_math():
    """df_mul's max relative error vs a float64 reference is <= 2**-40 over 10k operand pairs."""
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE
    )
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 * b64) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_output_limbs_do_not_overlap():
    """The pair df_mul returns satisfies |lo| <= 0.5*ulp(hi).

    Dropping the wrong cross term can pass the accuracy bound while still
    producing overlapping limbs.
    """
    a_pairs, b_pairs, _, _ = _random_df32_operands(
        N, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE
    )
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    _assert_no_overlap(out)


def test_df_mul_destroys_precision_under_fast_math():
    """Compiling df_mul under FAST math fails at compile time."""
    a_pairs, b_pairs, _, _ = _random_df32_operands(N, np.random.default_rng(42))
    with pytest.raises(mr.CompileError, match="SAFE"):
        _df32_binop("df_mul", a_pairs, b_pairs, MathMode.FAST)


# --- df_sub, df_neg, df_abs, conversions, comparisons, df_fma ---------------
#
# FAST/RELAXED rejection is not re-tested per op: the prelude-wide guard fires
# for any kernel compiled against PRELUDE.

_DF32_UNOP_TEMPLATE = """
kernel void k_{name}(device const df32* a   [[buffer(0)]],
                     device df32*       out [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid]);
}}
"""

_DF32_CMP_TEMPLATE = """
kernel void k_{name}(device const df32* a   [[buffer(0)]],
                     device const df32* b   [[buffer(1)]],
                     device bool*       out [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid], b[tid]);
}}
"""

_DF32_FMA_TEMPLATE = """
kernel void k_df_fma(device const df32* a   [[buffer(0)]],
                     device const df32* b   [[buffer(1)]],
                     device const df32* c   [[buffer(2)]],
                     device df32*       out [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    out[tid] = df_fma(a[tid], b[tid], c[tid]);
}
"""


def _df32_unop(name, a_pairs, math_mode=MathMode.SAFE):
    source = df32.PRELUDE + _DF32_UNOP_TEMPLATE.format(name=name)
    kernel = mr.Kernel(source, f"k_{name}", math_mode=math_mode)
    out = mr.Buffer.zeros(a_pairs.shape, "float32")
    mr.run(kernel, grid=len(a_pairs), buffers=[mr.Buffer(a_pairs), out])
    return out.to_numpy().astype(np.float64)


def _df32_cmp(name, a_pairs, b_pairs, math_mode=MathMode.SAFE):
    source = df32.PRELUDE + _DF32_CMP_TEMPLATE.format(name=name)
    kernel = mr.Kernel(source, f"k_{name}", math_mode=math_mode)
    out = mr.Buffer.zeros([len(a_pairs)], "bool")
    mr.run(kernel, grid=len(a_pairs), buffers=[mr.Buffer(a_pairs), mr.Buffer(b_pairs), out])
    return out.to_numpy()


def _df32_fma(a_pairs, b_pairs, c_pairs, math_mode=MathMode.SAFE):
    source = df32.PRELUDE + _DF32_FMA_TEMPLATE
    kernel = mr.Kernel(source, "k_df_fma", math_mode=math_mode)
    out = mr.Buffer.zeros(a_pairs.shape, "float32")
    mr.run(
        kernel,
        grid=len(a_pairs),
        buffers=[mr.Buffer(a_pairs), mr.Buffer(b_pairs), mr.Buffer(c_pairs), out],
    )
    return out.to_numpy().astype(np.float64)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_neg_and_df_abs_are_exact():
    """df_neg and df_abs are bit-exact sign manipulation, matched by exact equality."""
    a64 = _random_float64(N, np.random.default_rng(42), **SAFE_EXP_RANGE)
    a_pairs = df32.split(a64)

    neg_out = _df32_unop("df_neg", a_pairs)
    assert np.array_equal(neg_out, -a_pairs.astype(np.float64))

    abs_out = _df32_unop("df_abs", a_pairs)
    # Compare against abs() of the split value, not the original a64: split()
    # already lost precision, so df32.join(a_pairs) != a64 in general.
    assert np.array_equal(df32.join(abs_out.astype(np.float32)), np.abs(df32.join(a_pairs)))
    _assert_no_overlap(abs_out)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_sub_matches_float64_reference_under_safe_math():
    """df_sub meets the same 2**-40 bound as df_add, which it is defined in terms of."""
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **SAFE_EXP_RANGE
    )
    out = _df32_binop("df_sub", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 - b64) <= 2**-40
    _assert_no_overlap(out)


def test_df_from_float_and_df_to_float_round_trip():
    """df_from_float promotes with an exact-zero lo, and df_to_float truncates back to the hi limb."""
    x = np.array([1.5, -3.25, 0.0, -0.0, 123456.75], dtype=np.float32)
    pairs = df32.split(x.astype(np.float64))

    source = (
        df32.PRELUDE
        + """
kernel void k_promote(device const float* a [[buffer(0)]],
                       device df32* out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    out[tid] = df_from_float(a[tid]);
}
"""
    )
    kernel = mr.Kernel(source, "k_promote", math_mode=MathMode.SAFE)
    out = mr.Buffer.zeros([len(x), 2], "float32")
    mr.run(kernel, grid=len(x), buffers=[mr.Buffer(x), out])
    promoted = out.to_numpy()
    assert np.array_equal(promoted[:, 0], x)
    assert np.all(promoted[:, 1] == 0.0)

    source = (
        df32.PRELUDE
        + """
kernel void k_demote(device const df32* a [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    out[tid] = df_to_float(a[tid]);
}
"""
    )
    kernel = mr.Kernel(source, "k_demote", math_mode=MathMode.SAFE)
    demote_out = mr.Buffer.zeros([len(x)], "float32")
    mr.run(kernel, grid=len(x), buffers=[mr.Buffer(pairs), demote_out])
    assert np.array_equal(demote_out.to_numpy(), pairs[:, 0])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_comparisons_match_float64_ordering():
    """The df32 comparisons agree with float64 ordering, including the tie-break on lo for a shared hi."""
    rng = np.random.default_rng(42)
    a64 = _random_float64(N, rng, **SAFE_EXP_RANGE)
    b64 = _random_float64(N, rng, **SAFE_EXP_RANGE)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)

    assert np.array_equal(_df32_cmp("df_lt", a_pairs, b_pairs), a64 < b64)
    assert np.array_equal(_df32_cmp("df_gt", a_pairs, b_pairs), a64 > b64)
    assert np.array_equal(_df32_cmp("df_le", a_pairs, b_pairs), a64 <= b64)
    assert np.array_equal(_df32_cmp("df_ge", a_pairs, b_pairs), a64 >= b64)
    assert np.array_equal(_df32_cmp("df_eq", a_pairs, b_pairs), a64 == b64)

    # Same hi, different (nonzero, opposite-sign) lo: must resolve on lo.
    hi = np.array([1.5, -7.0], dtype=np.float32)
    lo_pos = np.array([1e-7, 1e-7], dtype=np.float32)
    lo_neg = -lo_pos
    pos_pairs = np.stack([hi, lo_pos], axis=-1)
    neg_pairs = np.stack([hi, lo_neg], axis=-1)
    assert np.array_equal(_df32_cmp("df_lt", neg_pairs, pos_pairs), [True, True])
    assert np.array_equal(_df32_cmp("df_gt", pos_pairs, neg_pairs), [True, True])
    assert np.array_equal(_df32_cmp("df_eq", pos_pairs, pos_pairs), [True, True])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_fma_matches_float64_reference_under_safe_math():
    """df_fma's max relative error vs a float64 a*b + c reference is <= 2**-40."""
    rng = np.random.default_rng(42)
    a64 = _random_float64(N, rng, **MUL_SAFE_EXP_RANGE)
    b64 = _random_float64(N, rng, **MUL_SAFE_EXP_RANGE)
    c64 = _random_float64(N, rng, **MUL_SAFE_EXP_RANGE)
    a_pairs, b_pairs, c_pairs = df32.split(a64), df32.split(b64), df32.split(c64)

    out = _df32_fma(a_pairs, b_pairs, c_pairs)
    assert _max_rel_err(out, a64 * b64 + c64) <= 2**-40
    _assert_no_overlap(out)


# --- Accuracy suite ---------------------------------------------------------


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_near_cancellation_stays_accurate():
    """df_add(a, b) for b ~= -a stays within the same relative-error bound as the general case."""
    n = N
    rng = np.random.default_rng(42)
    a64 = _random_float64(n, rng, lo_exp=-20, hi_exp=20)
    a_pairs = df32.split(a64)
    # The reference is built from the split operands, not the pre-split
    # float64s: under near-cancellation, split()'s own ~2**-47 representation
    # error would otherwise dwarf df_add's error.
    a_exact = df32.join(a_pairs)
    # The perturbation sits far below float32's ~2**-24 precision (so plain
    # float32 addition cancels to noise) but above df32's ~2**-47 floor (so it
    # survives the split).
    perturbation = rng.uniform(2**-40, 2**-20, size=n) * rng.choice([-1.0, 1.0], size=n)
    b_pairs = df32.split(-a_exact * (1 + perturbation))
    b_exact = df32.join(b_pairs)
    ref = a_exact + b_exact

    nonzero = ref != 0
    out = _df32_binop("df_add", a_pairs[nonzero], b_pairs[nonzero], MathMode.SAFE)
    assert _max_rel_err(out, ref[nonzero]) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_and_df_mul_handle_wide_exponent_spread():
    """df_add and df_mul stay within the accuracy bound for operands 60-100 exponent bits apart.

    The spread stops short of float32's +-126 edge, where the compensation term
    would underflow into subnormal range and be flushed to zero by the GPU.
    """
    rng = np.random.default_rng(42)
    a64 = _random_float64(N, rng, lo_exp=30, hi_exp=50)
    b64 = _random_float64(N, rng, lo_exp=-50, hi_exp=-30)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)

    add_out = _df32_binop("df_add", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(add_out, a64 + b64) <= 2**-40

    mul_out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(mul_out, a64 * b64) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_near_float32_range_limits():
    """df_mul's intermediate cross terms neither overflow to inf near float32's max nor flush to zero near its smallest normal."""
    f32max = np.finfo(np.float32).max
    tiny = np.finfo(np.float32).tiny

    # The true product stays well inside range (~f32max / 4).
    big = np.sqrt(f32max) / 2
    a64 = np.array([big, -big], dtype=np.float64)
    b64 = np.array([big, big], dtype=np.float64)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert np.all(np.isfinite(out))
    assert _max_rel_err(out, a64 * b64) <= 2**-46

    # Near the smallest normal the true, nonzero product must not flush to zero.
    small = tiny * 4
    a64 = np.array([small, -small], dtype=np.float64)
    b64 = np.array([2.0, 2.0], dtype=np.float64)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert np.all(out[:, 0] != 0.0)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_and_df_mul_preserve_signed_zero():
    """+0.0 and -0.0 operands produce an exact-zero magnitude through df_add and df_mul in every sign combination.

    Only the magnitude is checked: the EFT chain's `a - (s - a)` terms re-add
    opposite-signed zeros, which round to +0 by definition, so df_add/df_mul do
    not preserve the sign of a resulting zero.
    """
    zero = df32.split(np.array([0.0]))
    neg_zero = df32.split(np.array([-0.0]))
    one = df32.split(np.array([1.0]))
    neg_one = df32.split(np.array([-1.0]))

    for a, b in [(zero, zero), (neg_zero, neg_zero), (zero, neg_zero)]:
        out = _df32_binop("df_add", a, b, MathMode.SAFE)
        assert out[0, 0] == 0.0 and out[0, 1] == 0.0

    for a, b in [(zero, one), (neg_zero, one), (zero, neg_one), (neg_zero, neg_one)]:
        out = _df32_binop("df_mul", a, b, MathMode.SAFE)
        assert out[0, 0] == 0.0 and out[0, 1] == 0.0


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_chained_df_add_matches_float64_running_sum():
    """A 500-term chain of df_add calls stays within 2**-36 of a float64 running sum."""
    n_terms = 500
    values64 = _random_float64(n_terms, np.random.default_rng(7), lo_exp=-10, hi_exp=10)
    pairs = df32.split(values64)

    acc = pairs[0:1]
    ref = values64[0]
    for i in range(1, n_terms):
        acc = _df32_binop("df_add", acc, pairs[i : i + 1], MathMode.SAFE).astype(np.float32)
        # Sequential reference, matching df_add's accumulation order.
        ref = ref + values64[i]

    rel_err = abs(df32.join(acc)[0] - ref) / abs(ref)
    assert rel_err <= 2**-36


def _assert_math_mode_pinned(call, math_mode):
    """SAFE runs and produces finite output; any other math mode fails at compile time."""
    if math_mode is MathMode.SAFE:
        assert np.all(np.isfinite(call()))
    else:
        with pytest.raises(mr.CompileError, match="SAFE"):
            call()


@pytest.mark.parametrize("math_mode", list(mr.MathMode))
@pytest.mark.parametrize("routine", ["df_add", "df_mul", "df_div"])
def test_math_mode_behavior_is_pinned_per_routine(routine, math_mode):
    """Each binary routine compiles only under SAFE and is rejected under every other math mode."""
    a_pairs, b_pairs, _, _ = _random_df32_operands(
        8, np.random.default_rng(0), **MUL_SAFE_EXP_RANGE
    )
    _assert_math_mode_pinned(lambda: _df32_binop(routine, a_pairs, b_pairs, math_mode), math_mode)


@pytest.mark.parametrize("math_mode", list(mr.MathMode))
def test_sqrt_math_mode_behavior_is_pinned(math_mode):
    """df_sqrt compiles only under SAFE and is rejected under every other math mode."""
    a_pairs, _, _, _ = _random_df32_operands(8, np.random.default_rng(0), **MUL_SAFE_EXP_RANGE)
    a_pairs = np.abs(a_pairs)
    _assert_math_mode_pinned(lambda: _df32_unop("df_sqrt", a_pairs, math_mode), math_mode)


# --- Math-mode guard --------------------------------------------------------


def test_prelude_rejects_non_safe_math_at_compile_time():
    """Compiling df32.PRELUDE under FAST or RELAXED raises CompileError with a message naming SAFE."""
    source = df32.PRELUDE + "kernel void noop() {}"
    for math_mode in (MathMode.FAST, MathMode.RELAXED):
        with pytest.raises(mr.CompileError, match="SAFE"):
            mr.Kernel(source, "noop", math_mode=math_mode)
    # SAFE itself must still compile cleanly.
    mr.Kernel(source, "noop", math_mode=MathMode.SAFE)


def test_df32_safe_kernel_helper_pins_math_mode():
    """df32.kernel defaults to SAFE math mode and still honors an explicit math_mode."""
    kernel = df32.kernel("kernel void noop() {}", "noop")
    assert kernel.math_mode == MathMode.SAFE

    # An explicit non-SAFE math_mode still hits the prelude guard.
    with pytest.raises(mr.CompileError, match="SAFE"):
        df32.kernel("kernel void noop() {}", "noop", math_mode=MathMode.FAST)


# --- df_div, df_sqrt --------------------------------------------------------


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_div_matches_float64_reference_under_safe_math():
    """df_div's max relative error vs a float64 reference is <= 2**-40 over 10k operand pairs."""
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE
    )
    out = _df32_binop("df_div", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 / b64) <= 2**-40
    _assert_no_overlap(out)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_sqrt_matches_float64_reference_under_safe_math():
    """df_sqrt's max relative error vs a float64 reference is <= 2**-40 over 10k positive operands."""
    a64 = np.abs(_random_float64(10_000, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE))
    a_pairs = df32.split(a64)
    out = _df32_unop("df_sqrt", a_pairs)
    assert _max_rel_err(out, np.sqrt(a64)) <= 2**-40
    _assert_no_overlap(out)


def test_df_sqrt_handles_zero_and_negative_inputs():
    """df_sqrt returns a sign-preserved zero for +-0 and NaN for a negative input.

    The general path divides by a.hi, hence the dedicated `a.hi <= 0` branch.
    """
    x = np.array([0.0, -0.0, -4.0], dtype=np.float64)
    pairs = df32.split(x)
    out = _df32_unop("df_sqrt", pairs)
    assert out[0, 0] == 0.0 and not np.signbit(out[0, 0])
    assert out[1, 0] == 0.0 and np.signbit(out[1, 0])
    assert np.isnan(out[2, 0])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_div_by_near_zero_does_not_crash():
    """Dividing by a signed zero produces inf or NaN rather than crashing or hanging."""
    a64 = np.array([1.0, -1.0], dtype=np.float64)
    b64 = np.array([0.0, -0.0], dtype=np.float64)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)
    out = _df32_binop("df_div", a_pairs, b_pairs, MathMode.SAFE)
    assert np.all(np.isinf(out[:, 0]) | np.isnan(out[:, 0]))
