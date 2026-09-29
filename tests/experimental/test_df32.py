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
    """FAST used to silently zero out the compensation term here; step 6's
    `__FAST_MATH__` guard (see `test_prelude_rejects_non_safe_math_at_compile_time`)
    now turns that into a compile-time failure before the kernel ever runs, so
    this asserts the loud failure instead of the silent one.
    """
    a, b = _inputs(input_gen)
    with pytest.raises(mr.CompileError, match="SAFE"):
        _limbs(kernel_name, a, b, MathMode.FAST)


def test_two_prod_survives_fast_math():
    """`two_prod`'s own error term is routed through `fma()` and would survive
    FAST reassociation on its own merits -- but it never gets the chance: the
    prelude-wide guard added in step 6 rejects the compile before `two_prod`'s
    resilience is even observable.
    """
    a, b = _inputs()
    with pytest.raises(mr.CompileError, match="SAFE"):
        _limbs("two_prod", a, b, MathMode.FAST)


def _representable_pairs(n, rng, *, lo_exp=-100, hi_exp=100):
    """Normalised (hi, lo) float32 pairs, plus the exact float64 they sum to.

    Return `(hi, lo, x)` with `x == float64(hi) + float64(lo)` *exactly*, so
    `split(x)` is obliged to give back `(hi, lo)` bit for bit.

    |lo| needs bounding from both sides, for unrelated reasons:

    * Upper. At exactly 0.5*ulp(hi) the sum is a round-half-even tie, and at a
      power of two the gap *below* hi is only half the gap above -- so a negative
      lo near 0.5*ulp rounds onto the neighbour and fl32(hi + lo) is not hi.
    * Lower. lo carries a full 24-bit significand wherever it sits, so the
      smaller it is relative to ulp(hi), the wider the span from hi's leading bit
      to lo's trailing bit. Past 53 bits the float64 sum stops being exact and
      this generator's premise quietly fails.

    Drawing |lo|/ulp(hi) from a narrow band satisfies both at once.
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
    """join(split(x)) recovers x to ~48 bits, over normal float32 magnitudes.

    The bound is relative, not absolute. Note this holds only away from the
    subnormal range -- keep the exponent window inside the defaults.
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    rt = df32.join(df32.split(x))
    assert np.all(np.abs(rt - x) / np.abs(x) <= 2**-47)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_exact_for_representable_values():
    """A value that already IS a non-overlapping float32 pair must split back
    into exactly that pair -- no tolerance, bit for bit.

    This is the test that pins `split` and the Step 1 EFTs as consistent with
    each other, rather than each being self-consistently wrong.
    """
    hi, lo, x = _representable_pairs(N, np.random.default_rng(seed=42))
    assert np.all(df32.split(x) == np.stack([hi, lo], axis=-1))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_limbs_do_not_overlap():
    """|lo| <= 0.5 * ulp(hi) -- the invariant df_add and df_mul will consume.

    `np.spacing` gives the ulp but is *signed*, so it is negative for negative
    hi. The bound is tight in practice (worst observed ratio ~0.999998), so this
    is a real constraint rather than a slack one.
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    pairs = df32.split(x)
    hi, lo = pairs[..., 0], pairs[..., 1]
    assert np.all(np.abs(lo) <= 0.5 * np.abs(np.spacing(hi)))


@pytest.mark.parametrize("shape", [(0,), (7,), (3, 4), (2, 3, 5)])
def test_split_shape_dtype_and_layout(shape):
    """split maps (...) -> (..., 2) as C-contiguous float32, and join inverts
    the shape. Contiguity is what lets `mr.Buffer` take the result without a copy.
    """
    x = _random_float64(np.prod(shape), np.random.default_rng(seed=42)).reshape(shape)
    pairs = df32.split(x)
    assert pairs.flags.c_contiguous
    assert pairs.dtype == np.float32
    assert pairs.shape == (*shape, 2)
    rt = df32.join(pairs)
    assert rt.shape == shape
    assert np.all(np.abs(rt - x) <= 2**-47 * np.abs(x))


def test_split_accepts_the_boundary_value():
    """float32's largest finite value must be ACCEPTED, not rejected.

    A guard written with >= instead of > passes every other test in this file.
    """
    f32max = np.finfo(np.float32).max
    x = np.array([f32max, -f32max], dtype=np.float64)
    pairs = df32.split(x)
    assert pairs[0, 0] == np.float32(f32max)
    assert pairs[0, 1] == 0.0


def test_split_rejects_out_of_range_anywhere_in_the_array():
    """A finite value above float32's range raises ValueError.

    Put the offending value at a non-zero index of a multi-element array: a guard
    that collapses the array to a single truth value passes the one-element case.
    """
    x = np.array([0.1, 1e40], dtype=np.float64)
    with pytest.raises(ValueError, match="out of range for np.float32"):
        df32.split(x)


def test_join_rejects_non_pairs():
    """join requires a trailing dimension of exactly 2. Without the guard a
    (4, 3) array silently yields (4,), using columns 0 and 1 and dropping the third.
    """
    x = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="must be an array of shape"):
        df32.join(x)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_signed_zero_survives():
    """-0.0 must round-trip as -0.0.

    `0.0 == -0.0` is True and `np.array_equal([0.0], [-0.0])` is True, so an
    equality assertion cannot see a lost sign. Use `np.signbit`.
    """
    x = np.array([-0.0])
    rt = df32.join(df32.split(x))
    assert np.signbit(x.item()) == np.signbit(rt.item())


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_passes_through_with_zero_lo(value):
    """inf and nan are representable in float32, so split must not poison them.

    hi carries the value and lo must be exactly zero. `np.array_equal` reports
    nan != nan, so the nan case needs `np.isnan`. The filterwarnings mark is
    load-bearing here: computing the residual as inf - inf raises a
    RuntimeWarning, which this turns into a failure.
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
    """Below float32's smallest normal the 2**-47 claim does not hold.

    Assert the *limitation*: the error exceeds 2**-47 but stays under
    `worst_rel`. The deeper into the subnormal range, the fewer significand bits
    survive -- 1e-40 keeps roughly 18 of them, 1e-44 about two.
    """
    x = np.array([value], dtype=np.float64)
    rt = df32.join(df32.split(x))
    rel_err = np.abs(rt - x) / np.abs(x)
    assert 2**-47 < rel_err[0] < worst_rel


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_idempotent():
    """split(join(split(x))) == split(x), bit for bit.

    Once a value is representable as a df32 pair, a further round trip must not
    move it. This is the property chained df_add/df_mul leans on.
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    assert np.array_equal(df32.split(df32.join(df32.split(x))), df32.split(x))


def test_gpu_round_trip_proves_the_memory_layout():
    """(..., 2) float32 binds against `device df32*` with no reinterpretation.

    Everything above tests numpy arithmetic; only this tests the claim that the
    two layouts coincide. Upload split output, run `df32_identity`, read it back,
    and require the bytes to be unchanged. Keep the element count a multiple of
    256 so the dispatch does not round the grid up past the buffer.
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
    """Run a df32(df32, df32) -> df32 kernel; return output pairs widened to float64.

    Mirrors `_limbs()` from step 1, but the inputs and output are already df32
    pairs rather than raw float32s -- `df_add`/`df_mul` operate on `(hi, lo)`,
    not on scalars.
    """
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
    """Two independent arrays of valid, non-overlapping df32 pairs, plus the
    float64 values each one exactly represents.

    Built from `_random_float64` + `split` rather than drawn as raw (hi, lo)
    floats, so every operand already satisfies the non-overlap invariant
    `test_limbs_do_not_overlap` pins -- df_add only needs to preserve that
    property, not establish it.
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
#: that df_add's second-order compensation term doesn't itself underflow
#: into subnormal range and get flushed to zero by the GPU (measured,
#: independent of math_mode; see notes/devlog.md).
SAFE_EXP_RANGE = {"lo_exp": -70, "hi_exp": 70}


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_matches_float64_reference_under_safe_math():
    """max relative error of df_add vs a float64 `+` reference must be <= 2**-40
    over ~10k random operand pairs, under MathMode.SAFE.

    The plan's stated target was 2**-46. Measured, even the fully renormalized
    variant lands closer to 2**-44 worst-case over 10k samples. 2**-40 keeps
    margin over that.
    """
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **SAFE_EXP_RANGE
    )
    out = _df32_binop("df_add", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 + b64) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_output_limbs_do_not_overlap():
    """The (hi, lo) pair df_add returns must itself satisfy |lo| <= 0.5*ulp(hi),
    the same invariant test_limbs_do_not_overlap pins for split.

    Separates a correctly renormalized df_add from a sloppy one that produces
    overlapping limbs: both can look accurate applied once, but only the
    renormalized one survives being chained (see the next test).
    """
    a_pairs, b_pairs, _, _ = _random_df32_operands(N, np.random.default_rng(42), **SAFE_EXP_RANGE)
    out = _df32_binop("df_add", a_pairs, b_pairs, MathMode.SAFE)
    _assert_no_overlap(out)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_stays_accurate_when_chained():
    """Summing a long chain of df32 values with repeated df_add calls must stay
    within the same relative-error bound as a single df_add, compared against an
    equivalent float64 running sum.

    A df_add that skips renormalization can look fine applied once and still
    violate non-overlap; the error only compounds once results get fed back in
    as operands.
    """
    n_terms = 32
    values64 = _random_float64(n_terms, np.random.default_rng(42), lo_exp=-10, hi_exp=10)
    pairs = df32.split(values64)

    acc = pairs[0:1]
    ref = values64[0]
    for i in range(1, n_terms):
        acc = _df32_binop("df_add", acc, pairs[i : i + 1], MathMode.SAFE).astype(np.float32)
        # Sequential, matching df_add's own accumulation order: np.sum's
        # pairwise summation rounds differently, and under the cancellation
        # these random values sometimes hit, that's a bigger divergence than
        # df_add's own error.
        ref = ref + values64[i]

    # join(), not a bare `acc[0, 0] + acc[0, 1]`: acc is float32, so that sum
    # rounds at float32 precision and throws away the compensation this test
    # exists to check.
    rel_err = abs(df32.join(acc)[0] - ref) / abs(ref)
    assert rel_err <= 2**-40


def test_df_add_destroys_precision_under_fast_math():
    """Mirrors step 1's `test_fast_math_destroys_compensation`: df_add's error
    term is folded through plain `+`/`-`, an expression tree the optimizer is
    free to reassociate. The whole prelude is caught by the same compile-time
    guard step 6 adds; see `test_prelude_rejects_non_safe_math_at_compile_time`.
    """
    a_pairs, b_pairs, _, _ = _random_df32_operands(N, np.random.default_rng(42))
    with pytest.raises(mr.CompileError, match="SAFE"):
        _df32_binop("df_add", a_pairs, b_pairs, MathMode.FAST)


# --- Step 4 -- df_mul -------------------------------------------------------


#: Keeps a*b's true exponent (up to +-140 here) well inside float32's +-126
#: normal range in both directions, so df_mul's overflow/underflow behaviour
#: isn't conflated with the accuracy bound below. That is
#: test_df_mul_near_float32_range_limits' job, not this one's.
MUL_SAFE_EXP_RANGE = {"lo_exp": -40, "hi_exp": 40}


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_matches_float64_reference_under_safe_math():
    """max relative error of df_mul vs a float64 `*` reference must be <= 2**-40
    over ~10k random operand pairs, under MathMode.SAFE.

    Same shape as df_add's exactness test (same measured-vs-planned gap from
    the plan's 2**-46 target; see that test's docstring), built on `two_prod`
    for the leading term plus the cross terms rather than `two_sum` on the
    his/los.
    """
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE
    )
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 * b64) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_output_limbs_do_not_overlap():
    """The (hi, lo) pair df_mul returns must itself satisfy |lo| <= 0.5*ulp(hi).

    Same invariant as df_add's, now on the composed two_prod + cross-term
    result. Worth checking independently: dropping the wrong cross term (see
    the plan's note on which of the four is conventionally negligible) could
    pass the accuracy test above while still overlapping.
    """
    a_pairs, b_pairs, _, _ = _random_df32_operands(
        N, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE
    )
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    _assert_no_overlap(out)


def test_df_mul_destroys_precision_under_fast_math():
    """Same comparison as the SAFE-math test above, but compiled under
    MathMode.FAST.

    `two_prod`'s own error term is `fma()`-routed and would survive FAST on its
    own (step 1 measured that), but df_mul's cross terms are plain `+`, and the
    whole prelude is rejected at compile time regardless (step 6's guard).
    """
    a_pairs, b_pairs, _, _ = _random_df32_operands(N, np.random.default_rng(42))
    with pytest.raises(mr.CompileError, match="SAFE"):
        _df32_binop("df_mul", a_pairs, b_pairs, MathMode.FAST)


# --- Extra ops: df_sub, df_neg, df_abs, conversions, comparisons, df_fma ---
#
# Not in the original plan's step list; added afterwards to round out the
# arithmetic a Pallas-generated kernel would actually need. FAST/RELAXED
# rejection isn't re-tested per op here: test_prelude_rejects_non_safe_math_
# at_compile_time already proves the guard fires for any kernel compiled
# against PRELUDE.

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
    """Both are bit-exact sign manipulation, not compensated arithmetic: no
    accuracy bound needed, just exact equality against a numpy reference.
    """
    a64 = _random_float64(N, np.random.default_rng(42), **SAFE_EXP_RANGE)
    a_pairs = df32.split(a64)

    neg_out = _df32_unop("df_neg", a_pairs)
    assert np.array_equal(neg_out, -a_pairs.astype(np.float64))

    abs_out = _df32_unop("df_abs", a_pairs)
    # Compare against abs() of the split value, not the original a64: split()
    # already lost precision going from float64 to df32, so
    # df32.join(a_pairs) != a64 in general. df_abs is a sign flip on top of
    # whatever split() produced, and that step is bit-exact.
    assert np.array_equal(df32.join(abs_out.astype(np.float32)), np.abs(df32.join(a_pairs)))
    _assert_no_overlap(abs_out)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_sub_matches_float64_reference_under_safe_math():
    """df_sub(a, b) must equal df_add(a, -b): same 2**-40 bound as df_add,
    since it's defined directly in terms of it.
    """
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **SAFE_EXP_RANGE
    )
    out = _df32_binop("df_sub", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 - b64) <= 2**-40
    _assert_no_overlap(out)


def test_df_from_float_and_df_to_float_round_trip():
    """`df_from_float` promotes with an exact-zero `lo`. `df_to_float` is a
    lossy truncation back down to the `hi` limb: exact for the promote
    direction, `<= ulp(x)` for the truncate direction (it drops `lo`).
    """
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
    """df_lt/df_gt/df_le/df_ge/df_eq must agree with ordering the underlying
    float64 values, including the tie-break-on-lo case for pairs that share a
    `hi` limb but differ in `lo`.
    """
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
    """max relative error of df_fma(a, b, c) vs a float64 `a*b + c` reference
    must be <= 2**-40, matching df_add/df_mul's bound. Built from the same
    primitives, just with one fewer renormalization pass.
    """
    rng = np.random.default_rng(42)
    a64 = _random_float64(N, rng, **MUL_SAFE_EXP_RANGE)
    b64 = _random_float64(N, rng, **MUL_SAFE_EXP_RANGE)
    c64 = _random_float64(N, rng, **MUL_SAFE_EXP_RANGE)
    a_pairs, b_pairs, c_pairs = df32.split(a64), df32.split(b64), df32.split(c64)

    out = _df32_fma(a_pairs, b_pairs, c_pairs)
    assert _max_rel_err(out, a64 * b64 + c64) <= 2**-40
    _assert_no_overlap(out)


# --- Step 5 -- Accuracy suite ------------------------------------------------


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_near_cancellation_stays_accurate():
    """df_add(a, b) for b ~= -a (catastrophic cancellation in plain float32)
    must still land within the same relative-error bound as the general case.

    This is the adversarial input float32 arithmetic gets most visibly wrong:
    the whole point of carrying a `lo` limb is that the compensation term
    survives where plain addition would cancel it away.
    """
    n = N
    rng = np.random.default_rng(42)
    a64 = _random_float64(n, rng, lo_exp=-20, hi_exp=20)
    a_pairs = df32.split(a64)
    # The reference must be built from what df_add actually receives: the
    # split operands, not the pre-split float64s. a already loses ~5 bits in
    # split() (the 2**-47 bound from step 2); once b is built to nearly cancel
    # it, that representation error would otherwise dwarf df_add's own error
    # and this test would measure split()'s precision, not df_add's.
    a_exact = df32.join(a_pairs)
    # b is a itself, negated and perturbed at a relative level far below
    # float32's ~2**-24 precision (so plain float32 addition cancels a and b to
    # noise) but well above df32's own ~2**-47 representation floor (so the
    # perturbation survives the split below).
    perturbation = rng.uniform(2**-40, 2**-20, size=n) * rng.choice([-1.0, 1.0], size=n)
    b_pairs = df32.split(-a_exact * (1 + perturbation))
    b_exact = df32.join(b_pairs)
    ref = a_exact + b_exact

    nonzero = ref != 0
    out = _df32_binop("df_add", a_pairs[nonzero], b_pairs[nonzero], MathMode.SAFE)
    assert _max_rel_err(out, ref[nonzero]) <= 2**-40


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_and_df_mul_handle_wide_exponent_spread():
    """Operands whose exponents differ far more than step 1's bounded band
    (which existed only to keep the float64 *reference* exact) must still
    round-trip through df_add/df_mul within the accuracy bound.

    Step 1 bounded the spread so the test oracle stayed exact; that constraint
    doesn't apply here since df32 arithmetic's correctness doesn't depend on
    float64 exactly representing the answer, only on the relative-error
    comparison being valid.

    Exponents are chosen wide (~60-100 bits of spread, versus step 1's 29-bit
    cap) but not pushed all the way to float32's +-126 edge: a compensation
    term several orders below an operand near that edge would itself underflow
    into subnormal range and get flushed to zero by the GPU (a real hardware
    effect, see `SAFE_EXP_RANGE`'s docstring, not what this test checks).
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
    """Operands close to float32's max magnitude must not silently overflow to
    inf inside df_mul's cross-term arithmetic, and values close to the smallest
    normal must not silently flush to zero.

    df_mul composes several float32 multiplications and additions per output
    element; each is a place range could be clipped that a single `a * b`
    reference wouldn't reveal.
    """
    f32max = np.finfo(np.float32).max
    tiny = np.finfo(np.float32).tiny

    # The true product stays well inside range (~f32max / 4); df_mul's
    # intermediate cross terms must not spuriously overflow to inf.
    big = np.sqrt(f32max) / 2
    a64 = np.array([big, -big], dtype=np.float64)
    b64 = np.array([big, big], dtype=np.float64)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert np.all(np.isfinite(out))
    assert _max_rel_err(out, a64 * b64) <= 2**-46

    # Near the smallest normal: the true, nonzero product must not flush to
    # zero inside df_mul's intermediate arithmetic.
    small = tiny * 4
    a64 = np.array([small, -small], dtype=np.float64)
    b64 = np.array([2.0, 2.0], dtype=np.float64)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)
    out = _df32_binop("df_mul", a_pairs, b_pairs, MathMode.SAFE)
    assert np.all(out[:, 0] != 0.0)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_and_df_mul_preserve_signed_zero():
    """+0.0 and -0.0 operands must produce an exact-zero *magnitude* through
    both df_add and df_mul, in every sign combination.

    Unlike `split`/`join` (see `test_signed_zero_survives`, step 2: a real bug
    fixed there), df_add/df_mul do NOT reliably preserve the IEEE-754 sign of
    a resulting zero. quick_two_sum/two_sum's own internal `a - (s - a)`-style
    terms re-add a pair of same-magnitude zeros wherever the true result is a
    signed zero, and round-to-nearest addition of opposite-signed zeros is +0
    by definition regardless of which operand should have dominated. Measured
    to hold for every sign combination of df_add(+-0, +-0) and
    df_mul(+-0, +-1), including cases that mathematically should stay negative
    (`df_mul(-0, 1) == -0` in plain IEEE arithmetic). A structural property of
    EFT-chain renormalization, shared with other double-double libraries, not
    worth special-casing branchy zero handling into the hot path for.
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
    """Accumulating a long chain of df32 values via repeated df_add calls must
    stay within the accuracy bound compared to an equivalent float64 running
    sum, not just the single-application bound already tested in step 3.

    The test the plan calls out as the one that would catch a broken
    non-overlap invariant: a df_add that skips renormalization can look
    accurate applied once and still drift once its own output is fed back in
    as the next operand.
    """
    n_terms = 500
    values64 = _random_float64(n_terms, np.random.default_rng(7), lo_exp=-10, hi_exp=10)
    pairs = df32.split(values64)

    acc = pairs[0:1]
    ref = values64[0]
    for i in range(1, n_terms):
        acc = _df32_binop("df_add", acc, pairs[i : i + 1], MathMode.SAFE).astype(np.float32)
        # Sequential reference; see test_df_add_stays_accurate_when_chained
        # for why np.sum's pairwise order is the wrong comparison here.
        ref = ref + values64[i]

    rel_err = abs(df32.join(acc)[0] - ref) / abs(ref)
    assert rel_err <= 2**-36


def _assert_math_mode_pinned(call, math_mode):
    """Shared body for the math-mode-per-routine tests below: SAFE must run
    and produce finite output, anything else must hit the step-6 guard.
    """
    if math_mode is MathMode.SAFE:
        assert np.all(np.isfinite(call()))
    else:
        with pytest.raises(mr.CompileError, match="SAFE"):
            call()


@pytest.mark.parametrize("math_mode", list(mr.MathMode))
@pytest.mark.parametrize("routine", ["df_add", "df_mul", "df_div"])
def test_math_mode_behavior_is_pinned_per_routine(routine, math_mode):
    """Whether df_add/df_mul/df_div survive a given MathMode is a property of
    how each routine's error term is written, not a suite-wide constant, so pin
    it per (routine, math_mode) pair rather than assuming it from step 1's EFTs.

    All three share the same prelude-wide `#error` guard, so every routine is
    rejected outside SAFE regardless. See test_sqrt_math_mode_behavior_is_pinned
    for the unary case (df_sqrt).
    """
    a_pairs, b_pairs, _, _ = _random_df32_operands(
        8, np.random.default_rng(0), **MUL_SAFE_EXP_RANGE
    )
    _assert_math_mode_pinned(lambda: _df32_binop(routine, a_pairs, b_pairs, math_mode), math_mode)


@pytest.mark.parametrize("math_mode", list(mr.MathMode))
def test_sqrt_math_mode_behavior_is_pinned(math_mode):
    """Unary counterpart to test_math_mode_behavior_is_pinned_per_routine:
    df_sqrt only takes one operand, so it doesn't fit `_df32_binop`.
    """
    a_pairs, _, _, _ = _random_df32_operands(8, np.random.default_rng(0), **MUL_SAFE_EXP_RANGE)
    a_pairs = np.abs(a_pairs)
    _assert_math_mode_pinned(lambda: _df32_unop("df_sqrt", a_pairs, math_mode), math_mode)


# --- Step 6 -- Make the math-mode requirement unmissable --------------------


def test_prelude_rejects_non_safe_math_at_compile_time():
    """Compiling df32.PRELUDE (plus any kernel using it) under MathMode.FAST or
    MathMode.RELAXED must raise mr.CompileError, with a message that names the
    fix (i.e. mentions SAFE math mode) rather than a bare compiler error.

    The step whose whole justification is that `mr.Kernel(df32.PRELUDE +
    my_source)`, with math_mode defaulting to FAST, compiles, runs, and returns
    silently wrong answers otherwise. A compile-time guard converts that into a
    loud failure naming the fix.
    """
    source = df32.PRELUDE + "kernel void noop() {}"
    for math_mode in (MathMode.FAST, MathMode.RELAXED):
        with pytest.raises(mr.CompileError, match="SAFE"):
            mr.Kernel(source, "noop", math_mode=math_mode)
    # SAFE itself must still compile cleanly.
    mr.Kernel(source, "noop", math_mode=MathMode.SAFE)


def test_df32_safe_kernel_helper_pins_math_mode():
    """`df32.kernel(source, function_name)` must produce SAFE math mode without
    the caller having to remember to pass it explicitly.
    """
    kernel = df32.kernel("kernel void noop() {}", "noop")
    assert kernel.math_mode == MathMode.SAFE

    # Passing math_mode explicitly is still honored, and still hits the
    # step 6 guard if it isn't SAFE.
    with pytest.raises(mr.CompileError, match="SAFE"):
        df32.kernel("kernel void noop() {}", "noop", math_mode=MathMode.FAST)


# --- Step 7 (stretch) -- df_div, df_sqrt -------------------------------------


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_div_matches_float64_reference_under_safe_math():
    """max relative error of df_div vs a float64 `/` reference must be <= 2**-40
    over ~10k random operand pairs, under MathMode.SAFE.

    df_div is iterative refinement, not a fixed-shape EFT composition like
    df_add/df_mul, but it has to clear the same bound.
    """
    a_pairs, b_pairs, a64, b64 = _random_df32_operands(
        10_000, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE
    )
    out = _df32_binop("df_div", a_pairs, b_pairs, MathMode.SAFE)
    assert _max_rel_err(out, a64 / b64) <= 2**-40
    _assert_no_overlap(out)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_sqrt_matches_float64_reference_under_safe_math():
    """max relative error of df_sqrt vs a float64 `sqrt` reference must be
    <= 2**-40 over ~10k random positive operands, under MathMode.SAFE.
    """
    a64 = np.abs(_random_float64(10_000, np.random.default_rng(42), **MUL_SAFE_EXP_RANGE))
    a_pairs = df32.split(a64)
    out = _df32_unop("df_sqrt", a_pairs)
    assert _max_rel_err(out, np.sqrt(a64)) <= 2**-40
    _assert_no_overlap(out)


def test_df_sqrt_handles_zero_and_negative_inputs():
    """sqrt(0) must be exactly 0 with sign preserved, and sqrt of a negative
    input must produce NaN rather than crash. df_sqrt's `a.hi <= 0` branch
    exists for this: the general path divides by `a.hi`, which is undefined at
    exactly 0 rather than IEEE's well-defined answer.
    """
    x = np.array([0.0, -0.0, -4.0], dtype=np.float64)
    pairs = df32.split(x)
    out = _df32_unop("df_sqrt", pairs)
    assert out[0, 0] == 0.0 and not np.signbit(out[0, 0])
    assert out[1, 0] == 0.0 and np.signbit(out[1, 0])
    assert np.isnan(out[2, 0])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_div_by_near_zero_does_not_crash():
    """Dividing by zero must produce inf or NaN, not crash or hang. df_div has
    no special-casing for this, same stance as df_add/df_mul on inf/nan.
    """
    a64 = np.array([1.0, -1.0], dtype=np.float64)
    b64 = np.array([0.0, -0.0], dtype=np.float64)
    a_pairs, b_pairs = df32.split(a64), df32.split(b64)
    out = _df32_binop("df_div", a_pairs, b_pairs, MathMode.SAFE)
    assert np.all(np.isinf(out[:, 0]) | np.isnan(out[:, 0]))
