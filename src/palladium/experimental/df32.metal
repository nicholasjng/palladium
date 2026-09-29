// An emulation of float64 as float32x2.
#if defined(__FAST_MATH__) || defined(__RELAXED_MATH__)
#error \
    "df32.metal requires MathMode.SAFE, as FAST/RELAXED reassociate away the compensation terms this prelude depends on."
#endif

#include <metal_stdlib>
using namespace metal;

struct df32 {
    float hi, lo;
};

// requires |a| >= |b|.
inline df32 quick_two_sum(float a, float b) {
    float s = a + b;
    return df32{s, b - (s - a)};
}

// No ordering required, 6 flops.
inline df32 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return df32{s, (a - (s - bb)) + (b - bb)};
}

// 2 flops, and the reason this is affordable on Apple GPUs at all.
inline df32 two_prod(float a, float b) {
    float p = a * b;
    return df32{p, fma(a, b, -p)};
}

// Accurate variant, not the cheaper sloppy one (two_sum the his, plain-add
// the los): sloppy measures fine typically but hits 2^-35 error on
// near-cancellation inputs, well outside the target bound.
inline df32 df_add(df32 a, df32 b) {
    df32 s = two_sum(a.hi, b.hi);
    df32 t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    s = quick_two_sum(s.hi, s.lo);
    return s;
}

// a.lo*b.lo is dropped: second-order small relative to the other two cross
// terms.
inline df32 df_mul(df32 a, df32 b) {
    df32 p = two_prod(a.hi, b.hi);
    float e = p.lo + (a.hi * b.lo + a.lo * b.hi);
    return quick_two_sum(p.hi, e);
}

// Skips df_mul's own renormalization pass: df_add renormalizes the whole sum
// anyway, and two_sum/quick_two_sum are exact for any float pair, not just
// already-non-overlapping ones, so the unnormalized product limbs are a
// valid input to df_add as-is.
inline df32 df_fma(df32 a, df32 b, df32 c) {
    df32 p = two_prod(a.hi, b.hi);
    float e = p.lo + (a.hi * b.lo + a.lo * b.hi);
    return df_add(df32{p.hi, e}, c);
}

inline df32 df_neg(df32 a) { return df32{-a.hi, -a.lo}; }

inline df32 df_abs(df32 a) { return a.hi < 0.0f ? df_neg(a) : a; }

inline df32 df_sub(df32 a, df32 b) { return df_add(a, df_neg(b)); }

inline df32 df_from_float(float x) { return df32{x, 0.0f}; }

inline float df_to_float(df32 a) { return a.hi; }

// Iterative refinement from a correctly-rounded float32 seed (SAFE math
// guarantees x/y is correctly rounded): each residual gets its own quotient
// term folded back in. One round already clears 2^-45; the second buys about
// another bit of margin for the same shape of code (measured, not assumed).
// q0/q1/q2 are strictly decreasing in magnitude by construction, so the
// combine is quick_two_sum, not the general-ordering df_add.
inline df32 df_div(df32 a, df32 b) {
    float q0 = a.hi / b.hi;
    df32 r = df_sub(a, df_mul(df_from_float(q0), b));

    float q1 = df_to_float(r) / b.hi;
    r = df_sub(r, df_mul(df_from_float(q1), b));

    float q2 = df_to_float(r) / b.hi;

    df32 t = quick_two_sum(q1, q2);
    df32 s = quick_two_sum(q0, t.hi);
    return quick_two_sum(s.hi, s.lo + t.lo);
}

// Karp's trick: a correctly-rounded rsqrt seed x gives a correctly-rounded
// float32 sqrt ax = a.hi*x; the correction term is second-order, so it only
// needs float32 precision, not a second full df32 division. The general path
// divides by a.hi (inside rsqrt), which is 0*inf = NaN at exactly 0 rather
// than IEEE's 0/-0/NaN for zero/negative -- the branch exists for that case,
// not to avoid a crash.
inline df32 df_sqrt(df32 a) {
    if (a.hi <= 0.0f) {
        return df32{sqrt(a.hi), 0.0f};
    }
    float x = rsqrt(a.hi);
    float ax = a.hi * x;
    df32 residual = df_sub(a, df_mul(df_from_float(ax), df_from_float(ax)));
    float correction = df_to_float(residual) * (x * 0.5f);
    return quick_two_sum(ax, correction);
}

// hi first, lo as tiebreak: valid only for non-overlapping pairs
// (|lo| <= 0.5*ulp(hi)), which split()/df_add()/df_mul() always produce.
// Not NaN-aware, same as the rest of this prelude.
inline bool df_eq(df32 a, df32 b) { return a.hi == b.hi && a.lo == b.lo; }

inline bool df_lt(df32 a, df32 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

inline bool df_gt(df32 a, df32 b) { return df_lt(b, a); }

inline bool df_le(df32 a, df32 b) { return !df_lt(b, a); }

inline bool df_ge(df32 a, df32 b) { return !df_lt(a, b); }
