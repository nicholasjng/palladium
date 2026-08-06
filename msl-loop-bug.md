# A Metal compiler bug in loop-carried arrays, found by the fuzzer

Status: open, not fixed, exposure understood and narrow. Found 2026-08-11 by
`test_fuzz_loops` (`tests/test_fuzz_emitter.py`). Regression test:
`test_carry_read_and_alias_conflict` (`tests/test_05_loops.py`, `xfail`).
`_rule_scan` (`src/palladium/emit.py`) was reverted to its pre-diagnosis form
after the attempted fix turned out not to work; see below. A second,
narrower-than-stretch-13 fix attempt (loop fusion without expression
folding) was also tried and reverted 2026-08-12; see "Why it isn't fixed
yet" for why a smaller patch isn't viable.

## Symptom

A `fori_loop`/`scan` carry that is both read directly by a fresh
(non-pass-through) computation *and* separately aliased into a different
carry's output comes out wrong, but only from the second loop iteration
onward.

Minimal repro:

```python
def kernel(x0_ref, x1_ref, x2_ref, o0_ref, o1_ref, o2_ref):
    def step(_, carry):
        c0, c1, c2 = carry
        return c2, jnp.tanh(c0 + 0.0 * c0), c0  # c0<->c2 swap, c1 = fresh(c0)

    f0, f1, f2 = jax.lax.fori_loop(0, 2, step, (x0_ref[...], x1_ref[...], x2_ref[...]))
    o0_ref[...] = f0
    o1_ref[...] = f1
    o2_ref[...] = f2
```

Inputs `x0=x1=[0]*16`, `x2=[1]*16`. Expected (and what `interpret=True`
gives): `(0.0, tanh(1.0), 1.0)` = `(0.0, 0.7616, 1.0)`. Actual GPU output:
`(1.0, 0.7616, 1.0)`: only the first slot is wrong.

- At `length=1`: correct.
- At `length=2` or higher: wrong, consistently, every run.
- Reproduces identically under `MathMode.FAST`, `SAFE`, and `RELAXED`,
  ruling out floating-point reassociation as the cause.

## Diagnosis

**First, ruled out an emitter logic error.** `_rule_scan` emits a two-phase
copy-back: carries that a body pass-through would alias into a *different*
carry's slot get snapshotted into a temp *before* any carry is overwritten
(so a swap like `c0, c1 = c1, c0` doesn't read a value that's already been
clobbered), then the temps are copied into the real carries. The emitted MSL
for the repro above:

```c
float t6[16];  // = c0
float t8[16];  // = c1
float t10[16]; // = c2
for (uint _s = 0; _s < 2; ++_s) {
    float t13[16]; for (...) t13[i] = 0.0f * t6[i];
    float t15[16]; for (...) t15[i] = t6[i] + t13[i];
    float t17[16]; for (...) t17[i] = tanh(t15[i]);   // fresh value, reads t6
    float t19[16]; for (...) t19[i] = t10[i];          // snapshot for slot 0 (aliases c2)
    float t21[16]; for (...) t21[i] = t6[i];            // snapshot for slot 2 (aliases c0)
    for (...) t6[i]  = t19[i];
    for (...) t8[i]  = t17[i];
    for (...) t10[i] = t21[i];
}
```

Hand-simulating this statement by statement in Python, iteration by
iteration, with concrete numbers, gives `(0.0, 0.7616, 1.0)`, the correct
answer. Plain `jax.lax.fori_loop` with the same body, run directly with no
palladium involved, also gives `(0.0, 0.7616, 1.0)`. **The C semantics of
the emitted source are correct.** Something between "correct C" and "what
the GPU actually computes" is wrong.

**Second, reproduced independently in hand-written MSL**, no palladium code
involved at all, ruling out anything specific to how palladium constructs
`Buffer`/`Kernel`/dispatch:

```c
kernel void k(... uint3 _pid [[thread_position_in_grid]]) {
    float c0[16]; for (i) c0[i]=arg0[i];
    float c1[16]; for (i) c1[i]=arg1[i];
    float c2[16]; for (i) c2[i]=arg2[i];
    for (uint s = 0; s < 2; ++s) {
        float s0[16]; for (i) s0[i]=c0[i];
        float s1[16]; for (i) s1[i]=c1[i];
        float s2[16]; for (i) s2[i]=c2[i];
        float t[16];  for (i) t[i]=tanh(s0[i] + 0.0f*s0[i]);
        for (i) c0[i]=s2[i];
        for (i) c1[i]=t[i];
        for (i) c2[i]=s0[i];
    }
    for (i) arg3[i]=c0[i]; for (i) arg4[i]=c1[i]; for (i) arg5[i]=c2[i];
}
```

This is the "always snapshot everything, unconditionally" pattern (see
below), still wrong on real hardware: `got=(1.0, 0.7616, 1.0)`,
`want=(0.0, 0.7616, 1.0)`. Same wrong answer, same shape, with zero
palladium code in the loop.

**Third, found the structural trigger.** A version of the same computation
using **scalars** instead of 16-element arrays (one value per thread, no
inner per-element `for` loops) gives the **correct** answer:

```c
float c0 = arg0[tid], c1 = arg1[tid], c2 = arg2[tid];
for (uint s = 0; s < 2; ++s) {
    float s0 = c0, s1 = c1, s2 = c2;
    float t = tanh(s0 + 0.0f * s0);
    c0 = s2; c1 = t; c2 = s0;
}
```

This is structurally identical to the failing array version, just without
the per-element `for` loops. It works. So the bug is not about the
read+alias data pattern in the abstract; it's specific to representing each
array operation as its own separate `for` loop, several of which touch
same-sized arrays, inside a loop body that itself repeats.

**Fourth, found that partial fusion doesn't help.** Fusing only the carry
snapshot and write-back loops into one pass each (one loop that copies
`c0,c1,c2` into `s0,s1,s2` together, one loop that writes `c0,c1,c2` back
together), while leaving the body's own elementwise chain (`t19`, `t21`,
`t23` above) as separate loops, still reproduces the bug. The snapshot/
write-back loops were never the problem in isolation.

**Fifth, found what actually works.** Fully fusing *every* per-element
operation for one loop iteration into a single pass (one `for` loop per
outer iteration, not several) gives the correct answer:

```c
for (uint s = 0; s < 2; ++s) {
    for (uint i = 0; i < 16; ++i) {
        float s0 = c0[i], s1 = c1[i], s2 = c2[i];
        float t = tanh(s0 + 0.0f * s0);
        c0[i] = s2; c1[i] = t; c2[i] = s0;
    }
}
```

Confirmed correct on real hardware, both as hand-written MSL and as the
output of a rewritten `_rule_scan` that always snapshots every carry
(removing the alias-detection special-casing entirely): the *rewrite*
didn't fix anything on its own; what fixed it was that particular rewrite
happening to also collapse the loop count in a case I tested, and further
testing showed the loop *count/fusion*, not the snapshot strategy, is what
matters. Going back to a version of the always-snapshot rewrite that still
emitted separate per-op loops reproduced the bug again.

## Conclusion

This is best characterized as a **Metal shader compiler code-generation
bug** (or at least a very sharp, undocumented edge case), not a palladium
logic error:

- The C emitted by `_rule_scan` is provably correct by direct simulation.
- The bug requires *both* multiple per-element sub-loops *and* an outer
  loop that repeats (>= 2 iterations): scalars are fine, a single fused
  loop is fine, only "several loops, repeated" fails.
- It is insensitive to `MathMode`, ruling out reassociation-style
  explanations.
- It was reproduced with zero palladium code involved (hand-written MSL,
  dispatched directly via `metal_runtime.Kernel`/`run`).

Plausible mechanism (not confirmed, no access to Metal's internal IR): the
AIR compiler's loop-fusion or register-allocation pass merges or reorders
the several small per-element loops across outer-loop iterations in a way
that's unsound when a value is both read by one fused group and later
written by another in the same iteration. This is exactly the class of
optimization that a single fused loop has no room to get wrong.

## Why it isn't fixed yet

The confirmed fix, fully fusing each loop iteration's body into one pass,
is not a small patch to `_rule_scan`. It requires the emitter to build
expressions instead of eagerly materializing every intermediate into its
own thread-local array with its own copy loop, i.e. ROADMAP's stretch 13
(the expression-AST rewrite: fold single-use temporaries, render with
precedence-aware minimal parens). That's real emitter-architecture work,
not something to improvise under pressure while chasing a fuzzer failure.

`_rule_scan` was reverted to its original, pre-diagnosis form (selective
snapshot: only carries detected to alias a *different* carry's output get
a temp, self-forwards and fresh values don't). The always-snapshot rewrite
tried during diagnosis was **removed**, not kept, because it changed the
emitted structure (breaking golden snapshots) and added overhead without
fixing anything, once tested against the real array-loop shape rather than
the misleadingly-passing scalar experiment that first suggested it worked.

**A second attempt (2026-08-12), scoped as narrowly as possible before
committing to full stretch 13, also failed and was reverted.** Tried:
loop fusion only, no expression folding, giving `EmitState` a shared
per-element loop that `copy`/`_rule_elementwise`/`_rule_random_bits`
reuse instead of each opening its own, closed when the enclosing block
exits. This alone reproduces the confirmed regression fix (the
`test_carry_read_and_alias_conflict` repro passed) but breaks other
kernels: `declare()` still allocates every intermediate as its own full
array, and a *new* array's declaration can't be textually placed inside
a loop that's already open and shared with an *earlier*, unrelated op
without either redeclaring it every iteration (wrong) or somehow
hoisting the declaration above the loop (which the single-pass, no
look-ahead emission model can't do). Concretely, a kernel mixing scalar
and array carries in one `fori_loop` (`test_negative_inf_sentinel`) got
an entire unrelated block nested inside a leftover open loop, corrupting
a scalar carry (`git diff` reverted, not landed).

The confirmed-working example in this doc's Diagnosis section uses
**scalar, per-iteration locals** for intermediates (`float s0 = c0[i]`,
not `float s0[16]`), not "keep full arrays, just share the loop." That's
the actual reason fusion alone isn't a smaller patch than stretch 13:
avoiding the declare-ordering problem needs intermediates to stop being
separately-declared storage at all, which needs deferred/lazy expression
construction, i.e., a form of `CVal.expr` becoming an expression rather
than a name. One real narrowing this attempt did surface, worth keeping
for whoever picks up stretch 13: every current `ELEMENTWISE` template is
already fully self-parenthesizing (`"({a} + {b})"`), so substituting a
folded expression as an operand is correctness-safe today, verbose but
correct, without precedence-aware minimal-paren rendering. That means
stretch 13's two stated pieces aren't equally load-bearing: "fold
single-use temporaries into their consumers" is what this bug actually
needs; "render with precedence-aware minimal parens" is a separate
readability pass that could ship later, or not at all, without blocking
the fix.

## Practical exposure

Narrow. Every kernel currently in this repo (the RK4 capstone, the
adaptive controller, the Gray-Scott stencil's launches, the SDE
Monte Carlo loop) either doesn't stage this specific
read-by-fresh-computation-plus-alias-into-a-different-carry shape, or
wasn't affected in testing. It took the fuzzer's random composition to
find it (`test_fuzz_loops`, not a hand-written test), exactly the kind of
thing the fuzzer exists to catch. Running `uv run pytest -m fuzz` will
intermittently rediscover this on its own, depending on Hypothesis's
random seed; that's expected, not new breakage.

If you write a kernel with a scan/fori_loop carry that is (a) read by
another carry's fresh computation in the same body and (b) also
pass-through-aliased into a different carry's slot, diff it against
`.interpret` before trusting it, the way every kernel here already should.

## Pointers

- `tests/test_05_loops.py::test_carry_read_and_alias_conflict`, `xfail`,
  the tracked regression test.
- `tests/test_fuzz_emitter.py::test_fuzz_loops`, the fuzz test that found
  this; will fail intermittently across runs until fixed.
- `src/palladium/emit.py`, `_rule_scan`, short pointer to this file.
- Fixing it needs full loop fusion in the emitter (fold single-use
  temporaries into their consumers, one loop per outer iteration instead
  of several), not a scan-specific patch.
