# Emitter work for GEMM-shaped training: tiling, transposed lhs, colvec

**Scratch, 2026-08-18.** What `examples/08_mlp_training.py` had to work
around, why, and what emitter work would remove each workaround. The
example's measurements are the gates.

> **Outcome (same day): item 1 variant (b) implemented.** The
> cooperative MMA dot now column-tiles wide outputs (at most
> `_MMA_FRAGMENT_CAP` fragments live at once, threadgroup result tile up
> to `_MMA_TG_TILE_LIMIT` = 16KB, so n up to 512 at R = 8), and the
> detection walker's accumulator cap exempts MMA-eligible dots. The
> single-tile case emits byte-identical MSL (scored attention hashes and
> goldens unchanged). MLP step: 0.10x -> 0.47x at batch 512 (now
> dispatch-floor-bound) and 0.09x -> 0.70x at batch 2048-4096 (now
> GEMM-throughput-bound). Isolated mmt kernel: 247 GFLOP/s at
> 4096x256x256 and 402 GFLOP/s at 4096x512x512 vs XLA-CPU's 359/472.
> Measured non-result, do not retry without staging: R = 16 (more rows
> per instance to amortize rhs reads) is a wash (227 vs 247 GFLOP/s),
> the same reuse-vs-occupancy reciprocal coupling
> query-blocking-scratch.md documented. The remaining lever is item 1c
> below; items 2 and 3 measured as secondary.

## 1. 2-D output tiling (the headline)

The contiguous-block discipline (`_check_contiguous`: only the leading
block dim may be partial) applies to inputs and outputs alike, which
forecloses every standard GEMM decomposition:

- an `(8, 64)` output tile of a `(B, N)` array is non-contiguous;
- a `(K, 64)` rhs column block is non-contiguous;
- in-kernel column slices (`ref[:, dslice(...)]`) are rejected too
  (partial slice only on the first kept dim), so a kernel cannot loop
  over column tiles of a full-width block either.

What remains expressible is one full-width row stripe per instance,
which caps a cooperative matmul at n = 256 (the 64-accumulator budget at
R = 8) and pushes it off the MMA path (tile and fragment caps) onto the
float4 path with 8x8 float4 accumulators per lane, which spills
registers. Measured consequence in example 08: ~24 GFLOP/s effective on
the 4096x256x256 GEMMs, ~40x below what the MMA tile demonstrably does
in the attention kernel. **The widths the emitter can express keep every
GEMM inside the always-lose region; no kernel authorship gets around
it.**

Two designs, not mutually exclusive:

a. **Column-blocked BlockSpecs** (general fix): teach `_block_offset` /
   `_ref_view` strided views. Blocks become (offset, shape, row-stride)
   instead of flat pointer+extent; every rule that flat-indexes a device
   view must consult the stride. Touches the load/store rules, both dot
   rules and their alignment analysis (`CVal.align` is per-flat-element
   today), and `_check_contiguous` itself. Substantial, and it is the
   honest version of "support real GEMM grids".

b. **In-kernel column loop with MMA per tile** (narrow fix): keep blocks
   full-width, let the cooperative dot emit `n/64` MMA tile passes with
   `simdgroup_store` directly to the device output (strided stores are
   fine for `simdgroup_store`, it takes an elements-per-row argument),
   instead of one giant accumulator set. No BlockSpec changes; the dot
   rule grows a column-tile loop and stores per tile. This alone lifts
   the width cap and re-enables MMA at any n, for the dot only;
   elementwise epilogues over the wide output still walk the full
   stripe, which is fine (they are bandwidth-bound).

Recommendation: (b) first. It is contained in `_coop_rule_dot_general` /
`_coop_emit_dot_mma`, measurable against example 08 directly, and does
not disturb the flat-view model everything else relies on. (a) only if
column-blocked *inputs* to non-dot kernels ever matter.

**(b) is implemented; see the outcome banner.** What it did not fix, and
the follow-up it defines:

c. **Cross-instance rhs reuse: implemented (2026-08-18,
   `emit/gemm.py`).** A pattern-gated whole-kernel lowering for
   load/load/(transpose)/dot/store jaxprs: G SIMD-groups per
   threadgroup (one instance each) share a staged rhs slab, fragments
   store straight to the device output (no result tile), and the
   detection walker exempts matched kernels from its accumulator caps
   (so n is unbounded by the tile limit; n = 1024+ works). Geometry is
   adaptive per shape (`_select_geometry`, sweep measurements in its
   docstring); the deciding factor was occupancy, not bandwidth:
   staging alone at G = 4 changed nothing, G = 16 with a 128-deep slab
   hit 835 GF/s. Isolated mmt at batch 4096: 551/787/734 GF/s at
   n = 256/512/1024 vs XLA-CPU's 435/479/459, i.e. **bare-GEMM offload
   now wins 1.3-1.6x**. The MLP *step* remains overhead-bound (0.37x
   at batch 512 to 0.93x at batch 8192/hidden 512): the GEMMs are now a
   minority of the step; the floor, the backward's host transposes
   (~0.5 ms each at batch 4096), and CPU elementwise dominate. That
   re-orders the remaining items: item 2 (transposed-lhs dots) is now
   measurably material, and item 3 (epilogue fusion) next. One
   structural limit remains: a GEMM's grid extent is its m/8, so
   small-m GEMMs (weight gradients of narrow layers) under-occupy the
   GPU regardless of staging; example 08 works around it by computing
   the gradient in the wider orientation and transposing the small
   result on the host.

## 2. Transposed-lhs dots (dim-0 contraction)

`dW = x^T @ g` stages `dot_general` with contraction `(((0,), (0,)))`,
which both dot rules reject; a pre-transposed `x^T` fed as an operand is
also unreachable because a materialized transpose of a device view is
classic-only. Example 08 works around it with XLA-side `.T`s (three per
layer-step: `p.T`, `g.T`, `a.T`), which are bandwidth-cheap but pure
overhead, and at larger widths they scale with the weights.

The fix is small compared to item 1: canonicalize dim-0-contraction
dots by treating the lhs as transposed storage, symmetric to the
existing lazy rhs transpose. The MMA path already has the hardware
transposed load (`simdgroup_load(..., true)`) used for the rhs; the
lhs fragment load takes the same flag. The scalar/vectorized paths read
`lhs[kk * m + r]` instead of `lhs[r * k + kk]` (column-major walk; the
vectorized path would need its own eligibility check since the lhs rows
stop being unit-stride).

Gate: measure the three `.T`s' share of example 08's step first (delta
between the step as-is and the step with transposes excluded from the
timed region). If item 1 lands and transposes are under ~10% of the
step, this can wait.

## 3. Colvec layout for fused bias (deferred, cheap CPU workaround)

`tanh(x @ W + b)` with `b` shaped `(n,)` fits no cooperative layout at
R > 1 (`_coop_geom`: scalar, `(R,)`, `(R, 1)`, `(R, W)`), so one bias
add drops the whole kernel to the classic model. Example 08 sidesteps
it by keeping bias + tanh in jnp on the CPU, which measures as noise
next to items 1 and 2.

Sketch, for when fusion matters (it will if item 1 makes single-kernel
layers competitive, since unfused epilogues then cost an extra pass over
the output):

- The cheap 80% case is a *device-view* colvec: a `(W,)`-shaped input
  ref operand in an elementwise op needs no storage or new layout field,
  only (i) the detection walker accepting a `(W,)`/`(1, W)` device
  operand whose W matches the output width, and (ii)
  `_coop_rule_elementwise`'s `elem_sh` reading it at the global column
  index (`expr[gc]`), exactly how shard-shaped device operands are read
  minus the row term. Bias from an input ref is always this case.
- The full version is a fourth `CVal.layout` kind `"colvec"` (one value
  per column, lane t owns columns `[t*ppw, (t+1)*ppw)`, storage
  identical to a shard at R = 1). Touchpoints: `_coop_geom` (a shape
  taxonomy entry), `_coop_declare`/`_coop_copy`/`_coop_storage_len`,
  elementwise, broadcast (`(W,) -> (R, W)` fill), reduce (colvec source
  along W is one simd reduction). Ambiguity to resolve: `(n,)` with
  n == R currently classifies as rowvec; a computed `(W,)` value with
  W == R is indistinguishable by shape alone. Options: classify by
  producer/consumer context in the walker, or only admit rank-2
  `(1, W)` as colvec and keep bare `(W,)` rowvec-when-square. The
  device-view case dodges this entirely (no storage, so the read style
  can be decided per consumer).

Gate: only after item 1; measure the epilogue pass it would fuse away.

## Order

1 (variant b, done) -> re-measured example 08 (0.47x/0.70x) -> next is
1c if MLP training stays a goal; 2 and 3 measured as secondary (the
host transposes are ~0.55 ms per 4096x512 next to 5-7 ms GEMMs, and the
epilogue pass is noise). Everything after 1c is speculative until it is
measured.

Small loose end found on the way: the "stack space" pipeline error is
wrapped with a helpful message on the eager path (`dispatch.bind`) but
surfaces as a raw JaxRuntimeError on the FFI path, where compilation
happens inside the native handler. Cosmetic; noted here rather than
fixed.
