# Stacked scan support (dense-output ODE steppers)

Status: implemented, including the v2 streaming-store fusion and
`reverse=True`; see `emit/rules.py` (`_rule_scan`, `_split_scan_operands`,
`_xs_slice`, `_ys_stream_target`), `tests/test_21_stacked_scan.py`, the
`test_fuzz_scan_xs_ys` fuzzer strategy (500-example soak clean, no new
Metal miscompile), and the `dense_output_scan` golden snapshot. The
pre-existing goldens are byte-identical, so pure-carry emission is
unchanged. The input-side pairing noted under step 4 is closed too: a
full-block get consumed only as scan xs binds the ref directly
(`_consumed_only_as_scan_xs`), so a save-time grid never touches the
per-thread stack; scan consts keep their deliberate cache copy.
(2026-08-25)

## What and why

`lax.scan` with stacked ys and scanned xs used to be typed-rejected
(`emit/rules.py`, then `_check_pure_carry_scan`), which ruled out the
idiom a dense-output stepper wants to be written in:

    ts = ts_ref[...]
    _, ys = jax.lax.scan(step, y0, ts)   # ys: (n_save, y_dim)
    o_ref[...] = ys

The capability itself is expressible today via `fori_loop` plus indexed
ref writes (`o_ref[i] = y`), so this is idiom compatibility, not new
power. It matters anyway: diffrax-style code and ordinary jax users
write the scan form.

## Verified layout (jax 0.11)

Probed on a traced palladium kernel: scan params carry no
`num_carry`/`num_consts` (only `reverse`, `length`, `ft_in`/`ft_out`,
`unroll`), so the split must stay structural, extending the rank-pairing
`_check_pure_carry_scan` already does. `eqn.invars = [consts..,
carries.., xs..]` where an xs has the body invar's rank + 1 with leading
dim `length`; `eqn.outvars = [carries.., stacked ys..]`, same rule. In
the dense-output shape above, the stacked ys' sole consumer is a
full-block `swap`: the fusion target below exists in exactly the form
needed.

## Plan

1. Replace `_check_pure_carry_scan` with a structural classifier
   (consts/carries/xs, carries/ys) using the rank + leading-dim pairing.
   Small; pure-Python unit tests.
2. Scanned xs: bind each body x-slice per iteration as a strided view
   into the xs CVal (pointer-arithmetic expr; xs is an immutable SSA
   value, so viewing without copy is safe). Scalar xs like `ts` bind as
   `xs[_s]` directly. Small-medium.
3. Stacked ys, v1: declare the stacked outvar as ordinary thread-local
   storage and copy the body's y-slice into `ys + _s * y_size` each
   iteration. Small, but bounded by the per-thread stack (a few KB):
   fine for moderate dense output, not long trajectories. The existing
   stack-space -> EmitError translation already reports overflow with
   the actionable message.
4. Stacked ys, v2 (the real dense-output enabler): when a ys outvar's
   sole consumer (`Environment.consumers`) is a full-block swap to an
   output ref, skip thread-local storage and stream each y-slice
   straight to the device ref at `_s * y_size`. Removes the stack
   ceiling; emits what the manual fori workaround emits by hand.
   Medium. Symmetric input-side pairing: large `ts` still costs a
   full-block stack copy before the scan; the full-block alias
   relaxation from jaxpr-effects item 2 (or direct device-view xs)
   fixes that side.
5. `reverse=True`: loop header and index flip, nearly free once 2-4
   exist.
6. Fuzzer extension, non-optional: the copy-back volatile workaround
   exists because Metal's optimizer miscompiled scan loops with three or
   more live thread-local arrays, and a ys buffer adds another live
   array to exactly that loop shape. Extend the differential fuzzer with
   xs/ys strategies and budget a soak; a fresh miscompile appearing here
   is the risk tail.
7. Docs (`supported-subset.md`), a golden MSL snapshot, dispatch/ffi and
   the trace validators need no changes (ys is a value consumed by the
   existing swap machinery).

## Estimate

v1 (steps 1-3, 5, tests): one focused session. v2 fusion plus the
fuzzer extension and soak: a second session. Dense-output steppers
realistically need v2 (trajectory sizes clear the stack quickly), so
plan 2-3 sessions total, with the uncertainty concentrated in step 6's
soak.
