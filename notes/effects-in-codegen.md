# Ref effects in codegen

JAX records, per equation, which Refs it may read or write, including
writes inside nested `scan`, `cond`, and `while` bodies
(`palladium.effects.eqn_reads_ref` / `eqn_writes_ref`). Two checks use
this today: the alias ordering check and the parallel write-race check in
`trace.py`. The remaining ideas, with what is known about each:

## View binding for Refs nothing writes later (measured, not worth it)

`_rule_get` binds a pointer view instead of copying only for `const device`
refs. Effects could widen that to any Ref with no write between the load
and its last use. Counting loads over the test suite (2026-09-30): 558 of
608 copies come from read-only inputs, and those copies are deliberate (a
local copy stays in registers across loops instead of re-reading device
memory). Only 7 copies are from writable device refs. The ordering
analysis through nested jaxprs and fused lazy values is not worth that.
Revisit if a kernel hits the per-thread stack limit because of a large
block read back from an output or scratch ref.

## Missing-barrier lint (open)

Barriers carry `_ThreadgroupEffect`, so they survive DCE and stay visible in
the jaxpr. A pass could flag threadgroup scratch written and then read with
no barrier between them. `_validate_barriers` checks only that barriers
are not in lane-dependent control flow. Deciding whether the reading lane
is different from the writing lane needs index analysis of `thread_index()`
arithmetic, so a first version would warn on any write-then-read of the
same scratch ref with no barrier in between.

## Hoisting loads out of loop bodies (open, measure first)

If a `scan` equation has no write effect on a Ref, its loads can move out
of the loop body. The Metal compiler cannot do this for non-const `device`
pointers because it cannot rule out aliasing. For `const device` inputs the
compiler may already hoist, so measure on a loop-heavy kernel (RK4
ensemble, SIR simulator) before building it.
