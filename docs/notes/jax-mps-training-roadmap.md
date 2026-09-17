# jax-mps training roadmap: parameter recovery to continuous normalizing flows

Date: 2026-09-17. Follow-up to the `palladium.dispatch` / jax-mps custom-call
spike. This is a staged validation plan, not a promise of an unrestricted
autodiff backend.

## Current baseline

Palladium can trace a Pallas kernel, emit MSL, and lower it as a
`stablehlo.custom_call @palladium.dispatch`. jax-mps encodes that dispatch on
its existing MLX/Metal stream, so surrounding JAX work remains on MPS. The
initial endpoint is forward-only: it has a portable interpreter fallback and
an MPS integration test, but no differentiation rule.

The RK4 ensemble measurement supplies the motivating workload: one fused
fixed-step solve per independent trajectory was 1.059 ms median for 100,000
Lotka--Volterra systems over 500 steps, versus 63.008 ms for the equivalent
MPS `jit(vmap(scan(...)))` expression. Timings are synchronized, warmed
medians; first-call compilation is reported separately in the benchmark.

## Phase 1: parameter recovery

Build a small, rigorous differentiable-simulation example before attempting a
learned vector field:

1. Generate observations from batched Lotka--Volterra trajectories with known
   parameters `(a, b, c, d)` and optionally noisy partial observations.
2. Optimize per-trajectory or shared parameters by minimizing trajectory or
   sampled-timepoint error. Keep the optimizer, loss reduction, batching, and
   parameter representation as ordinary JAX/jax-mps code.
3. Supply a custom VJP for the Palladium call. Its backward computation must
   produce gradients for inputs/parameters and should be validated against the
   portable JAX reference for values and derivatives.
4. Start with fixed-step RK4 and either store/checkpoint sufficient state or
   implement a discrete reverse-time adjoint. Do not claim continuous-adjoint
   semantics while differentiating a discrete solver.
5. Compare loss curves, gradients, and final recovered parameters across CPU
   reference, jax-mps baseline, and Palladium. Measure full optimizer-step
   time as well as forward-only time.

This isolates the key product question: can a fused Pallas solve participate
in a real `value_and_grad` training loop while preserving a clear numerical
contract?

## Phase 2: conditional continuous normalizing flow

Use the differentiated solver in a deliberately kernel-friendly CNF:

```text
state (z, logp), condition c
dz/dt    = f_theta(z, t, c)
dlogp/dt = -trace(df_theta/dz)
```

Initial constraints:

- fixed-step RK4 and a low-dimensional state (2--8 dimensions);
- exact Jacobian trace, not a stochastic estimator;
- a small conditioned MLP or similarly simple vector field;
- likelihood loss, with known/simple target distributions before a real data
  application;
- numerical/gradient checks against a pure-JAX fixed-step reference.

For higher-dimensional flows, evaluate a Hutchinson trace estimator only after
the exact-trace path is correct; its RNG, variance, and gradient behavior are
part of the model contract.

## Compilation boundary to evaluate

There are two useful designs, which should be measured separately.

1. **Fused solve and vector field.** Compile RK4 plus the small vector field
   into one Pallas kernel. This gives the clearest dispatch reduction and is
   the compelling performance demonstration.
2. **Palladium integrator with jax-mps vector field.** Keep a more general
   learned MLP in regular JAX and use Palladium only where a kernel boundary is
   profitable. This is less likely to eliminate all dispatches, but is the
   better composability test and supports a broader model surface.

Begin with (1) to establish the performance ceiling, then implement (2) only
if its interface does not require host synchronization or an excessive number
of crossings.

## Exit criteria

- No device-to-host transfer between the custom call and surrounding training
  operations.
- CPU-reference agreement for forward values and input/parameter gradients,
  with stated tolerances and deterministic test cases.
- A demonstrated optimizer run that recovers parameters rather than merely
  evaluates a loss.
- Full-step timing, including backward and optimizer work; never extrapolate
  training speedup from the forward-only RK4 benchmark.
- Explicit rejection or documented behavior for unsupported features such as
  aliases, external `vmap`, adaptive stepping, and non-FAST math modes.
