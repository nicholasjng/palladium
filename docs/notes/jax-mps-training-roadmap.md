# jax-mps training roadmap: parameter recovery to continuous normalizing flows

Date: 2026-09-17. Follow-up to the `palladium.dispatch` / jax-mps custom-call
spike. This is a staged validation plan, not a promise of an unrestricted
autodiff backend.

## Current baseline

Palladium can trace a Pallas kernel, emit MSL, and lower it as a
`stablehlo.custom_call @palladium.dispatch`. jax-mps encodes that dispatch on
its existing MLX/Metal stream, so surrounding JAX work remains on MPS. It has
a portable interpreter fallback and an MPS integration test. Calls may opt
into a correctness-first custom VJP backed by a supplied pure-JAX reference,
or pair a forward and an explicit backward Pallas call. The latter makes both
directions fused custom calls when the supplied kernel implements a valid VJP.

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

### Validated training workflow

The reusable experimental workload lives in `palladium.ode_training`.
`make_solver` selects pure JAX, fused forward with a reference VJP, the
six-direction tangent VJP, or the discrete reverse VJP. A forward-only variant
is also available. State count, step count, timestep, and checkpoint interval
are explicit factory arguments; the final partial checkpoint segment is handled
without integrating beyond the requested time.

The reverse kernel stores one state per checkpoint and recomputes the prefix
inside that segment for each reverse step. It uses O(steps * interval) work and
8 * trajectories * ceil(steps / interval) bytes of saved float32 states.
It does not yet replay each segment once into a temporary local tape.
Interval 1 supplies the full-history oracle.

Tests compare arbitrary signed output cotangents and all six input gradients
against a CPU JAX VJP for multiple random inputs, horizons and intervals,
including intervals longer than the trajectory and nondivisible segments.
Forward, tangent, reference-VJP and reverse variants share the test workload.
The recovery test checks loss reduction, parameter accuracy, and several Adam
updates against the pure-JAX implementation.

Loss, VJP, and Adam now run inside one `jax.jit`. Observations are generated
on CPU outside training. Model inputs, checkpoints, gradients and optimizer
updates remain on MPS during each training step.

The default 4,096-trajectory recovery run completed 500 updates in 0.471 s
(one synchronization at the end), reducing loss from 0.24893 to 3.10e-08.
Recovered parameters were (1.099902, 0.399927, 0.101211, 0.401802).
This total is separate from the individually synchronized latency benchmark.

Run from the jax-mps checkout (with the Palladium custom-call handler installed):

```sh
JAX_PLATFORMS=mps,cpu env -u VIRTUAL_ENV uv run python ../palladium/examples/07_mps_parameter_recovery.py --n 4096 --steps 100 --interval 10 --iterations 500
JAX_PLATFORMS=mps,cpu env -u VIRTUAL_ENV uv run python ../palladium/benchmarks/bench_mps_training.py --repeats 15
```

The benchmark measures complete Adam updates from identical optimizer states,
rotates variant order, synchronizes every sample, and excludes input transfers.
It reports lowering/compilation and first execution separately; first execution
can include lazy Metal compilation and persistent caches are not cleared.

Measured locally with JAX 0.11.1, 4,096 trajectories, 100 RK4 steps, and
15 repetitions:

| Variant | Median update (ms) | Min–max (ms) | Saved checkpoint states (MiB) | Compile / first execute (ms) |
|---|---:|---:|---:|---:|
| Pure JAX on MPS | 39.173 | 37.674–42.363 | — | 56.8 / 76.3 |
| Fused forward, reference VJP | 39.064 | 38.192–41.684 | — | 49.1 / 128.9 |
| Fused tangent VJP | 0.503 | 0.478–0.578 | 0 | 83.6 / 4.2 |
| Reverse, interval 1 | 0.479 | 0.451–1.007 | 3.125 | 33.5 / 97.8 |
| Reverse, interval 5 | 0.496 | 0.471–0.603 | 0.625 | 28.7 / 92.5 |
| Reverse, interval 10 | 0.482 | 0.470–0.502 | 0.3125 | 28.6 / 3.0 |
| Reverse, interval 25 | 0.622 | 0.595–0.925 | 0.125 | 28.1 / 93.5 |

This run gives about 81x lower median update latency for interval 10 than pure
JAX on MPS. Saved-state bytes are not peak device memory; JAX's internal adjoint
storage is not measured by that column. Results are specific to this small
Lotka–Volterra workload and do not establish a CNF speedup. Earlier multi-second
training totals included Python-dispatched optimizer operations and are not
directly comparable to these complete compiled steps.

Shared scalar parameters are expanded to per-trajectory inputs because the
current MLX adapter may place small inputs in Metal's constant address space,
whereas Palladium expects device pointers. This adapter limitation remains.
IEEE NaN and signed-zero semantics for sign/remainder are tested through
metal-runtime SAFE mode; the MPS bridge currently supports FAST only.

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
