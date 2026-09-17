# ODE trajectory compiler: project direction

Date: 2026-09-11. Design exploration, not a Palladium implementation commitment.

## Goal

Build an MLIR-oriented compiler project demonstrating IR design, transformation
legality, memory optimization, and numerical validation through ODE and dynamics
applications. Reuse experience and workloads from Palladium without requiring a
rewrite of its JAX-to-Metal emitter.

The central question is how requested outputs, storage decisions, and numerical
semantics interact in compiled dynamical systems.

## Proposed stack

- **Python package with xDSL/PyRDL:** prototype domain operations, verifiers,
  interpreters, and passes with a short development loop.
- **Existing dialects:** use ordinary arithmetic and structured control flow
  where possible. Introduce domain operations only to preserve useful concepts
  such as stepping, sampling, and trajectory production.
- **Optional egglog pass:** optimize pure expressions inside a step using
  explicit rule sets and a cost model. Keep mutable memory and loop scheduling
  outside the initial e-graph boundary.
- **Native MLIR backend:** lower the supported representation to standard
  dialects, then use bufferization and LLVM lowering for CPU execution.
- **Later C++ milestone:** port one substantial analysis/transformation into
  native MLIR to demonstrate upstream-style compiler engineering.

Conceptual pipeline (individual lowering passes still need to be selected and
validated):

```text
RHS + fixed-step method + requested outputs
  -> domain IR retaining step and observation structure
  -> observation/storage analysis and rewrites
     + optional equality saturation of pure step expressions
  -> scf / tensor / arith / math
  -> bufferization and remaining standard lowerings
  -> LLVM -> CPU execution
```

xDSL is a separate Python compiler framework, not the native MLIR Python
bindings. Textual IR exchange requires compatible operation definitions and
versions; Python passes do not automatically become native MLIR passes. Lower
custom operations before handing IR to a native tool that does not know them.

## First useful optimization

Start with a fixed-step solver whose complete trajectory is only consumed by a
pure pointwise observable:

```python
ys = integrate(rhs, y0, times)
energy = map(hamiltonian, ys)
return energy
```

Fuse observation into stepping and store only the energy sequence. If only the
final state is requested, eliminate trajectory storage entirely. If the full
trajectory is required, write directly to its destination when aliasing and
read/write ordering permit, eliminating an intermediate and copy.

Specify output timing precisely: whether the initial state is included, when
samples are taken, and the ordering and shape of observations. Preserve the
integrator's arithmetic ordering in the first optimization.

Legality must account for other trajectory consumers, observable effects,
aliasing, and accesses to earlier/later states. Reject unsupported cases
explicitly. An ordinary analysis and directed rewrite is sufficient here;
equality saturation is an optional second experiment.

## Equality saturation experiment

Explore alternate factorizations and shared expressions in the RHS or stage
calculations. Compare against canonicalization and common-subexpression
elimination to establish the additional value of an e-graph.

Use explicit budgets for graph growth and optimization time. Evaluate operation
cost, shared subexpressions across outputs, and live temporary storage: a simple
expression-tree operation count can miss both sharing and memory pressure.

Separate strict floating-point transformations from explicitly enabled algebraic
transformations. Real-number identities involving reassociation, distributivity,
or trigonometry are not automatically floating-point equalities. Approximate
alternatives must not silently enter strict equality classes. Check rounding,
overflow, exceptional values, and any required assumptions.

## Milestones and evidence

1. Define a narrow fixed-step execution contract and a reference interpreter.
2. Run one small ODE end to end on CPU without optimization.
3. Implement observation fusion and/or direct trajectory output placement, with
   negative tests for cases where the rewrite is illegal.
4. Record before/after IR, temporary allocation/copy counts, peak storage,
   compilation time, and execution time across trajectory lengths and state sizes.
5. Add optional egglog expression optimization and compare against the baseline.
6. Validate long-horizon dynamics alongside local numerical error; report runtime
   and accuracy tradeoffs rather than treating smaller expressions as a win.
7. Port a meaningful pass to native MLIR C++, with regression tests and a short
   design document explaining its legality conditions.

Candidate workloads: Lotka–Volterra with RK4, a harmonic oscillator, and the
Kepler problem with a symplectic integrator. Initially defer adaptive solvers,
autodiff/checkpointing, GPU execution, and a general JAX frontend.

## Python package scaffold

Yes: a small standalone Python package is a sensible starting point. Prefer a
sibling of Palladium so compiler experiments and dependencies can evolve
independently. The final name is undecided; `ode_compiler` below is a placeholder.

```text
pyproject.toml
src/ode_compiler/
    dialects/
    transforms/
    lowering/
    interpreter.py
    driver.py
tests/
    ir/
    execution/
examples/
benchmarks/
docs/
```

Start with xDSL and testing dependencies; add egglog when its experiment begins.
Pin a tested xDSL/native MLIR combination and document the native toolchain
separately from Python installation. A Python package alone does not provision
the LLVM/MLIR backend. A minimal CLI that prints IR and selects passes is useful;
a general Python tracing frontend can wait. This note records the proposed
scaffold; it does not create the package.

## Tooling map and references

- [xDSL](https://docs.xdsl.dev/): Python dialect and pass prototyping.
- [ODS/TableGen](https://mlir.llvm.org/docs/DefiningDialects/Operations/): native
  MLIR operation definitions and generated infrastructure.
- [IRDL](https://mlir.llvm.org/docs/Dialects/IRDL/): dialect schemas represented
  as MLIR, with declarative constraints; not automatic semantic inference.
- [Dialect conversion](https://mlir.llvm.org/docs/DialectConversion/) and
  [interfaces](https://mlir.llvm.org/docs/Interfaces/): explicit lowering and
  capability contracts connecting operations to reusable transformations.
- [PDLL](https://mlir.llvm.org/docs/PDLL/): declarative directed rewrites.
- [Transform dialect](https://mlir.llvm.org/docs/Tutorials/transform/): scheduling
  and controlling transformations; distinct from equality saturation.
- [Bufferization](https://mlir.llvm.org/docs/Bufferization/): tensor-to-buffer
  analysis and conversion infrastructure to investigate before duplicating it.
- [egglog Python](https://egglog-python.readthedocs.io/latest/) and
  [egglog paper](https://arxiv.org/abs/2304.04332): equality saturation combined
  with relational reasoning.
- [Diehl's MLIR/e-graph article](https://www.stephendiehl.com/posts/mlir_egraphs/):
  expression optimization before MLIR generation; an approachable example,
  not an arbitrary-region MLIR optimizer or strict numerical rule library.
- [DialEgg](https://github.com/AzizZayed/DialEgg): research infrastructure for
  applying egglog rules to MLIR. Evaluate compatibility and supported constructs
  before adoption; the reviewed README pins dependencies and lists limitations.
- [Arithmetic dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/): operation
  semantics and floating-point permissions.

See [the other project ideas](compiler-project-ideas.md) for alternative paths.
