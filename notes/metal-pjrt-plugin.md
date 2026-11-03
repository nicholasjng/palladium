# The `metal` PJRT plugin (M1 complete)

Implementation record and handoff for milestone M1 of
`notes/metal-compute-stack.md`: a PJRT plugin that runs `jax.jit` programs
on the Apple GPU without MLX. This note says what exists, how it was built
and why, what to keep invariant, what comes next, and when M1 counts as
done. Status as of 2026-11-02, on jax and jaxlib 0.12.0.1. Initial
measurements were on M2; the recent operation and lifetime work was validated
on Apple M1 Pro and rechecked on M2. Each measurement section identifies its
machine.

## Status

Works:

- `jax.devices("metal")`, `device_put`, `np.asarray`, deletion, copies to
  CPU, `memory_stats()`, and `apple_gpu_family` as a device attribute.
- `jax.jit` of programs made of elementwise ops (float, integer, unsigned,
  bool, f16, bf16; comparisons, select, convert, bitcast, shifts) and index
  maps (broadcast, reshape, transpose, slice, reverse, iota), with constants,
  multiple outputs, inputs returned as outputs, calls and composites.
- Guarded index maps: `concatenate`, `pad` (including interior padding and
  negative edge padding), `dynamic_slice`, and `dynamic_update_slice`.
  They fuse with producers and consumers without intermediate buffers.
- `jax.random.key` on the Metal default device.
- PJRT-local int64/uint64 elementwise arithmetic, comparisons, conversions,
  same-width bitcasts and index maps, including exact wide literals and bounded
  shifts. Threefry key split/fold-in, random bits, and float32 uniform/normal
  samples run on Metal.
- Single-result `reduce` with add, maximum, minimum, and, or bodies over
  arbitrary axes, including empty reductions. Producers fuse into reduction
  reads; consumers read intermediate slots in later launches. This enables
  softmax, means, variances and normalization programs.
- Variadic scalar elementwise reducers, including argmax/argmin with first-index
  ties and NaN behavior, using a sequential fold per output element. SIMD
  acceleration of these reducers remains future tuning.
- Gather for JAX indexing, including batching, windows, negative indices and
  clip/fill lowering. Scatter-add supports float32, float16, bfloat16, int32
  and uint32 data with 32-bit indices, including duplicate indices and
  embedding gradients. Non-add and variadic scatter bodies remain unsupported.
- Static counted `while`, including `lax.scan` trajectories and nested loops:
  a scalar i32/ui32 counter starting at a constant, compared with LT to a
  constant bound, incremented by one. Body kernels replay in one command
  buffer with alternating carry buffers and no host condition reads.
- Floating `dot_general`: plain/transposed/batched products, multiple batch
  and contraction axes, vectors and empty dimensions. It reuses the Pallas
  GEMM emitter; gradients and dots inside counted loops work too.
- Pure `stablehlo.case` branches (`lax.cond`/`lax.switch`), including nested
  conditionals, tuples, gradients and composition with reductions/GEMMs/static
  loops. Elementwise branch reads are guarded in the fused kernel. Materialized
  branch computations run eagerly before result selection; there is no host
  condition read, but untaken expensive branches still consume work and storage.
- Asynchronous execution with completion events, a buffer pool, and at most
  16 executions in flight.
- Donation via `tf.aliasing_output`: ordered GPU copyback transfers input
  storage to the output after all reads. Runtime non-donatable inputs and
  repeated arguments retain their storage. This still allocates temporary
  outputs; in-place kernel planning is not implemented.
- Concurrent calls with shared inputs retain independent storage owners until
  GPU completion. Passthrough and repeated outputs are copied on the GPU queue.
- The complete SBI step compiles and executes, with a ten-step CPU comparison
  of losses, parameters and optimizer state (details and tolerance below).
- Palladium kernels inside `jit`: `pl.pallas_call` (any math mode, aliases)
  and `metal_call` (including under `jax.vmap`) become launches of the
  program, also inside counted loops. The CNF update of
  `examples/cnf_density.py` matches CPU over ten steps.
- Kernels bind at most 31 buffers, Metal's limit: wider fusions split by
  output, and values reading more buffers store intermediate operands.

Fails with UNIMPLEMENTED, naming the op (`palladium.stablehlo` raises
`UnsupportedProgram`): unsupported reducer bodies (including single-result
product), integer or explicitly algorithm-selected dots, dynamic or unrecognized `while`,
non-add scatter and unsupported scatter dtypes, `reduce_window`, `sort`,
`convolution`, float64 arithmetic and
64-bit integer reductions. Shape-changing bitcasts and wide integer operations
without explicit lowering (such as power) are rejected too. In
`examples/training_steps.py`, SBI and LLM now run. The LLM has a one-update
CPU regression test with varied repeated tokens at rtol 1e-4, atol 2e-6; this
is not an M3 performance claim. SAM2-like still stops at `convolution`. CPU setup remains useful for reference inputs:

```python
with jax.default_device(jax.devices("cpu")[0]):
    key = jax.random.key(0)
```

Not supported by design yet: host callbacks (`jax.debug.print`, `pure_callback`), `eigh` and
`svd`, the persistent compilation cache (executables do not serialize),
more than one device.

## Build, run, test

Requirements: macOS 26 on Apple silicon, uv, CMake and Ninja (`brew install
cmake ninja`), and network access on the first configure, which downloads
metal-cpp and the PJRT C API header. Xcode is not needed.

```sh
uv sync
# uv does not rebuild for C++-only edits; after changing native/:
uv sync --reinstall-package palladium

JAX_PLATFORMS=metal,cpu uv run python -c "import jax; print(jax.devices())"
uv run pytest -q tests/runtime/test_pjrt.py tests/codegen/test_stablehlo.py
uv run python examples/training_steps.py   # StableHLO op census
```

The plugin registers only when JAX_PLATFORMS names `metal`; `metal,cpu`
makes it the default device, `cpu,metal` keeps CPU the default. Plugin tests
run in subprocesses because jax reads JAX_PLATFORMS once per process. The
full check is `ruff check`, `ruff format --check`, `ty check` and `pytest`.

A segfault inside `libjax_common.dylib` at a `pjrt::` or `PjRtCApi*` frame
almost always means jaxlib called an API slot the plugin left null. Run with
`python -X faulthandler` to see the frame, then implement that slot.

## Layout

| Path | What |
|---|---|
| `native/pjrt/plugin.h` | Shared types: `Error`, `Memory` (extending the C API's function-table structs), `Completion`, `PJRT_Buffer`, the pool interface |
| `native/pjrt/plugin.cpp` | `GetPjrtApi`: errors, events, client, device, memory, topology, buffers, the pool |
| `native/pjrt/executable.cpp` | `PJRT_Client_Compile`, the `pp_*` builder ABI, `Execute`, executable queries, in-flight throttling |
| `native/runtime/dispatch.*` | `CommandBatch::commit(on_completed)`, the non-blocking commit Execute uses |
| `src/palladium/_pjrt.py` | `jax_plugins` entry point (`initialize`), ctypes signatures, the compile callback |
| `src/palladium/stablehlo.py` | StableHLO to `Program`: slots, launches, outputs, MSL |
| `src/palladium/_stablehlo_gemm.py` | Canonical batched products to the Pallas GEMM emitter |
| `CMakeLists.txt` | `palladium_pjrt` target, the pinned header download, the native dependency test |
| `tests/runtime/test_pjrt.py` | End to end through jax, against the CPU backend |
| `tests/runtime/test_pjrt_native.py` | Builds and runs `tests/native/pjrt_dependencies.cpp` |
| `tests/codegen/test_stablehlo.py` | Generated programs, no GPU needed |
| `examples/training_steps.py` | The SBI, LLM and SAM2-like training steps |
| `examples/pjrt_m1_benchmark.py` | The M1 overhead, bandwidth and compile-time artifact |
| `examples/pjrt_sbi_audit.py` | The SBI Metal/CPU numerical audit |
| `examples/pjrt_m2_profile.py` | M2: wall, host, encode and GPU time per SBI step |

## How it was built

1. **What JAX hands a new platform.** Registering a stub backend factory
   (`xla_bridge.register_backend_factory("metal", ...)`) and lowering with
   `jit(f).trace(*args).lower(lowering_platforms=("metal",))` shows the
   StableHLO and the failures without any plugin; findings are in
   `notes/metal-compute-stack.md`, "What JAX hands a new platform".
2. **Pinning the C API.** jaxlib ships only the XLA FFI headers, not
   `pjrt_c_api.h`. CMake downloads it from the XLA commit jax 0.12.0.1 pins
   in its `MODULE.bazel` (`d867969800d4a8fd045505eabc65e1e66f30794f`,
   PJRT C API 0.116) and checks its SHA-256. A jaxlib upgrade means updating
   the commit and the hash; jaxlib requires the same major version (0) and
   tolerates a lagging minor one.
3. **Skeleton.** One client, one device, one `device` memory, buffers in
   shared MTLBuffers, with host data copied in (strided arrays copy one
   dense block at a time). The 0.116 header defines `PJRT_Error` and
   `PJRT_Memory` as structs holding a function table that jaxlib calls
   directly, so the plugin subclasses them and fills the tables. jaxlib
   calls these slots without a null check, and each one crashed until
   implemented: `PJRT_Client_TopologyDescription`,
   `PJRT_Error_ForEachPayload`, `PJRT_TopologyDescription_Fingerprint`
   (compilation cache keys, once a CPU compile has run), and
   `PJRT_LoadedExecutable_GetDeviceAssignment`. Its `DeviceAssignmentProto`
   for one device is 8 bytes, encoded by hand to avoid protobuf.
4. **Registration.** `[project.entry-points.jax_plugins] metal =
   "palladium._pjrt"`; `initialize` registers the plugin only when
   JAX_PLATFORMS names `metal`. Otherwise every jax process would load it and
   print jax's "experimental platform" warning while most programs cannot run.
5. **Compiling in Python.** The C++ route needs LLVM, MLIR and StableHLO
   built at StableHLO's pinned LLVM commit (Homebrew's LLVM does not match),
   an hour or more per machine and a pin per jaxlib bump. Instead,
   `PJRT_Client_Compile` calls a ctypes callback registered by
   `palladium._pjrt`. Programs arrive as VHLO portable artifacts;
   `jaxlib.mlir.dialects.stablehlo.deserialize_portable_artifact` reads
   them with jaxlib's own bindings, so the versions always match.
   `palladium.stablehlo.compile_program` returns a `Program`, and `_pjrt`
   replays it through the builder ABI:
   - `pp_slot(kind, type, dims, data)`: a parameter (numbered in order), a
     fresh buffer per execution, or a constant uploaded once.
   - `pp_kernel(msl, function, math_mode, slots, grid, threadgroup)`:
     compiles the MSL now; an empty threadgroup lets the pipeline choose.
   - `pp_output(slot)`, `pp_alias(parameter, output)` and `pp_error(code, message)`.

   `UnsupportedProgram` becomes UNIMPLEMENTED; any other exception becomes
   INTERNAL with its traceback. No exception may unwind into the plugin.
6. **Code generation.** Outputs are grouped by shape; each group is one
   kernel with one thread per output element. A thread derives its
   coordinates from its flat index, and every value is generated at the
   coordinates it is read at, memoized per (value, coordinates). Index maps
   therefore cost index arithmetic and never a copy, and a reshape read at
   the kernel's own coordinates reads at the flat index. Names and literals
   are inlined instead of getting a variable. Scalar ops reuse palladium's
   templates (`emit.numeric.format_scalar`, keyed by JAX primitive name
   through `_ELEMENTWISE`); `_EXTRA` adds those the emitter lacks (sign,
   remainder, rounding, popcount, clz, not). `func.call` and
   `stablehlo.composite` (jax wraps `erf` in one) are inlined. Non-splat
   constants become constant slots. Math mode is FAST, palladium's default.
7. **Execution.** `Execute` binds slots to buffers, encodes every launch
   into one command buffer, commits it with a completion handler and
   returns. Passthrough/repeated outputs and donation copybacks use blit
   commands after the compute launches, in two passes: snapshots into new
   outputs, then copybacks into donated storage. No copy in a pass writes a
   buffer another one uses, so each pass is one blit encoder whatever order
   Metal runs its copies in. The handler retains program and
   storage owners, then sets the execution's `Completion`, which is each
   output's `ready`. GPU copies ensure subsequent commands cannot race a
   delayed host completion callback.
8. **Buffer reuse.** Buffers come from a pool keyed by requested size (up
   to 1 GiB), since a fresh 16 MB MTLBuffer is zeroed on allocation. A
   released buffer returns to the pool when its last storage owner disappears.
   Each dispatch holds independent owners of all input, intermediate and output
   storage until completion; it does not mutate a shared last-use pointer.
   `test_executions_run_ahead_of_the_gpu` fails on every element when the
   deferral is removed. Without a bound on executions in flight, a caller
   running ahead allocated a new output per queued call and large calls got
   2× slower; at most 16 are now in flight (4 cost small calls 70%).

Measured on the M2, `tanh(2x + 1) * x` per call:

| Elements | Pipelined | Blocking each call | CPU backend |
|---|---:|---:|---:|
| 1,024 | 29 µs | 151 µs | 3 µs |
| 4M | 464 µs | 635 µs | 899 µs |

A blocking call costs one command buffer round trip (about 150 µs). A
pipelined small call is host time: jax's dispatch through the C API plus
encoding. A large call moves 33.5 MB at about 72 GB/s against the 91 GB/s
copy ceiling.

## Invariants

- **Ordering.** All work goes to one MTLCommandQueue, so command buffers
  run in commit order, and Metal tracks hazards on the shared-storage
  buffers across them. An execution may therefore read buffers whose
  producers are still queued, with no host wait. Within one command buffer
  the compute encoder dispatches serially, so a launch sees the writes of
  the launches before it; multi-kernel programs rely on this.
- **Host access.** Host reads wait for `ready` (`wait_ready`). Host writes
  go only into buffers fresh from the pool, which never holds a buffer the
  GPU may still use.
- **Lifetimes.** Committed command buffers retain the MTLBuffers they bind,
  and the completion handler holds the `Program`, so jax may delete inputs,
  outputs and executables while work is in flight. The pool is leaked on
  purpose: buffers released during static destruction still find it.
- **Errors.** Unsupported input fails loudly with the op's name, never by
  computing something else.

## Next, with the main considerations

In order of what M2's SBI step needs first.

1. **Multi-kernel programs and `reduce` (single-result slice done).**
   Dependencies are scheduled before consumers, with each reduction computed
   once even when returned or consumed multiple times. Producers fuse into
   input reads inside the reduction loop; consumers read fresh intermediate
   slots, which Execute already keeps out of the pool until completion.
   Kernels use 128 threads per output, SIMD collectives and four threadgroup
   partials. When there are fewer than 128 outputs and more than 4096 reduced
   elements, independent 4096-element chunks feed a second reduction launch.
   The same coordinate machinery handles arbitrary axes; column reductions
   are correct but still need coalesced tiling.

   The compiler validates the reducer region: exactly one binary add, max,
   min, and or or returning its result. Other single-result bodies
   fail explicitly. Variadic scalar elementwise bodies have a separate
   sequential-fold implementation, including argmax/argmin. Half/bfloat accumulators and partial buffers use float32.
   Reduction launches use SAFE math; min/max propagate NaNs and resolve
   signed-zero ties explicitly around Metal's SIMD collectives. Elementwise
   launches retain FAST math. The supplied initializer is combined once in
   the final pass, and is returned for empty reductions. Non-identity
   initializers need not match XLA's reduction schedule, which can insert
   them multiple times ([StableHLO semantics](https://openxla.org/stablehlo/spec#reduce)).

   Reductions over at most 16 elements fold into their consumers instead
   (see the CNF section below). Remaining: parallel variadic reductions,
   coalesced column reductions, one thread per output for rows of 17 to
   about 128 elements, and epilogue fusion.
2. **`concatenate`, `pad`, dynamic slices (done).** `_Kernel.choose` emits
   each branch's entire producer chain inside its guard. Value and reshape
   memoization is scoped to that branch, with unique temporary names across
   scopes. Empty operands are never read; padding supports cropping and
   interior holes. Dynamic offsets are scalar buffer reads, clamped to the
   operand minus slice/update size, preserving unsigned values above INT_MAX.
   CPU differential tests cover nesting, empty and scalar tensors, negative
   and oversized offsets, and integer, boolean and floating operands.
3. **64-bit integers (RNG slice done).** `long` and `ulong` live only in the
   PJRT compiler's type map; the shared Pallas `CTYPES` contract is unchanged.
   Wide literals never pass through float32. Wrapping arithmetic uses unsigned
   intermediates, shifts guard counts at/above 64 (arithmetic right shifts
   sign-extend), and divide/remainder handle zero and signed-min/-1 explicitly.
   CPU tests cover extrema, values above 2^53, wrapping, shifts, population
   counts, conversions and bitcasts. RNG bits (uint32/uint64), split and fold-in
   match exactly; float32 uniform/normal match at rtol 2e-5, atol 1e-6.
   Wide reductions and 64-bit loop counters remain unsupported.
4. **`while` without host round trips (counted slice done).** The compiler
   proves the constant-start, LT-bound, unit-increment pattern, including
   invariant bounds and hoisted constants in the PJRT artifact. It compiles
   a body template once and rebinds its launches per iteration. Changed
   carries have two output banks; even carry permutations are materialized,
   so outputs cannot overwrite inputs still needed by that iteration.
   Invariant carries retain their original slots. Body temporaries are reused
   between iterations. Nested counted loops, reductions inside loops,
   forward/reverse scans, zero trips and queued executions have CPU comparisons.
   A per-graph 100,000-launch limit rejects excessive replay expansion.
   `pp_kernel` caches libraries/pipelines per executable by source, entry point
   and math mode, so replayed launches do not recompile identical shaders.

   Dynamic conditions and other induction patterns remain UNIMPLEMENTED.
   They would need host condition reads or GPU-driven dispatch. Pure `case`
   now lowers to guarded per-result selection: scalar/index-map producers stay
   inside each branch, while reductions, dots and counted loops are computed
   eagerly. All branches are validated; unsupported/effectful ops still fail,
   including in an untaken branch. Out-of-range raw StableHLO indices select
   the last branch; JAX switch supplies its own clamp. Selective dispatch of
   expensive branches remains an optimization.

   Replay binds induction variables (integer scalar carries with a constant
   start and step, including the proven counter and `fori_loop`'s index) to
   one constant array per variable, at a byte offset per iteration, instead
   of computing them with a kernel. A carry updated only by a
   `dynamic_update_slice` of itself, and read nowhere else in the body, lives
   in one buffer per execution: each iteration writes just its update, fused
   into the body's kernel of the same element count. This covers `lax.scan`
   outputs. Fused kernels group outputs by element count, not shape.
5. **`dot_general` (floating slice done).** Normalize dimension numbers to
   `(batch, M, K)` and `(batch, K, N)`, preserving the specified pairing/order
   of batch and contracting axes. Canonical buffers bind directly; packing
   kernels fuse transposes, reshapes and elementwise producers when necessary.
   `_stablehlo_gemm.py` traces a tiled Pallas matmul and calls the existing
   cooperative emitter, with batches as a grid axis. Aligned float32 products
   use `simdgroup_matrix`; output/K tails and f16/bf16 use Metal 4 TensorOps.
   Both paths accumulate in float32. Matching float operands are supported,
   including widening the result to float32 by converting packed operands.
   Empty contraction dimensions fill zero without reading either input.
   Integer/complex products and explicit algorithm attributes are rejected.
   Generated sources are cached by canonical shape/dtype; native pipelines
   retain the existing per-executable cache. Remaining: epilogue fusion,
   avoiding transposed packing where possible, and tuning small/tail products.

   Accuracy at large K (M2, 2026-11-02): each output accumulates all K
   products in one float32 register. For uniform(-1, 1) operands of shape
   64×8192 by 8192×64, the Metal result equals, bit for bit, a host
   simulation of a sequential per-product FMA chain. Its maximum error
   against float64 is 3.05e-4, versus 5.56e-5 for XLA:CPU; the factor grows
   with K (about 3× at K=2048, 13× at K=16384). Simulated split-K with 4
   chunks gives 4.2e-5, below CPU. Split-K would also occupy more of the GPU
   when M×N has few tiles (this product runs one threadgroup). This is not an
   M1 gate: the SBI step contracts at most 128 elements and the tests cover
   K ≤ 128. It matters for M3's GEMMs.
6. **Pallas kernels inside `jit` (done for `pallas_call`, `metal_call` and
   `plmtl`).** On `metal`, `pallas_call` lowers through
   `palladium.ffi.pallas_lowering` (registered next to the jax-mps rule in
   `palladium/mps.py`) to the same typed-FFI custom call, `palladium_dispatch`,
   that `metal_call` and `plmtl.kernel` emit on every platform. Its
   `mhlo.backend_config` dictionary carries the MSL, function name, thread
   grid, threadgroup, math mode, and the `batch_size`/`elem_strides` of
   `metal_call`'s vmap rule. `stablehlo.py` turns the call into one `Launch`
   per batch element, binding inputs then results, offset by the strides.
   An aliased result starts as a copy of its input, so elements the kernel
   does not write keep the input's values. Other custom-call targets and
   non-row-major layouts are rejected. Remaining: a `metal` rule for
   `pl.kernel`'s `mpmd_map` outside `plmtl`; avoiding the copy for an aliased
   input that is not read again.
7. **Donation (correctness implemented).** `_pjrt.initialize` appends `metal`
   to JAX's private donation-platform list. Argument `tf.aliasing_output`
   attributes pass through the compiler to native execution. Outputs compute
   into fresh storage, then copy back into donated storage after all reads and
   passthrough snapshots. This handles permutations and repeated outputs.
   PJRT runtime non-donatable indices override the compile-time aliases.
   Remaining: avoid temporary allocation/copyback when in-place execution is safe.
8. **Host time per call.** 29 µs pipelined against 3 µs on CPU. Profile
   the split between jax's C API dispatch and Execute (command buffer and
   encoder creation, per-launch vectors, `std::function` and `shared_ptr`
   churn), and compare against jax-mps on the same machine in its own
   environment (`uv sync --group mps --no-group latest-jax`, since jax-mps
   pins jax 0.11).
9. **Smaller gaps.**
   - Dependency failures now propagate through output readiness and execution
     events. GPU work may already be queued, but its results cannot report
     success when an input failed. Readiness joins use callbacks, without
     blocking dispatch or Metal callback threads.
   - Concurrent read-only uses of shared buffers are covered. Concurrently
     deleting or donating the same Python array from another thread is not
     part of this guarantee.
   - Libraries/pipelines are cached per executable; sharing across executables
     remains future work.
   - Compile time: the first call of a small program takes about 90 ms.
10. **Dropping the JAX_PLATFORMS gate** once the success criteria below hold.

### Reduction measurements, 2026-11-01

Apple M1 Pro, jax/jaxlib 0.12.0.1, float32 ones of shape (1024, 4096),
resident inputs. Wall time per call: minimum of five batches of 100 compiled
calls, blocking only on the last output in each batch. Compile + first call
includes lowering, compilation and one completed execution; Metal's disk
shader cache may be warm. These are host timings, not GPU timestamps.

| Sum axis | Metal pipelined | CPU pipelined | Metal compile + first call |
|---|---:|---:|---:|
| Last (rows) | 63.9 µs | 150.9 µs | 86.3 ms |
| First (columns) | 410.3 µs | 248.9 µs | 21.0 ms |
| All | 71.4 µs | 140.0 µs | 25.7 ms |

The column path's strided reads remain slower than CPU. These measurements
do not establish M1's overhead target against jax-mps.

### Counted-loop checkpoint, 2026-11-01

`examples/training_steps.py` now exposes `sbi_trajectory(theta)`, the same
200-step RK4 ensemble used by `sbi_step`. With 512 parameter vectors sampled
on the host in [0.5, 1.5], the complete trajectory and concatenated
mean/std/final-state summary match CPU within rtol 2e-5, atol 1e-5 in a PJRT
subprocess test. This isolates the simulation from the remaining RNG and
MLP gaps; it is not the full SBI training-step milestone.

On the Apple M1 Pro, the trajectory-plus-summary program's first completed
call took 91 ms on Metal and 56 ms on CPU (shader cache may be warm).
Pipelined wall time per call was 2.59 ms on Metal versus 0.345 ms on CPU,
taking the minimum of five batches of ten calls, blocking on each batch's
last result. The replay implementation is correct but does not yet beat CPU.
Full trajectory copies, multiple dispatches per iteration and strided summary
reductions are candidates for profiling and tuning in M2.

### Dot checkpoint, 2026-11-01

The SBI dense loss is exposed as `sbi_loss` so its forward/backward GEMMs and
AdamW updates can run independently of RNG. With host-generated fixed data,
ten updates match CPU at rtol 1e-4, atol 2e-6 for parameters, optimizer state
and losses. The observed maximum parameter difference was 1.31e-6; loss
differences stayed below 5e-7. Tiny gradients amplify float32 summation
differences under Adam, hence the explicit absolute tolerance. This does not
claim the full ten-step SBI success criterion: simulation and RNG are excluded.

Apple M1 Pro, float32 resident operands of ones, `(M,N,K)` below. Pipelined
wall time is the minimum of five batches of 30 calls, waiting only on the
last result. First call includes tracing, compilation and completion; shader
caches may be warm. These are host timings, not GPU timestamps.

| M, N, K | Metal pipelined | CPU pipelined | Metal first call |
|---|---:|---:|---:|
| 512, 128, 128 | 28.5 µs | 187.9 µs | 246 ms |
| 1024, 1024, 1024 | 571.9 µs | 3281.7 µs | 115 ms |

Both use the reused SIMD-group GEMM. Packing, tail and whole-training costs
are not represented by these aligned-product measurements. At this checkpoint,
SBI reached missing uint64 RNG operations, subsequently implemented below.

### Full SBI checkpoint, 2026-11-01

The actual `sbi_step` now executes RNG, the 200-step RK4 simulation, summary
statistics, the MDN loss/gradients and AdamW together on Metal. A permanent
PJRT test initializes on CPU, folds in a distinct key for each of ten updates,
and compares every updated parameter, optimizer-state leaf and loss against
CPU. It passes with rtol 1e-4 and **atol 2e-6**; measured maximum state
difference was 1.22e-6 and maximum loss difference was 9.54e-7. This includes
an absolute tolerance for near-zero parameters; it does not establish literal
relative-only agreement there.

On the Apple M1 Pro, the first completed Metal update took 0.82 s during
validation, below the five-second compilation target. A later process with
warm shader caches took 0.27 s. Pipelined full-step calls on fixed resident
inputs measured 2.83 ms on Metal versus 1.47 ms on CPU (minimum of five
batches of ten calls, waiting only on the last result). The throughput probe
reuses initial state; the correctness test chains ten updates with new keys.
The full step is correct at the stated tolerance, but does not yet beat CPU.

The operation groups and common API/concurrency cases now have tests,
including argmax, gather/scatter-add and donation. M1 is still incomplete.
Overhead versus jax-mps and bandwidth targets on M2 and M4 Pro have not been established. M2 still
needs profiling/tuning of replay, trajectory updates and reductions.

### M1 completion assessment, after indexing and lifetime support (2026-11-01)

The milestone is not complete just because SBI runs or the test suite passes.
The remaining acceptance gates, against the criteria below, are:

| Gate | Evidence now | Remaining work |
|---|---|---|
| Operation groups | All listed groups have differential tests, now including argmax/argmin and JAX gather/scatter-add in f32/f16/bf16 | Scoped coverage, not every StableHLO dtype/body; partially out-of-bounds low-level scatter windows differ from CPU as described below |
| Ten SBI updates | Full program runs with new keys; every parameter, optimizer-state array and loss within normwise rtol 1e-4 (worst 3.6e-5 on M2) | Resolved 2026-11-02: the criterion is now normwise per array (see "Agreement rule") |
| Compile time | First completed SBI step 0.256 s on M1 Pro, 2.51 s on M2, 1.64 s on M4 Pro; persistent JAX cache disabled, driver cache unchanged | Met on both target machines |
| Common API/lifetimes | `lower().compile()`, analysis APIs, donation/storage reuse, deletion after dispatch, and four-thread shared-input dispatch have tests; shared last-use mutation removed | Further stress/profiling on target machines; no known remaining implementation gap in the listed common API cases |
| Memory analysis contract | PJRT returns UNIMPLEMENTED; JAX 0.12 returns `None`, covered by the API test | Resolved: the criterion now describes the public JAX behavior |
| Dispatch overhead/bandwidth | M1 Pro: 23.1 µs small-call median, 89.7% of copy. M2: 26.2 µs versus 175.7 µs for jax-mps, 96.4% of copy. M4 Pro: 10.2 µs versus 126.4 µs for jax-mps, 94.0% of copy | Met on both target machines |

**M1 is complete as of 2026-11-02.** Every gate has evidence: the operation,
SBI and API tests pass on the M1 Pro, the M2 and the M4 Pro under the normwise agreement
rule (resolved 2026-11-02, see "Agreement rule"), and the performance and
compile-time artifacts for both target machines are below. The
current full-SBI slowdown is a warning for M2, **not an M1 gate**:
M1's performance requirements are the stated dispatch and bandwidth targets.
Dynamic while, convolution, LLM/SAM training, and Pallas custom-call integration
are not reasons to hold M1 open beyond its listed criteria.

### Indexing compatibility and validation

Gather fuses source/index expressions into reads. Scatter initializes fresh
output storage and uses atomic adds for f32/i32/u32. For f16/bf16 it uses one
atomic uint word per element, compare/exchange with rounding after each
successful update, then unpacks; odd-sized outputs never access a neighboring
halfword outside their allocation. Duplicate floating updates have an
implementation-dependent order, so general floating results are not promised
bitwise identical to CPU.

There is one observed low-level compatibility difference: a `lax.scatter_add`
window of length 4 starting at -1 or 5 in a length-7 operand is discarded as a
whole by XLA CPU. Our lowering retains its in-bounds elements, following the
[StableHLO scatter specification](https://openxla.org/stablehlo/spec#scatter).
A scalar reference tests this behavior separately. The common `x[idx]` and
`x.at[idx].add` cases in the M1 census match CPU; do not describe arbitrary
out-of-bounds window scatter as CPU-compatible.

### Reproducing the performance gates

Run in fresh processes on **each** target machine, using separate environments
for current JAX and jax-mps (which pins JAX 0.11). Never install jax-mps into
the project's `.venv`: its plugin registers in every jax process, with priority
500 over CPU's 0, and becomes the default backend. Palladium registers
`metal_call` for CPU only, so most of the suite then fails inside jax-mps
("Output count mismatch"). `uv sync` removes it again. Do not run other GPU
workloads concurrently:

```sh
JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_m1_benchmark.py --platform metal --output metal.json
# In the jax-mps environment:
JAX_PLATFORMS=mps,cpu python examples/pjrt_m1_benchmark.py --platform mps --output mps.json
```

The JSON records versions, machine, cache policy, timing samples, SBI compile
plus first completed call, and median small-call microseconds. Metal also
records a 256 MiB elementwise wall-time bandwidth and the same copy kernel/GPU
time used by `bench_copy_bandwidth`. Compare `small_median_us` across the two
artifacts; require `elementwise_fraction_of_copy >= 0.8` and SBI first call
under 5 seconds. The correctness suite remains a separate prerequisite.
JAX persistent caching is disabled; the Metal driver's cache is not cleared.

Schema version 2 also records Git/jj revisions and working-copy dirtiness.
With `--output`, the artifact is atomically checkpointed before and after each
section: an unsupported SBI step does not erase latency/bandwidth measurements.
Failed sections contain their exception and the process exits nonzero; an
interrupted section remains marked `running`. Successful sections remain usable.
Timing batches are calibrated to at least `--sample-seconds` (default 0.2 s)
before collecting five samples, with the actual calls per sample recorded.
Large elementwise calls synchronize individually to bound memory use; small
calls remain pipelined. Copy samples average GPU time across repeated calls.

[M1 Pro artifact, 2026-11-01](metal-m1-pro-2026-11-01.json): 23.1 µs median
small-call latency, 127.0 GB/s elementwise versus 141.6 GB/s copy (89.7%),
and 0.256 s SBI compile plus first execution. These are five-sample medians
except for the single first-call measurement; the artifact retains every
sample. They do not establish either target-machine gate or a jax-mps speedup.
This artifact predates schema version 2: it has no revision, sections or
sample calibration, and used fixed batches of 1000 small calls. The M1 Pro is
not a target machine, so it was not regenerated.

[M2 artifact, 2026-11-02](metal-m2-2026-11-02.json), schema 2, clean working
copy at the Codex audit commit `972ad13`: 26.2 µs median small-call latency,
83.5 GB/s elementwise versus 86.6 GB/s copy (96.4%), and 2.51 s SBI compile
plus first execution. The Metal driver's shader cache was probably cold for
the SBI kernels.

[jax-mps M2 artifact, 2026-11-02](mps-m2-2026-11-02.json): the published
jax-mps 0.11.1 wheel with jax/jaxlib 0.11.2, in a separate virtual environment
on the same machine, same script and session. 175.7 µs median small-call
latency, 75.3 GB/s elementwise, 3.96 s SBI compile plus first execution, and
the same first loss as Metal to six digits. The Metal plugin's small call is
6.7× cheaper. jax-mps' latency is close to one blocking command-buffer round
trip; whether it waits per call was not checked. A jax-mps build from source
needs MLX installed and was not used. Its artifact's `revision` describes
this repository (with uncommitted note edits), not jax-mps.

On the M2, all three performance gates hold: overhead below jax-mps,
elementwise bandwidth at 96.4% of copy, and SBI compile time under 5 s.

[M4 Pro artifact](metal-m4-pro-2026-11-02.json) and
[jax-mps M4 Pro artifact](mps-m4-pro-2026-11-02.json), 2026-11-02, macOS 27.2:
the Metal plugin's small call costs 10.2 µs median versus 126.4 µs for
jax-mps (12×). Elementwise reaches 218.6 GB/s against a 232.4 GB/s copy
(94.0%); jax-mps also reaches 218.6 GB/s, so both are at the memory ceiling.
SBI compile plus first execution takes 1.64 s on Metal and 2.54 s on
jax-mps; both report the same first loss. All three M4 Pro gates hold.

Provenance differs from the M2 comparison. Both M4 Pro runs used the
project's main virtual environment with jax/jaxlib 0.12.0.1, so jax-mps ran
outside its declared `jax<0.12` range, and the artifact does not record
which jax-mps build was installed. The working copy was dirty with a
`uv.lock` refresh and a uv pre-commit hook bump, at revision `abe1975`
(the normwise-agreement change). Neither touches the measured code. With a
12× margin, the jax-mps configuration cannot change the overhead verdict.
The full test suite passes on the M4 Pro once jax-mps is removed from the
main environment (see the warning under "Reproducing the performance gates").
Validation after these changes: 504 tests passed, 2 skipped, 4 fuzz tests
deselected; lint and type checks on the changed Python files passed.

### SBI numerical audit, 2026-11-02

Run `JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_sbi_audit.py --output audit.json`.
The [M1 Pro audit artifact](metal-sbi-audit-m1-pro-2026-11-02.json) records seed 0,
ten updates with new keys, per-leaf and per-step errors, and isolated diagnostics.
Nonfinite values fail the audit. `required_atol` is the maximum positive
`abs(metal - cpu) - 1e-4 * abs(cpu)`, not simply the maximum absolute error.

| Comparison | Relative-only failures / elements | Required additional atol | Maximum absolute error |
|---|---:|---:|---:|
| Ten chained updates, parameters + optimizer + loss | 4,945 / 687,380 | 5.5341e-7 | 1.2117e-6 |
| Each update starts from identical CPU state | 1,060 / 687,380 | 1.4464e-8 | 8.4937e-7 |
| Chained losses alone | 0 / 10 | 0 | 9.5368e-7 |

The largest tolerance excess is parameter `wa[90,1]` at step 4:
Metal -0.000120491961 versus CPU -0.000121057470. Large relative errors near
zero do not indicate similarly large absolute differences. All comparisons
pass the existing rtol 1e-4 plus atol 2e-6 check; the original relative-only
criterion is **not satisfied**.

The first-step diagnostic has exactly matching sampled theta. Trajectory and
features differ by at most 1.9074e-6 and pass relative-only tolerance. Gradients
on identical CPU data differ by at most 1.0478e-9; AdamW on identical gradients
changes parameters by at most 3.7253e-9. Together with the CPU-anchored updates,
this is evidence of accumulated numerical drift, not proof that every source
of error has been isolated.

The audit recommended adopting this mixed tolerance. It was not adopted; see
"Agreement rule" below.
The memory-analysis wording is resolved separately: public JAX returns `None`
when native compiled memory statistics report UNIMPLEMENTED; this is tested.

### Agreement rule, 2026-11-02

The mixed tolerance (rtol 1e-4 plus atol 2e-6) does not check most of the
optimizer state. After ten chained SBI steps on the M2, Adam's second
moments peak between 1.5e-11 (`wa`) and 2.3e-6 (`w1`): an all-zero `v` would
pass. The first moment of `wa` peaks at 2.3e-5, so the allowance is about 10%.
The criteria now use normwise relative error per array, `max|metal − cpu| ≤
rtol · max|cpu|`: the standard scale-invariant measure (Higham, *Accuracy and
Stability of Numerical Algorithms*, ch. 1), which tolerates near-zero entries
of an array only as much as the array's own rounding does. The rtol values
are unchanged: 1e-4 for SBI, 2e-5 for float32 operations, 1e-2 for f16/bf16.
These are budgets, not error estimates.

Measured on the M2 (normwise error per array, maximum over ten chained
steps): the worst is 3.6e-5 (`v['wa']`), parameters 8e-7 to 1.8e-5, the loss
4.6e-7.
Every operation-group test passes under the same rule.

Rounding floor. Perturbing CPU's initial parameters and optimizer state by
one unit roundoff (u = 2⁻²⁴, five seeds) and repeating the ten CPU steps
moves the arrays by 2–66 u normwise: the ten-step map amplifies initial
rounding only mildly. Metal's errors are 0.3–11× this floor for parameters,
3–24× for first moments and 23–89× for second moments. So Metal's per-step
arithmetic differs from CPU's by tens of ulps, more than reordered float32
summation alone would explain. Untested candidates: FAST math in elementwise
kernels (`exp`, `log`, `tanh`, `erf`) and the 200-step simulation, whose
trajectory already differs by 1.9e-6 (about 32 u). A rule derived from the
floor would be near 1e-5 and would fail the moments. Locating this gap is M2
work: rerun with SAFE elementwise math first.

## M2 baseline: where an SBI step's time goes (2026-11-02, M2)

M2 requires GPU time to be most of a step's wall time. The plugin counts
executions, command buffers and launches; host seconds inside Execute
(encoding; waiting for the in-flight bound); buffer-pool hits, new MTLBuffers
and their creation time; and GPU time from each command buffer's
`GPUStartTime`/`GPUEndTime`, as the sum and the exact union (busy) of the
intervals. `palladium._pjrt.stats(reset=...)` reads them;
`examples/pjrt_m2_profile.py` reports them per step for the SBI step and its
two halves. Completion handlers arrive out of start order, so the union is
computed when read; an earlier incremental merge assumed start order and
reported about half the true busy time.

The simulation dominated the original step: 603 of 657 launches, three per
RK4 iteration, each over only 512×2 elements. One launch computed the loop
counter, whose values are known when replaying; a second rewrote the whole
200×512×2 trajectory (204,800 threads) to append one row. Binding induction
variables to constant arrays, updating scan outputs in place, and grouping
kernels by element count leave one 1,024-thread launch per iteration:

| SBI step on the M2, pipelined | Before | After |
|---|---:|---:|
| Full step wall | 2.145 ms | 1.005 ms |
| Simulation wall | 1.805 ms | 0.650 ms |
| Launches per step | 657 | 256 |
| Host encoding per step | 0.738 ms | 0.274 ms |

After the change, per step (medians of five calibrated batches, fixed
resident inputs):

| Workload, pipelined | Wall | Host | Encode | GPU busy | GPU/wall | Launches |
|---|---:|---:|---:|---:|---:|---:|
| Full SBI step | 1.005 ms | 0.327 ms | 0.274 ms | 0.989 ms | 98.4% | 256 |
| Simulation (200 RK4 steps) | 0.650 ms | 0.239 ms | 0.224 ms | 0.639 ms | 98.2% | 202 |
| Dense update (MDN loss, grads, AdamW) | 0.302 ms | 0.057 ms | 0.039 ms | 0.290 ms | 95.8% | 48 |

**On the M2, pipelined SBI steps meet M2's GPU-time target**: the GPU is busy
98% of the wall time, and a Metal System Trace shows no gaps between command
buffers. The host is not the limit: it spends most of each call waiting for
the 16-execution in-flight bound. The step takes 1.0 ms against 1.32 ms on
XLA:CPU and 18.66 ms on jax-mps 0.11.1 (min of five batches of 20). The GPU
busy time is a command buffer's whole residency, including any time it
waits behind an overlapping predecessor; it is not pure kernel time.

Buffer allocation is negligible after warm-up: over 1,000 steady-state steps,
15 new MTLBuffers took 0.8 ms in total (70 pool requests per step). Warm-up
creates about 90 MiB, one set per execution in flight, in about 3 ms. In a
`sample` of the same run, the pool-miss path (Execute → `make_buffer` →
`acquire` → `Buffer::Buffer` → `AGXBuffer initWithDevice`) has 2 samples.
But 23 of the 65 samples inside Execute wait on the pool's mutex: each step
locks it about 70 times while Metal's completion threads lock it to return
the previous steps' buffers. Not on the critical path while the GPU is the
bottleneck; acquiring and releasing an execution's buffers under one lock
would remove it for small, host-bound calls.

Profiling with symbols: the installed plugin is stripped by default.
`uv sync --reinstall-package palladium -C install.strip=false` keeps its
symbols for Instruments or `sample`; a later plain `uv sync` strips it again.
The unstripped library in `build/<wheel tag>` also resolves offsets with
`nm -n`.

### Pallas kernels and the CNF premise check (2026-11-02, M2)

M2's premise test from `notes/metal-compute-stack.md`: the CNF update of
`examples/cnf_density.py` (256 points, width 4, 16 RK4 steps, checkpoint
interval 4, Palladium's forward and adjoint kernels, Adam). jax-mps with
`JAX_MPS_ASYNC_DISPATCH=1` took 149 µs per pipelined update, 61 µs of it GPU
time (xla-metal-replay `notes/plan.md`, 2026-10-28, which places it on the M1
machine; not rerun here, since PyPI jax-mps has no `palladium.dispatch`
handler).
`examples/pjrt_m2_profile.py` on the M2:

| Workload, pipelined | Wall | Host | Encode | GPU busy | GPU/wall | Launches |
|---|---:|---:|---:|---:|---:|---:|
| CNF update, Palladium kernels (`pallas_call`) | 0.057 ms | 0.016 ms | 0.009 ms | 0.051 ms | 90.1% | 9 |
| CNF update, plain JAX (before the fixes below) | 1.863 ms | 0.856 ms | 0.822 ms | 1.843 ms | 98.9% | 705 |
| CNF update, plain JAX (after) | 0.551 ms | 0.273 ms | 0.263 ms | 0.541 ms | 98.2% | 464 |

For comparison on the M2 (min of five batches of 200): XLA:CPU takes 0.91 ms
for the plain-JAX update, and the same kernels through `metal_call` on the
CPU platform 0.49–0.52 ms, since every FFI call waits for its command buffer.
**The premise holds**: owning execution leaves about 6 µs of host time
outside the GPU's 52 µs, where jax-mps spent about 88 µs outside 61 µs. All
three Metal paths match CPU over ten updates at normwise rtol 1e-4
(`test_cnf_training_matches_cpu_over_ten_steps`).

Plain JAX on `metal` first lost to XLA:CPU (1.86 ms against 0.91 ms), as
the replay estimate predicted (2.58 ms there). Two fixes:

- **Metal's 31-buffer limit.** One fused value in the backward loop read 50
  buffers. `_compile_graph` now splits fusions whose outputs and reads exceed
  31 buffers into several kernels, and stores the largest operand that fits
  in one kernel when a single value reads more (descending into operands
  that are too large themselves).
- **Short reductions.** The gap was kernel work, not launch overhead: a
  trivial launch costs 0.70 µs of GPU time (705 launches ≈ 0.49 ms), and
  the update's time grew 12× for 16× the batch with the same launches. 517
  of the 705 launches were reductions over axes of 1, 2 or 4 elements, each
  with 128 threads per output; one costs 4–6 µs. Reductions over at most 16
  elements now fold into the kernel that reads them, unrolled from the
  initial value as XLA's sequential reduce. A kernel that folds a float
  maximum or minimum runs RELAXED: under FAST, `x * inf - inf` lost its NaN
  and maxima of negative zeros their sign; SAFE kept both but slowed the
  SBI dense update by 10% through its `exp` and `log`; RELAXED keeps both at
  FAST's speed (M2). Folding recomputes a short reduction in every thread
  that reads it, which measured faster than storing it in both workloads.
  It folds only at the kernel body's coordinates, though: guarded branches
  name their shifted coordinates afresh, so a fold read inside branches was
  regenerated, producers and all, in each one. Folding everywhere made one
  plain-JAX CNF kernel 9.2 MB of MSL and its compile 18.4 s (from 0.08 s);
  a short reduction read at branch coordinates is now stored once by a
  one-thread-per-output kernel and read from its buffer (compile 0.11 s,
  largest kernel 16 KB; `test_plain_jax_cnf_kernels_stay_small`).

| Plain-JAX CNF update, M2 | Metal before | Metal after | XLA:CPU |
|---|---:|---:|---:|
| n = 256 | 1.83 ms, 705 launches | 0.56 ms, 464 launches | 0.91 ms |
| n = 1,024 | 5.76 ms | 0.86 ms | 8.12 ms |
| n = 4,096 | 22.0 ms | 2.40 ms | 30.3 ms |

The SBI step is unchanged within noise (0.62 ms, 249 launches); its dense
update dropped from 48 to 41 launches (0.30 ms).

### Recomputation in guarded branches

The RK4 right-hand side stacks each stage's two components with
`concatenate`, and the next stage slices them apart again. Guarded branches
used to keep their memoized values private, and equal coordinates had several
spellings (`c0`, `(i / 2u)`, `((i / 2u))`, `i / 2u % 512u`), so each stage
recomputed the stages before it. The loop kernel did 4,032 float operations
and 1,795 buffer reads per thread for a body of 46 arithmetic ops.

Now each guarded branch is a scope of its own names: the coordinates its guard
shifts and the temporaries it defines. A value whose coordinates use none of
them is computed once before the branch chain. That is safe for `concatenate`
and `dynamic_update_slice`: unshifted coordinates index their operands within
bounds, since those match the output in every unshifted dimension. `pad`
shifts every coordinate, so its operand reads stay guarded, and `case`
branches are never hoisted, so untaken branches stay unevaluated. Constant
index arithmetic is folded, a concatenate read at a constant coordinate
selects its operand without a guard, and coordinates have one canonical
spelling. The loop kernel now has 58 float operations and 8 buffer reads per
thread.

| SBI on the M2, pipelined | Before | After |
|---|---:|---:|
| Full step wall | 1.005 ms | 0.614 ms |
| Simulation wall | 0.650 ms | 0.243 ms |
| GPU busy / wall (step) | 98.4% | 97.8% |

The step takes 0.614 ms against 1.33 ms on XLA:CPU (2.2× faster) and 18.66 ms
on jax-mps. The dense update (0.30 ms, 48 launches) is now half of it. The
simulation's 202 launches take about 1.2 µs each, close to a per-launch floor.

Open: a single isolated step took 6.2 ms of GPU time before the loop changes,
against about 1.1 ms pipelined, which suggests low GPU clocks under sparse
work; not yet re-measured. Unary calls such as `exp(v2)` count as atoms and
are inlined at each use rather than given a temporary; the Metal compiler
likely merges them, but that is unverified. The M4 Pro has not been profiled.

## Success criteria for M1, against jax 0.12

All on jax and jaxlib 0.12.0.1 (PJRT C API 0.116) with
`JAX_PLATFORMS=metal,cpu`, as subprocess tests in `tests/runtime/test_pjrt.py`.

Floating agreement is normwise per array: `max|metal − cpu| ≤ rtol ·
max|cpu|`, with nonfinite values matching in place
(`tests/runtime/pjrt_tolerance.py`). Componentwise relative error cannot be
met where values cancel to near zero, and an absolute tolerance is vacuous
for arrays smaller than it. This replaced the componentwise wording on
2026-11-02; see "Agreement rule" below.

1. **Correctness against the CPU backend.** One test per op group of the
   census, matching exactly for integers and bools, within normwise rtol 2e-5
   for float32 and 1e-2 for f16 and bf16:
   - elementwise (done)
   - reductions over each axis and over all (sum, max, min, argmax)
   - `dot_general`, plain, batched and transposed
   - concatenate, pad, dynamic_slice and dynamic_update_slice
   - gather and scatter-add as `x[idx]` and `x.at[idx].add` produce them
   - `while` with a static trip count, and `cond`
   - `jax.random.normal` and `jax.random.uniform` with threefry keys
2. **The SBI step of `examples/training_steps.py`** compiles without
   UNIMPLEMENTED, and its loss, updated parameters and every optimizer-state
   array match the CPU backend within normwise rtol 1e-4 over ten steps. This is also M2's entry criterion. The
   LLM and SAM2-like steps belong to M3 to M5.
3. **No crashes on jax's common API surface:**
   - `lower().compile()`
   - `compiled.cost_analysis()` (empty) and `memory_analysis()` (`None` on JAX 0.12; native PJRT reports UNIMPLEMENTED)
   - `jit` with `donate_argnums`
   - deleting arrays and executables while work is in flight
   - dispatch from several Python threads
4. **Overhead and bandwidth**, on the M2 and on the M4 Pro:
   - a pipelined small call costs less than jax-mps's on the same machine
   - large elementwise programs reach 80% of `bench_copy_bandwidth`
5. **Compile time** is recorded for the SBI step's first call and stays
   under 5 s; the plan rejects a faster step that takes a minute to compile.

On a jaxlib upgrade:
1. Bump the header commit and its hash in `CMakeLists.txt`.
2. Rebuild.
3. Rerun the tests above.

The native dependency-error regression deliberately fails producer readiness
events, without causing GPU faults. It covers already-failed and late-failed
inputs, transitive chains, passthrough/empty/donated outputs, event errors,
host reads, and handle destruction while work is pending. The target is
excluded from the default build, so the wheel does not contain it.
`tests/runtime/test_pjrt_native.py` builds it in the interpreter's build
directory (`build/<wheel tag>`, after `uv sync`) and runs it as part of
`pytest`. To run it by hand:

```sh
cmake --build build/<wheel tag> --target palladium_pjrt_dependencies_test
ctest --test-dir build/<wheel tag> -R pjrt_dependencies --output-on-failure
```

StableHLO needs no pin of its own, because it is read with jaxlib's
bindings.
