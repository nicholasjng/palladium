# The `metal` PJRT plugin (M1 in progress)

Implementation record and handoff for milestone M1 of
`notes/metal-compute-stack.md`: a PJRT plugin that runs `jax.jit` programs
on the Apple GPU without MLX. This note says what exists, how it was built
and why, what to keep invariant, what comes next, and when M1 counts as
done. Status as of 2026-11-01, on jax and jaxlib 0.12.0.1, measured on an
M2 (Apple GPU family 8).

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
- Static counted `while`, including `lax.scan` trajectories and nested loops:
  a scalar i32/ui32 counter starting at a constant, compared with LT to a
  constant bound, incremented by one. Body kernels replay in one command
  buffer with alternating carry buffers and no host condition reads.
- Floating `dot_general`: plain/transposed/batched products, multiple batch
  and contraction axes, vectors and empty dimensions. It reuses the Pallas
  GEMM emitter; gradients and dots inside counted loops work too.
- Asynchronous execution with completion events, a buffer pool, and at most
  16 executions in flight.
- The complete SBI step compiles and executes, with a ten-step CPU comparison
  of losses, parameters and optimizer state (details and tolerance below).

Fails with UNIMPLEMENTED, naming the op (`palladium.stablehlo` raises
`UnsupportedProgram`): variadic or other reducer bodies (including argmax and
product), integer or explicitly algorithm-selected dots, dynamic or unrecognized `while`, `case`,
`gather`,
`scatter`, `reduce_window`, `sort`, `convolution`, float64 arithmetic and
64-bit integer reductions. Shape-changing bitcasts and wide integer operations
without explicit lowering (such as power) are rejected too. In
`examples/training_steps.py`, SBI now runs; LLM still stops at `gather` and
SAM2-like at `convolution`. CPU setup remains useful for reference inputs:

```python
with jax.default_device(jax.devices("cpu")[0]):
    key = jax.random.key(0)
```

Not supported by design yet: donation (jax refuses it for unknown
platforms), host callbacks (`jax.debug.print`, `pure_callback`), `eigh` and
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
| `CMakeLists.txt` | `palladium_pjrt` target and the pinned header download |
| `tests/runtime/test_pjrt.py` | End to end through jax, against the CPU backend |
| `tests/codegen/test_stablehlo.py` | Generated programs, no GPU needed |
| `examples/training_steps.py` | The SBI, LLM and SAM2-like training steps |

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
   - `pp_output(slot)` and `pp_error(code, message)`.

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
   returns. The handler copies outputs that are not fresh slots (inputs or
   constants returned directly, repeated outputs) and sets the execution's
   `Completion`, which is every output's `ready` and every input's and
   output's `last_use`.
8. **Buffer reuse.** Buffers come from a pool keyed by requested size (up
   to 1 GiB), since a fresh 16 MB MTLBuffer is zeroed on allocation. A
   released buffer returns to the pool only when its `last_use` completes.
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
   min, and or or returning its result. Other bodies and variadic reducers
   fail explicitly. Half/bfloat accumulators and partial buffers use float32.
   Reduction launches use SAFE math; min/max propagate NaNs and resolve
   signed-zero ties explicitly around Metal's SIMD collectives. Elementwise
   launches retain FAST math. The supplied initializer is combined once in
   the final pass, and is returned for empty reductions. Non-identity
   initializers need not match XLA's reduction schedule, which can insert
   them multiple times ([StableHLO semantics](https://openxla.org/stablehlo/spec#reduce)).

   Remaining: variadic argmax/argmin (palladium already lowers these for
   jaxprs), coalesced column reductions, tuning small rows, and epilogue fusion.
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
   They would need host condition reads or GPU-driven dispatch. `case` with
   a device predicate remains open. Trajectory updates currently copy the
   full carried array per iteration; in-place disjoint updates are future work.
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
6. **Pallas kernels inside `jit`.** Register lowering rules for
   `pallas_call_p` and `mpmd_map_p` with `platform="metal"`; the stub probe
   showed both work. They emit a `stablehlo.custom_call` whose typed-FFI
   `backend_config` must be a dictionary, carrying the MSL and launch.
   `stablehlo.py` turns that custom call into a `Launch`. This brings
   `metal_call` and `plmtl` kernels into whole programs.
7. **Donation.** jax refuses donation unless the platform is listed in
   `jax._src.interpreters.mlir._platforms_with_donation` (jax-mps appends
   `mps` there); `_pjrt.initialize` can append `metal`. Then check how the
   aliases arrive (expected: `tf.aliasing_output` argument attributes) and
   let the output take over the input's storage. Optimizer state makes this
   matter from M2 on.
8. **Host time per call.** 29 µs pipelined against 3 µs on CPU. Profile
   the split between jax's C API dispatch and Execute (command buffer and
   encoder creation, per-launch vectors, `std::function` and `shared_ptr`
   churn), and compare against jax-mps on the same machine in its own
   environment (`uv sync --group mps --no-group latest-jax`, since jax-mps
   pins jax 0.11).
9. **Smaller gaps.**
   - A failed execution does not fail executions that consume its outputs.
   - `PJRT_Buffer::last_use` is assigned without a lock, a race if two
     threads execute with the same buffer.
   - Each kernel compiles its own MTLLibrary; cache by source hash once
     programs grow.
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

M1 is still incomplete: variadic argmax, gather/scatter-add, `cond`, donation
and the full API/concurrency checklist remain open. Overhead versus jax-mps
and bandwidth targets on M2 and M4 Pro have not been established. M2 still
needs profiling/tuning of replay, trajectory updates and reductions.

## Success criteria for M1, against jax 0.12

All on jax and jaxlib 0.12.0.1 (PJRT C API 0.116) with
`JAX_PLATFORMS=metal,cpu`, as subprocess tests in `tests/runtime/test_pjrt.py`.

1. **Correctness against the CPU backend.** One test per op group of the
   census, matching exactly for integers and bools, within rtol 2e-5 for
   float32 and 1e-2 for f16 and bf16:
   - elementwise (done)
   - reductions over each axis and over all (sum, max, min, argmax)
   - `dot_general`, plain, batched and transposed
   - concatenate, pad, dynamic_slice and dynamic_update_slice
   - gather and scatter-add as `x[idx]` and `x.at[idx].add` produce them
   - `while` with a static trip count, and `cond`
   - `jax.random.normal` and `jax.random.uniform` with threefry keys
2. **The SBI step of `examples/training_steps.py`** compiles without
   UNIMPLEMENTED, and its loss and updated parameters match the CPU backend
   within rtol 1e-4 over ten steps. This is also M2's entry criterion. The
   LLM and SAM2-like steps belong to M3 to M5.
3. **No crashes on jax's common API surface:**
   - `lower().compile()`
   - `compiled.cost_analysis()` (empty) and `memory_analysis()` (raises)
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

StableHLO needs no pin of its own, because it is read with jaxlib's
bindings.
