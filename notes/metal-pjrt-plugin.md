# The `metal` PJRT plugin

A PJRT plugin that runs `jax.jit` programs on the Apple GPU without MLX:
milestone M1 of `notes/metal-compute-stack.md`, and the base of M2 to M5.
This note says what exists, how it works, what to keep invariant, what is
left, and how to measure it. It targets jax and jaxlib 0.12 (PJRT C API
0.116). Measurements are not recorded here: run the benchmarks
(`docs/benchmarking.md`) and keep their records next to this note.

## Status

Works:

- `jax.devices("metal")`, `device_put`, `np.asarray`, deletion, copies to
  CPU, `memory_stats()`, and `apple_gpu_family` as a device attribute.
- `jax.jit` of programs made of elementwise ops (float, f16, bf16, bool, and
  signed and unsigned integers of 8 to 64 bits; comparisons, select, convert,
  bitcast, shifts) and index maps (broadcast, reshape, transpose, slice,
  reverse, iota), with constants, multiple outputs, inputs returned as
  outputs, calls and composites.
- Guarded index maps: `concatenate`, `pad` (including interior padding and
  negative edge padding), `dynamic_slice` and `dynamic_update_slice`. They
  fuse with producers and consumers without intermediate buffers.
- Threefry on the device: key split and fold-in, random bits, and float32
  uniform and normal samples.
- `reduce` with add, maximum, minimum, and, or bodies over arbitrary axes,
  including empty reductions, and variadic scalar reducers (argmax and argmin
  with first-index ties and NaN behavior).
- Gather for JAX indexing, including batching, windows, negative indices and
  clip/fill lowering. Scatter-add for float32, float16, bfloat16, int32 and
  uint32 data with 32-bit indices, including duplicate indices.
- Static counted `while`, including `lax.scan` trajectories and nested loops.
- Floating `dot_general`: plain, transposed and batched products, multiple
  batch and contraction axes, vectors and empty dimensions.
- `convolution` (every layout, stride, padding, both dilations, window
  reversal), `reduce_window` and `select_and_scatter`.
- Pure `stablehlo.case` (`lax.cond`, `lax.switch`), including nested
  conditionals.
- Asynchronous execution with completion events, a buffer pool, and at most
  16 executions in flight.
- Donation through `tf.aliasing_output`.
- Palladium kernels inside `jit`: `pl.pallas_call` (any math mode, aliases)
  and `metal_call` (including under `jax.vmap`), also inside counted loops.
- The SBI, LLM and SAM2-like steps of `examples/training_steps.py`, and the
  CNF update of `examples/cnf_density.py`.

Fails with UNIMPLEMENTED, naming the op (`palladium.stablehlo` raises
`UnsupportedProgram`):

- reducer bodies other than the ones above (including a single-result
  product), and 64-bit integer reductions;
- integer or explicitly algorithm-selected dots;
- dynamic or unrecognized `while`;
- non-add and variadic scatter, and scatter on other data types;
- `reduce_window` over more than 64 elements per output;
- `sort`, and comparisons in total order;
- grouped convolutions (`feature_group_count` or `batch_group_count` above 1);
- float64 arithmetic, shape-changing bitcasts, and 64-bit integer operations
  without an explicit lowering (such as power).

Not supported by design yet: host callbacks (`jax.debug.print`,
`pure_callback`), `eigh` and `svd`, the persistent compilation cache
(executables do not serialize), more than one device.

Reference inputs can be made on CPU:

```python
with jax.default_device(jax.devices("cpu")[0]):
    key = jax.random.key(0)
```

## Build, run, test

Requirements: macOS 26 on Apple silicon, uv, CMake and Ninja (`brew install
cmake ninja`), and network access on the first configure, which downloads
metal-cpp and the PJRT C API header, both checked against pinned hashes.
Xcode is not needed.

```sh
uv sync
# uv does not rebuild for C++-only edits; after changing native/:
uv sync --reinstall-package palladium

JAX_PLATFORMS=metal,cpu uv run python -c "import jax; print(jax.devices())"
uv run pytest -q tests/runtime/test_pjrt.py tests/codegen/test_stablehlo.py
uv run python examples/training_steps.py   # StableHLO op census
```

The plugin registers only when JAX_PLATFORMS names `metal`; `metal,cpu`
makes it the default device, `cpu,metal` keeps CPU the default. Otherwise
every jax process would load it and print jax's "experimental platform"
warning. Plugin tests run in subprocesses because jax reads JAX_PLATFORMS
once per process. The full check is `ruff check`, `ruff format --check`,
`ty check`, `pytest` and `pytest -m fuzz`.

To see which lines of the Python package the tests leave out, run them under
coverage; it follows the plugin tests into their subprocesses:

```sh
uv run coverage run -m pytest -q
uv run coverage combine
uv run coverage report
```

A segfault inside `libjax_common.dylib` at a `pjrt::` or `PjRtCApi*` frame
almost always means jaxlib called an API slot the plugin left null. Run with
`python -X faulthandler` to see the frame, then implement that slot.

The installed plugin is stripped. `uv sync --reinstall-package palladium -C
install.strip=false` keeps its symbols for Instruments or `sample`; a later
plain `uv sync` strips it again. The unstripped library in `build/<wheel
tag>` also resolves offsets with `nm -n`.

## Layout

| Path | What |
|---|---|
| `native/pjrt/plugin.h` | Shared types: `Error`, `Memory` (extending the C API's function-table structs), `Completion`, `PJRT_Buffer`, the pool interface |
| `native/pjrt/plugin.cpp` | `GetPjrtApi`: errors, events, client, device, memory, topology, buffers, the pool |
| `native/pjrt/executable.cpp` | `PJRT_Client_Compile`, the `pp_*` builder ABI, `Execute`, executable queries, in-flight throttling, counters |
| `native/runtime/dispatch.*` | `CommandBatch::commit(on_completed)`, the non-blocking commit Execute uses |
| `src/palladium/_pjrt.py` | `jax_plugins` entry point (`initialize`), ctypes signatures, the compile callback, `stats()` |
| `src/palladium/stablehlo/` | StableHLO to `Program`: graph import, terms, kernel text, scheduling |
| `src/palladium/stablehlo/gemm.py` | Canonical batched products to the Pallas GEMM emitter |
| `CMakeLists.txt` | `palladium_pjrt` target, the pinned downloads, the native dependency test |
| `tests/runtime/test_pjrt.py` | End to end through jax, against the CPU backend |
| `tests/runtime/test_pjrt_native.py` | Builds and runs `tests/native/pjrt_dependencies.cpp` |
| `tests/codegen/test_stablehlo.py` | Generated programs, no GPU needed |
| `examples/training_steps.py` | The SBI, LLM and SAM2-like training steps |
| `benchmarks/bench_dispatch.py` | Per-call overhead and bandwidth, per backend |
| `benchmarks/bench_training.py` | Pipelined training steps with GPU busy fraction, launches and host time; compile plus first call |
| `benchmarks/bench_kernels.py` | GEMM, attention and embedding kernels against MLX |
| `examples/pjrt_sbi_audit.py` | The SBI Metal/CPU numerical audit |
| `examples/pjrt_long_run.py` | Long training runs against CPU's own rounding sensitivity |

## How it works

1. **What JAX hands a new platform.** Registering a stub backend factory
   (`xla_bridge.register_backend_factory("metal", ...)`) and lowering with
   `jit(f).trace(*args).lower(lowering_platforms=("metal",))` shows the
   StableHLO and the failures without any plugin; see
   `notes/metal-compute-stack.md`, "What JAX hands a new platform".
2. **The C API.** jaxlib ships only the XLA FFI headers, not
   `pjrt_c_api.h`. CMake downloads it from the XLA commit that jax pins in
   its `MODULE.bazel` and checks its SHA-256. jaxlib requires the same major
   version and tolerates a lagging minor one. The header defines
   `PJRT_Error` and `PJRT_Memory` as structs holding a function table that
   jaxlib calls directly, so the plugin subclasses them and fills the
   tables. jaxlib calls some slots without a null check:
   `PJRT_Client_TopologyDescription`, `PJRT_Error_ForEachPayload`,
   `PJRT_TopologyDescription_Fingerprint` (compilation cache keys) and
   `PJRT_LoadedExecutable_GetDeviceAssignment`, whose
   `DeviceAssignmentProto` for one device is 8 bytes, encoded by hand to
   avoid protobuf.
3. **Device and buffers.** One client, one device, one `device` memory.
   Buffers are shared MTLBuffers with host data copied in (strided arrays
   copy one dense block at a time).
4. **Compiling in Python.** A C++ compiler would need LLVM, MLIR and
   StableHLO built at StableHLO's pinned LLVM commit. Instead,
   `PJRT_Client_Compile` calls a ctypes callback registered by
   `palladium._pjrt`. Programs arrive as VHLO portable artifacts, which
   `jaxlib.mlir.dialects.stablehlo.deserialize_portable_artifact` reads with
   jaxlib's own bindings, so the versions always match and StableHLO needs
   no pin of its own. `palladium.stablehlo.compile_program` returns a
   `Program`, and `_pjrt` replays it through the builder ABI:
   - `pp_slot(kind, type, dims, data)`: a parameter (numbered in order), a
     fresh buffer per execution, or a constant uploaded once.
   - `pp_kernel(msl, function, math_mode, slots, grid, threadgroup,
     offsets)`: compiles the MSL, cached per executable by source, entry
     point and math mode; an empty threadgroup lets the pipeline choose.
   - `pp_output(slot)`, `pp_alias(parameter, output)` and
     `pp_error(code, message)`.

   `UnsupportedProgram` becomes UNIMPLEMENTED; any other exception becomes
   INTERNAL with its traceback. No exception may unwind into the plugin.
5. **Code generation.** Outputs are grouped by element count; each group is
   one kernel with one thread per output element. A thread derives its
   coordinates from its flat index, and every value is generated at the
   coordinates it is read at, memoized per (value, coordinates). Index maps
   therefore cost index arithmetic and never a copy. Scalar ops reuse
   palladium's templates (`emit.numeric.format_scalar`). `func.call` and
   `stablehlo.composite` are inlined. Non-splat constants become constant
   slots. Fused kernels run RELAXED math: FAST folds `x != x` to false and
   loses the NaN of `x - x`. GEMMs and user kernels keep their own mode.
6. **Execution.** `Execute` binds slots to buffers, encodes every launch
   into one command buffer, commits it with a completion handler and
   returns. Passthrough and repeated outputs and donation copybacks use blit
   commands after the compute launches, in two passes: snapshots into new
   outputs, then copybacks into donated storage. No copy in a pass writes a
   buffer another one uses. The handler retains the program and the storage
   owners, then sets the execution's `Completion`, which, joined with the
   inputs' completions, is each output's `ready`.
7. **Buffer reuse.** Buffers come from a pool keyed by requested size,
   since a fresh MTLBuffer is zeroed on allocation. The pool holds up to a
   quarter of the device's recommended working set (`PALLADIUM_POOL_BYTES`
   overrides it) and evicts its oldest free buffers. Each dispatch holds
   independent owners of all input, intermediate and output storage until
   completion. At most 16 executions are in flight: without the bound, a
   caller running ahead of the GPU allocates a new output per queued call.

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
  computing something else. A failed input fails every output that depends
  on it, through readiness joins that never block a Metal callback thread.
- **Concurrency.** Concurrent read-only uses of shared buffers are
  supported. Deleting or donating the same Python array from another thread
  at the same time is not.

## Lowerings

**Reductions.** Dependencies are scheduled before consumers, and a
reduction is computed once however often it is read. Producers fuse into
the reads inside the reduction loop. Kernels use 128 threads per output,
SIMD collectives and four threadgroup partials; with fewer than 128 outputs
and more than 4,096 reduced elements, independent 4,096-element chunks feed
a second reduction launch. The reducer region must be exactly one binary
add, max, min, and or or. Half and bfloat accumulate in float32. Reduction
launches use SAFE math; min and max propagate NaNs and resolve signed-zero
ties around Metal's SIMD collectives. The initializer is combined once in
the final pass and returned for empty reductions, so non-identity
initializers need not match XLA's schedule, which may insert them several
times ([StableHLO semantics](https://openxla.org/stablehlo/spec#reduce)).
Variadic scalar reducers fold sequentially per output element.

Reductions over at most 16 elements fold into the kernel that reads them,
unrolled from the initial value as XLA's sequential reduce. Folding
recomputes the reduction in every thread that reads it, which is cheaper
than a launch. A fold read at the shifted coordinates of a guarded branch
would be regenerated in each branch, so there it is stored once by a
one-thread-per-output kernel.

**Guarded index maps.** `Kernel.choose` emits each branch's producer chain
inside its guard. Each guarded branch is a scope of its own names: the
coordinates its guard shifts and the temporaries it defines. A value whose
coordinates use none of them is computed once before the branch chain. That
is safe for `concatenate` and `dynamic_update_slice`, whose unshifted
coordinates index their operands within bounds; `pad` shifts every
coordinate, so its operand reads stay guarded, and `case` branches are never
hoisted, so untaken branches stay unevaluated. Empty operands are never
read. Dynamic offsets are scalar buffer reads, clamped to the operand minus
the slice or update size. A reshape names its flat index in a `uint`
variable, so chains of reshapes and transposes do not repeat it in every
coordinate.

**Integers.** `long` and `ulong` live only in the PJRT compiler's type map,
as do the 8- and 16-bit types; the Pallas `CTYPES` contract is unchanged.
Wide literals never pass through float32. Wrapping arithmetic uses unsigned
intermediates, shifts guard counts at or above the width, and divide and
remainder handle zero and signed-min / -1 explicitly. Narrow integers
compute in 32 bits and wrap on the way back.

**Counted loops.** The compiler proves a scalar i32/ui32 counter starting
at a constant, compared with LT to a constant bound and incremented by one,
including invariant bounds and hoisted constants. It compiles the body once
and rebinds its launches per iteration, in one command buffer with no host
condition read. Changed carries have two output banks, and carry
permutations are materialized, so outputs cannot overwrite inputs the
iteration still needs. A carry written by a GEMM, a long reduction, a
scatter or a kernel dispatch goes straight into its bank. Invariant carries
keep their slots, and body temporaries are reused between iterations.
Induction variables (integer scalar carries with a constant start and step)
bind one constant array per variable at a byte offset per iteration. A
carry updated only by a `dynamic_update_slice` of itself, and read nowhere
else in the body, lives in one buffer per execution, each iteration writing
just its update: this covers `lax.scan` outputs. A graph may replay at most
100,000 launches.

**`case`.** Guarded per-result selection: scalar and index-map producers
stay inside each branch, while reductions, dots and counted loops are
computed eagerly before the selection. All branches are validated, so an
unsupported op fails even in an untaken branch. Out-of-range indices select
the last branch.

**`dot_general`.** Dimension numbers are normalized to `(batch, M, K)` and
`(batch, K, N)`. Canonical buffers bind directly; packing kernels fuse
transposes, reshapes and elementwise producers when necessary.
`stablehlo/gemm.py` traces a tiled Pallas matmul and calls the cooperative
emitter, with batches as a grid axis. Aligned products use
`simdgroup_matrix` on families 7 to 9; output and K tails and family 10 use
Metal 4 TensorOps. Both accumulate in float32. Empty contraction dimensions
fill zero without reading either input. A float32 product with fewer than
32 output tiles splits K into up to 16 power-of-two chunks of at least 256:
a batched GEMM writes float32 partials that a short reduction sums. This
occupies more of the GPU and bounds the rounding error of one long float32
accumulation. Half types are not split, so their partials do not round.

**Convolutions and pooling.** `convolution` lowers to an index map and a
`dot_general`: `conv_patches` maps the input to (batch, output spatial...,
window..., input feature), zero on padding and between dilated input
elements, and the kernel is transposed to (window..., input, output
feature). An input dilated by n would put n - 1 zeros between patch
elements; when stride and kernel dilation are 1 and the window and output
divide by n, `conv_phases` builds undilated patches per output phase and
`conv_taps` gathers each phase's taps, as one batched product or, when every
phase starts at the same input offset, one product against all phases' taps.
The convolution's input is stored before the patches read it. `reduce_window`
folds its window inline per output element. `select_and_scatter` is a
gather without atomics: each element revisits the windows that contain it,
recomputes each window's selection, and adds the window's source value
where that selection is itself.

**Gather and scatter.** Gather fuses source and index expressions into
reads. Scatter initializes fresh output storage and uses atomic adds for
f32, i32 and u32. For f16 and bf16 it uses one atomic uint word per
element, with compare/exchange and rounding after each update, then
unpacks; odd-sized outputs never touch a neighboring halfword outside their
allocation.

**Palladium kernels.** On `metal`, `pallas_call` lowers through
`palladium.ffi.pallas_lowering` to the same typed-FFI custom call,
`palladium_dispatch`, that `metal_call` and `plmtl.kernel` emit on every
platform. Its `mhlo.backend_config` carries the MSL, function name, thread
grid, threadgroup, math mode, and the `batch_size` and `elem_strides` of
`metal_call`'s vmap rule. The compiler turns the call into one launch per
batch element. An aliased result starts as a copy of its input, so elements
the kernel does not write keep the input's values. Other custom-call
targets and non-row-major layouts are rejected.

**Donation.** `_pjrt.initialize` appends `metal` to JAX's private
donation-platform list. Outputs compute into fresh storage, then copy back
into donated storage after all reads and passthrough snapshots, which
handles permutations and repeated outputs. Runtime non-donatable indices
and repeated arguments override the compile-time aliases.

**Metal's 31 buffers.** A fusion whose outputs and reads exceed 31 buffers
splits into several kernels, and a value that reads more by itself stores
its largest operand that fits.

## Numerical agreement

Floating agreement with the CPU backend is normwise per array: `max|metal −
cpu| ≤ rtol · max|cpu|`, with nonfinite values matching in place
(`tests/runtime/pjrt_tolerance.py`). Componentwise relative error cannot be
met where values cancel to near zero, and an absolute tolerance is vacuous
for arrays smaller than it: with rtol 1e-4 plus atol 2e-6, an all-zero
second-moment estimate would pass. The budgets are 1e-4 for training steps,
2e-5 for float32 operations and 1e-2 for f16 and bf16; integers and bools
match exactly.

Two float32 training runs drift apart even on one backend, so over long
runs the yardstick is CPU against itself from parameters moved by one ulp,
and a float64 run on the same data (`examples/pjrt_long_run.py`).

Known differences from XLA:CPU:

- Duplicate floating scatter updates apply in an implementation-dependent
  order.
- A `lax.scatter_add` window that is partially out of bounds is discarded
  as a whole by XLA:CPU. This lowering keeps its in-bounds elements, as the
  [StableHLO scatter specification](https://openxla.org/stablehlo/spec#scatter)
  says. The `x[idx]` and `x.at[idx].add` forms match CPU.
- Float-to-narrow-integer conversion of out-of-range values is undefined on
  both backends.

## Remaining work

- Reductions: parallel variadic reductions, coalesced tiling for column
  reductions, one thread per output for rows of 17 to about 128 elements,
  epilogue fusion.
- Loops: dynamic conditions and other induction patterns (host condition
  reads or GPU-driven dispatch), 64-bit loop counters, selective dispatch
  of expensive `case` branches.
- Products: epilogue fusion, avoiding transposed packing, reading split-K
  chunks in place instead of copying the left operand, small and tail
  products.
- Kernels: a `metal` rule for `pl.kernel`'s `mpmd_map` outside `plmtl`;
  avoiding the copy for an aliased input that is not read again.
- Donation: in-place execution without the temporary and the copyback.
- Host time per call: the pool's mutex is taken once per buffer while
  completion threads return the previous executions' buffers; one lock per
  execution would remove the contention for small calls.
- Pipelines are cached per executable; sharing them across executables
  would cut compile time.
- `compile_program` raises Python's recursion limit to fit the program and
  leaves it raised; depth-bounded fusion would avoid that at the cost of
  launches.
- Dropping the JAX_PLATFORMS gate.

## Measuring

`palladium._pjrt.stats(reset=...)` reads the plugin's counters: executions,
command buffers and launches; host seconds inside Execute, split into
encoding and waiting for the in-flight bound; buffer-pool hits, new
MTLBuffers and their creation time; and GPU time from each command
buffer's `GPUStartTime` and `GPUEndTime`, as the sum and the union (busy)
of the intervals. Busy time is a command buffer's whole residency,
including time it waits behind a predecessor, not pure kernel time.
Completion handlers arrive out of start order, so synchronize on the last
outputs before reading.

The benchmarks are mew benchmarks, run once per backend and compared with
`mew compare` (`docs/benchmarking.md`). Do not run other GPU workloads at
the same time.

```sh
JAX_PLATFORMS=metal,cpu uv run mew run benchmarks/bench_dispatch.py benchmarks/bench_training.py -o metal.jsonl
JAX_PLATFORMS=mps,cpu ../jax-mps/.venv/bin/mew run benchmarks/bench_dispatch.py benchmarks/bench_training.py -o mps.jsonl
uv run mew compare metal.jsonl mps.jsonl
```

Never install jax-mps into the project's `.venv`: its plugin registers in
every jax process with priority over CPU and becomes the default backend.
Keep it in a venv inside the jax-mps checkout.

`bench_small_call` is the per-call overhead; `bench_elementwise` reports
bandwidth as a `bytes` rate, to compare with
`bench_roofline.py::bench_copy_bandwidth`. `bench_first_call` measures
compile plus first call after clearing JAX's caches in a running process,
which is shorter than a fresh process: that includes JAX's own first-use
costs, and the Metal driver's shader cache may be warm or cold.
`examples/pjrt_sbi_audit.py` writes the per-leaf SBI comparison against
CPU, and `examples/pjrt_long_run.py` the long-run comparison.

## Acceptance criteria

With `JAX_PLATFORMS=metal,cpu`, as subprocess tests in
`tests/runtime/test_pjrt.py`, under the agreement rule above.

1. **Correctness against the CPU backend**, one test per op group:
   elementwise; reductions over each axis and over all (sum, max, min,
   argmax); `dot_general`, plain, batched and transposed; concatenate, pad,
   dynamic_slice and dynamic_update_slice; gather and scatter-add as
   `x[idx]` and `x.at[idx].add` produce them; `while` with a static trip
   count, and `cond`; `jax.random.normal` and `jax.random.uniform` with
   threefry keys.
2. **The SBI step** of `examples/training_steps.py` compiles without
   UNIMPLEMENTED, and its loss, updated parameters and every optimizer-state
   array match the CPU backend over ten steps.
3. **No crashes on jax's common API surface:** `lower().compile()`;
   `compiled.cost_analysis()` (empty) and `memory_analysis()` (`None`: the
   plugin reports UNIMPLEMENTED); `jit` with `donate_argnums`; deleting
   arrays and executables while work is in flight; dispatch from several
   Python threads.
4. **Overhead and bandwidth:** a pipelined small call costs less than
   jax-mps's on the same machine, and large elementwise programs reach 80%
   of `bench_copy_bandwidth`.
5. **Compile time** of the SBI step's first call stays under 5 s.

On a jaxlib upgrade: bump the header commit and its hash in
`CMakeLists.txt`, widen the jax bound in `pyproject.toml`, rebuild, and
rerun the tests above.

The native dependency test deliberately fails producer readiness events,
without causing GPU faults. It covers already-failed and late-failed
inputs, transitive chains, passthrough, empty and donated outputs, event
errors, host reads, and handle destruction while work is pending. The
target is excluded from the default build, so the wheel does not contain
it. `tests/runtime/test_pjrt_native.py` builds it in the interpreter's
build directory and runs it as part of `pytest`. By hand:

```sh
cmake --build build/<wheel tag> --target palladium_pjrt_dependencies_test
ctest --test-dir build/<wheel tag> -R pjrt_dependencies --output-on-failure
```
