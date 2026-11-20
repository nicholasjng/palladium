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
- `reduce` with add, multiply, maximum, minimum, and, or, xor bodies over
  arbitrary axes, including empty reductions; any other scalar reducer body,
  variadic ones (argmax and argmin with first-index ties and NaN behavior)
  and 64-bit integers as a sequential fold per output.
- Gather for JAX indexing, including batching, windows, negative indices and
  clip/fill lowering. Scatter with 32-bit indices: assignment on every type,
  and add, multiply, min, max or any scalar body on types of up to 32 bits,
  including duplicate indices.
- Static counted `while`, including `lax.scan` trajectories and nested loops,
  with 32- or 64-bit counters.
- Any other `while` (`lax.while_loop` with a data-dependent condition,
  `fori_loop` with a traced bound), nested in counted loops and conditionals:
  the host reads the condition before each iteration.
- Host callbacks: `jax.pure_callback`, `io_callback` (ordered or not),
  `jax.debug.print` and `jax.debug.callback`, and `checkify` checks, also
  inside loops.
- Dense linear algebra on LAPACK, between launches: Cholesky, triangular
  solves, LU, QR, `eigh` and SVD, and what JAX builds on them (`solve`, `inv`,
  `det`, `slogdet`, `pinv`, `lstsq`, Gaussian log densities), with gradients
  and under `vmap`.
- `sort` along any dimension with any comparator and several operands
  (`jnp.sort`, `argsort`, `lax.sort` by keys, `top_k`, `median`,
  `searchsorted`, `unique` with a size, `random.permutation`), stable.
- Cumulative reductions (`cumsum`, `cumprod`, `cummax`, `cummin`,
  `cumlogsumexp`, forward and reverse) along axes of any length.
- `rng_bit_generator` (the `rbg` PRNGs), as Threefry on the element index.
- Floating `dot_general`: plain, transposed and batched products, multiple
  batch and contraction axes, vectors and empty dimensions. Integer and
  boolean `dot_general`, as a reduction over products.
- `convolution` (every layout, stride, padding, both dilations, window
  reversal, feature and batch groups), `reduce_window` with windows of any
  length and any scalar reducer body, and `select_and_scatter`.
- Comparisons in total order, `fma`, `exp2`, `log2`, `reduce_precision`, and
  bitcasts between element widths.
- Pure `stablehlo.case` (`lax.cond`, `lax.switch`), including nested
  conditionals.
- Asynchronous execution with completion events, a buffer pool, and at most
  16 executions in flight.
- Donation through `tf.aliasing_output`.
- Palladium kernels inside `jit`: `pl.pallas_call` (any math mode, aliases)
  and `metal_call` (including under `jax.vmap`), also inside counted loops.
- The SBI, LLM and SAM2-like steps of `examples/training_steps.py`, the
  CNF update of `examples/cnf_density.py`, and LoRA training of SAM2
  (`examples/sam2`, Hiera-tiny at 1024 px) and of Qwen2.5-0.5B
  (`examples/lora`, float32 and bfloat16).

Fails with UNIMPLEMENTED, naming the op (`palladium.stablehlo` raises
`UnsupportedProgram`):

- complex numbers, and with them `fft` and `eig`;
- reducer, scatter and comparator bodies that are not scalar elementwise
  code, and `reduce_window` or `scatter` of several operands;
- scatter that combines 64-bit elements, or takes 64-bit indices;
- `select_and_scatter` reading more than 64 elements per output;
- explicitly algorithm-selected dots;
- a host callback with effects inside a loop condition or a reducer;
- float64 arithmetic, and 64-bit integer operations without an explicit
  lowering (such as power).

Not supported by design yet: the persistent compilation cache (executables
do not serialize), more than one device.

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
| `native/pjrt/executable.cpp` | `PJRT_Client_Compile`, the `pp_*` builder ABI, `Execute` with its host steps and branches, executable queries, in-flight throttling, counters |
| `native/runtime/dispatch.*` | `CommandBatch::commit(on_completed)`, the non-blocking commit Execute uses |
| `src/palladium/_pjrt.py` | `jax_plugins` entry point (`initialize`), ctypes signatures, the compile callback, `stats()` |
| `src/palladium/stablehlo/` | StableHLO to `Program`: graph import, terms, kernel text, scheduling, storage sharing |
| `src/palladium/host.py` | Lowering rules for work on the host: Python callbacks and LAPACK |
| `src/palladium/stablehlo/gemm.py` | Canonical batched products to the Pallas GEMM emitter |
| `CMakeLists.txt` | `palladium_pjrt` target, the pinned downloads, the native dependency test |
| `tests/runtime/test_pjrt.py` | End to end through jax, against the CPU backend |
| `tests/runtime/test_pjrt_native.py` | Builds and runs `tests/native/pjrt_dependencies.cpp` |
| `tests/codegen/test_stablehlo.py` | Generated programs, no GPU needed |
| `examples/training_steps.py` | The SBI, LLM and SAM2-like training steps |
| `examples/sam2/` | SAM2 in JAX with LoRA training, and its parity check against the MLX port |
| `examples/lora/` | Qwen2.5-0.5B in JAX with mlx-lm's LoRA training, and its parity check against mlx-lm |
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
   - `pp_host(id, slots, offsets)`: a call of a Python function on the
     contents of the slots, and `pp_branch(slot, skip)`: skip steps (back,
     when negative) if the boolean in the slot is false, or always without
     a slot.
   - `pp_output(slot)`, `pp_alias(parameter, output)` and
     `pp_error(code, message)`.

   The plugin reports jaxlib's StableHLO version as its own
   (`stablehlo_current_version`). Without it, jaxlib serializes for a
   release months old and rejects programs with newer ops such as `fma`,
   which the special functions and several samplers use.

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
7. **Host steps and branches.** A host step or a conditional branch needs
   values the GPU has yet to compute. `Execute` commits the command buffer
   encoded so far, waits for it (and, the first time, for the arguments
   other executions still compute), and goes on in a new one; whatever
   follows the last such step is committed without waiting, as always. A
   host step calls `palladium._pjrt._host` with pointers into the buffers'
   shared memory; NumPy views them without a copy. A program with such
   steps therefore blocks its caller and does not pipeline.
8. **Storage sharing.** Fresh slots that are not outputs share storage when
   their lifetimes, from the first step binding them to the last, do not
   overlap and their byte sizes match (`program.share_storage`). A slot
   bound inside a host-driven loop lives for the whole loop. An execution's
   memory is then bounded by the program's widest point: the SAM2 step at
   1024 px exhausted a 16 GB M2 without it.
9. **Buffer reuse.** Buffers come from a pool keyed by requested size,
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

**Counted loops.** The compiler proves a scalar integer counter starting
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

**Dynamic loops.** A `while` that is not provably counted keeps its
condition as a program of its own. The carries start as copies in slots of
their own, which hold the results in the end. Reading the condition costs a
command buffer and a wait, so a body made of launches alone runs several
iterations between two reads: an iteration is the condition, the body into a
second bank, and copies back that a kernel gates on the condition. Once the
condition is false the carries stop changing, so it stays false, and the
iterations past the last change nothing. Groups of 1, 1, 2 and 4 iterations
end in a branch out of the loop, then groups of 8 repeat: a short loop
wastes little, and a long one waits once in eight iterations. A body with
host steps of its own (a callback, a nested dynamic loop) cannot run
unobserved. It is laid out twice, between the carries' slots and the second
bank, each copy behind a read of the condition. Induction variables and
in-place carries are optimizations of counted loops only.

**Host calls.** `palladium.host` registers lowering rules for `metal` that
emit a `palladium_host` custom call naming a Python function by index: for
`pure_callback`, `io_callback`, `debug_callback`, `debug_print` and
`checkify`'s check, and for `cholesky`, `triangular_solve`, `lu`, `qr`,
`eigh` and `svd`, which run on LAPACK through NumPy and SciPy (MLX runs them
on the CPU too). The compiler schedules the inputs into buffers, then the
step, then reads the results as buffers. Calls with effects run in program
order whether or not anything reads them, and a loop holding one runs for
it; the token of an ordered effect is an empty buffer. Kernels compute every
branch of a `case` and select, but a host call may raise, print or cost
seconds, and a host-driven loop may never end outside its branch: a branch
step skips each unless every `case` branch around it is the one taken.
diffrax, which steps with a `while_loop` and reports errors from
conditionals, solves on `metal` this way.

**Sort.** A merge sort by rank, one launch per doubling of the sorted runs,
all with one kernel: each pass binds its run width at an offset into a
constant array and alternates between two banks of buffers. One thread per
element binary-searches the other run of its pair with the program's own
comparator and writes itself at its rank; elements of the left run go
before equal ones of the right, so the sort is stable. All operands move
together. Work is n log² n comparisons, with no data-dependent launches.

**Cumulative reductions and long windows.** JAX lowers `cumsum` and its
relatives to a `reduce_window` as long as the axis. Along more than 128
elements it becomes three launches: the total of each block, for each block
the total of the blocks before it, and each element folded from its block's
start. Blocks hold about the cube root of four times the length, which
balances the last two. The reducer must be associative and the initial
value its identity. Any other window of more than 64 elements, or with a
reducer body of its own, folds in a loop per output element and is stored
once for all readers.

**Scatter.** One thread per update element. An assignment stores. An
addition of 32-bit elements is an atomic add. Any other update is a
compare-and-exchange loop on the element's 32-bit word, with the scatter's
own body between load and exchange; narrower elements get a word each while
they are updated. Updates of one element apply in no particular order.

**Grouped convolutions.** Each group is one product of a batch: feature
groups split the patches' features, batch groups (the kernel gradient of a
grouped convolution) their batch. A group contracting at most 16 elements,
as in a depthwise 3×3, is a sum that the reading kernel unrolls, with no
GEMM and no patch buffer.

**Attention.** A floating product whose left operand is a row softmax
(`exp(s - max s) / sum`, as `jax.nn.softmax` writes it) of a product,
possibly times a constant, is attention if nothing else reads the scores or
the probabilities. It runs as one Pallas kernel through the TensorOps
recognizer: a threadgroup per tile of queries with the softmax in
registers, each head a batch, so the queries-by-keys scores take no buffer.
The match is by structure through the reshapes `jnp.matmul` adds; lengths
must fill the kernel's tiles and heads be 16 to 128 wide in steps of 16,
otherwise the graph is scheduled as written. A gradient is: there the
probabilities have other readers.

**Shared reduction operands.** A reduction stores its operand first when
other nodes read it too and it binds more than two buffers. A normalization
reads what it took the mean of twice more, and fused, each reader recomputed
a residual sum of dozens of buffers.

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

**Gather.** Gather fuses source and index expressions into reads.

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
- `rng_bit_generator` draws different bits than XLA:CPU: StableHLO leaves
  the generator to the backend. The default Threefry PRNG matches.
- Linear algebra runs LAPACK through NumPy and SciPy, which may pick other
  drivers than jaxlib's; results agree within rounding, and eigenvectors
  and singular vectors up to their sign conventions.

## Remaining work

- Reductions: parallel variadic reductions, coalesced tiling for column
  reductions, one thread per output for rows of 17 to about 128 elements,
  epilogue fusion.
- Loops: a dynamic loop inside a dynamic loop, and every host call in a
  conditional branch, still cost a wait per read, where the outer loop
  could carry the inner condition as data. A 200-step diffrax solve nests
  such loops: 1,002 command buffers and 243 ms on an M2 with 34 ms of GPU
  time, against 0.4 ms on CPU; a flat loop of 1,000 small iterations takes
  31 ms. Indirect dispatch would also save the iterations run past the end.
  Selective dispatch of expensive `case` branches made of kernels, as host
  calls in branches already have.
- Complex numbers and `fft`.
- Linear algebra on the GPU: one thread per matrix for batches of small
  Cholesky factorizations and triangular solves, which SBI's Gaussian
  posteriors use; the host steps stop the pipeline.
- Sort: start from short runs sorted in registers, to save the first
  passes; a sort of a million elements is 20 launches of a million threads.
- Storage sharing ignores slots of different sizes; the pool then drops and
  reallocates buffers when a step's working set exceeds it (175 new buffers
  per SAM2 step at 1024 px on a 16 GB M2).
- Products: epilogue fusion, avoiding transposed packing, reading split-K
  chunks in place instead of copying the left operand, small and tail
  products.
- Grouped convolutions with 17 or more elements per group go through the
  batched GEMM, padded to 16×16 tiles.
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

With `PALLADIUM_PJRT_PROFILE` set in the environment, every launch runs in
a command buffer of its own, and `palladium._pjrt.profile()` returns the GPU
seconds of each step of the last execution, in the order of the compiled
program's steps. The sum exceeds the normal GPU time, by roughly 0.1 ms per
launch on an M2, so it ranks large launches and overstates small ones.

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
