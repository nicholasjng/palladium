# Emitter walkthrough

The emitter translates a `KernelSpec` into self-contained Metal Shading
Language source. It does not own input arrays, compile a Metal library, or
submit GPU work. Both dispatch integrations consume the same emitted source.

Start with [`emit_msl_stats`](../src/palladium/emit/kernel.py), then read
[`core.py`](../src/palladium/emit/core.py),
[`addressing.py`](../src/palladium/emit/addressing.py), and the rule for the
primitive you are following. [Architecture](architecture.md) explains how
JAX produces the specification and where dispatch happens.

## 1. Choose the execution model

`emit_msl` delegates to `emit_msl_stats` and returns only its source result.
The latter picks one of four lowerings:

- a `plmtl.kernel` on a mesh goes to the mesh-program lowering
  ([`mesh.py`](../src/palladium/emit/mesh.py)): one threadgroup per program,
  with dots on SIMD-group matrix fragments;
- `program="threadgroup"` goes to the row-program lowering
  ([`threadgroup.py`](../src/palladium/emit/threadgroup.py)): one threadgroup
  per tile of rows, each thread owning some columns of a row;
- a kernel with a dot may go to the cooperative TensorOps compiler, depending
  on the dot policy. A scan-containing candidate goes to attention
  recognition; otherwise it goes to matmul recognition. Under `"auto"`, an
  `EmitError` from recognition or cooperative emission causes primitive
  fallback;
- everything else, and that fallback, is primitive emission: one Metal thread
  per program. Sections 2 to 5 describe it.

Mesh and row programs raise `EmitError` for kernels outside their vocabulary;
they have no fallback.

Primitive emission runs once with ordinary input snapshots. If declared
thread storage reaches 512 bytes, it emits again with eligible input-load
fusion enabled. The second pass can replace single-read immutable input
copies with direct device reads. This is a resource heuristic, not a proof
about the Metal compiler's register allocation.

## 2. Assemble parameters and program-local Refs

Primitive emission binds inputs, then outputs, to successive `[[buffer(k)]]`
indices. Inputs use `const device` pointers and outputs use writable `device`
pointers. `_pid` is `thread_position_in_grid`, so each thread runs an entire
Pallas program, including its serial loops and block operations.

Each Ref starts as a pointer to its block's origin, computed from the
`BlockSpec` index map:

```text
element_offset = sum(block_index[d] * full_block_extent[d] * array_stride[d])
```

The map is itself a jaxpr, lowered by the same rule walker, and a map of
constants folds to a literal offset. `flat_index` renders such sums of
index-times-stride terms, dropping zero terms and unit factors, for block
origins, element addresses and Ref views alike.

A block that spans the array's trailing dimensions is contiguous from that
pointer. Any other block, which includes every two-dimensional tile, also
carries a template that maps a logical element index to its offset from the
origin. A block that reaches past the end of its array additionally carries
a bounds predicate: invalid loads use dtype-specific Pallas padding and
invalid stores are skipped. Blocks inside the array need no predicate and can
be read and written in place like contiguous ones, element by element. Ref
indexing composes scalar indices and positive static slice strides with
these layouts. The guards support partial tiles; they do not establish
safety for arbitrary user-supplied out-of-range indices.

Scratch Refs become uninitialized per-thread declarations. They follow input
and output Refs in the kernel body's operand order and must be written before
being read.

## 3. Walk the jaxpr using three small abstractions

| Abstraction | What it holds | Why it matters |
|---|---|---|
| `CVal` | Expression, logical shape, MSL type, address space, readonly status, alignment, and optional layout/lazy-expression templates | Lets rules reason about how to access a value without losing storage facts |
| `Environment` | Var-to-CVal bindings, producer/consumer information, escaping values, inlined-JIT output links, and the number of bindings holding each declared variable | Enables local optimizations while checking whether other uses can observe them |
| `Cursor` | Indented source lines, deterministic fresh names, helper functions, and declared-storage counters | Gives primitive and cooperative emitters a common way to write typed MSL |

`emit_jaxpr` binds inputs, rebuilds use information for that jaxpr level, and
dispatches each equation through `RULES[eqn.primitive.name]`. A `@rule(...)`
decorator registers implementations when the rule modules are imported.
Missing rules raise `UnsupportedPrimitiveError` with the primitive name as a
field. An unsupported case of an existing rule raises `EmitError`.

Literal atoms become typed scalar expressions. Captured array constvars are
rejected: arrays should be explicit kernel operands. Nested JIT bodies are
walked in the same output scope; their use maps must be rebuilt because JAX can
reuse a cached body and its Var identities across calls.

## 4. Lower the primitive families

| Module | Main lowering behavior |
|---|---|
| [`memory.py`](../src/palladium/emit/rules/memory.py) | Ref get/swap, snapshot preservation, eligible readonly views, and JIT inlining |
| [`elementwise.py`](../src/palladium/emit/rules/elementwise.py) | Scalar templates, broadcasting, lazy expression fusion, and eligible direct output computation |
| [`array.py`](../src/palladium/emit/rules/array.py) | Reshape/squeeze views, permuted copies, transpose, multiway selection, and broadcasts |
| [`structural.py`](../src/palladium/emit/rules/structural.py) | Iotas, cumulative operations, concatenation, and clamped dynamic slices |
| [`dot.py`](../src/palladium/emit/rules/dot.py) | Serial matrix/vector contractions, selected vectorized cases, and reductions over kept/reduced axes |
| [`control.py`](../src/palladium/emit/rules/control.py) | Scan/for loops, while loops, branches, carry storage, and scan-output streaming |
| [`random.py`](../src/palladium/emit/rules/random.py) | Threefry-2x32-20 hashing for 32-bit random bits and fold-in |

[`numeric.py`](../src/palladium/emit/numeric.py) centralizes scalar expressions
used by these rules and cooperative epilogues. Dtype-aware variants preserve
integer boundaries, shift behavior, extrema identities, and requested narrow
intermediates. Helper functions implement operations absent from MSL or improve
small-argument accuracy, and are emitted once per cursor. IEEE edge behavior
still depends on the compilation math mode: FAST permits transformations that
SAFE prevents.

## 5. Preserve state and control-flow semantics

Several optimizations depend on observable reads and writes, not just shapes:

- **Snapshot reads:** a loaded mutable Ref must keep the values observed at the
  read, even if the Ref changes later. A pointer view is safe only when the
  relevant storage remains immutable. Aliased input/output buffers disable
  optimizations that would otherwise conceal writes through a second Ref.
- **Lazy expressions:** eligible intermediate elementwise results stay as
  templates evaluated at a flat index, saving temporary arrays. Use counts,
  matching shapes, escapes, and consumer capabilities determine eligibility.
- **Direct outputs:** a value may compute into a device output when its sole
  full-block store and the absence of other Ref uses make earlier writes
  unobservable. Indexed, aliased, or incompatible targets use local storage.
- **Loop carries:** iteration outputs must update simultaneously. Carry
  permutations require snapshot/copy ordering; the implementation also uses
  volatile accesses at specific alias hazards to avoid a Metal optimizer
  miscompile covered by regression tests.
- **Scan streaming:** stacked ys can be written into their output slots each
  iteration instead of accumulating a full local trajectory. Full-block xs
  consumed only as scanned inputs can be read as device views. Other uses,
  constants, or aliasing may require local snapshots.

Static for loops arrive as scans, whose operands split into constants,
carries, and scanned values by the scan's own input tree (`scan_operands`). A
carry takes over its initial value's thread-local variable when no other
binding holds that variable (`Environment.storage_users`), and gets a copy
otherwise. Reverse scans reverse iteration while preserving stacked-output
positions. While loops emit condition evaluation and a break inside a loop;
conditionals emit branch chains. Each independent program can take a different
branch or trip count.

## 6. Recognize cooperative TensorOps kernels

The cooperative path is a specialized compiler for known dataflow and layout
patterns. It emits MSL using Metal Performance Primitives TensorOps and runs
one program across four SIMD groups, rather than translating each block
operation into work performed by one thread.

**Matmul** recognition checks the input reads, one matrix contraction, supported
epilogue chain, output store, and Pallas block maps. Rank-2 matrices and matching
rank-3 batches are supported with matching float32, float16, or bfloat16
buffers. Each Pallas tile spans the full K dimension; the generated kernel
chunks K internally, handles its tail, and masks partial output tiles. Half
and bfloat products accumulate in float and narrow on output. Transposed
rank-2 operands and supported scalar/residual/bias epilogues have dedicated
handling. Scratch, aliases, arbitrary batch broadcasting, and unmatched
dataflow are rejected by this recognizer.

**Attention** recognition follows dependencies through a forward key-tile scan:
score matmul, scaling/masking, online max and sum updates, probability formation,
value matmul, and final normalization. Merely containing the right primitive
names is insufficient. Reordering independent equations preserves recognition;
miswiring probability operands fails it. The current contract uses rank-4
float32 `[batch, sequence, heads, D]` buffers, standard block maps, and tile/head
dimensions divisible by 16. Lengths must tile evenly, and causal attention
requires equal query/key lengths.

The attention emitter stores the score tile and row state in shared memory,
assigns rows to SIMD groups and columns to lanes, and retains the value
accumulator in an MPP cooperative tensor. Barriers separate phases that share
data. Typed tensor views and operation descriptors live in
[`tensorops/_shared.py`](../src/palladium/emit/tensorops/_shared.py);
[`softmax.py`](../src/palladium/emit/tensorops/softmax.py) emits the row updates.

Cooperative source, like mesh and row programs, uses
`threadgroup_position_in_grid` and declares its group width in a
`// palladium: threadgroup width N` line. `launch_geometry` reads that line,
fixes the threadgroup to that width, and scales the thread grid's first axis.
Every dispatch path uses that calculation, including after an automatic
fallback to primitive emission.

## 7. Read the output and validate a change

For a small kernel, this is the shortest path from Python to inspectable MSL:

```python
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import palladium


def saxpy(x, y, out):
    out[...] = 2.0 * x[...] + y[...]


shape = jax.ShapeDtypeStruct((8,), jnp.float32)
block = pl.BlockSpec((1,), lambda i: (i,))
print(
    palladium.debug_msl(
        saxpy,
        shape,
        shape,
        grid=(8,),
        in_specs=(block, block),
        out_specs=block,
        out_shape=shape,
    )
)
```

Follow the two input buffer bindings, `_pid.x`-derived offsets, scalar loads,
multiply/add expression, and output store. Then compare a gridless block to
see how a single program processes multiple elements serially. A tiled matmul
shows the switch to threadgroup indexing and MPP operations.

When adding a primitive rule, document its shape/dtype limits, preserve Ref
snapshots and address spaces, and compare behavior against the interpreter
or an independent reference. Source assertions explain
which optimization or layout was selected; device execution establishes that
the source compiles and computes correctly. Declared storage statistics help
explain resource failures but are not compiler-measured stack or register use.
