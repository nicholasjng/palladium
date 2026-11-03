# API guide

The supported user entry points are exported by `palladium.__all__`.
Their NumPy-style docstrings describe parameters, results, errors, and edge
cases; use `help(palladium.metal_call)` or an editor's documentation view for
the full callable contract. [Getting started](getting-started.md) provides
complete usage examples.

## Kernel construction and inspection

| API | Purpose | Important boundary |
|---|---|---|
| `metal_call(kernel, **pallas_kwargs)` | Construct a CPU JAX callable dispatching a Pallas kernel to Metal | Native FFI and a Metal device are required at execution; the PJRT plugin is not |
| `plmtl.kernel(body, out_type, *, mesh, scratch_types=())` | Like `pl.kernel` on a `plmtl.Mesh` of threadgroups, with `plmtl.THREADGROUP` scratch, dispatched like `metal_call` | Use `pl.program_id` for mesh coordinates; dots take threadgroup tiles, `a @ b.T` included; vectors broadcast along rows or columns; runs from CPU programs and inside programs on the `metal` plugin |
| `CompilerParams(...)` | Select dot policy, threadgroup dimensions, and math mode | Cooperative kernels fix their group size |
| `debug_msl(kernel, *example_args, **pallas_kwargs)` | Return source directly from a kernel and shape/dtype examples | Traces and emits; no Metal compilation or dispatch |
| `trace(pallas_fn, *example_args)` | Extract a reusable `KernelSpec` | Exactly one Pallas call, including through JIT wrappers |
| `emit_msl(spec, kernel_name=None, *, dot_general="auto")` | Generate complete source from a specification | Launch geometry must match the emitted primitive/cooperative model |
| `with_vjp(forward, backward, *, residuals=0)` | Supply an explicit reverse-mode derivative | Backward receives primals, residuals, then public-output cotangents |

Inside programs on the `metal` PJRT plugin (`JAX_PLATFORMS=metal,cpu`), use a
plain `pl.pallas_call`: the plugin registers Palladium's lowering for the
platform when it loads. Pass `CompilerParams` through Pallas's
`compiler_params` keyword. Both paths emit the same `palladium_dispatch`
custom call; see [architecture](architecture.md).

The `FfiCallable` returned by `metal_call` exposes:

- `call(*arrays)`: return one JAX array or a tuple for multiple outputs;
  specialization is cached by input shape and dtype.
- `call.interpret(*arrays)`: run the same Pallas kernel through the interpreter.
- `call.explain(*examples)`: return `KernelDiagnostics` after tracing and
  emitting, without compiling or dispatching Metal work.

Do not pass `interpret` to `metal_call`; use its returned interpreter callable.
FAST math may change rounding, reassociation, and special-value behavior;
use SAFE through CPU FFI when required, especially for compensated arithmetic.

## Metadata records

| API | Information |
|---|---|
| `KernelSpec` | Kernel jaxpr, normalized grid, inputs, outputs, scratch, and validated alias pairs |
| `BlockInfo` | Logical tile shape, backing array shape/dtype, staged index map, and unsqueezed block axes |
| `ScratchInfo` | Shape and dtype of an uninitialized, private scratch buffer |
| `KernelDiagnostics` | Launch dimensions, source line count, declared storage, available shared-memory limit, and cooperative status |

Prefer obtaining specifications through `trace` and diagnostics through
`explain`. Constructing these dataclasses directly does not repeat all tracing
or backend validation. Diagnostic storage counts describe declarations rather
than liveness or final compiler resource allocation.

## Exceptions

All Palladium-defined validation exceptions derive from `PalladiumError`:

| Exception | Stage and meaning | Additional base |
|---|---|---|
| `TraceError` | Unsupported call structure, effects, layouts, aliases, or obvious write collisions | `ValueError` |
| `EmitError` | Unsupported lowering case, addressing, shape, or checked resource limit | — |
| `UnsupportedPrimitiveError` | Missing primitive rule; inspect its `primitive` field | `EmitError`, `NotImplementedError` |
| `DispatchError` | Unsupported input dtype on the CPU FFI path | `TypeError` |

JAX, option validation, native compilation, and runtime execution may also
raise their own exceptions. Catching `PalladiumError` covers Palladium's
defined hierarchy rather than every possible failure in the toolchain.

## Compiler extension interfaces

Submodules expose typed building blocks for compiler work: `EmitStats`,
`emit_msl_stats`, `CVal`, `Cursor`, `Environment`, addressing helpers and
rule registration. Their docstrings document the
required invariants. Start with the [emitter guide](emitter.md) before reusing
them: they consume staged JAX internals and are coupled to the supported JAX
version and emitted source format.
