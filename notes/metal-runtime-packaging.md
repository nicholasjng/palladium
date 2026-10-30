# Packaging metal-runtime inside palladium (scratch)

## What palladium uses today

- Native: the FFI bridge links `libmetal_runtime.dylib` and calls only its C
  API (`mr_compile_library`, `mr_get_pipeline`, `mr_wrap_buffer`, the
  `mr_batch_*` dispatch). CMake finds the header and library through the
  Python package (`metal_runtime.c_api`); the installed bridge finds the
  dylib through an rpath into that package.
- Python: `MathMode` (launch.py, mps.py, ffi.py) and `device_info()`
  (device.py: SIMD width, threadgroup limits).
- The mps path uses none of it; jax-mps executes through MLX.
- Oracles do not use `mr.Buffer`: `metal_call` checks against
  `.interpret` (the Pallas interpreter) and NumPy. `mr.Buffer`/`mr.Kernel`
  appear only in the runtime smoke test, one load/store test, df32 (example
  and tests), the symplectic example, and the attention benchmark.

## Why

Distribution. A user needs two packages with matching versions: the bridge
requires the batch C API. Neither repo has CI or wheels; the path dependency
is fine for development, not for anyone else installing palladium.

## Steps

1. Build metal-runtime's C library in palladium's CMake from a pinned commit
   (submodule or FetchContent), statically linked into `libpalladium_ffi`.
   metal-runtime only builds a shared library today; it needs a static
   target.
2. Own `MathMode` (a three-value StrEnum with the same values).
3. Device queries: the C API has none. Add one to metal-runtime (SIMD width,
   max threads per threadgroup, max threadgroup memory), or export one from
   the bridge against metal-cpp and call it through ctypes.
4. Keep `metal-runtime` as a dev dependency for df32, the smoke test, and the
   attention benchmark (or move the benchmark to `metal_call`; its recorded
   numbers use direct dispatch).

## Costs

- palladium owns the pin and its rebuilds.
- A process that also imports `metal_runtime` loads two runtime copies with
  separate pipeline caches; macOS keeps the symbols apart, but the state is
  duplicated.

## When

When publishing a wheel. Cheapest preparation: the C API device-info
function in metal-runtime, the only non-enum thing palladium's library code
takes from the Python package.
