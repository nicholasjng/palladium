# Packaging the Metal runtime inside palladium (done 2026-10-30)

palladium used one part of metal-runtime, its C API, which the jax.ffi
handler linked as a shared library found through the Python package. That
made distribution need two packages of matching versions, and changes on
both sides of the API two commits in two repositories.

What was done:

- The runtime core and C API live in `native/runtime` (from metal-runtime
  82f39a0 plus its wrap cache and GPU family query), built by palladium's
  CMake as the static `palladium::runtime` target and linked into
  `libpalladium_ffi`, which exports the `mr_*` functions. CMake fetches
  metal-cpp itself.
- Pruned to what the C API uses: no library cache, GPU capture,
  preprocessor defines, function constants, indirect dispatch, concurrent
  encoders or batch discarding.
- `palladium.MathMode` replaces metal-runtime's enum. Device queries read
  the C API's `MRDeviceInfo` (now with the device name and Apple GPU
  family) through ctypes in `palladium._runtime`, which also has a small
  `Kernel`/`run` helper with GPU timing (`mr_batch_gpu_time`) for the
  benchmarks, hand-written-MSL baselines and the df32 example.
- metal-runtime is no longer a dependency; it stays a separate playground.

No process loads two runtimes any more, so pipeline state and the wrap
cache's budget are not duplicated.
