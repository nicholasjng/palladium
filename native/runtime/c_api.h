// C entry point into metal-runtime for callers without Python or the GIL,
// such as an XLA FFI handler. Opaque handles and POD structs only.
//
// Compile once (mr_compile_library + mr_get_pipeline), hold the handles for
// the process lifetime, release at teardown.
#pragma once

#include <stddef.h>

#if defined(__GNUC__)
#define MR_EXPORT __attribute__((visibility("default")))
#else
#define MR_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MRLibrary MRLibrary;
typedef struct MRPipeline MRPipeline;
typedef struct MRBuffer MRBuffer;
typedef struct MRBatch MRBatch;

typedef enum MRStatus {
    MR_OK = 0,
    MR_ERROR_NO_DEVICE = 1,
    MR_ERROR_COMPILE = 2,
    MR_ERROR_FUNCTION_NOT_FOUND = 3,
    MR_ERROR_DISPATCH = 4,
    MR_ERROR_INVALID_ARGUMENT = 5,
    MR_ERROR_UNKNOWN = 6,
    MR_ERROR_ALLOCATION = 7,
} MRStatus;

// Mirrors library.h's MathMode. FAST (Metal's default) permits reassociation;
// compensated arithmetic needs SAFE.
typedef enum MRMathMode {
    MR_MATH_MODE_SAFE = 0,
    MR_MATH_MODE_RELAXED = 1,
    MR_MATH_MODE_FAST = 2,
} MRMathMode;

// No-op on NULL. Every *out_err_msg below is untouched on success.
MR_EXPORT void mr_free_error_message(char* msg);

MR_EXPORT MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len,
                                      MRMathMode math_mode, MRLibrary** out_library,
                                      char** out_err_msg);
MR_EXPORT void mr_release_library(MRLibrary* library);

// Cached per library and function name.
MR_EXPORT MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name,
                                   MRPipeline** out_pipeline, char** out_err_msg);
MR_EXPORT void mr_release_pipeline(MRPipeline* pipeline);

// Limits of the device every mr_* call runs on. Metal has no device-wide SIMD
// width; see MRPipelineInfo.
typedef struct MRDeviceInfo {
    size_t max_threads_per_threadgroup;
    size_t max_threadgroup_memory_length;  // bytes
    size_t max_buffer_length;              // bytes
    size_t recommended_max_working_set_size;
    int has_unified_memory;
    int supports_non_uniform_threadgroups;  // else the grid must divide by the threadgroup
} MRDeviceInfo;

MR_EXPORT MRStatus mr_device_info(MRDeviceInfo* out_info, char** out_err_msg);

// One kernel's limits, which register pressure can push below the device's.
typedef struct MRPipelineInfo {
    size_t thread_execution_width;  // the SIMD-group width
    size_t max_threads_per_threadgroup;
    size_t static_threadgroup_memory_length;  // bytes the kernel declares itself
} MRPipelineInfo;

MR_EXPORT MRStatus mr_pipeline_info(const MRPipeline* pipeline, MRPipelineInfo* out_info,
                                    char** out_err_msg);

// Wraps `ptr` for mr_dispatch without copying: the range is rounded out to
// page boundaries (a Metal requirement) and dispatch binds at the true
// offset. Falls back to an owned copy, synced back by mr_buffer_flush_to,
// only if the rounded range is not all mapped read-write memory. The caller
// keeps owning `ptr`.
//
// Wraps of 1 MB or more are cached and reused while the range keeps the VM
// objects it was wrapped over, which saves the GPU's page-table setup on
// repeated calls. A cached wrap keeps its pages alive after the caller frees
// them, until the next wrap call notices or mr_clear_wrap_cache runs. The
// budget is a quarter of the recommended working set; MR_WRAP_CACHE_BYTES
// overrides it, and 0 disables the cache.
MR_EXPORT MRStatus mr_wrap_buffer(void* ptr, size_t size_bytes, MRBuffer** out_buffer,
                                  char** out_err_msg);

// Copies the contents back to `ptr` after a dispatch. No-op for a zero-copy wrap.
MR_EXPORT MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg);

// Never frees the wrapped pointer.
MR_EXPORT void mr_release_buffer(MRBuffer* buffer);

// Releases every cached wrap (see mr_wrap_buffer).
MR_EXPORT void mr_clear_wrap_cache(void);

// One kernel launch. buffers/buffer_offsets and scalars/scalar_sizes are
// parallel arrays. threadgroup_x == 0 picks a default threadgroup.
typedef struct MRLaunchDesc {
    MRPipeline* pipeline;
    MRBuffer* const* buffers;
    const size_t* buffer_offsets;
    size_t buffer_count;
    const void* const* scalars;
    const size_t* scalar_sizes;
    size_t scalar_count;
    const size_t* threadgroup_memory;
    size_t threadgroup_memory_count;
    size_t grid_x, grid_y, grid_z;
    size_t threadgroup_x, threadgroup_y, threadgroup_z;
} MRLaunchDesc;

// Blocks until the launch completes.
MR_EXPORT MRStatus mr_dispatch(const MRLaunchDesc* launch, char** out_err_msg);

// Launches in one command buffer: create, add each launch (the descriptor is
// copied, so its arrays may change afterwards), commit, wait. Commit submits
// without blocking. Releasing without waiting leaves faults unreported.
// Metal's queue holds at most 64 command buffers, committed or not: past that,
// mr_batch_create and mr_dispatch block until one completes or is released.
MR_EXPORT MRStatus mr_batch_create(MRBatch** out_batch, char** out_err_msg);
MR_EXPORT MRStatus mr_batch_add(MRBatch* batch, const MRLaunchDesc* launch, char** out_err_msg);
MR_EXPORT MRStatus mr_batch_commit(MRBatch* batch, char** out_err_msg);

// Commits if needed, then blocks; a faulted command buffer is MR_ERROR_DISPATCH.
MR_EXPORT MRStatus mr_batch_wait(MRBatch* batch, char** out_err_msg);

MR_EXPORT void mr_release_batch(MRBatch* batch);

#ifdef __cplusplus
}  // extern "C"
#endif
