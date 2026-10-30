#pragma once
#include <stdexcept>

#include "export.h"

// Base of every runtime error; MetalError in Python.
struct MR_API MetalError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// No Metal device on this machine, or no command queue on it.
struct MR_API NoDeviceError : MetalError {
    using MetalError::MetalError;
};

struct MR_API MSLCompileError : MetalError {
    using MetalError::MetalError;
};

// The source compiled but has no such entry point.
struct MR_API MSLFunctionNotFoundError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// newComputePipelineState failed: the source compiled, the back end for this GPU did not.
struct MR_API PipelineBuildError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// The GPU rejected or aborted a committed command buffer.
struct MR_API DispatchError : MetalError {
    using MetalError::MetalError;
};

// Starting or stopping a GPU trace capture failed.
struct MR_API CaptureError : MetalError {
    using MetalError::MetalError;
};

// Metal refused to allocate or wrap memory. MemoryError in Python, so not a MetalError.
struct MR_API AllocationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
