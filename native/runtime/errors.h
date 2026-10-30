#pragma once
#include <stdexcept>

// Base of every runtime error but AllocationError.
struct MetalError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// No Metal device on this machine, or no command queue on it.
struct NoDeviceError : MetalError {
    using MetalError::MetalError;
};

struct MSLCompileError : MetalError {
    using MetalError::MetalError;
};

// The source compiled but has no such entry point.
struct MSLFunctionNotFoundError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// newComputePipelineState failed: the source compiled, the back end for this GPU did not.
struct PipelineBuildError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// The GPU rejected or aborted a committed command buffer.
struct DispatchError : MetalError {
    using MetalError::MetalError;
};

// Metal refused to allocate or wrap memory; MR_ERROR_ALLOCATION in the C API.
struct AllocationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
