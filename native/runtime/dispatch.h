#pragma once
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "buffer.h"
#include "ns_ptr.h"

namespace MTL {
class Device;
class CommandBuffer;
class ComputeCommandEncoder;
class ComputePipelineState;
class Function;
}  // namespace MTL

class MetalRuntime;

// A 1-, 2- or 3-dimensional extent; unused dimensions are 1.
struct Dim3 {
    size_t x = 1, y = 1, z = 1;
    size_t volume() const { return x * y * z; }
};

// A kernel argument the compiler reports as used.
struct BindingInfo {
    size_t index;
    std::string name;
    size_t data_size = 0;  // bytes per element of a buffer binding; 0 if unknown
};

class ComputePipeline {
   public:
    // `label` is the function name, used in error messages.
    ComputePipeline(MTL::Device* device, MTL::Function* function, const std::string& label);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&&) = delete;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    MTL::ComputePipelineState* handle() const { return pipeline_.get(); }
    const std::string& label() const { return label_; }

    // This kernel's limits, which register pressure can push below the device's.
    size_t max_threads_per_threadgroup() const { return max_threads_per_threadgroup_; }
    size_t thread_execution_width() const { return thread_execution_width_; }
    size_t static_threadgroup_memory_length() const { return static_threadgroup_memory_length_; }

    // Used buffer and [[threadgroup(i)]] arguments.
    const std::vector<BindingInfo>& buffer_bindings() const { return buffer_bindings_; }
    const std::vector<BindingInfo>& threadgroup_bindings() const { return threadgroup_bindings_; }

    // Whole SIMD groups up to the kernel's ceiling.
    Dim3 default_threadgroup(Dim3 grid) const;

   private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    std::string label_;
    std::vector<BindingInfo> buffer_bindings_;
    std::vector<BindingInfo> threadgroup_bindings_;

    size_t max_threads_per_threadgroup_ = 0;
    size_t thread_execution_width_ = 0;
    size_t static_threadgroup_memory_length_ = 0;
};

// One kernel launch. Buffers bind at indices 0..n-1, scalars (setBytes) at
// the indices after.
struct Launch {
    struct BufferBinding {
        Buffer* buffer;
        size_t offset;  // bytes
    };
    struct Scalar {
        const void* data;
        size_t size;
    };

    ComputePipeline* pipeline = nullptr;
    std::vector<BufferBinding> buffers;
    std::vector<Scalar> scalars;
    std::vector<size_t> threadgroup_memory;
    Dim3 grid;
    Dim3 threadgroup;
};

// Several launches in one command buffer, run in order. Methods are synchronized.
class CommandBatch {
   public:
    explicit CommandBatch(MetalRuntime& rt);
    ~CommandBatch();
    CommandBatch(CommandBatch&&) = delete;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    // Validates, then encodes. Throws before touching the encoder if invalid.
    void add(const Launch& launch);

    // Submits without blocking.
    void commit();

    // Commits if needed, then blocks until done. Throws DispatchError on a
    // faulted command buffer. Every caller blocks, however many times.
    void wait();

    // Device-side execution seconds for the whole batch; set by wait().
    std::optional<double> gpu_time() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return gpu_time_;
    }

   private:
    void commit_locked();

    // Declaration order: the encoder is released before its command buffer.
    NS::SharedPtr<MTL::CommandBuffer> command_buffer_;
    NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
    mutable std::mutex state_mutex_;
    bool committed_ = false;
    bool non_uniform_ = false;
    size_t max_threadgroup_memory_ = 0;
    std::optional<double> gpu_time_;
};

void dispatch(MetalRuntime& rt, const Launch& launch);
