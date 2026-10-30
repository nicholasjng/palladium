#pragma once
#include <cstddef>
#include <string>

#include "ns_ptr.h"

namespace MTL {
class Device;
class CommandQueue;
}  // namespace MTL

// The process's Metal device and command queue.
class MetalRuntime {
   public:
    MetalRuntime();
    ~MetalRuntime();
    MetalRuntime(MetalRuntime&&) = delete;
    MetalRuntime(const MetalRuntime&) = delete;
    MetalRuntime& operator=(const MetalRuntime&) = delete;

    MTL::Device* device() const { return device_.get(); }
    MTL::CommandQueue* queue() const { return queue_.get(); }

    std::string device_name() const;
    bool has_unified_memory() const;
    size_t recommended_max_working_set_size() const;
    size_t max_threads_per_threadgroup() const;
    size_t max_buffer_length() const;

    size_t max_threadgroup_memory_length() const { return max_threadgroup_memory_; }

    // dispatchThreads() needs Apple family 4+ or Mac family 2; older GPUs
    // take dispatchThreadgroups, which needs the grid to divide evenly.
    bool supports_non_uniform_threadgroups() const { return non_uniform_threadgroups_; }

    // Highest N for which the device supports MTLGPUFamilyAppleN (7 for M1, 8
    // for M2, 9 for M3 and M4, 10 for M5), or 0.
    // Kernels choose between matrix instructions by it.
    int apple_gpu_family() const { return apple_gpu_family_; }

   private:
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::CommandQueue> queue_;
    bool non_uniform_threadgroups_ = false;
    int apple_gpu_family_ = 0;
    size_t max_threadgroup_memory_ = 0;
};

MetalRuntime& runtime();
