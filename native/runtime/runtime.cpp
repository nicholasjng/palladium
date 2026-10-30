#include "runtime.h"

#include <string>

#include "errors.h"
#include "metal.h"

MetalRuntime::MetalRuntime() : device_(NS::TransferPtr(MTL::CreateSystemDefaultDevice())) {
    if (!device_) {
        throw NoDeviceError(
            "no Metal device found: this runtime needs macOS with a "
            "Metal-capable GPU");
    }
    queue_ = NS::TransferPtr(device_->newCommandQueue());
    if (!queue_) {
        throw NoDeviceError("could not create a Metal command queue on this device");
    }
    non_uniform_threadgroups_ = device_->supportsFamily(MTL::GPUFamilyApple4) ||
                                device_->supportsFamily(MTL::GPUFamilyMac2);
    max_threadgroup_memory_ = (size_t)device_->maxThreadgroupMemoryLength();
    for (int family = 20; family > 0 && !apple_gpu_family_; --family) {
        if (device_->supportsFamily((MTL::GPUFamily)(MTL::GPUFamilyApple1 - 1 + family))) {
            apple_gpu_family_ = family;
        }
    }
}

MetalRuntime::~MetalRuntime() = default;

std::string MetalRuntime::device_name() const {
    AutoreleaseScope scope;
    return std::string(device_->name()->utf8String());
}

bool MetalRuntime::has_unified_memory() const { return device_->hasUnifiedMemory(); }

size_t MetalRuntime::recommended_max_working_set_size() const {
    return (size_t)(device_->recommendedMaxWorkingSetSize());
}

size_t MetalRuntime::max_threads_per_threadgroup() const {
    return (size_t)(device_->maxThreadsPerThreadgroup().width);
}

size_t MetalRuntime::max_buffer_length() const { return (size_t)(device_->maxBufferLength()); }

MetalRuntime& runtime() {
    static MetalRuntime rt;
    return rt;
}
