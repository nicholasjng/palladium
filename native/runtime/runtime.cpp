#include "runtime.h"

#include <stdexcept>
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

std::shared_ptr<Library> MetalRuntime::library_for(const std::string& msl_source,
                                                   const CompileOptions& options) {
    std::string serialized = options.cache_key();
    const std::string key = std::to_string(serialized.size()) + ":" + serialized + msl_source;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto hit = libraries_.find(key)) return hit;
    }
    // Compiled outside the lock.
    auto library = std::make_shared<Library>(device_.get(), msl_source, options);
    std::lock_guard<std::mutex> lock(mutex_);
    return libraries_.insert(key, std::move(library));
}

void MetalRuntime::set_library_cache_limit(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    libraries_.set_limit(limit);
}

size_t MetalRuntime::library_cache_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return libraries_.size();
}

void MetalRuntime::clear_library_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    libraries_.clear();
}

void MetalRuntime::start_capture(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("start_capture: output path must not be empty");

    std::lock_guard<std::mutex> lock(capture_mutex_);
    AutoreleaseScope scope;
    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (manager->isCapturing()) {
        throw CaptureError("a Metal capture is already in progress");
    }
    if (!manager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument)) {
        throw CaptureError("this system does not support GPU trace document captures");
    }

    MTL::CaptureDescriptor* descriptor = MTL::CaptureDescriptor::alloc()->init()->autorelease();
    // A device capture includes queues owned by other bindings in the process.
    descriptor->setCaptureObject(device_.get());
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    NS::String* output_path = to_ns_string(path, "capture path");
    descriptor->setOutputURL(NS::URL::fileURLWithPath(output_path));

    NS::Error* error = nullptr;
    if (!manager->startCapture(descriptor, &error)) {
        throw CaptureError("failed to start Metal capture at " + path + ": " + describe(error));
    }
    capture_started_ = true;
}

void MetalRuntime::stop_capture() {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!capture_started_) throw CaptureError("no capture started by metal_runtime is active");
    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (!manager->isCapturing()) {
        capture_started_ = false;
        throw CaptureError("the Metal capture was stopped outside metal_runtime");
    }
    manager->stopCapture();
    capture_started_ = false;
}

bool MetalRuntime::is_capturing() const {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    return MTL::CaptureManager::sharedCaptureManager()->isCapturing();
}

MetalRuntime& runtime() {
    static MetalRuntime rt;
    return rt;
}
