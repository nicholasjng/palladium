#include "buffer.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "errors.h"
#include "metal.h"

// newBuffer(0) returns nullptr; empty buffers get one byte.
Buffer::Buffer(MTL::Device* device, size_t size_bytes)
    : buffer_(NS::TransferPtr(
          device->newBuffer(std::max<size_t>(size_bytes, 1), MTL::ResourceStorageModeShared))),
      size_(size_bytes) {
    if (!buffer_) {
        throw AllocationError("failed to allocate Metal buffer of " + std::to_string(size_bytes) +
                              " bytes");
    }
}

Buffer::Buffer(MTL::Device* device, void* external_ptr, size_t size_bytes) : size_(size_bytes) {
    size_t page_size = (size_t)getpagesize();
    if ((uintptr_t)external_ptr % page_size != 0 || size_bytes % page_size != 0) {
        throw std::invalid_argument(
            "Buffer: external_ptr and size_bytes must both be a multiple of the page size (" +
            std::to_string(page_size) + " bytes) to wrap without copying");
    }
    buffer_ = NS::TransferPtr(
        device->newBuffer(external_ptr, size_bytes, MTL::ResourceStorageModeShared, nullptr));
    if (!buffer_) {
        throw AllocationError("failed to wrap external memory of " + std::to_string(size_bytes) +
                              " bytes as a Metal buffer");
    }
}

Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&& other) noexcept = default;

void* Buffer::contents() const { return buffer_->contents(); }
