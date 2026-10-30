#pragma once
#include <cstddef>

#include "export.h"
#include "ns_ptr.h"

namespace MTL {
class Device;
class Buffer;
}  // namespace MTL

// Shared storage: host and GPU read and write the same bytes.
class MR_API Buffer {
   public:
    Buffer(MTL::Device* device, size_t size_bytes);

    // Wraps caller-owned, page-aligned memory without copying. The caller
    // keeps it alive and frees it after this Buffer is destroyed.
    Buffer(MTL::Device* device, void* external_ptr, size_t size_bytes);

    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* contents() const;

    // Requested size; an empty buffer reports 0.
    size_t size() const { return size_; }
    MTL::Buffer* handle() const { return buffer_.get(); }

   private:
    NS::SharedPtr<MTL::Buffer> buffer_;
    size_t size_ = 0;
};
