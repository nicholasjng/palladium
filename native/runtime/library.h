#pragma once
#include <memory>
#include <string>

#include "ns_ptr.h"

namespace MTL {
class Device;
class Library;
}  // namespace MTL

class ComputePipeline;

// `Fast` (Metal's default) permits reassociation, which silently deletes
// compensated-arithmetic error terms; such code needs `Safe`.
enum class MathMode { Safe, Relaxed, Fast };

// MSL source compiled at runtime via newLibrary.
class Library {
   public:
    Library(MTL::Device* device, const std::string& msl_source, MathMode math_mode);
    ~Library();
    Library(Library&&) = delete;
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // A new pipeline per call; callers cache. Throws MSLFunctionNotFoundError
    // for an unknown name.
    std::shared_ptr<ComputePipeline> pipeline(const std::string& name) const;

   private:
    MTL::Device* device_ = nullptr;  // borrowed
    NS::SharedPtr<MTL::Library> library_;
};
