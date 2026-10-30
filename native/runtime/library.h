#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

#include "dtype.h"
#include "export.h"
#include "lru.h"
#include "ns_ptr.h"

namespace MTL {
class Device;
class Library;
class Function;
}  // namespace MTL

class ComputePipeline;

// `Fast` (Metal's default) permits reassociation, which silently deletes
// compensated-arithmetic error terms; such code needs `Safe`.
enum class MathMode { Safe, Relaxed, Fast };

// Part of the library cache key.
struct MR_API CompileOptions {
    MathMode math_mode = MathMode::Fast;
    std::map<std::string, std::string> defines;  // preprocessor macros

    std::string cache_key() const;
};

// One MSL function constant, baked in at pipeline build without recompiling the
// library. bool, int64_t and double coerce to the declared type; an ExactScalar
// (a numpy scalar) must match it.
struct FunctionConstant {
    struct ExactScalar {
        DType dtype;
        std::array<uint8_t, 8> bytes{};  // little-endian
    };

    std::string name;
    std::variant<bool, int64_t, double, ExactScalar> value;
};
using FunctionConstants = std::vector<FunctionConstant>;

// MSL source compiled at runtime via newLibrary.
class MR_API Library {
   public:
    Library(MTL::Device* device, const std::string& msl_source, const CompileOptions& options = {});
    ~Library();
    Library(Library&&) = delete;
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // Cached per (name, constants as given). Throws MSLFunctionNotFoundError
    // for an unknown name, MSLCompileError for constants that don't match
    // the function's declaration.
    std::shared_ptr<ComputePipeline> pipeline_for(const std::string& name,
                                                  const FunctionConstants& constants = {});

   private:
    // Validates `constants` against reflection, then specializes.
    NS::SharedPtr<MTL::Function> create_specialized(const std::string& name,
                                                    const FunctionConstants& constants) const;

    MTL::Device* device_ = nullptr;  // borrowed
    NS::SharedPtr<MTL::Library> library_;

    static constexpr size_t kMaxCachedPipelines = 128;
    std::mutex mutex_;
    LruCache<ComputePipeline> pipelines_{kMaxCachedPipelines};
};
