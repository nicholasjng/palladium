#include "library.h"

#include "dispatch.h"
#include "errors.h"
#include "metal.h"

namespace {

MTL::MathMode to_mtl(MathMode mode) {
    switch (mode) {
        case MathMode::Safe:
            return MTL::MathModeSafe;
        case MathMode::Relaxed:
            return MTL::MathModeRelaxed;
        case MathMode::Fast:
            break;
    }
    return MTL::MathModeFast;
}

// Autoreleased into the enclosing AutoreleaseScope.
MTL::CompileOptions* build_options(MathMode math_mode) {
    MTL::CompileOptions* mtl_options = MTL::CompileOptions::alloc()->init()->autorelease();
    if (__builtin_available(macOS 26.0, *)) {
        mtl_options->setLanguageVersion(MTL::LanguageVersion4_0);
    }
    if (__builtin_available(macOS 15.0, *)) {
        mtl_options->setMathMode(to_mtl(math_mode));
    } else {
        // mathMode is macOS 15+; the legacy boolean still keeps SAFE exact.
        mtl_options->setFastMathEnabled(math_mode != MathMode::Safe);
    }
    return mtl_options;
}

}  // namespace

Library::Library(MTL::Device* device, const std::string& msl_source, MathMode math_mode)
    : device_(device) {
    AutoreleaseScope scope;
    NS::Error* error = nullptr;
    NS::String* source = to_ns_string(msl_source, "MSL source");
    library_ = NS::TransferPtr(device->newLibrary(source, build_options(math_mode), &error));
    if (!library_) {
        throw MSLCompileError("MSL compile error: " + describe(error));
    }
}

Library::~Library() = default;

std::shared_ptr<ComputePipeline> Library::pipeline(const std::string& name) const {
    AutoreleaseScope scope;
    NS::SharedPtr<MTL::Function> fn =
        NS::TransferPtr(library_->newFunction(to_ns_string(name, "function name")));
    if (!fn) throw MSLFunctionNotFoundError("no such MSL function: " + name);
    return std::make_shared<ComputePipeline>(device_, fn.get(), name);
}
