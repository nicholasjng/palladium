#include "library.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <type_traits>
#include <variant>
#include <vector>

#include "dispatch.h"
#include "errors.h"
#include "metal.h"

std::string CompileOptions::cache_key() const {
    // Length-prefixed fields keep the key injective.
    std::string out = "math=" + std::to_string((int)(math_mode));
    for (const auto& [name, value] : defines) {
        out += ";" + std::to_string(name.size()) + ":" + name;
        out += "=" + std::to_string(value.size()) + ":" + value;
    }
    return out;
}

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

// Everything here is autoreleased into the enclosing AutoreleaseScope.
MTL::CompileOptions* build_options(const CompileOptions& options) {
    MTL::CompileOptions* mtl_options = MTL::CompileOptions::alloc()->init()->autorelease();
    if (__builtin_available(macOS 26.0, *)) {
        mtl_options->setLanguageVersion(MTL::LanguageVersion4_0);
    }
    if (__builtin_available(macOS 15.0, *)) {
        mtl_options->setMathMode(to_mtl(options.math_mode));
    } else {
        // mathMode is macOS 15+; the legacy boolean still keeps SAFE exact.
        mtl_options->setFastMathEnabled(options.math_mode != MathMode::Safe);
    }

    if (!options.defines.empty()) {
        std::vector<NS::Object*> keys;
        std::vector<NS::Object*> values;
        keys.reserve(options.defines.size());
        values.reserve(options.defines.size());
        for (const auto& [name, value] : options.defines) {
            keys.push_back(to_ns_string(name, "preprocessor macro name"));
            values.push_back(to_ns_string(value, "preprocessor macro value"));
        }
        mtl_options->setPreprocessorMacros(
            NS::Dictionary::dictionary(values.data(), keys.data(), values.size()));
    }
    return mtl_options;
}

// The scalar MSL types a function constant can have.
struct ConstantType {
    MTL::DataType mtl;
    DType dtype;
};

constexpr ConstantType kConstantTypes[] = {
    {MTL::DataTypeBool, {DType::Bool, 8}},    {MTL::DataTypeChar, {DType::Int, 8}},
    {MTL::DataTypeShort, {DType::Int, 16}},   {MTL::DataTypeInt, {DType::Int, 32}},
    {MTL::DataTypeLong, {DType::Int, 64}},    {MTL::DataTypeUChar, {DType::UInt, 8}},
    {MTL::DataTypeUShort, {DType::UInt, 16}}, {MTL::DataTypeUInt, {DType::UInt, 32}},
    {MTL::DataTypeULong, {DType::UInt, 64}},  {MTL::DataTypeHalf, {DType::Float, 16}},
    {MTL::DataTypeFloat, {DType::Float, 32}}, {MTL::DataTypeBFloat, {DType::Bfloat, 16}},
};

const DType* constant_dtype(MTL::DataType t) {
    for (const ConstantType& entry : kConstantTypes) {
        if (entry.mtl == t) return &entry.dtype;
    }
    return nullptr;
}

struct DeclaredConstant {
    MTL::DataType type;
    bool required;
};

std::map<std::string, DeclaredConstant> read_declared(MTL::Function* fn) {
    std::map<std::string, DeclaredConstant> out;
    NS::Dictionary* dict = fn->functionConstantsDictionary();
    NS::Enumerator<NS::String>* keys = dict->keyEnumerator<NS::String>();
    while (NS::String* key = keys->nextObject()) {
        auto* constant = dict->object<MTL::FunctionConstant>(key);
        out.emplace(key->utf8String(), DeclaredConstant{constant->type(), constant->required()});
    }
    return out;
}

using ConstantBytes = std::array<uint8_t, 8>;

template <typename T>
ConstantBytes to_bytes(T value) {
    ConstantBytes out{};
    std::memcpy(out.data(), &value, sizeof(T));
    return out;
}

// `v` as a T, or an error naming the declared type if it doesn't fit.
template <typename T>
ConstantBytes int_to_bytes(int64_t v, const std::string& name, const char* msl_name) {
    bool fits;
    if constexpr (std::is_signed_v<T>) {
        fits = v >= std::numeric_limits<T>::min() && v <= std::numeric_limits<T>::max();
    } else {
        fits = v >= 0 && (uint64_t)v <= std::numeric_limits<T>::max();
    }
    if (!fits) {
        throw MSLCompileError("function constant '" + name + "' = " + std::to_string(v) +
                              " is out of range for declared type '" + msl_name + "'");
    }
    return to_bytes((T)v);
}

// The bytes of one provided constant, coerced to its declared MSL type.
ConstantBytes coerce(const FunctionConstant& c, MTL::DataType declared) {
    const DType* target = constant_dtype(declared);
    if (!target) {
        throw MSLCompileError(
            "function constant '" + c.name + "' has an unsupported MSL type (MTLDataType " +
            std::to_string((int)declared) + "); only scalar constants are supported");
    }
    const char* msl_name = msl_type_name(*target);
    auto type_error = [&](const std::string& what) {
        return MSLCompileError("function constant '" + c.name + "' is declared as '" + msl_name +
                               "', but the value is " + what);
    };

    if (const auto* exact = std::get_if<FunctionConstant::ExactScalar>(&c.value)) {
        if (exact->dtype != *target) {
            throw type_error("a '" + std::string(dtype_name(exact->dtype)) + "' scalar");
        }
        return exact->bytes;
    }
    if (const bool* b = std::get_if<bool>(&c.value)) {
        if (declared != MTL::DataTypeBool) throw type_error("a Python bool");
        return to_bytes<uint8_t>(*b ? 1 : 0);
    }
    if (const double* f = std::get_if<double>(&c.value)) {
        if (declared == MTL::DataTypeHalf)
            throw type_error("a Python float; pass numpy.float16(value)");
        if (declared != MTL::DataTypeFloat) throw type_error("a Python float");
        return to_bytes((float)*f);
    }
    int64_t v = std::get<int64_t>(c.value);
    switch (declared) {
        case MTL::DataTypeChar:
            return int_to_bytes<int8_t>(v, c.name, msl_name);
        case MTL::DataTypeShort:
            return int_to_bytes<int16_t>(v, c.name, msl_name);
        case MTL::DataTypeInt:
            return int_to_bytes<int32_t>(v, c.name, msl_name);
        case MTL::DataTypeLong:
            return int_to_bytes<int64_t>(v, c.name, msl_name);
        case MTL::DataTypeUChar:
            return int_to_bytes<uint8_t>(v, c.name, msl_name);
        case MTL::DataTypeUShort:
            return int_to_bytes<uint16_t>(v, c.name, msl_name);
        case MTL::DataTypeUInt:
            return int_to_bytes<uint32_t>(v, c.name, msl_name);
        case MTL::DataTypeULong:
            return int_to_bytes<uint64_t>(v, c.name, msl_name);
        case MTL::DataTypeFloat:
            return to_bytes((float)v);
        default:
            throw type_error("a Python int");
    }
}

// Injective, like CompileOptions::cache_key.
std::string pipeline_key(const std::string& name, const FunctionConstants& constants) {
    std::string key = std::to_string(name.size()) + ":" + name;
    for (const FunctionConstant& c : constants) {
        key += ";" + std::to_string(c.name.size()) + ":" + c.name + "=";
        if (const bool* b = std::get_if<bool>(&c.value)) {
            key += *b ? "B1" : "B0";
        } else if (const int64_t* i = std::get_if<int64_t>(&c.value)) {
            key += "I" + std::to_string(*i);
        } else if (const double* f = std::get_if<double>(&c.value)) {
            uint64_t bits;
            std::memcpy(&bits, f, sizeof(bits));
            key += "F" + std::to_string(bits);
        } else {
            const auto& exact = std::get<FunctionConstant::ExactScalar>(c.value);
            key += "E" + std::to_string((int)exact.dtype.code) + "." +
                   std::to_string((int)exact.dtype.bits) + ":";
            for (size_t i = 0; i < exact.dtype.itemsize(); ++i) {
                key += std::to_string((int)exact.bytes[i]) + ",";
            }
        }
    }
    return key;
}

}  // namespace

Library::Library(MTL::Device* device, const std::string& msl_source, const CompileOptions& options)
    : device_(device) {
    AutoreleaseScope scope;
    NS::Error* error = nullptr;
    NS::String* source = to_ns_string(msl_source, "MSL source");
    library_ = NS::TransferPtr(device->newLibrary(source, build_options(options), &error));
    if (!library_) {
        throw MSLCompileError("MSL compile error: " + describe(error));
    }
}

Library::~Library() = default;

NS::SharedPtr<MTL::Function> Library::create_specialized(const std::string& name,
                                                         const FunctionConstants& constants) const {
    AutoreleaseScope scope;
    NS::String* fn_name = to_ns_string(name, "function name");

    NS::SharedPtr<MTL::Function> probe = NS::TransferPtr(library_->newFunction(fn_name));
    if (!probe) throw MSLFunctionNotFoundError("no such MSL function: " + name);
    std::map<std::string, DeclaredConstant> declared = read_declared(probe.get());

    // No constants declared, none provided: the probe *is* the function.
    if (declared.empty() && constants.empty()) return probe;

    // Metal silently accepts an unset required constant (it specializes to an
    // undefined value) and a misspelled name, so both are checked here.
    std::string missing;
    for (const auto& entry : declared) {
        const std::string& declared_name = entry.first;
        bool provided =
            std::any_of(constants.begin(), constants.end(),
                        [&](const FunctionConstant& c) { return c.name == declared_name; });
        if (entry.second.required && !provided) {
            missing += missing.empty() ? "'" : ", '";
            missing += declared_name + "'";
        }
    }
    std::string unknown;
    for (const FunctionConstant& c : constants) {
        if (!declared.count(c.name)) {
            unknown += unknown.empty() ? "'" : ", '";
            unknown += c.name + "'";
        }
    }
    if (!missing.empty() || !unknown.empty()) {
        std::string message;
        if (!missing.empty()) {
            message += "MSL function '" + name + "' requires function constant(s) " + missing +
                       ", but they were not set; pass values via Kernel(..., constants={...})";
        }
        if (!unknown.empty()) {
            if (!missing.empty()) message += "; additionally, ";
            message += "constant(s) " + unknown + " do not exist in MSL function '" + name + "'";
        }
        throw MSLCompileError(message);
    }

    MTL::FunctionConstantValues* values =
        MTL::FunctionConstantValues::alloc()->init()->autorelease();
    for (const FunctionConstant& c : constants) {
        MTL::DataType type = declared.at(c.name).type;
        ConstantBytes bytes = coerce(c, type);
        values->setConstantValue(bytes.data(), type,
                                 to_ns_string(c.name, "function constant name"));
    }
    NS::Error* error = nullptr;
    NS::SharedPtr<MTL::Function> fn =
        NS::TransferPtr(library_->newFunction(fn_name, values, &error));
    if (!fn) {
        throw MSLCompileError("failed to specialize MSL function '" + name +
                              "' with the given constants: " + describe(error));
    }
    return fn;
}

std::shared_ptr<ComputePipeline> Library::pipeline_for(const std::string& name,
                                                       const FunctionConstants& constants) {
    const std::string key = pipeline_key(name, constants);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto hit = pipelines_.find(key)) return hit;
    }
    // Built outside the lock; a racing build of the same key is dropped.
    NS::SharedPtr<MTL::Function> fn = create_specialized(name, constants);
    auto pipeline = std::make_shared<ComputePipeline>(device_, fn.get(), name);
    std::lock_guard<std::mutex> lock(mutex_);
    return pipelines_.insert(key, std::move(pipeline));
}
