#include "dtype.h"

#include <stdexcept>
#include <string>

namespace {

struct Entry {
    const char* name;
    const char* msl_name;
    DType dtype;
};

constexpr Entry kEntries[] = {
    {"bool", "bool", {DType::Bool, 8}},       {"int8", "char", {DType::Int, 8}},
    {"int16", "short", {DType::Int, 16}},     {"int32", "int", {DType::Int, 32}},
    {"int64", "long", {DType::Int, 64}},      {"uint8", "uchar", {DType::UInt, 8}},
    {"uint16", "ushort", {DType::UInt, 16}},  {"uint32", "uint", {DType::UInt, 32}},
    {"uint64", "ulong", {DType::UInt, 64}},   {"float16", "half", {DType::Float, 16}},
    {"float32", "float", {DType::Float, 32}}, {"bfloat16", "bfloat", {DType::Bfloat, 16}},
};

}  // namespace

DType dtype_from_name(const std::string& name) {
    for (const Entry& entry : kEntries) {
        if (name == entry.name) return entry.dtype;
    }
    if (name == "float64" || name == "double") {
        throw std::invalid_argument(
            "dtype 'float64' is not supported: Metal has no double-precision type. "
            "Convert with .astype(numpy.float32) before uploading.");
    }
    throw std::invalid_argument("unsupported dtype '" + name +
                                "'; supported: " + supported_dtype_names());
}

const char* dtype_name(DType dt) {
    for (const Entry& entry : kEntries) {
        if (dt == entry.dtype) return entry.name;
    }
    return nullptr;
}

const char* msl_type_name(DType dt) {
    for (const Entry& entry : kEntries) {
        if (dt == entry.dtype) return entry.msl_name;
    }
    return nullptr;
}

std::string supported_dtype_names() {
    std::string out;
    for (const Entry& entry : kEntries) {
        if (!out.empty()) out += ", ";
        out += entry.name;
    }
    return out;
}
