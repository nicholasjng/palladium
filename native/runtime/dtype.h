#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "export.h"

// Element type of a Buffer, as DLPack's (code, bits).
struct DType {
    enum Code : uint8_t { Int = 0, UInt = 1, Float = 2, Bfloat = 4, Bool = 6 };

    uint8_t code = Float;
    uint8_t bits = 32;

    size_t itemsize() const { return (size_t)(bits + 7) / 8; }
    bool operator==(const DType& other) const { return code == other.code && bits == other.bits; }
    bool operator!=(const DType& other) const { return !(*this == other); }
};

// Throws std::invalid_argument for an unsupported name.
MR_API DType dtype_from_name(const std::string& name);

// nullptr if `dt` isn't a dtype this runtime supports.
MR_API const char* dtype_name(DType dt);

// The MSL spelling, e.g. "uint" for uint32; nullptr if unsupported.
const char* msl_type_name(DType dt);

// Comma-separated, for error messages.
MR_API std::string supported_dtype_names();
