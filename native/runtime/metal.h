#pragma once
#include <stdexcept>
#include <string>

// IWYU pragma: begin_exports
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
// IWYU pragma: end_exports

// Metal may return no NSError at all.
inline std::string describe(NS::Error* error) {
    return error ? error->localizedDescription()->utf8String() : "unknown error";
}

// Autoreleased. NS::String::string stops at a NUL and returns nil for invalid
// UTF-8, which Metal asserts on instead of reporting; `what` names the argument.
inline NS::String* to_ns_string(const std::string& s, const char* what) {
    if (s.find('\0') != std::string::npos) {
        throw std::invalid_argument(std::string(what) + " contains a NUL byte");
    }
    NS::String* out = NS::String::string(s.c_str(), NS::UTF8StringEncoding);
    if (!out) throw std::invalid_argument(std::string(what) + " is not valid UTF-8");
    return out;
}

// Without a pool, autoreleased Cocoa/Metal objects leak for the process lifetime.
struct AutoreleaseScope {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~AutoreleaseScope() { pool->release(); }
};
