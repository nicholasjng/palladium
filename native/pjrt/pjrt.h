// Types and helpers shared by the `metal` PJRT plugin's translation units.
#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "buffer.h"
#include "xla/pjrt/c/pjrt_c_api.h"

// The C API defines PJRT_Error and PJRT_Memory as function tables that
// callers may use directly; these extend them.
struct Error : PJRT_Error {
    PJRT_Error_Code code;
    std::string message;
};

struct PJRT_Event {};

struct PJRT_DeviceDescription {
    std::string kind, debug_string, to_string;
    int64_t apple_gpu_family;
    PJRT_NamedValue attributes[1];
};

struct Memory : PJRT_Memory {
    PJRT_Device* device;
    std::mutex mutex;
    std::map<const void*, std::pair<void*, void (*)(void*)>> user_data;
    ~Memory() {
        for (auto& [key, entry] : user_data) {
            if (entry.second) entry.second(entry.first);
        }
    }
};

struct PJRT_Device {
    PJRT_DeviceDescription description;
    Memory memory;
    PJRT_Memory* memory_ptr = &memory;
};

struct PJRT_TopologyDescription {
    PJRT_DeviceDescription* description;
};

struct PJRT_Client {
    PJRT_Device device;
    PJRT_Device* device_ptr = &device;
    PJRT_TopologyDescription topology{&device.description};
};

struct PJRT_Buffer {
    PJRT_Device* device;
    PJRT_Buffer_Type type;
    std::vector<int64_t> dims, minor_to_major;
    std::unique_ptr<Buffer> storage;  // null once deleted
};

PJRT_Error* error(PJRT_Error_Code code, std::string message);

// Bytes per element, or 0 for types that are not byte-addressable.
size_t element_size(PJRT_Buffer_Type type);
size_t byte_size(const PJRT_Buffer* buffer);

// Allocates a dense row-major buffer; the caller fills it.
PJRT_Error* make_buffer(PJRT_Device* device, PJRT_Buffer_Type type, const int64_t* dims,
                        size_t num_dims, PJRT_Buffer** out);

// An error if the buffer has been deleted.
PJRT_Error* deleted(const PJRT_Buffer* buffer);

// Compilation and execution: PJRT_Client_Compile and the executable functions.
void add_executable_api(PJRT_Api& api);
