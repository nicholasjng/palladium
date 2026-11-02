// Types and helpers shared by the `metal` PJRT plugin's translation units.
#pragma once
#include <condition_variable>
#include <functional>
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

// Completion of host or GPU work, shared by the PJRT_Events that report it
// and the buffers it defines or reads.
class Completion {
   public:
    // An already completed one, for work done synchronously.
    static std::shared_ptr<Completion> done();

    // Complete after all dependencies, preserving the first error in input
    // order. Registration never waits; duplicate dependencies are coalesced.
    static std::shared_ptr<Completion> join(std::vector<std::shared_ptr<Completion>> dependencies);

    void set(PJRT_Error_Code code = PJRT_Error_Code_OK, std::string message = {});
    bool ready();
    void wait();
    // A new error if the work failed, else null; call once ready.
    PJRT_Error* error();
    // Runs `f` once complete: at once if it is, else on the completing thread.
    void then(std::function<void()> f);

   private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool ready_ = false;
    PJRT_Error_Code code_ = PJRT_Error_Code_OK;
    std::string message_;
    std::vector<std::function<void()>> callbacks_;
};

struct PJRT_Event {
    std::shared_ptr<Completion> completion;
};

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

// Buffers come from a pool keyed by size: GPU allocations cost tens of
// microseconds and are zeroed. Each dispatch retains the storage it uses;
// only the final owner returns it to the pool.
std::unique_ptr<Buffer> acquire(size_t size_bytes);
void release(std::unique_ptr<Buffer> buffer);

// Pool counters since the last reset: requests served from the pool, new
// MTLBuffers created (and their bytes and seconds), and releases the pool
// dropped because it was full.
struct PoolCounters {
    uint64_t hits, misses, miss_bytes, dropped;
    double miss_seconds;
};
PoolCounters pool_counters(bool reset);

struct PJRT_Buffer {
    PJRT_Device* device;
    PJRT_Buffer_Type type;
    std::vector<int64_t> dims, minor_to_major;
    std::shared_ptr<Buffer> storage;  // null once deleted; dispatches retain their own copy
    // Its contents are defined. Published once, before returning the buffer.
    std::shared_ptr<Completion> ready = Completion::done();
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

// Waits until the buffer's contents are defined; their failure as an error.
PJRT_Error* wait_ready(const PJRT_Buffer* buffer);

// Compilation and execution: PJRT_Client_Compile and the executable functions.
void add_executable_api(PJRT_Api& api);
