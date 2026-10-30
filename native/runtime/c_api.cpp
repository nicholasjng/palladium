#include "c_api.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "buffer.h"
#include "dispatch.h"
#include "errors.h"
#include "library.h"
#include "metal.h"
#include "runtime.h"

struct MRLibrary {
    Library library;
};

struct MRPipeline {
    std::shared_ptr<ComputePipeline> pipeline;
};

struct MRBuffer {
    std::shared_ptr<Buffer> buffer;  // shared with the wrap cache
    void* external_ptr = nullptr;    // set even on the copy path, for flush_to
    size_t external_offset = 0;      // buffer.contents() + external_offset == external_ptr
    bool owns_copy = false;          // true: buffer is a private copy, flush_to must run
    size_t logical_size = 0;         // excludes the enclosing page padding
};

namespace {

void set_error(char** out_err_msg, const std::string& message) {
    if (!out_err_msg) return;
    *out_err_msg = (char*)std::malloc(message.size() + 1);
    if (!*out_err_msg) return;
    std::memcpy(*out_err_msg, message.c_str(), message.size() + 1);
}

// Maps exceptions to MRStatus.
template <typename Fn>
MRStatus mr_guard(char** out_err_msg, Fn&& fn) {
    AutoreleaseScope scope;
    try {
        fn();
        return MR_OK;
    } catch (const MSLFunctionNotFoundError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_FUNCTION_NOT_FOUND;
    } catch (const MSLCompileError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_COMPILE;
    } catch (const DispatchError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_DISPATCH;
    } catch (const NoDeviceError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_NO_DEVICE;
    } catch (const AllocationError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_ALLOCATION;
    } catch (const std::invalid_argument& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_UNKNOWN;
    }
}

// The VM objects behind [start, start + length): (entry start, object, offset)
// for each map entry it spans, or nullopt if part of it is unmapped, not
// read-write, or not backed by an object. One allocation can span several
// entries; XNU splits anonymous mappings into entries of at most 128 MB.
std::optional<std::vector<uint64_t>> region_signature(uintptr_t start, size_t length) {
    std::vector<uint64_t> signature;
    mach_vm_address_t addr = start;
    const mach_vm_address_t end = start + length;
    while (addr < end) {
        mach_vm_address_t entry = addr;
        mach_vm_size_t size = 0;
        natural_t depth = 0;
        vm_region_submap_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
        if (mach_vm_region_recurse(mach_task_self(), &entry, &size, &depth,
                                   (vm_region_recurse_info_t)&info, &count) != KERN_SUCCESS) {
            return std::nullopt;
        }
        // mach_vm_region_recurse moves `entry` forward when `addr` falls in a hole.
        if (entry > addr || info.is_submap || info.object_id_full == 0) return std::nullopt;
        if (!(info.protection & VM_PROT_READ) || !(info.protection & VM_PROT_WRITE)) {
            return std::nullopt;
        }
        signature.insert(signature.end(), {entry, info.object_id_full, info.offset});
        addr = entry + size;
    }
    return signature;
}

// No-copy wraps kept across mr_wrap_buffer calls. A new wrap costs the GPU
// page-table setup on its first dispatch, about 15 us per MB, so a caller that
// wraps the same memory on every call (an XLA FFI handler) pays it every
// call. A wrap pins the VM objects it maps, and the GPU keeps reading those
// pages even after the range is freed and mapped again; but while the wrap
// lives, no other object can take their ids. So a cached wrap is reused
// exactly while its range still has the signature it was made over.
class WrapCache {
   public:
    static constexpr size_t kMinBytes = 1 << 20;  // smaller wraps are cheap

    static WrapCache& instance() {
        static WrapCache* cache = new WrapCache();  // never destroyed: no exit-order issues
        return *cache;
    }

    std::shared_ptr<Buffer> find(uintptr_t start, size_t length,
                                 const std::vector<uint64_t>& signature) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->start != start || it->length != length) continue;
            if (it->signature != signature) {
                erase(it);
                return nullptr;
            }
            entries_.splice(entries_.begin(), entries_, it);
            return it->buffer;
        }
        return nullptr;
    }

    // Also drops every entry whose range has changed since, so freed memory
    // is not pinned past the next insertion, then the least recently used
    // entries beyond the byte budget.
    void insert(uintptr_t start, size_t length, std::vector<uint64_t> signature,
                std::shared_ptr<Buffer> buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end();) {
            auto next = std::next(it);
            if ((it->start == start && it->length == length) ||
                region_signature(it->start, it->length) != it->signature) {
                erase(it);
            }
            it = next;
        }
        entries_.push_front({start, length, std::move(signature), std::move(buffer)});
        bytes_ += length;
        while (bytes_ > capacity_ && !entries_.empty()) erase(std::prev(entries_.end()));
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        bytes_ = 0;
    }

    size_t capacity() const { return capacity_; }

   private:
    struct Entry {
        uintptr_t start;
        size_t length;
        std::vector<uint64_t> signature;
        std::shared_ptr<Buffer> buffer;
    };

    // MR_WRAP_CACHE_BYTES overrides the budget, and 0 disables the cache.
    WrapCache() {
        const char* env = std::getenv("MR_WRAP_CACHE_BYTES");
        char* end = nullptr;
        unsigned long long bytes = env && *env ? std::strtoull(env, &end, 10) : 0;
        capacity_ =
            env && *env && !*end ? (size_t)bytes : runtime().recommended_max_working_set_size() / 4;
    }

    void erase(std::list<Entry>::iterator it) {
        bytes_ -= it->length;
        entries_.erase(it);
    }

    std::mutex mutex_;
    std::list<Entry> entries_;  // most recently used first
    size_t bytes_ = 0;
    size_t capacity_ = 0;
};

}  // namespace

void mr_free_error_message(char* msg) { std::free(msg); }

MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len, MRMathMode math_mode,
                            MRLibrary** out_library, char** out_err_msg) {
    if (out_library) *out_library = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_library) throw std::invalid_argument("mr_compile_library: out_library is null");
        if (!msl_source) throw std::invalid_argument("mr_compile_library: msl_source is null");
        MathMode mode;
        switch (math_mode) {
            case MR_MATH_MODE_SAFE:
                mode = MathMode::Safe;
                break;
            case MR_MATH_MODE_RELAXED:
                mode = MathMode::Relaxed;
                break;
            case MR_MATH_MODE_FAST:
                mode = MathMode::Fast;
                break;
            default:
                throw std::invalid_argument("mr_compile_library: unknown math_mode");
        }
        auto* lib = new MRLibrary{
            Library(runtime().device(), std::string(msl_source, msl_source_len), mode)};
        *out_library = lib;
    });
}

void mr_release_library(MRLibrary* library) { delete library; }

MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name, MRPipeline** out_pipeline,
                         char** out_err_msg) {
    if (out_pipeline) *out_pipeline = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_pipeline) throw std::invalid_argument("mr_get_pipeline: out_pipeline is null");
        if (!library) throw std::invalid_argument("mr_get_pipeline: library is null");
        if (!function_name) throw std::invalid_argument("mr_get_pipeline: function_name is null");
        auto* pipeline = new MRPipeline{library->library.pipeline(function_name)};
        *out_pipeline = pipeline;
    });
}

void mr_release_pipeline(MRPipeline* pipeline) { delete pipeline; }

MRStatus mr_device_info(MRDeviceInfo* out_info, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!out_info) throw std::invalid_argument("mr_device_info: out_info is null");
        MetalRuntime& rt = runtime();
        *out_info = MRDeviceInfo{
            rt.max_threads_per_threadgroup(), rt.max_threadgroup_memory_length(),
            rt.max_buffer_length(),           rt.recommended_max_working_set_size(),
            rt.has_unified_memory(),          rt.supports_non_uniform_threadgroups(),
            rt.apple_gpu_family(),            {},
        };
        std::strncpy(out_info->name, rt.device_name().c_str(), sizeof(out_info->name) - 1);
    });
}

MRStatus mr_pipeline_info(const MRPipeline* pipeline, MRPipelineInfo* out_info,
                          char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!pipeline) throw std::invalid_argument("mr_pipeline_info: pipeline is null");
        if (!out_info) throw std::invalid_argument("mr_pipeline_info: out_info is null");
        const ComputePipeline& p = *pipeline->pipeline;
        *out_info = MRPipelineInfo{
            p.thread_execution_width(),
            p.max_threads_per_threadgroup(),
            p.static_threadgroup_memory_length(),
        };
    });
}

MRStatus mr_wrap_buffer(void* ptr, size_t size_bytes, MRBuffer** out_buffer, char** out_err_msg) {
    if (out_buffer) *out_buffer = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_buffer) throw std::invalid_argument("mr_wrap_buffer: out_buffer is null");
        if (!ptr && size_bytes > 0) {
            throw std::invalid_argument("mr_wrap_buffer: ptr is null for a nonzero size");
        }

        if (size_bytes > 0) {
            size_t page_size = (size_t)getpagesize();
            uintptr_t addr = (uintptr_t)ptr;
            uintptr_t page_start = addr & ~(uintptr_t)(page_size - 1);
            size_t offset = addr - page_start;
            constexpr size_t max_size = std::numeric_limits<size_t>::max();
            if (size_bytes > std::numeric_limits<uintptr_t>::max() - addr ||
                size_bytes > max_size - offset ||
                offset + size_bytes > max_size - (page_size - 1)) {
                throw std::invalid_argument("mr_wrap_buffer: memory range overflows");
            }
            size_t wrapped_length = ((offset + size_bytes + page_size - 1) / page_size) * page_size;
            if (wrapped_length > std::numeric_limits<uintptr_t>::max() - page_start) {
                throw std::invalid_argument("mr_wrap_buffer: page-rounded memory range overflows");
            }

            if (auto signature = region_signature(page_start, wrapped_length)) {
                WrapCache& cache = WrapCache::instance();
                bool cacheable =
                    wrapped_length >= WrapCache::kMinBytes && wrapped_length <= cache.capacity();
                std::shared_ptr<Buffer> wrapped;
                if (cacheable) wrapped = cache.find(page_start, wrapped_length, *signature);
                if (!wrapped) {
                    wrapped = std::make_shared<Buffer>(runtime().device(), (void*)page_start,
                                                       wrapped_length);
                    // Wrapping can fault in an object for untouched memory, so the
                    // entry records the signature from after the wrap.
                    if (cacheable) {
                        if (auto after = region_signature(page_start, wrapped_length)) {
                            cache.insert(page_start, wrapped_length, std::move(*after), wrapped);
                        }
                    }
                }
                *out_buffer = new MRBuffer{std::move(wrapped), ptr, offset, false, size_bytes};
                return;
            }
        }

        auto owned = std::make_shared<Buffer>(runtime().device(), size_bytes);
        if (size_bytes > 0) std::memcpy(owned->contents(), ptr, size_bytes);
        auto* buf = new MRBuffer{std::move(owned), ptr, 0, true, size_bytes};
        *out_buffer = buf;
    });
}

MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!buffer) throw std::invalid_argument("mr_buffer_flush_to: buffer is null");
        if (!buffer->owns_copy) return;
        if (buffer->buffer->size() > 0) {
            std::memcpy(buffer->external_ptr, buffer->buffer->contents(), buffer->buffer->size());
        }
    });
}

void mr_release_buffer(MRBuffer* buffer) { delete buffer; }

void mr_clear_wrap_cache(void) { WrapCache::instance().clear(); }

namespace {

// `who` names the entry point in error messages.
Launch build_launch(const MRLaunchDesc* launch_desc, const char* who) {
    if (!launch_desc) throw std::invalid_argument(std::string(who) + ": launch is null");
    if (!launch_desc->pipeline)
        throw std::invalid_argument(std::string(who) + ": launch has no pipeline");
    if (launch_desc->buffer_count && (!launch_desc->buffers || !launch_desc->buffer_offsets)) {
        throw std::invalid_argument(std::string(who) +
                                    ": buffers and buffer_offsets are required when buffer_count "
                                    "is nonzero");
    }
    if (launch_desc->scalar_count && (!launch_desc->scalars || !launch_desc->scalar_sizes)) {
        throw std::invalid_argument(std::string(who) +
                                    ": scalars and scalar_sizes are required when scalar_count is "
                                    "nonzero");
    }
    if (launch_desc->threadgroup_memory_count && !launch_desc->threadgroup_memory) {
        throw std::invalid_argument(std::string(who) +
                                    ": threadgroup_memory is required when its count is nonzero");
    }

    Launch launch;
    launch.pipeline = launch_desc->pipeline->pipeline.get();
    for (size_t i = 0; i < launch_desc->buffer_count; ++i) {
        if (!launch_desc->buffers[i]) {
            throw std::invalid_argument(std::string(who) + ": buffers[" + std::to_string(i) +
                                        "] is null");
        }
        const MRBuffer& buffer = *launch_desc->buffers[i];
        size_t offset = launch_desc->buffer_offsets[i];
        if ((offset >= buffer.logical_size && !(offset == 0 && buffer.logical_size == 0)) ||
            offset > std::numeric_limits<size_t>::max() - buffer.external_offset) {
            throw std::invalid_argument(std::string(who) + ": buffers[" + std::to_string(i) +
                                        "] offset " + std::to_string(offset) +
                                        " is out of bounds for a logical buffer of " +
                                        std::to_string(buffer.logical_size) + " bytes");
        }
        launch.buffers.push_back(
            {launch_desc->buffers[i]->buffer.get(), offset + buffer.external_offset});
    }
    for (size_t i = 0; i < launch_desc->scalar_count; ++i) {
        if (!launch_desc->scalars[i]) {
            throw std::invalid_argument(std::string(who) + ": scalars[" + std::to_string(i) +
                                        "] is null");
        }
        launch.scalars.push_back({launch_desc->scalars[i], launch_desc->scalar_sizes[i]});
    }
    if (launch_desc->threadgroup_memory_count) {
        launch.threadgroup_memory.assign(
            launch_desc->threadgroup_memory,
            launch_desc->threadgroup_memory + launch_desc->threadgroup_memory_count);
    }
    launch.grid = {launch_desc->grid_x, launch_desc->grid_y, launch_desc->grid_z};

    if (launch_desc->threadgroup_x == 0) {
        launch.threadgroup = launch.pipeline->default_threadgroup(launch.grid);
    } else {
        launch.threadgroup = {launch_desc->threadgroup_x, launch_desc->threadgroup_y,
                              launch_desc->threadgroup_z};
    }
    return launch;
}

}  // namespace

MRStatus mr_dispatch(const MRLaunchDesc* launch_desc, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        Launch launch = build_launch(launch_desc, "mr_dispatch");
        dispatch(runtime(), launch);
    });
}

struct MRBatch {
    explicit MRBatch(MetalRuntime& rt) : batch(rt) {}
    CommandBatch batch;
};

MRStatus mr_batch_wait(MRBatch* batch, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_wait: batch is null");
        batch->batch.wait();
    });
}

void mr_release_batch(MRBatch* batch) { delete batch; }

MRStatus mr_batch_create(MRBatch** out_batch, char** out_err_msg) {
    if (out_batch) *out_batch = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_batch) throw std::invalid_argument("mr_batch_create: out_batch is null");
        *out_batch = new MRBatch(runtime());
    });
}

MRStatus mr_batch_add(MRBatch* batch, const MRLaunchDesc* launch_desc, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_add: batch is null");
        Launch launch = build_launch(launch_desc, "mr_batch_add");
        batch->batch.add(launch);
    });
}

MRStatus mr_batch_commit(MRBatch* batch, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_commit: batch is null");
        batch->batch.commit();
    });
}

MRStatus mr_batch_gpu_time(const MRBatch* batch, double* out_seconds, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_gpu_time: batch is null");
        if (!out_seconds) throw std::invalid_argument("mr_batch_gpu_time: out_seconds is null");
        std::optional<double> seconds = batch->batch.gpu_time();
        if (!seconds) throw std::invalid_argument("mr_batch_gpu_time: the batch was not waited on");
        *out_seconds = *seconds;
    });
}
