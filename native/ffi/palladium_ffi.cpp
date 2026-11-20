// One generic XLA FFI handler for every palladium kernel, registered as
// "palladium_dispatch" on the "cpu" platform. MSL source, entry point, and
// grid travel as FFI Attrs baked into the HLO; buffers arrive as
// RemainingArgs/RemainingRets, so one handler covers any input/output
// count. Uses only the runtime's C API (native/runtime/c_api.h). XLA:CPU
// runs FFI handlers on its own thread pool, so KernelCache is mutex-guarded.

#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "c_api.h"
#include "xla/ffi/api/ffi.h"

namespace {

// LRU of compiled pipelines keyed by (source digest, entry point, math
// mode); mr_compile_library does not cache.
// PALLADIUM_KERNEL_CACHE_SIZE overrides the cap, and 0 disables eviction.
constexpr size_t kDefaultKernelCacheSize = 256;

size_t kernel_cache_capacity() {
  const char *env = std::getenv("PALLADIUM_KERNEL_CACHE_SIZE");
  if (!env || !*env)
    return kDefaultKernelCacheSize;
  char *end = nullptr;
  unsigned long v = std::strtoul(env, &end, 10);
  if (end == env || *end)
    return kDefaultKernelCacheSize;
  return (size_t)v;
}

// Owning handles: every return path releases what it wrapped or created.
struct Release {
  void operator()(MRBuffer *b) const { mr_release_buffer(b); }
  void operator()(MRBatch *b) const { mr_release_batch(b); }
};

// Takes ownership of a C API error message, or `fallback` if there is none.
std::string take_error(char *err, const char *fallback) {
  std::string message = err ? err : fallback;
  mr_free_error_message(err);
  return message;
}

class KernelCache {
public:
  static KernelCache &instance() {
    static KernelCache cache;
    return cache;
  }

  // Returns null and fills `error` on failure. `source_id` is palladium's
  // content digest of `msl_source`, computed once at trace time, so the
  // per-call key never copies or hashes the source itself. The caller shares
  // ownership, so eviction on another thread cannot free a pipeline in use.
  std::shared_ptr<MRPipeline> get_or_compile(std::string_view source_id,
                                             std::string_view msl_source,
                                             std::string_view function_name,
                                             MRMathMode math_mode,
                                             std::string *error) {
    std::string key;
    key.reserve(source_id.size() + function_name.size() + 3);
    key.append(source_id);
    key.push_back('\0');
    key.append(function_name);
    key.push_back('\0');
    key.push_back((char)math_mode);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = entries_.find(key);
      if (it != entries_.end()) {
        touch(it);
        return it->second->second;
      }
    }

    // Compiled outside the lock so threads compiling different kernels do
    // not serialize. A race double-compiles; the loser is released below.
    char *err = nullptr;
    MRLibrary *library = nullptr;
    if (mr_compile_library(msl_source.data(), msl_source.size(), math_mode,
                           &library, &err) != MR_OK) {
      *error = take_error(err, "mr_compile_library failed");
      return nullptr;
    }

    std::string fn_name(function_name);
    MRPipeline *raw = nullptr;
    if (mr_get_pipeline(library, fn_name.c_str(), &raw, &err) != MR_OK) {
      *error = take_error(err, "mr_get_pipeline failed");
      mr_release_library(library);
      return nullptr;
    }
    // The library lives as long as its pipeline.
    std::shared_ptr<MRPipeline> pipeline(raw, [library](MRPipeline *p) {
      mr_release_pipeline(p);
      mr_release_library(library);
    });

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      // Lost the compile race; ours is released on return.
      touch(it);
      return it->second->second;
    }
    order_.push_front({key, pipeline});
    entries_.emplace(std::move(key), order_.begin());
    if (capacity_ != 0) {
      while (entries_.size() > capacity_) {
        entries_.erase(order_.back().first);
        order_.pop_back();
      }
    }
    return pipeline;
  }

private:
  using Node = std::pair<std::string, std::shared_ptr<MRPipeline>>;
  using Iter = std::list<Node>::iterator;

  // Most-recently-used at the front. Called with mutex_ held.
  void touch(std::unordered_map<std::string, Iter>::iterator it) {
    order_.splice(order_.begin(), order_, it->second);
  }

  std::mutex mutex_;
  const size_t capacity_ = kernel_cache_capacity();
  std::list<Node> order_;
  std::unordered_map<std::string, Iter> entries_;
};

xla::ffi::Error PalladiumDispatch(std::string_view source_id,
                                  std::string_view msl_source,
                                  std::string_view function_name,
                                  int64_t grid_x, int64_t grid_y,
                                  int64_t grid_z, int64_t threadgroup_x,
                                  int64_t threadgroup_y, int64_t threadgroup_z,
                                  int64_t math_mode, int64_t batch_size,
                                  xla::ffi::Span<const int64_t> elem_strides,
                                  xla::ffi::RemainingArgs args,
                                  xla::ffi::RemainingRets rets) {
  if (math_mode < MR_MATH_MODE_SAFE || math_mode > MR_MATH_MODE_FAST) {
    return xla::ffi::Error::InvalidArgument(
        "palladium_dispatch: math_mode out of range");
  }
  if (batch_size < 1) {
    return xla::ffi::Error::InvalidArgument(
        "palladium_dispatch: batch_size must be >= 1");
  }
  std::string error;
  std::shared_ptr<MRPipeline> pipeline = KernelCache::instance().get_or_compile(
      source_id, msl_source, function_name, (MRMathMode)math_mode, &error);
  if (!pipeline)
    return xla::ffi::Error::Internal(error);

  char *err = nullptr;
  std::vector<std::unique_ptr<MRBuffer, Release>> owned;
  std::vector<MRBuffer *> wrapped;
  owned.reserve(args.size() + rets.size());
  wrapped.reserve(args.size() + rets.size());
  auto wrap = [&](xla::ffi::AnyBuffer buf) {
    MRBuffer *b = nullptr;
    if (mr_wrap_buffer(buf.untyped_data(), buf.size_bytes(), &b, &err) != MR_OK)
      return false;
    owned.emplace_back(b);
    wrapped.push_back(b);
    return true;
  };
  for (size_t i = 0; i < args.size(); ++i) {
    auto buf = args.get<xla::ffi::AnyBuffer>(i);
    if (!buf.has_value())
      return buf.error();
    if (!wrap(*buf))
      return xla::ffi::Error::Internal(take_error(err, "mr_wrap_buffer failed"));
  }
  size_t n_inputs = wrapped.size();
  for (size_t i = 0; i < rets.size(); ++i) {
    auto buf = rets.get<xla::ffi::AnyBuffer>(i);
    if (!buf.has_value())
      return buf.error();
    if (!wrap(**buf))
      return xla::ffi::Error::Internal(take_error(err, "mr_wrap_buffer failed"));
  }

  if (elem_strides.size() != wrapped.size()) {
    return xla::ffi::Error::InvalidArgument(
        "palladium_dispatch: elem_strides must have one entry per buffer");
  }

  std::vector<size_t> offsets(wrapped.size());
  MRLaunchDesc desc{};
  desc.pipeline = pipeline.get();
  desc.buffers = wrapped.data();
  desc.buffer_offsets = offsets.data();
  desc.buffer_count = wrapped.size();
  desc.grid_x = (size_t)grid_x;
  desc.grid_y = (size_t)grid_y;
  desc.grid_z = (size_t)grid_z;
  // threadgroup_x == 0 lets the runtime choose.
  desc.threadgroup_x = (size_t)threadgroup_x;
  desc.threadgroup_y = (size_t)threadgroup_y;
  desc.threadgroup_z = (size_t)threadgroup_z;

  // One grid dispatch per batch element, offset element * elem_strides[i]
  // bytes into each buffer (stride 0 = shared), all in one command buffer
  // so the fixed cost of a command buffer is paid once per FFI call.
  // mr_batch_add copies the descriptor, so offsets is rewritten in place.
  // A batch that fails before its commit is released without running: the
  // buffers wrap the caller's memory, which nothing may write after return.
  MRBatch *raw_batch = nullptr;
  if (mr_batch_create(&raw_batch, &err) != MR_OK)
    return xla::ffi::Error::Internal(take_error(err, "mr_batch_create failed"));
  std::unique_ptr<MRBatch, Release> batch(raw_batch);
  for (int64_t element = 0; element < batch_size; ++element) {
    for (size_t i = 0; i < wrapped.size(); ++i)
      offsets[i] = (size_t)(element * elem_strides[i]);
    if (mr_batch_add(batch.get(), &desc, &err) != MR_OK)
      return xla::ffi::Error::Internal(take_error(err, "mr_batch_add failed"));
  }
  // Waiting commits first, and returns only once the GPU is done.
  if (mr_batch_wait(batch.get(), &err) != MR_OK)
    return xla::ffi::Error::Internal(take_error(err, "mr_batch_wait failed"));

  // Flush outputs only (no-op on the zero-copy path), once per buffer
  // after the whole batch.
  for (size_t i = n_inputs; i < wrapped.size(); ++i) {
    if (mr_buffer_flush_to(wrapped[i], &err) != MR_OK) {
      return xla::ffi::Error::Internal(
          take_error(err, "mr_buffer_flush_to failed"));
    }
  }
  return xla::ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(palladium_dispatch, PalladiumDispatch,
                              xla::ffi::Ffi::Bind()
                                  .Attr<std::string_view>("source_id")
                                  .Attr<std::string_view>("msl_source")
                                  .Attr<std::string_view>("function_name")
                                  .Attr<int64_t>("grid_x")
                                  .Attr<int64_t>("grid_y")
                                  .Attr<int64_t>("grid_z")
                                  .Attr<int64_t>("threadgroup_x")
                                  .Attr<int64_t>("threadgroup_y")
                                  .Attr<int64_t>("threadgroup_z")
                                  .Attr<int64_t>("math_mode")
                                  .Attr<int64_t>("batch_size")
                                  .Attr<xla::ffi::Span<const int64_t>>(
                                      "elem_strides")
                                  .RemainingArgs()
                                  .RemainingRets());
