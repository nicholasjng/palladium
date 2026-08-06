// jax.ffi entry point: one generic XLA FFI handler for every palladium
// kernel, registered as "palladium_dispatch" on the "cpu" platform. MSL
// source, entry point name, and grid travel as FFI Attrs (baked into
// the HLO at trace time, free per call); buffers arrive as
// RemainingArgs/RemainingRets, so one handler covers any input/output
// count. Bridges through metal-runtime's C API (c_api.h) only, no
// Metal/Obj-C type here. XLA:CPU runs FFI handlers on its own compute
// thread pool with no opt-out trait, so KernelCache below is
// mutex-guarded the same way metal-runtime's own caches are.

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "c_api.h"
#include "xla/ffi/api/ffi.h"

namespace {

// Compiles lazily, once per (msl_source, function_name) pair, holds the
// MRLibrary/MRPipeline for the process lifetime. mr_compile_library
// itself doesn't cache (unlike the Python bindings' library_for()), so
// this exists because nothing upstream provides it.
class KernelCache {
public:
  static KernelCache &instance() {
    static KernelCache cache;
    return cache;
  }

  // Returns {nullptr, nullptr} and fills `error` on failure.
  std::pair<MRLibrary *, MRPipeline *>
  get_or_compile(std::string_view msl_source, std::string_view function_name,
                 MRMathMode math_mode, std::string *error) {
    std::string key;
    key.reserve(msl_source.size() + function_name.size() + 2);
    key.append(msl_source);
    key.push_back('\0');
    key.append(function_name);
    key.push_back('\0');
    key.push_back((char)math_mode);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = entries_.find(key);
      if (it != entries_.end())
        return it->second;
    }

    // Compiled outside the lock: holding it across a compile/build would
    // serialize threads compiling *different* kernels. A race
    // double-compiles; the loser is released below. Same trade-off as
    // runtime.h's library_for().
    char *err = nullptr;
    MRLibrary *library = nullptr;
    if (mr_compile_library(msl_source.data(), msl_source.size(), math_mode,
                           &library, &err) != MR_OK) {
      *error = err ? err : "mr_compile_library failed";
      if (err)
        mr_free_error_message(err);
      return {nullptr, nullptr};
    }

    std::string fn_name(function_name);
    MRPipeline *pipeline = nullptr;
    if (mr_get_pipeline(library, fn_name.c_str(), &pipeline, &err) != MR_OK) {
      *error = err ? err : "mr_get_pipeline failed";
      if (err)
        mr_free_error_message(err);
      mr_release_library(library);
      return {nullptr, nullptr};
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      mr_release_pipeline(pipeline);
      mr_release_library(library);
      return it->second;
    }
    auto entry = std::make_pair(library, pipeline);
    entries_.emplace(std::move(key), entry);
    return entry;
  }

private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::pair<MRLibrary *, MRPipeline *>>
      entries_;
};

// Wraps one FFI buffer via mr_wrap_buffer, appending to `wrapped`/`offsets`.
// Returns false and fills `error` on failure; caller must not dispatch.
bool wrap_one(xla::ffi::AnyBuffer buf, std::vector<MRBuffer *> &wrapped,
              std::vector<size_t> &offsets, std::string *error) {
  MRBuffer *b = nullptr;
  char *err = nullptr;
  if (mr_wrap_buffer(buf.untyped_data(), buf.size_bytes(), &b, &err) != MR_OK) {
    *error = err ? err : "mr_wrap_buffer failed";
    if (err)
      mr_free_error_message(err);
    return false;
  }
  wrapped.push_back(b);
  offsets.push_back(0);
  return true;
}

xla::ffi::Error PalladiumDispatch(std::string_view msl_source,
                                  std::string_view function_name,
                                  int64_t grid_x, int64_t grid_y,
                                  int64_t grid_z, int64_t threadgroup_x,
                                  int64_t threadgroup_y, int64_t threadgroup_z,
                                  int64_t math_mode,
                                  xla::ffi::RemainingArgs args,
                                  xla::ffi::RemainingRets rets) {
  if (math_mode < MR_MATH_MODE_SAFE || math_mode > MR_MATH_MODE_FAST) {
    return xla::ffi::Error::InvalidArgument(
        "palladium_dispatch: math_mode out of range");
  }
  std::string error;
  auto [library, pipeline] = KernelCache::instance().get_or_compile(
      msl_source, function_name, (MRMathMode)math_mode, &error);
  if (!pipeline)
    return xla::ffi::Error::Internal(error);
  (void)library; // kept alive by the cache; not needed past pipeline lookup

  std::vector<MRBuffer *> wrapped;
  std::vector<size_t> offsets;
  wrapped.reserve(args.size() + rets.size());
  offsets.reserve(args.size() + rets.size());

  for (size_t i = 0; i < args.size(); ++i) {
    auto buf = args.get<xla::ffi::AnyBuffer>(i);
    if (!buf.has_value())
      return buf.error();
    if (!wrap_one(*buf, wrapped, offsets, &error))
      return xla::ffi::Error::Internal(error);
  }
  size_t n_inputs = wrapped.size();
  for (size_t i = 0; i < rets.size(); ++i) {
    auto buf = rets.get<xla::ffi::AnyBuffer>(i);
    if (!buf.has_value())
      return buf.error();
    if (!wrap_one(**buf, wrapped, offsets, &error))
      return xla::ffi::Error::Internal(error);
  }

  MRLaunchDesc desc{};
  desc.pipeline = pipeline;
  desc.buffers = wrapped.data();
  desc.buffer_offsets = offsets.data();
  desc.buffer_count = wrapped.size();
  desc.grid_x = (size_t)grid_x;
  desc.grid_y = (size_t)grid_y;
  desc.grid_z = (size_t)grid_z;
  // threadgroup_x == 0 lets the runtime choose (c_api.cpp); the
  // cooperative execution model requires an explicit (32, 1, 1).
  desc.threadgroup_x = (size_t)threadgroup_x;
  desc.threadgroup_y = (size_t)threadgroup_y;
  desc.threadgroup_z = (size_t)threadgroup_z;

  char *err = nullptr;
  MRStatus status = mr_dispatch(&desc, &err);
  if (status != MR_OK) {
    xla::ffi::Error e =
        xla::ffi::Error::Internal(err ? err : "mr_dispatch failed");
    if (err)
      mr_free_error_message(err);
    for (auto *b : wrapped)
      mr_release_buffer(b);
    return e;
  }

  // Only outputs need flushing back (no-op on the zero-copy path).
  // Inputs are read-only to the kernel.
  for (size_t i = n_inputs; i < wrapped.size(); ++i) {
    if (mr_buffer_flush_to(wrapped[i], &err) != MR_OK) {
      xla::ffi::Error e =
          xla::ffi::Error::Internal(err ? err : "mr_buffer_flush_to failed");
      if (err)
        mr_free_error_message(err);
      for (auto *b : wrapped)
        mr_release_buffer(b);
      return e;
    }
  }

  for (auto *b : wrapped)
    mr_release_buffer(b);
  return xla::ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(palladium_dispatch, PalladiumDispatch,
                              xla::ffi::Ffi::Bind()
                                  .Attr<std::string_view>("msl_source")
                                  .Attr<std::string_view>("function_name")
                                  .Attr<int64_t>("grid_x")
                                  .Attr<int64_t>("grid_y")
                                  .Attr<int64_t>("grid_z")
                                  .Attr<int64_t>("threadgroup_x")
                                  .Attr<int64_t>("threadgroup_y")
                                  .Attr<int64_t>("threadgroup_z")
                                  .Attr<int64_t>("math_mode")
                                  .RemainingArgs()
                                  .RemainingRets());
