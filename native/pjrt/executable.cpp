// Compilation and execution for the `metal` PJRT plugin. PJRT_Client_Compile
// hands the StableHLO program to a compiler that palladium registers from
// Python; it describes the program through the pp_* builder functions as
// buffer slots and steps. Execute encodes kernel launches into one command
// buffer and commits without waiting. Host steps and branches need values the
// GPU has yet to compute: Execute commits what it has encoded, waits, and
// goes on in a new command buffer.
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "dispatch.h"
#include "errors.h"
#include "library.h"
#include "plugin.h"
#include "runtime.h"

namespace {

// A buffer an executable reads or writes: one of its parameters, a buffer
// allocated per execution, or a constant uploaded at compile time.
struct Slot {
    enum Kind { Parameter = 0, Fresh = 1, Constant = 2 } kind;
    PJRT_Buffer_Type type;
    std::vector<int64_t> dims;
    size_t bytes = 0;
    int parameter = -1;
    std::unique_ptr<Buffer> constant;
};

// One step of a program: a kernel launch, a call into the host, or a branch.
struct Kernel {
    enum Kind { Launch, Host, Branch } kind = Launch;
    // Host: the function's id; its buffers are `slots`. Branch: the steps to
    // skip (negative: back) when the first byte of `slots[0]` is zero, or
    // always when there is no slot.
    int host = -1;
    int64_t skip = 0;
    std::shared_ptr<Library> library;
    std::shared_ptr<ComputePipeline> pipeline;
    std::vector<int> slots;  // bound at buffer indices 0..n-1
    std::vector<size_t> offsets;  // byte offset of each binding
    Dim3 grid, threadgroup;  // an empty threadgroup lets the pipeline choose
};

struct Program {
    std::string code, compile_options;
    std::vector<Slot> slots;
    std::vector<Kernel> kernels;
    // A static loop replays launches with different slots. Compile each source
    // once per executable, retaining its library with the pipeline.
    std::map<std::tuple<std::string, std::string, int>,
             std::pair<std::shared_ptr<Library>, std::shared_ptr<ComputePipeline>>> pipelines;
    std::vector<int> outputs;
    std::vector<std::pair<int, int>> aliases;  // parameter, output
    int num_parameters = 0;

    // What the output queries return, kept alive here.
    std::vector<PJRT_Buffer_Type> output_types;
    std::vector<int64_t> output_dims;
    std::vector<size_t> output_ranks, memory_kind_sizes;
    std::vector<const char*> memory_kinds;
    std::string fingerprint;

    // Set by the compiler on failure.
    PJRT_Error_Code error_code = PJRT_Error_Code_OK;
    std::string error_message;

    ~Program();
};

using Compiler = int (*)(const char* code, size_t code_size, Program* builder);
Compiler compiler = nullptr;

// Runs host function `id` on the contents of its buffers; on failure returns
// nonzero and writes a message of at most `capacity` bytes.
using Host = int (*)(int id, void* const* data, size_t count, char* message, size_t capacity);
Host host = nullptr;

// Host functions of destroyed programs, for palladium._pjrt to forget. A
// program may be destroyed while the interpreter shuts down, so it does not
// call into Python itself.
std::mutex released_mutex;
std::vector<int>& released() {
    static auto* ids = new std::vector<int>;  // leaked: programs may outlive statics
    return *ids;
}

Program::~Program() {
    std::lock_guard lock(released_mutex);
    for (const Kernel& k : kernels) {
        if (k.kind == Kernel::Host) released().push_back(k.host);
    }
}

constexpr char kName[] = "metal_program";
constexpr char kMemoryKind[] = "device";

}  // namespace

struct PJRT_Executable {
    std::shared_ptr<Program> program;
};

// One replica and one computation on device 0, as a DeviceAssignmentProto:
// replica_count = 1, computation_count = 1, computation_devices { replica_device_ids: 0 }.
struct PJRT_DeviceAssignmentSerialized {
    const char bytes[8] = {0x08, 0x01, 0x10, 0x01, 0x1a, 0x02, 0x08, 0x00};
};

struct PJRT_LoadedExecutable {
    std::shared_ptr<Program> program;
    PJRT_Client* client;
    bool deleted = false;
};

extern "C" {

#define EXPORT __attribute__((visibility("default")))

// The compiler PJRT_Client_Compile calls; palladium._pjrt registers it.
EXPORT void palladium_pjrt_set_compiler(Compiler c) { compiler = c; }

EXPORT void pp_error(Program* p, int code, const char* message) {
    p->error_code = (PJRT_Error_Code)code;
    p->error_message = message;
}

// Adds a slot and returns its id; parameters number in the order they are added.
EXPORT int pp_slot(Program* p, int kind, int type, const int64_t* dims, size_t rank,
                   const void* data, size_t data_size) {
    Slot slot{(Slot::Kind)kind, (PJRT_Buffer_Type)type, {dims, dims + rank}};
    slot.bytes = element_size(slot.type);
    for (int64_t d : slot.dims) slot.bytes *= (size_t)d;
    if (kind == Slot::Parameter) slot.parameter = p->num_parameters++;
    if (kind == Slot::Constant) {
        try {
            slot.constant = std::make_unique<Buffer>(runtime().device(), data_size);
        } catch (const AllocationError& e) {
            pp_error(p, PJRT_Error_Code_RESOURCE_EXHAUSTED, e.what());
            return -1;
        }
        std::memcpy(slot.constant->contents(), data, data_size);
    }
    p->slots.push_back(std::move(slot));
    return (int)p->slots.size() - 1;
}

EXPORT int pp_kernel(Program* p, const char* msl, const char* function, int math_mode,
                     const int* slots, size_t num_slots, const size_t* grid,
                     const size_t* threadgroup, const size_t* offsets) {
    try {
        auto key = std::make_tuple(std::string(msl), std::string(function), math_mode);
        auto found = p->pipelines.find(key);
        if (found == p->pipelines.end()) {
            auto library = std::make_shared<Library>(runtime().device(), msl, (MathMode)math_mode);
            auto pipeline = library->pipeline(function);
            found = p->pipelines.emplace(std::move(key), std::make_pair(library, pipeline)).first;
        }
        const auto& [library, pipeline] = found->second;
        auto bound_offsets = offsets ? std::vector<size_t>(offsets, offsets + num_slots)
                                     : std::vector<size_t>(num_slots);
        Kernel kernel;
        kernel.library = library;
        kernel.pipeline = pipeline;
        kernel.slots.assign(slots, slots + num_slots);
        kernel.offsets = std::move(bound_offsets);
        kernel.grid = {grid[0], grid[1], grid[2]};
        kernel.threadgroup = {threadgroup[0], threadgroup[1], threadgroup[2]};
        p->kernels.push_back(std::move(kernel));
        return 0;
    } catch (const MetalError& e) {
        pp_error(p, PJRT_Error_Code_INTERNAL, e.what());
        return -1;
    }
}

// A call of host function `id` on the contents of `slots`, each from its byte
// offset, once the GPU has computed them.
EXPORT void pp_host(Program* p, int id, const int* slots, size_t num_slots,
                    const size_t* offsets) {
    Kernel step;
    step.kind = Kernel::Host;
    step.host = id;
    step.slots.assign(slots, slots + num_slots);
    step.offsets.assign(offsets, offsets + num_slots);
    p->kernels.push_back(std::move(step));
}

// Skips `skip` steps (back, when negative) if the first byte of `slot` is
// zero; always, when `slot` is negative.
EXPORT void pp_branch(Program* p, int slot, int64_t skip) {
    Kernel step;
    step.kind = Kernel::Branch;
    step.skip = skip;
    if (slot >= 0) step.slots.push_back(slot);
    p->kernels.push_back(std::move(step));
}

EXPORT void palladium_pjrt_set_host(Host h) { host = h; }

// Moves up to `capacity` ids of host functions whose programs are gone into
// `ids`, and returns how many.
EXPORT size_t palladium_pjrt_released(int* ids, size_t capacity) {
    std::lock_guard lock(released_mutex);
    size_t count = std::min(capacity, released().size());
    std::copy(released().end() - count, released().end(), ids);
    released().resize(released().size() - count);
    return count;
}

EXPORT void pp_output(Program* p, int slot) { p->outputs.push_back(slot); }
EXPORT void pp_alias(Program* p, int parameter, int output) {
    p->aliases.emplace_back(parameter, output);
}

}  // extern "C"

namespace {

PJRT_Error* compile(PJRT_Client_Compile_Args* args) {
    if (!compiler) {
        return error(PJRT_Error_Code_FAILED_PRECONDITION,
                     "metal: no compiler registered; load the plugin through palladium._pjrt");
    }
    const PJRT_Program* program = args->program;
    if (std::string(program->format, program->format_size) != "mlir") {
        return error(PJRT_Error_Code_UNIMPLEMENTED, "metal: programs must be StableHLO (mlir)");
    }
    auto p = std::make_shared<Program>();
    p->code.assign(program->code, program->code_size);
    p->compile_options.assign(args->compile_options, args->compile_options_size);
    if (compiler(p->code.data(), p->code.size(), p.get()) != 0) {
        return error(p->error_code ? p->error_code : PJRT_Error_Code_INTERNAL,
                     p->error_message.empty() ? "metal: compilation failed" : p->error_message);
    }
    for (int s : p->outputs) {
        const Slot& slot = p->slots[s];
        p->output_types.push_back(slot.type);
        p->output_dims.insert(p->output_dims.end(), slot.dims.begin(), slot.dims.end());
        p->output_ranks.push_back(slot.dims.size());
        p->memory_kinds.push_back(kMemoryKind);
        p->memory_kind_sizes.push_back(std::strlen(kMemoryKind));
    }
    p->fingerprint = std::to_string(std::hash<std::string>{}(p->code));
    args->executable = new PJRT_LoadedExecutable{p, args->client};
    return nullptr;
}

PJRT_Error* new_output(PJRT_Device* device, const Slot& slot, PJRT_Buffer** out,
                       std::shared_ptr<Buffer> storage = nullptr) {
    return make_buffer(device, slot.type, slot.dims.data(), slot.dims.size(), out,
                       std::move(storage));
}

// Executions the GPU has not finished, oldest first. Execute waits for the
// oldest beyond kMaxInFlight, so a caller running ahead of the GPU reuses
// pooled buffers instead of allocating for every queued call.
constexpr size_t kMaxInFlight = 16;
std::mutex in_flight_mutex;
std::deque<std::shared_ptr<Completion>> in_flight;

// Execution counters for profiling, read through palladium_pjrt_stats. GPU
// intervals come from completion handlers, which Metal may call out of start
// order, so each one is merged into the union wherever it falls.
struct Stats {
    std::mutex mutex;
    uint64_t executions = 0, command_buffers = 0, launches = 0;
    std::chrono::duration<double> encode{}, throttle{};
    // Command-buffer GPU intervals: their sum, and their union as disjoint
    // intervals by start time. Only the newest kMaxIntervals stay mergeable;
    // older ones are folded into `gpu_busy_folded`.
    double gpu_sum = 0, gpu_busy_folded = 0;
    std::map<double, double> gpu_busy;
};
// Far more than the executions in flight, whose handlers may arrive out of
// start order; an interval this old no longer overlaps a new one.
constexpr size_t kMaxIntervals = 1024;
Stats& stats() {
    static Stats* s = new Stats;  // leaked: completion handlers may outlive statics
    return *s;
}

// Merges the interval into the union, so memory stays bounded however long
// the process runs.
void record_gpu(double start, double end) {
    if (!(end > start)) return;
    Stats& s = stats();
    std::lock_guard lock(s.mutex);
    s.gpu_sum += end - start;
    auto it = s.gpu_busy.upper_bound(start);
    if (it != s.gpu_busy.begin() && std::prev(it)->second >= start) start = (--it)->first;
    while (it != s.gpu_busy.end() && it->first <= end) {
        end = std::max(end, it->second);
        it = s.gpu_busy.erase(it);
    }
    s.gpu_busy.emplace(start, end);
    while (s.gpu_busy.size() > kMaxIntervals) {
        s.gpu_busy_folded += s.gpu_busy.begin()->second - s.gpu_busy.begin()->first;
        s.gpu_busy.erase(s.gpu_busy.begin());
    }
}

// With PALLADIUM_PJRT_PROFILE set, every launch runs in a command buffer of
// its own and the GPU seconds of each step of the last execution are kept,
// for palladium_pjrt_profile. Slower: nothing overlaps.
bool profiling() {
    static const bool on = std::getenv("PALLADIUM_PJRT_PROFILE") != nullptr;
    return on;
}
std::mutex profile_mutex;
std::vector<double>& last_profile() {
    static auto* seconds = new std::vector<double>;
    return *seconds;
}

void throttle() {
    std::unique_lock lock(in_flight_mutex);
    while (!in_flight.empty() && (in_flight.front()->ready() || in_flight.size() >= kMaxInFlight)) {
        std::shared_ptr<Completion> oldest = in_flight.front();
        in_flight.pop_front();
        lock.unlock();
        oldest->wait();
        lock.lock();
    }
}

// Encodes the kernels into one command buffer and returns without waiting:
// the outputs are defined, and `completion` set, once the GPU is done. A host
// step or a conditional branch first waits for the work encoded so far.
PJRT_Error* execute(std::shared_ptr<Program> program, PJRT_Device* device,
                    PJRT_Buffer* const* args, size_t num_args, PJRT_Buffer** outputs,
                    std::shared_ptr<Completion> completion, const PJRT_ExecuteOptions* options) {
    const Program& p = *program;
    if ((int)num_args != p.num_parameters) {
        return error(PJRT_Error_Code_INVALID_ARGUMENT,
                     "metal: the program takes " + std::to_string(p.num_parameters) +
                         " arguments, got " + std::to_string(num_args));
    }
    auto throttle_start = std::chrono::steady_clock::now();
    throttle();
    auto encode_start = std::chrono::steady_clock::now();
    size_t launches = 0;
    bool committed = false;
    std::vector<Buffer*> bound(p.slots.size());
    std::vector<std::shared_ptr<Buffer>> retained;
    // Storage only: most fresh slots are intermediates, which need no handle.
    std::vector<std::shared_ptr<Buffer>> fresh(p.slots.size());
    for (size_t s = 0; s < p.slots.size(); ++s) {
        const Slot& slot = p.slots[s];
        if (slot.kind == Slot::Parameter) {
            if (PJRT_Error* e = deleted(args[slot.parameter])) return e;
            bound[s] = args[slot.parameter]->storage.get();
            retained.push_back(args[slot.parameter]->storage);
        } else if (slot.kind == Slot::Constant) {
            bound[s] = slot.constant.get();
        } else {
            try {
                fresh[s] = make_storage(slot.bytes);
            } catch (const AllocationError& e) {
                return error(PJRT_Error_Code_RESOURCE_EXHAUSTED, e.what());
            }
            bound[s] = fresh[s].get();
            retained.push_back(fresh[s]);
        }
    }
    // A fresh slot becomes its first output; parameters, constants and repeated
    // outputs are copied into new buffers once the kernels are done.
    std::vector<std::unique_ptr<PJRT_Buffer>> results(p.outputs.size());
    // Two blit passes: outputs first snapshot parameters, constants and
    // repeated slots, then results are copied back into donated storage. Each
    // pass writes distinct buffers that no other copy in it reads.
    std::vector<std::pair<const Buffer*, Buffer*>> snapshots, copybacks;
    for (size_t i = 0; i < p.outputs.size(); ++i) {
        int s = p.outputs[i];
        bool first = fresh[s] != nullptr;
        PJRT_Buffer* buffer;
        if (PJRT_Error* e = new_output(device, p.slots[s], &buffer, std::move(fresh[s]))) return e;
        results[i].reset(buffer);
        if (first) continue;
        snapshots.emplace_back(bound[s], buffer->storage.get());
        retained.push_back(buffer->storage);
    }
    std::vector<int> donated;
    for (auto [parameter, output] : p.aliases) {
        bool allowed = true;
        if (options) {
            for (size_t i = 0; i < options->num_non_donatable_input_indices; ++i) {
                if (options->non_donatable_input_indices[i] == parameter) allowed = false;
            }
        }
        // A repeated argument must remain available through its other position.
        if (std::count(args, args + num_args, args[parameter]) != 1) allowed = false;
        if (!allowed) continue;
        auto storage = args[parameter]->storage;
        copybacks.emplace_back(results[output]->storage.get(), storage.get());
        results[output]->storage = std::move(storage);
        donated.push_back(parameter);
    }
    // Independent owners avoid a shared last-use marker, whose update can race
    // or occur in a different order from command-buffer submission.
    auto finish = [program, completion, retained = std::move(retained)](
                      const char* failure, double gpu_start, double gpu_end) mutable {
        record_gpu(gpu_start, gpu_end);
        // The GPU is done with the storage. Give it up before publishing
        // readiness, so a caller that then drops its outputs and runs again
        // finds every buffer of this execution back in the pool.
        retained.clear();
        if (failure) return completion->set(PJRT_Error_Code_INTERNAL, failure);
        completion->set();
    };
    size_t waits = 0;
    try {
        auto batch = std::make_unique<CommandBatch>(runtime());
        bool encoded = false;
        bool inputs_ready = false;
        // Makes every buffer readable on the host: the work encoded so far
        // and, the first time, the arguments other executions still compute.
        auto synchronize = [&]() -> PJRT_Error* {
            if (encoded) {
                batch->wait();
                if (auto seconds = batch->gpu_time()) {
                    // No other execution overlaps a command buffer this one waits for.
                    Stats& st = stats();
                    std::lock_guard lock(st.mutex);
                    st.gpu_sum += *seconds;
                    st.gpu_busy_folded += *seconds;
                }
                batch = std::make_unique<CommandBatch>(runtime());
                encoded = false;
                ++waits;
            }
            if (!inputs_ready) {
                for (size_t i = 0; i < num_args; ++i) {
                    if (PJRT_Error* e = wait_ready(args[i])) return e;
                }
                inputs_ready = true;
            }
            return nullptr;
        };
        Launch launch;  // reused: add() encodes it at once, and buffers keep their capacity
        std::vector<void*> contents;
        std::vector<double> profile(profiling() ? p.kernels.size() : 0);
        for (size_t pc = 0; pc < p.kernels.size(); ++pc) {
            const Kernel& k = p.kernels[pc];
            if (k.kind == Kernel::Branch) {
                bool taken = true;
                if (!k.slots.empty()) {
                    if (PJRT_Error* e = synchronize()) return e;
                    taken = *static_cast<const char*>(bound[k.slots[0]]->contents()) == 0;
                }
                if (taken) pc += k.skip;
                continue;
            }
            if (k.kind == Kernel::Host) {
                if (PJRT_Error* e = synchronize()) return e;
                contents.clear();
                for (size_t j = 0; j < k.slots.size(); ++j) {
                    char* data = static_cast<char*>(bound[k.slots[j]]->contents());
                    contents.push_back(data ? data + k.offsets[j] : nullptr);
                }
                char message[4096] = "metal: no host function handler registered";
                if (!host || host(k.host, contents.data(), contents.size(), message,
                                  sizeof message - 1) != 0) {
                    message[sizeof message - 1] = 0;
                    return error(PJRT_Error_Code_INTERNAL, message);
                }
                continue;
            }
            if (!k.grid.volume()) continue;
            launch.pipeline = k.pipeline.get();
            launch.buffers.clear();
            for (size_t j = 0; j < k.slots.size(); ++j) {
                launch.buffers.push_back({bound[k.slots[j]], k.offsets[j]});
            }
            launch.grid = k.grid;
            launch.threadgroup =
                k.threadgroup.volume() ? k.threadgroup : k.pipeline->default_threadgroup(k.grid);
            batch->add(launch);
            encoded = true;
            ++launches;
            if (profiling()) {
                batch->wait();
                profile[pc] += batch->gpu_time().value_or(0);
                batch = std::make_unique<CommandBatch>(runtime());
                encoded = false;
                ++waits;
            }
        }
        if (profiling()) {
            std::lock_guard lock(profile_mutex);
            last_profile() = std::move(profile);
        }
        // A host memcpy in the completion handler can lag the next command
        // buffer. GPU copies make passthrough/repeated outputs safe to chain.
        for (const auto* pass : {&snapshots, &copybacks}) {
            batch->copy(*pass);
            for (auto [src, dst] : *pass) encoded |= dst->size() != 0;
        }
        if (encoded) {
            batch->commit(std::move(finish));
            committed = true;
            std::lock_guard lock(in_flight_mutex);
            in_flight.push_back(completion);
        } else {
            // Empty work still joins its input readiness below, asynchronously.
            finish(nullptr, 0, 0);
        }
    } catch (const std::exception& e) {
        return error(PJRT_Error_Code_INTERNAL, std::string("metal: ") + e.what());
    }
    // Publish deletion only after successful submission. All original reads and
    // output snapshots precede the copybacks, including swapped/aliased outputs.
    for (int parameter : donated) args[parameter]->storage.reset();
    // The caller publishes readiness: this completion joined with the inputs'.
    for (size_t i = 0; i < results.size(); ++i) outputs[i] = results[i].release();
    {
        Stats& s = stats();
        std::lock_guard lock(s.mutex);
        ++s.executions;
        s.command_buffers += committed + waits;
        s.launches += launches;
        s.throttle += encode_start - throttle_start;
        s.encode += std::chrono::steady_clock::now() - encode_start;
    }
    return nullptr;
}

}  // namespace

extern "C" {

// Counters since the last reset. Host seconds are spent inside Execute:
// `throttle_seconds` waiting for the in-flight bound, `encode_seconds` binding,
// encoding and committing. `gpu_busy_seconds` is the union of command-buffer
// GPU intervals, `gpu_sum_seconds` their sum. Completion handlers still in
// flight are not counted yet: synchronize before reading.
struct PalladiumPjrtStats {
    uint64_t executions, command_buffers, launches;
    double encode_seconds, throttle_seconds, gpu_busy_seconds, gpu_sum_seconds;
    // Buffer pool: hits, new MTLBuffers (count, bytes, seconds), dropped releases.
    uint64_t pool_hits, pool_misses, allocated_bytes, pool_dropped;
    double allocation_seconds;
};

// Copies the GPU seconds of each step of the last execution into `seconds`,
// up to `capacity`, and returns the number of steps. Empty unless
// PALLADIUM_PJRT_PROFILE is set.
EXPORT size_t palladium_pjrt_profile(double* seconds, size_t capacity) {
    std::lock_guard lock(profile_mutex);
    const auto& profile = last_profile();
    std::copy_n(profile.begin(), std::min(capacity, profile.size()), seconds);
    return profile.size();
}

EXPORT void palladium_pjrt_stats(PalladiumPjrtStats* out, int reset) {
    Stats& s = stats();
    std::lock_guard lock(s.mutex);
    PoolCounters pool = pool_counters(reset);
    double gpu_busy = s.gpu_busy_folded;
    for (auto [start, end] : s.gpu_busy) gpu_busy += end - start;
    *out = {s.executions, s.command_buffers, s.launches, s.encode.count(), s.throttle.count(),
            gpu_busy,     s.gpu_sum,         pool.hits,  pool.misses,   pool.miss_bytes,
            pool.dropped, pool.miss_seconds.count()};
    if (reset) {
        s.executions = s.command_buffers = s.launches = 0;
        s.encode = s.throttle = {};
        s.gpu_sum = s.gpu_busy_folded = 0;
        s.gpu_busy.clear();
    }
}

}  // extern "C"

void add_executable_api(PJRT_Api& api) {
    api.PJRT_Client_Compile = compile;

    api.PJRT_Executable_Destroy = [](PJRT_Executable_Destroy_Args* args) -> PJRT_Error* {
        delete args->executable;
        return nullptr;
    };
    api.PJRT_Executable_Name = [](PJRT_Executable_Name_Args* args) -> PJRT_Error* {
        args->executable_name = kName;
        args->executable_name_size = std::strlen(kName);
        return nullptr;
    };
    api.PJRT_Executable_NumReplicas = [](PJRT_Executable_NumReplicas_Args* args) -> PJRT_Error* {
        args->num_replicas = 1;
        return nullptr;
    };
    api.PJRT_Executable_NumPartitions =
        [](PJRT_Executable_NumPartitions_Args* args) -> PJRT_Error* {
        args->num_partitions = 1;
        return nullptr;
    };
    api.PJRT_Executable_NumOutputs = [](PJRT_Executable_NumOutputs_Args* args) -> PJRT_Error* {
        args->num_outputs = args->executable->program->outputs.size();
        return nullptr;
    };
    api.PJRT_Executable_SizeOfGeneratedCodeInBytes =
        [](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) -> PJRT_Error* {
        args->size_in_bytes = 0;
        return nullptr;
    };
    api.PJRT_Executable_GetCostAnalysis =
        [](PJRT_Executable_GetCostAnalysis_Args* args) -> PJRT_Error* {
        args->num_properties = 0;
        args->properties = nullptr;
        return nullptr;
    };
    api.PJRT_Executable_OutputElementTypes =
        [](PJRT_Executable_OutputElementTypes_Args* args) -> PJRT_Error* {
        Program& p = *args->executable->program;
        args->output_types = p.output_types.data();
        args->num_output_types = p.output_types.size();
        return nullptr;
    };
    api.PJRT_Executable_OutputDimensions =
        [](PJRT_Executable_OutputDimensions_Args* args) -> PJRT_Error* {
        Program& p = *args->executable->program;
        args->num_outputs = p.outputs.size();
        args->dims = p.output_dims.data();
        args->dim_sizes = p.output_ranks.data();
        return nullptr;
    };
    api.PJRT_Executable_OutputMemoryKinds =
        [](PJRT_Executable_OutputMemoryKinds_Args* args) -> PJRT_Error* {
        Program& p = *args->executable->program;
        args->num_outputs = p.outputs.size();
        args->memory_kinds = p.memory_kinds.data();
        args->memory_kind_sizes = p.memory_kind_sizes.data();
        return nullptr;
    };
    // The input program; palladium does not rewrite StableHLO.
    api.PJRT_Executable_OptimizedProgram =
        [](PJRT_Executable_OptimizedProgram_Args* args) -> PJRT_Error* {
        const std::string& code = args->executable->program->code;
        PJRT_Program* program = args->program;
        if (program->code) std::memcpy(program->code, code.data(), code.size());
        program->code_size = code.size();
        program->format = "mlir";
        program->format_size = 4;
        return nullptr;
    };
    api.PJRT_Executable_Fingerprint = [](PJRT_Executable_Fingerprint_Args* args) -> PJRT_Error* {
        const std::string& f = args->executable->program->fingerprint;
        args->executable_fingerprint = f.data();
        args->executable_fingerprint_size = f.size();
        return nullptr;
    };
    api.PJRT_Executable_Serialize = [](PJRT_Executable_Serialize_Args*) -> PJRT_Error* {
        return error(PJRT_Error_Code_UNIMPLEMENTED, "metal: executables do not serialize");
    };
    api.PJRT_Executable_DeserializeAndLoad =
        [](PJRT_Executable_DeserializeAndLoad_Args*) -> PJRT_Error* {
        return error(PJRT_Error_Code_UNIMPLEMENTED, "metal: executables do not serialize");
    };
    api.PJRT_Executable_GetCompiledMemoryStats =
        [](PJRT_Executable_GetCompiledMemoryStats_Args*) -> PJRT_Error* {
        return error(PJRT_Error_Code_UNIMPLEMENTED, "metal: no compiled memory stats");
    };

    api.PJRT_LoadedExecutable_Destroy =
        [](PJRT_LoadedExecutable_Destroy_Args* args) -> PJRT_Error* {
        delete args->executable;
        return nullptr;
    };
    api.PJRT_LoadedExecutable_GetExecutable =
        [](PJRT_LoadedExecutable_GetExecutable_Args* args) -> PJRT_Error* {
        args->executable = new PJRT_Executable{args->loaded_executable->program};
        return nullptr;
    };
    api.PJRT_LoadedExecutable_AddressableDevices =
        [](PJRT_LoadedExecutable_AddressableDevices_Args* args) -> PJRT_Error* {
        args->addressable_devices = &args->executable->client->device_ptr;
        args->num_addressable_devices = 1;
        return nullptr;
    };
    api.PJRT_LoadedExecutable_GetDeviceAssignment =
        [](PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) -> PJRT_Error* {
        static PJRT_DeviceAssignmentSerialized assignment;
        args->serialized_bytes = assignment.bytes;
        args->serialized_bytes_size = sizeof(assignment.bytes);
        args->serialized_device_assignment = &assignment;
        args->serialized_device_assignment_deleter = [](PJRT_DeviceAssignmentSerialized*) {};
        return nullptr;
    };
    api.PJRT_LoadedExecutable_Delete = [](PJRT_LoadedExecutable_Delete_Args* args) -> PJRT_Error* {
        args->executable->deleted = true;
        return nullptr;
    };
    api.PJRT_LoadedExecutable_IsDeleted =
        [](PJRT_LoadedExecutable_IsDeleted_Args* args) -> PJRT_Error* {
        args->is_deleted = args->executable->deleted;
        return nullptr;
    };
    api.PJRT_LoadedExecutable_Fingerprint =
        [](PJRT_LoadedExecutable_Fingerprint_Args* args) -> PJRT_Error* {
        const std::string& f = args->executable->program->fingerprint;
        args->executable_fingerprint = f.data();
        args->executable_fingerprint_size = f.size();
        return nullptr;
    };
    api.PJRT_LoadedExecutable_Execute =
        [](PJRT_LoadedExecutable_Execute_Args* args) -> PJRT_Error* {
        PJRT_LoadedExecutable* exe = args->executable;
        if (exe->deleted) {
            return error(PJRT_Error_Code_FAILED_PRECONDITION, "metal: the executable is deleted");
        }
        if (args->num_devices != 1) {
            return error(PJRT_Error_Code_INVALID_ARGUMENT, "metal: one device per execution");
        }
        PJRT_Device* device =
            args->execute_device ? args->execute_device : &exe->client->device;
        auto completion = std::make_shared<Completion>();
        std::vector<std::shared_ptr<Completion>> dependencies;
        for (size_t i = 0; i < args->num_args; ++i) {
            dependencies.push_back(args->argument_lists[0][i]->ready);
        }
        if (PJRT_Error* e = execute(exe->program, device, args->argument_lists[0],
                                    args->num_args, args->output_lists[0], completion, args->options)) {
            return e;
        }
        // Queue ordering covers GPU data dependencies, but completion callbacks
        // may run out of order. Publish success only after both the GPU work and
        // all input completions have succeeded, including transitive failures.
        dependencies.push_back(completion);
        completion = Completion::join(std::move(dependencies));
        for (size_t i = 0; i < exe->program->outputs.size(); ++i) {
            args->output_lists[0][i]->ready = completion;
        }
        if (args->device_complete_events) {
            args->device_complete_events[0] = new PJRT_Event{completion};
        }
        return nullptr;
    };
}
