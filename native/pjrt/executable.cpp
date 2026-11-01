// Compilation and execution for the `metal` PJRT plugin. PJRT_Client_Compile
// hands the StableHLO program to a compiler that palladium registers from
// Python; it describes the program through the pp_* builder functions as
// buffer slots and kernel launches, which Execute encodes into one command
// buffer and commits without waiting.
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
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
    int parameter = -1;
    std::unique_ptr<Buffer> constant;
};

struct Kernel {
    std::shared_ptr<Library> library;
    std::shared_ptr<ComputePipeline> pipeline;
    std::vector<int> slots;  // bound at buffer indices 0..n-1
    Dim3 grid, threadgroup;  // an empty threadgroup lets the pipeline choose
};

struct Program {
    std::string code, compile_options;
    std::vector<Slot> slots;
    std::vector<Kernel> kernels;
    std::vector<int> outputs;
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
};

using Compiler = int (*)(const char* code, size_t code_size, Program* builder);
Compiler compiler = nullptr;

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
                     const size_t* threadgroup) {
    try {
        auto library =
            std::make_shared<Library>(runtime().device(), msl, (MathMode)math_mode);
        auto pipeline = library->pipeline(function);
        p->kernels.push_back({library, pipeline, {slots, slots + num_slots},
                              {grid[0], grid[1], grid[2]},
                              {threadgroup[0], threadgroup[1], threadgroup[2]}});
        return 0;
    } catch (const MetalError& e) {
        pp_error(p, PJRT_Error_Code_INTERNAL, e.what());
        return -1;
    }
}

EXPORT void pp_output(Program* p, int slot) { p->outputs.push_back(slot); }

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

PJRT_Error* new_output(PJRT_Device* device, const Slot& slot, PJRT_Buffer** out) {
    return make_buffer(device, slot.type, slot.dims.data(), slot.dims.size(), out);
}

// Executions the GPU has not finished, oldest first. Execute waits for the
// oldest beyond kMaxInFlight, so a caller running ahead of the GPU reuses
// pooled buffers instead of allocating for every queued call.
constexpr size_t kMaxInFlight = 16;
std::mutex in_flight_mutex;
std::deque<std::shared_ptr<Completion>> in_flight;

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

// Encodes every kernel into one command buffer and returns without waiting:
// the outputs are defined, and `completion` set, once the GPU is done.
PJRT_Error* execute(std::shared_ptr<Program> program, PJRT_Device* device,
                    PJRT_Buffer* const* args, size_t num_args, PJRT_Buffer** outputs,
                    std::shared_ptr<Completion> completion) {
    const Program& p = *program;
    if ((int)num_args != p.num_parameters) {
        return error(PJRT_Error_Code_INVALID_ARGUMENT,
                     "metal: the program takes " + std::to_string(p.num_parameters) +
                         " arguments, got " + std::to_string(num_args));
    }
    throttle();
    std::vector<Buffer*> bound(p.slots.size());
    std::vector<std::unique_ptr<PJRT_Buffer>> fresh(p.slots.size());
    for (size_t s = 0; s < p.slots.size(); ++s) {
        const Slot& slot = p.slots[s];
        if (slot.kind == Slot::Parameter) {
            if (PJRT_Error* e = deleted(args[slot.parameter])) return e;
            bound[s] = args[slot.parameter]->storage.get();
        } else if (slot.kind == Slot::Constant) {
            bound[s] = slot.constant.get();
        } else {
            PJRT_Buffer* buffer;
            if (PJRT_Error* e = new_output(device, slot, &buffer)) return e;
            fresh[s].reset(buffer);
            bound[s] = buffer->storage.get();
        }
    }
    // A fresh slot becomes its first output; parameters, constants and repeated
    // outputs are copied into new buffers once the kernels are done.
    std::vector<std::unique_ptr<PJRT_Buffer>> results(p.outputs.size());
    std::vector<std::pair<Buffer*, Buffer*>> copies;
    for (size_t i = 0; i < p.outputs.size(); ++i) {
        int s = p.outputs[i];
        if (fresh[s]) {
            results[i] = std::move(fresh[s]);
            continue;
        }
        PJRT_Buffer* buffer;
        if (PJRT_Error* e = new_output(device, p.slots[s], &buffer)) return e;
        results[i].reset(buffer);
        copies.emplace_back(bound[s], buffer->storage.get());
    }
    auto finish = [program, completion, copies](const char* failure) {
        if (failure) return completion->set(PJRT_Error_Code_INTERNAL, failure);
        for (auto [src, dst] : copies) std::memcpy(dst->contents(), src->contents(), dst->size());
        completion->set();
    };
    try {
        CommandBatch batch(runtime());
        bool encoded = false;
        for (const Kernel& k : p.kernels) {
            if (!k.grid.volume()) continue;
            Launch launch;
            launch.pipeline = k.pipeline.get();
            for (int s : k.slots) launch.buffers.push_back({bound[s], 0});
            launch.grid = k.grid;
            launch.threadgroup =
                k.threadgroup.volume() ? k.threadgroup : k.pipeline->default_threadgroup(k.grid);
            batch.add(launch);
            encoded = true;
        }
        if (encoded) {
            batch.commit(finish);
            std::lock_guard lock(in_flight_mutex);
            in_flight.push_back(completion);
        } else {
            // Nothing for the GPU: copy once the parameters are defined.
            for (int i = 0; i < p.num_parameters; ++i) {
                if (PJRT_Error* e = wait_ready(args[i])) return e;
            }
            finish(nullptr);
        }
    } catch (const std::exception& e) {
        return error(PJRT_Error_Code_INTERNAL, std::string("metal: ") + e.what());
    }
    // Buffers keep their memory out of the pool until the GPU is done with them.
    for (int i = 0; i < p.num_parameters; ++i) args[i]->last_use = completion;
    for (size_t i = 0; i < results.size(); ++i) {
        results[i]->ready = results[i]->last_use = completion;
        outputs[i] = results[i].release();
    }
    for (auto& buffer : fresh) {
        if (buffer) buffer->last_use = completion;  // computed but not an output
    }
    return nullptr;
}

}  // namespace

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
        if (PJRT_Error* e = execute(exe->program, device, args->argument_lists[0],
                                    args->num_args, args->output_lists[0], completion)) {
            return e;
        }
        if (args->device_complete_events) {
            args->device_complete_events[0] = new PJRT_Event{completion};
        }
        return nullptr;
    };
}
