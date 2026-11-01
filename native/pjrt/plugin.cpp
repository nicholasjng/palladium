// PJRT C API plugin for the `metal` platform: the system's Apple GPU as one
// device with one memory, buffers in shared MTLBuffers on unified memory.
// Work is synchronous so far, so every event is created ready; functions left
// null in the API table are unsupported. Compilation is in executable.cpp.
#include "pjrt.h"

#include <cstring>

#include "errors.h"
#include "metal.h"
#include "runtime.h"

namespace {

constexpr char kPlatformName[] = "metal";
constexpr char kPlatformVersion[] = "palladium 0.1.0";
constexpr char kMemoryKind[] = "device";

const Error& as_error(const PJRT_Error* e) { return *static_cast<const Error*>(e); }

const PJRT_Error_FunctionTable kErrorTable = {
    PJRT_Error_FunctionTable_STRUCT_SIZE,
    PJRT_Error_STRUCT_SIZE,
    nullptr,
    [](PJRT_Error* e) { delete static_cast<Error*>(e); },
    [](const PJRT_Error* e, const char** message, size_t* size) {
        *message = as_error(e).message.data();
        *size = as_error(e).message.size();
    },
    [](const PJRT_Error* e) { return as_error(e).code; },
    [](const PJRT_Error*, PJRT_Error_PayloadVisitor, void*) {},
};

const PJRT_Memory_FunctionTable kMemoryTable = {
    PJRT_Memory_FunctionTable_STRUCT_SIZE,
    nullptr,
    PJRT_Memory_STRUCT_SIZE,
    [](PJRT_Memory* m, const void* key) -> void* {
        auto* memory = static_cast<Memory*>(m);
        std::lock_guard lock(memory->mutex);
        auto it = memory->user_data.find(key);
        return it == memory->user_data.end() ? nullptr : it->second.first;
    },
    [](PJRT_Memory* m, const void* key, void* data, void (*dtor)(void*)) {
        auto* memory = static_cast<Memory*>(m);
        std::lock_guard lock(memory->mutex);
        auto [it, inserted] = memory->user_data.try_emplace(key, data, dtor);
        if (!inserted) {
            if (it->second.second) it->second.second(it->second.first);
            it->second = {data, dtor};
        }
    },
};

}  // namespace

PJRT_Error* error(PJRT_Error_Code code, std::string message) {
    auto* e = new Error;
    e->vtable = &kErrorTable;
    e->code = code;
    e->message = std::move(message);
    return e;
}

size_t element_size(PJRT_Buffer_Type type) {
    switch (type) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
        case PJRT_Buffer_Type_F8E5M2:
        case PJRT_Buffer_Type_F8E4M3FN:
        case PJRT_Buffer_Type_F8E4M3B11FNUZ:
        case PJRT_Buffer_Type_F8E5M2FNUZ:
        case PJRT_Buffer_Type_F8E4M3FNUZ:
        case PJRT_Buffer_Type_F8E4M3:
        case PJRT_Buffer_Type_F8E3M4:
        case PJRT_Buffer_Type_F8E8M0FNU:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
        case PJRT_Buffer_Type_F16:
        case PJRT_Buffer_Type_BF16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
        case PJRT_Buffer_Type_C64:
            return 8;
        case PJRT_Buffer_Type_C128:
            return 16;
        default:
            return 0;
    }
}

size_t byte_size(const PJRT_Buffer* buffer) {
    size_t n = element_size(buffer->type);
    for (int64_t d : buffer->dims) n *= (size_t)d;
    return n;
}

PJRT_Error* make_buffer(PJRT_Device* device, PJRT_Buffer_Type type, const int64_t* dims,
                        size_t num_dims, PJRT_Buffer** out) {
    if (!element_size(type)) {
        return error(PJRT_Error_Code_UNIMPLEMENTED,
                     "metal: buffers of PJRT element type " + std::to_string(type) +
                         " are not supported");
    }
    auto buffer = std::make_unique<PJRT_Buffer>();
    buffer->device = device;
    buffer->type = type;
    buffer->dims.assign(dims, dims + num_dims);
    for (size_t i = num_dims; i-- > 0;) buffer->minor_to_major.push_back((int64_t)i);
    try {
        buffer->storage = std::make_unique<Buffer>(runtime().device(), byte_size(buffer.get()));
    } catch (const AllocationError& e) {
        return error(PJRT_Error_Code_RESOURCE_EXHAUSTED, e.what());
    }
    *out = buffer.release();
    return nullptr;
}

PJRT_Error* deleted(const PJRT_Buffer* buffer) {
    return buffer->storage ? nullptr
                           : error(PJRT_Error_Code_FAILED_PRECONDITION,
                                   "metal: the buffer has been deleted");
}

namespace {

Memory* as_memory(PJRT_Memory* m) { return static_cast<Memory*>(m); }

// Copies a strided host array into dense row-major memory, one memcpy per
// dense block.
void copy_strided(char* dst, const char* src, const int64_t* dims, const int64_t* strides,
                  size_t rank, size_t element) {
    size_t size = element;
    bool dense = true;
    for (size_t i = rank; i-- > 0;) {
        dense = dense && (dims[i] == 1 || strides[i] == (int64_t)size);
        size *= (size_t)dims[i];
    }
    if (dense || size == 0) {
        std::memcpy(dst, src, size);
        return;
    }
    size_t inner = size / (size_t)dims[0];
    for (int64_t i = 0; i < dims[0]; ++i) {
        copy_strided(dst + i * inner, src + i * strides[0], dims + 1, strides + 1, rank - 1,
                     element);
    }
}

bool is_row_major(const PJRT_Buffer_MemoryLayout* layout, size_t rank) {
    if (!layout) return true;
    if (layout->type != PJRT_Buffer_MemoryLayout_Type_Tiled || layout->tiled.num_tiles) {
        return false;
    }
    for (size_t i = 0; i < layout->tiled.minor_to_major_size; ++i) {
        if (layout->tiled.minor_to_major[i] != (int64_t)(rank - 1 - i)) return false;
    }
    return true;
}

PJRT_Error* copy_buffer(const PJRT_Buffer* src, PJRT_Device* device, PJRT_Buffer** out) {
    if (PJRT_Error* e = deleted(src)) return e;
    if (PJRT_Error* e = make_buffer(device, src->type, src->dims.data(), src->dims.size(), out)) {
        return e;
    }
    std::memcpy((*out)->storage->contents(), src->storage->contents(), byte_size(src));
    return nullptr;
}

PJRT_Api make_api() {
    PJRT_Api api{};
    api.struct_size = PJRT_Api_STRUCT_SIZE;
    api.pjrt_api_version = {PJRT_Api_Version_STRUCT_SIZE, nullptr, PJRT_API_MAJOR, PJRT_API_MINOR};

    api.PJRT_Error_Destroy = [](PJRT_Error_Destroy_Args* args) {
        delete static_cast<Error*>(args->error);
    };
    api.PJRT_Error_Message = [](PJRT_Error_Message_Args* args) {
        args->message = as_error(args->error).message.data();
        args->message_size = as_error(args->error).message.size();
    };
    api.PJRT_Error_GetCode = [](PJRT_Error_GetCode_Args* args) -> PJRT_Error* {
        args->code = as_error(args->error).code;
        return nullptr;
    };
    api.PJRT_Error_ForEachPayload = [](PJRT_Error_ForEachPayload_Args*) -> PJRT_Error* {
        return nullptr;
    };

    api.PJRT_Plugin_Initialize = [](PJRT_Plugin_Initialize_Args*) -> PJRT_Error* {
        return nullptr;
    };
    api.PJRT_Plugin_Attributes = [](PJRT_Plugin_Attributes_Args* args) -> PJRT_Error* {
        args->attributes = nullptr;
        args->num_attributes = 0;
        return nullptr;
    };

    api.PJRT_Event_Destroy = [](PJRT_Event_Destroy_Args* args) -> PJRT_Error* {
        delete args->event;
        return nullptr;
    };
    api.PJRT_Event_IsReady = [](PJRT_Event_IsReady_Args* args) -> PJRT_Error* {
        args->is_ready = true;
        return nullptr;
    };
    api.PJRT_Event_Error = [](PJRT_Event_Error_Args*) -> PJRT_Error* { return nullptr; };
    api.PJRT_Event_Await = [](PJRT_Event_Await_Args*) -> PJRT_Error* { return nullptr; };
    api.PJRT_Event_OnReady = [](PJRT_Event_OnReady_Args* args) -> PJRT_Error* {
        args->callback(nullptr, args->user_arg);
        return nullptr;
    };

    api.PJRT_Client_Create = [](PJRT_Client_Create_Args* args) -> PJRT_Error* {
        try {
            MetalRuntime& rt = runtime();
            auto client = std::make_unique<PJRT_Client>();
            client->device.memory.vtable = &kMemoryTable;
            client->device.memory.device = &client->device;
            PJRT_DeviceDescription& d = client->device.description;
            d.kind = rt.device_name();
            d.debug_string = "METAL_0";
            d.to_string = "MetalDevice(id=0, kind=" + d.kind + ")";
            d.apple_gpu_family = rt.apple_gpu_family();
            PJRT_NamedValue& family = d.attributes[0];
            family = {};
            family.struct_size = PJRT_NamedValue_STRUCT_SIZE;
            family.name = "apple_gpu_family";
            family.name_size = std::strlen(family.name);
            family.type = PJRT_NamedValue_kInt64;
            family.int64_value = d.apple_gpu_family;
            family.value_size = 1;
            args->client = client.release();
            return nullptr;
        } catch (const NoDeviceError& e) {
            return error(PJRT_Error_Code_UNAVAILABLE, e.what());
        }
    };
    api.PJRT_Client_Destroy = [](PJRT_Client_Destroy_Args* args) -> PJRT_Error* {
        delete args->client;
        return nullptr;
    };
    api.PJRT_Client_PlatformName = [](PJRT_Client_PlatformName_Args* args) -> PJRT_Error* {
        args->platform_name = kPlatformName;
        args->platform_name_size = std::strlen(kPlatformName);
        return nullptr;
    };
    api.PJRT_Client_ProcessIndex = [](PJRT_Client_ProcessIndex_Args* args) -> PJRT_Error* {
        args->process_index = 0;
        return nullptr;
    };
    api.PJRT_Client_PlatformVersion = [](PJRT_Client_PlatformVersion_Args* args) -> PJRT_Error* {
        args->platform_version = kPlatformVersion;
        args->platform_version_size = std::strlen(kPlatformVersion);
        return nullptr;
    };
    api.PJRT_Client_Devices = [](PJRT_Client_Devices_Args* args) -> PJRT_Error* {
        args->devices = &args->client->device_ptr;
        args->num_devices = 1;
        return nullptr;
    };
    api.PJRT_Client_AddressableDevices =
        [](PJRT_Client_AddressableDevices_Args* args) -> PJRT_Error* {
        args->addressable_devices = &args->client->device_ptr;
        args->num_addressable_devices = 1;
        return nullptr;
    };
    api.PJRT_Client_LookupDevice = [](PJRT_Client_LookupDevice_Args* args) -> PJRT_Error* {
        if (args->id != 0) {
            return error(PJRT_Error_Code_INVALID_ARGUMENT,
                         "metal: no device with id " + std::to_string(args->id));
        }
        args->device = &args->client->device;
        return nullptr;
    };
    api.PJRT_Client_LookupAddressableDevice =
        [](PJRT_Client_LookupAddressableDevice_Args* args) -> PJRT_Error* {
        if (args->local_hardware_id != 0) {
            return error(PJRT_Error_Code_INVALID_ARGUMENT,
                         "metal: no device with local hardware id " +
                             std::to_string(args->local_hardware_id));
        }
        args->addressable_device = &args->client->device;
        return nullptr;
    };
    api.PJRT_Client_AddressableMemories =
        [](PJRT_Client_AddressableMemories_Args* args) -> PJRT_Error* {
        args->addressable_memories = &args->client->device.memory_ptr;
        args->num_addressable_memories = 1;
        return nullptr;
    };
    api.PJRT_Client_DefaultDeviceAssignment =
        [](PJRT_Client_DefaultDeviceAssignment_Args* args) -> PJRT_Error* {
        if (args->num_replicas * args->num_partitions != 1) {
            return error(PJRT_Error_Code_INVALID_ARGUMENT,
                         "metal: one device, so one replica and one partition");
        }
        args->default_assignment[0] = 0;
        return nullptr;
    };
    api.PJRT_Client_TopologyDescription =
        [](PJRT_Client_TopologyDescription_Args* args) -> PJRT_Error* {
        args->topology = &args->client->topology;
        return nullptr;
    };

    api.PJRT_Client_BufferFromHostBuffer =
        [](PJRT_Client_BufferFromHostBuffer_Args* args) -> PJRT_Error* {
        if (args->device_layout && !is_row_major(args->device_layout, args->num_dims)) {
            return error(PJRT_Error_Code_UNIMPLEMENTED,
                         "metal: device buffers are dense and row-major");
        }
        PJRT_Device* device = args->device       ? args->device
                              : args->memory     ? as_memory(args->memory)->device
                                                 : &args->client->device;
        PJRT_Buffer* buffer;
        if (PJRT_Error* e = make_buffer(device, args->type, args->dims, args->num_dims, &buffer)) {
            return e;
        }
        // Always a copy, so the host data is free once this returns.
        auto* dst = (char*)buffer->storage->contents();
        if (args->byte_strides && args->num_byte_strides) {
            copy_strided(dst, (const char*)args->data, args->dims, args->byte_strides,
                         args->num_dims, element_size(args->type));
        } else {
            std::memcpy(dst, args->data, byte_size(buffer));
        }
        args->done_with_host_buffer = new PJRT_Event{};
        args->buffer = buffer;
        return nullptr;
    };
    api.PJRT_Client_CreateUninitializedBuffer =
        [](PJRT_Client_CreateUninitializedBuffer_Args* args) -> PJRT_Error* {
        PJRT_Device* device = args->device ? args->device : &args->client->device;
        return make_buffer(device, args->shape_element_type, args->shape_dims,
                           args->shape_num_dims, &args->buffer);
    };

    api.PJRT_TopologyDescription_Destroy =
        [](PJRT_TopologyDescription_Destroy_Args*) -> PJRT_Error* { return nullptr; };
    api.PJRT_TopologyDescription_PlatformName =
        [](PJRT_TopologyDescription_PlatformName_Args* args) -> PJRT_Error* {
        args->platform_name = kPlatformName;
        args->platform_name_size = std::strlen(kPlatformName);
        return nullptr;
    };
    api.PJRT_TopologyDescription_PlatformVersion =
        [](PJRT_TopologyDescription_PlatformVersion_Args* args) -> PJRT_Error* {
        args->platform_version = kPlatformVersion;
        args->platform_version_size = std::strlen(kPlatformVersion);
        return nullptr;
    };
    api.PJRT_TopologyDescription_GetDeviceDescriptions =
        [](PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) -> PJRT_Error* {
        args->descriptions = &args->topology->description;
        args->num_descriptions = 1;
        return nullptr;
    };
    // Compilation cache keys hash it: the device kind and GPU family.
    api.PJRT_TopologyDescription_Fingerprint =
        [](PJRT_TopologyDescription_Fingerprint_Args* args) -> PJRT_Error* {
        const PJRT_DeviceDescription* d = args->topology->description;
        args->fingerprint = std::hash<std::string>{}(d->kind + "/" +
                                                     std::to_string(d->apple_gpu_family));
        return nullptr;
    };
    api.PJRT_TopologyDescription_Serialize =
        [](PJRT_TopologyDescription_Serialize_Args*) -> PJRT_Error* {
        return error(PJRT_Error_Code_UNIMPLEMENTED, "metal: topologies do not serialize");
    };
    api.PJRT_TopologyDescription_Attributes =
        [](PJRT_TopologyDescription_Attributes_Args* args) -> PJRT_Error* {
        args->attributes = nullptr;
        args->num_attributes = 0;
        return nullptr;
    };

    api.PJRT_DeviceDescription_Id = [](PJRT_DeviceDescription_Id_Args* args) -> PJRT_Error* {
        args->id = 0;
        return nullptr;
    };
    api.PJRT_DeviceDescription_ProcessIndex =
        [](PJRT_DeviceDescription_ProcessIndex_Args* args) -> PJRT_Error* {
        args->process_index = 0;
        return nullptr;
    };
    api.PJRT_DeviceDescription_Attributes =
        [](PJRT_DeviceDescription_Attributes_Args* args) -> PJRT_Error* {
        args->attributes = args->device_description->attributes;
        args->num_attributes = 1;
        return nullptr;
    };
    api.PJRT_DeviceDescription_Kind = [](PJRT_DeviceDescription_Kind_Args* args) -> PJRT_Error* {
        args->device_kind = args->device_description->kind.data();
        args->device_kind_size = args->device_description->kind.size();
        return nullptr;
    };
    api.PJRT_DeviceDescription_DebugString =
        [](PJRT_DeviceDescription_DebugString_Args* args) -> PJRT_Error* {
        args->debug_string = args->device_description->debug_string.data();
        args->debug_string_size = args->device_description->debug_string.size();
        return nullptr;
    };
    api.PJRT_DeviceDescription_ToString =
        [](PJRT_DeviceDescription_ToString_Args* args) -> PJRT_Error* {
        args->to_string = args->device_description->to_string.data();
        args->to_string_size = args->device_description->to_string.size();
        return nullptr;
    };

    api.PJRT_Device_GetDescription = [](PJRT_Device_GetDescription_Args* args) -> PJRT_Error* {
        args->device_description = &args->device->description;
        return nullptr;
    };
    api.PJRT_Device_IsAddressable = [](PJRT_Device_IsAddressable_Args* args) -> PJRT_Error* {
        args->is_addressable = true;
        return nullptr;
    };
    api.PJRT_Device_LocalHardwareId = [](PJRT_Device_LocalHardwareId_Args* args) -> PJRT_Error* {
        args->local_hardware_id = 0;
        return nullptr;
    };
    api.PJRT_Device_AddressableMemories =
        [](PJRT_Device_AddressableMemories_Args* args) -> PJRT_Error* {
        args->memories = &args->device->memory_ptr;
        args->num_memories = 1;
        return nullptr;
    };
    api.PJRT_Device_DefaultMemory = [](PJRT_Device_DefaultMemory_Args* args) -> PJRT_Error* {
        args->memory = &args->device->memory;
        return nullptr;
    };
    api.PJRT_Device_MemoryStats = [](PJRT_Device_MemoryStats_Args* args) -> PJRT_Error* {
        MTL::Device* device = runtime().device();
        // Clears the outputs and every *_is_set flag the caller did not.
        std::memset(&args->bytes_in_use, 0,
                    args->struct_size - offsetof(PJRT_Device_MemoryStats_Args, bytes_in_use));
        args->bytes_in_use = (int64_t)device->currentAllocatedSize();
        args->bytes_limit = (int64_t)device->recommendedMaxWorkingSetSize();
        args->bytes_limit_is_set = true;
        return nullptr;
    };

    api.PJRT_Memory_Id = [](PJRT_Memory_Id_Args* args) -> PJRT_Error* {
        args->id = 0;
        return nullptr;
    };
    api.PJRT_Memory_Kind = [](PJRT_Memory_Kind_Args* args) -> PJRT_Error* {
        args->kind = kMemoryKind;
        args->kind_size = std::strlen(kMemoryKind);
        return nullptr;
    };
    api.PJRT_Memory_Kind_Id = [](PJRT_Memory_Kind_Id_Args* args) -> PJRT_Error* {
        args->kind_id = 0;
        return nullptr;
    };
    api.PJRT_Memory_DebugString = [](PJRT_Memory_DebugString_Args* args) -> PJRT_Error* {
        args->debug_string = kMemoryKind;
        args->debug_string_size = std::strlen(kMemoryKind);
        return nullptr;
    };
    api.PJRT_Memory_ToString = [](PJRT_Memory_ToString_Args* args) -> PJRT_Error* {
        args->to_string = kMemoryKind;
        args->to_string_size = std::strlen(kMemoryKind);
        return nullptr;
    };
    api.PJRT_Memory_AddressableByDevices =
        [](PJRT_Memory_AddressableByDevices_Args* args) -> PJRT_Error* {
        args->devices = &as_memory(args->memory)->device;
        args->num_devices = 1;
        return nullptr;
    };

    api.PJRT_Buffer_Destroy = [](PJRT_Buffer_Destroy_Args* args) -> PJRT_Error* {
        delete args->buffer;
        return nullptr;
    };
    api.PJRT_Buffer_ElementType = [](PJRT_Buffer_ElementType_Args* args) -> PJRT_Error* {
        args->type = args->buffer->type;
        return nullptr;
    };
    api.PJRT_Buffer_Dimensions = [](PJRT_Buffer_Dimensions_Args* args) -> PJRT_Error* {
        args->dims = args->buffer->dims.data();
        args->num_dims = args->buffer->dims.size();
        return nullptr;
    };
    api.PJRT_Buffer_UnpaddedDimensions =
        [](PJRT_Buffer_UnpaddedDimensions_Args* args) -> PJRT_Error* {
        args->unpadded_dims = args->buffer->dims.data();
        args->num_dims = args->buffer->dims.size();
        return nullptr;
    };
    api.PJRT_Buffer_DynamicDimensionIndices =
        [](PJRT_Buffer_DynamicDimensionIndices_Args* args) -> PJRT_Error* {
        args->dynamic_dim_indices = nullptr;
        args->num_dynamic_dims = 0;
        return nullptr;
    };
    api.PJRT_Buffer_GetMemoryLayout = [](PJRT_Buffer_GetMemoryLayout_Args* args) -> PJRT_Error* {
        args->layout = {};
        args->layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
        args->layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
        args->layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
        args->layout.tiled.minor_to_major = args->buffer->minor_to_major.data();
        args->layout.tiled.minor_to_major_size = args->buffer->minor_to_major.size();
        return nullptr;
    };
    api.PJRT_Buffer_OnDeviceSizeInBytes =
        [](PJRT_Buffer_OnDeviceSizeInBytes_Args* args) -> PJRT_Error* {
        args->on_device_size_in_bytes = byte_size(args->buffer);
        return nullptr;
    };
    api.PJRT_Buffer_Device = [](PJRT_Buffer_Device_Args* args) -> PJRT_Error* {
        args->device = args->buffer->device;
        return nullptr;
    };
    api.PJRT_Buffer_Memory = [](PJRT_Buffer_Memory_Args* args) -> PJRT_Error* {
        args->memory = &args->buffer->device->memory;
        return nullptr;
    };
    api.PJRT_Buffer_Delete = [](PJRT_Buffer_Delete_Args* args) -> PJRT_Error* {
        args->buffer->storage.reset();
        return nullptr;
    };
    api.PJRT_Buffer_IsDeleted = [](PJRT_Buffer_IsDeleted_Args* args) -> PJRT_Error* {
        args->is_deleted = !args->buffer->storage;
        return nullptr;
    };
    api.PJRT_Buffer_IsOnCpu = [](PJRT_Buffer_IsOnCpu_Args* args) -> PJRT_Error* {
        args->is_on_cpu = false;
        return nullptr;
    };
    api.PJRT_Buffer_ReadyEvent = [](PJRT_Buffer_ReadyEvent_Args* args) -> PJRT_Error* {
        args->event = new PJRT_Event{};
        return nullptr;
    };
    api.PJRT_Buffer_UnsafePointer = [](PJRT_Buffer_UnsafePointer_Args* args) -> PJRT_Error* {
        if (PJRT_Error* e = deleted(args->buffer)) return e;
        args->buffer_pointer = (uintptr_t)args->buffer->storage->contents();
        return nullptr;
    };
    api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer =
        [](PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) -> PJRT_Error* {
        if (PJRT_Error* e = deleted(args->buffer)) return e;
        args->device_memory_ptr = args->buffer->storage->handle();
        return nullptr;
    };
    api.PJRT_Buffer_ToHostBuffer = [](PJRT_Buffer_ToHostBuffer_Args* args) -> PJRT_Error* {
        PJRT_Buffer* src = args->src;
        if (PJRT_Error* e = deleted(src)) return e;
        size_t size = byte_size(src);
        if (!args->dst) {
            args->dst_size = size;
            return nullptr;
        }
        if (!is_row_major(args->host_layout, src->dims.size())) {
            return error(PJRT_Error_Code_UNIMPLEMENTED,
                         "metal: copies to the host are dense and row-major");
        }
        if (args->dst_size < size) {
            return error(PJRT_Error_Code_INVALID_ARGUMENT,
                         "metal: the host buffer holds " + std::to_string(args->dst_size) +
                             " bytes, the device buffer " + std::to_string(size));
        }
        std::memcpy(args->dst, src->storage->contents(), size);
        args->event = new PJRT_Event{};
        return nullptr;
    };
    api.PJRT_Buffer_CopyToDevice = [](PJRT_Buffer_CopyToDevice_Args* args) -> PJRT_Error* {
        return copy_buffer(args->buffer, args->dst_device, &args->dst_buffer);
    };
    api.PJRT_Buffer_CopyToMemory = [](PJRT_Buffer_CopyToMemory_Args* args) -> PJRT_Error* {
        return copy_buffer(args->buffer, as_memory(args->dst_memory)->device, &args->dst_buffer);
    };
    add_executable_api(api);
    return api;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
    static const PJRT_Api api = make_api();
    return &api;
}
