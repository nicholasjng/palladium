// Exercise native readiness/errors with initialized storage and synthetic
// producer failures. No invalid GPU memory accesses or production fault hooks.
#include "plugin.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

extern "C" {
const PJRT_Api* GetPjrtApi();
void palladium_pjrt_set_compiler(int (*)(const char*, size_t, void*));
int pp_slot(void*, int, int, const int64_t*, size_t, const void*, size_t);
int pp_kernel(void*, const char*, const char*, int, const int*, size_t, const size_t*, const size_t*);
void pp_output(void*, int);
void pp_alias(void*, int, int);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(PJRT_Error* error) {
    if (!error) return;
    std::unique_ptr<Error> owned(static_cast<Error*>(error));
    throw std::runtime_error(owned->message);
}

void expect_failure(const std::shared_ptr<Completion>& completion) {
    completion->wait();
    std::unique_ptr<Error> failure(static_cast<Error*>(completion->error()));
    require(failure && failure->code == PJRT_Error_Code_DATA_LOSS, "missing dependency error code");
    require(failure->message == "injected producer failure", "missing original error message");
}

void test_join() {
    auto first = std::make_shared<Completion>(), second = std::make_shared<Completion>();
    auto joined = Completion::join({first, second, first, Completion::done()});
    std::atomic<int> callbacks = 0;
    joined->then([&] { ++callbacks; });
    second->set(PJRT_Error_Code_INTERNAL, "later dependency failed first");
    require(!joined->ready(), "join completed before all dependencies");
    first->set(PJRT_Error_Code_DATA_LOSS, "injected producer failure");
    expect_failure(joined);
    require(callbacks == 1, "completion callback count");
    expect_failure(Completion::join({first, second}));  // already completed errors
    for (int i = 0; i < 100; ++i) {
        auto a = std::make_shared<Completion>(), b = std::make_shared<Completion>();
        auto result = Completion::join({a, b, a});
        std::thread t1([a] { a->set(); }), t2([b] { b->set(); });
        result->wait();
        check(result->error());
        t1.join();
        t2.join();
    }
}

// c: compute; p: passthrough/blit; e: zero-sized work; d: donated output.
int compile(const char* code, size_t, void* p) {
    int64_t n = *code == 'e' ? 0 : 32;
    int input = pp_slot(p, 0, PJRT_Buffer_Type_F32, &n, 1, nullptr, 0);
    int output = input;
    if (*code != 'p') {
        output = pp_slot(p, 1, PJRT_Buffer_Type_F32, &n, 1, nullptr, 0);
        const char* source = R"(
#include <metal_stdlib>
using namespace metal;
kernel void add1(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) { y[i] = x[i] + 1; }
)";
        int slots[] = {input, output};
        size_t grid[] = {size_t(n), 1, 1}, group[] = {0, 0, 0};
        if (pp_kernel(p, source, "add1", 0, slots, 2, grid, group)) return 1;
    }
    pp_output(p, output);
    if (*code == 'd') pp_alias(p, 0, 0);
    return 0;
}

void test_execute(char kind, bool already_failed) {
    const auto* api = GetPjrtApi();
    PJRT_Client client;
    PJRT_Program program{};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.code = &kind;
    program.code_size = 1;
    program.format = "mlir";
    program.format_size = 4;
    PJRT_Client_Compile_Args compile_args{};
    compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = &client;
    compile_args.program = &program;
    compile_args.compile_options = "";
    check(api->PJRT_Client_Compile(&compile_args));
    int64_t n = kind == 'e' ? 0 : 32;
    PJRT_Buffer* input;
    check(make_buffer(&client.device, PJRT_Buffer_Type_F32, &n, 1, &input));
    std::memset(input->storage->contents(), 0, byte_size(input));
    auto producer = std::make_shared<Completion>();
    input->ready = producer;
    if (already_failed) producer->set(PJRT_Error_Code_DATA_LOSS, "injected producer failure");
    std::vector<std::unique_ptr<PJRT_Buffer>> buffers;
    std::vector<std::unique_ptr<PJRT_Event>> events;
    buffers.emplace_back(input);
    for (int i = 0; i < 3; ++i) {
        PJRT_Buffer* output = nullptr;
        PJRT_Buffer** inputs = &input;
        PJRT_Buffer** outputs = &output;
        PJRT_Event* event = nullptr;
        PJRT_LoadedExecutable_Execute_Args args{};
        args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        args.executable = compile_args.executable;
        args.num_devices = args.num_args = 1;
        args.argument_lists = &inputs;
        args.output_lists = &outputs;
        args.device_complete_events = &event;
        check(api->PJRT_LoadedExecutable_Execute(&args));
        if (!already_failed) require(!output->ready->ready(), "premature consumer success");
        if (kind == 'd') require(!input->storage, "input not donated");
        buffers.emplace_back(output);
        events.emplace_back(event);
        input = output;
    }
    // Release producer/intermediate handles and executable before resolving the
    // dependency: callbacks must own everything needed to complete the chain.
    auto last = std::move(buffers.back());
    buffers.clear();
    PJRT_LoadedExecutable_Destroy_Args destroy{};
    destroy.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    destroy.executable = compile_args.executable;
    check(api->PJRT_LoadedExecutable_Destroy(&destroy));
    if (!already_failed) producer->set(PJRT_Error_Code_DATA_LOSS, "injected producer failure");
    expect_failure(last->ready);
    check(deleted(last.get()));
    std::unique_ptr<Error> host_error(static_cast<Error*>(wait_ready(last.get())));
    require(host_error && host_error->code == PJRT_Error_Code_DATA_LOSS, "host read lost failure");
    for (const auto& event : events) expect_failure(event->completion);
}

int main() {
    try {
        test_join();
        palladium_pjrt_set_compiler(compile);
        for (char kind : {'c', 'p', 'e', 'd'}) {
            test_execute(kind, false);
            test_execute(kind, true);
        }
        std::cout << "PJRT dependency tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
