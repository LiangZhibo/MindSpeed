// Thin C ABI over the frozen AscendC release. No tensor operators or ACL context
// initialization: the owner supplies all buffers and the existing NPU stream.
#include "vendor/splitzip_v2.h"
#include <cstring>
#include <memory>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

using namespace splitzip_v2;

namespace {
struct PreparedContext {
    Plan plan;
    Codebook book;
    Workspace workspace;
    Buffer header;
};
struct PreparedPacket {
    Plan plan;
    PacketInfo info;
};

struct AsyncRequest {
    Plan plan;
    PacketInfo info;
    Buffer packet, host, header;
    aclrtStream encoder = nullptr, dma = nullptr;
    aclrtContext context = nullptr;
    aclrtEvent header_ready = nullptr, payload_ready = nullptr;
    std::mutex mutex;
    std::condition_variable changed;
    bool submitted = false;
    Status status = Status::ok;
    aclError error = ACL_SUCCESS;

    void finish(Status value, aclError runtime = ACL_SUCCESS) {
        { std::lock_guard<std::mutex> lock(mutex); status = value; error = runtime; submitted = true; }
        changed.notify_all();
    }
    Status wait_submitted() {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return submitted; });
        return status;
    }
};
using AsyncHandle = std::shared_ptr<AsyncRequest>;

struct ContextScope {
    aclrtContext previous = nullptr;
    bool changed = false;
    aclError status = ACL_SUCCESS;
    explicit ContextScope(aclrtContext target) {
        aclrtGetCurrentContext(&previous);
        changed = previous != target;
        if (changed) status = aclrtSetCurrentContext(target);
    }
    ~ContextScope() { if (changed && previous) aclrtSetCurrentContext(previous); }
};

struct AsyncEngine {
    aclrtStream dma;
    aclrtContext context;
    std::mutex mutex;
    std::condition_variable changed, idle;
    std::deque<AsyncHandle> queue;
    bool stopping = false, active = false;
    Status failure = Status::ok;
    std::thread worker;

    AsyncEngine(aclrtStream stream, aclrtContext owner) : dma(stream), context(owner), worker([this] { run(); }) {}
    ~AsyncEngine() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        changed.notify_all();
        worker.join();
    }
    void enqueue(const AsyncHandle& request) {
        { std::lock_guard<std::mutex> lock(mutex); queue.push_back(request); }
        changed.notify_one();
    }
    Status flush() {
        std::unique_lock<std::mutex> lock(mutex);
        idle.wait(lock, [&] { return queue.empty() && !active; });
        return failure;
    }
    void process(const AsyncHandle& request, aclError context_status) {
        auto error = context_status;
        if (!error) error = aclrtSynchronizeEvent(request->header_ready);
        Status status = error ? Status::runtime_error : parse_header(
            request->plan, request->header.data, kHeaderBytes, request->packet.bytes, &request->info);
        if (status == Status::ok) error = aclrtStreamWaitEvent(dma, request->header_ready);
        if (status == Status::ok && !error) error = aclrtMemcpyAsync(
            request->host.data, request->host.bytes, request->packet.data,
            request->info.stored_bytes(), ACL_MEMCPY_DEVICE_TO_HOST, dma);
        if (status == Status::ok && !error) error = aclrtRecordEvent(request->payload_ready, dma);
        if (error) status = Status::runtime_error;
        request->finish(status, error);
        if (status != Status::ok) { std::lock_guard<std::mutex> lock(mutex); failure = status; }
    }
    void run() {
        const auto context_status = aclrtSetCurrentContext(context);
        for (;;) {
            AsyncHandle request;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty() && stopping) return;
                request = queue.front(); queue.pop_front(); active = true;
            }
            try { process(request, context_status); }
            catch (...) {
                request->finish(Status::runtime_error);
                std::lock_guard<std::mutex> lock(mutex); failure = Status::runtime_error;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                active = false;
            }
            idle.notify_all();
        }
    }
};
}

struct SzPackedResult {
    void* packet = nullptr;
    const void* header = nullptr;
    uint32_t stored = 0;
    uint32_t raw = 0;
};

extern "C" {
int sz_async_engine_create(void* dma_stream, void** result) {
    if (!dma_stream || !result) return static_cast<int>(Status::invalid_argument);
    *result = nullptr;
    aclrtContext context = nullptr;
    if (aclrtGetCurrentContext(&context) || !context) return static_cast<int>(Status::runtime_error);
    try { *result = new AsyncEngine(dma_stream, context); }
    catch (...) { return static_cast<int>(Status::runtime_error); }
    return static_cast<int>(Status::ok);
}
void sz_async_engine_destroy(void* pointer) { delete static_cast<AsyncEngine*>(pointer); }
int sz_async_engine_flush(void* pointer) {
    if (!pointer) return static_cast<int>(Status::invalid_argument);
    return static_cast<int>(static_cast<AsyncEngine*>(pointer)->flush());
}
int sz_async_engine_sync(void* pointer) {
    if (!pointer) return static_cast<int>(Status::invalid_argument);
    auto& engine = *static_cast<AsyncEngine*>(pointer);
    const auto status = engine.flush();
    ContextScope scope(engine.context);
    if (scope.status || aclrtSynchronizeStream(engine.dma)) return static_cast<int>(Status::runtime_error);
    return static_cast<int>(status);
}

int sz_async_submit(void* engine_pointer, void* context_pointer, void* input, size_t input_bytes,
                    void* packet, size_t capacity, void* header, void* host, size_t host_bytes,
                    void* encoder, void** result) {
    if (!engine_pointer || !context_pointer || !result) return static_cast<int>(Status::invalid_argument);
    *result = nullptr;
    try {
        auto& engine = *static_cast<AsyncEngine*>(engine_pointer);
        auto& context = *static_cast<PreparedContext*>(context_pointer);
        if (!input || !packet || !header || !host || !encoder || host_bytes < context.plan.packet_capacity)
            return static_cast<int>(Status::invalid_argument);
        if (reinterpret_cast<uintptr_t>(host)%32 || reinterpret_cast<uintptr_t>(header)%32)
            return static_cast<int>(Status::misaligned);
        auto request = std::make_shared<AsyncRequest>();
        request->plan = context.plan;
        request->packet = {packet, capacity}; request->host = {host, host_bytes};
        request->header = {header, kHeaderBytes};
        request->encoder = encoder; request->dma = engine.dma; request->context = engine.context;
        *result = new AsyncHandle(request);
        auto error = aclrtCreateEvent(&request->header_ready);
        if (!error) error = aclrtCreateEvent(&request->payload_ready);
        Status status = error ? Status::runtime_error : encode_async(
            context.plan, context.book, {input, input_bytes}, {packet, capacity}, context.workspace, encoder);
        if (status == Status::ok) error = aclrtMemcpyAsync(
            header, kHeaderBytes, packet, kHeaderBytes, ACL_MEMCPY_DEVICE_TO_HOST, encoder);
        if (status == Status::ok && !error) error = aclrtRecordEvent(request->header_ready, encoder);
        if (error) status = Status::runtime_error;
        if (status == Status::ok) engine.enqueue(request);
        else request->finish(status, error);
        return static_cast<int>(status);
    } catch (...) {
        if (*result) (*static_cast<AsyncHandle*>(*result))->finish(Status::runtime_error);
        return static_cast<int>(Status::runtime_error);
    }
}

int sz_async_wait_stream(void* pointer, void* stream) {
    if (!pointer || !stream) return static_cast<int>(Status::invalid_argument);
    auto request = *static_cast<AsyncHandle*>(pointer);
    auto status = request->wait_submitted();
    if (status != Status::ok) return static_cast<int>(status);
    ContextScope scope(request->context);
    if (scope.status || aclrtStreamWaitEvent(stream, request->payload_ready)) return static_cast<int>(Status::runtime_error);
    return static_cast<int>(Status::ok);
}
int sz_async_query(void* pointer, uint32_t* ready) {
    if (!pointer || !ready) return static_cast<int>(Status::invalid_argument);
    auto request = *static_cast<AsyncHandle*>(pointer);
    *ready = 0;
    {
        std::lock_guard<std::mutex> lock(request->mutex);
        if (!request->submitted) return static_cast<int>(Status::ok);
        if (request->status != Status::ok) return static_cast<int>(request->status);
    }
    ContextScope scope(request->context);
    aclrtEventRecordedStatus status;
    if (scope.status || aclrtQueryEventStatus(request->payload_ready, &status)) return static_cast<int>(Status::runtime_error);
    *ready = status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return static_cast<int>(Status::ok);
}
int sz_async_stats(void* pointer, uint32_t* stored, uint32_t* raw) {
    if (!pointer || !stored || !raw) return static_cast<int>(Status::invalid_argument);
    auto request = *static_cast<AsyncHandle*>(pointer);
    auto status = request->wait_submitted();
    if (status != Status::ok) return static_cast<int>(status);
    *stored = request->info.stored_bytes(); *raw = request->info.raw();
    return static_cast<int>(Status::ok);
}
int sz_async_sync(void* pointer) {
    if (!pointer) return static_cast<int>(Status::invalid_argument);
    auto request = *static_cast<AsyncHandle*>(pointer);
    auto status = request->wait_submitted();
    if (status != Status::ok) return static_cast<int>(status);
    ContextScope scope(request->context);
    if (scope.status || aclrtSynchronizeEvent(request->payload_ready)) return static_cast<int>(Status::runtime_error);
    return static_cast<int>(Status::ok);
}
int sz_async_decode(void* pointer, void* packet, size_t bytes, void* output, size_t output_bytes, void* stream) {
    if (!pointer) return static_cast<int>(Status::invalid_argument);
    auto request = *static_cast<AsyncHandle*>(pointer);
    auto status = request->wait_submitted();
    if (status != Status::ok) return static_cast<int>(status);
    return static_cast<int>(decode_async(request->plan, request->info, {packet, bytes}, {output, output_bytes}, stream));
}
int sz_async_destroy(void* pointer) {
    if (!pointer) return static_cast<int>(Status::ok);
    auto request = *static_cast<AsyncHandle*>(pointer);
    auto status = request->wait_submitted();
    ContextScope scope(request->context);
    if (scope.status) return static_cast<int>(Status::runtime_error);
    aclError error;
    if (status == Status::ok) error = aclrtSynchronizeEvent(request->payload_ready);
    else {
        error = aclrtSynchronizeStream(request->encoder);
        if (!error) error = aclrtSynchronizeStream(request->dma);
    }
    if (error) return static_cast<int>(Status::runtime_error);
    if (request->header_ready) aclrtDestroyEvent(request->header_ready);
    if (request->payload_ready) aclrtDestroyEvent(request->payload_ready);
    delete static_cast<AsyncHandle*>(pointer);
    return static_cast<int>(Status::ok);
}

// The opaque objects contain CPU metadata only. Device buffers and the stream
// remain caller-owned. No global ACL initialization or allocator is introduced.
int sz_context_create(uint64_t elements, const void* host_table, uint32_t base,
                      void* sparse, void* counts, void* table, void* header, void** result) {
    if (!result || !host_table) return static_cast<int>(Status::invalid_argument);
    *result = nullptr;
    try {
        auto context = std::make_unique<PreparedContext>();
        auto status = make_plan(elements, &context->plan);
        if (status != Status::ok) return static_cast<int>(status);
        std::memcpy(context->book.lut.data(), host_table, sizeof(context->book.lut));
        context->book.code_base = base;
        status = validate_codebook(context->book);
        if (status != Status::ok) return static_cast<int>(status);
        context->workspace = {{sparse, context->plan.sparse_bytes},
                              {counts, context->plan.count_bytes}, {table, context->plan.table_bytes}};
        context->header = {header, kHeaderBytes};
        *result = context.release();
        return static_cast<int>(Status::ok);
    } catch (...) { return static_cast<int>(Status::runtime_error); }
}

void sz_context_destroy(void* pointer) { delete static_cast<PreparedContext*>(pointer); }
void sz_packet_destroy(void* pointer) { delete static_cast<PreparedPacket*>(pointer); }

int sz_encode_prepared(void* pointer, void* input, size_t input_bytes,
                       void* packet, size_t capacity, void* stream, SzPackedResult* result) {
    if (!pointer || !result) return static_cast<int>(Status::invalid_argument);
    *result = SzPackedResult{};
    auto& context = *static_cast<PreparedContext*>(pointer);
    try {
        auto output = std::make_unique<PreparedPacket>();
        output->plan = context.plan;
        auto status = encode_async(context.plan, context.book, {input, input_bytes},
                                   {packet, capacity}, context.workspace, stream);
        if (status != Status::ok) return static_cast<int>(status);
        status = read_packet_info(context.plan, {packet, capacity}, context.header, stream, &output->info);
        if (status != Status::ok) return static_cast<int>(status);
        result->stored = output->info.stored_bytes();
        result->raw = output->info.raw();
        result->header = output->info.header().data();
        result->packet = output.release();
        return static_cast<int>(Status::ok);
    } catch (...) { return static_cast<int>(Status::runtime_error); }
}

int sz_decode_prepared(void* pointer, void* packet, size_t packet_bytes,
                       void* output, size_t output_bytes, void* stream) {
    if (!pointer) return static_cast<int>(Status::invalid_argument);
    auto& state = *static_cast<PreparedPacket*>(pointer);
    return static_cast<int>(decode_async(state.plan, state.info, {packet, packet_bytes},
                                        {output, output_bytes}, stream));
}

const char* sz_status(int code) { return status_name(static_cast<Status>(code)); }

int sz_book(const void* sample, size_t elements, void* table, uint32_t* base) {
    Codebook book;
    auto status = make_codebook(static_cast<const uint16_t*>(sample), elements, &book);
    if (status == Status::ok) {
        std::memcpy(table, book.lut.data(), sizeof(book.lut));
        *base = book.code_base;
    }
    return static_cast<int>(status);
}

int sz_encode(uint64_t elements, const void* host_table, uint32_t base,
              void* input, size_t input_bytes, void* packet, size_t packet_bytes,
              void* sparse, void* counts, void* table, void* stream) {
    Plan plan;
    auto status = make_plan(elements, &plan);
    if (status != Status::ok) return static_cast<int>(status);
    Codebook book;
    std::memcpy(book.lut.data(), host_table, sizeof(book.lut));
    book.code_base = base;
    Workspace workspace{{sparse, plan.sparse_bytes}, {counts, plan.count_bytes}, {table, plan.table_bytes}};
    return static_cast<int>(encode_async(plan, book, {input, input_bytes},
                                       {packet, packet_bytes}, workspace, stream));
}

int sz_info(uint64_t elements, void* packet, size_t capacity, void* header,
            void* stream, uint32_t* stored, uint32_t* raw) {
    Plan plan;
    auto status = make_plan(elements, &plan);
    if (status != Status::ok) return static_cast<int>(status);
    PacketInfo info;
    status = read_packet_info(plan, {packet, capacity}, {header, kHeaderBytes}, stream, &info);
    if (status == Status::ok) { *stored = info.stored_bytes(); *raw = info.raw(); }
    return static_cast<int>(status);
}

int sz_decode(uint64_t elements, const void* header, void* packet, size_t packet_bytes,
              void* output, size_t output_bytes, void* stream) {
    Plan plan;
    auto status = make_plan(elements, &plan);
    if (status != Status::ok) return static_cast<int>(status);
    PacketInfo info;
    status = parse_header(plan, header, kHeaderBytes, packet_bytes, &info);
    if (status != Status::ok) return static_cast<int>(status);
    return static_cast<int>(decode_async(plan, info, {packet, packet_bytes},
                                       {output, output_bytes}, stream));
}
}
