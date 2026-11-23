#include "runtime.h"
#include "gguf_loader.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace {
thread_local char error_text[512]{};
struct Failure { int status; };
[[noreturn]] void fail(int code, const char* message) {
    std::snprintf(error_text, sizeof(error_text), "%s", message);
    throw Failure{code};
}
void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) fail(ML_CUDA, cudaGetErrorString(result));
}
struct DeviceGuard {
    int previous = -1;
    explicit DeviceGuard(int device) {
        cuda_check(cudaGetDevice(&previous));
        cuda_check(cudaSetDevice(device));
    }
    ~DeviceGuard() noexcept { if (previous >= 0) (void)cudaSetDevice(previous); }
};
struct Engine {
    int device;
    uint64_t budget, max_sessions, sessions=0, peak=0;
    cudaStream_t stream=nullptr;
    Engine(int d, uint64_t b, uint64_t m): device(d), budget(b), max_sessions(m) {
        DeviceGuard guard(device);
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Engine() noexcept {
        int previous=-1;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        if (stream) { (void)cudaStreamSynchronize(stream); (void)cudaStreamDestroy(stream); }
        if (previous >= 0) (void)cudaSetDevice(previous);
    }
};
struct Session {
    std::shared_ptr<Engine> engine;
    uint64_t capacity, position=0;
    int32_t* device=nullptr;
    int32_t* host=nullptr;
    bool counted=false;
    Session(std::shared_ptr<Engine> e, uint64_t c): engine(std::move(e)), capacity(c) {}
    ~Session() noexcept {
        int previous=-1;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(engine->device);
        (void)cudaStreamSynchronize(engine->stream);
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (previous >= 0) (void)cudaSetDevice(previous);
        if (counted) --engine->sessions;
    }
};
// Declared in this order so sessions are destroyed before engine registry at exit.
std::mutex mutex;
std::unordered_map<ml_handle, std::shared_ptr<Engine>> engines;
std::unordered_map<ml_handle, std::shared_ptr<Session>> sessions;
std::unordered_map<ml_handle, std::shared_ptr<ml::GGUF>> ggufs;
uint64_t next_handle=1;
uint64_t new_handle() {
    if (next_handle == std::numeric_limits<uint64_t>::max()) fail(ML_CAPACITY, "handle space exhausted");
    return next_handle++;
}
template<class Map> auto lookup(Map& map, ml_handle handle) {
    auto it=map.find(handle);
    if (it==map.end()) fail(ML_INVALID, "invalid or closed handle");
    return it->second;
}
template<class F> int boundary(F&& fn) noexcept {
    error_text[0]='\0';
    try { std::lock_guard lock(mutex); fn(); return ML_OK; }
    catch (const Failure& f) { return f.status; }
        catch (const ml::GGUFError& e) { std::snprintf(error_text, sizeof(error_text), "%s", e.what()); return ML_INVALID; }
    catch (const std::exception& e) { std::snprintf(error_text, sizeof(error_text), "%s", e.what()); return ML_INTERNAL; }
    catch (...) { std::snprintf(error_text, sizeof(error_text), "unknown native exception"); return ML_INTERNAL; }
}
}
extern "C" {
const char* ml_last_error() noexcept { return error_text; }
int ml_gguf_validate_schema(ml_handle id, const ml_tensor_requirement* schema, uint64_t count) noexcept {
    return boundary([&] {
        auto store=lookup(ggufs,id);
        if(!schema || !count) fail(ML_INVALID,"empty/null required tensor schema");
        std::unordered_map<std::string,const ml_gguf_tensor*> tensors;
        for(const auto& t:store->tensors) tensors.emplace(t.name,&t.desc);
        std::unordered_set<std::string> seen;
        for(uint64_t i=0;i<count;++i) {
            const auto& r=schema[i];
            if(!r.name || !*r.name || r.rank<1 || r.rank>4) fail(ML_INVALID,"invalid tensor requirement");
            if(!seen.emplace(r.name).second) fail(ML_INVALID,"duplicate tensor requirement");
            auto it=tensors.find(r.name);
            if(it==tensors.end()) throw ml::GGUFError(std::string(r.name)+": missing required tensor");
            const auto& d=*it->second;
            if(d.rank!=r.rank || !std::equal(r.ggml_shape,r.ggml_shape+r.rank,d.ggml_shape))
                throw ml::GGUFError(std::string(r.name)+": required shape mismatch");
            if(d.type_id!=r.type_id) throw ml::GGUFError(std::string(r.name)+": required type mismatch");
        }
        if(seen.size()!=tensors.size()) fail(ML_INVALID,"unsupported tensors outside required schema");
    });
}
int ml_gguf_open(const char* const* paths, uint64_t count, ml_handle* out) noexcept {
    return boundary([&] {
        if(!out) fail(ML_INVALID,"null output");
        *out=0;
        auto store=std::make_shared<ml::GGUF>(paths,count);
        auto id=new_handle(); ggufs.emplace(id,std::move(store)); *out=id;
    });
}
int ml_gguf_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(ggufs,id); ggufs.erase(id); } });
}
int ml_gguf_counts(ml_handle id,uint64_t* nt,uint64_t* nm) noexcept {
    return boundary([&] { if(!nt || !nm) fail(ML_INVALID,"null output"); auto s=lookup(ggufs,id); *nt=s->tensors.size(); *nm=s->metadata.size(); });
}
int ml_gguf_tensor_get(ml_handle id,uint64_t index,const char** name,ml_gguf_tensor* out) noexcept {
    return boundary([&] { auto s=lookup(ggufs,id); if(!name || !out || index>=s->tensors.size()) fail(ML_INVALID,"invalid tensor output/index"); *name=s->tensors[index].name.c_str(); *out=s->tensors[index].desc; });
}
int ml_gguf_metadata_get(ml_handle id,uint64_t index,const char** key,uint32_t* type,const void** value,uint64_t* size,uint64_t* shard) noexcept {
    return boundary([&] {
        auto s=lookup(ggufs,id);
        if(!key || !type || !value || !size || !shard || index>=s->metadata.size()) fail(ML_INVALID,"invalid metadata output/index");
        const auto& m=s->metadata[index]; *key=m.key.c_str(); *type=m.type; *value=m.value.data(); *size=m.value.size(); *shard=m.shard;
    });
}
int ml_engine_create(int device, uint64_t budget, uint64_t maximum, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID, "null output");
        *out=0;
        if (device<0 || budget<12 || maximum==0) fail(ML_INVALID, "invalid engine configuration");
        auto engine=std::make_shared<Engine>(device, budget, maximum);
        auto id=new_handle(); engines.emplace(id, std::move(engine)); *out=id;
    });
}
int ml_engine_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(engines,id); engines.erase(id); } });
}
int ml_engine_stats(ml_handle id, ml_stats* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID, "null output");
        auto e=lookup(engines,id);
        *out={e->sessions*4,e->sessions*8,e->peak,e->sessions,e->budget};
    });
}
int ml_session_create(ml_handle id, uint64_t capacity, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID, "null output");
        *out=0;
        auto e=lookup(engines,id);
        if (!capacity) fail(ML_INVALID, "capacity must be positive");
        if (e->sessions>=e->max_sessions || e->sessions>=e->budget/12) fail(ML_CAPACITY, "session allocation budget exhausted");
        auto s=std::make_shared<Session>(e,capacity);
        DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&s->device),4));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&s->host),8));
        auto handle=new_handle(); sessions.emplace(handle,s);
        ++e->sessions; s->counted=true; e->peak=std::max(e->peak,e->sessions*12); *out=handle;
    });
}
int ml_session_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(sessions,id); sessions.erase(id); } });
}
int ml_session_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto s=lookup(sessions,id); DeviceGuard guard(s->engine->device);
        cuda_check(cudaStreamSynchronize(s->engine->stream)); s->position=0;
    });
}
int ml_session_step(ml_handle id, int32_t token, int32_t* out, uint64_t* position) noexcept {
    return boundary([&] {
        if (!out || !position || token<0) fail(ML_INVALID, "invalid step arguments");
        auto s=lookup(sessions,id);
        if (s->position>=s->capacity) fail(ML_CAPACITY, "token capacity exhausted; reset session");
        DeviceGuard guard(s->engine->device);
        s->host[0]=token;
        cuda_check(cudaMemcpyAsync(s->device,s->host,4,cudaMemcpyHostToDevice,s->engine->stream));
        cuda_check(cudaMemcpyAsync(s->host+1,s->device,4,cudaMemcpyDeviceToHost,s->engine->stream));
        cuda_check(cudaStreamSynchronize(s->engine->stream));
        *out=s->host[1]; *position=s->position++;
    });
}
}
