#include "runtime.h"
#include "gguf_loader.h"
#include "qsa_selection.h"
#include "qsa_attention.h"
#include "qsa_projection.h"
#include "qsa_frontend.h"
#include "hyperconnection.h"
#include "recurrent.h"
#include "ple.h"
#include "ple_hash.h"
#include "moe.h"
#include "../ops/ggml_packed_cpu.h"
#include <dlfcn.h>
#include <array>
#include <climits>
#include <cmath>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

cudaError_t ml_qsa_layer_finite(cudaStream_t, const float*, uint64_t, int*);

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
    uint64_t budget, max_sessions, sessions=0, peak=0, device_bytes=0, pinned_bytes=0;
        void admit(uint64_t bytes) const {
            // Counters are bounded by budget; subtraction avoids sum/size overflow.
            if (bytes>budget-device_bytes-pinned_bytes) fail(ML_CAPACITY,"engine allocation budget exhausted");
        }
        void record(uint64_t device, uint64_t pinned) {
            device_bytes+=device; pinned_bytes+=pinned;
            peak=std::max(peak,device_bytes+pinned_bytes);
        }
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
        if (counted) { --engine->sessions; engine->device_bytes-=4; engine->pinned_bytes-=8; }
    }
};
// Declared in this order so sessions are destroyed before engine registry at exit.
std::mutex mutex;
std::unordered_map<ml_handle, std::shared_ptr<Engine>> engines;
std::unordered_map<ml_handle, std::shared_ptr<Session>> sessions;
std::unordered_map<ml_handle, std::shared_ptr<ml::GGUF>> ggufs;
struct PayloadLease {
    std::shared_ptr<ml::GGUF> store;
    size_t tensor_index;
    const ml_gguf_tensor& descriptor() const { return store->tensors[tensor_index].desc; }
};
std::unordered_map<ml_handle, std::shared_ptr<PayloadLease>> payload_leases;
struct UploadedTensor {
    std::shared_ptr<Engine> engine;
    std::shared_ptr<PayloadLease> lease;
    ml_gguf_tensor descriptor;
    void* device=nullptr; // Internal native operator access only; never exposed by ABI.
    cudaEvent_t event=nullptr;
    bool counted=false, retired=false;
    UploadedTensor(std::shared_ptr<Engine> e, std::shared_ptr<PayloadLease> l)
        : engine(std::move(e)), lease(std::move(l)), descriptor(lease->descriptor()) {}
    void retire() { retired=true; lease.reset(); }
    void wait() {
        if (!retired) { cuda_check(cudaEventSynchronize(event)); retire(); }
    }
    ~UploadedTensor() noexcept {
        int previous=-1;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(engine->device);
        // Also covers enqueue success followed by event-record/registry failure.
        // Never release the mmap on a failed event wait; drain before freeing.
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (event) (void)cudaEventDestroy(event);
        lease.reset();
        if (counted) engine->device_bytes-=descriptor.nbytes;
        if (previous>=0) (void)cudaSetDevice(previous);
    }
};
// Last registry constructed: uploads drain before leases, stores and engines at exit.
std::unordered_map<ml_handle, std::shared_ptr<UploadedTensor>> uploads;
struct QSAWorkspace {
    std::shared_ptr<Engine> engine;
    uint64_t capacity, n=0, bytes;
    void* device=nullptr;
    void* host=nullptr;
    uint64_t* parameter=nullptr;
    int64_t* ids=nullptr;
    float* scores=nullptr;
    int* status=nullptr;
    cudaEvent_t start=nullptr, end=nullptr;
    bool counted=false, valid=false;
    QSAWorkspace(std::shared_ptr<Engine> e, uint64_t c): engine(std::move(e)), capacity(c),
        bytes(8+8*std::min(c,uint64_t(2051))+4*c+4) {}
    ~QSAWorkspace() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (start) (void)cudaEventDestroy(start);
        if (end) (void)cudaEventDestroy(end);
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=12; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void set_n(uint64_t value) {
        // Ensure the pinned scalar is not overwritten while a previous copy uses it.
        cuda_check(cudaStreamSynchronize(engine->stream));
        valid=false; n=0;
        *static_cast<uint64_t*>(host)=value;
        cuda_check(cudaMemcpyAsync(parameter,host,8,cudaMemcpyHostToDevice,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream)); n=value;
    }
};
std::unordered_map<ml_handle, std::shared_ptr<QSAWorkspace>> qsa_workspaces;
struct QSAAttention {
    std::shared_ptr<QSAWorkspace> selection;
    void* device=nullptr;
    int* host=nullptr;
    float* contexts=nullptr;
    float* gated=nullptr;
    int* status=nullptr;
    cudaEvent_t start=nullptr, end=nullptr;
    bool counted=false, valid=false;
    explicit QSAAttention(std::shared_ptr<QSAWorkspace> s): selection(std::move(s)) {}
    ~QSAAttention() noexcept {
        auto e=selection->engine;
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(e->device);
        if (cudaStreamSynchronize(e->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (start) (void)cudaEventDestroy(start);
        if (end) (void)cudaEventDestroy(end);
        if (counted) { e->device_bytes-=49156; e->pinned_bytes-=8; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
};
std::unordered_map<ml_handle,std::shared_ptr<QSAAttention>> qsa_attentions;
struct QSAProjection {
    std::shared_ptr<QSAAttention> attention;
    std::shared_ptr<UploadedTensor> weight;
    void* device=nullptr;
    int* host=nullptr;
    cudaEvent_t start=nullptr, end=nullptr;
    uint64_t rows, bytes;
    bool counted=false, valid=false;
    QSAProjection(std::shared_ptr<QSAAttention> a, std::shared_ptr<UploadedTensor> w)
        : attention(std::move(a)), weight(std::move(w)),
          rows(weight->descriptor.ggml_shape[1]), bytes(rows*4+4) {}
    ~QSAProjection() noexcept {
        auto e=weight->engine;
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(e->device);
        if (cudaStreamSynchronize(e->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (start) (void)cudaEventDestroy(start);
        if (end) (void)cudaEventDestroy(end);
        if (counted) { e->device_bytes-=bytes; e->pinned_bytes-=4; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
};
std::unordered_map<ml_handle,std::shared_ptr<QSAProjection>> qsa_projections;
struct QSAFrontend {
    static constexpr uint64_t bytes=54288, pinned=16, hidden_bytes=55808;
    void* hidden_device=nullptr;
    float* projections[5]{}; // qg, k, v, raw, iq
    bool hidden_valid=false;
    std::shared_ptr<Engine> engine;
    void* device=nullptr;
    void* host=nullptr;
    uint64_t* parameter=nullptr;
    float* outputs[6]{};
    int* ready=nullptr;
    int* status=nullptr;
    uint64_t n=0;
    bool counted=false, valid=false;
    explicit QSAFrontend(std::shared_ptr<Engine> e): engine(std::move(e)) {}
    ~QSAFrontend() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (hidden_device) { (void)cudaFree(hidden_device); engine->device_bytes-=hidden_bytes; }
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=pinned; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void set_n(uint64_t value) {
        valid=false; hidden_valid=false; n=0;
        cuda_check(cudaStreamSynchronize(engine->stream));
        *static_cast<uint64_t*>(host)=value;
        cuda_check(cudaMemcpyAsync(parameter,host,8,cudaMemcpyHostToDevice,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream)); n=value;
    }
};
std::unordered_map<ml_handle,std::shared_ptr<QSAFrontend>> qsa_frontends;
// Must be declared after input owners in each run: drain before their release
// even if enqueue, event recording, or status transfer throws.
struct StreamCompletion {
    cudaStream_t stream;
    ~StreamCompletion() noexcept {
        if (cudaStreamSynchronize(stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
    }
};
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
#include "qsa_layer_runtime.inc"
#include "hyperconnection_runtime.inc"
#include "recurrent_runtime.inc"
#include "ple_runtime.inc"
#include "cpu_expert_runtime.inc"
#include "moe_runtime.inc"
#include "chaining_runtime.inc"
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
int ml_gguf_lease_create(ml_handle id, const char* name, ml_handle* out) noexcept {
    return boundary([&] {
        if(!out) fail(ML_INVALID,"null lease output");
        *out=0;
        auto store=lookup(ggufs,id);
        if(!name) fail(ML_INVALID,"null tensor name");
        auto it=std::find_if(store->tensors.begin(),store->tensors.end(),
                             [&](const auto& t) { return t.name==name; });
        if(it==store->tensors.end()) fail(ML_INVALID,"unknown tensor name");
        auto lease=std::make_shared<PayloadLease>(PayloadLease{
            store,static_cast<size_t>(it-store->tensors.begin())});
        auto handle=new_handle(); payload_leases.emplace(handle,std::move(lease)); *out=handle;
    });
}
int ml_payload_lease_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(payload_leases,id); payload_leases.erase(id); } });
}
int ml_payload_lease_descriptor(ml_handle id, ml_gguf_tensor* out) noexcept {
    return boundary([&] {
        if(!out) fail(ML_INVALID,"null descriptor output");
        auto lease=lookup(payload_leases,id); *out=lease->descriptor();
    });
}
int ml_payload_lease_copy(ml_handle id, uint64_t offset, uint64_t size,
                         void* destination, uint64_t capacity, uint64_t* copied) noexcept {
    return boundary([&] {
        if(!copied) fail(ML_INVALID,"null copied output");
        *copied=0;
        auto lease=lookup(payload_leases,id);
        const auto& d=lease->descriptor();
        if(offset>d.nbytes || size>d.nbytes-offset) fail(ML_INVALID,"payload range out of bounds");
        if(size>capacity || size>SIZE_MAX || (size && !destination))
            fail(ML_INVALID,"invalid destination/capacity");
        if(size) std::memcpy(destination,lease->store->maps[d.shard_index]->data+d.data_offset+offset,
                             static_cast<size_t>(size));
        *copied=size;
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
        *out={e->device_bytes,e->pinned_bytes,e->peak,e->sessions,e->budget};
    });
}
int ml_engine_upload(ml_handle id, ml_handle lease_id, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null upload output");
        *out=0;
        auto e=lookup(engines,id);
        auto lease=lookup(payload_leases,lease_id);
        const auto& d=lease->descriptor();
        if (d.nbytes>SIZE_MAX) fail(ML_CAPACITY,"upload exceeds host address space");
        e->admit(d.nbytes);
        auto u=std::make_shared<UploadedTensor>(e,lease);
        DeviceGuard guard(e->device);
        cuda_check(cudaEventCreateWithFlags(&u->event,cudaEventDisableTiming));
        cuda_check(cudaMalloc(&u->device,static_cast<size_t>(d.nbytes)));
        // Account actual requested live storage even if a later enqueue fails.
        e->record(d.nbytes,0); u->counted=true;
        const auto* source=lease->store->maps[d.shard_index]->data+d.data_offset;
        cuda_check(cudaMemcpyAsync(u->device,source,static_cast<size_t>(d.nbytes),
                                   cudaMemcpyHostToDevice,e->stream));
        cuda_check(cudaEventRecord(u->event,e->stream));
        auto handle=new_handle(); uploads.emplace(handle,u); *out=handle;
    });
}
int ml_uploaded_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(uploads,id); uploads.erase(id); } });
}
int ml_uploaded_descriptor(ml_handle id, ml_gguf_tensor* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null descriptor output");
        *out=lookup(uploads,id)->descriptor;
    });
}
int ml_uploaded_query(ml_handle id, int* ready) noexcept {
    return boundary([&] {
        if (!ready) fail(ML_INVALID,"null completion output");
        *ready=0;
        auto u=lookup(uploads,id); DeviceGuard guard(u->engine->device);
        if (!u->retired) {
            auto result=cudaEventQuery(u->event);
            if (result==cudaErrorNotReady) return;
            cuda_check(result); u->retire();
        }
        *ready=1;
    });
}
int ml_uploaded_wait(ml_handle id) noexcept {
    return boundary([&] { auto u=lookup(uploads,id); DeviceGuard guard(u->engine->device); u->wait(); });
}
int ml_uploaded_readback_for_test(ml_handle id, uint64_t offset, uint64_t size,
                                 void* destination, uint64_t capacity, uint64_t* copied) noexcept {
    return boundary([&] {
        if (!copied) fail(ML_INVALID,"null copied output");
        *copied=0;
        auto u=lookup(uploads,id);
        if (offset>u->descriptor.nbytes || size>u->descriptor.nbytes-offset)
            fail(ML_INVALID,"upload range out of bounds");
        if (size>capacity || size>SIZE_MAX || (size && !destination))
            fail(ML_INVALID,"invalid destination/capacity");
        DeviceGuard guard(u->engine->device); u->wait();
        // Synchronous D2H: no outstanding writes to caller memory on return.
        if (size) cuda_check(cudaMemcpy(destination,static_cast<const char*>(u->device)+offset,
                                       static_cast<size_t>(size),cudaMemcpyDeviceToHost));
        *copied=size;
    });
}
int ml_qsa_frontend_create(ml_handle id, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null frontend output");
        *out=0;
        auto e=lookup(engines,id);
        e->admit(QSAFrontend::bytes+QSAFrontend::pinned);
        DeviceGuard guard(e->device);
        auto w=std::make_shared<QSAFrontend>(e);
        cuda_check(cudaMalloc(&w->device,QSAFrontend::bytes));
        cuda_check(cudaMallocHost(&w->host,QSAFrontend::pinned));
        w->parameter=static_cast<uint64_t*>(w->device);
        float* cursor=reinterpret_cast<float*>(w->parameter+1);
        const int sizes[]={6144,512,6144,512,128,128};
        for (int j=0;j<6;++j) { w->outputs[j]=cursor; cursor+=sizes[j]; }
        w->ready=reinterpret_cast<int*>(cursor); w->status=w->ready+1;
        e->record(QSAFrontend::bytes,QSAFrontend::pinned); w->counted=true;
        cuda_check(cudaMemsetAsync(w->device,0,QSAFrontend::bytes,e->stream));
        w->set_n(0);
        auto handle=new_handle(); qsa_frontends.emplace(handle,w); *out=handle;
    });
}
int ml_qsa_frontend_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(qsa_frontends,id); qsa_frontends.erase(id); } });
}
int ml_qsa_frontend_set_n(ml_handle id, uint64_t n) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_frontends,id); w->valid=false; w->hidden_valid=false;
        if (!n) fail(ML_INVALID,"frontend n must be positive");
        DeviceGuard guard(w->engine->device); w->set_n(n);
    });
}
int ml_qsa_frontend_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_frontends,id); DeviceGuard guard(w->engine->device); w->set_n(0);
    });
}
int ml_qsa_frontend_run(ml_handle id, ml_handle qg, ml_handle k,
    ml_handle raw, ml_handle iq, ml_handle first_raw, ml_handle partial,
    ml_handle gamma_q, ml_handle gamma_k, ml_handle gamma_iq, ml_handle gamma_ik,
    float epsilon) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_frontends,id); w->valid=false; w->hidden_valid=false;
        if (!w->n || !std::isfinite(epsilon) || epsilon<=0)
            fail(ML_INVALID,"invalid frontend n/epsilon");
        ml_handle handles[]={qg,k,raw,iq,first_raw,partial,gamma_q,gamma_k,gamma_iq,gamma_ik};
        const uint64_t widths[]={512,256,128,128,128,128,256,256,128,128};
        const uint64_t rows[]={24,2,0,4,0,3,0,0,0,0};
        std::shared_ptr<UploadedTensor> u[10];
        for (int j=0;j<10;++j) {
            u[j]=lookup(uploads,handles[j]);
            if (u[j]->engine!=w->engine) fail(ML_INVALID,"frontend cross-engine input");
            const auto& d=u[j]->descriptor;
            if (d.type_id!=0 || d.rank!=(rows[j] ? 2u : 1u) ||
                d.ggml_shape[0]!=widths[j] || (rows[j] && d.ggml_shape[1]!=rows[j]))
                fail(ML_INVALID,"frontend requires exact F32 input shapes");
        }
        DeviceGuard guard(w->engine->device);
        auto stream=w->engine->stream;
        StreamCompletion completion{stream};
        for (const auto& input:u) input->wait();
        auto ptr=[&](int j) { return static_cast<const float*>(u[j]->device); };
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        cuda_check(ml_qsa_frontend_launch(stream,w->parameter,
            ptr(0),ptr(1),ptr(2),ptr(3),ptr(4),ptr(5),ptr(6),ptr(7),ptr(8),ptr(9),epsilon,
            w->outputs[0],w->outputs[1],w->outputs[2],w->outputs[3],w->outputs[4],
            w->ready,w->outputs[5],w->status));
        auto host_status=reinterpret_cast<int*>(static_cast<char*>(w->host)+8);
        cuda_check(cudaMemcpyAsync(host_status,w->status,4,cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if (*host_status) fail(ML_INVALID,"frontend nonfinite source or arithmetic");
        w->valid=true;
    });
}
int ml_qsa_frontend_from_hidden(ml_handle id, ml_handle hidden,
    ml_handle qg, ml_handle k, ml_handle v, ml_handle iq, ml_handle raw,
    ml_handle gamma_q, ml_handle gamma_k, ml_handle gamma_iq, ml_handle gamma_ik,
    ml_handle first_raw, ml_handle partial, uint64_t n, float epsilon) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_frontends,id); w->valid=false; w->hidden_valid=false;
        if (!n || !std::isfinite(epsilon) || epsilon<=0)
            fail(ML_INVALID,"invalid frontend n/epsilon");
        ml_handle handles[]={hidden,qg,k,v,raw,iq,gamma_q,gamma_k,gamma_iq,gamma_ik,first_raw,partial};
        const uint64_t widths[]={2560,2560,2560,2560,2560,2560,256,256,128,128,128,128};
        const uint64_t rows[]={0,12288,512,512,128,512,0,0,0,0,0,3};
        std::shared_ptr<UploadedTensor> u[12];
        for (int j=0;j<12;++j) {
            u[j]=lookup(uploads,handles[j]);
            if (u[j]->engine!=w->engine) fail(ML_INVALID,"frontend cross-engine input");
            const auto& d=u[j]->descriptor;
            bool weight=j>=1 && j<=5;
            if (weight && d.type_id!=0 && d.type_id!=30 && d.type_id!=12 &&
                d.type_id!=13 && d.type_id!=14 && d.type_id!=23)
                fail(ML_INVALID,"unsupported frontend projection format; requires F32/BF16/Q4_K/Q5_K/Q6_K/IQ4_XS");
            if ((!weight && d.type_id!=0) || d.rank!=(rows[j]?2u:1u) ||
                d.ggml_shape[0]!=widths[j] || (rows[j] && d.ggml_shape[1]!=rows[j]))
                fail(ML_INVALID,"frontend hidden requires exact input shapes and F32 activations/gammas/history");
        }
        DeviceGuard guard(w->engine->device);
        auto stream=w->engine->stream;
        StreamCompletion completion{stream};
        if (!w->hidden_device) {
            w->engine->admit(QSAFrontend::hidden_bytes);
            cuda_check(cudaMalloc(&w->hidden_device,QSAFrontend::hidden_bytes));
            w->engine->record(QSAFrontend::hidden_bytes,0);
            float* cursor=static_cast<float*>(w->hidden_device);
            const int sizes[]={12288,512,512,128,512};
            for (int j=0;j<5;++j) { w->projections[j]=cursor; cursor+=sizes[j]; }
        }
        for (const auto& input:u) input->wait();
        w->set_n(n);
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        for (int j=0;j<5;++j) {
            const auto& d=u[j+1]->descriptor;
            cuda_check(ml_qsa_projection_launch(stream,u[j+1]->device,d.type_id,
                d.ggml_shape[1],2560,d.row_bytes,static_cast<const float*>(u[0]->device),
                w->projections[j],w->status));
        }
        auto ptr=[&](int j) { return static_cast<const float*>(u[j]->device); };
        cuda_check(ml_qsa_frontend_launch(stream,w->parameter,
            w->projections[0],w->projections[1],w->projections[3],w->projections[4],
            ptr(10),ptr(11),ptr(6),ptr(7),ptr(8),ptr(9),epsilon,
            w->outputs[0],w->outputs[1],w->outputs[2],w->outputs[3],w->outputs[4],
            w->ready,w->outputs[5],w->status));
        auto host_status=reinterpret_cast<int*>(static_cast<char*>(w->host)+8);
        cuda_check(cudaMemcpyAsync(host_status,w->status,4,cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if (*host_status) fail(ML_INVALID,"frontend nonfinite source or arithmetic");
        w->valid=true; w->hidden_valid=true;
    });
}
int ml_qsa_frontend_hidden_readback_for_test(ml_handle id,
    float* qg, uint64_t qc, float* k, uint64_t kc, float* v, uint64_t vc,
    float* raw, uint64_t rc, float* iq, uint64_t ic) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_frontends,id);
        if (!w->valid || !w->hidden_valid) fail(ML_INVALID,"frontend hidden results unavailable");
        float* destinations[]={qg,k,v,raw,iq};
        const uint64_t capacities[]={qc,kc,vc,rc,ic}, sizes[]={12288,512,512,128,512};
        for (int j=0;j<5;++j)
            if (!destinations[j] || capacities[j]<sizes[j])
                fail(ML_INVALID,"frontend hidden readback destination/capacity");
        DeviceGuard guard(w->engine->device);
        for (int j=0;j<5;++j)
            cuda_check(cudaMemcpy(destinations[j],w->projections[j],sizes[j]*4,cudaMemcpyDeviceToHost));
    });
}
int ml_qsa_frontend_readback_for_test(ml_handle id,
    float* q, uint64_t qc, float* k, uint64_t kc, float* gate, uint64_t gc,
    float* iq, uint64_t ic, float* dead, uint64_t dc, float* pool, uint64_t pc,
    int* ready, uint64_t* n) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_frontends,id);
        if (!w->valid) fail(ML_INVALID,"frontend results unavailable");
        float* destinations[]={q,k,gate,iq,dead,pool};
        const uint64_t capacities[]={qc,kc,gc,ic,dc,pc}, sizes[]={6144,512,6144,512,128,128};
        if (!ready || !n) fail(ML_INVALID,"null frontend diagnostic scalar");
        for (int j=0;j<6;++j)
            if (!destinations[j] || capacities[j]<sizes[j])
                fail(ML_INVALID,"frontend readback destination/capacity");
        DeviceGuard guard(w->engine->device);
        for (int j=0;j<6;++j)
            cuda_check(cudaMemcpy(destinations[j],w->outputs[j],sizes[j]*4,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(ready,w->ready,4,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(n,w->parameter,8,cudaMemcpyDeviceToHost));
    });
}
int ml_qsa_create(ml_handle id, uint64_t capacity, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null workspace output");
        *out=0;
        auto e=lookup(engines,id);
        if (!capacity || capacity%4 || capacity>uint64_t(INT32_MAX)-3)
            fail(ML_INVALID,"QSA capacity must be a positive multiple of 4 <= INT32_MAX-3");
        auto w=std::make_shared<QSAWorkspace>(e,capacity);
        e->admit(w->bytes+12);
        DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(&w->device,static_cast<size_t>(w->bytes)));
        cuda_check(cudaMallocHost(&w->host,12));
        w->parameter=static_cast<uint64_t*>(w->device);
        w->ids=reinterpret_cast<int64_t*>(w->parameter+1);
        w->scores=reinterpret_cast<float*>(w->ids+std::min(capacity,uint64_t(2051)));
        w->status=reinterpret_cast<int*>(w->scores+capacity);
        e->record(w->bytes,12); w->counted=true;
        cuda_check(cudaEventCreate(&w->start)); cuda_check(cudaEventCreate(&w->end));
        w->set_n(0);
        auto handle=new_handle(); qsa_workspaces.emplace(handle,w); *out=handle;
    });
}
int ml_qsa_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(qsa_workspaces,id); qsa_workspaces.erase(id); } });
}
int ml_qsa_set_n(ml_handle id, uint64_t n) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_workspaces,id);
        if (!n || n>w->capacity) fail(ML_INVALID,"QSA n out of bounds");
        DeviceGuard guard(w->engine->device); w->set_n(n);
    });
}
int ml_qsa_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_workspaces,id); DeviceGuard guard(w->engine->device); w->set_n(0);
    });
}
int ml_qsa_run(ml_handle id, ml_handle query, ml_handle pooled, ml_handle dead,
               int policy, uint64_t repeats, float* milliseconds) noexcept {
    return boundary([&] {
        if (!milliseconds) fail(ML_INVALID,"null timing output");
        *milliseconds=0;
        auto w=lookup(qsa_workspaces,id);
        if (!w->n || (policy!=0 && policy!=1) || !repeats || repeats>10000)
            fail(ML_INVALID,"invalid QSA n/policy/repeats");
        auto q=lookup(uploads,query), p=lookup(uploads,pooled), d=lookup(uploads,dead);
        auto check=[&](const auto& u, uint64_t rank, uint64_t second) {
            const auto& desc=u->descriptor;
            if (u->engine!=w->engine) fail(ML_INVALID,"QSA cross-engine input");
            if (desc.type_id!=0 || desc.rank!=rank || desc.ggml_shape[0]!=128 ||
                (rank==2 && desc.ggml_shape[1]!=second)) fail(ML_INVALID,"QSA requires exact F32 input shapes");
        };
        check(q,2,4); check(p,2,w->capacity/4); check(d,1,0);
        DeviceGuard guard(w->engine->device);
        w->valid=false;
        // Uploads use the same stream; waiting also safely retires mmap leases.
        q->wait(); p->wait(); d->wait();
        auto stream=w->engine->stream;
        cuda_check(cudaEventRecord(w->start,stream));
        for (uint64_t i=0;i<repeats;++i)
            cuda_check(ml_qsa_launch(stream,w->capacity,w->parameter,
                static_cast<const float*>(q->device),static_cast<const float*>(p->device),
                static_cast<const float*>(d->device),w->scores,w->ids,w->status,policy));
        cuda_check(cudaEventRecord(w->end,stream));
        auto host_status=reinterpret_cast<int*>(static_cast<char*>(w->host)+8);
        cuda_check(cudaMemcpyAsync(host_status,w->status,4,cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if (*host_status) fail(ML_INVALID,"QSA nonfinite input or score arithmetic");
        float elapsed=0; cuda_check(cudaEventElapsedTime(&elapsed,w->start,w->end));
        *milliseconds=elapsed/static_cast<float>(repeats); w->valid=true;
    });
}
int ml_qsa_readback_for_test(ml_handle id, float* scores, uint64_t score_capacity,
                            int64_t* ids, uint64_t id_capacity, uint64_t* n) noexcept {
    return boundary([&] {
        if (!n) fail(ML_INVALID,"null n output");
        *n=0;
        auto w=lookup(qsa_workspaces,id);
        auto width=std::min(w->n,uint64_t(2051));
        if (!w->valid) fail(ML_INVALID,"QSA results unavailable; run after set_n/reset");
        if (!scores || !ids || score_capacity<w->n || id_capacity<width)
            fail(ML_INVALID,"QSA readback destination/capacity");
        DeviceGuard guard(w->engine->device);
        cuda_check(cudaMemcpy(scores,w->scores,w->n*4,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(ids,w->ids,width*8,cudaMemcpyDeviceToHost)); *n=w->n;
    });
}
int ml_qsa_attention_create(ml_handle id, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null attention output");
        *out=0;
        auto s=lookup(qsa_workspaces,id);
        auto e=s->engine;
        e->admit(49164);
        auto a=std::make_shared<QSAAttention>(s);
        DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(&a->device,49156));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&a->host),8));
        a->contexts=static_cast<float*>(a->device);
        a->gated=a->contexts+6144;
        a->status=reinterpret_cast<int*>(a->gated+6144);
        cuda_check(cudaEventCreate(&a->start)); cuda_check(cudaEventCreate(&a->end));
        e->record(49156,8); a->counted=true;
        auto handle=new_handle(); qsa_attentions.emplace(handle,a); *out=handle;
    });
}
int ml_qsa_attention_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(qsa_attentions,id); qsa_attentions.erase(id); } });
}
int ml_qsa_attention_run(ml_handle id, ml_handle q, ml_handle gate,
    ml_handle k, ml_handle v, ml_handle keys, ml_handle values,
    ml_handle index_query, ml_handle pooled, ml_handle dead, int policy,
    float scale, uint64_t repeats, float* milliseconds) noexcept {
    return boundary([&] {
        if (!milliseconds) fail(ML_INVALID,"null timing output");
        *milliseconds=0;
        auto a=lookup(qsa_attentions,id);
        a->valid=false;
        auto s=a->selection;
        const bool combined=index_query || pooled || dead;
        if (!s->n || (!combined && !s->valid) || policy<0 || policy>1 ||
            !std::isfinite(scale) || scale<=0 || !repeats || repeats>10000)
            fail(ML_INVALID,"invalid attention selection/scale/policy/repeats");
        std::shared_ptr<UploadedTensor> u[9];
        ml_handle handles[]={q,gate,k,v,keys,values,index_query,pooled,dead};
        for (int j=0; j<(combined ? 9 : 6); ++j) {
            u[j]=lookup(uploads,handles[j]);
            if (u[j]->engine!=s->engine) fail(ML_INVALID,"attention cross-engine input");
        }
        auto check=[&](int j, uint64_t rank, uint64_t x, uint64_t y, uint64_t z) {
            const auto& d=u[j]->descriptor;
            if (d.type_id!=0 || d.rank!=rank || d.ggml_shape[0]!=x ||
                (rank>1 && d.ggml_shape[1]!=y) || (rank>2 && d.ggml_shape[2]!=z))
                fail(ML_INVALID,"attention requires exact F32 input shapes");
        };
        check(0,2,256,24,0); check(1,2,256,24,0);
        check(2,2,256,2,0); check(3,2,256,2,0);
        check(4,3,256,2,s->capacity); check(5,3,256,2,s->capacity);
        if (combined) { check(6,2,128,4,0); check(7,2,128,s->capacity/4,0); check(8,1,128,0,0); }
        DeviceGuard guard(s->engine->device);
        StreamCompletion completion{s->engine->stream};
        for (const auto& input:u) if (input) input->wait();
        auto stream=s->engine->stream;
        auto ptr=[&](int j) { return static_cast<const float*>(u[j]->device); };
        if (combined) s->valid=false;
        cuda_check(cudaEventRecord(a->start,stream));
        cuda_check(cudaMemsetAsync(a->status,0,4,stream));
        for (uint64_t i=0; i<repeats; ++i) {
            if (combined) cuda_check(ml_qsa_launch(stream,s->capacity,s->parameter,
                ptr(6),ptr(7),ptr(8),s->scores,s->ids,s->status,policy));
            cuda_check(ml_qsa_attention_launch(stream,s->parameter,s->ids,
                ptr(0),ptr(1),ptr(2),ptr(3),ptr(4),ptr(5),a->contexts,a->gated,a->status,scale));
        }
        cuda_check(cudaEventRecord(a->end,stream));
        cuda_check(cudaMemcpyAsync(a->host,a->status,4,cudaMemcpyDeviceToHost,stream));
        if (combined) cuda_check(cudaMemcpyAsync(a->host+1,s->status,4,cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if (combined) s->valid=!a->host[1];
        if (a->host[0] || (combined && a->host[1])) fail(ML_INVALID,"attention nonfinite source or arithmetic");
        float elapsed=0; cuda_check(cudaEventElapsedTime(&elapsed,a->start,a->end));
        *milliseconds=elapsed/static_cast<float>(repeats); a->valid=true;
    });
}
int ml_qsa_attention_readback_for_test(ml_handle id, float* contexts,
    uint64_t context_capacity, float* gated, uint64_t gated_capacity) noexcept {
    return boundary([&] {
        auto a=lookup(qsa_attentions,id);
        if (!a->valid) fail(ML_INVALID,"attention results unavailable");
        if (!contexts || !gated || context_capacity<6144 || gated_capacity<6144)
            fail(ML_INVALID,"attention readback destination/capacity");
        DeviceGuard guard(a->selection->engine->device);
        cuda_check(cudaMemcpy(contexts,a->contexts,24576,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(gated,a->gated,24576,cudaMemcpyDeviceToHost));
    });
}
int ml_qsa_projection_create(ml_handle aid, ml_handle wid, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null projection output");
        *out=0;
        auto a=lookup(qsa_attentions,aid);
        auto w=lookup(uploads,wid);
        auto e=a->selection->engine;
        const auto& d=w->descriptor;
        if (w->engine!=e) fail(ML_INVALID,"projection cross-engine weight");
        if (d.rank!=2 || d.ggml_shape[0]!=6144 || !d.ggml_shape[1] || d.ggml_shape[1]>2560 ||
            (d.type_id!=0 && d.type_id!=30 && d.type_id!=12 && d.type_id!=14))
            fail(ML_INVALID,"projection requires F32/BF16/Q4_K/Q6_K [6144,rows<=2560]");
        auto p=std::make_shared<QSAProjection>(a,w);
        e->admit(p->bytes+4);
        DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(&p->device,p->bytes));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&p->host),4));
        cuda_check(cudaEventCreate(&p->start)); cuda_check(cudaEventCreate(&p->end));
        e->record(p->bytes,4); p->counted=true;
        auto id=new_handle(); qsa_projections.emplace(id,p); *out=id;
    });
}
int ml_qsa_projection_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(qsa_projections,id); qsa_projections.erase(id); } });
}
int ml_qsa_projection_run(ml_handle id, uint64_t repeats, float* milliseconds) noexcept {
    return boundary([&] {
        auto p=lookup(qsa_projections,id); p->valid=false;
        if (!milliseconds) fail(ML_INVALID,"null projection timing");
        *milliseconds=0;
        if (!p->attention->valid) fail(ML_INVALID,"attention results unavailable");
        if (!repeats || repeats>10000) fail(ML_INVALID,"invalid projection repeats");
        auto e=p->weight->engine;
        DeviceGuard guard(e->device);
        StreamCompletion completion{e->stream};
        p->weight->wait();
        auto output=static_cast<float*>(p->device);
        auto status=reinterpret_cast<int*>(output+p->rows);
        cuda_check(cudaEventRecord(p->start,e->stream));
        cuda_check(cudaMemsetAsync(status,0,4,e->stream));
        for (uint64_t i=0;i<repeats;++i)
            cuda_check(ml_qsa_projection_launch(e->stream,p->weight->device,
                p->weight->descriptor.type_id,p->rows,6144,p->weight->descriptor.row_bytes,
                p->attention->gated,output,status));
        cuda_check(cudaEventRecord(p->end,e->stream));
        cuda_check(cudaMemcpyAsync(p->host,status,4,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaStreamSynchronize(e->stream));
        if (*p->host) fail(ML_INVALID,"projection nonfinite source or arithmetic");
        float elapsed=0; cuda_check(cudaEventElapsedTime(&elapsed,p->start,p->end));
        *milliseconds=elapsed/static_cast<float>(repeats); p->valid=true;
    });
}
int ml_qsa_projection_readback_for_test(ml_handle id, float* output, uint64_t capacity) noexcept {
    return boundary([&] {
        auto p=lookup(qsa_projections,id);
        if (!p->valid) fail(ML_INVALID,"projection results unavailable");
        if (!output || capacity<p->rows) fail(ML_INVALID,"projection readback destination/capacity");
        DeviceGuard guard(p->weight->engine->device);
        cuda_check(cudaMemcpy(output,p->device,p->rows*4,cudaMemcpyDeviceToHost));
    });
}
int ml_session_create(ml_handle id, uint64_t capacity, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID, "null output");
        *out=0;
        auto e=lookup(engines,id);
        if (!capacity) fail(ML_INVALID, "capacity must be positive");
        if (e->sessions>=e->max_sessions) fail(ML_CAPACITY, "session allocation budget exhausted");
                e->admit(12);
        auto s=std::make_shared<Session>(e,capacity);
        DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&s->device),4));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&s->host),8));
        auto handle=new_handle(); sessions.emplace(handle,s);
        ++e->sessions; s->counted=true; e->record(4,8); *out=handle;
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
