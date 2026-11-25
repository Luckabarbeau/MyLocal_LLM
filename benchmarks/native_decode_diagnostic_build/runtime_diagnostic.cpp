#include "runtime.h"
#include "sampling.h"
#include "generation.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include "gguf_loader.h"
#include "qsa_selection.h"
#include "qsa_attention.h"
#include "kv_pages.h"
#include "kv_streamed.h"
#include "qsa_paged.h"
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
cudaError_t ml_vocab_projection_launch(cudaStream_t,const void*,uint32_t,uint64_t,uint64_t,uint64_t,const float*,float*,int*);


// WP9 standalone diagnostic ONLY. No added CUDA synchronization or changed math.
namespace wp9 {
using Clock=std::chrono::steady_clock;
struct Entry { unsigned long long count=0; double seconds=0; };
thread_local bool enabled=false;
thread_local Entry entries[10];
const char* names[]={"hc_read", "moe_submit", "moe_finish", "recurrent",
    "qsa", "hc_write", "ple", "cpu_expert_exposed_wait", "final_head_tail", "cudaStreamSynchronize"};
struct Timer {
    int slot; bool on; Clock::time_point start;
    explicit Timer(int s):slot(s),on(enabled) { if(on) start=Clock::now(); }
    ~Timer() { if(on) { entries[slot].count++; entries[slot].seconds+=std::chrono::duration<double>(Clock::now()-start).count(); } }
};
cudaError_t sync(cudaStream_t s) { Timer t(9); return ::cudaStreamSynchronize(s); }
}
extern "C" void ml_wp9_diag_start() { for(auto& e:wp9::entries) e={}; wp9::enabled=true; }
extern "C" const char* ml_wp9_diag_stop() {
    wp9::enabled=false; static thread_local char out[4096]; size_t n=0;
    n+=std::snprintf(out+n,sizeof(out)-n,"{");
    for(int i=0;i<10;++i) n+=std::snprintf(out+n,sizeof(out)-n,
        "%s\"%s\":{\"count\":%llu,\"seconds\":%.12g}",i?",":"",wp9::names[i],wp9::entries[i].count,wp9::entries[i].seconds);
    std::snprintf(out+n,sizeof(out)-n,"}"); return out;
}
#define cudaStreamSynchronize wp9::sync

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
// Optional complete-layer backing. Existing standalone stores remain unchanged.
struct QSAPagedState {
    std::shared_ptr<Engine> engine;
    uint64_t capacity,page_size,slots,logical_pages,row_bytes,page_bytes,host_page_bytes;
    bool quantized,raw_diagnostics;
    ml_kvi8_stats stats{};
    std::unordered_map<uint64_t,std::unique_ptr<unsigned char[]>> backing;
    std::unique_ptr<uint64_t[]> host_table,slot_page,slot_generation;
    KVPageRequests host_requests{};
    void* device=nullptr;
    uint64_t *table=nullptr,*compact_n=nullptr;
    int64_t* compact_ids=nullptr;
    KVPageRequests* requests=nullptr;
    float* scratch=nullptr;
    unsigned char *pages=nullptr,*stage=nullptr;
    cudaEvent_t completion=nullptr;
    bool counted=false,pending=false;
    QSAPagedState(std::shared_ptr<Engine> e,uint64_t c,uint64_t p,uint64_t s,
        uint64_t budget,bool quant,bool debug):engine(std::move(e)),capacity(c),page_size(p),
        slots(s),logical_pages((c+p-1)/p),row_bytes(quant?1056:4096),page_bytes(p*row_bytes),
        host_page_bytes(page_bytes+(debug?p*512:0)),quantized(quant),raw_diagnostics(debug) {
        stats.generation=1; stats.host_budget=budget;
        stats.host_bytes=sizeof(KVPageRequests)+8*(logical_pages+2*slots);
        stats.device_bytes=logical_pages*8+sizeof(KVPageRequests)+8+2051*8+
            2051*4096+(slots+1)*page_bytes;
    }
    ~QSAPagedState() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (completion) (void)cudaEventDestroy(completion);
        if (counted) engine->device_bytes-=stats.device_bytes;
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void wait() { if (pending) { cuda_check(cudaEventSynchronize(completion)); pending=false; } }
    void finish() { cuda_check(cudaEventRecord(completion,engine->stream)); pending=true; wait(); }
    void allocate() {
        if (stats.host_bytes>stats.host_budget) fail(ML_CAPACITY,"paged QSA host metadata budget exhausted");
        engine->admit(stats.device_bytes);
        host_table=std::make_unique<uint64_t[]>(logical_pages);
        slot_page=std::make_unique<uint64_t[]>(slots); slot_generation=std::make_unique<uint64_t[]>(slots);
        std::fill_n(slot_page.get(),slots,UINT64_MAX);
        cuda_check(cudaEventCreateWithFlags(&completion,cudaEventDisableTiming));
        cuda_check(cudaMalloc(&device,stats.device_bytes)); engine->record(stats.device_bytes,0); counted=true;
        auto* cursor=static_cast<unsigned char*>(device);
        table=reinterpret_cast<uint64_t*>(cursor); cursor+=logical_pages*8;
        requests=reinterpret_cast<KVPageRequests*>(cursor); cursor+=sizeof(KVPageRequests);
        compact_n=reinterpret_cast<uint64_t*>(cursor); cursor+=8;
        compact_ids=reinterpret_cast<int64_t*>(cursor); cursor+=2051*8;
        scratch=reinterpret_cast<float*>(cursor); cursor+=2051*4096;
        pages=cursor; cursor+=slots*page_bytes; stage=cursor;
        cuda_check(cudaMemsetAsync(device,0,stats.device_bytes,engine->stream)); finish();
    }
    void reserve(uint64_t logical) {
        if (backing.count(logical)) return;
        if (host_page_bytes>stats.host_budget-stats.host_bytes)
            fail(ML_CAPACITY,"paged QSA host page budget exhausted");
        auto page=std::make_unique<unsigned char[]>(host_page_bytes);
        backing.emplace(logical,std::move(page)); stats.host_bytes+=host_page_bytes;
    }
    void update(uint64_t logical,uint64_t entry) {
        host_table[logical]=entry;
        cuda_check(cudaMemcpyAsync(table+logical,host_table.get()+logical,8,cudaMemcpyHostToDevice,engine->stream));
        finish();
    }
    void remove(uint64_t slot) {
        if (slot_page[slot]!=UINT64_MAX) {
            update(slot_page[slot],0); slot_page[slot]=UINT64_MAX; slot_generation[slot]=0;
            --stats.resident_pages;
        }
    }
    void reset() {
        wait();
        if (stats.generation==UINT64_MAX) fail(ML_CAPACITY,"paged QSA generation exhausted");
        cuda_check(cudaMemsetAsync(table,0,logical_pages*8,engine->stream)); finish();
        std::fill_n(host_table.get(),logical_pages,0); std::fill_n(slot_page.get(),slots,UINT64_MAX);
        std::fill_n(slot_generation.get(),slots,0); backing.clear();
        stats.host_bytes=sizeof(KVPageRequests)+8*(logical_pages+2*slots);
        ++stats.generation; stats.length=stats.populated_bytes=stats.resident_pages=0;
        stats.hits=stats.misses=stats.h2d_bytes=stats.metadata_bytes=0;
    }
    void truncate(uint64_t length) {
        if(length>stats.length) fail(ML_INVALID,"paged QSA rollback beyond committed prefix");
        if(stats.generation==UINT64_MAX) fail(ML_CAPACITY,"paged QSA generation exhausted");
        // Retire every gather/copy before invalidating mappings or freeing suffix
        // pages. Prefix backing is append-only and is deliberately not copied.
        wait(); cuda_check(cudaStreamSynchronize(engine->stream));
        cuda_check(cudaMemsetAsync(table,0,logical_pages*8,engine->stream)); finish();
        std::fill_n(host_table.get(),logical_pages,0);
        std::fill_n(slot_page.get(),slots,UINT64_MAX);
        std::fill_n(slot_generation.get(),slots,0);
        const uint64_t keep=(length+page_size-1)/page_size;
        for(auto it=backing.begin();it!=backing.end();) {
            if(it->first>=keep) { it=backing.erase(it); stats.host_bytes-=host_page_bytes; }
            else ++it;
        }
        ++stats.generation; stats.length=length; stats.populated_bytes=length*row_bytes;
        stats.resident_pages=0;
    }
    void pack(const float* k,const float* v,uint64_t rows,int* status) {
        wait(); cuda_check(ml_qsa_paged_pack(engine->stream,k,v,rows,quantized,stage,status));
    }
    // Called only after output validation; failure leaves the parent layer poisoned.
    void commit(uint64_t start,uint64_t rows,const float* raw) {
        const uint64_t logical=start/page_size,offset=start%page_size;
        auto* host=backing.at(logical).get();
        cuda_check(cudaMemcpyAsync(host+offset*row_bytes,stage,rows*row_bytes,cudaMemcpyDeviceToHost,engine->stream));
        if (raw_diagnostics) cuda_check(cudaMemcpyAsync(host+page_bytes+offset*512,raw,
            rows*512,cudaMemcpyDeviceToHost,engine->stream));
        if (host_table[logical]) remove(host_table[logical]-1);
        finish();
    }
    void gather(const uint64_t* n,const int64_t* ids,uint64_t active_n,
        const float* q,const float* gate,float* contexts,float* gated,int* status,float scale) {
        cuda_check(cudaMemsetAsync(requests,0,sizeof(KVPageRequests),engine->stream));
        cuda_check(ml_kvi8_requests(engine->stream,n,ids,table,page_size,slots,requests));
        cuda_check(cudaMemcpyAsync(&host_requests,requests,sizeof(KVPageRequests),cudaMemcpyDeviceToHost,engine->stream));
        finish(); stats.metadata_bytes+=sizeof(KVPageRequests);
        if (host_requests.error) fail(ML_INVALID,"paged QSA invalid device page request");
        auto gather_page=[&](uint64_t logical,const unsigned char* source) {
            cuda_check(ml_qsa_paged_gather(engine->stream,n,ids,logical,page_size,quantized,
                source,scratch,compact_n,compact_ids));
        };
        gather_page(UINT64_MAX,stage);
        // Gather all hits before permitting any selected resident page to retire.
        for (uint64_t j=0;j<host_requests.count;++j) {
            const uint64_t logical=host_requests.pages[j],entry=host_table[logical];
            if (!entry) continue;
            if (slot_generation[entry-1]!=stats.generation) fail(ML_INVALID,"paged QSA slot generation mismatch");
            gather_page(logical,pages+(entry-1)*page_bytes);
        }
        finish();
        for (uint64_t j=0;j<host_requests.misses;++j) {
            const uint64_t logical=host_requests.missing[j];
            uint64_t slot=j%slots;
            for (uint64_t i=0;i<slots;++i) if (slot_page[i]==UINT64_MAX) { slot=i; break; }
            remove(slot);
            const uint64_t rows=std::min(page_size,stats.length-logical*page_size);
            cuda_check(cudaMemcpyAsync(pages+slot*page_bytes,backing.at(logical).get(),
                rows*row_bytes,cudaMemcpyHostToDevice,engine->stream));
            update(logical,slot+1); slot_page[slot]=logical; slot_generation[slot]=stats.generation;
            ++stats.resident_pages; stats.h2d_bytes+=rows*row_bytes;
            gather_page(logical,pages+slot*page_bytes);
            finish(); // The slot's last reader retires before the next reuse.
        }
        stats.hits+=host_requests.count-host_requests.misses; stats.misses+=host_requests.misses;
        const uint64_t width=std::min(active_n,uint64_t(2051));
        cuda_check(ml_qsa_attention_launch(engine->stream,compact_n,compact_ids,q,gate,
            scratch+(width-1)*512,scratch+2051*512+(width-1)*512,
            scratch,scratch+2051*512,contexts,gated,status,scale));
    }
    void read(uint64_t length,int which,float* out,uint64_t start=0) {
        if (which==1 && !raw_diagnostics) fail(ML_INVALID,"paged QSA raw diagnostics disabled");
        wait();
        for (uint64_t t=start;t<length;++t) {
            const auto* page=backing.at(t/page_size).get();
            if (which==1) { std::memcpy(out+(t-start)*128,page+page_bytes+(t%page_size)*512,512); continue; }
            const auto* row=page+(t%page_size)*row_bytes;
            const int base=which==2?0:512;
            if (!quantized) std::memcpy(out+(t-start)*512,row+base*4,2048);
            else for (int d=0;d<512;++d) {
                const auto* group=row+((base+d)/64)*66;
                uint16_t bits; std::memcpy(&bits,group+64,2);
                const int exponent=bits>>10, mantissa=bits&1023;
                const float sf=std::ldexp(float(exponent?1024+mantissa:mantissa),exponent?exponent-25:-24);
                out[(t-start)*512+d]=float(reinterpret_cast<const int8_t*>(group)[d%64])*sf;
            }
        }
    }
};

// Included in runtime.cpp's private namespace after boundary/lookup helpers.
struct QSALayer {
    std::shared_ptr<Engine> engine;
    std::shared_ptr<QSAPagedState> paged;
    std::shared_ptr<UploadedTensor> tensors[10]; // Q,K,V,IQ,IK,O; four gammas
    uint64_t capacity, length=0, bytes=0;
    float epsilon, scale;
    int policy;
    bool counted=false, invalid=false, output_valid=false;
    void* device=nullptr;
    void* host=nullptr;
    uint64_t* n=nullptr;
    int64_t* ids=nullptr;
    int *status=nullptr, *ready=nullptr;
    float *raw=nullptr, *keys=nullptr, *values=nullptr, *pooled=nullptr, *staged_pool=nullptr;
    float *projected[5]{}, *q=nullptr, *k=nullptr, *gate=nullptr, *iq=nullptr, *dead=nullptr;
    float *pool=nullptr, *contexts=nullptr, *gated=nullptr, *output=nullptr, *scores=nullptr;
    QSALayer(std::shared_ptr<Engine> e, uint64_t c, int p, float eps, float s)
        : engine(std::move(e)), capacity(c), epsilon(eps), scale(s), policy(p) {
        // uint64 n, 2051 int64 IDs, two ints; all remaining storage is F32.
        bytes=8+2051*8+8+4*(c*(128+512+512+32+32+1)+13952+6144+512+6144+512+128+128+6144+6144+2560);
    }
    ~QSALayer() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=16; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void allocate() {
        engine->admit(bytes+16);
        cuda_check(cudaMalloc(&device,bytes)); cuda_check(cudaMallocHost(&host,16));
        engine->record(bytes,16); counted=true;
        n=static_cast<uint64_t*>(device); ids=reinterpret_cast<int64_t*>(n+1);
        status=reinterpret_cast<int*>(ids+2051); ready=status+1;
        float* cursor=reinterpret_cast<float*>(ready+1);
        auto take=[&](uint64_t count) { float* result=cursor; cursor+=count; return result; };
        if (paged) raw=take(128+4*128);
        else { raw=take(capacity*128); keys=take(capacity*512); values=take(capacity*512); }
        pooled=take(capacity*32); staged_pool=take(capacity*32); scores=take(capacity);
        const int sizes[]={12288,512,512,128,512};
        for (int j=0;j<5;++j) projected[j]=take(sizes[j]);
        q=take(6144); k=take(512); gate=take(6144); iq=take(512); dead=take(128); pool=take(128);
        contexts=take(6144); gated=take(6144); output=take(2560);
        cuda_check(cudaMemsetAsync(device,0,bytes,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
    }
    void check_status() {
        cuda_check(cudaMemcpyAsync(static_cast<char*>(host)+8,status,4,cudaMemcpyDeviceToHost,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
        if (*reinterpret_cast<int*>(static_cast<char*>(host)+8)) fail(ML_INVALID,"QSA layer nonfinite source or arithmetic");
    }
    void copy(float* dst, const float* src, uint64_t count) {
        if (count) cuda_check(cudaMemcpyAsync(dst,src,count*4,cudaMemcpyDeviceToDevice,engine->stream));
    }
};
std::unordered_map<ml_handle,std::shared_ptr<QSALayer>> qsa_layers;
void layer_shape(const std::shared_ptr<UploadedTensor>& u, const std::shared_ptr<Engine>& e,
                 uint64_t width, uint64_t rows, bool weight=false) {
    const auto& d=u->descriptor;
    if (u->engine!=e) fail(ML_INVALID,"QSA layer cross-engine input");
    if (d.rank!=(rows?2u:1u) || d.ggml_shape[0]!=width || (rows && d.ggml_shape[1]!=rows))
        fail(ML_INVALID,"QSA layer exact input shape required");
    if (d.type_id!=0 && (!weight || (d.type_id!=30 && d.type_id!=12 && d.type_id!=13 && d.type_id!=14 && d.type_id!=23)))
        fail(ML_INVALID,"QSA layer unsupported format");
}
void qsa_layer_step_device(const std::shared_ptr<QSALayer>& w, const float* hidden,
                           uint64_t position, uint64_t* committed) {
        auto stream=w->engine->stream;
        const uint64_t n=position+1;
        if (w->paged) w->paged->reserve(position/w->paged->page_size);
        *static_cast<uint64_t*>(w->host)=n;
        try {
            cuda_check(cudaMemcpyAsync(w->n,w->host,8,cudaMemcpyHostToDevice,stream));
            cuda_check(cudaMemsetAsync(w->status,0,4,stream));
            const int order[]={0,1,2,4,3};
            for (int j=0;j<5;++j) {
                auto u=w->tensors[order[j]]; const auto& d=u->descriptor;
                cuda_check(ml_qsa_projection_launch(stream,u->device,d.type_id,d.ggml_shape[1],2560,
                    d.row_bytes,hidden,w->projected[j],w->status));
            }
            auto gamma=[&](int j) { return static_cast<const float*>(w->tensors[j+6]->device); };
            cuda_check(ml_qsa_frontend_launch(stream,w->n,w->projected[0],w->projected[1],w->projected[3],w->projected[4],
                w->raw,w->paged ? w->raw+128 : w->raw+(n%4==0 ? n-4 : 0)*128,gamma(0),gamma(1),gamma(2),gamma(3),w->epsilon,
                w->q,w->k,w->gate,w->iq,w->dead,w->ready,w->pool,w->status));
            w->copy(w->staged_pool,w->pooled,(position/4)*128);
            if (n%4==0) w->copy(w->staged_pool+(n/4-1)*128,w->pool,128);
            cuda_check(ml_qsa_launch(stream,w->paged ? (n+3)/4*4 : w->capacity,w->n,w->iq,w->staged_pool,w->dead,
                w->scores,w->ids,w->status,w->policy,false));
            if (w->paged) {
                w->paged->pack(w->k,w->projected[2],1,w->status);
                w->check_status(); // No page scheduling on an invalid frontend/selection/current operand.
                w->paged->gather(w->n,w->ids,n,w->q,w->gate,w->contexts,w->gated,w->status,w->scale);
            } else cuda_check(ml_qsa_attention_launch(stream,w->n,w->ids,w->q,w->gate,w->k,w->projected[2],
                w->keys,w->values,w->contexts,w->gated,w->status,w->scale));
            auto u=w->tensors[5]; const auto& d=u->descriptor;
            cuda_check(ml_qsa_projection_launch(stream,u->device,d.type_id,2560,6144,d.row_bytes,w->gated,w->output,w->status));
            w->check_status();
        } catch (const Failure& f) { if (f.status==ML_CUDA) w->invalid=true; throw; }
        // No committed cache writes before successful output/status validation.
        w->invalid=true;
        if (w->paged) {
            w->paged->commit(position,1,w->projected[3]);
            if (!position) w->copy(w->raw,w->projected[3],128);
            w->copy(w->raw+128+(position%4)*128,w->projected[3],128);
        } else {
            w->copy(w->raw+position*128,w->projected[3],128);
            w->copy(w->keys+position*512,w->k,512);
            w->copy(w->values+position*512,w->projected[2],512);
        }
        if (n%4==0) w->copy(w->pooled+(n/4-1)*128,w->pool,128);
        cuda_check(cudaStreamSynchronize(stream));
        if (w->paged) { w->paged->stats.length=n; w->paged->stats.populated_bytes=n*w->paged->row_bytes; }
        w->length=n; w->invalid=false; w->output_valid=true; *committed=n;
}
extern "C" {
int ml_qsa_layer_create(ml_handle engine, uint64_t capacity, const ml_handle* tensors,
                        int policy, float epsilon, float scale, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out || !tensors || (capacity!=4096 && capacity!=16384) || (policy!=0 && policy!=1) ||
            !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(scale) || scale<=0)
            fail(ML_INVALID,"invalid QSA layer config");
        auto e=lookup(engines,engine);
        auto w=std::make_shared<QSALayer>(e,capacity,policy,epsilon,scale);
        const uint64_t widths[]={2560,2560,2560,2560,2560,6144,256,256,128,128};
        const uint64_t rows[]={12288,512,512,512,128,2560,0,0,0,0};
        for (int j=0;j<10;++j) {
            w->tensors[j]=lookup(uploads,tensors[j]);
            layer_shape(w->tensors[j],e,widths[j],rows[j],j<6);
        }
        DeviceGuard guard(e->device);
        for (auto& u:w->tensors) u->wait();
        w->allocate();
        auto h=new_handle(); qsa_layers.emplace(h,w); *out=h;
    });
}
int ml_qsa_layer_create_paged(ml_handle engine,uint64_t capacity,const ml_handle* tensors,
    int policy,float epsilon,float scale,uint64_t page_size,uint64_t slots,uint64_t host_budget,
    const char* format,int raw_diagnostics,ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null paged QSA output");
        *out=0;
        if (!tensors || !capacity || capacity>262144 || capacity%4 || !page_size || page_size>4096 ||
            !slots || slots>(capacity+page_size-1)/page_size || (policy!=0 && policy!=1) ||
            !std::isfinite(epsilon) || epsilon<=0 || !std::isfinite(scale) || scale<=0 ||
            !format || (std::strcmp(format,"reference_fp32") && std::strcmp(format,"strata_int8_per64_fp16")) ||
            (raw_diagnostics!=0 && raw_diagnostics!=1)) fail(ML_INVALID,"invalid paged QSA config/format");
        auto e=lookup(engines,engine);
        auto w=std::make_shared<QSALayer>(e,capacity,policy,epsilon,scale);
        const uint64_t widths[]={2560,2560,2560,2560,2560,6144,256,256,128,128};
        const uint64_t rows[]={12288,512,512,512,128,2560,0,0,0,0};
        for (int j=0;j<10;++j) { w->tensors[j]=lookup(uploads,tensors[j]); layer_shape(w->tensors[j],e,widths[j],rows[j],j<6); }
        w->paged=std::make_shared<QSAPagedState>(e,capacity,page_size,slots,host_budget,
            !std::strcmp(format,"strata_int8_per64_fp16"),raw_diagnostics!=0);
        w->bytes=8+2051*8+8+4*(640+capacity*65+13952+6144+512+6144+512+128+128+6144+6144+2560);
        DeviceGuard guard(e->device); for (auto& u:w->tensors) u->wait();
        e->admit(w->bytes+16+w->paged->stats.device_bytes);
        w->paged->allocate(); w->allocate();
        const auto h=new_handle(); qsa_layers.emplace(h,w); *out=h;
    });
}
int ml_qsa_layer_paged_stats(ml_handle id,ml_kvi8_stats* out) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!out || !w->paged || w->invalid) fail(ML_INVALID,"paged QSA stats unavailable");
        *out=w->paged->stats; out->device_bytes+=w->bytes; out->pending=w->paged->pending?1:0;
    });
}
int ml_qsa_layer_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(qsa_layers,id); qsa_layers.erase(id); } });
}
int ml_qsa_layer_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id); DeviceGuard guard(w->engine->device);
        w->invalid=true; w->output_valid=false;
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,w->engine->stream));
        if (w->paged) w->paged->reset();
        cuda_check(cudaStreamSynchronize(w->engine->stream)); w->length=0; w->invalid=false;
    });
}
int ml_qsa_layer_position(ml_handle id, uint64_t* length) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!length || w->invalid) fail(ML_INVALID,"QSA layer invalid; reset required");
        *length=w->length;
    });
}
int ml_qsa_layer_step(ml_handle id, ml_handle hidden, uint64_t position, uint64_t* committed) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!committed || w->invalid) fail(ML_INVALID,"QSA layer invalid; reset required");
        if (position!=w->length || position>=w->capacity) fail(ML_INVALID,"QSA layer position/capacity mismatch");
        auto h=lookup(uploads,hidden); layer_shape(h,w->engine,2560,0);
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        h->wait(); w->output_valid=false;
        qsa_layer_step_device(w,static_cast<const float*>(h->device),position,committed);
    });
}
int ml_qsa_layer_import(ml_handle id, uint64_t length, ml_handle raw, ml_handle keys,
                         ml_handle values, ml_handle pooled) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (w->invalid || w->length || length<4 || length>w->capacity)
            fail(ML_INVALID,"QSA layer import requires empty valid state and length in [4,capacity]");
        ml_handle handles[]={raw,keys,values,pooled};
        uint64_t widths[]={128,512,512,128}, rows[]={length,length,length,length/4};
        std::shared_ptr<UploadedTensor> u[4];
        for (int j=0;j<4;++j) { u[j]=lookup(uploads,handles[j]); layer_shape(u[j],w->engine,widths[j],rows[j]); }
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        for (auto& x:u) x->wait();
        cuda_check(cudaMemsetAsync(w->status,0,4,w->engine->stream));
        for (int j=0;j<4;++j) cuda_check(ml_qsa_layer_finite(w->engine->stream,
            static_cast<const float*>(u[j]->device),rows[j]*widths[j],w->status));
        w->check_status();
        w->invalid=true; w->output_valid=false;
        if (w->paged) {
            auto p=w->paged;
            for (uint64_t start=0;start<length;start+=p->page_size) p->reserve(start/p->page_size);
            for (uint64_t start=0;start<length;start+=p->page_size) {
                const uint64_t count=std::min(p->page_size,length-start);
                p->pack(static_cast<const float*>(u[1]->device)+start*512,
                    static_cast<const float*>(u[2]->device)+start*512,count,w->status);
                w->check_status();
                p->commit(start,count,static_cast<const float*>(u[0]->device)+start*128);
            }
            w->copy(w->raw,static_cast<const float*>(u[0]->device),128);
            for (uint64_t t=length-4;t<length;++t)
                w->copy(w->raw+128+(t%4)*128,static_cast<const float*>(u[0]->device)+t*128,128);
            w->copy(w->pooled,static_cast<const float*>(u[3]->device),(length/4)*128);
            cuda_check(cudaStreamSynchronize(w->engine->stream));
            p->stats.length=length; p->stats.populated_bytes=length*p->row_bytes;
        } else {
            float* dst[]={w->raw,w->keys,w->values,w->pooled};
            for (int j=0;j<4;++j) w->copy(dst[j],static_cast<const float*>(u[j]->device),rows[j]*widths[j]);
        }
        cuda_check(cudaStreamSynchronize(w->engine->stream)); w->length=length; w->invalid=false;
    });
}
int ml_qsa_layer_readback_for_test(ml_handle id, int which, void* dst, uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (w->invalid || which<0 || which>(w->paged?10:6) || !dst) fail(ML_INVALID,"QSA layer diagnostic invalid");
        if ((which==0 || which==5 || which==6 || which>=9) && !w->output_valid) fail(ML_INVALID,"QSA layer output unavailable");
        const void* src[]={w->output,w->raw,w->keys,w->values,w->pooled,w->ids,w->scores,w->raw,w->raw+128,w->q,w->gate};
        uint64_t sizes[]={2560,w->length*128,w->length*512,w->length*512,w->length/4*128,std::min(w->length,uint64_t(2051)),w->length,128,512,6144,6144};
        if (count<sizes[which]) fail(ML_INVALID,"QSA layer diagnostic capacity");
        DeviceGuard guard(w->engine->device);
        if (w->paged && which>=1 && which<=3) w->paged->read(w->length,which,static_cast<float*>(dst));
        else cuda_check(cudaMemcpy(dst,src[which],sizes[which]*(which==5?8:4),cudaMemcpyDeviceToHost));
    });
}
}

// Bounded prepared-history transport and diagnostics; no model computation.
extern "C" {
int ml_qsa_layer_import_chunk(ml_handle id,uint64_t position,ml_handle raw,
    ml_handle keys,ml_handle values,ml_handle pooled) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!w->paged || w->invalid || position!=w->length)
            fail(ML_INVALID,"paged QSA chunk requires valid state and exact position");
        auto r=lookup(uploads,raw);
        const uint64_t rows=r->descriptor.ggml_shape[1];
        if (!rows || rows>4096 || rows>w->capacity-position)
            fail(ML_CAPACITY,"paged QSA chunk exceeds 4096 rows or logical capacity");
        std::shared_ptr<UploadedTensor> u[4]={r,lookup(uploads,keys),lookup(uploads,values),nullptr};
        const uint64_t widths[]={128,512,512,128};
        for (int j=0;j<3;++j) layer_shape(u[j],w->engine,widths[j],rows);
        const uint64_t pools=(position+rows)/4-position/4;
        if (pools) { u[3]=lookup(uploads,pooled); layer_shape(u[3],w->engine,128,pools); }
        else if (pooled) fail(ML_INVALID,"paged QSA chunk without completed blocks requires zero pool handle");
        DeviceGuard guard(w->engine->device); StreamCompletion drain{w->engine->stream};
        for (auto x:u) if (x) x->wait();
        cuda_check(cudaMemsetAsync(w->status,0,4,w->engine->stream));
        for (int j=0;j<4;++j) if (u[j]) cuda_check(ml_qsa_layer_finite(w->engine->stream,
            static_cast<const float*>(u[j]->device),(j==3?pools:rows)*widths[j],w->status));
        w->check_status();
        w->invalid=true; w->output_valid=false;
        auto p=w->paged;
        const uint64_t end=position+rows;
        for (uint64_t page=position/p->page_size;page<=(end-1)/p->page_size;++page) p->reserve(page);
        for (uint64_t offset=0;offset<rows;) {
            const uint64_t count=std::min(rows-offset,p->page_size-(position+offset)%p->page_size);
            p->pack(static_cast<const float*>(u[1]->device)+offset*512,
                static_cast<const float*>(u[2]->device)+offset*512,count,w->status);
            w->check_status();
            p->commit(position+offset,count,static_cast<const float*>(u[0]->device)+offset*128);
            offset+=count;
        }
        if (!position) w->copy(w->raw,static_cast<const float*>(u[0]->device),128);
        const uint64_t begin=rows>4?end-4:position;
        for (uint64_t t=begin;t<end;++t)
            w->copy(w->raw+128+(t%4)*128,static_cast<const float*>(u[0]->device)+(t-position)*128,128);
        if (pools) w->copy(w->pooled+(position/4)*128,static_cast<const float*>(u[3]->device),pools*128);
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        p->stats.length=end; p->stats.populated_bytes=end*p->row_bytes;
        w->length=end; w->invalid=false;
    });
}
int ml_qsa_layer_paged_range_for_test(ml_handle id,int which,uint64_t start,uint64_t rows,
    void* destination,uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!w->paged || w->invalid || which<1 || which>4 || !destination)
            fail(ML_INVALID,"paged QSA range diagnostic invalid");
        const uint64_t length=which==4?w->length/4:w->length;
        const uint64_t width=(which==1 || which==4)?128:512;
        if (start>length || rows>length-start || count<rows*width)
            fail(ML_INVALID,"paged QSA range diagnostic bounds");
        DeviceGuard guard(w->engine->device);
        if (which==4) {
            if (rows) cuda_check(cudaMemcpy(destination,w->pooled+start*128,rows*512,cudaMemcpyDeviceToHost));
        } else w->paged->read(start+rows,which,static_cast<float*>(destination),start);
    });
}
}

// Private independent session/layer state. Public calls serialize on boundary's
// lock; retained owners and StreamCompletion form a synchronous page-use lease.
struct KVPages {
    std::shared_ptr<Engine> engine;
    uint64_t capacity, page_size, physical_pages, logical_pages, bytes;
    uint64_t length=0, generation=1;
    bool counted=false, invalid=false;
    void* device=nullptr;
    uint64_t *table=nullptr, *compact_n=nullptr;
    int64_t* compact_ids=nullptr;
    float *pages=nullptr, *scratch=nullptr, *remap_scratch=nullptr;
    std::unordered_map<uint64_t,uint64_t> mapping;
    KVPages(std::shared_ptr<Engine> e, uint64_t c, uint64_t p, uint64_t b)
        : engine(std::move(e)), capacity(c), page_size(p), physical_pages(b),
          logical_pages((c+p-1)/p),
          bytes(logical_pages*8+b*p*4096+2051*4096+8+2051*8+p*4096) {}
    ~KVPages() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (counted) engine->device_bytes-=bytes;
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void allocate() {
        engine->admit(bytes); cuda_check(cudaMalloc(&device,bytes));
        engine->record(bytes,0); counted=true;
        table=static_cast<uint64_t*>(device); compact_n=table+logical_pages;
        compact_ids=reinterpret_cast<int64_t*>(compact_n+1);
        pages=reinterpret_cast<float*>(compact_ids+2051);
        scratch=pages+physical_pages*page_size*1024;
        remap_scratch=scratch+2051*1024;
        cuda_check(cudaMemsetAsync(device,0,bytes,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
    }
    void check(uint64_t expected) {
        if (invalid || expected!=generation) fail(ML_INVALID,"paged KV stale generation or invalid state; reset required");
    }
    void publish(uint64_t logical, uint64_t slot) {
        uint64_t entry=slot+1;
        cuda_check(cudaMemcpyAsync(table+logical,&entry,8,cudaMemcpyHostToDevice,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
        mapping[logical]=slot;
    }
};
std::unordered_map<ml_handle,std::shared_ptr<KVPages>> kv_stores;
extern "C" {
int ml_kv_create(ml_handle engine, uint64_t capacity, uint64_t page_size,
                 uint64_t physical_pages, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null paged KV output");
        *out=0;
        if (!capacity || capacity>262144 || !page_size || page_size>4096 ||
            !physical_pages || physical_pages>(capacity+page_size-1)/page_size)
            fail(ML_INVALID,"invalid paged KV config");
        auto e=lookup(engines,engine); DeviceGuard guard(e->device);
        auto w=std::make_shared<KVPages>(e,capacity,page_size,physical_pages);
        w->allocate(); auto id=new_handle(); kv_stores.emplace(id,w); *out=id;
    });
}
int ml_kv_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(kv_stores,id); kv_stores.erase(id); } });
}
int ml_kv_state(ml_handle id, uint64_t* length, uint64_t* generation) noexcept {
    return boundary([&] {
        auto w=lookup(kv_stores,id);
        if (!length || !generation || w->invalid) fail(ML_INVALID,"invalid paged KV state");
        *length=w->length; *generation=w->generation;
    });
}
int ml_kv_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(kv_stores,id); DeviceGuard guard(w->engine->device);
        if (w->generation==UINT64_MAX) fail(ML_CAPACITY,"paged KV generation exhausted");
        w->invalid=true;
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        cuda_check(cudaMemsetAsync(w->table,0,w->logical_pages*8,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        w->mapping.clear(); w->length=0; ++w->generation; w->invalid=false;
    });
}
int ml_kv_append(ml_handle id, uint64_t generation, ml_handle keys, ml_handle values) noexcept {
    return boundary([&] {
        auto w=lookup(kv_stores,id); w->check(generation);
        auto k=lookup(uploads,keys), v=lookup(uploads,values);
        const uint64_t rows=k->descriptor.ggml_shape[2];
        for (auto u:{k,v}) {
            const auto& d=u->descriptor;
            if (u->engine!=w->engine || d.type_id!=0 || d.rank!=3 ||
                d.ggml_shape[0]!=256 || d.ggml_shape[1]!=2 || d.ggml_shape[2]!=rows || !rows)
                fail(ML_INVALID,"paged KV requires same-engine F32 [rows,2,256]");
        }
        if (rows>w->capacity-w->length || (w->length+rows+w->page_size-1)/w->page_size>w->physical_pages)
            fail(ML_CAPACITY,"paged KV physical page budget exhausted");
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        k->wait(); v->wait();
        // Validate before any store mutation; finite checking uses bounded scratch.
        int* status=reinterpret_cast<int*>(w->compact_n); int host=0;
        cuda_check(cudaMemsetAsync(status,0,4,w->engine->stream));
        for (auto u:{k,v}) cuda_check(ml_qsa_layer_finite(w->engine->stream,
            static_cast<const float*>(u->device),rows*512,status));
        cuda_check(cudaMemcpyAsync(&host,status,4,cudaMemcpyDeviceToHost,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        if (host) fail(ML_INVALID,"paged KV nonfinite source");
        w->invalid=true;
        for (uint64_t row=0; row<rows; ++row) {
            uint64_t token=w->length+row, logical=token/w->page_size;
            auto it=w->mapping.find(logical);
            if (it==w->mapping.end()) {
                uint64_t slot=0;
                for (;;++slot) {
                    bool used=false; for (const auto& entry:w->mapping) used|=entry.second==slot;
                    if (!used) break;
                }
                w->publish(logical,slot); it=w->mapping.find(logical);
            }
            float* dst=w->pages+(it->second*w->page_size+token%w->page_size)*1024;
            cuda_check(cudaMemcpyAsync(dst,static_cast<const float*>(k->device)+row*512,2048,cudaMemcpyDeviceToDevice,w->engine->stream));
            cuda_check(cudaMemcpyAsync(dst+512,static_cast<const float*>(v->device)+row*512,2048,cudaMemcpyDeviceToDevice,w->engine->stream));
        }
        cuda_check(cudaStreamSynchronize(w->engine->stream)); w->length+=rows; w->invalid=false;
    });
}
int ml_kv_remap(ml_handle id, uint64_t generation, uint64_t logical, uint64_t slot) noexcept {
    return boundary([&] {
        auto w=lookup(kv_stores,id); w->check(generation);
        auto it=w->mapping.find(logical);
        if (it==w->mapping.end() || slot>=w->physical_pages) fail(ML_INVALID,"invalid paged KV remap");
        const uint64_t old=it->second;
        if (old==slot) return;
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        cuda_check(cudaStreamSynchronize(w->engine->stream)); w->invalid=true;
        uint64_t other=UINT64_MAX;
        for (const auto& entry:w->mapping) if (entry.second==slot) other=entry.first;
        const uint64_t floats=w->page_size*1024;
        auto copy=[&](float* d,const float* s) { cuda_check(cudaMemcpyAsync(d,s,floats*4,cudaMemcpyDeviceToDevice,w->engine->stream)); };
        copy(w->remap_scratch,w->pages+old*floats);
        if (other!=UINT64_MAX) copy(w->pages+old*floats,w->pages+slot*floats);
        copy(w->pages+slot*floats,w->remap_scratch);
        if (other!=UINT64_MAX) w->publish(other,old);
        w->publish(logical,slot); w->invalid=false;
    });
}
int ml_kv_attention(ml_handle id, uint64_t generation, ml_handle attention,
    ml_handle query, ml_handle gate, ml_handle key, ml_handle value, float scale) noexcept {
    return boundary([&] {
        auto w=lookup(kv_stores,id); w->check(generation);
        auto a=lookup(qsa_attentions,attention); a->valid=false;
        auto s=a->selection;
        if (s->engine!=w->engine || !s->valid || s->n!=w->length+1 || s->n>w->capacity ||
            !std::isfinite(scale) || scale<=0) fail(ML_INVALID,"paged KV selection/length/scale mismatch");
        ml_handle handles[]={query,gate,key,value}; std::shared_ptr<UploadedTensor> u[4];
        for (int j=0;j<4;++j) {
            u[j]=lookup(uploads,handles[j]); const auto& d=u[j]->descriptor;
            if (u[j]->engine!=w->engine || d.type_id!=0 || d.rank!=2 ||
                d.ggml_shape[0]!=256 || d.ggml_shape[1]!=(j<2?24u:2u))
                fail(ML_INVALID,"paged KV attention exact F32 shapes required");
        }
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        for (auto x:u) x->wait();
        auto ptr=[&](int j) { return static_cast<const float*>(u[j]->device); };
        cuda_check(cudaMemsetAsync(a->status,0,4,w->engine->stream));
        for (int j=0;j<4;++j) cuda_check(ml_qsa_layer_finite(w->engine->stream,
            ptr(j),j<2?6144:512,a->status));
        cuda_check(ml_kv_selected_launch(w->engine->stream,s->parameter,s->ids,s->n,w->table,
            w->page_size,w->physical_pages,w->pages,ptr(0),ptr(1),ptr(2),ptr(3),
            w->scratch,w->compact_n,w->compact_ids,a->contexts,a->gated,a->status,scale));
        cuda_check(cudaMemcpyAsync(a->host,a->status,4,cudaMemcpyDeviceToHost,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        if (*a->host) fail(ML_INVALID,"paged KV attention nonfinite source or addressing");
        a->valid=true;
    });
}
int ml_kv_readback_for_test(ml_handle id, uint64_t generation, float* output, uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(kv_stores,id); w->check(generation);
        if (!output || count<w->length*1024) fail(ML_INVALID,"paged KV diagnostic capacity");
        DeviceGuard guard(w->engine->device);
        for (uint64_t row=0;row<w->length;++row) {
            const auto slot=w->mapping.at(row/w->page_size);
            cuda_check(cudaMemcpy(output+row*1024,w->pages+(slot*w->page_size+row%w->page_size)*1024,4096,cudaMemcpyDeviceToHost));
        }
    });
}
}

// Explicit INT8/pageable-host policy, independent of the FP32 paged owner.
struct KVStreamed {
    std::shared_ptr<Engine> engine;
    uint64_t capacity, page_size, slots, logical_pages, page_bytes, backing_bytes;
    ml_kvi8_stats stats{};
    void* device=nullptr;
    unsigned char *pages=nullptr, *stage=nullptr;
    float* scratch=nullptr;
    uint64_t *table=nullptr, *compact_n=nullptr;
    int64_t* compact_ids=nullptr;
    int* status=nullptr;
    KVPageRequests* requests=nullptr;
    KVPageRequests host_requests{};
    std::unique_ptr<unsigned char[]> backing;
    std::unique_ptr<uint64_t[]> host_table, slot_page, slot_generation;
    cudaEvent_t completion=nullptr;
    bool counted=false, invalid=false, pending=false;
    KVStreamed(std::shared_ptr<Engine> e,uint64_t c,uint64_t p,uint64_t s,uint64_t budget)
        : engine(std::move(e)),capacity(c),page_size(p),slots(s),logical_pages((c+p-1)/p),
          page_bytes(p*ML_KV_I8_ROW),backing_bytes(logical_pages*page_bytes) {
        stats.generation=1; stats.host_budget=budget;
        stats.host_bytes=backing_bytes+sizeof(KVPageRequests)+8*(logical_pages+2*slots);
        stats.device_bytes=8*logical_pages+sizeof(KVPageRequests)+8+2051*8+8+
            2051*4096+(slots+1)*page_bytes;
    }
    ~KVStreamed() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (completion) (void)cudaEventDestroy(completion);
        if (counted) engine->device_bytes-=stats.device_bytes;
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void allocate() {
        if (stats.host_bytes>stats.host_budget) fail(ML_CAPACITY,"streamed KV ordinary-RAM budget exhausted");
        engine->admit(stats.device_bytes);
        backing=std::make_unique<unsigned char[]>(backing_bytes);
        host_table=std::make_unique<uint64_t[]>(logical_pages);
        slot_page=std::make_unique<uint64_t[]>(slots);
        slot_generation=std::make_unique<uint64_t[]>(slots);
        std::fill_n(slot_page.get(),slots,UINT64_MAX);
        cuda_check(cudaEventCreateWithFlags(&completion,cudaEventDisableTiming));
        cuda_check(cudaMalloc(&device,stats.device_bytes));
        engine->record(stats.device_bytes,0); counted=true;
        auto* cursor=static_cast<unsigned char*>(device);
        table=reinterpret_cast<uint64_t*>(cursor); cursor+=logical_pages*8;
        requests=reinterpret_cast<KVPageRequests*>(cursor); cursor+=sizeof(KVPageRequests);
        compact_n=reinterpret_cast<uint64_t*>(cursor); cursor+=8;
        compact_ids=reinterpret_cast<int64_t*>(cursor); cursor+=2051*8;
        status=reinterpret_cast<int*>(cursor); cursor+=8;
        scratch=reinterpret_cast<float*>(cursor); cursor+=2051*4096;
        pages=cursor; cursor+=slots*page_bytes; stage=cursor;
        cuda_check(cudaMemsetAsync(device,0,stats.device_bytes,engine->stream)); finish();
    }
    void wait() {
        if (pending) { cuda_check(cudaEventSynchronize(completion)); pending=false; }
    }
    void finish() {
        cuda_check(cudaEventRecord(completion,engine->stream)); pending=true; wait();
    }
    void check(uint64_t generation) {
        if (invalid || generation!=stats.generation)
            fail(ML_INVALID,"streamed KV stale generation or invalid state; reset required");
    }
    void check_status() {
        int host=0;
        cuda_check(cudaMemcpyAsync(&host,status,4,cudaMemcpyDeviceToHost,engine->stream)); finish();
        if (host) fail(ML_INVALID,"streamed KV nonfinite source or nonrepresentable FP16 scale");
    }
    void update(uint64_t logical,uint64_t entry) {
        host_table[logical]=entry;
        cuda_check(cudaMemcpyAsync(table+logical,host_table.get()+logical,8,cudaMemcpyHostToDevice,engine->stream));
                finish(); // Retire the borrowed host metadata before it can be overwritten.
    }
    void remove(uint64_t slot) {
        if (slot_page[slot]!=UINT64_MAX) {
            update(slot_page[slot],0); slot_page[slot]=UINT64_MAX; slot_generation[slot]=0;
            --stats.resident_pages;
        }
    }
    void load(uint64_t logical,uint64_t slot) {
        wait();
        remove(slot);
        const uint64_t rows=std::min(page_size,stats.length-logical*page_size);
        cuda_check(cudaMemcpyAsync(pages+slot*page_bytes,backing.get()+logical*page_bytes,
            rows*ML_KV_I8_ROW,cudaMemcpyHostToDevice,engine->stream));
        // Publication and use are ordered after the copy on the same stream.
        update(logical,slot+1); slot_page[slot]=logical; slot_generation[slot]=stats.generation;
        ++stats.resident_pages; stats.h2d_bytes+=rows*ML_KV_I8_ROW;
    }
};
std::unordered_map<ml_handle,std::shared_ptr<KVStreamed>> kv_streamed;
extern "C" {
int ml_kvi8_create(ml_handle engine,uint64_t capacity,uint64_t page_size,uint64_t slots,
    uint64_t host_budget,const char* policy,ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null streamed KV output");
        *out=0;
        if (!policy || std::strcmp(policy,"strata_int8_per64_fp16") || !capacity || capacity>262144 ||
            !page_size || page_size>4096 || !slots || slots>(capacity+page_size-1)/page_size)
            fail(ML_INVALID,"invalid streamed KV config/policy");
        auto e=lookup(engines,engine); DeviceGuard guard(e->device);
        auto w=std::make_shared<KVStreamed>(e,capacity,page_size,slots,host_budget);
        w->allocate(); const auto id=new_handle(); kv_streamed.emplace(id,w); *out=id;
    });
}
int ml_kvi8_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(kv_streamed,id); kv_streamed.erase(id); } });
}
int ml_kvi8_stats_get(ml_handle id,ml_kvi8_stats* out) noexcept {
    return boundary([&] {
        auto w=lookup(kv_streamed,id);
        if (!out || w->invalid) fail(ML_INVALID,"invalid streamed KV stats");
        w->stats.pending=w->pending?1:0;
        *out=w->stats;
    });
}
int ml_kvi8_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(kv_streamed,id); DeviceGuard guard(w->engine->device);
        StreamCompletion drain{w->engine->stream}; w->wait();
        if (w->stats.generation==UINT64_MAX) fail(ML_CAPACITY,"streamed KV generation exhausted");
        w->invalid=true;
        cuda_check(cudaMemsetAsync(w->table,0,w->logical_pages*8,w->engine->stream)); w->finish();
        std::fill_n(w->host_table.get(),w->logical_pages,0);
        std::fill_n(w->slot_page.get(),w->slots,UINT64_MAX);
        std::fill_n(w->slot_generation.get(),w->slots,0);
        ++w->stats.generation; w->stats.length=w->stats.populated_bytes=w->stats.resident_pages=0;
        w->stats.hits=w->stats.misses=w->stats.h2d_bytes=w->stats.metadata_bytes=0;
        w->invalid=false;
    });
}
int ml_kvi8_append(ml_handle id,uint64_t generation,ml_handle keys,ml_handle values) noexcept {
    return boundary([&] {
        auto w=lookup(kv_streamed,id); w->check(generation);
        auto k=lookup(uploads,keys),v=lookup(uploads,values);
        const uint64_t rows=k->descriptor.ggml_shape[2];
        for (auto u:{k,v}) {
            const auto& d=u->descriptor;
            if (u->engine!=w->engine || d.type_id!=0 || d.rank!=3 || d.ggml_shape[0]!=256 ||
                d.ggml_shape[1]!=2 || d.ggml_shape[2]!=rows || !rows)
                fail(ML_INVALID,"streamed KV requires same-engine F32 [rows,2,256]");
        }
        if (rows>w->page_size || rows>w->capacity-w->stats.length)
            fail(ML_CAPACITY,"streamed KV append exceeds staging page or logical capacity");
        DeviceGuard guard(w->engine->device); StreamCompletion drain{w->engine->stream};
        w->wait(); k->wait(); v->wait();
        cuda_check(cudaMemsetAsync(w->status,0,4,w->engine->stream));
        cuda_check(ml_kvi8_quantize(w->engine->stream,static_cast<const float*>(k->device),
            static_cast<const float*>(v->device),rows,w->stage,w->status));
        w->check_status();
        w->invalid=true;
        cuda_check(cudaMemcpyAsync(w->backing.get()+w->stats.length*ML_KV_I8_ROW,w->stage,
            rows*ML_KV_I8_ROW,cudaMemcpyDeviceToHost,w->engine->stream));
        const uint64_t first=w->stats.length/w->page_size,last=(w->stats.length+rows-1)/w->page_size;
        for (uint64_t page=first;page<=last;++page)
            if (w->host_table[page]) w->remove(w->host_table[page]-1);
        w->finish(); w->stats.length+=rows; w->stats.populated_bytes=w->stats.length*ML_KV_I8_ROW;
        w->invalid=false;
    });
}
int ml_kvi8_remap(ml_handle id,uint64_t generation,uint64_t logical,uint64_t slot) noexcept {
    return boundary([&] {
        auto w=lookup(kv_streamed,id); w->check(generation);
        if (logical>=(w->stats.length+w->page_size-1)/w->page_size || slot>=w->slots)
            fail(ML_INVALID,"invalid streamed KV remap");
        DeviceGuard guard(w->engine->device); StreamCompletion drain{w->engine->stream}; w->wait();
        if (w->host_table[logical]==slot+1) return;
        w->invalid=true;
        if (w->host_table[logical]) w->remove(w->host_table[logical]-1);
        w->load(logical,slot); w->finish(); w->invalid=false;
    });
}
int ml_kvi8_attention(ml_handle id,uint64_t generation,ml_handle attention,
    ml_handle query,ml_handle gate,ml_handle key,ml_handle value,float scale) noexcept {
    return boundary([&] {
        auto w=lookup(kv_streamed,id); w->check(generation);
        auto a=lookup(qsa_attentions,attention); a->valid=false; auto s=a->selection;
        if (s->engine!=w->engine || !s->valid || s->n!=w->stats.length+1 || s->n>w->capacity ||
            !std::isfinite(scale) || scale<=0) fail(ML_INVALID,"streamed KV selection/length/scale mismatch");
        ml_handle handles[]={query,gate,key,value}; std::shared_ptr<UploadedTensor> u[4];
        for (int j=0;j<4;++j) {
            u[j]=lookup(uploads,handles[j]); const auto& d=u[j]->descriptor;
            if (u[j]->engine!=w->engine || d.type_id!=0 || d.rank!=2 || d.ggml_shape[0]!=256 ||
                d.ggml_shape[1]!=(j<2?24u:2u)) fail(ML_INVALID,"streamed KV attention exact F32 shapes required");
        }
        DeviceGuard guard(w->engine->device); StreamCompletion drain{w->engine->stream}; w->wait();
        for (auto x:u) x->wait();
        auto ptr=[&](int j) { return static_cast<const float*>(u[j]->device); };
        cuda_check(cudaMemsetAsync(w->status,0,4,w->engine->stream));
        for (int j=0;j<2;++j) cuda_check(ml_qsa_layer_finite(w->engine->stream,ptr(j),6144,w->status));
        cuda_check(ml_kvi8_quantize(w->engine->stream,ptr(2),ptr(3),1,w->stage,w->status));
        w->check_status();
        cuda_check(cudaMemsetAsync(w->requests,0,sizeof(KVPageRequests),w->engine->stream));
        cuda_check(ml_kvi8_requests(w->engine->stream,s->parameter,s->ids,w->table,w->page_size,w->slots,w->requests));
        cuda_check(cudaMemcpyAsync(&w->host_requests,w->requests,sizeof(KVPageRequests),cudaMemcpyDeviceToHost,w->engine->stream));
        w->finish(); w->stats.metadata_bytes+=sizeof(KVPageRequests);
        const auto& request=w->host_requests;
        if (request.error) fail(ML_INVALID,"streamed KV invalid device page request");
        // Preflight before any eviction or payload transfer; no full-resident fallback.
        if (request.count>w->slots) fail(ML_CAPACITY,"streamed KV selected-page working set exceeds GPU slots");
        std::unordered_set<uint64_t> protected_pages;
        for (uint64_t j=0;j<request.count;++j) protected_pages.insert(request.pages[j]);
        w->invalid=true;
        for (uint64_t j=0;j<request.count;++j) {
            const auto entry=w->host_table[request.pages[j]];
            if (entry && w->slot_generation[entry-1]!=generation) fail(ML_INVALID,"streamed KV slot generation mismatch");
        }
        for (uint64_t j=0;j<request.misses;++j) {
            uint64_t slot=0;
            for (;slot<w->slots;++slot)
                if (w->slot_page[slot]==UINT64_MAX || !protected_pages.count(w->slot_page[slot])) break;
            if (slot==w->slots) fail(ML_INTERNAL,"streamed KV protected working set allocation");
            w->load(request.missing[j],slot);
        }
        w->stats.hits+=request.count-request.misses; w->stats.misses+=request.misses;
        cuda_check(cudaMemsetAsync(a->status,0,4,w->engine->stream));
        cuda_check(ml_kvi8_attention_launch(w->engine->stream,s->parameter,s->ids,s->n,w->table,
            w->page_size,w->slots,w->pages,w->stage,ptr(0),ptr(1),w->scratch,w->compact_n,
            w->compact_ids,a->contexts,a->gated,a->status,scale));
        cuda_check(cudaMemcpyAsync(a->host,a->status,4,cudaMemcpyDeviceToHost,w->engine->stream)); w->finish();
        w->invalid=false;
        if (*a->host) fail(ML_INVALID,"streamed KV nonfinite attention or addressing");
        a->valid=true;
    });
}
int ml_kvi8_readback_for_test(ml_handle id,uint64_t generation,uint64_t start,uint64_t rows,
    void* output,uint64_t bytes) noexcept {
    return boundary([&] {
        auto w=lookup(kv_streamed,id); w->check(generation);
        if (start>w->stats.length || rows>w->stats.length-start || !output || bytes<rows*ML_KV_I8_ROW)
            fail(ML_INVALID,"streamed KV packed diagnostic bounds");
        DeviceGuard guard(w->engine->device); w->wait();
        std::memcpy(output,w->backing.get()+start*ML_KV_I8_ROW,rows*ML_KV_I8_ROW);
    });
}
}

// Private HC ownership/transactions; included after runtime lookup/boundary helpers.
struct HCWeights {
    std::shared_ptr<Engine> engine;
    std::shared_ptr<UploadedTensor> uploads[4]; // converted gamma, down, up, inject
    const float* values[4]{};
    int d, rank;
    float epsilon;
    uint64_t bytes=4;
    void* device=nullptr;
    int* host=nullptr;
    int* status=nullptr;
    bool counted=false;
    HCWeights(std::shared_ptr<Engine> e, int width, int r, float eps)
        : engine(std::move(e)), d(width), rank(r), epsilon(eps) {}
    ~HCWeights() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=4; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
};
struct HCState {
    std::shared_ptr<Engine> engine;
    std::shared_ptr<HCWeights> pending;
    int d, rank_capacity;
    uint64_t bytes, ticket=0;
    void* device=nullptr;
    int *host=nullptr, *status=nullptr;
    float *streams=nullptr, *staged=nullptr, *normalized=nullptr, *low=nullptr;
    float *hidden=nullptr, *gains=nullptr;
    bool counted=false, invalid=false, initialized=false, collapsed=false, hidden_valid=false;
    HCState(std::shared_ptr<Engine> e, int width, int rank)
        : engine(std::move(e)), d(width), rank_capacity(rank), bytes((13ull*width+rank+5)*4) {}
    ~HCState() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=4; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void begin() {
        if (invalid) fail(ML_INVALID,"HC state invalid; reset required");
        // Every failed mutating attempt poisons the state, including metadata and
        // stale-ticket errors, so old gains can never silently resume a transaction.
        invalid=true; hidden_valid=false;
    }
    void ready() const {
        if (!initialized || collapsed) fail(ML_INVALID,"HC state must be initialized and not collapsed");
    }
    void check_status() {
        cuda_check(cudaMemcpyAsync(host,status,4,cudaMemcpyDeviceToHost,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
        if (*host) fail(ML_INVALID,"HC nonfinite source or arithmetic; reset required");
    }
};
// Declared after upload/engine registries: owners release before their dependencies.
std::unordered_map<ml_handle,std::shared_ptr<HCWeights>> hc_weights;
std::unordered_map<ml_handle,std::shared_ptr<HCState>> hc_states;
void hc_shape(const std::shared_ptr<UploadedTensor>& u, const std::shared_ptr<Engine>& e,
              uint64_t width, uint64_t rows, bool dense=false) {
    const auto& desc=u->descriptor;
    if (u->engine!=e) fail(ML_INVALID,"HC cross-engine input");
    if (desc.rank!=(rows?2u:1u) || desc.ggml_shape[0]!=width ||
        (rows && desc.ggml_shape[1]!=rows)) fail(ML_INVALID,"HC exact input shape required");
    if (desc.type_id!=0 && (!dense || desc.type_id!=30)) fail(ML_INVALID,"HC supports F32/BF16 weights and F32 activations only");
}
void hc_read_device(const std::shared_ptr<HCState>& w, const std::shared_ptr<HCWeights>& v,
                    uint64_t* snapshot) {
        auto stream=w->engine->stream;
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        cuda_check(ml_hc_read_launch(stream,w->d,v->rank,v->epsilon,w->streams,
            v->values[0],v->values[1],v->values[2],v->values[3],w->normalized,w->low,w->hidden,w->gains,w->status));
        w->check_status();
        w->ticket=new_handle(); w->pending=v; w->hidden_valid=true; w->invalid=false; *snapshot=w->ticket;
}
void hc_write_device(const std::shared_ptr<HCState>& w, const float* output) {
        auto stream=w->engine->stream;
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        cuda_check(ml_hc_write_launch(stream,w->d,w->streams,output,w->gains,w->staged,w->status));
        w->check_status();
        // Preserve stable stream address; no committed write before finite gating.
        cuda_check(cudaMemcpyAsync(w->streams,w->staged,uint64_t(w->d)*16,cudaMemcpyDeviceToDevice,stream));
        cuda_check(cudaStreamSynchronize(stream));
        w->pending.reset(); w->ticket=0; w->invalid=false;
}
extern "C" {
int ml_hc_weights_create(ml_handle engine, ml_handle gamma, ml_handle down,
    ml_handle up, ml_handle inject, float epsilon, int arithmetic, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null HC weight output");
        *out=0;
        if (arithmetic!=0 || !std::isfinite(epsilon) || epsilon<=0)
            fail(ML_INVALID,"HC requires explicit reference_fp32 and finite positive epsilon");
        auto e=lookup(engines,engine);
        auto dn=lookup(uploads,down);
        const auto& desc=dn->descriptor;
        if (desc.rank!=2 || !desc.ggml_shape[0] || desc.ggml_shape[0]%4 ||
            desc.ggml_shape[0]>uint64_t(INT_MAX) || !desc.ggml_shape[1] ||
            desc.ggml_shape[1]>uint64_t(INT_MAX-4)) fail(ML_INVALID,"HC down shape/index limits");
        auto w=std::make_shared<HCWeights>(e,int(desc.ggml_shape[0]/4),int(desc.ggml_shape[1]),epsilon);
        w->uploads[0]=lookup(uploads,gamma); w->uploads[1]=dn; w->uploads[2]=lookup(uploads,up);
        if (inject) w->uploads[3]=lookup(uploads,inject);
        auto g=w->uploads[0];
        // The Python architecture authority permits flat or grouped converted gamma.
        if (g->descriptor.rank==1) hc_shape(g,e,4ull*w->d,0,true);
        else hc_shape(g,e,w->d,4,true);
        hc_shape(dn,e,4ull*w->d,w->rank,true);
        hc_shape(w->uploads[2],e,w->rank,4ull*w->d,true);
        if (inject) hc_shape(w->uploads[3],e,4ull*w->d,4,true);
        for (auto& u:w->uploads) if (u && u->descriptor.type_id==30) w->bytes+=u->descriptor.nbytes*2;
        e->admit(w->bytes+4);
        DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        for (auto& u:w->uploads) if (u) u->wait();
        cuda_check(cudaMalloc(&w->device,w->bytes));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&w->host),4));
        e->record(w->bytes,4); w->counted=true;
        w->status=static_cast<int*>(w->device);
        float* cursor=reinterpret_cast<float*>(w->status+1);
        cuda_check(cudaMemsetAsync(w->status,0,4,e->stream));
        for (int j=0;j<4;++j) if (auto u=w->uploads[j]) {
            uint64_t count=u->descriptor.nbytes/(u->descriptor.type_id==30?2:4);
            if (u->descriptor.type_id==30) {
                w->values[j]=cursor;
                cuda_check(ml_hc_decode_launch(e->stream,u->device,30,count,cursor,w->status));
                cursor+=count;
            } else {
                w->values[j]=static_cast<const float*>(u->device);
                cuda_check(ml_qsa_layer_finite(e->stream,w->values[j],count,w->status));
            }
        }
        cuda_check(cudaMemcpyAsync(w->host,w->status,4,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaStreamSynchronize(e->stream));
        if (*w->host) fail(ML_INVALID,"HC nonfinite weight");
        auto handle=new_handle(); hc_weights.emplace(handle,w); *out=handle;
    });
}
int ml_hc_weights_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(hc_weights,id); hc_weights.erase(id); } });
}
int ml_hc_create(ml_handle engine, int d, int rank_capacity, ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null HC state output");
        *out=0;
        if (d<=0 || d>INT_MAX/4 || rank_capacity<=0 || rank_capacity>INT_MAX-4)
            fail(ML_INVALID,"HC state dimension/index limits");
        auto e=lookup(engines,engine);
        auto w=std::make_shared<HCState>(e,d,rank_capacity);
        e->admit(w->bytes+4); DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(&w->device,w->bytes));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&w->host),4));
        e->record(w->bytes,4); w->counted=true;
        float* cursor=static_cast<float*>(w->device);
        auto take=[&](uint64_t n) { float* result=cursor; cursor+=n; return result; };
        w->streams=take(4ull*d); w->staged=take(4ull*d); w->normalized=take(4ull*d);
        w->low=take(rank_capacity); w->hidden=take(d); w->gains=take(4);
        w->status=reinterpret_cast<int*>(cursor);
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,e->stream));
        cuda_check(cudaStreamSynchronize(e->stream));
        auto handle=new_handle(); hc_states.emplace(handle,w); *out=handle;
    });
}
int ml_hc_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(hc_states,id); hc_states.erase(id); } });
}
int ml_hc_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(hc_states,id); DeviceGuard guard(w->engine->device);
        w->invalid=true; w->hidden_valid=false;
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        w->pending.reset(); w->ticket=0; w->initialized=false; w->collapsed=false; w->invalid=false;
    });
}
int ml_hc_initialize(ml_handle id, ml_handle input) noexcept {
    return boundary([&] {
        auto w=lookup(hc_states,id); w->begin();
        if (w->initialized) fail(ML_INVALID,"HC initialize requires reset/empty state");
        auto u=lookup(uploads,input);
        const bool embedding=u->descriptor.rank==1;
        hc_shape(u,w->engine,w->d,embedding?0:4);
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        u->wait(); auto stream=w->engine->stream;
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        cuda_check(ml_qsa_layer_finite(stream,static_cast<const float*>(u->device),uint64_t(w->d)*(embedding?1:4),w->status));
        w->check_status();
        if (embedding) for (int c=0;c<4;++c)
            cuda_check(cudaMemcpyAsync(w->streams+uint64_t(c)*w->d,u->device,uint64_t(w->d)*4,cudaMemcpyDeviceToDevice,stream));
        else cuda_check(cudaMemcpyAsync(w->streams,u->device,uint64_t(w->d)*16,cudaMemcpyDeviceToDevice,stream));
        cuda_check(cudaStreamSynchronize(stream)); w->initialized=true; w->invalid=false;
    });
}
int ml_hc_read(ml_handle id, ml_handle weights, uint64_t* snapshot) noexcept {
    return boundary([&] {
        auto w=lookup(hc_states,id); w->begin();
        if (!snapshot) fail(ML_INVALID,"null HC snapshot output");
        *snapshot=0; w->ready();
        if (w->pending) fail(ML_INVALID,"HC pending read must be written before another read");
        auto v=lookup(hc_weights,weights);
        if (v->engine!=w->engine) fail(ML_INVALID,"HC cross-engine weights");
        if (v->d!=w->d || v->rank>w->rank_capacity || !v->values[3])
            fail(ML_INVALID,"HC read weight shape/rank/injection mismatch");
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        hc_read_device(w,v,snapshot);
    });
}
int ml_hc_write(ml_handle id, uint64_t snapshot, ml_handle output) noexcept {
    return boundary([&] {
        auto w=lookup(hc_states,id); w->begin(); w->ready();
        if (!w->pending || !snapshot || snapshot!=w->ticket)
            fail(ML_INVALID,"HC snapshot identity/order mismatch; reset required");
        auto u=lookup(uploads,output); hc_shape(u,w->engine,w->d,0);
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        u->wait();
        hc_write_device(w,static_cast<const float*>(u->device));
    });
}
int ml_hc_collapse(ml_handle id, ml_handle weights) noexcept {
    return boundary([&] {
        auto w=lookup(hc_states,id); w->begin(); w->ready();
        if (w->pending) fail(ML_INVALID,"HC pending read must be written before collapse");
        auto v=lookup(hc_weights,weights);
        if (v->engine!=w->engine) fail(ML_INVALID,"HC cross-engine weights");
        if (v->d!=w->d || v->rank>w->rank_capacity || v->values[3])
            fail(ML_INVALID,"HC collapse requires read-only weights with matching shape/rank");
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        auto stream=w->engine->stream;
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        cuda_check(ml_hc_collapse_launch(stream,w->d,v->rank,v->epsilon,w->streams,
            v->values[0],v->values[1],v->values[2],w->normalized,w->low,w->hidden,w->status));
        w->check_status(); w->collapsed=true; w->hidden_valid=true; w->invalid=false;
    });
}
int ml_hc_readback_for_test(ml_handle id, int which, float* dst, uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(hc_states,id);
        if (w->invalid || !w->initialized || which<0 || which>2 || !dst)
            fail(ML_INVALID,"HC diagnostic invalid/uninitialized; reset may be required");
        if ((which==1 && !w->hidden_valid) || (which==2 && !w->pending))
            fail(ML_INVALID,"HC diagnostic snapshot unavailable");
        const float* src[]={w->streams,w->hidden,w->gains};
        uint64_t sizes[]={4ull*w->d,uint64_t(w->d),4};
        if (count<sizes[which]) fail(ML_INVALID,"HC diagnostic capacity");
        DeviceGuard guard(w->engine->device);
        cuda_check(cudaMemcpyAsync(dst,src[which],sizes[which]*4,cudaMemcpyDeviceToHost,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream));
    });
}
}

// Private bounded recurrent owner; parent runtime owns synchronization/errors.
struct RecurrentLayer {
    std::shared_ptr<Engine> engine;
    std::shared_ptr<UploadedTensor> tensors[9]; // QKV,Z,beta,alpha,O,conv,gamma,A,dt
    int d,hk,hv,dim,taps;
    float epsilon;
    uint64_t capacity,length=0,bytes,channels,inner,history_count,memory_count;
    bool counted=false,invalid=false,output_valid=false;
    void* device=nullptr;
    int *host=nullptr,*status=nullptr;
    float *history=nullptr,*memory=nullptr,*staged_history=nullptr,*staged_memory=nullptr;
    float *projected[4]{},*convolved=nullptr,*beta=nullptr,*log_decay=nullptr;
    float *readout=nullptr,*gated=nullptr,*output=nullptr;
    RecurrentLayer(std::shared_ptr<Engine> e,int width,int kh,int vh,int head,int k,
                   uint64_t cap,float eps): engine(std::move(e)),d(width),hk(kh),hv(vh),
        dim(head),taps(k),epsilon(eps),capacity(cap),channels(uint64_t(2*kh+vh)*head),
        inner(uint64_t(vh)*head),history_count(uint64_t(k-1)*channels),
        memory_count(uint64_t(vh)*head*head) {
        bytes=4*(2*history_count+2*memory_count+2*channels+3*inner+4*hv+d+1);
    }
    ~RecurrentLayer() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=4; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
    void copy(float* dst,const float* src,uint64_t n) {
        if (n) cuda_check(cudaMemcpyAsync(dst,src,n*4,cudaMemcpyDeviceToDevice,engine->stream));
    }
    void finite(const float* src,uint64_t n) {
        if (n) cuda_check(ml_qsa_layer_finite(engine->stream,src,n,status));
    }
    void check_status() {
        cuda_check(cudaMemcpyAsync(host,status,4,cudaMemcpyDeviceToHost,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
        if (*host) fail(ML_INVALID,"recurrent nonfinite source/arithmetic or nonnegative converted A");
    }
};
std::unordered_map<ml_handle,std::shared_ptr<RecurrentLayer>> recurrent_layers;
void recurrent_shape(const std::shared_ptr<UploadedTensor>& u,const std::shared_ptr<Engine>& e,
                     uint64_t width,uint64_t rows,bool projection=false) {
    const auto& d=u->descriptor;
    if (u->engine!=e) fail(ML_INVALID,"recurrent cross-engine input");
    if (d.rank!=(rows?2u:1u) || d.ggml_shape[0]!=width || (rows && d.ggml_shape[1]!=rows))
        fail(ML_INVALID,"recurrent exact input shape required");
    if (d.type_id!=0 && (!projection || (d.type_id!=30 && d.type_id!=12 &&
        d.type_id!=13 && d.type_id!=14 && d.type_id!=23)))
        fail(ML_INVALID,"recurrent unsupported format; reference_fp32 F32 controls/activations required");
    if (projection && d.type_id!=0 && d.type_id!=30 && width%256)
        fail(ML_INVALID,"recurrent packed projection block alignment");
}
void recurrent_step_device(const std::shared_ptr<RecurrentLayer>& w, const float* hidden,
                           uint64_t position, uint64_t* committed) {
        auto stream=w->engine->stream;
        try {
            cuda_check(cudaMemsetAsync(w->status,0,4,stream));
            w->copy(w->staged_history,w->history,w->history_count);
            w->copy(w->staged_memory,w->memory,w->memory_count);
            for (int j=0;j<4;++j) {
                auto u=w->tensors[j]; const auto& desc=u->descriptor;
                cuda_check(ml_qsa_projection_launch(stream,u->device,desc.type_id,desc.ggml_shape[1],
                    w->d,desc.row_bytes,hidden,w->projected[j],w->status));
            }
            auto control=[&](int j) { return static_cast<const float*>(w->tensors[j]->device); };
            cuda_check(ml_recurrent_launch(stream,w->hk,w->hv,w->dim,w->taps,w->epsilon,
                w->projected[0],w->projected[1],w->projected[2],w->projected[3],
                control(5),control(6),control(7),control(8),w->staged_history,w->staged_memory,
                w->convolved,w->beta,w->log_decay,w->readout,w->gated,w->status));
            auto u=w->tensors[4]; const auto& desc=u->descriptor;
            cuda_check(ml_qsa_projection_launch(stream,u->device,desc.type_id,w->d,w->inner,
                desc.row_bytes,w->gated,w->output,w->status));
            w->finite(w->staged_memory,w->memory_count); w->finite(w->staged_history,w->history_count);
            w->finite(w->convolved,w->channels); w->finite(w->readout,w->inner);
            w->finite(w->gated,w->inner); w->check_status();
            // No persistent mutation or visible snapshot before complete status validation.
            w->invalid=true;
            w->copy(w->history,w->staged_history,w->history_count);
            w->copy(w->memory,w->staged_memory,w->memory_count);
            cuda_check(cudaStreamSynchronize(stream));
            w->length=position+1; w->invalid=false; w->output_valid=true; *committed=w->length;
        } catch (const Failure& f) { if (f.status==ML_CUDA) w->invalid=true; throw; }
}
extern "C" {
int ml_recurrent_create(ml_handle engine,int d,int hk,int hv,int dim,int taps,
    uint64_t capacity,const ml_handle* tensors,float epsilon,int arithmetic,ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null recurrent output");
        *out=0;
        if (!tensors || d<1 || d>2560 || hk<1 || hk>16 || hv<1 || hv>48 || hv%hk ||
            dim<1 || dim>128 || taps<1 || taps>16 || !capacity ||
            !std::isfinite(epsilon) || epsilon<=0 || arithmetic!=0)
            fail(ML_INVALID,"invalid bounded recurrent config/reference_fp32 arithmetic");
        auto e=lookup(engines,engine);
        auto w=std::make_shared<RecurrentLayer>(e,d,hk,hv,dim,taps,capacity,epsilon);
        uint64_t widths[]={uint64_t(d),uint64_t(d),uint64_t(d),uint64_t(d),w->inner,
                           uint64_t(taps),uint64_t(dim),uint64_t(hv),uint64_t(hv)};
        uint64_t rows[]={w->channels,w->inner,uint64_t(hv),uint64_t(hv),uint64_t(d),w->channels,0,0,0};
        for (int j=0;j<9;++j) {
            w->tensors[j]=lookup(uploads,tensors[j]);
            recurrent_shape(w->tensors[j],e,widths[j],rows[j],j<5);
        }
        e->admit(w->bytes+4); DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        for (auto& u:w->tensors) u->wait();
        cuda_check(cudaMalloc(&w->device,w->bytes));
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&w->host),4));
        e->record(w->bytes,4); w->counted=true;
        float* cursor=static_cast<float*>(w->device);
        auto take=[&](uint64_t n) { auto p=cursor; cursor+=n; return p; };
        w->history=take(w->history_count); w->memory=take(w->memory_count);
        w->staged_history=take(w->history_count); w->staged_memory=take(w->memory_count);
        w->projected[0]=take(w->channels); w->projected[1]=take(w->inner);
        w->projected[2]=take(hv); w->projected[3]=take(hv);
        w->convolved=take(w->channels); w->beta=take(hv); w->log_decay=take(hv);
        w->readout=take(w->inner); w->gated=take(w->inner); w->output=take(d);
        w->status=reinterpret_cast<int*>(cursor);
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,e->stream));
        for (int j=5;j<9;++j) w->finite(static_cast<const float*>(w->tensors[j]->device),w->tensors[j]->descriptor.nbytes/4);
        w->check_status();
        auto h=new_handle(); recurrent_layers.emplace(h,w); *out=h;
    });
}
int ml_recurrent_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(recurrent_layers,id); recurrent_layers.erase(id); } });
}
int ml_recurrent_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(recurrent_layers,id); DeviceGuard guard(w->engine->device);
        w->invalid=true; w->output_valid=false;
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream)); w->length=0; w->invalid=false;
    });
}
int ml_recurrent_position(ml_handle id,uint64_t* length) noexcept {
    return boundary([&] {
        if (!length) fail(ML_INVALID,"null recurrent position");
        *length=0; auto w=lookup(recurrent_layers,id);
        if (w->invalid) fail(ML_INVALID,"recurrent invalid; reset required");
        *length=w->length;
    });
}
int ml_recurrent_step(ml_handle id,ml_handle hidden,uint64_t position,uint64_t* committed) noexcept {
    return boundary([&] {
        if (!committed) fail(ML_INVALID,"null recurrent committed output");
        *committed=0; auto w=lookup(recurrent_layers,id); w->output_valid=false;
        if (w->invalid) fail(ML_INVALID,"recurrent invalid; reset required");
        if (position!=w->length) fail(ML_INVALID,"recurrent position mismatch");
        if (position>=w->capacity) fail(ML_CAPACITY,"recurrent capacity exhausted");
        auto h=lookup(uploads,hidden); recurrent_shape(h,w->engine,w->d,0);
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        try {
            h->wait(); recurrent_step_device(w,static_cast<const float*>(h->device),position,committed);
        } catch (const Failure& f) { if (f.status==ML_CUDA) w->invalid=true; throw; }
    });
}
int ml_recurrent_readback_for_test(ml_handle id,int which,float* dst,uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(recurrent_layers,id);
        if (w->invalid || which<0 || which>2 || !dst) fail(ML_INVALID,"recurrent invalid diagnostic");
        if (which==0 && !w->output_valid) fail(ML_INVALID,"recurrent output unavailable");
        const float* src[]={w->output,w->history,w->memory};
        uint64_t sizes[]={uint64_t(w->d),w->history_count,w->memory_count};
        if (count<sizes[which]) fail(ML_INVALID,"recurrent diagnostic capacity");
        DeviceGuard guard(w->engine->device);
        if (sizes[which]) cuda_check(cudaMemcpyAsync(dst,src[which],sizes[which]*4,cudaMemcpyDeviceToHost,w->engine->stream));
        cuda_check(cudaStreamSynchronize(w->engine->stream));
    });
}
}

// Private bounded PLE owners. Registry order preserves engine/upload/mmap lifetimes.
struct PLEWeights {
    std::shared_ptr<Engine> engine;
    std::shared_ptr<PayloadLease> table;
    std::shared_ptr<UploadedTensor> uploads[6];
    MlPleHashConfig hash{};
    float epsilon;
    const float* controls[4]{};
    uint64_t bytes=4;
    void* device=nullptr;
    int *status=nullptr, *host=nullptr;
    bool counted=false;
    ~PLEWeights() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if (cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (counted) { engine->device_bytes-=bytes; engine->pinned_bytes-=4; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
};
struct PLEState {
    std::shared_ptr<PLEWeights> weights;
    // 2 histories, 2 outputs, embedding, key/value, RMS scratch, u, gates, z.
    static constexpr uint64_t floats=2*92160+2*10240+2560+10240+2560+3*10240+4+10240;
    static constexpr uint64_t bytes=floats*4+1440+4, pinned=1440+4;
    uint64_t capacity, position=0, rows[16]{};
    int32_t predecessors[2]{-1,-1};
    void* device=nullptr;
    unsigned char* host=nullptr;
    unsigned char* packed=nullptr;
    int *status=nullptr, *host_status=nullptr;
    float *history=nullptr,*next_history=nullptr,*output=nullptr,*staged=nullptr;
    float *embedding=nullptr,*key=nullptr,*value=nullptr,*kn=nullptr,*qn=nullptr,*u=nullptr,*gates=nullptr,*z=nullptr;
    bool counted=false, valid=false, poisoned=false;
    ~PLEState() noexcept {
        if (!weights) return; // Lookup can fail before ownership is installed.
        auto e=weights->engine;
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(e->device);
        if (cudaStreamSynchronize(e->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if (device) (void)cudaFree(device);
        if (host) (void)cudaFreeHost(host);
        if (counted) { e->device_bytes-=bytes; e->pinned_bytes-=pinned; }
        if (previous>=0) (void)cudaSetDevice(previous);
    }
};
std::unordered_map<ml_handle,std::shared_ptr<PLEWeights>> ple_weights;
std::unordered_map<ml_handle,std::shared_ptr<PLEState>> ple_states;
void ple_shape(const std::shared_ptr<UploadedTensor>& u,const std::shared_ptr<Engine>& e,
               uint64_t columns,uint64_t rows,bool gamma=false) {
    const auto& d=u->descriptor;
    if (u->engine!=e) fail(ML_INVALID,"PLE cross-engine input");
    bool shape=d.rank==2 && d.ggml_shape[0]==columns && d.ggml_shape[1]==rows;
    if (gamma) shape=(d.rank==1 && d.ggml_shape[0]==10240) ||
        (d.rank==2 && d.ggml_shape[0]==2560 && d.ggml_shape[1]==4);
    if (!shape) fail(ML_INVALID,"PLE exact tensor shape required");
}
extern "C" {
int ml_ple_weights_create(ml_handle engine,ml_handle table,const ml_handle* tensors,
    const uint64_t* multipliers,const uint64_t* offsets,const uint64_t* vocabs,
    int32_t eos,float epsilon,int arithmetic,ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null PLE weight output");
        *out=0;
        if (!tensors || !multipliers || !offsets || !vocabs || arithmetic!=0 ||
            !std::isfinite(epsilon) || epsilon<=0) fail(ML_INVALID,"PLE requires reference_fp32 and finite positive epsilon/config");
        auto e=lookup(engines,engine);
        auto w=std::make_shared<PLEWeights>(); w->engine=e; w->table=lookup(payload_leases,table); w->epsilon=epsilon;
        const auto& td=w->table->descriptor();
        if (td.rank!=2 || td.ggml_shape[0]!=160 || td.type_id!=20 || td.row_bytes!=90 ||
            td.block_elements!=32 || td.block_bytes!=18) fail(ML_INVALID,"PLE table must be IQ4_NL [rows,160]");
        w->hash.table_rows=td.ggml_shape[1]; w->hash.eos=eos;
        std::copy_n(multipliers,3,w->hash.multipliers); std::copy_n(offsets,16,w->hash.offsets);
        std::copy_n(vocabs,16,w->hash.vocabs);
        int32_t previous[2]{-1,-1}, next[2]; uint64_t rows[16];
        if (!ml_ple_hash_rows(w->hash,0,previous,rows,next)) fail(ML_INVALID,"invalid PLE hash config/vocabulary");
        for (int j=0;j<6;++j) {
            auto u=w->uploads[j]=lookup(uploads,tensors[j]); const auto& d=u->descriptor;
            ple_shape(u,e,j<2?2560:4,j==0?10240:j==1?2560:10240,j>=2 && j<5);
            if (j<2) {
                if (d.type_id!=0 && d.type_id!=30 && d.type_id!=12 && d.type_id!=13 && d.type_id!=14 && d.type_id!=23)
                    fail(ML_INVALID,"unsupported PLE packed projection type");
            } else {
                if (d.type_id!=0 && d.type_id!=1 && d.type_id!=30) fail(ML_INVALID,"PLE controls require F32/F16/BF16");
                if (d.type_id!=0) w->bytes+=(j==5?40960ull:10240ull)*4;
            }
        }
        e->admit(w->bytes+4); DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        for (auto& u:w->uploads) u->wait();
        cuda_check(cudaMalloc(&w->device,w->bytes)); cuda_check(cudaMallocHost(reinterpret_cast<void**>(&w->host),4));
        e->record(w->bytes,4); w->counted=true; w->status=static_cast<int*>(w->device);
        float* cursor=reinterpret_cast<float*>(w->status+1);
        cuda_check(cudaMemsetAsync(w->status,0,4,e->stream));
        for (int j=2;j<6;++j) {
            auto u=w->uploads[j]; const auto& d=u->descriptor; uint64_t count=j==5?40960:10240;
            if (d.type_id==0) {
                w->controls[j-2]=static_cast<const float*>(u->device);
                cuda_check(ml_qsa_layer_finite(e->stream,w->controls[j-2],count,w->status));
            } else {
                w->controls[j-2]=cursor;
                // Controls are dense: flatten to rows of width four for the common decoder.
                cuda_check(ml_qsa_decode_rows_launch(e->stream,u->device,d.type_id,count/4,4,8,cursor,w->status));
                cursor+=count;
            }
        }
        cuda_check(cudaMemcpyAsync(w->host,w->status,4,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaStreamSynchronize(e->stream));
        if (*w->host) fail(ML_INVALID,"PLE nonfinite controls");
        auto h=new_handle(); ple_weights.emplace(h,w); *out=h;
    });
}
int ml_ple_weights_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(ple_weights,id); ple_weights.erase(id); } });
}
int ml_ple_create(ml_handle weights,uint64_t capacity,ml_handle* out) noexcept {
    return boundary([&] {
        if (!out) fail(ML_INVALID,"null PLE state output");
        *out=0; if (!capacity) fail(ML_INVALID,"PLE positive capacity required");
        auto w=std::make_shared<PLEState>(); w->weights=lookup(ple_weights,weights); w->capacity=capacity;
        auto e=w->weights->engine; e->admit(w->bytes+w->pinned); DeviceGuard guard(e->device);
        cuda_check(cudaMalloc(&w->device,w->bytes)); cuda_check(cudaMallocHost(reinterpret_cast<void**>(&w->host),w->pinned));
        e->record(w->bytes,w->pinned); w->counted=true;
        float* cursor=static_cast<float*>(w->device);
        auto take=[&](uint64_t n) { auto p=cursor; cursor+=n; return p; };
        w->history=take(92160); w->next_history=take(92160); w->output=take(10240); w->staged=take(10240);
        w->embedding=take(2560); w->key=take(10240); w->value=take(2560);
        w->kn=take(10240); w->qn=take(10240); w->u=take(10240); w->gates=take(4); w->z=take(10240);
        w->packed=reinterpret_cast<unsigned char*>(cursor); w->status=reinterpret_cast<int*>(w->packed+1440);
        w->host_status=reinterpret_cast<int*>(w->host+1440);
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,e->stream)); cuda_check(cudaStreamSynchronize(e->stream));
        auto h=new_handle(); ple_states.emplace(h,w); *out=h;
    });
}
int ml_ple_destroy(ml_handle id) noexcept {
    return boundary([&] { if (id) { (void)lookup(ple_states,id); ple_states.erase(id); } });
}
int ml_ple_reset(ml_handle id) noexcept {
    return boundary([&] {
        auto w=lookup(ple_states,id); auto e=w->weights->engine; DeviceGuard guard(e->device);
        w->poisoned=true; w->valid=false;
        cuda_check(cudaMemsetAsync(w->device,0,w->bytes,e->stream)); cuda_check(cudaStreamSynchronize(e->stream));
        w->position=0; w->predecessors[0]=w->predecessors[1]=-1; std::fill_n(w->rows,16,0); w->poisoned=false;
    });
}
void ple_step_device(const std::shared_ptr<PLEState>& w, ml_handle input, int32_t token, uint64_t position, int hc, std::shared_ptr<HCState> owner={}) {
        w->valid=false;
        if (w->poisoned) fail(ML_INVALID,"PLE CUDA failure; reset required");
        if (position!=w->position) fail(ML_INVALID,"PLE position mismatch");
        if (position>=w->capacity) fail(ML_CAPACITY,"PLE capacity exhausted");
        auto weights=w->weights; auto e=weights->engine;
        uint64_t rows[16]; int32_t next[2];
        if (!ml_ple_hash_rows(weights->hash,token,w->predecessors,rows,next)) fail(ML_INVALID,"invalid PLE token");
        std::shared_ptr<UploadedTensor> upload;
        const float* streams=nullptr;
        if (hc==1) {
            if (!owner) owner=lookup(hc_states,input);
            if (owner->engine!=e || owner->d!=2560 || owner->pending || owner->invalid)
                fail(ML_INVALID,"PLE HC engine/width/transaction mismatch");
            owner->ready(); streams=owner->streams;
        } else if (hc==0) {
            upload=lookup(uploads,input); ple_shape(upload,e,2560,4);
            if (upload->descriptor.type_id!=0) fail(ML_INVALID,"PLE input must be F32 [4,2560]");
            streams=static_cast<const float*>(upload->device);
        } else fail(ML_INVALID,"invalid PLE input kind");
        DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        // Any CUDA exception after this point quarantines scratch until explicit reset.
        w->poisoned=true;
        if (upload) upload->wait();
        const auto& td=weights->table->descriptor();
        const auto* source=weights->table->store->maps[td.shard_index]->data+td.data_offset;
        for (int h=0;h<16;++h) std::memcpy(w->host+h*90,source+rows[h]*90,90);
        auto stream=e->stream;
        cuda_check(cudaMemsetAsync(w->status,0,4,stream));
        cuda_check(cudaMemcpyAsync(w->packed,w->host,1440,cudaMemcpyHostToDevice,stream));
        cuda_check(ml_qsa_decode_rows_launch(stream,w->packed,20,16,160,90,w->embedding,w->status));
        for (int j=0;j<2;++j) {
            auto u=weights->uploads[j]; const auto& d=u->descriptor;
            cuda_check(ml_qsa_projection_launch(stream,u->device,d.type_id,d.ggml_shape[1],2560,d.row_bytes,
                w->embedding,j==0?w->key:w->value,w->status));
        }
        cuda_check(ml_ple_apply_launch(stream,weights->epsilon,streams,w->key,w->value,
            weights->controls[0],weights->controls[1],weights->controls[2],weights->controls[3],w->history,
            w->kn,w->qn,w->u,w->gates,w->z,w->staged,w->next_history,w->status));
        cuda_check(cudaMemcpyAsync(w->host_status,w->status,4,cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream)); w->poisoned=false;
        if (*w->host_status) fail(ML_INVALID,"PLE nonfinite source or arithmetic; committed state preserved");
        if (owner) {
            // Copy into HC's uncommitted buffer; never exchange allocation ownership.
            w->poisoned=true;
            cuda_check(cudaMemcpyAsync(owner->staged,w->staged,40960,cudaMemcpyDeviceToDevice,stream));
            cuda_check(cudaStreamSynchronize(stream));
            w->poisoned=false;
            std::swap(owner->streams,owner->staged); owner->hidden_valid=false;
        }
        std::swap(w->output,w->staged);
        std::swap(w->history,w->next_history);
        std::copy_n(next,2,w->predecessors); std::copy_n(rows,16,w->rows); ++w->position; w->valid=true;
}
int ml_ple_step(ml_handle id,ml_handle input,int32_t token,uint64_t position,int hc) noexcept {
    return boundary([&] { ple_step_device(lookup(ple_states,id),input,token,position,hc); });
}
int ml_ple_metadata(ml_handle id,int32_t* predecessors,uint64_t* rows,uint64_t* position) noexcept {
    return boundary([&] {
        auto w=lookup(ple_states,id);
        if (!predecessors || !rows || !position || w->poisoned) fail(ML_INVALID,"invalid PLE metadata output/state");
        std::copy_n(w->predecessors,2,predecessors); std::copy_n(w->rows,16,rows); *position=w->position;
    });
}
int ml_ple_readback_for_test(ml_handle id,int which,float* dst,uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(ple_states,id);
        if (w->poisoned || (which!=1 && !w->valid)) fail(ML_INVALID,"PLE output unavailable");
        const float* source=nullptr; uint64_t required=0;
        switch (which) {
            case 0: source=w->output; required=10240; break;
            case 1: source=w->history; required=92160; break;
            case 2: source=w->embedding; required=2560; break;
            case 3: source=w->gates; required=4; break;
            default: fail(ML_INVALID,"invalid PLE diagnostic");
        }
        if (!dst || count<required) fail(ML_INVALID,"PLE diagnostic capacity");
        auto e=w->weights->engine; DeviceGuard guard(e->device);
        cuda_check(cudaMemcpyAsync(dst,source,required*4,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaStreamSynchronize(e->stream));
    });
}
}

// SPDX-License-Identifier: LGPL-3.0-or-later
// Private native caller only. All worker arithmetic remains in the supplied bridge.
struct CPUSharedLibrary {
    void* handle=nullptr;
    explicit CPUSharedLibrary(const char* path,bool process_resident=false) {
        if (!path || !*path) fail(ML_INVALID,"explicit CPU library path required");
        int flags=RTLD_NOW|RTLD_LOCAL;
        if(process_resident) {
            // GGML installs a process-global C++ terminate handler. Unmapping it
            // leaves that handler dangling; reloading can assert in its initializer.
#ifdef RTLD_NODELETE
            flags|=RTLD_NODELETE;
#else
            fail(ML_INVALID,"GGML process residency requires RTLD_NODELETE");
#endif
        }
        handle=dlopen(path,flags);
        if (!handle) fail(ML_INVALID,dlerror());
    }
    ~CPUSharedLibrary() { if(handle) dlclose(handle); }
    CPUSharedLibrary(const CPUSharedLibrary&)=delete;
    template<class T> T symbol(const char* name) {
        dlerror(); void* p=dlsym(handle,name);
        if (const char* error=dlerror()) fail(ML_INVALID,error);
        if (!p) fail(ML_INVALID,"null CPU library symbol");
        return reinterpret_cast<T>(p);
    }
};
struct CPUExpertPool {
    std::shared_ptr<Engine> engine;
    CPUSharedLibrary bridge, ggml;
    decltype(&mylocal_ggml_pool_create) create;
    decltype(&mylocal_ggml_pool_register) reg;
    decltype(&mylocal_ggml_pool_submit_registered) submit;
    decltype(&mylocal_ggml_pool_wait) wait;
    decltype(&mylocal_ggml_pool_destroy) destroy;
    struct Functions { int type, activation; Quantize quantize; Dot dot; };
    std::array<Functions,6> functions;
    std::vector<std::array<std::shared_ptr<PayloadLease>,3>> sources;
    void* pool=nullptr;
    int d, f, capacity;
    bool pending=false, closed=false;
    CPUExpertPool(std::shared_ptr<Engine> e,int width,int ff,int threads,int cap,
                  const char* bp,const char* gp)
        : engine(std::move(e)),bridge(bp),ggml(gp,true),d(width),f(ff),capacity(cap) {
        create=bridge.symbol<decltype(create)>("mylocal_ggml_pool_create");
        reg=bridge.symbol<decltype(reg)>("mylocal_ggml_pool_register");
        submit=bridge.symbol<decltype(submit)>("mylocal_ggml_pool_submit_registered");
        wait=bridge.symbol<decltype(wait)>("mylocal_ggml_pool_wait");
        destroy=bridge.symbol<decltype(destroy)>("mylocal_ggml_pool_destroy");
        ggml.symbol<void(*)()>("ggml_cpu_init")();
        // Same ABI and activation-type checks as GGMLPackedCPUExecutor._Traits.
        struct Traits { void* from_float; void* vec_dot; int vec_dot_type; int64_t nrows; };
        auto traits=ggml.symbol<const Traits*(*)(int)>("ggml_get_type_traits_cpu");
        constexpr int types[]={22,21,18,23,20,42};
        const char* dots[]={"ggml_vec_dot_iq2_s_q8_K","ggml_vec_dot_iq3_s_q8_K",
            "ggml_vec_dot_iq3_xxs_q8_K","ggml_vec_dot_iq4_xs_q8_K",
            "ggml_vec_dot_iq4_nl_q8_0","ggml_vec_dot_q2_0_q8_0"};
        for (size_t i=0;i<functions.size();++i) {
            int activation=i<4?15:8;
            const auto* t=traits(types[i]);
            if(!t || t->vec_dot_type!=activation || !t->vec_dot)
                fail(ML_INVALID,"incompatible GGML CPU traits");
            functions[i]={types[i],activation,ggml.symbol<Quantize>(
                activation==15?"quantize_row_q8_K":"quantize_row_q8_0"),ggml.symbol<Dot>(dots[i])};
        }
        pool=create(d,f,capacity,threads);
        if(!pool) fail(ML_CAPACITY,"cannot allocate existing GGML row pool");
    }
    void drain() noexcept { if(pending) { wait(pool); pending=false; } }
    void close() noexcept { drain(); closed=true; }
    ~CPUExpertPool() { if(pool) { drain(); destroy(pool); } }
    Projection projection(const std::shared_ptr<PayloadLease>& lease,uint64_t offset,int rows,int columns) {
        const auto& t=lease->descriptor();
        const Functions* fn=nullptr;
        for(const auto& v:functions) if(uint32_t(v.type)==t.type_id) fn=&v;
        if(!fn) fail(ML_INVALID,"unsupported GGML expert format");
        if((t.rank!=2 && t.rank!=3) || t.ggml_shape[0]!=uint64_t(columns) ||
           t.ggml_shape[1]!=uint64_t(rows) || !t.row_bytes ||
           (fn->activation==15 && columns%256) || (fn->activation==8 && columns%32))
            fail(ML_INVALID,"CPU expert projection shape/alignment mismatch");
        const uint64_t bytes=uint64_t(rows)*t.row_bytes;
        if(offset>t.nbytes || bytes>t.nbytes-offset || offset%bytes)
            fail(ML_INVALID,"CPU expert subtensor offset out of bounds/alignment");
        const auto& mapping=lease->store->maps[t.shard_index];
        return {mapping->data+t.data_offset+offset,size_t(t.row_bytes),fn->activation,fn->quantize,fn->dot};
    }
};
struct CPUExpertJob {
    std::shared_ptr<CPUExpertPool> owner;
    std::vector<float> input, output;
    bool complete=false, submitted=false;
    CPUExpertJob(std::shared_ptr<CPUExpertPool> p,const float* x,size_t n)
        : owner(std::move(p)),input(x,x+owner->d),output(n*owner->d) {}
    void finish() noexcept {
        if(submitted && !complete) { owner->drain(); complete=true; }
    }
    ~CPUExpertJob() { finish(); }
};
// Jobs retire before pools, and both before source/engine registries at exit.
std::unordered_map<ml_handle,std::shared_ptr<CPUExpertPool>> cpu_expert_pools;
std::unordered_map<ml_handle,std::shared_ptr<CPUExpertJob>> cpu_expert_jobs;
extern "C" int ml_cpu_experts_create(ml_handle engine,int d,int f,int threads,int capacity,
        const char* bp,const char* gp,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto e=lookup(engines,engine);
        if(!out || d<1 || d>65536 || f<1 || f>65536 || d%32 || f%32 ||
           threads<1 || threads>256 || capacity<1 || capacity>512)
            fail(ML_INVALID,"invalid CPU expert dimensions/threads/capacity/output");
        auto p=std::make_shared<CPUExpertPool>(e,d,f,threads,capacity,bp,gp);
        auto id=new_handle(); cpu_expert_pools.emplace(id,std::move(p)); *out=id;
    });
}
extern "C" int ml_cpu_experts_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { auto p=lookup(cpu_expert_pools,id); p->close(); cpu_expert_pools.erase(id); } });
}
extern "C" int ml_cpu_experts_register(ml_handle id,const ml_handle* leases,
        const uint64_t* offsets,int32_t* index) noexcept {
    return boundary([&] {
        if(index) *index=-1;
        auto p=lookup(cpu_expert_pools,id);
        if(p->pending) fail(ML_INVALID,"CPU expert pool busy");
        if(!leases || !offsets || !index) fail(ML_INVALID,"null CPU expert registration");
        std::array<std::shared_ptr<PayloadLease>,3> sources;
        std::array<Projection,3> descriptors;
        for(int i=0;i<3;++i) {
            sources[i]=lookup(payload_leases,leases[i]);
            descriptors[i]=p->projection(sources[i],offsets[i],i==2?p->d:p->f,i==2?p->f:p->d);
        }
        PoolExpert expert{descriptors[0],descriptors[1],descriptors[2]};
        // Establish ownership before registration, so allocation failure cannot
        // leave a descriptor pointing at an unretained mapping.
        p->sources.push_back(std::move(sources));
        int registered=-1; int status=p->reg(p->pool,&expert,&registered);
        if(status) { p->sources.pop_back(); fail(ML_INVALID,"GGML expert registration rejected"); }
        *index=registered;
    });
}
extern "C" int ml_cpu_experts_submit(ml_handle id,const int32_t* indices,uint64_t count,
        const float* input,uint64_t input_count,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto p=lookup(cpu_expert_pools,id);
        if(p->pending) fail(ML_INVALID,"CPU expert pool busy");
        if(!out || !input || !indices || input_count!=uint64_t(p->d) || !count || count>uint64_t(p->capacity))
            fail(ML_INVALID,"invalid CPU expert batch/input");
        for(uint64_t i=0;i<count;++i)
            if(indices[i]<0 || size_t(indices[i])>=p->sources.size()) fail(ML_INVALID,"invalid registered expert index");
        for(int i=0;i<p->d;++i) if(!std::isfinite(input[i])) fail(ML_INVALID,"nonfinite CPU expert input");
        auto job=std::make_shared<CPUExpertJob>(p,input,size_t(count));
        auto handle=new_handle(); cpu_expert_jobs.emplace(handle,job);
        if(p->submit(p->pool,indices,int(count),job->input.data(),job->output.data())) {
            cpu_expert_jobs.erase(handle); fail(ML_INVALID,"GGML registered submit rejected");
        }
        job->submitted=true; p->pending=true; *out=handle;
    });
}
extern "C" int ml_cpu_job_wait(ml_handle id) noexcept {
    return boundary([&] { lookup(cpu_expert_jobs,id)->finish(); });
}
extern "C" int ml_cpu_job_readback_for_test(ml_handle id,float* output,uint64_t capacity) noexcept {
    return boundary([&] {
        auto j=lookup(cpu_expert_jobs,id);
        if(!output || capacity<j->output.size()) fail(ML_INVALID,"CPU job output capacity too small");
        j->finish(); std::copy(j->output.begin(),j->output.end(),output);
    });
}
extern "C" int ml_cpu_job_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { auto j=lookup(cpu_expert_jobs,id); j->finish(); cpu_expert_jobs.erase(id); } });
}

// SPDX-License-Identifier: LGPL-3.0-or-later
// Static placement, explicit heterogeneous arithmetic; no Python dispatch seam.
struct MoeOwner {
    std::shared_ptr<Engine> engine;
    std::array<std::shared_ptr<UploadedTensor>,5> weights; // router, shared G/U/D, scalar
    std::array<std::shared_ptr<PayloadLease>,3> banks;
    std::shared_ptr<CPUExpertPool> cpu;
    std::array<int,512> resident{}, registered{};
        std::array<MLMoeExpert,512> host_descriptors{};
    std::vector<std::pair<void*,uint64_t>> allocations;
    int d=0,f=0,sf=0;
    bool busy=false, invalid=false, result_valid=false;
    uint64_t length=0; // HC mix commits only; standalone jobs are stateless.
    MLMoeExpert shared{};
    MLMoeExpert* experts=nullptr;
    int *placement=nullptr,*ids=nullptr,*status=nullptr;
    float *x=nullptr,*logits=nullptr,*probabilities=nullptr,*route_weights=nullptr;
    float *hidden=nullptr,*terms=nullptr,*cpu_rows=nullptr,*routed=nullptr,*output=nullptr;
    float *sg=nullptr,*su=nullptr,*sh=nullptr,*score=nullptr,*raw=nullptr,*shared_out=nullptr,*scalar_gate=nullptr;
    explicit MoeOwner(std::shared_ptr<Engine> e):engine(std::move(e)) { registered.fill(-1); }
    ~MoeOwner() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if(cpu) cpu->drain();
        if(cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        for(auto [p,n]:allocations) { (void)cudaFree(p); engine->device_bytes-=n; }
        if(previous>=0) (void)cudaSetDevice(previous);
    }
    template<class T> T* alloc(uint64_t count) {
        uint64_t n=count*sizeof(T); engine->admit(n);
        // Reserve bookkeeping before allocation, so every successful allocation is owned.
        allocations.reserve(allocations.size()+1);
        void* p=nullptr; cuda_check(cudaMalloc(&p,n));
        allocations.emplace_back(p,n); engine->record(n,0); return static_cast<T*>(p);
    }
    MLMoeMatrix bank_matrix(int p,int id,bool copy) {
        const auto& t=banks[p]->descriptor();
        int rows=p==2?d:f,cols=p==2?f:d;
        bool allowed=p==2?(t.type_id==20 || t.type_id==42):
            (t.type_id==18 || t.type_id==21 || t.type_id==22 || t.type_id==23);
        if(!allowed || t.rank!=3 || t.ggml_shape[0]!=uint64_t(cols) ||
           t.ggml_shape[1]!=uint64_t(rows) || t.ggml_shape[2]!=512 || !t.row_bytes ||
           t.nbytes!=uint64_t(512)*rows*t.row_bytes)
            fail(ML_INVALID,"MoE packed bank shape/type mismatch");
        uint64_t n=uint64_t(rows)*t.row_bytes;
        const auto& map=banks[p]->store->maps[t.shard_index];
        const void* source=map->data+t.data_offset+uint64_t(id)*n;
        if(!copy) return {source,t.row_bytes,t.type_id,0};
        void* target=alloc<unsigned char>(n);
        cuda_check(cudaMemcpyAsync(target,source,n,cudaMemcpyHostToDevice,engine->stream));
        return {target,t.row_bytes,t.type_id,0};
    }
};
struct MoeJob {
    std::shared_ptr<MoeOwner> owner;
    std::shared_ptr<CPUExpertJob> cpu;
    std::array<int,10> ids{},slots{};
    std::array<float,10> weights{};
    std::vector<float> input,rows,result;
    bool complete=false,failed=false,enqueued=false;
    explicit MoeJob(std::shared_ptr<MoeOwner> p):owner(std::move(p)),
        input(owner->d),rows(10*owner->d),result(owner->d) {}
    void finish(bool snapshot=true) {
        if(complete) { if(failed) fail(ML_INVALID,"failed MoE job"); return; }
        auto p=owner; auto e=p->engine; DeviceGuard guard(e->device);
        // Always drain both branches on error before permitting the next epoch.
        try {
            if(cpu) {
                { wp9::Timer t(7); cpu->finish(); }
                for(size_t i=0;i<cpu->output.size()/p->d;++i)
                    std::copy_n(cpu->output.data()+i*p->d,p->d,rows.data()+slots[i]*p->d);
            }
            cuda_check(cudaMemcpyAsync(p->cpu_rows,rows.data(),rows.size()*4,cudaMemcpyHostToDevice,e->stream));
            cuda_check(ml_moe_finish_launch(e->stream,p->d,p->ids,p->placement,p->route_weights,
                p->cpu_rows,p->terms,p->routed,p->shared_out,p->output,p->status));
            int status=0;
            cuda_check(cudaMemcpyAsync(&status,p->status,4,cudaMemcpyDeviceToHost,e->stream));
            if(snapshot)
                cuda_check(cudaMemcpyAsync(result.data(),p->output,p->d*4,cudaMemcpyDeviceToHost,e->stream));
            cuda_check(cudaStreamSynchronize(e->stream));
            if(status) fail(ML_INVALID,"nonfinite/invalid MoE device result");
            complete=true; p->busy=false; p->result_valid=true;
        } catch(...) {
            if(cpu) cpu->finish();
            if(cudaStreamSynchronize(e->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
            complete=true; failed=true; p->busy=false; throw;
        }
    }
    ~MoeJob() {
            if(!complete) {
                if(enqueued) { try { finish(); } catch(...) {} }
                else {
                    if(cpu) cpu->finish();
                    int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(owner->engine->device);
                    if(cudaStreamSynchronize(owner->engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
                    if(previous>=0) (void)cudaSetDevice(previous);
                    owner->busy=false;
                }
            }
        }
};
std::unordered_map<ml_handle,std::shared_ptr<MoeOwner>> moe_owners;
std::unordered_map<ml_handle,std::shared_ptr<MoeJob>> moe_jobs;
extern "C" int ml_moe_create(ml_handle engine,int d,int f,const ml_handle* weights,
    const ml_handle* banks,const int32_t* gpu_ids,uint64_t gpu_count,int threads,
    const char* bridge,const char* ggml,const char* policy,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto e=lookup(engines,engine);
        if(!out || !weights || !banks || (gpu_count && !gpu_ids) || gpu_count>512 ||
           d<256 || d>6144 || d%256 || f<256 || f>12288 || f%64 || threads<1 || threads>256 ||
           !policy || std::strcmp(policy,"cpu_ggml_q8_gpu_reference_fp32"))
            fail(ML_INVALID,"invalid MoE configuration/explicit mixed arithmetic policy");
        auto p=std::make_shared<MoeOwner>(e); p->d=d; p->f=f;
        DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        for(uint64_t i=0;i<gpu_count;++i) {
            int id=gpu_ids[i];
            if(id<0 || id>=512 || p->resident[id]) fail(ML_INVALID,"invalid/duplicate MoE GPU placement");
            p->resident[id]=1;
        }
        for(int i=0;i<5;++i) {
            p->weights[i]=lookup(uploads,weights[i]);
            if(p->weights[i]->engine!=e) fail(ML_INVALID,"MoE upload engine mismatch");
        }
        for(int i=0;i<3;++i) p->banks[i]=lookup(payload_leases,banks[i]);
        const auto& rt=p->weights[0]->descriptor;
        if((rt.type_id!=0 && rt.type_id!=1 && rt.type_id!=30) || rt.rank!=2 || rt.ggml_shape[0]!=uint64_t(d) || rt.ggml_shape[1]!=512)
            fail(ML_INVALID,"MoE router must be F32/F16/BF16 [512,D]");
        p->sf=int(p->weights[1]->descriptor.ggml_shape[1]);
        if(p->sf<256 || p->sf>12288 || p->sf%64) fail(ML_INVALID,"invalid MoE shared width");
        MLMoeMatrix* sm[]={&p->shared.gate,&p->shared.up,&p->shared.down};
        for(int i=0;i<3;++i) {
            auto w=p->weights[i+1]; const auto& t=w->descriptor;
            bool allowed=i==2?(t.type_id==20 || t.type_id==8):
                (t.type_id==12 || t.type_id==13 || t.type_id==14 || t.type_id==23);
            if(!allowed || t.rank!=2 || t.ggml_shape[0]!=uint64_t(i==2?p->sf:d) ||
               t.ggml_shape[1]!=uint64_t(i==2?d:p->sf)) fail(ML_INVALID,"MoE shared shape/type mismatch");
            *sm[i]={w->device,t.row_bytes,t.type_id,0};
        }
        const auto& scalar=p->weights[4]->descriptor;
        if((scalar.type_id!=0 && scalar.type_id!=1 && scalar.type_id!=30) || scalar.rank!=1 || scalar.ggml_shape[0]!=uint64_t(d))
            fail(ML_INVALID,"MoE shared scalar gate must be F32/F16/BF16 [D]");
        for(int i=0;i<3;++i) (void)p->bank_matrix(i,0,false);
        if(gpu_count<512) p->cpu=std::make_shared<CPUExpertPool>(e,d,f,threads,10,bridge,ggml);
        auto& descriptors=p->host_descriptors;
        for(int id=0;id<512;++id) {
            if(p->resident[id]) {
                descriptors[id]={p->bank_matrix(0,id,true),p->bank_matrix(1,id,true),p->bank_matrix(2,id,true)};
            } else {
                std::array<Projection,3> projections;
                for(int i=0;i<3;++i) {
                    uint64_t bytes=uint64_t(i==2?d:f)*p->banks[i]->descriptor().row_bytes;
                    projections[i]=p->cpu->projection(p->banks[i],id*bytes,i==2?d:f,i==2?f:d);
                }
                p->cpu->sources.push_back(p->banks);
                PoolExpert expert{projections[0],projections[1],projections[2]};
                if(p->cpu->reg(p->cpu->pool,&expert,&p->registered[id])) fail(ML_INVALID,"MoE CPU registration rejected");
            }
        }
        p->experts=p->alloc<MLMoeExpert>(512); p->placement=p->alloc<int>(512);
        cuda_check(cudaMemcpyAsync(p->experts,descriptors.data(),sizeof(descriptors),cudaMemcpyHostToDevice,e->stream));
        cuda_check(cudaMemcpyAsync(p->placement,p->resident.data(),512*4,cudaMemcpyHostToDevice,e->stream));
        p->ids=p->alloc<int>(10); p->status=p->alloc<int>(1);
        p->x=p->alloc<float>(d); p->logits=p->alloc<float>(512); p->probabilities=p->alloc<float>(512);
        p->route_weights=p->alloc<float>(10); p->hidden=p->alloc<float>(10*f);
        p->terms=p->alloc<float>(10*d); p->cpu_rows=p->alloc<float>(10*d);
        p->routed=p->alloc<float>(d); p->output=p->alloc<float>(d);
        p->sg=p->alloc<float>(p->sf); p->su=p->alloc<float>(p->sf); p->sh=p->alloc<float>(p->sf);
        p->scalar_gate=p->alloc<float>(d);
                p->score=p->alloc<float>(1); p->raw=p->alloc<float>(d); p->shared_out=p->alloc<float>(d);
        cuda_check(cudaStreamSynchronize(e->stream));
        auto id=new_handle(); moe_owners.emplace(id,p); *out=id;
    });
}
std::shared_ptr<MoeJob> moe_submit_input(const std::shared_ptr<MoeOwner>& p,
    const float* input,float scale,bool device_input) {
        if(p->invalid) fail(ML_INVALID,"MoE owner invalid; reset required");
        if(p->busy) fail(ML_INVALID,"MoE owner busy");
        if(!input || !std::isfinite(scale) || scale<0) fail(ML_INVALID,"invalid MoE input/scale");
        if(!device_input) for(int i=0;i<p->d;++i)
            if(!std::isfinite(input[i])) fail(ML_INVALID,"nonfinite MoE input");
        auto j=std::make_shared<MoeJob>(p);
        if(!device_input) std::copy_n(input,p->d,j->input.data());
        auto e=p->engine; DeviceGuard guard(e->device);
        // On a partial launch failure, job destruction drains before releasing inputs.
        p->busy=true; p->result_valid=false;
        cuda_check(cudaMemsetAsync(p->status,0,4,e->stream));
        cuda_check(cudaMemcpyAsync(p->x,device_input?input:j->input.data(),p->d*4,
            device_input?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice,e->stream));
        const auto& rt=p->weights[0]->descriptor;
        cuda_check(ml_qsa_projection_launch(e->stream,p->weights[0]->device,rt.type_id,512,p->d,rt.row_bytes,p->x,p->logits,p->status));
        cuda_check(ml_moe_route_launch(e->stream,p->logits,scale,p->probabilities,p->ids,p->route_weights,p->status));
        int status=0;
        cuda_check(cudaMemcpyAsync(j->ids.data(),p->ids,40,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaMemcpyAsync(j->weights.data(),p->route_weights,40,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaMemcpyAsync(&status,p->status,4,cudaMemcpyDeviceToHost,e->stream));
        cuda_check(cudaStreamSynchronize(e->stream));
        if(status) fail(ML_INVALID,"nonfinite/invalid MoE routing");
        std::array<int,10> indices{}; int n=0;
        for(int slot=0;slot<10;++slot) if(!p->resident[j->ids[slot]]) {
            j->slots[n]=slot; indices[n++]=p->registered[j->ids[slot]];
        }
        if(n) {
            if(device_input) {
                cuda_check(cudaMemcpyAsync(j->input.data(),p->x,p->d*4,cudaMemcpyDeviceToHost,e->stream));
                cuda_check(cudaStreamSynchronize(e->stream));
            }
            j->cpu=std::make_shared<CPUExpertJob>(p->cpu,j->input.data(),n);
            if(p->cpu->submit(p->cpu->pool,indices.data(),n,j->cpu->input.data(),j->cpu->output.data()))
                fail(ML_INVALID,"MoE CPU submit rejected");
            j->cpu->submitted=true; p->cpu->pending=true;
        }
        const auto& scalar=p->weights[4]->descriptor;
        cuda_check(ml_qsa_decode_rows_launch(e->stream,p->weights[4]->device,scalar.type_id,
            1,p->d,scalar.row_bytes,p->scalar_gate,p->status));
        cuda_check(ml_moe_shared_launch(e->stream,p->d,p->sf,p->shared,
            p->scalar_gate,p->x,p->sg,p->su,p->sh,p->score,p->raw,p->shared_out,p->status));
        cuda_check(ml_moe_resident_launch(e->stream,p->d,p->f,p->experts,p->placement,
            p->ids,p->route_weights,p->x,p->hidden,p->terms,p->status));
        j->enqueued=true;
        return j;
}
extern "C" int ml_moe_submit(ml_handle id,const float* input,uint64_t count,float scale,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto p=lookup(moe_owners,id);
        if(!out || count!=uint64_t(p->d)) fail(ML_INVALID,"invalid MoE input/output count");
        auto j=moe_submit_input(p,input,scale,false);
        auto handle=new_handle(); moe_jobs.emplace(handle,j); *out=handle;
    });
}
extern "C" int ml_moe_reset(ml_handle id) noexcept {
    return boundary([&] {
            auto p=lookup(moe_owners,id); if(p->busy) fail(ML_INVALID,"MoE owner busy");
            p->invalid=false; p->length=0; p->result_valid=false;
        });
}
extern "C" int ml_moe_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) {
        auto p=lookup(moe_owners,id);
        for(auto& [key,j]:moe_jobs) if(j->owner==p && !j->complete) {
                    try { j->finish(); } catch(...) { /* finish drained; close discards failed output. */ }
                }
        moe_owners.erase(id);
    } });
}
extern "C" int ml_moe_input_readback_for_test(ml_handle id,float* output,uint64_t count) noexcept {
    return boundary([&] {
        auto p=lookup(moe_owners,id);
        if(p->busy || p->invalid || !p->result_valid || !output || count<uint64_t(p->d))
            fail(ML_INVALID,"MoE diagnostic input unavailable/capacity");
        DeviceGuard guard(p->engine->device);
        cuda_check(cudaMemcpyAsync(output,p->x,p->d*4,cudaMemcpyDeviceToHost,p->engine->stream));
        cuda_check(cudaStreamSynchronize(p->engine->stream));
    });
}
// Read-only diagnostics of the latest completed owner epoch. No arithmetic or
// allocation is added to submit/finish; decoder use additionally requires trace.
void moe_diagnostic_copy(const std::shared_ptr<MoeOwner>& p,int which,void* out,uint64_t count) {
    if(p->busy || p->invalid || !p->result_valid || !out)
        fail(ML_INVALID,"MoE diagnostic unavailable");
    const void* source=nullptr; uint64_t n=0;
    switch(which) {
        case 0: source=p->ids; n=10; break; // int32, all other cases float32
        case 1: source=p->route_weights; n=10; break;
        case 2: source=p->logits; n=512; break;
        case 3: source=p->probabilities; n=512; break;
        case 4: source=p->cpu_rows; n=10*p->d; break; // unweighted, route-slot order
        case 5: source=p->terms; n=10*p->d; break; // weighted, route-slot order
        case 6: source=p->routed; n=p->d; break;
        case 7: source=p->shared_out; n=p->d; break;
        case 8: source=p->output; n=p->d; break;
        case 9: source=p->x; n=p->d; break;
        case 10: source=p->sg; n=p->sf; break;
        case 11: source=p->su; n=p->sf; break;
        case 12: source=p->sh; n=p->sf; break;
        case 13: source=p->score; n=1; break;
        case 14: source=p->raw; n=p->d; break;
        default: fail(ML_INVALID,"invalid MoE diagnostic kind");
    }
    if(count<n) fail(ML_INVALID,"MoE diagnostic capacity");
    DeviceGuard guard(p->engine->device);
    cuda_check(cudaMemcpyAsync(out,source,n*4,cudaMemcpyDeviceToHost,p->engine->stream));
    cuda_check(cudaStreamSynchronize(p->engine->stream));
}
extern "C" int ml_moe_diagnostic_for_test(ml_handle id,int which,void* out,uint64_t count) noexcept {
    return boundary([&] { moe_diagnostic_copy(lookup(moe_owners,id),which,out,count); });
}
extern "C" int ml_moe_job_wait(ml_handle id) noexcept {
    return boundary([&] { lookup(moe_jobs,id)->finish(); });
}
extern "C" int ml_moe_job_readback_for_test(ml_handle id,float* output,uint64_t capacity,
    int32_t* ids,float* weights) noexcept {
    return boundary([&] {
        auto j=lookup(moe_jobs,id);
        if(!output || capacity<j->result.size() || !ids || !weights) fail(ML_INVALID,"invalid MoE diagnostic buffers");
        j->finish(); std::copy(j->result.begin(),j->result.end(),output);
        std::copy(j->ids.begin(),j->ids.end(),ids); std::copy(j->weights.begin(),j->weights.end(),weights);
    });
}
extern "C" int ml_moe_job_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) {
            auto j=lookup(moe_jobs,id);
            try { j->finish(); } catch(...) { /* Failed jobs must still be releasable. */ }
            moe_jobs.erase(id);
        } });
}

// Private orchestration only: all arithmetic remains in the validated operators.
// Called under the parent boundary lock; shared owners outlive stream completion.
void hc_mix_impl(ml_handle id, ml_handle weights, ml_handle branch,
                 int kind, uint64_t position, float scale, uint64_t* committed) {
        if (committed) *committed=0;
        auto w=lookup(hc_states,id); w->begin(); w->ready();
        if (!committed) fail(ML_INVALID,"null HC mix committed output");
        if (w->pending) fail(ML_INVALID,"HC pending read must be written before mix");
        auto v=lookup(hc_weights,weights);
        if (v->engine!=w->engine || v->d!=w->d || v->rank>w->rank_capacity || !v->values[3])
            fail(ML_INVALID,"HC mix weight engine/shape/rank/injection mismatch");
        std::shared_ptr<QSALayer> q;
        std::shared_ptr<RecurrentLayer> r;
        std::shared_ptr<MoeOwner> m;
        if (kind==0) q=lookup(qsa_layers,branch);
        else if (kind==1) r=lookup(recurrent_layers,branch);
        else if (kind==2) m=lookup(moe_owners,branch);
        else fail(ML_INVALID,"invalid HC mix branch kind");
        if ((q && (q->engine!=w->engine || w->d!=2560 || q->invalid)) ||
            (r && (r->engine!=w->engine || r->d!=w->d || r->invalid)) ||
            (m && (m->engine!=w->engine || m->d!=w->d || m->invalid)))
            fail(ML_INVALID,"HC mix branch engine/width/invalid mismatch; reset required");
        if(m && m->busy) fail(ML_INVALID,"MoE owner busy");
        if(!std::isfinite(scale) || scale<0 || (!m && scale!=1)) fail(ML_INVALID,"invalid HC branch scale");
        const uint64_t length=q?q->length:r?r->length:m->length;
        const uint64_t capacity=q?q->capacity:r?r->capacity:std::numeric_limits<uint64_t>::max();
        if (position!=length) fail(ML_INVALID,"HC mix position mismatch");
        if (position>=capacity) fail(ML_CAPACITY,"HC mix capacity exhausted");
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        uint64_t ticket=0, done=0;
        try {
            hc_read_device(w,v,&ticket);
            if (q) { q->output_valid=false; qsa_layer_step_device(q,w->hidden,position,&done); }
            else if(r) { r->output_valid=false; recurrent_step_device(r,w->hidden,position,&done); }
            else {
                auto job=moe_submit_input(m,w->hidden,scale,true);
                job->finish(false); // Status fence only; HC consumes the native device output.
                done=position+1;
            }
            // The exact read's gains are consumed once, without exporting its ticket.
            w->begin();
            if (!w->pending || w->ticket!=ticket) fail(ML_INTERNAL,"HC mix lost read ticket");
            hc_write_device(w,q?q->output:r?r->output:m->output);
            if(m) m->length=done;
            *committed=done;
        } catch (...) {
            w->invalid=true; w->hidden_valid=false;
            // A branch may already have committed before HC write failed. Quarantine
            // both owners rather than presenting a resumable partial combined step.
            if (q) { q->invalid=true; q->output_valid=false; }
            if (r) { r->invalid=true; r->output_valid=false; }
            if (m) m->invalid=true;
            throw;
        }
}
extern "C" int ml_hc_mix(ml_handle id, ml_handle weights, ml_handle branch,
                          int kind, uint64_t position, uint64_t* committed) noexcept {
    return boundary([&] { hc_mix_impl(id,weights,branch,kind,position,1.0f,committed); });
}
extern "C" int ml_hc_mix_scaled(ml_handle id, ml_handle weights, ml_handle branch,
    int kind, uint64_t position, float scale, uint64_t* committed) noexcept {
    return boundary([&] { hc_mix_impl(id,weights,branch,kind,position,scale,committed); });
}

// Architecture-aware batch-one reference decoder. Mutable bindings are consumed.
struct DecoderLayer {
    std::shared_ptr<HCWeights> attention,ffn;
    std::shared_ptr<RecurrentLayer> recurrent;
    std::shared_ptr<QSALayer> qsa;
    std::shared_ptr<PLEState> ple;
    std::shared_ptr<MoeOwner> moe;
};
struct DecoderPromptCheckpoint;
struct Decoder {
    std::shared_ptr<DecoderPromptCheckpoint> checkpoint;
    uint64_t checkpoint_epoch=0;
    std::shared_ptr<Engine> engine;
    std::shared_ptr<PayloadLease> embedding;
    std::shared_ptr<UploadedTensor> output;
    std::shared_ptr<HCWeights> final;
    std::shared_ptr<HCState> hc;
    std::vector<DecoderLayer> layers;
    std::vector<float> trace;
    uint64_t capacity=0,position=0,bytes=0;
    int d=0,vocab=0;
    float scale=1;
    bool trace_valid=false,failed=false,counted=false,session_counted=false,valid=false;
    void* device=nullptr;
    unsigned char* packed=nullptr;
    float *row=nullptr,*logits=nullptr;
    int* status=nullptr;
    ~Decoder() noexcept {
        DeviceGuardNoThrow();
    }
    void DeviceGuardNoThrow() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if(cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if(device) (void)cudaFree(device);
        if(counted) engine->device_bytes-=bytes;
        if(session_counted) --engine->sessions;
        if(previous>=0) (void)cudaSetDevice(previous);
    }
    void snapshot(size_t layer,int stage,const float* source,uint64_t count) {
        if(trace.empty()) return;
        auto* dst=trace.data()+(layer*4+stage)*4*d;
        cuda_check(cudaMemcpyAsync(dst,source,count*4,cudaMemcpyDeviceToHost,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
    }
    void mix(DecoderLayer& b,bool ffn) {
        auto w=hc; uint64_t ticket=0,done=0;
        w->begin(); w->ready();
        { wp9::Timer t(0); hc_read_device(w,ffn?b.ffn:b.attention,&ticket); }
        if(ffn) {
            std::shared_ptr<MoeJob> job;
            { wp9::Timer t(1); job=moe_submit_input(b.moe,w->hidden,scale,true); }
            { wp9::Timer t(2); job->finish(false); }
        } else if(b.recurrent) { wp9::Timer t(3); recurrent_step_device(b.recurrent,w->hidden,position,&done); }
        else { wp9::Timer t(4); qsa_layer_step_device(b.qsa,w->hidden,position,&done); }
        w->begin();
        if(!w->pending || w->ticket!=ticket) fail(ML_INTERNAL,"decoder HC ticket lost");
        { wp9::Timer t(5); hc_write_device(w,ffn?b.moe->output:b.recurrent?b.recurrent->output:b.qsa->output); }
        if(ffn) b.moe->length=position+1;
    }
    void reset() {
        checkpoint.reset(); ++checkpoint_epoch;
        failed=true; valid=false;
        auto stream=engine->stream;
        for(auto& b:layers) {
            if(b.recurrent) {
                auto w=b.recurrent; w->invalid=true; w->output_valid=false;
                cuda_check(cudaMemsetAsync(w->device,0,w->bytes,stream));
                w->length=0; w->invalid=false;
            } else {
                auto w=b.qsa; w->invalid=true; w->output_valid=false;
                cuda_check(cudaMemsetAsync(w->device,0,w->bytes,stream));
                if(w->paged) w->paged->reset();
                w->length=0; w->invalid=false;
            }
            if(b.ple) {
                auto w=b.ple; w->poisoned=true; w->valid=false;
                cuda_check(cudaMemsetAsync(w->device,0,w->bytes,stream));
                w->position=0; w->predecessors[0]=w->predecessors[1]=-1;
                std::fill_n(w->rows,16,0); w->poisoned=false;
            }
            b.moe->invalid=false; b.moe->result_valid=false; b.moe->length=0;
        }
        cuda_check(cudaMemsetAsync(hc->device,0,hc->bytes,stream));
        cuda_check(cudaMemsetAsync(device,0,bytes,stream));
        cuda_check(cudaStreamSynchronize(stream));
        hc->pending.reset(); hc->ticket=0; hc->initialized=false; hc->collapsed=false;
        hc->hidden_valid=false; hc->invalid=false;
        std::fill(trace.begin(),trace.end(),0); position=0; failed=false;
    }
};
std::unordered_map<ml_handle,std::shared_ptr<Decoder>> decoders;

// Ordinary host snapshots have an explicit per-checkpoint budget. No full QSA
// raw/K/V/pooled prefix is copied; those committed rows cannot be overwritten by
// append-only decode. The raw ring is the only mutable QSA dependency copied.
struct PromptCopy {
    void* device;
    uint64_t offset,bytes;
};
struct DecoderPromptCheckpoint {
    std::weak_ptr<Decoder> owner;
    uint64_t epoch=0,position=0,host_bytes=0;
    ml_handle owner_id=0;
    bool logits_valid=false,fail_restore_for_test=false;
    std::vector<PromptCopy> copies;
    std::vector<unsigned char> data;
    struct PleMetadata {
        int32_t predecessors[2]; uint64_t rows[16];
        float *history=nullptr,*next_history=nullptr;
    };
    std::vector<PleMetadata> ple;
};
std::unordered_map<ml_handle,std::weak_ptr<DecoderPromptCheckpoint>> prompt_checkpoints;

extern "C" int ml_decoder_checkpoint_capture(ml_handle id,uint64_t budget,ml_handle* out,
    uint64_t* host_bytes) noexcept {
    return boundary([&] {
        if(out) *out=0;
        if(host_bytes) *host_bytes=0;
        auto p=lookup(decoders,id); // Unavailable while native worker owns it.
        if(!out || !host_bytes || !budget || p->failed)
            fail(ML_INVALID,"invalid prompt checkpoint capture/session");
        auto c=std::make_shared<DecoderPromptCheckpoint>();
        c->owner=p; c->owner_id=id; c->epoch=p->checkpoint_epoch; c->position=p->position; c->logits_valid=p->valid;
        uint64_t bytes=0;
        auto add=[&](void* device,uint64_t count) {
            c->copies.push_back({device,bytes,count}); bytes+=count;
        };
        for(auto& b:p->layers) {
            if(b.recurrent) {
                auto w=b.recurrent;
                if(w->invalid || w->length!=p->position) fail(ML_INVALID,"checkpoint recurrent position");
                add(w->history,w->history_count*4); add(w->memory,w->memory_count*4);
            } else {
                auto w=b.qsa;
                if(w->invalid || w->length!=p->position || (w->paged && w->paged->stats.length!=p->position))
                    fail(ML_INVALID,"checkpoint QSA position");
                if(w->paged) add(w->raw,640*4);
            }
            if(b.ple) {
                auto w=b.ple;
                if(w->poisoned || w->position!=p->position) fail(ML_INVALID,"checkpoint PLE position");
                add(w->history,92160*4);
                DecoderPromptCheckpoint::PleMetadata meta{};
                meta.history=w->history; meta.next_history=w->next_history;
                std::copy_n(w->predecessors,2,meta.predecessors); std::copy_n(w->rows,16,meta.rows);
                c->ple.push_back(meta);
            }
        }
        if(p->valid) add(p->logits,uint64_t(p->vocab)*4);
        c->host_bytes=bytes+sizeof(DecoderPromptCheckpoint)+c->copies.capacity()*sizeof(PromptCopy)+
            c->ple.capacity()*sizeof(DecoderPromptCheckpoint::PleMetadata);
        if(c->host_bytes>budget) fail(ML_CAPACITY,"prompt checkpoint host budget exhausted");
        // Drop the previous host buffer before allocating its replacement: one
        // bounded retained checkpoint, never two complete state copies.
        p->checkpoint.reset();
        c->data.resize(bytes);
        DeviceGuard guard(p->engine->device); StreamCompletion completion{p->engine->stream};
        for(auto& copy:c->copies) if(copy.bytes)
            cuda_check(cudaMemcpyAsync(c->data.data()+copy.offset,copy.device,copy.bytes,
                cudaMemcpyDeviceToHost,p->engine->stream));
        cuda_check(cudaStreamSynchronize(p->engine->stream));
        auto h=new_handle(); prompt_checkpoints.emplace(h,c); p->checkpoint=c;
        *host_bytes=c->host_bytes; *out=h;
    });
}
extern "C" int ml_decoder_checkpoint_restore(ml_handle id,ml_handle checkpoint,uint64_t* committed) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        auto c=lookup(prompt_checkpoints,checkpoint).lock();
        if(!committed || !c || c->owner.lock()!=p || p->checkpoint!=c ||
           c->epoch!=p->checkpoint_epoch || p->failed || p->position<c->position)
            fail(ML_INVALID,"stale prompt checkpoint/session; reset required");
        for(auto& b:p->layers) {
            if(b.recurrent && (b.recurrent->invalid || b.recurrent->length!=p->position))
                fail(ML_INVALID,"checkpoint recurrent history changed");
            if(b.qsa && (b.qsa->invalid || b.qsa->length!=p->position ||
                (b.qsa->paged && b.qsa->paged->stats.length!=p->position)))
                fail(ML_INVALID,"checkpoint QSA history changed");
            if(b.ple && (b.ple->poisoned || b.ple->position!=p->position))
                fail(ML_INVALID,"checkpoint PLE history changed");
        }
        DeviceGuard guard(p->engine->device); StreamCompletion completion{p->engine->stream};
        // Publish no usable session until all copies, pager truncation and fences
        // succeed. A partial failure quarantines the decoder and checkpoint.
        p->failed=true; p->valid=false; p->trace_valid=false;
        try {
            cuda_check(cudaStreamSynchronize(p->engine->stream));
            for(auto& copy:c->copies) if(copy.bytes) {
                cuda_check(cudaMemcpyAsync(copy.device,c->data.data()+copy.offset,copy.bytes,
                    cudaMemcpyHostToDevice,p->engine->stream));
                if(c->fail_restore_for_test) fail(ML_CUDA,"injected prompt checkpoint restore failure");
            }
            size_t ple_index=0;
            for(auto& b:p->layers) {
                if(b.recurrent) {
                    b.recurrent->length=c->position; b.recurrent->output_valid=false;
                } else {
                    auto w=b.qsa;
                    if(w->paged) w->paged->truncate(c->position);
                    // New pools beyond floor(position/4) are ignored/recomputed;
                    // stale IDs/scores/staging are not diagnostic-valid after restore.
                    cuda_check(cudaMemcpyAsync(w->n,&c->position,8,cudaMemcpyHostToDevice,p->engine->stream));
                    w->length=c->position; w->output_valid=false;
                }
                if(b.ple) {
                    auto& meta=c->ple[ple_index++]; auto w=b.ple;
                    // PLE swaps these buffers each step. Copies above target the
                    // captured allocation; restore its committed/scratch roles too.
                    w->history=meta.history; w->next_history=meta.next_history;
                    std::copy_n(meta.predecessors,2,w->predecessors); std::copy_n(meta.rows,16,w->rows);
                    w->position=c->position; w->valid=false;
                }
                b.moe->length=c->position; b.moe->result_valid=false;
            }
            auto w=p->hc;
            w->pending.reset(); w->ticket=0; w->initialized=false; w->collapsed=false; w->hidden_valid=false;
            cuda_check(cudaStreamSynchronize(p->engine->stream));
            p->position=c->position; p->valid=c->logits_valid; p->failed=false; *committed=p->position;
        } catch(...) { p->checkpoint.reset(); ++p->checkpoint_epoch; throw; }
    });
}
extern "C" int ml_decoder_paged_stats_for_test(ml_handle id,uint64_t layer,ml_kvi8_stats* out) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!out || layer>=p->layers.size() || !p->layers[layer].qsa || !p->layers[layer].qsa->paged)
            fail(ML_INVALID,"decoder paged diagnostic unavailable");
        auto w=p->layers[layer].qsa->paged;
        *out=w->stats; out->pending=w->pending?1:0;
    });
}
extern "C" int ml_decoder_checkpoint_fail_restore_for_test(ml_handle id) noexcept {
    return boundary([&] {
        auto c=lookup(prompt_checkpoints,id).lock();
        if(!c || !decoders.count(c->owner_id)) fail(ML_INVALID,"stale/worker-owned checkpoint");
        c->fail_restore_for_test=true;
    });
}
extern "C" int ml_decoder_checkpoint_readback_for_test(ml_handle id,void* out,uint64_t count,uint64_t* size) noexcept {
    return boundary([&] {
        auto c=lookup(prompt_checkpoints,id).lock();
        if(!c || !size) fail(ML_INVALID,"stale prompt checkpoint diagnostic");
        *size=c->data.size();
        if(!out && !count) return;
        if(!out || count<c->data.size()) fail(ML_INVALID,"checkpoint diagnostic capacity");
        std::memcpy(out,c->data.data(),c->data.size());
    });
}
extern "C" int ml_decoder_checkpoint_destroy(ml_handle id) noexcept {
    return boundary([&] {
        if(!id) return;
        auto c=lookup(prompt_checkpoints,id).lock();
        if(c) if(auto p=c->owner.lock(); p && p->checkpoint==c) {
            if(!decoders.count(c->owner_id)) fail(ML_INVALID,"checkpoint owner handle unavailable during generation");
            p->checkpoint.reset();
        }
        prompt_checkpoints.erase(id);
    });
}
extern "C" int ml_decoder_create(ml_handle engine,ml_handle embedding,ml_handle output,
    ml_handle final,ml_handle hc,const ml_decoder_layer* bindings,uint64_t count,
    uint64_t capacity,float scale,int trace,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto e=lookup(engines,engine);
        if(!out || !bindings || !count || count>48 || !capacity || capacity>262144 ||
           !std::isfinite(scale) || scale<=0 || (trace!=0 && trace!=1))
            fail(ML_INVALID,"invalid decoder configuration");
        if(e->sessions>=e->max_sessions) fail(ML_CAPACITY,"decoder session limit");
        auto p=std::make_shared<Decoder>(); p->engine=e; p->capacity=capacity; p->scale=scale;
        p->embedding=lookup(payload_leases,embedding); p->output=lookup(uploads,output);
        p->final=lookup(hc_weights,final); p->hc=lookup(hc_states,hc); p->d=p->hc->d;
        const auto& ed=p->embedding->descriptor(); const auto& od=p->output->descriptor;
        if(ed.rank!=2 || ed.ggml_shape[0]!=uint64_t(p->d) || ed.type_id!=23 ||
           !ed.ggml_shape[1] || ed.ggml_shape[1]>248320 || p->d%256 ||
           p->output->engine!=e || od.rank!=2 || od.ggml_shape[0]!=ed.ggml_shape[0] ||
           od.ggml_shape[1]!=ed.ggml_shape[1] ||
           (od.type_id!=20 && od.type_id!=8 && od.type_id!=23 && od.type_id!=0 &&
                       od.type_id!=12 && od.type_id!=13 && od.type_id!=14))
            fail(ML_INVALID,"decoder embedding/output geometry or packed format");
        p->vocab=int(ed.ggml_shape[1]);
        auto check_hc=[&](const std::shared_ptr<HCWeights>& w,bool inject) {
            if(w->engine!=e || w->d!=p->d || w->rank>p->hc->rank_capacity || bool(w->values[3])!=inject)
                fail(ML_INVALID,"decoder HC weight mismatch");
        };
        if(p->hc->engine!=e || p->hc->initialized || p->hc->invalid || p->hc->pending)
            fail(ML_INVALID,"decoder requires fresh exclusive HC state");
        check_hc(p->final,false);
        std::unordered_set<ml_handle> mutable_ids{hc};
        for(uint64_t i=0;i<count;++i) {
            const auto& v=bindings[i]; DecoderLayer b;
            b.attention=lookup(hc_weights,v.attention_hc); b.ffn=lookup(hc_weights,v.ffn_hc);
            check_hc(b.attention,true); check_hc(b.ffn,true);
            if(!mutable_ids.insert(v.branch).second || !mutable_ids.insert(v.moe).second ||
               (v.ple && !mutable_ids.insert(v.ple).second)) fail(ML_INVALID,"duplicate decoder mutable binding");
            if(v.kind==1) {
                b.recurrent=lookup(recurrent_layers,v.branch); auto w=b.recurrent;
                if(w->engine!=e || w->d!=p->d || w->length || w->invalid || w->capacity<capacity)
                    fail(ML_INVALID,"decoder recurrent binding mismatch");
            } else if(v.kind==0) {
                b.qsa=lookup(qsa_layers,v.branch); auto w=b.qsa;
                if(w->engine!=e || p->d!=2560 || w->length || w->invalid || w->capacity<capacity)
                    fail(ML_INVALID,"decoder QSA binding mismatch");
            } else fail(ML_INVALID,"decoder branch kind");
            b.moe=lookup(moe_owners,v.moe);
            if(b.moe->engine!=e || b.moe->d!=p->d || b.moe->busy || b.moe->invalid || b.moe->length)
                fail(ML_INVALID,"decoder MoE binding mismatch");
            if(v.ple) {
                b.ple=lookup(ple_states,v.ple); auto w=b.ple;
                if(p->d!=2560 || w->weights->engine!=e || w->position || w->poisoned || w->capacity<capacity)
                    fail(ML_INVALID,"decoder PLE binding mismatch");
            }
            p->layers.push_back(std::move(b));
        }
        p->bytes=ed.row_bytes+4*(uint64_t(p->d)+p->vocab+1);
        e->admit(p->bytes); DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        if(trace) p->trace.resize(count*4*4*p->d);
        p->output->wait();
        cuda_check(cudaMalloc(&p->device,p->bytes)); e->record(p->bytes,0); p->counted=true;
        p->packed=static_cast<unsigned char*>(p->device);
        p->row=reinterpret_cast<float*>(p->packed+ed.row_bytes);
        p->logits=p->row+p->d; p->status=reinterpret_cast<int*>(p->logits+p->vocab);
        auto id=new_handle(); decoders.emplace(id,p);
        // Publication succeeded: transfer mutable bindings; old aliases become stale.
        hc_states.erase(hc);
        for(uint64_t i=0;i<count;++i) {
            auto& v=bindings[i];
            if(v.kind) recurrent_layers.erase(v.branch); else qsa_layers.erase(v.branch);
            moe_owners.erase(v.moe); if(v.ple) ple_states.erase(v.ple);
        }
        ++e->sessions; p->session_counted=true; *out=id;
    });
}
extern "C" int ml_decoder_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(decoders,id); decoders.erase(id); } });
}
extern "C" int ml_decoder_reset(ml_handle id) noexcept {
    return boundary([&] { auto p=lookup(decoders,id); DeviceGuard g(p->engine->device); p->reset(); });
}
extern "C" int ml_decoder_state(ml_handle id,uint64_t* position,int* failed) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id); if(!position || !failed) fail(ML_INVALID,"null decoder state");
        *position=p->position; *failed=p->failed;
    });
}
void decoder_step_native(const std::shared_ptr<Decoder>& p,int32_t token,float* logits,
    uint64_t count,uint64_t* committed,bool capture_trace=true) {
        if(committed) *committed=0;
        if(p->failed) fail(ML_INVALID,"decoder unresumable; reset required");
        p->valid=false; p->trace_valid=false;
        try {
            if(!committed || (logits ? count<uint64_t(p->vocab) : count!=0) || token<0 || token>=p->vocab)
                fail(ML_INVALID,"decoder token/output capacity");
            if(p->position>=p->capacity) fail(ML_CAPACITY,"decoder active session capacity");
            DeviceGuard guard(p->engine->device); auto stream=p->engine->stream; StreamCompletion completion{stream};
            const auto& ed=p->embedding->descriptor();
            const auto* source=p->embedding->store->maps[ed.shard_index]->data+ed.data_offset+uint64_t(token)*ed.row_bytes;
            cuda_check(cudaMemsetAsync(p->status,0,4,stream));
            cuda_check(cudaMemcpyAsync(p->packed,source,ed.row_bytes,cudaMemcpyHostToDevice,stream));
            cuda_check(ml_qsa_decode_rows_launch(stream,p->packed,23,1,p->d,ed.row_bytes,p->row,p->status));
            auto w=p->hc;
            w->pending.reset(); w->ticket=0; w->collapsed=false; w->hidden_valid=false;
            for(int c=0;c<4;++c) cuda_check(cudaMemcpyAsync(w->streams+c*p->d,p->row,p->d*4,cudaMemcpyDeviceToDevice,stream));
            w->initialized=true; w->invalid=false;
            for(size_t i=0;i<p->layers.size();++i) {
                auto& b=p->layers[i];
                if(b.ple) { wp9::Timer t(6); ple_step_device(b.ple,0,token,p->position,1,w); }
                p->mix(b,false); if(capture_trace) { p->snapshot(i,0,w->hidden,p->d); p->snapshot(i,1,w->streams,4*p->d); }
                p->mix(b,true); if(capture_trace) { p->snapshot(i,2,w->hidden,p->d); p->snapshot(i,3,w->streams,4*p->d); }
            }
            wp9::Timer head_tail(8);
            w->begin(); w->ready(); auto v=p->final;
            cuda_check(cudaMemsetAsync(w->status,0,4,stream));
            cuda_check(ml_hc_collapse_launch(stream,w->d,v->rank,v->epsilon,w->streams,
                v->values[0],v->values[1],v->values[2],w->normalized,w->low,w->hidden,w->status));
            w->check_status(); w->collapsed=true; w->hidden_valid=true; w->invalid=false;
            const auto& od=p->output->descriptor;
            cuda_check(ml_vocab_projection_launch(stream,p->output->device,od.type_id,p->vocab,p->d,od.row_bytes,
                w->hidden,p->logits,p->status));
            int status=0; cuda_check(cudaMemcpyAsync(&status,p->status,4,cudaMemcpyDeviceToHost,stream));
            cuda_check(cudaStreamSynchronize(stream));
            if(status) fail(ML_INVALID,"decoder nonfinite embedding/logits");
            // Null logits with zero count explicitly keeps the completed vector on device.
            if(logits) {
                cuda_check(cudaMemcpyAsync(logits,p->logits,p->vocab*4,cudaMemcpyDeviceToHost,stream));
                cuda_check(cudaStreamSynchronize(stream));
            }
            ++p->position; p->valid=true; p->trace_valid=capture_trace; *committed=p->position;
        } catch(...) { p->failed=true; throw; }
}
extern "C" int ml_decoder_consume(ml_handle id,const int32_t* tokens,uint64_t count,uint64_t* committed) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!committed || (count && !tokens)) fail(ML_INVALID,"null decoder consume input/output");
        *committed=p->position;
        if(p->failed) fail(ML_INVALID,"decoder unresumable; reset required");
        if(count>p->capacity-p->position) fail(ML_CAPACITY,"decoder consume capacity");
        // Validate the entire vector before any mutation. Execution failures still
        // poison the decoder and retain the last successfully committed position.
        for(uint64_t i=0;i<count;++i)
            if(tokens[i]<0 || tokens[i]>=p->vocab) fail(ML_INVALID,"decoder consume token");
        for(uint64_t i=0;i<count;++i) {
            uint64_t position=0;
            decoder_step_native(p,tokens[i],nullptr,0,&position,false);
            *committed=position;
        }
    });
}
extern "C" int ml_decoder_step(ml_handle id,int32_t token,float* logits,uint64_t count,uint64_t* committed) noexcept {
    return boundary([&] { decoder_step_native(lookup(decoders,id),token,logits,count,committed); });
}
extern "C" int ml_decoder_moe_diagnostic_for_test(ml_handle id,uint64_t layer,int which,void* out,uint64_t count) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!p->valid || !p->trace_valid || p->failed || p->trace.empty() || layer>=p->layers.size())
            fail(ML_INVALID,"decoder MoE diagnostic unavailable");
        moe_diagnostic_copy(p->layers[layer].moe,which,out,count);
    });
}
extern "C" int ml_decoder_trace(ml_handle id,uint64_t layer,int stage,float* out,uint64_t count) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!p->valid || !p->trace_valid || p->failed || p->trace.empty() || layer>=p->layers.size() || stage<0 || stage>3 || !out)
            fail(ML_INVALID,"decoder trace unavailable");
        uint64_t n=(stage%2?4:1)*p->d; if(count<n) fail(ML_INVALID,"decoder trace capacity");
        std::copy_n(p->trace.data()+(layer*4+stage)*4*p->d,n,out);
    });
}

// Private WP8 owner; included after Decoder so no pointers escape the C ABI.
struct Sampler {
    std::shared_ptr<Engine> engine;
    int n=0;
    uint64_t seed=0,counter=0,bytes=0;
    size_t temporary_bytes=0;
    void* device=nullptr;
    float *a=nullptr,*b=nullptr,*weights=nullptr,*cdf=nullptr;
    int *ia=nullptr,*ib=nullptr;
    void* temporary=nullptr;
    ml_sample_result* result=nullptr;
    bool counted=false,valid=false;
    ~Sampler() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if(cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if(device) (void)cudaFree(device);
        if(counted) engine->device_bytes-=bytes;
        if(previous>=0) (void)cudaSetDevice(previous);
    }
};
std::unordered_map<ml_handle,std::shared_ptr<Sampler>> samplers;
extern "C" int ml_sampler_create(ml_handle engine,uint64_t vocab,uint64_t seed,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto e=lookup(engines,engine);
        if(!out || !vocab || vocab>248320) fail(ML_INVALID,"sampler vocabulary must be 1..248320");
        DeviceGuard guard(e->device);
        auto p=std::make_shared<Sampler>(); p->engine=e; p->n=int(vocab); p->seed=seed;
        p->temporary_bytes=ml_sampling_temp_bytes(p->n);
        if(!p->temporary_bytes) fail(ML_CUDA,"sampler CUB workspace query failed");
        uint64_t offset=(24*vocab+sizeof(ml_sample_result)+255)&~uint64_t(255);
        p->bytes=offset+p->temporary_bytes;
        e->admit(p->bytes);
        cuda_check(cudaMalloc(&p->device,p->bytes));
        auto* base=static_cast<float*>(p->device);
        p->a=base; p->b=base+vocab;
        p->ia=reinterpret_cast<int*>(base+2*vocab); p->ib=p->ia+vocab;
        p->weights=base+4*vocab; p->cdf=base+5*vocab;
        p->result=reinterpret_cast<ml_sample_result*>(base+6*vocab);

        p->temporary=static_cast<char*>(p->device)+offset;
        e->record(p->bytes,0); p->counted=true;
        auto id=new_handle(); samplers.emplace(id,p); *out=id;
    });
}
extern "C" int ml_sampler_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(samplers,id); samplers.erase(id); } });
}
extern "C" int ml_sampler_reset(ml_handle id,uint64_t seed) noexcept {
    return boundary([&] { auto p=lookup(samplers,id); p->seed=seed; p->counter=0; p->valid=false; });
}
void sampler_sample_native(const std::shared_ptr<Sampler>& p,const float* logits,double t,
    double top_p,uint64_t k,int supplied,double uniform,ml_sample_result* out) {
        DeviceGuard guard(p->engine->device); auto stream=p->engine->stream;
        StreamCompletion completion{stream};
        cuda_check(ml_sampling_launch(stream,logits,p->n,t,top_p,int(k?std::min(k,uint64_t(p->n)):p->n),
            p->seed,p->counter,supplied,uniform,p->a,p->b,p->ia,p->ib,p->weights,p->cdf,
            p->temporary,p->temporary_bytes,p->result));
        cuda_check(cudaMemcpyAsync(out,p->result,sizeof(*out),cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if(out->status) fail(ML_INVALID,out->status==1 ? "sampler nonfinite logits" :
            out->status==2 ? "sampler temperature scaling overflow" : "sampler invalid probability mass");
        if(!supplied && t>0) ++p->counter;
        p->valid=t>0;
}
extern "C" int ml_sampler_sample(ml_handle id,ml_handle source,int decoder,double t,double top_p,
    uint64_t k,int supplied,double uniform,ml_sample_result* out) noexcept {
    return boundary([&] {
        if(out) *out={-1,0,0,0,0,0,0,0,0};
        auto p=lookup(samplers,id); p->valid=false;
        if(!out || !std::isfinite(t) || t<0 || !std::isfinite(float(t)) ||
                   (t>0 && float(t)==0) || !std::isfinite(top_p) || top_p<=0 || top_p>1 ||
           (decoder!=0 && decoder!=1) || (supplied!=0 && supplied!=1) ||
           (supplied && (!std::isfinite(uniform) || uniform<0 || uniform>=1)))
            fail(ML_INVALID,"invalid sampler parameters/uniform");
        std::shared_ptr<UploadedTensor> upload;
        std::shared_ptr<Decoder> dec;
        const float* logits=nullptr;
        if(decoder) {
            dec=lookup(decoders,source);
            if(dec->engine!=p->engine || dec->vocab!=p->n || !dec->valid || dec->failed)
                fail(ML_INVALID,"sampler requires valid same-engine decoder logits");
            logits=dec->logits;
        } else {
            upload=lookup(uploads,source); const auto& d=upload->descriptor;
            if(upload->engine!=p->engine || d.type_id!=0 || d.nbytes!=uint64_t(p->n)*4 ||
               !(d.rank==1 || (d.rank==2 && d.ggml_shape[1]==1)))
                fail(ML_INVALID,"sampler requires same-engine F32 logits vector");
            logits=static_cast<const float*>(upload->device);
        }
        sampler_sample_native(p,logits,t,top_p,k,supplied,uniform,out);
    });
}
extern "C" int ml_sampler_readback_for_test(ml_handle id,int32_t* ids,float* probabilities,float* cdf,uint64_t count) noexcept {
    return boundary([&] {
        auto p=lookup(samplers,id);
        if(!p->valid || !ids || !probabilities || !cdf || count<uint64_t(p->n))
            fail(ML_INVALID,"sampler diagnostic unavailable/capacity");
        DeviceGuard guard(p->engine->device); auto s=p->engine->stream; StreamCompletion completion{s};
        cuda_check(cudaMemcpyAsync(ids,p->ia,p->n*4,cudaMemcpyDeviceToHost,s));
        cuda_check(cudaMemcpyAsync(probabilities,p->weights,p->n*4,cudaMemcpyDeviceToHost,s));
        cuda_check(cudaMemcpyAsync(cdf,p->cdf,p->n*4,cudaMemcpyDeviceToHost,s));
        cuda_check(cudaStreamSynchronize(s));
    });
}

// Lock order: runtime -> generation registry at publication only. Queue waits,
// cancel, registry lookup, and join never hold the runtime mutex.
struct Generation {
    std::shared_ptr<Decoder> decoder;
    std::shared_ptr<Sampler> sampler;
    ml_handle decoder_id=0,sampler_id=0;
    std::atomic<bool> cancelled{false};
    std::mutex queue_mutex,close_mutex;
    std::condition_variable changed;
    std::vector<ml_generation_event> queue;
    std::unordered_set<int32_t> eos;
    size_t head=0,size=0;
    uint64_t maximum=0,k=0;
    double temperature=0,top_p=1;
    int32_t input=0;
    bool done=false,delivered=false,closed=false,running=false;
    ml_generation_event terminal{},published{};
    std::thread worker;
    void cancel() noexcept { cancelled.store(true); changed.notify_all(); }
    ~Generation() noexcept {
        cancel();
        if(worker.joinable()) {
            worker.join();
            // Static teardown may overlap another generation on the same engine.
            // Serialize final owner destruction/accounting with its GPU work.
            std::lock_guard lock(mutex); decoder.reset(); sampler.reset();
        }
    }
    void run() noexcept {
        ml_generation_event state{};
        state.position=decoder->position; state.pending_token=-1;
        int reason=ML_GEN_NONE;
        try {
            for(;;) {
                {
                    std::unique_lock lock(queue_mutex);
                    changed.wait(lock,[&] { return cancelled.load() || size<queue.size(); });
                }
                if(cancelled.load()) { reason=ML_GEN_CANCELLED; break; }
                if(state.sampled>=maximum) { reason=ML_GEN_MAX_TOKENS; break; }
                if(state.position>=decoder->capacity) { reason=ML_GEN_CAPACITY; break; }
                {
                    std::lock_guard lock(mutex);
                    // A cancellation observed before dispatch does not consume input.
                    if(cancelled.load()) { reason=ML_GEN_CANCELLED; break; }
                    { std::lock_guard queue_lock(queue_mutex); running=true; }
                    auto decode_start=std::chrono::steady_clock::now();
                    uint64_t committed=0;
                    decoder_step_native(decoder,input,nullptr,0,&committed,false);
                    state.position=committed;
                    state.pending=0; state.pending_token=-1;
                    sampler->valid=false;
                    sampler_sample_native(sampler,decoder->logits,temperature,top_p,k,0,0,&state.sample);
                    input=state.sample.token; ++state.sampled;
                    state.pending=1; state.pending_token=input;
                    state.decode_seconds+=std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-decode_start).count();
                }
                state.kind=1;
                {
                    std::lock_guard lock(queue_mutex);
                    queue[(head+size)%queue.size()]=state; ++size; published=state; running=false;
                }
                changed.notify_all();
                // Completion of running work is published even when cancel raced it.
                if(eos.count(input)) { reason=ML_GEN_EOS; break; }
                if(state.sampled==maximum) { reason=ML_GEN_MAX_TOKENS; break; }
                if(state.position==decoder->capacity) { reason=ML_GEN_CAPACITY; break; }
            }
        } catch(const Failure& f) {
            reason=ML_GEN_ERROR; state.status=f.status;
            std::snprintf(state.error,sizeof(state.error),"%s",error_text);
        } catch(const std::exception& e) {
            reason=ML_GEN_ERROR; state.status=ML_INTERNAL;
            std::snprintf(state.error,sizeof(state.error),"%s",e.what());
        } catch(...) {
            reason=ML_GEN_ERROR; state.status=ML_INTERNAL;
            std::snprintf(state.error,sizeof(state.error),"unknown generation exception");
        }
        state.kind=2; state.reason=reason; state.failed=decoder->failed;
        {
            std::lock_guard lock(queue_mutex); terminal=state; done=true; running=false;
        }
        changed.notify_all();
    }
};
std::mutex generation_mutex;
std::unordered_map<ml_handle,std::shared_ptr<Generation>> generations;

template<class F> int generation_boundary(F&& fn) noexcept {
    error_text[0]='\0';
    try { fn(); return ML_OK; }
    catch(const Failure& f) { return f.status; }
    catch(const std::exception& e) {
        std::snprintf(error_text,sizeof(error_text),"%s",e.what()); return ML_INTERNAL;
    }
    catch(...) { std::snprintf(error_text,sizeof(error_text),"unknown generation exception"); return ML_INTERNAL; }
}
std::shared_ptr<Generation> generation_lookup(ml_handle id) {
    std::lock_guard lock(generation_mutex); return lookup(generations,id);
}
extern "C" int ml_generation_start(ml_handle decoder,ml_handle sampler,int32_t input,
    uint64_t maximum,const int32_t* eos,uint64_t eos_count,double temperature,
    double top_p,uint64_t k,uint64_t capacity,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto d=lookup(decoders,decoder); auto s=lookup(samplers,sampler);
        if(!out || !capacity || capacity>4096 || eos_count>uint64_t(d->vocab) ||
           (eos_count && !eos) || input<0 || input>=d->vocab || d->failed ||
           d->engine!=s->engine || d->vocab!=s->n ||
           !std::isfinite(temperature) || temperature<0 || !std::isfinite(float(temperature)) ||
           (temperature>0 && float(temperature)==0) || !std::isfinite(top_p) || top_p<=0 || top_p>1)
            fail(ML_INVALID,"invalid generation configuration/owners");
        auto p=std::make_shared<Generation>(); p->decoder=d; p->sampler=s;
        p->decoder_id=decoder; p->sampler_id=sampler; p->input=input;
        p->maximum=maximum; p->temperature=temperature; p->top_p=top_p; p->k=k;
                p->published.position=d->position; p->published.pending_token=-1;
        p->queue.resize(capacity);
        for(uint64_t i=0;i<eos_count;++i) {
            if(eos[i]<0 || eos[i]>=d->vocab) fail(ML_INVALID,"invalid generation EOS ID");
            p->eos.insert(eos[i]);
        }
        auto id=new_handle();
        std::lock_guard registry_lock(generation_mutex);
        generations.emplace(id,p);
        try { p->worker=std::thread([raw=p.get()] { raw->run(); }); }
        catch(...) { generations.erase(id); throw; }
        // Worker cannot enter GPU work until this boundary releases runtime mutex.
        decoders.erase(decoder); samplers.erase(sampler); *out=id;
    });
}
extern "C" int ml_generation_state(ml_handle id,ml_generation_event* out,uint64_t* queued,int32_t* done) noexcept {
    return generation_boundary([&] {
        if(!out || !queued || !done) fail(ML_INVALID,"null generation state output");
        auto p=generation_lookup(id); std::lock_guard lock(p->queue_mutex);
        *out=p->done?p->terminal:p->published; out->in_flight=p->running;
                *queued=p->size; *done=p->done;
    });
}
extern "C" int ml_generation_cancel(ml_handle id) noexcept {
    return generation_boundary([&] { generation_lookup(id)->cancel(); });
}
extern "C" int ml_generation_poll(ml_handle id,uint64_t timeout,ml_generation_event* out) noexcept {
    return generation_boundary([&] {
        if(!out || timeout>120000) fail(ML_INVALID,"invalid generation poll output/timeout");
        *out={}; auto p=generation_lookup(id);
        std::unique_lock lock(p->queue_mutex);
        p->changed.wait_for(lock,std::chrono::milliseconds(timeout),[&] { return p->size || p->done; });
        if(p->size) { *out=p->queue[p->head]; p->head=(p->head+1)%p->queue.size(); --p->size; }
        else if(p->done && !p->delivered) { *out=p->terminal; p->delivered=true; }
        lock.unlock(); p->changed.notify_all();
    });
}
extern "C" int ml_generation_close(ml_handle id,ml_generation_event* out) noexcept {
    return generation_boundary([&] {
        if(!out) fail(ML_INVALID,"null generation close output");
        *out={}; if(!id) return;
        auto p=generation_lookup(id);
        std::lock_guard closer(p->close_mutex);
        if(p->closed) fail(ML_INVALID,"generation already closed");
        p->cancel();
        if(p->worker.joinable()) p->worker.join();
        {
            std::lock_guard lock(p->queue_mutex); *out=p->terminal; p->size=0;
        }
        {
            std::lock_guard lock(mutex);
            decoders.emplace(p->decoder_id,p->decoder);
            try { samplers.emplace(p->sampler_id,p->sampler); }
            catch(...) { decoders.erase(p->decoder_id); throw; }
            p->decoder.reset(); p->sampler.reset();
        }
        p->closed=true;
        { std::lock_guard lock(generation_mutex); generations.erase(id); }
    });
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
