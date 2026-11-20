// SPDX-License-Identifier: LGPL-3.0-or-later
// Original bounded staging seam. The native service fills complete pinned
// bundles; the GPU pulls them into VRAM after publication. No CUDA calls here.
#include <cstring>

class NativeExpertStage {
public:
    NativeHandoff *h;
    uint8_t *host;
    int limit, minimum;
    size_t stride, sizes[3];
    int *map;
    std::vector<uint64_t> aliases;
    uint64_t *selected_aliases;
    bool pinned_only;
    std::atomic<uint32_t> *ready;
    uint64_t staged=0, bytes=0, copy_ns=0;

    NativeExpertStage(NativeHandoff *handoff,uint8_t *src,int slots,int threshold,
        size_t slot_stride,const size_t *parts,int *selection,uint32_t *flag,
        const uint64_t *direct,uint64_t *chosen,bool only)
        : h(handoff),host(src),limit(slots),minimum(threshold),stride(slot_stride),map(selection),
          aliases(direct,direct+handoff->registry.size()),selected_aliases(chosen),pinned_only(only),
          ready(new(flag) std::atomic<uint32_t>(0)) {
        std::copy(parts,parts+3,sizes);
    }

    void run(uint32_t request,std::atomic<uint32_t> *failure) noexcept {
        int indices[16], selected[2], count=0, misses=0, ns=0, error=0, previous=-1;
        bool submitted=false;
        const auto start=std::chrono::steady_clock::now();
        ++h->runs;
        try {
            std::fill(map,map+h->k,-1);
            std::fill(selected_aliases,selected_aliases+2,uint64_t(0));
            if(h->metadata[h->k]) error=1;
            for(int i=0;i<h->k && !error;++i) {
                int id=h->metadata[i];
                if(id<=previous || id<0 || size_t(id)>=h->registry.size()) { error=1; break; }
                previous=id;
                if(h->registry[id]>=0) ++misses;
            }
            const int wanted=misses>=minimum ? std::min(limit,misses) : 0;
            for(int i=0;i<h->k && !error;++i) {
                const int id=h->registry[h->metadata[i]];
                if(id<0) continue;
                if(size_t(id)>=h->pool->registered.size()) { error=1; break; }
                const uint64_t alias=aliases[h->metadata[i]];
                if(ns<wanted && (!pinned_only || alias)) {
                    map[i]=ns; selected_aliases[ns]=alias; selected[ns++]=id;
                } else indices[count++]=id;
            }
            if(!error && !h->route_heat.empty())
                for(int i=0;i<h->k;++i) ++h->route_heat[h->metadata[i]];
            if(!error && count) {
                error=h->pool->submit_registered(indices,count,h->input,h->output);
                submitted=!error;
            }
            const auto copying=std::chrono::steady_clock::now();
            for(int i=0;i<ns && !error;++i) {
                const auto &e=h->pool->registered[selected[i]];
                const Projection parts[3]={e.gate,e.up,e.down};
                const size_t rows[3]={size_t(h->pool->f),size_t(h->pool->f),size_t(h->pool->d)};
                size_t offset=size_t(i)*stride;
                for(int p=0;p<3;++p) {
                    if(parts[p].stride*rows[p]!=sizes[p]) { error=1; break; }
                    if(!selected_aliases[i]) std::memcpy(host+offset,parts[p].weight,sizes[p]);
                    offset+=sizes[p];
                }
            }
            if(ns) copy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-copying).count();
            if(error) failure->store(2,std::memory_order_release);
            // The map and complete host bundles precede publication. A GPU
            // copy kernel, ordered before packed arithmetic, owns VRAM readiness.
            ready->store(request,std::memory_order_release);
            if(submitted) h->pool->wait();
        } catch(...) {
            if(submitted) h->pool->wait();
            error=3;
            failure->store(2,std::memory_order_release);
            ready->store(request,std::memory_order_release);
        }
        if(error) {
            ++h->errors;
            std::fill(h->output,h->output+size_t(h->k)*h->pool->d,NAN);
        } else {
            h->misses+=misses; h->hits+=h->k-misses; h->batches+=bool(count);
            staged+=ns; bytes+=size_t(ns)*(sizes[0]+sizes[1]+sizes[2]);
        }
        h->elapsed_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
    }
};

extern "C" void *mylocal_ggml_stage_create(void *handoff,void *host,int slots,int minimum,
    size_t stride,const size_t *parts,int *map,uint32_t *ready,
    const uint64_t *direct,uint64_t *chosen,int pinned_only) noexcept {
    if(!handoff || !host || !parts || !map || !ready || !direct || !chosen ||
       (pinned_only!=0 && pinned_only!=1) ||
       uintptr_t(ready)%64 || slots<1 || slots>2 || minimum<slots || minimum>16 ||
       stride>8*1024*1024 || !stride || !parts[0] || !parts[1] || !parts[2] ||
       parts[0]>stride || parts[1]>stride-parts[0] || parts[2]>stride-parts[0]-parts[1]) return nullptr;
    try { return new NativeExpertStage(static_cast<NativeHandoff*>(handoff),
        static_cast<uint8_t*>(host),slots,minimum,stride,parts,map,ready,direct,chosen,pinned_only); }
    catch(...) { return nullptr; }
}
extern "C" void mylocal_ggml_stage_stats(void *stage,uint64_t *out) noexcept {
    if(!stage || !out) return;
    const auto &s=*static_cast<NativeExpertStage*>(stage);
    out[0]=s.staged; out[1]=s.bytes; out[2]=s.copy_ns;
}
extern "C" void mylocal_ggml_stage_destroy(void *stage) noexcept {
    delete static_cast<NativeExpertStage*>(stage);
}
