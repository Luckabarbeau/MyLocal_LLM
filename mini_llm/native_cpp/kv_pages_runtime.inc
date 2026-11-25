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
