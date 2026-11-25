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
