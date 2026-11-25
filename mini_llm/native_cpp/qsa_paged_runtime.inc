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
