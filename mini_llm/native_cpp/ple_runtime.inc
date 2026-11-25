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
