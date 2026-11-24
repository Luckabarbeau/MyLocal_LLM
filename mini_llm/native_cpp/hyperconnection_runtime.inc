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
