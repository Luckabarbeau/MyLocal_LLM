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
