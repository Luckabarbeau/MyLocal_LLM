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
