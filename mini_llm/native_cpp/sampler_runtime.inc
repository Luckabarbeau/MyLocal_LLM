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
