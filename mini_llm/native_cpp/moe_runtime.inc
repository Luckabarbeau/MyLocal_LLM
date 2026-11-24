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
                cpu->finish();
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
