// SPDX-License-Identifier: LGPL-3.0-or-later
// Private native caller only. All worker arithmetic remains in the supplied bridge.
struct CPUSharedLibrary {
    void* handle=nullptr;
    explicit CPUSharedLibrary(const char* path) {
        if (!path || !*path) fail(ML_INVALID,"explicit CPU library path required");
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if (!handle) fail(ML_INVALID,dlerror());
    }
    ~CPUSharedLibrary() { if(handle) dlclose(handle); }
    CPUSharedLibrary(const CPUSharedLibrary&)=delete;
    template<class T> T symbol(const char* name) {
        dlerror(); void* p=dlsym(handle,name);
        if (const char* error=dlerror()) fail(ML_INVALID,error);
        if (!p) fail(ML_INVALID,"null CPU library symbol");
        return reinterpret_cast<T>(p);
    }
};
struct CPUExpertPool {
    std::shared_ptr<Engine> engine;
    CPUSharedLibrary bridge, ggml;
    decltype(&mylocal_ggml_pool_create) create;
    decltype(&mylocal_ggml_pool_register) reg;
    decltype(&mylocal_ggml_pool_submit_registered) submit;
    decltype(&mylocal_ggml_pool_wait) wait;
    decltype(&mylocal_ggml_pool_destroy) destroy;
    struct Functions { int type, activation; Quantize quantize; Dot dot; };
    std::array<Functions,6> functions;
    std::vector<std::array<std::shared_ptr<PayloadLease>,3>> sources;
    void* pool=nullptr;
    int d, f, capacity;
    bool pending=false, closed=false;
    CPUExpertPool(std::shared_ptr<Engine> e,int width,int ff,int threads,int cap,
                  const char* bp,const char* gp)
        : engine(std::move(e)),bridge(bp),ggml(gp),d(width),f(ff),capacity(cap) {
        create=bridge.symbol<decltype(create)>("mylocal_ggml_pool_create");
        reg=bridge.symbol<decltype(reg)>("mylocal_ggml_pool_register");
        submit=bridge.symbol<decltype(submit)>("mylocal_ggml_pool_submit_registered");
        wait=bridge.symbol<decltype(wait)>("mylocal_ggml_pool_wait");
        destroy=bridge.symbol<decltype(destroy)>("mylocal_ggml_pool_destroy");
        ggml.symbol<void(*)()>("ggml_cpu_init")();
        // Same ABI and activation-type checks as GGMLPackedCPUExecutor._Traits.
        struct Traits { void* from_float; void* vec_dot; int vec_dot_type; int64_t nrows; };
        auto traits=ggml.symbol<const Traits*(*)(int)>("ggml_get_type_traits_cpu");
        constexpr int types[]={22,21,18,23,20,42};
        const char* dots[]={"ggml_vec_dot_iq2_s_q8_K","ggml_vec_dot_iq3_s_q8_K",
            "ggml_vec_dot_iq3_xxs_q8_K","ggml_vec_dot_iq4_xs_q8_K",
            "ggml_vec_dot_iq4_nl_q8_0","ggml_vec_dot_q2_0_q8_0"};
        for (size_t i=0;i<functions.size();++i) {
            int activation=i<4?15:8;
            const auto* t=traits(types[i]);
            if(!t || t->vec_dot_type!=activation || !t->vec_dot)
                fail(ML_INVALID,"incompatible GGML CPU traits");
            functions[i]={types[i],activation,ggml.symbol<Quantize>(
                activation==15?"quantize_row_q8_K":"quantize_row_q8_0"),ggml.symbol<Dot>(dots[i])};
        }
        pool=create(d,f,capacity,threads);
        if(!pool) fail(ML_CAPACITY,"cannot allocate existing GGML row pool");
    }
    void drain() noexcept { if(pending) { wait(pool); pending=false; } }
    void close() noexcept { drain(); closed=true; }
    ~CPUExpertPool() { if(pool) { drain(); destroy(pool); } }
    Projection projection(const std::shared_ptr<PayloadLease>& lease,uint64_t offset,int rows,int columns) {
        const auto& t=lease->descriptor();
        const Functions* fn=nullptr;
        for(const auto& v:functions) if(uint32_t(v.type)==t.type_id) fn=&v;
        if(!fn) fail(ML_INVALID,"unsupported GGML expert format");
        if((t.rank!=2 && t.rank!=3) || t.ggml_shape[0]!=uint64_t(columns) ||
           t.ggml_shape[1]!=uint64_t(rows) || !t.row_bytes ||
           (fn->activation==15 && columns%256) || (fn->activation==8 && columns%32))
            fail(ML_INVALID,"CPU expert projection shape/alignment mismatch");
        const uint64_t bytes=uint64_t(rows)*t.row_bytes;
        if(offset>t.nbytes || bytes>t.nbytes-offset || offset%bytes)
            fail(ML_INVALID,"CPU expert subtensor offset out of bounds/alignment");
        const auto& mapping=lease->store->maps[t.shard_index];
        return {mapping->data+t.data_offset+offset,size_t(t.row_bytes),fn->activation,fn->quantize,fn->dot};
    }
};
struct CPUExpertJob {
    std::shared_ptr<CPUExpertPool> owner;
    std::vector<float> input, output;
    bool complete=false, submitted=false;
    CPUExpertJob(std::shared_ptr<CPUExpertPool> p,const float* x,size_t n)
        : owner(std::move(p)),input(x,x+owner->d),output(n*owner->d) {}
    void finish() noexcept {
        if(submitted && !complete) { owner->drain(); complete=true; }
    }
    ~CPUExpertJob() { finish(); }
};
// Jobs retire before pools, and both before source/engine registries at exit.
std::unordered_map<ml_handle,std::shared_ptr<CPUExpertPool>> cpu_expert_pools;
std::unordered_map<ml_handle,std::shared_ptr<CPUExpertJob>> cpu_expert_jobs;
extern "C" int ml_cpu_experts_create(ml_handle engine,int d,int f,int threads,int capacity,
        const char* bp,const char* gp,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto e=lookup(engines,engine);
        if(!out || d<1 || d>65536 || f<1 || f>65536 || d%32 || f%32 ||
           threads<1 || threads>256 || capacity<1 || capacity>512)
            fail(ML_INVALID,"invalid CPU expert dimensions/threads/capacity/output");
        auto p=std::make_shared<CPUExpertPool>(e,d,f,threads,capacity,bp,gp);
        auto id=new_handle(); cpu_expert_pools.emplace(id,std::move(p)); *out=id;
    });
}
extern "C" int ml_cpu_experts_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { auto p=lookup(cpu_expert_pools,id); p->close(); cpu_expert_pools.erase(id); } });
}
extern "C" int ml_cpu_experts_register(ml_handle id,const ml_handle* leases,
        const uint64_t* offsets,int32_t* index) noexcept {
    return boundary([&] {
        if(index) *index=-1;
        auto p=lookup(cpu_expert_pools,id);
        if(p->pending) fail(ML_INVALID,"CPU expert pool busy");
        if(!leases || !offsets || !index) fail(ML_INVALID,"null CPU expert registration");
        std::array<std::shared_ptr<PayloadLease>,3> sources;
        std::array<Projection,3> descriptors;
        for(int i=0;i<3;++i) {
            sources[i]=lookup(payload_leases,leases[i]);
            descriptors[i]=p->projection(sources[i],offsets[i],i==2?p->d:p->f,i==2?p->f:p->d);
        }
        PoolExpert expert{descriptors[0],descriptors[1],descriptors[2]};
        // Establish ownership before registration, so allocation failure cannot
        // leave a descriptor pointing at an unretained mapping.
        p->sources.push_back(std::move(sources));
        int registered=-1; int status=p->reg(p->pool,&expert,&registered);
        if(status) { p->sources.pop_back(); fail(ML_INVALID,"GGML expert registration rejected"); }
        *index=registered;
    });
}
extern "C" int ml_cpu_experts_submit(ml_handle id,const int32_t* indices,uint64_t count,
        const float* input,uint64_t input_count,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto p=lookup(cpu_expert_pools,id);
        if(p->pending) fail(ML_INVALID,"CPU expert pool busy");
        if(!out || !input || !indices || input_count!=uint64_t(p->d) || !count || count>uint64_t(p->capacity))
            fail(ML_INVALID,"invalid CPU expert batch/input");
        for(uint64_t i=0;i<count;++i)
            if(indices[i]<0 || size_t(indices[i])>=p->sources.size()) fail(ML_INVALID,"invalid registered expert index");
        for(int i=0;i<p->d;++i) if(!std::isfinite(input[i])) fail(ML_INVALID,"nonfinite CPU expert input");
        auto job=std::make_shared<CPUExpertJob>(p,input,size_t(count));
        auto handle=new_handle(); cpu_expert_jobs.emplace(handle,job);
        if(p->submit(p->pool,indices,int(count),job->input.data(),job->output.data())) {
            cpu_expert_jobs.erase(handle); fail(ML_INVALID,"GGML registered submit rejected");
        }
        job->submitted=true; p->pending=true; *out=handle;
    });
}
extern "C" int ml_cpu_job_wait(ml_handle id) noexcept {
    return boundary([&] { lookup(cpu_expert_jobs,id)->finish(); });
}
extern "C" int ml_cpu_job_readback_for_test(ml_handle id,float* output,uint64_t capacity) noexcept {
    return boundary([&] {
        auto j=lookup(cpu_expert_jobs,id);
        if(!output || capacity<j->output.size()) fail(ML_INVALID,"CPU job output capacity too small");
        j->finish(); std::copy(j->output.begin(),j->output.end(),output);
    });
}
extern "C" int ml_cpu_job_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { auto j=lookup(cpu_expert_jobs,id); j->finish(); cpu_expert_jobs.erase(id); } });
}
