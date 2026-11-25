// Lock order: runtime -> generation registry at publication only. Queue waits,
// cancel, registry lookup, and join never hold the runtime mutex.
struct Generation {
    std::shared_ptr<Decoder> decoder;
    std::shared_ptr<Sampler> sampler;
    ml_handle decoder_id=0,sampler_id=0;
    std::atomic<bool> cancelled{false};
    std::mutex queue_mutex,close_mutex;
    std::condition_variable changed;
    std::vector<ml_generation_event> queue;
    std::unordered_set<int32_t> eos;
    size_t head=0,size=0;
    uint64_t maximum=0,k=0;
    double temperature=0,top_p=1;
    int32_t input=0;
    bool done=false,delivered=false,closed=false,running=false;
    ml_generation_event terminal{},published{};
    std::thread worker;
    void cancel() noexcept { cancelled.store(true); changed.notify_all(); }
    ~Generation() noexcept {
        cancel();
        if(worker.joinable()) {
            worker.join();
            // Static teardown may overlap another generation on the same engine.
            // Serialize final owner destruction/accounting with its GPU work.
            std::lock_guard lock(mutex); decoder.reset(); sampler.reset();
        }
    }
    void run() noexcept {
        ml_generation_event state{};
        state.position=decoder->position; state.pending_token=-1;
        int reason=ML_GEN_NONE;
        try {
            for(;;) {
                {
                    std::unique_lock lock(queue_mutex);
                    changed.wait(lock,[&] { return cancelled.load() || size<queue.size(); });
                }
                if(cancelled.load()) { reason=ML_GEN_CANCELLED; break; }
                if(state.sampled>=maximum) { reason=ML_GEN_MAX_TOKENS; break; }
                if(state.position>=decoder->capacity) { reason=ML_GEN_CAPACITY; break; }
                {
                    std::lock_guard lock(mutex);
                    // A cancellation observed before dispatch does not consume input.
                    if(cancelled.load()) { reason=ML_GEN_CANCELLED; break; }
                    { std::lock_guard queue_lock(queue_mutex); running=true; }
                    uint64_t committed=0;
                    decoder_step_native(decoder,input,nullptr,0,&committed,false);
                    state.position=committed;
                    state.pending=0; state.pending_token=-1;
                    sampler->valid=false;
                    sampler_sample_native(sampler,decoder->logits,temperature,top_p,k,0,0,&state.sample);
                    input=state.sample.token; ++state.sampled;
                    state.pending=1; state.pending_token=input;
                }
                state.kind=1;
                {
                    std::lock_guard lock(queue_mutex);
                    queue[(head+size)%queue.size()]=state; ++size; published=state; running=false;
                }
                changed.notify_all();
                // Completion of running work is published even when cancel raced it.
                if(eos.count(input)) { reason=ML_GEN_EOS; break; }
                if(state.sampled==maximum) { reason=ML_GEN_MAX_TOKENS; break; }
                if(state.position==decoder->capacity) { reason=ML_GEN_CAPACITY; break; }
            }
        } catch(const Failure& f) {
            reason=ML_GEN_ERROR; state.status=f.status;
            std::snprintf(state.error,sizeof(state.error),"%s",error_text);
        } catch(const std::exception& e) {
            reason=ML_GEN_ERROR; state.status=ML_INTERNAL;
            std::snprintf(state.error,sizeof(state.error),"%s",e.what());
        } catch(...) {
            reason=ML_GEN_ERROR; state.status=ML_INTERNAL;
            std::snprintf(state.error,sizeof(state.error),"unknown generation exception");
        }
        state.kind=2; state.reason=reason; state.failed=decoder->failed;
        {
            std::lock_guard lock(queue_mutex); terminal=state; done=true; running=false;
        }
        changed.notify_all();
    }
};
std::mutex generation_mutex;
std::unordered_map<ml_handle,std::shared_ptr<Generation>> generations;

template<class F> int generation_boundary(F&& fn) noexcept {
    error_text[0]='\0';
    try { fn(); return ML_OK; }
    catch(const Failure& f) { return f.status; }
    catch(const std::exception& e) {
        std::snprintf(error_text,sizeof(error_text),"%s",e.what()); return ML_INTERNAL;
    }
    catch(...) { std::snprintf(error_text,sizeof(error_text),"unknown generation exception"); return ML_INTERNAL; }
}
std::shared_ptr<Generation> generation_lookup(ml_handle id) {
    std::lock_guard lock(generation_mutex); return lookup(generations,id);
}
extern "C" int ml_generation_start(ml_handle decoder,ml_handle sampler,int32_t input,
    uint64_t maximum,const int32_t* eos,uint64_t eos_count,double temperature,
    double top_p,uint64_t k,uint64_t capacity,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto d=lookup(decoders,decoder); auto s=lookup(samplers,sampler);
        if(!out || !capacity || capacity>4096 || eos_count>uint64_t(d->vocab) ||
           (eos_count && !eos) || input<0 || input>=d->vocab || d->failed ||
           d->engine!=s->engine || d->vocab!=s->n ||
           !std::isfinite(temperature) || temperature<0 || !std::isfinite(float(temperature)) ||
           (temperature>0 && float(temperature)==0) || !std::isfinite(top_p) || top_p<=0 || top_p>1)
            fail(ML_INVALID,"invalid generation configuration/owners");
        auto p=std::make_shared<Generation>(); p->decoder=d; p->sampler=s;
        p->decoder_id=decoder; p->sampler_id=sampler; p->input=input;
        p->maximum=maximum; p->temperature=temperature; p->top_p=top_p; p->k=k;
                p->published.position=d->position; p->published.pending_token=-1;
        p->queue.resize(capacity);
        for(uint64_t i=0;i<eos_count;++i) {
            if(eos[i]<0 || eos[i]>=d->vocab) fail(ML_INVALID,"invalid generation EOS ID");
            p->eos.insert(eos[i]);
        }
        auto id=new_handle();
        std::lock_guard registry_lock(generation_mutex);
        generations.emplace(id,p);
        try { p->worker=std::thread([raw=p.get()] { raw->run(); }); }
        catch(...) { generations.erase(id); throw; }
        // Worker cannot enter GPU work until this boundary releases runtime mutex.
        decoders.erase(decoder); samplers.erase(sampler); *out=id;
    });
}
extern "C" int ml_generation_state(ml_handle id,ml_generation_event* out,uint64_t* queued,int32_t* done) noexcept {
    return generation_boundary([&] {
        if(!out || !queued || !done) fail(ML_INVALID,"null generation state output");
        auto p=generation_lookup(id); std::lock_guard lock(p->queue_mutex);
        *out=p->done?p->terminal:p->published; out->in_flight=p->running;
                *queued=p->size; *done=p->done;
    });
}
extern "C" int ml_generation_cancel(ml_handle id) noexcept {
    return generation_boundary([&] { generation_lookup(id)->cancel(); });
}
extern "C" int ml_generation_poll(ml_handle id,uint64_t timeout,ml_generation_event* out) noexcept {
    return generation_boundary([&] {
        if(!out || timeout>120000) fail(ML_INVALID,"invalid generation poll output/timeout");
        *out={}; auto p=generation_lookup(id);
        std::unique_lock lock(p->queue_mutex);
        p->changed.wait_for(lock,std::chrono::milliseconds(timeout),[&] { return p->size || p->done; });
        if(p->size) { *out=p->queue[p->head]; p->head=(p->head+1)%p->queue.size(); --p->size; }
        else if(p->done && !p->delivered) { *out=p->terminal; p->delivered=true; }
        lock.unlock(); p->changed.notify_all();
    });
}
extern "C" int ml_generation_close(ml_handle id,ml_generation_event* out) noexcept {
    return generation_boundary([&] {
        if(!out) fail(ML_INVALID,"null generation close output");
        *out={}; if(!id) return;
        auto p=generation_lookup(id);
        std::lock_guard closer(p->close_mutex);
        if(p->closed) fail(ML_INVALID,"generation already closed");
        p->cancel();
        if(p->worker.joinable()) p->worker.join();
        {
            std::lock_guard lock(p->queue_mutex); *out=p->terminal; p->size=0;
        }
        {
            std::lock_guard lock(mutex);
            decoders.emplace(p->decoder_id,p->decoder);
            try { samplers.emplace(p->sampler_id,p->sampler); }
            catch(...) { decoders.erase(p->decoder_id); throw; }
            p->decoder.reset(); p->sampler.reset();
        }
        p->closed=true;
        { std::lock_guard lock(generation_mutex); generations.erase(id); }
    });
}
