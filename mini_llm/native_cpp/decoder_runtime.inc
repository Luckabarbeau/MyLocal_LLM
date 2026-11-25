// Architecture-aware batch-one reference decoder. Mutable bindings are consumed.
struct DecoderLayer {
    std::shared_ptr<HCWeights> attention,ffn;
    std::shared_ptr<RecurrentLayer> recurrent;
    std::shared_ptr<QSALayer> qsa;
    std::shared_ptr<PLEState> ple;
    std::shared_ptr<MoeOwner> moe;
};
struct DecoderPromptCheckpoint;
struct Decoder {
    std::shared_ptr<DecoderPromptCheckpoint> checkpoint;
    uint64_t checkpoint_epoch=0;
    std::shared_ptr<Engine> engine;
    std::shared_ptr<PayloadLease> embedding;
    std::shared_ptr<UploadedTensor> output;
    std::shared_ptr<HCWeights> final;
    std::shared_ptr<HCState> hc;
    std::vector<DecoderLayer> layers;
    std::vector<float> trace;
    uint64_t capacity=0,position=0,bytes=0;
    int d=0,vocab=0;
    float scale=1;
    bool trace_valid=false,failed=false,counted=false,session_counted=false,valid=false;
    void* device=nullptr;
    unsigned char* packed=nullptr;
    float *row=nullptr,*logits=nullptr;
    int* status=nullptr;
    ~Decoder() noexcept {
        DeviceGuardNoThrow();
    }
    void DeviceGuardNoThrow() noexcept {
        int previous=-1; (void)cudaGetDevice(&previous); (void)cudaSetDevice(engine->device);
        if(cudaStreamSynchronize(engine->stream)!=cudaSuccess) (void)cudaDeviceSynchronize();
        if(device) (void)cudaFree(device);
        if(counted) engine->device_bytes-=bytes;
        if(session_counted) --engine->sessions;
        if(previous>=0) (void)cudaSetDevice(previous);
    }
    void snapshot(size_t layer,int stage,const float* source,uint64_t count) {
        if(trace.empty()) return;
        auto* dst=trace.data()+(layer*4+stage)*4*d;
        cuda_check(cudaMemcpyAsync(dst,source,count*4,cudaMemcpyDeviceToHost,engine->stream));
        cuda_check(cudaStreamSynchronize(engine->stream));
    }
    void mix(DecoderLayer& b,bool ffn) {
        auto w=hc; uint64_t ticket=0,done=0;
        w->begin(); w->ready();
        hc_read_device(w,ffn?b.ffn:b.attention,&ticket);
        if(ffn) {
            auto job=moe_submit_input(b.moe,w->hidden,scale,true); job->finish(false);
        } else if(b.recurrent) recurrent_step_device(b.recurrent,w->hidden,position,&done);
        else qsa_layer_step_device(b.qsa,w->hidden,position,&done);
        w->begin();
        if(!w->pending || w->ticket!=ticket) fail(ML_INTERNAL,"decoder HC ticket lost");
        hc_write_device(w,ffn?b.moe->output:b.recurrent?b.recurrent->output:b.qsa->output);
        if(ffn) b.moe->length=position+1;
    }
    void reset() {
        checkpoint.reset(); ++checkpoint_epoch;
        failed=true; valid=false;
        auto stream=engine->stream;
        for(auto& b:layers) {
            if(b.recurrent) {
                auto w=b.recurrent; w->invalid=true; w->output_valid=false;
                cuda_check(cudaMemsetAsync(w->device,0,w->bytes,stream));
                w->length=0; w->invalid=false;
            } else {
                auto w=b.qsa; w->invalid=true; w->output_valid=false;
                cuda_check(cudaMemsetAsync(w->device,0,w->bytes,stream));
                if(w->paged) w->paged->reset();
                w->length=0; w->invalid=false;
            }
            if(b.ple) {
                auto w=b.ple; w->poisoned=true; w->valid=false;
                cuda_check(cudaMemsetAsync(w->device,0,w->bytes,stream));
                w->position=0; w->predecessors[0]=w->predecessors[1]=-1;
                std::fill_n(w->rows,16,0); w->poisoned=false;
            }
            b.moe->invalid=false; b.moe->result_valid=false; b.moe->length=0;
        }
        cuda_check(cudaMemsetAsync(hc->device,0,hc->bytes,stream));
        cuda_check(cudaMemsetAsync(device,0,bytes,stream));
        cuda_check(cudaStreamSynchronize(stream));
        hc->pending.reset(); hc->ticket=0; hc->initialized=false; hc->collapsed=false;
        hc->hidden_valid=false; hc->invalid=false;
        std::fill(trace.begin(),trace.end(),0); position=0; failed=false;
    }
};
std::unordered_map<ml_handle,std::shared_ptr<Decoder>> decoders;

// Ordinary host snapshots have an explicit per-checkpoint budget. No full QSA
// raw/K/V/pooled prefix is copied; those committed rows cannot be overwritten by
// append-only decode. The raw ring is the only mutable QSA dependency copied.
struct PromptCopy {
    void* device;
    uint64_t offset,bytes;
};
struct DecoderPromptCheckpoint {
    std::weak_ptr<Decoder> owner;
    uint64_t epoch=0,position=0,host_bytes=0;
    ml_handle owner_id=0;
    bool logits_valid=false,fail_restore_for_test=false;
    std::vector<PromptCopy> copies;
    std::vector<unsigned char> data;
    struct PleMetadata {
        int32_t predecessors[2]; uint64_t rows[16];
        float *history=nullptr,*next_history=nullptr;
    };
    std::vector<PleMetadata> ple;
};
std::unordered_map<ml_handle,std::weak_ptr<DecoderPromptCheckpoint>> prompt_checkpoints;

extern "C" int ml_decoder_checkpoint_capture(ml_handle id,uint64_t budget,ml_handle* out,
    uint64_t* host_bytes) noexcept {
    return boundary([&] {
        if(out) *out=0;
        if(host_bytes) *host_bytes=0;
        auto p=lookup(decoders,id); // Unavailable while native worker owns it.
        if(!out || !host_bytes || !budget || p->failed)
            fail(ML_INVALID,"invalid prompt checkpoint capture/session");
        auto c=std::make_shared<DecoderPromptCheckpoint>();
        c->owner=p; c->owner_id=id; c->epoch=p->checkpoint_epoch; c->position=p->position; c->logits_valid=p->valid;
        uint64_t bytes=0;
        auto add=[&](void* device,uint64_t count) {
            c->copies.push_back({device,bytes,count}); bytes+=count;
        };
        for(auto& b:p->layers) {
            if(b.recurrent) {
                auto w=b.recurrent;
                if(w->invalid || w->length!=p->position) fail(ML_INVALID,"checkpoint recurrent position");
                add(w->history,w->history_count*4); add(w->memory,w->memory_count*4);
            } else {
                auto w=b.qsa;
                if(w->invalid || w->length!=p->position || (w->paged && w->paged->stats.length!=p->position))
                    fail(ML_INVALID,"checkpoint QSA position");
                if(w->paged) add(w->raw,640*4);
            }
            if(b.ple) {
                auto w=b.ple;
                if(w->poisoned || w->position!=p->position) fail(ML_INVALID,"checkpoint PLE position");
                add(w->history,92160*4);
                DecoderPromptCheckpoint::PleMetadata meta{};
                meta.history=w->history; meta.next_history=w->next_history;
                std::copy_n(w->predecessors,2,meta.predecessors); std::copy_n(w->rows,16,meta.rows);
                c->ple.push_back(meta);
            }
        }
        if(p->valid) add(p->logits,uint64_t(p->vocab)*4);
        c->host_bytes=bytes+sizeof(DecoderPromptCheckpoint)+c->copies.capacity()*sizeof(PromptCopy)+
            c->ple.capacity()*sizeof(DecoderPromptCheckpoint::PleMetadata);
        if(c->host_bytes>budget) fail(ML_CAPACITY,"prompt checkpoint host budget exhausted");
        // Drop the previous host buffer before allocating its replacement: one
        // bounded retained checkpoint, never two complete state copies.
        p->checkpoint.reset();
        c->data.resize(bytes);
        DeviceGuard guard(p->engine->device); StreamCompletion completion{p->engine->stream};
        for(auto& copy:c->copies) if(copy.bytes)
            cuda_check(cudaMemcpyAsync(c->data.data()+copy.offset,copy.device,copy.bytes,
                cudaMemcpyDeviceToHost,p->engine->stream));
        cuda_check(cudaStreamSynchronize(p->engine->stream));
        auto h=new_handle(); prompt_checkpoints.emplace(h,c); p->checkpoint=c;
        *host_bytes=c->host_bytes; *out=h;
    });
}
extern "C" int ml_decoder_checkpoint_restore(ml_handle id,ml_handle checkpoint,uint64_t* committed) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        auto c=lookup(prompt_checkpoints,checkpoint).lock();
        if(!committed || !c || c->owner.lock()!=p || p->checkpoint!=c ||
           c->epoch!=p->checkpoint_epoch || p->failed || p->position<c->position)
            fail(ML_INVALID,"stale prompt checkpoint/session; reset required");
        for(auto& b:p->layers) {
            if(b.recurrent && (b.recurrent->invalid || b.recurrent->length!=p->position))
                fail(ML_INVALID,"checkpoint recurrent history changed");
            if(b.qsa && (b.qsa->invalid || b.qsa->length!=p->position ||
                (b.qsa->paged && b.qsa->paged->stats.length!=p->position)))
                fail(ML_INVALID,"checkpoint QSA history changed");
            if(b.ple && (b.ple->poisoned || b.ple->position!=p->position))
                fail(ML_INVALID,"checkpoint PLE history changed");
        }
        DeviceGuard guard(p->engine->device); StreamCompletion completion{p->engine->stream};
        // Publish no usable session until all copies, pager truncation and fences
        // succeed. A partial failure quarantines the decoder and checkpoint.
        p->failed=true; p->valid=false; p->trace_valid=false;
        try {
            cuda_check(cudaStreamSynchronize(p->engine->stream));
            for(auto& copy:c->copies) if(copy.bytes) {
                cuda_check(cudaMemcpyAsync(copy.device,c->data.data()+copy.offset,copy.bytes,
                    cudaMemcpyHostToDevice,p->engine->stream));
                if(c->fail_restore_for_test) fail(ML_CUDA,"injected prompt checkpoint restore failure");
            }
            size_t ple_index=0;
            for(auto& b:p->layers) {
                if(b.recurrent) {
                    b.recurrent->length=c->position; b.recurrent->output_valid=false;
                } else {
                    auto w=b.qsa;
                    if(w->paged) w->paged->truncate(c->position);
                    // New pools beyond floor(position/4) are ignored/recomputed;
                    // stale IDs/scores/staging are not diagnostic-valid after restore.
                    cuda_check(cudaMemcpyAsync(w->n,&c->position,8,cudaMemcpyHostToDevice,p->engine->stream));
                    w->length=c->position; w->output_valid=false;
                }
                if(b.ple) {
                    auto& meta=c->ple[ple_index++]; auto w=b.ple;
                    // PLE swaps these buffers each step. Copies above target the
                    // captured allocation; restore its committed/scratch roles too.
                    w->history=meta.history; w->next_history=meta.next_history;
                    std::copy_n(meta.predecessors,2,w->predecessors); std::copy_n(meta.rows,16,w->rows);
                    w->position=c->position; w->valid=false;
                }
                b.moe->length=c->position; b.moe->result_valid=false;
            }
            auto w=p->hc;
            w->pending.reset(); w->ticket=0; w->initialized=false; w->collapsed=false; w->hidden_valid=false;
            cuda_check(cudaStreamSynchronize(p->engine->stream));
            p->position=c->position; p->valid=c->logits_valid; p->failed=false; *committed=p->position;
        } catch(...) { p->checkpoint.reset(); ++p->checkpoint_epoch; throw; }
    });
}
extern "C" int ml_decoder_paged_stats_for_test(ml_handle id,uint64_t layer,ml_kvi8_stats* out) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!out || layer>=p->layers.size() || !p->layers[layer].qsa || !p->layers[layer].qsa->paged)
            fail(ML_INVALID,"decoder paged diagnostic unavailable");
        auto w=p->layers[layer].qsa->paged;
        *out=w->stats; out->pending=w->pending?1:0;
    });
}
extern "C" int ml_decoder_checkpoint_fail_restore_for_test(ml_handle id) noexcept {
    return boundary([&] {
        auto c=lookup(prompt_checkpoints,id).lock();
        if(!c || !decoders.count(c->owner_id)) fail(ML_INVALID,"stale/worker-owned checkpoint");
        c->fail_restore_for_test=true;
    });
}
extern "C" int ml_decoder_checkpoint_readback_for_test(ml_handle id,void* out,uint64_t count,uint64_t* size) noexcept {
    return boundary([&] {
        auto c=lookup(prompt_checkpoints,id).lock();
        if(!c || !size) fail(ML_INVALID,"stale prompt checkpoint diagnostic");
        *size=c->data.size();
        if(!out && !count) return;
        if(!out || count<c->data.size()) fail(ML_INVALID,"checkpoint diagnostic capacity");
        std::memcpy(out,c->data.data(),c->data.size());
    });
}
extern "C" int ml_decoder_checkpoint_destroy(ml_handle id) noexcept {
    return boundary([&] {
        if(!id) return;
        auto c=lookup(prompt_checkpoints,id).lock();
        if(c) if(auto p=c->owner.lock(); p && p->checkpoint==c) {
            if(!decoders.count(c->owner_id)) fail(ML_INVALID,"checkpoint owner handle unavailable during generation");
            p->checkpoint.reset();
        }
        prompt_checkpoints.erase(id);
    });
}
extern "C" int ml_decoder_create(ml_handle engine,ml_handle embedding,ml_handle output,
    ml_handle final,ml_handle hc,const ml_decoder_layer* bindings,uint64_t count,
    uint64_t capacity,float scale,int trace,ml_handle* out) noexcept {
    return boundary([&] {
        if(out) *out=0;
        auto e=lookup(engines,engine);
        if(!out || !bindings || !count || count>48 || !capacity || capacity>262144 ||
           !std::isfinite(scale) || scale<=0 || (trace!=0 && trace!=1))
            fail(ML_INVALID,"invalid decoder configuration");
        if(e->sessions>=e->max_sessions) fail(ML_CAPACITY,"decoder session limit");
        auto p=std::make_shared<Decoder>(); p->engine=e; p->capacity=capacity; p->scale=scale;
        p->embedding=lookup(payload_leases,embedding); p->output=lookup(uploads,output);
        p->final=lookup(hc_weights,final); p->hc=lookup(hc_states,hc); p->d=p->hc->d;
        const auto& ed=p->embedding->descriptor(); const auto& od=p->output->descriptor;
        if(ed.rank!=2 || ed.ggml_shape[0]!=uint64_t(p->d) || ed.type_id!=23 ||
           !ed.ggml_shape[1] || ed.ggml_shape[1]>248320 || p->d%256 ||
           p->output->engine!=e || od.rank!=2 || od.ggml_shape[0]!=ed.ggml_shape[0] ||
           od.ggml_shape[1]!=ed.ggml_shape[1] ||
           (od.type_id!=20 && od.type_id!=8 && od.type_id!=23 && od.type_id!=0 &&
                       od.type_id!=12 && od.type_id!=13 && od.type_id!=14))
            fail(ML_INVALID,"decoder embedding/output geometry or packed format");
        p->vocab=int(ed.ggml_shape[1]);
        auto check_hc=[&](const std::shared_ptr<HCWeights>& w,bool inject) {
            if(w->engine!=e || w->d!=p->d || w->rank>p->hc->rank_capacity || bool(w->values[3])!=inject)
                fail(ML_INVALID,"decoder HC weight mismatch");
        };
        if(p->hc->engine!=e || p->hc->initialized || p->hc->invalid || p->hc->pending)
            fail(ML_INVALID,"decoder requires fresh exclusive HC state");
        check_hc(p->final,false);
        std::unordered_set<ml_handle> mutable_ids{hc};
        for(uint64_t i=0;i<count;++i) {
            const auto& v=bindings[i]; DecoderLayer b;
            b.attention=lookup(hc_weights,v.attention_hc); b.ffn=lookup(hc_weights,v.ffn_hc);
            check_hc(b.attention,true); check_hc(b.ffn,true);
            if(!mutable_ids.insert(v.branch).second || !mutable_ids.insert(v.moe).second ||
               (v.ple && !mutable_ids.insert(v.ple).second)) fail(ML_INVALID,"duplicate decoder mutable binding");
            if(v.kind==1) {
                b.recurrent=lookup(recurrent_layers,v.branch); auto w=b.recurrent;
                if(w->engine!=e || w->d!=p->d || w->length || w->invalid || w->capacity<capacity)
                    fail(ML_INVALID,"decoder recurrent binding mismatch");
            } else if(v.kind==0) {
                b.qsa=lookup(qsa_layers,v.branch); auto w=b.qsa;
                if(w->engine!=e || p->d!=2560 || w->length || w->invalid || w->capacity<capacity)
                    fail(ML_INVALID,"decoder QSA binding mismatch");
            } else fail(ML_INVALID,"decoder branch kind");
            b.moe=lookup(moe_owners,v.moe);
            if(b.moe->engine!=e || b.moe->d!=p->d || b.moe->busy || b.moe->invalid || b.moe->length)
                fail(ML_INVALID,"decoder MoE binding mismatch");
            if(v.ple) {
                b.ple=lookup(ple_states,v.ple); auto w=b.ple;
                if(p->d!=2560 || w->weights->engine!=e || w->position || w->poisoned || w->capacity<capacity)
                    fail(ML_INVALID,"decoder PLE binding mismatch");
            }
            p->layers.push_back(std::move(b));
        }
        p->bytes=ed.row_bytes+4*(uint64_t(p->d)+p->vocab+1);
        e->admit(p->bytes); DeviceGuard guard(e->device); StreamCompletion completion{e->stream};
        if(trace) p->trace.resize(count*4*4*p->d);
        p->output->wait();
        cuda_check(cudaMalloc(&p->device,p->bytes)); e->record(p->bytes,0); p->counted=true;
        p->packed=static_cast<unsigned char*>(p->device);
        p->row=reinterpret_cast<float*>(p->packed+ed.row_bytes);
        p->logits=p->row+p->d; p->status=reinterpret_cast<int*>(p->logits+p->vocab);
        auto id=new_handle(); decoders.emplace(id,p);
        // Publication succeeded: transfer mutable bindings; old aliases become stale.
        hc_states.erase(hc);
        for(uint64_t i=0;i<count;++i) {
            auto& v=bindings[i];
            if(v.kind) recurrent_layers.erase(v.branch); else qsa_layers.erase(v.branch);
            moe_owners.erase(v.moe); if(v.ple) ple_states.erase(v.ple);
        }
        ++e->sessions; p->session_counted=true; *out=id;
    });
}
extern "C" int ml_decoder_destroy(ml_handle id) noexcept {
    return boundary([&] { if(id) { (void)lookup(decoders,id); decoders.erase(id); } });
}
extern "C" int ml_decoder_reset(ml_handle id) noexcept {
    return boundary([&] { auto p=lookup(decoders,id); DeviceGuard g(p->engine->device); p->reset(); });
}
extern "C" int ml_decoder_state(ml_handle id,uint64_t* position,int* failed) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id); if(!position || !failed) fail(ML_INVALID,"null decoder state");
        *position=p->position; *failed=p->failed;
    });
}
void decoder_step_native(const std::shared_ptr<Decoder>& p,int32_t token,float* logits,
    uint64_t count,uint64_t* committed,bool capture_trace=true) {
        if(committed) *committed=0;
        if(p->failed) fail(ML_INVALID,"decoder unresumable; reset required");
        p->valid=false; p->trace_valid=false;
        try {
            if(!committed || (logits ? count<uint64_t(p->vocab) : count!=0) || token<0 || token>=p->vocab)
                fail(ML_INVALID,"decoder token/output capacity");
            if(p->position>=p->capacity) fail(ML_CAPACITY,"decoder active session capacity");
            DeviceGuard guard(p->engine->device); auto stream=p->engine->stream; StreamCompletion completion{stream};
            const auto& ed=p->embedding->descriptor();
            const auto* source=p->embedding->store->maps[ed.shard_index]->data+ed.data_offset+uint64_t(token)*ed.row_bytes;
            cuda_check(cudaMemsetAsync(p->status,0,4,stream));
            cuda_check(cudaMemcpyAsync(p->packed,source,ed.row_bytes,cudaMemcpyHostToDevice,stream));
            cuda_check(ml_qsa_decode_rows_launch(stream,p->packed,23,1,p->d,ed.row_bytes,p->row,p->status));
            auto w=p->hc;
            w->pending.reset(); w->ticket=0; w->collapsed=false; w->hidden_valid=false;
            for(int c=0;c<4;++c) cuda_check(cudaMemcpyAsync(w->streams+c*p->d,p->row,p->d*4,cudaMemcpyDeviceToDevice,stream));
            w->initialized=true; w->invalid=false;
            for(size_t i=0;i<p->layers.size();++i) {
                auto& b=p->layers[i];
                if(b.ple) ple_step_device(b.ple,0,token,p->position,1,w);
                p->mix(b,false); if(capture_trace) { p->snapshot(i,0,w->hidden,p->d); p->snapshot(i,1,w->streams,4*p->d); }
                p->mix(b,true); if(capture_trace) { p->snapshot(i,2,w->hidden,p->d); p->snapshot(i,3,w->streams,4*p->d); }
            }
            w->begin(); w->ready(); auto v=p->final;
            cuda_check(cudaMemsetAsync(w->status,0,4,stream));
            cuda_check(ml_hc_collapse_launch(stream,w->d,v->rank,v->epsilon,w->streams,
                v->values[0],v->values[1],v->values[2],w->normalized,w->low,w->hidden,w->status));
            w->check_status(); w->collapsed=true; w->hidden_valid=true; w->invalid=false;
            const auto& od=p->output->descriptor;
            cuda_check(ml_vocab_projection_launch(stream,p->output->device,od.type_id,p->vocab,p->d,od.row_bytes,
                w->hidden,p->logits,p->status));
            int status=0; cuda_check(cudaMemcpyAsync(&status,p->status,4,cudaMemcpyDeviceToHost,stream));
            cuda_check(cudaStreamSynchronize(stream));
            if(status) fail(ML_INVALID,"decoder nonfinite embedding/logits");
            // Null logits with zero count explicitly keeps the completed vector on device.
            if(logits) {
                cuda_check(cudaMemcpyAsync(logits,p->logits,p->vocab*4,cudaMemcpyDeviceToHost,stream));
                cuda_check(cudaStreamSynchronize(stream));
            }
            ++p->position; p->valid=true; p->trace_valid=capture_trace; *committed=p->position;
        } catch(...) { p->failed=true; throw; }
}
extern "C" int ml_decoder_consume(ml_handle id,const int32_t* tokens,uint64_t count,uint64_t* committed) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!committed || (count && !tokens)) fail(ML_INVALID,"null decoder consume input/output");
        *committed=p->position;
        if(p->failed) fail(ML_INVALID,"decoder unresumable; reset required");
        if(count>p->capacity-p->position) fail(ML_CAPACITY,"decoder consume capacity");
        // Validate the entire vector before any mutation. Execution failures still
        // poison the decoder and retain the last successfully committed position.
        for(uint64_t i=0;i<count;++i)
            if(tokens[i]<0 || tokens[i]>=p->vocab) fail(ML_INVALID,"decoder consume token");
        for(uint64_t i=0;i<count;++i) {
            uint64_t position=0;
            decoder_step_native(p,tokens[i],nullptr,0,&position,false);
            *committed=position;
        }
    });
}
extern "C" int ml_decoder_step(ml_handle id,int32_t token,float* logits,uint64_t count,uint64_t* committed) noexcept {
    return boundary([&] { decoder_step_native(lookup(decoders,id),token,logits,count,committed); });
}
extern "C" int ml_decoder_moe_diagnostic_for_test(ml_handle id,uint64_t layer,int which,void* out,uint64_t count) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!p->valid || !p->trace_valid || p->failed || p->trace.empty() || layer>=p->layers.size())
            fail(ML_INVALID,"decoder MoE diagnostic unavailable");
        moe_diagnostic_copy(p->layers[layer].moe,which,out,count);
    });
}
extern "C" int ml_decoder_trace(ml_handle id,uint64_t layer,int stage,float* out,uint64_t count) noexcept {
    return boundary([&] {
        auto p=lookup(decoders,id);
        if(!p->valid || !p->trace_valid || p->failed || p->trace.empty() || layer>=p->layers.size() || stage<0 || stage>3 || !out)
            fail(ML_INVALID,"decoder trace unavailable");
        uint64_t n=(stage%2?4:1)*p->d; if(count<n) fail(ML_INVALID,"decoder trace capacity");
        std::copy_n(p->trace.data()+(layer*4+stage)*4*p->d,n,out);
    });
}
