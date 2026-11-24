// Private orchestration only: all arithmetic remains in the validated operators.
// Called under the parent boundary lock; shared owners outlive stream completion.
void hc_mix_impl(ml_handle id, ml_handle weights, ml_handle branch,
                 int kind, uint64_t position, float scale, uint64_t* committed) {
        if (committed) *committed=0;
        auto w=lookup(hc_states,id); w->begin(); w->ready();
        if (!committed) fail(ML_INVALID,"null HC mix committed output");
        if (w->pending) fail(ML_INVALID,"HC pending read must be written before mix");
        auto v=lookup(hc_weights,weights);
        if (v->engine!=w->engine || v->d!=w->d || v->rank>w->rank_capacity || !v->values[3])
            fail(ML_INVALID,"HC mix weight engine/shape/rank/injection mismatch");
        std::shared_ptr<QSALayer> q;
        std::shared_ptr<RecurrentLayer> r;
        std::shared_ptr<MoeOwner> m;
        if (kind==0) q=lookup(qsa_layers,branch);
        else if (kind==1) r=lookup(recurrent_layers,branch);
        else if (kind==2) m=lookup(moe_owners,branch);
        else fail(ML_INVALID,"invalid HC mix branch kind");
        if ((q && (q->engine!=w->engine || w->d!=2560 || q->invalid)) ||
            (r && (r->engine!=w->engine || r->d!=w->d || r->invalid)) ||
            (m && (m->engine!=w->engine || m->d!=w->d || m->invalid)))
            fail(ML_INVALID,"HC mix branch engine/width/invalid mismatch; reset required");
        if(m && m->busy) fail(ML_INVALID,"MoE owner busy");
        if(!std::isfinite(scale) || scale<0 || (!m && scale!=1)) fail(ML_INVALID,"invalid HC branch scale");
        const uint64_t length=q?q->length:r?r->length:m->length;
        const uint64_t capacity=q?q->capacity:r?r->capacity:std::numeric_limits<uint64_t>::max();
        if (position!=length) fail(ML_INVALID,"HC mix position mismatch");
        if (position>=capacity) fail(ML_CAPACITY,"HC mix capacity exhausted");
        DeviceGuard guard(w->engine->device); StreamCompletion completion{w->engine->stream};
        uint64_t ticket=0, done=0;
        try {
            hc_read_device(w,v,&ticket);
            if (q) { q->output_valid=false; qsa_layer_step_device(q,w->hidden,position,&done); }
            else if(r) { r->output_valid=false; recurrent_step_device(r,w->hidden,position,&done); }
            else {
                auto job=moe_submit_input(m,w->hidden,scale,true);
                job->finish(false); // Status fence only; HC consumes the native device output.
                done=position+1;
            }
            // The exact read's gains are consumed once, without exporting its ticket.
            w->begin();
            if (!w->pending || w->ticket!=ticket) fail(ML_INTERNAL,"HC mix lost read ticket");
            hc_write_device(w,q?q->output:r?r->output:m->output);
            if(m) m->length=done;
            *committed=done;
        } catch (...) {
            w->invalid=true; w->hidden_valid=false;
            // A branch may already have committed before HC write failed. Quarantine
            // both owners rather than presenting a resumable partial combined step.
            if (q) { q->invalid=true; q->output_valid=false; }
            if (r) { r->invalid=true; r->output_valid=false; }
            if (m) m->invalid=true;
            throw;
        }
}
extern "C" int ml_hc_mix(ml_handle id, ml_handle weights, ml_handle branch,
                          int kind, uint64_t position, uint64_t* committed) noexcept {
    return boundary([&] { hc_mix_impl(id,weights,branch,kind,position,1.0f,committed); });
}
extern "C" int ml_hc_mix_scaled(ml_handle id, ml_handle weights, ml_handle branch,
    int kind, uint64_t position, float scale, uint64_t* committed) noexcept {
    return boundary([&] { hc_mix_impl(id,weights,branch,kind,position,scale,committed); });
}
