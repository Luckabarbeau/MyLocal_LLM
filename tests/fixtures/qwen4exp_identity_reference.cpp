// Diagnostic only: reuse API setup utilities, never the occurrence association.
#define main occurrence_harness_main
#include "qwen4exp_reference.cpp"
#undef main
#include <cstdint>

struct Identity {
    std::ofstream events;
    std::string dir;
    size_t occurrence = 0;
    std::map<ggml_tensor *, size_t> counts;
};
static uint64_t addr(const void * p) { return reinterpret_cast<uintptr_t>(p); }
static void node(std::ostream & o, ggml_tensor * t) {
    o << "{\"ptr\":" << addr(t) << ",\"name\":\"" << ggml_get_name(t)
      << "\",\"op\":\"" << ggml_op_name(t->op) << "\",\"type\":" << int(t->type)
      << ",\"data\":" << addr(t->data) << ",\"view_src\":" << addr(t->view_src)
      << ",\"view_offs\":" << t->view_offs << ",\"buffer\":" << addr(t->buffer)
      << ",\"buffer_name\":\"" << (t->buffer ? ggml_backend_buffer_name(t->buffer) : "")
      << "\",\"ne\":[";
    for (int i=0;i<4;++i) o << (i ? "," : "") << t->ne[i];
    o << "],\"nb\":[";
    for (int i=0;i<4;++i) o << (i ? "," : "") << t->nb[i];
    o << "],\"src\":[";
    for (int i=0;i<GGML_MAX_SRC;++i) o << (i ? "," : "") << addr(t->src[i]);
    o << "]}";
}
static void ancestry(ggml_tensor * t, std::set<ggml_tensor *> & seen) {
    if (!t || !seen.insert(t).second) return;
    for (auto s : t->src) ancestry(s, seen);
}
static bool identity_callback(ggml_tensor * t, bool ask, void * opaque) {
    auto & c = *static_cast<Identity *>(opaque);
    bool wanted = std::string(ggml_get_name(t)) == "hc_mixed-0";
    if (ask) return wanted;
    if (!wanted) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t)>65536)
        return false;
    // Own the bytes now: graph allocator may alias this storage later.
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size()*sizeof(float));
    auto id = c.occurrence++;
    save(c.dir+"/eval_"+std::to_string(id)+".f32",values.data(),values.size());
    c.events << "{\"eval_occurrence\":" << id << ",\"pointer_occurrence\":" << c.counts[t]++
             << ",\"tensor\":";
    node(c.events,t);
    c.events << ",\"nodes\":[";
    std::set<ggml_tensor *> seen; ancestry(t,seen);
    bool comma=false;
    for (auto s:seen) { if(comma)c.events << ','; comma=true; node(c.events,s); }
    c.events << "]}\n"; c.events.flush();
    return true;
}
int main(int argc, char ** argv) {
    if(argc!=3) return 2;
    llama_backend_init();
    Identity c; c.dir=argv[2]; c.events.open(c.dir+"/events.jsonl");
    auto mp=llama_model_default_params(); ggml_backend_dev_t devices[]={nullptr};
    mp.devices=devices; mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode=LLAMA_LAZY_MODE_ON; mp.use_extra_bufts=false; mp.check_tensors=false;
    auto model=llama_model_load_from_file(argv[1],mp); if(!model)return 3;
    auto cp=llama_context_default_params(); cp.n_ctx=32; cp.n_seq_max=1;
    cp.n_batch=cp.n_ubatch=1; cp.n_threads=cp.n_threads_batch=1;
    cp.type_k=cp.type_v=GGML_TYPE_F32; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=cp.op_offload=false; cp.cb_eval=identity_callback; cp.cb_eval_user_data=&c;
    auto ctx=llama_init_from_model(model,cp); if(!ctx)return 4;
    auto b=llama_batch_init(1,0,1); b.n_tokens=1; b.token[0]=1; b.pos[0]=0;
    b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=1;
    int status=llama_decode(ctx,b); llama_synchronize(ctx);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return status || !c.occurrence ? 5 : 0;
}
