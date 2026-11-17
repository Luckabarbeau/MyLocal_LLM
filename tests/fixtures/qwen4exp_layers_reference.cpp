// Diagnostic-only bounded ancestry; reuse A's tensor metadata utilities.
#define main unused_reference_main
#include "qwen4exp_reference.cpp"
#undef main
#include <cstdint>
namespace identity_utilities {
#include "qwen4exp_identity_reference.cpp"
}
using identity_utilities::node;

struct Layers {
    std::ofstream events;
    std::string dir;
    int position = -1;
    size_t eval = 0;
    bool failed = false;
};
static void nearest(ggml_tensor * t, ggml_tensor * root, std::set<ggml_tensor *> & seen) {
    if (!t || !seen.insert(t).second) return;
    std::string name = ggml_get_name(t);
    if (seen.size() > 512) throw std::runtime_error("ancestry bound exceeded");
    if (t->op == GGML_OP_RMS_NORM ||
        (t != root && (name.rfind("hc_mixed-",0)==0 ||
                       t->op == GGML_OP_DSV4_HC_POST))) return;
    for (auto s : t->src) nearest(s, root, seen);
}
static bool layers_callback(ggml_tensor * t, bool ask, void * opaque) {
    auto & c = *static_cast<Layers *>(opaque);
    const std::string name = ggml_get_name(t);
        // Prefix matches include scheduler-created " (reshaped)" aliases, which
        // are not new mathematical boundaries and must never overwrite captures.
        const bool wanted = selected(name) && (name == "result_norm" ||
            name.find_first_not_of("0123456789", name.find_last_of('-') + 1) == std::string::npos);
    if (ask) return wanted;
    if (!wanted || c.position < 0) return true;
    try {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t)>10240)
            throw std::runtime_error("unsupported boundary layout");
        std::vector<float> values(ggml_nelements(t));
        ggml_backend_tensor_get(t, values.data(), 0, values.size()*sizeof(float));
        auto id = c.eval++;
        save(c.dir+"/eval_"+std::to_string(id)+".f32",values.data(),values.size());
        c.events << "{\"position\":" << c.position << ",\"eval_id\":" << id << ",\"tensor\":";
        node(c.events,t);
        std::set<ggml_tensor *> seen;
        if (t->op == GGML_OP_DSV4_HC_POST) {
            seen.insert(t);
            // Block/residual metadata only: do not dump their entire history.
            for (auto s:t->src) if(s) seen.insert(s);
            std::set<ggml_tensor *> gates;
            nearest(t->src[2],t->src[2],gates);
            seen.insert(gates.begin(),gates.end());
        } else nearest(t,t,seen);
        c.events << ",\"nodes\":[";
        bool comma=false;
        for(auto s:seen) { if(comma)c.events << ','; comma=true; node(c.events,s); }
        c.events << "]}\n"; c.events.flush();
    } catch(const std::exception & e) {
        std::cerr << e.what() << '\n'; c.failed=true; return false;
    }
    return true;
}
int main(int argc,char ** argv) {
    if(argc!=3)return 2;
    llama_backend_init();
    Layers c; c.dir=argv[2]; c.events.open(c.dir+"/events.jsonl");
    auto mp=llama_model_default_params(); ggml_backend_dev_t devices[]={nullptr};
    mp.devices=devices; mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode=LLAMA_LAZY_MODE_ON; mp.use_extra_bufts=false; mp.check_tensors=false;
    auto model=llama_model_load_from_file(argv[1],mp); if(!model)return 3;
    auto cp=llama_context_default_params(); cp.n_ctx=32; cp.n_seq_max=1;
    cp.n_batch=cp.n_ubatch=1; cp.n_threads=cp.n_threads_batch=1;
    cp.type_k=cp.type_v=GGML_TYPE_F32; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=cp.op_offload=false; cp.cb_eval=layers_callback; cp.cb_eval_user_data=&c;
    auto ctx=llama_init_from_model(model,cp); if(!ctx)return 4;
    auto b=llama_batch_init(1,0,1);
    int status=0;
    for(int p=0;p<2;++p) {
        c.position=p; b.n_tokens=1; b.token[0]=p+1; b.pos[0]=p;
        b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=1;
        status=llama_decode(ctx,b); llama_synchronize(ctx);
        if(status || c.failed)break;
        save(c.dir+"/logits_"+std::to_string(p)+".f32",llama_get_logits_ith(ctx,0),
             llama_vocab_n_tokens(llama_model_get_vocab(model)));
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return status || c.failed || c.eval!=386 ? 5 : 0;
}
