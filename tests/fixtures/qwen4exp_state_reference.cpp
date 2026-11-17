// Sampled persistent-state diagnostics, CPU-only linked independent graph.
#define main original_driver_unused
#include "qwen4exp_reference.cpp"
#undef main
#include <cstdint>
namespace metadata {
#include "qwen4exp_identity_reference.cpp"
}
struct States { std::ofstream events; std::string dir; int position=-1; bool failed=false; };
static bool state_callback(ggml_tensor * t,bool ask,void * opaque) {
    auto & c=*static_cast<States *>(opaque); std::string name=ggml_get_name(t);
    bool wanted=name=="state_predelta-0" || name=="conv_state_at-0" || name=="conv_state_at-1" ||
        (name=="Kcur-3" && t->op==GGML_OP_ROPE) ||
                (name=="Vcur-3" && t->ne[0]==256 && t->ne[1]==2) || name=="indexer_k_raw-3" || name=="ple_embd";
    if(ask)return wanted;
    if(!wanted || c.position<0)return true;
    try {
        if(t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t)>786432)
            throw std::runtime_error("state capture layout unavailable: "+name);
        std::string key=name;
        if(name=="conv_state_at-1") key+=t->ne[0]==9 ? "_ple" : "_gdn";
        std::vector<float> data(ggml_nelements(t)); ggml_backend_tensor_get(t,data.data(),0,data.size()*4);
        std::string file="p"+std::to_string(c.position)+"_"+key+".f32";
        save(c.dir+"/"+file,data.data(),data.size());
        c.events<<"{\"position\":"<<c.position<<",\"file\":\""<<file<<"\",\"tensor\":";
        metadata::node(c.events,t); c.events<<"}\n"; c.events.flush();
        if(name=="ple_embd") {
            auto gather=t;
                        while(gather && (gather->op==GGML_OP_RESHAPE || gather->op==GGML_OP_VIEW || gather->op==GGML_OP_CONT)) gather=gather->src[0];
                        auto ids=gather && gather->op==GGML_OP_GET_ROWS ? gather->src[1] : nullptr;
            if(!ids || ids->type!=GGML_TYPE_I32 || ggml_nelements(ids)!=16 || !ggml_is_contiguous(ids))
                throw std::runtime_error("PLE row IDs unavailable");
            std::vector<int32_t> rows(16); ggml_backend_tensor_get(ids,rows.data(),0,64);
            std::ofstream out(c.dir+"/p"+std::to_string(c.position)+"_ple_indices.i32",std::ios::binary);
            out.write((char *)rows.data(),64);
        }
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n'; c.failed=true;return false;}
    return true;
}
int main(int argc,char ** argv) {
    if(argc!=3)return 2; llama_backend_init();
    States s; s.dir=argv[2]; s.events.open(s.dir+"/events.jsonl");
    auto mp=llama_model_default_params(); ggml_backend_dev_t devices[]={nullptr};
    mp.devices=devices; mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.use_extra_bufts=false;mp.check_tensors=false;
    auto model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
    auto cp=llama_context_default_params();cp.n_ctx=32;cp.n_seq_max=1;cp.n_batch=cp.n_ubatch=1;
    cp.n_threads=cp.n_threads_batch=1;cp.type_k=cp.type_v=GGML_TYPE_F32;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;
    cp.cb_eval=state_callback;cp.cb_eval_user_data=&s;
    auto ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
    auto b=llama_batch_init(1,0,1);int status=0;
    // Fifth token exposes pre-state after the four-token fixed prefix.
    for(int p=0;p<5;++p) {
        s.position=p;b.n_tokens=1;b.token[0]=p+1;b.pos[0]=p;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
        status=llama_decode(ctx,b);llama_synchronize(ctx);if(status||s.failed)break;
        save(s.dir+"/logits_"+std::to_string(p)+".f32",llama_get_logits_ith(ctx,0),
             llama_vocab_n_tokens(llama_model_get_vocab(model)));
    }
    llama_batch_free(b);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return status||s.failed ? 5:0;
}
