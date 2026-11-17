// Diagnostic-only captures from the linked pinned graph. No equations changed.
#define main original_driver_unused
#include "qwen4exp_reference.cpp"
#undef main
#include <cstdint>
namespace metadata {
#include "qwen4exp_identity_reference.cpp"
}
struct BlockCapture {
    std::ofstream events;std::string directory;int position=-1;bool failed=false,stopped=false;
    std::set<std::string> names;
};
static std::string logical_name(ggml_tensor *t) {
    if(t->op==GGML_OP_MUL_MAT && t->src[0]) {
        std::string weight=ggml_get_name(t->src[0]);
        if(weight=="blk.3.indexer.q_proj.weight")return "indexer_q_projection-3";
        if(weight=="blk.3.attn_k.weight")return "K_projection-3";
    }
    return ggml_get_name(t);
}
static bool wanted(ggml_tensor *t) {
    const std::string n=logical_name(t);
    if(n=="indexer_q_projection-3" || n=="K_projection-3")return true;
    static const std::set<std::string> names={
        "hc_mixed-0","linear_attn_qkv_mixed-0","z-0","beta-0","alpha-0",
        "conv_state_at-0","state_predelta-0","new_state-0","final_output-0","linear_attn_out-0",
        "hc_mixed-3","Qcur_full-3","Kcur-3","Vcur-3","indexer_k_raw-3","indexer_k-3",
        "indexer_score-3","indexer_score_tokens-3","indexer_top_k-3","attn_gated-3","attn_output-3"};
    if(!names.count(n))return false;
    // HC helper names are genuinely repeated; discriminate through converted gamma.
    if(n=="hc_mixed-0" || n=="hc_mixed-3") {
        int layer=n.back()-'0';std::set<ggml_tensor*> seen;
        std::vector<ggml_tensor*> todo={t};bool attn=false,ffn=false;
        while(!todo.empty()) {
            auto a=todo.back();todo.pop_back();if(!a||!seen.insert(a).second)continue;
            std::string name=ggml_get_name(a);
            if(name=="blk."+std::to_string(layer)+".hc_attn_norm.weight")attn=true;
            if(name=="blk."+std::to_string(layer)+".hc_ffn_norm.weight")ffn=true;
            if(a->op==GGML_OP_RMS_NORM || (a!=t && name.rfind("hc_mixed-",0)==0))continue;
            for(auto s:a->src)todo.push_back(s);
        }
        return attn&&!ffn;
    }
    if(n=="Kcur-3")return t->op==GGML_OP_ROPE;
    if(n=="Vcur-3")return t->ne[0]==256 && t->ne[1]==2;
    return true;
}
static bool capture_blocks(ggml_tensor *t,bool ask,void *opaque) {
    auto &c=*static_cast<BlockCapture*>(opaque);
    if(ask)return c.position>=0 && wanted(t);
    if(!wanted(t) || c.position<0)return true;
    try {
        std::string name=logical_name(t);
        if(c.stopped || !c.names.insert(name).second)throw std::runtime_error("duplicate or late boundary: "+name);
        if((t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_I32) || !ggml_is_contiguous(t) || ggml_nelements(t)>786432)
            throw std::runtime_error("unsupported capture layout: "+name);
        std::vector<unsigned char> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t,data.data(),0,data.size());
        std::string file="p"+std::to_string(c.position)+"_"+name+(t->type==GGML_TYPE_I32?".i32":".f32");
        std::ofstream out(c.directory+"/"+file,std::ios::binary);out.write((char*)data.data(),data.size());
        if(!out)throw std::runtime_error("artifact write failed");
        c.events<<"{\"position\":"<<c.position<<",\"key\":\""<<name<<"\",\"file\":\""<<file<<"\",\"tensor\":";
        metadata::node(c.events,t);c.events<<"}\n";c.events.flush();
        // CPU-only scheduler has one split. Prefix layers have completed all
        // their recurrence/KV writes; later decoder layers cannot feed them.
        if(name=="attn_output-3") {c.stopped=true;return false;}
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';c.failed=true;return false;}
    return true;
}
int main(int argc,char**argv) {
    if(argc!=3)return 2;llama_backend_init();
    BlockCapture c;c.directory=argv[2];c.events.open(c.directory+"/events.jsonl");
    auto mp=llama_model_default_params();ggml_backend_dev_t devices[]={nullptr};
    mp.devices=devices;mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.use_extra_bufts=false;mp.check_tensors=false;
    auto model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
    auto cp=llama_context_default_params();cp.n_ctx=32;cp.n_seq_max=1;cp.n_batch=cp.n_ubatch=1;
    cp.n_threads=cp.n_threads_batch=1;cp.type_k=cp.type_v=GGML_TYPE_F32;
    cp.offload_kqv=cp.op_offload=false;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval=capture_blocks;cp.cb_eval_user_data=&c;
    auto ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
    auto batch=llama_batch_init(1,0,1);int status=0;
    for(int p=0;p<16;++p) {
        c.position=p;c.names.clear();c.stopped=false;
        batch.n_tokens=1;batch.token[0]=p+1;batch.pos[0]=p;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=1;
        status=llama_decode(ctx,batch);llama_synchronize(ctx);
        if(status || c.failed || !c.stopped) {status=5;break;}
    }
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return status||c.failed ? 5:0;
}
