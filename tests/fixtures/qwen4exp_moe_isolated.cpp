// Diagnostic graph using the pinned GGML operators; no quantized kernels copied.
#include <cmath>
#define main arithmetic_driver_unused
#include "qwen4exp_arithmetic_reference.cpp"
#undef main
#include <map>
#include <limits>
int main(int argc,char **argv) {
 try {
    if(argc!=3)throw std::runtime_error("directory route|experts");
    std::string dir=argv[1],mode=argv[2];int d,f,e,k,gt,ut,dt,sgt,sut,sdt;
    std::ifstream spec(dir+"/spec.txt");spec>>d>>f>>e>>k>>gt>>ut>>dt>>sgt>>sut>>sdt;
    if(!spec || d<=0 || d>2560 || f<=0 || f>640 || e<k || e>512 || k<=0 || k>10)throw std::runtime_error("bounded geometry");
    auto backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,1);
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    std::map<std::string,ggml_tensor *> inputs,outputs;
    auto tensor=[&](std::string name,int type,int n,int m,int bank=1) {
        auto t=ggml_new_tensor_3d(ctx,(ggml_type)type,n,m,bank);inputs[name]=t;return t;
    };
    auto x=tensor("input",GGML_TYPE_F32,d,1);
    ggml_tensor *final=nullptr;
    if(mode=="route") {
        auto router=tensor("router",GGML_TYPE_BF16,d,e);
        auto logits=ggml_mul_mat(ctx,router,x);
        auto probs=ggml_soft_max(ctx,logits);
        auto ids=ggml_argsort_top_k(ctx,probs,k);
        auto selected=ggml_get_rows(ctx,ggml_reshape_3d(ctx,probs,1,e,1),ids);
        auto weights=ggml_reshape_2d(ctx,selected,k,1);
        weights=ggml_div(ctx,weights,ggml_clamp(ctx,ggml_sum_rows(ctx,weights),6.103515625e-5f,INFINITY));
        outputs={{"route_logits",logits},{"probabilities",probs},{"ids",ids},{"weights",weights}};final=weights;
    } else if(mode=="experts") {
        auto gate=tensor("gate",gt,d,f,k),up=tensor("up",ut,d,f,k),down=tensor("down",dt,f,d,k);
        auto ids=tensor("local_ids",GGML_TYPE_I32,k,1);
        auto weights=tensor("weights",GGML_TYPE_F32,1,k);
        auto g=ggml_mul_mat_id(ctx,gate,x,ids),u=ggml_mul_mat_id(ctx,up,x,ids);
        auto h=ggml_swiglu_split(ctx,g,u);
        auto y=ggml_mul_mat_id(ctx,down,h,ids);
        auto weighted=ggml_mul(ctx,y,weights);
        auto routed=ggml_view_2d(ctx,weighted,d,1,weighted->nb[2],0);
        for(int i=1;i<k;++i)routed=ggml_add(ctx,routed,ggml_view_2d(ctx,weighted,d,1,weighted->nb[2],i*weighted->nb[1]));
        auto sg=tensor("shared_gate",sgt,d,f),su=tensor("shared_up",sut,d,f),sd=tensor("shared_down",sdt,f,d);
        auto sr=tensor("shared_router",GGML_TYPE_BF16,d,1);
        auto gs=ggml_mul_mat(ctx,sg,x),us=ggml_mul_mat(ctx,su,x);
        auto hs=ggml_swiglu_split(ctx,gs,us),ys=ggml_mul_mat(ctx,sd,hs);
        auto scalar=ggml_mul_mat(ctx,sr,x),gain=ggml_sigmoid(ctx,scalar);
        auto shared=ggml_mul(ctx,ys,gain);final=ggml_add(ctx,routed,shared);
        outputs={{"gate",g},{"up",u},{"swiglu",h},{"down",y},{"weighted",weighted},{"routed",routed},
                 {"shared_gate",gs},{"shared_up",us},{"shared_swiglu",hs},{"shared_down",ys},
                 {"shared_router",scalar},{"shared_gain",gain},{"shared",shared},{"final",final}};
    } else throw std::runtime_error("unknown mode");
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,final);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)throw std::runtime_error("allocation");
    for(auto item:inputs) {
        std::vector<unsigned char> bytes(ggml_nbytes(item.second));load((dir+"/"+item.first+".bin").c_str(),bytes.data(),bytes.size());
        ggml_backend_tensor_set(item.second,bytes.data(),0,bytes.size());
    }
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("graph compute");
    for(auto item:outputs) {
        auto t=item.second;
        if(!ggml_is_contiguous(t))throw std::runtime_error("noncontiguous output "+item.first);
        std::vector<unsigned char> bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
        save(dir+"/ref_"+item.first+(t->type==GGML_TYPE_I32?".i32":".f32"),bytes.data(),bytes.size());
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
 }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
