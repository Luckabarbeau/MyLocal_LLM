// Linked upstream codecs and CPU graph only; no local quantized dot implementation.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
static void load(const char * p, void * d, size_t n) {
    std::ifstream f(p, std::ios::binary); if (!f.read((char *)d,n)) throw std::runtime_error("short input");
}
static void save(const std::string & p, const void * d, size_t n) {
    std::ofstream f(p,std::ios::binary); f.write((const char *)d,n); if (!f) throw std::runtime_error("output failure");
}
int main(int argc, char ** argv) {
 try {
    if (argc != 8) throw std::runtime_error("type width rows packed input output_prefix synthetic");
    auto type = (ggml_type)std::stoi(argv[1]); int n=std::stoi(argv[2]), m=std::stoi(argv[3]);
    if (n<=0 || n>10240 || m<=0 || m>32) throw std::runtime_error("tile budget");
    ggml_backend_t backend=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend,1);
    auto wt=ggml_get_type_traits(type); auto cpu=ggml_get_type_traits_cpu(type);
    auto at=cpu->vec_dot_type; auto act=ggml_get_type_traits(at); auto acpu=ggml_get_type_traits_cpu(at);
    if (!wt->to_float || !acpu->from_float) throw std::runtime_error("missing codec");
    std::vector<unsigned char> w(ggml_row_size(type,n)*m), q(ggml_row_size(at,n)), qr(q.size());
    std::vector<float> x(n), xd(n), xr(n), dense(n*m), y(m);
    load(argv[5],x.data(),n*4);
    if (std::stoi(argv[7])) {
        std::vector<float> seed(n*m); for (int i=0;i<n*m;++i) seed[i]=0.1f*std::sin(i*0.37f);
        // Quantization initialization/codebooks are upstream-owned.
        ggml_quantize_init(type);
        ggml_quantize_chunk(type,seed.data(),w.data(),0,m,n,nullptr);
    } else load(argv[4],w.data(),w.size());
    for(int r=0;r<m;++r) wt->to_float(w.data()+r*ggml_row_size(type,n),dense.data()+r*n,n);
    acpu->from_float(x.data(),q.data(),n);
    if (act->from_float_ref) act->from_float_ref(x.data(),qr.data(),n); else qr=q;
    auto decode=[&](const void * p,float * d) {
        if(at==GGML_TYPE_Q8_K) dequantize_row_q8_K((const block_q8_K *)p,d,n);
        else if(act->to_float) act->to_float(p,d,n);
        else if(at==GGML_TYPE_F32) std::memcpy(d,p,n*4);
        else throw std::runtime_error("missing activation decoder");
    };
    decode(q.data(),xd.data()); decode(qr.data(),xr.data());
    ggml_init_params ip={2*1024*1024,nullptr,true}; auto ctx=ggml_init(ip);
    auto a=ggml_new_tensor_2d(ctx,type,n,m); auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,n,1);
    auto out=ggml_mul_mat(ctx,a,b); auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if (!buffer) throw std::runtime_error("allocation failure");
    ggml_backend_tensor_set(a,w.data(),0,w.size()); ggml_backend_tensor_set(b,x.data(),0,n*4);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("graph failure");
    ggml_backend_tensor_get(out,y.data(),0,m*4);
    std::string p=argv[6]; save(p+".w",dense.data(),dense.size()*4); save(p+".xq",xd.data(),n*4);
    save(p+".xr",xr.data(),n*4); save(p+".y",y.data(),m*4);
    std::cout << "{\"weight_type\":\"" << ggml_type_name(type) << "\",\"activation_type\":\"" << ggml_type_name(at)
              << "\",\"reference_quantizer_available\":" << (act->from_float_ref?"true":"false")
              << ",\"optimized_reference_bytes_equal\":" << (q==qr?"true":"false") << "}\n";
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
 } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
