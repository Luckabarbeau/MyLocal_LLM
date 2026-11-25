"""Bounded synthetic mixed decoder built from existing WP3/4/7 fixtures."""
from mini_llm.native import Runtime
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
from test_0201_gguf_store import write_gguf
from test_0228_iq_gpu_integration import packed
from test_native_engine_hc import make_weights, upload
from test_native_engine_recurrent import fixture as recurrent_fixture
from test_native_engine_qsa_layer import synthetic as qsa_fixture
from test_native_engine_ple import synthetic as ple_fixture


def mixed_setup(stack, path, libraries, fmt='reference_fp32', capacity=16):
    r = Runtime()
    e = stack.enter_context(r.engine(budget_bytes=512*2**20))
    d, f = 2560, 256
    # Sparse zero-valued valid packed weights. Only 10 CPU rows are read per
    # layer; no resident/decoded expert bank or full model allocation.
    layouts = [('router',(d,512),0,d*512*4),
               ('sg',(d,f),23,d*f//256*136), ('su',(d,f),23,d*f//256*136),
               ('sd',(f,d),20,f*d//32*18), ('scalar',(d,),0,d*4),
               ('bg',(d,f,512),23,d*f*512//256*136),
               ('bu',(d,f,512),23,d*f*512//256*136),
               ('bd',(f,d,512),20,d*f*512//32*18)]
    descriptors, offset = [], 0
    for name, shape, kind, size in layouts:
        offset = (offset+31)//32*32
        descriptors.append((name,shape,kind,offset,None))
        offset += size
    file = path/'sparse_moe.gguf'
    base = write_gguf(file, descriptors)
    with file.open('r+b') as stream:
        stream.truncate(base+offset)
    store = stack.enter_context(r.gguf(file))
    leases = [stack.enter_context(store.lease(name)) for name, *_ in layouts]
    weights = [stack.enter_context(e.upload(lease)) for lease in leases[:5]]
    def moe():
        return stack.enter_context(e.moe(*weights,*leases[5:],d_model=d,d_ff=f,
            gpu_experts=(),threads=1,**libraries))
    a, _, _ = make_weights(stack,path,r,e,d,5)
    b, _, _ = make_weights(stack,path,r,e,d,5,seed=5)
    final, _, _ = make_weights(stack,path,r,e,d,5,inject=False,seed=6)
    qsa_tensors, _ = qsa_fixture(stack,path,r,e,'ggml_cpu')
    qsa = stack.enter_context(e.qsa_layer(capacity,*qsa_tensors,policy='ggml_cpu',
        kv_policy=fmt,page_size=4,physical_pages=1,host_budget_bytes=2**20,raw_diagnostics=True))
    ple_weights, ref, _, _, _, _, _ = ple_fixture(stack,path,r,e,rows=4096)
    ref.close()
    ple = stack.enter_context(ple_weights.state(capacity))
    spec = Qwen4ExpMixerSpec(d,2,2,8,4,1e-6)
    recurrent_tensors, _ = upload(stack,path,r,e,recurrent_fixture(spec))
    rec = stack.enter_context(e.recurrent_layer(spec,capacity,*recurrent_tensors,arithmetic='reference_fp32'))
    embedding, output = packed('IQ4_XS',8,d,41), packed('IQ4_NL',8,d,42)
    file = path/'ends.gguf'
    write_gguf(file,[('embedding',(d,8),23,0,embedding.data.tobytes()),
        ('output',(d,8),20,len(embedding.data.tobytes()),output.data.tobytes())])
    ends = stack.enter_context(r.gguf(file))
    emb = stack.enter_context(ends.lease('embedding'))
    out = stack.enter_context(e.upload(stack.enter_context(ends.lease('output'))))
    hc = stack.enter_context(e.hc_state(d,5))
    decoder = stack.enter_context(e.decoder(emb,out,final,hc,
        [(a,qsa,b,moe(),ple),(a,rec,b,moe(),None)],capacity=capacity,trace=True))
    return decoder, r, e
