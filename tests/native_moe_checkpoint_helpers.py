"""Bounded WP5 setup/oracle shared by chaining tests and the whole-MoE benchmark.

No production arithmetic: use the unchanged model, heterogeneous executor and
packed projection operations. Placement is frozen at the existing prepare seam.
"""
from contextlib import contextmanager
import hashlib

import numpy as np

from mini_llm.ops.quantized_weight import PackedGGUFWeight
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear, PackedExpertFFNInference
from mini_llm.ops.router_inference import RouterInference
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor, ExpertKey
from mini_llm.external.qwen4exp_model import Qwen4ExpMoE

POLICY = 'cpu_ggml_q8_gpu_reference_fp32'
KINDS = {8:'Q8_0',12:'Q4_K',13:'Q5_K',14:'Q6_K',18:'IQ3_XXS',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',42:'Q2_0'}
NAMES = ('ffn_gate_inp.weight','ffn_gate_shexp.weight','ffn_up_shexp.weight',
         'ffn_down_shexp.weight','ffn_gate_inp_shexp.weight','ffn_gate_exps.weight',
         'ffn_up_exps.weight','ffn_down_exps.weight')


def control(lease):
    t=lease.descriptor; raw=lease.read(0,t.nbytes)
    if t.type_id==30:
        return (np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32)
    return np.frombuffer(raw,'<f4' if t.type_id==0 else '<f2').astype(np.float32)


class CheckpointMoE:
    d,f=2560,640

    def __init__(self,stack,engine,store,layer):
        self.engine=engine; self.layer=layer
        self.leases=[stack.enter_context(store.lease(f'blk.{layer}.'+n)) for n in NAMES]
        self.uploads=[stack.enter_context(engine.upload(l)) for l in self.leases[:5]]
        self.router=control(self.leases[0]).reshape(512,self.d)
        self.scalar=control(self.leases[4]); self.cache={}; self.identity=[]
        self.shared=[self.packed(l) for l in self.leases[1:4]]
        self.scale=float(store.metadata.get('qwen4exp.expert_weights_scale',1.0))
        for l in (self.leases[0],self.leases[4]):
            t=l.descriptor
            self.identity.append(dict(name=t.name,ggml_shape=t.ggml_shape,type_id=t.type_id,
                offset=0,nbytes=t.nbytes,sha256=hashlib.sha256(l.read(0,t.nbytes)).hexdigest()))

    def packed(self,lease,offset=0,shape=None):
        t=lease.descriptor; shape=shape or t.shape
        raw=lease.read(offset,shape[0]*t.row_bytes)
        self.identity.append(dict(name=t.name,ggml_shape=t.ggml_shape,type_id=t.type_id,
            offset=offset,nbytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        return PackedGGUFWeight(np.frombuffer(raw,np.uint8).copy(),shape,KINDS[t.type_id])

    def expert(self,i):
        if i not in self.cache:
            self.cache[i]=tuple(self.packed(l,i*(self.d if p==2 else self.f)*l.descriptor.row_bytes,
                (self.d,self.f) if p==2 else (self.f,self.d)) for p,l in enumerate(self.leases[5:]))
        return self.cache[i]

    def native(self,stack,placement,libraries,threads=4):
        return stack.enter_context(self.engine.moe(*self.uploads,*self.leases[5:],
            d_model=self.d,d_ff=self.f,gpu_experts=placement,threads=threads,**libraries))

    def route(self,cp,x):
        router=RouterInference(self.d,512,10,renormalize_selected=True,normalization_floor=2**-14)
        router._decode_fastpath_enabled=False
        router.W_router=cp.asarray(self.router.T.copy()); router.b_router=cp.zeros(512,cp.float32)
        return router,router.forward(cp.asarray(x).reshape(1,1,-1))

    @contextmanager
    def oracle(self,cp,placement,threads=4):
        router,_=self.route(cp,np.zeros(self.d,np.float32))
        devices={i:PackedExpertFFNInference(*(packed_to_device(w) for w in self.expert(i))) for i in placement}
        executor=HeterogeneousExpertExecutor(lambda key:self.expert(key.expert_id),model_id='wp5-static',
            d_model=self.d,d_ff=self.f,compute_dtype='float32',cpu_workers=threads,
            cpu_packed_backend='ggml_pool',group_after_admission=True,group_mixed_decode=True)
        executor._prepare=lambda layer,i,x:(ExpertKey('wp5-static',layer,i),devices.get(i),
            None if i in devices else self.expert(i),None)
        experts=ExpertsInference(self.d,self.f,512,10,expert_executor=executor,layer_id=self.layer)
        callbacks=[]
        for w in self.shared:
            dev=packed_to_device(w)
            callbacks.append(lambda x,w=dev:quantized_linear(x.reshape(1,-1),w).reshape(-1))
        model=Qwen4ExpMoE(router,experts,*callbacks,cp.asarray(self.scalar),scale=self.scale)
        try:
            yield model,executor
        finally:
            executor.close()
