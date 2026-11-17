"""Diagnostic-only activation-matched CPU replay; production files are untouched.

This intentionally measures a different arithmetic path, never a production fix.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from probe_0218_hc_identity import COPY, MODEL, provenance, bounded
from probe_0220_layer_evolution import metrics
from probe_0210_qwen4exp_reference import logits_metrics

TYPES={'Q8_0':8,'Q4_0':2,'Q3_K':11,'Q4_K':12,'Q5_K':13,'Q6_K':14,
       'IQ2_S':22,'IQ3_S':21,'IQ3_XXS':18,'IQ4_NL':20,'IQ4_XS':23,'Q2_0':42,'BF16':30}


def compile_codec(directory):
    lib=COPY/'build-cpu/bin'; target=Path(directory)/'activation.so'
    subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',
        str(ROOT/'tests/fixtures/qwen4exp_activation_codec.cpp'),
        '-I'+str(COPY/'ggml/include'),'-I'+str(COPY/'ggml/src'),
        '-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml','-lggml-base','-lggml-cpu','-o',str(target)],
        check=True,capture_output=True,text=True,timeout=60)
    return target


class Codec:
    def __init__(self,path):
        self.lib=ct.CDLL(str(path))
        self.lib.activation_roundtrip.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int]
        self.lib.activation_roundtrip.restype=ct.c_int
        self.counts={}
    def __call__(self,x,weight_type):
        x=np.ascontiguousarray(x,dtype=np.float32)
        y=np.empty_like(x)
        n=x.shape[-1]
        for a,b in zip(x.reshape(-1,n),y.reshape(-1,n)):
            rc=self.lib.activation_roundtrip(int(weight_type),a.ctypes.data,b.ctypes.data,n)
            if rc: raise ValueError(f'upstream activation codec rc={rc}, type={weight_type}, width={n}')
        self.counts[str(weight_type)]=self.counts.get(str(weight_type),0)+x.size//n
        return y


def install_diagnostic_arithmetic(codec):
    from mini_llm.external import qwen4exp_weights as storage
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection, _sigmoid
    from mini_llm.external.qwen4exp_model import Qwen4ExpMoE
    from mini_llm.ops import expert_residency
    from mini_llm.ops.native_packed_cpu import packed_cpu_linear
    original_projection=storage.Qwen4ExpWeights.projection
    def projection(self,name):
        original=original_projection(self,name)
        desc=self.store.tensors[name]
        kind=desc.format.type_id
        def apply(x):
            return original(codec(x,kind))
        return apply
    storage.Qwen4ExpWeights.projection=projection

    def mix(self,streams):
        self._streams(streams)
        norm=streams/np.sqrt(np.mean(streams*streams,axis=-1,keepdims=True,dtype=np.float32)+self.epsilon)*self.gamma
        flat=norm.reshape(streams.shape[:-2]+(4*self.width,))
        low=(codec(flat,30) @ self.down.T)*np.float32(.25)
        low*= _sigmoid(low)
        gate=_sigmoid(codec(low,30) @ self.up.T).reshape(streams.shape)
        return np.mean(norm*gate,axis=-2,dtype=np.float32),flat
    def read(self,streams):
        mixed,flat=self._mix(streams)
        gain=None if self.inject is None else np.float32(2)*_sigmoid((codec(flat,30)@self.inject.T)*np.float32(.25))
        return mixed,gain
    Qwen4ExpHyperConnection._mix=mix
    Qwen4ExpHyperConnection.read=read

    def expert(x,weights,*,backend='native'):
        gate,up,down=weights
        g=packed_cpu_linear(codec(x,TYPES[gate.quant_type]),gate)
        u=packed_cpu_linear(codec(x,TYPES[up.quant_type]),up)
        h=(g*_sigmoid(g))*u
        return packed_cpu_linear(codec(h,TYPES[down.quant_type]),down)
    expert_residency.packed_cpu_expert=expert

    def moe(self,hidden):
        x=hidden.reshape(1,1,-1)
        weights,ids=self.router.forward(codec(x,30))
        routed=self.experts.forward(x,weights*self.scale,ids).reshape(-1)
        g=self.gate(hidden)
        shared=self.down((g*_sigmoid(g))*self.up(hidden))
        return routed+_sigmoid(codec(hidden,30)@self.shared_gate)*shared
    Qwen4ExpMoE.__call__=moe


def child(output,codec_path):
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    codec=Codec(codec_path)
    install_diagnostic_arithmetic(codec)
    p=0
    def callback(layer,stage,value):
        if stage!='logits': np.save(output/f'p{p}_l{layer}_{stage}.npy',value)
    with Qwen4ExpGGUFModel.open(MODEL,max_length=32,cpu_workers=1,diagnostic_callback=callback) as model:
        for name,desc in model.index.store.tensors.items():
            if ('hc_' in name and name.endswith(('down.weight','up.weight','inject.weight'))) or name.endswith(('ffn_gate_inp.weight','ffn_gate_inp_shexp.weight')):
                if desc.quant_type!='BF16': raise ValueError(f'diagnostic BF16 assumption invalid: {name}')
        state=model.create_generation_state(1,32)
        for p,token in enumerate([1,2]):
            np.save(output/f'logits_{p}.npy',model.step(token,state)[0])
    (output/'codec_counts.json').write_text(json.dumps(codec.counts))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--child',type=Path)
    parser.add_argument('--codec',type=Path)
    args=parser.parse_args()
    if args.child:
        child(args.child,args.codec); return
    docs=ROOT/'docs/qwen4exp'
    output=docs/'0222_matched_activations_v1.json'; vectors=docs/'diagnostics_0222_v1'
    if output.exists() or vectors.exists(): raise FileExistsError('preserve prior artifacts')
    report=dict(scope='diagnostic arithmetic intervention, NOT production behavior',provenance=provenance(),
                production_files_modified=False,tokens=[1,2],context=32,threads=1)
    env=dict(os.environ,MINI_LLM_BACKEND='numpy',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    vectors.mkdir()
    with tempfile.TemporaryDirectory(prefix='matched-activation-') as tmp:
        so=compile_codec(tmp)
        report['run']=bounded([sys.executable,str(Path(__file__)),'--child',str(vectors),'--codec',str(so)],
                               time.monotonic()+120,Path(tmp),'matched',env)
    reference=docs/'diagnostics_0220_v2/reference'
    baseline=json.loads((docs/'0220_layer_evolution_v2.json').read_text())
    report['logits']=[]
    for p in range(2):
        report['logits'].append(dict(position=p,baseline_relative_l2=baseline['logits'][p]['relative_l2'],
            **logits_metrics(np.load(vectors/f'logits_{p}.npy'),np.fromfile(reference/f'logits_{p}.f32',np.float32))))
    events=[json.loads(s) for s in (reference/'events.jsonl').read_text().splitlines()]
    from probe_0220_layer_evolution import associate, validate_layout
    identities=associate(events)
    report['boundaries']=[]
    for event in events:
        p,l,s=identities[(event['position'],event['tensor']['ptr'])]
        a=validate_layout(event,np.load(vectors/f'p{p}_l{l}_{s}.npy'),s)
        r=np.fromfile(reference/f'eval_{event["eval_id"]}.f32',np.float32)
        report['boundaries'].append(dict(position=p,layer=l,stage=s,**metrics(a,r)))
    report['codec_counts']=json.loads((vectors/'codec_counts.json').read_text())
    report['interpretation']='Round input activations as the reference does, but retain local FP32 packed dot products and local operators. Remaining difference is not automatically accumulation alone.'
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(logits=report['logits'],first=report['boundaries'][:4],run=report['run']),indent=2))


if __name__=='__main__': main()
