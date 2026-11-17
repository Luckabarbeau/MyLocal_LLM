"""Bounded B isolation. Never loads a model graph or decodes a full weight bank."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_0218_hc_identity import COPY, MODEL, provenance


def compile_reference(directory):
    binary = Path(directory) / 'arithmetic'
    lib = COPY / 'build-cpu/bin'
    subprocess.run(['c++', '-std=c++17', '-O2', '-include', 'cmath',
        str(ROOT/'tests/fixtures/qwen4exp_arithmetic_reference.cpp'),
        '-I'+str(COPY/'ggml/include'), '-I'+str(COPY/'ggml/src'),
        '-L'+str(lib), '-Wl,-rpath,'+str(lib), '-lggml', '-lggml-base', '-lggml-cpu',
        '-o', str(binary)], check=True, timeout=120)
    return binary


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return dict(maxabs=float(np.max(np.abs(a-b))), relative_l2=float(np.linalg.norm(a-b)/max(bn,1e-30)),
                cosine=float(np.dot(a,b)/max(an*bn,1e-30)), rms=float(np.sqrt(np.mean(a*a))),
                norm=float(an), reference_norm=float(bn))


def inputs(n):
    rng = np.random.default_rng(219+n)
    real = np.load(ROOT/'docs/qwen4exp/diagnostics_0218_v1/hc_attention_read.npy').reshape(-1)
    baseline = real if n == real.size else (np.sin(np.arange(n)*0.173)*np.logspace(-3,2,n)).astype(np.float32)
    return [('real_A' if n == real.size else 'deterministic_width_substitute', baseline.astype(np.float32)),
            ('random_seed_'+str(219+n), rng.normal(size=n).astype(np.float32)),
            ('multirange_seed_'+str(219+n), (rng.normal(size=n)*np.logspace(-5,3,n)).astype(np.float32))]


def run_tile(binary, directory, type_id, packed, x, rows=32, synthetic=False):
    directory = Path(directory)
    x.astype(np.float32).tofile(directory/'x'); np.asarray(packed,dtype=np.uint8).tofile(directory/'w')
    p = directory/'result'
    call = subprocess.run([str(binary),str(type_id),str(x.size),str(rows),str(directory/'w'),
                           str(directory/'x'),str(p),str(int(synthetic))], check=True,
                          capture_output=True,text=True,timeout=60)
    traits = json.loads(call.stdout)
    w = np.fromfile(str(p)+'.w',np.float32).reshape(rows,x.size)
    xq = np.fromfile(str(p)+'.xq',np.float32)
    xr = np.fromfile(str(p)+'.xr',np.float32)
    y = np.fromfile(str(p)+'.y',np.float32)
    oracle = w.astype(np.float64) @ x.astype(np.float64)
    quantized = w.astype(np.float64) @ xq.astype(np.float64)
    dense32 = w @ x
    return dict(traits=traits, input_rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))),
        input_min=float(x.min()),input_max=float(x.max()),input_abs_dynamic_range=float(np.max(np.abs(x))/max(np.min(np.abs(x[x!=0])),1e-30)),
        fp32_vs_fp64=metric(dense32,oracle), activation_rounding_vs_fp64=metric(quantized,oracle),
        graph_vs_fp64=metric(y,oracle), graph_vs_rounded_oracle=metric(y,quantized),
        graph_vs_fp32=metric(y,dense32), activation_roundtrip=metric(xq,x),
        reference_vs_cpu_activation=metric(xr,xq), finite=bool(np.isfinite(y).all()))


def main():
    from mini_llm.external.gguf_store import MultiShardTensorStore
    paths = [ROOT/'docs/qwen4exp/activation_quantization_comparison.json',ROOT/'docs/qwen4exp/0219_activation_quantization_v1.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('B artifacts exist; refusing overwrite')
    report = dict(scope='B bounded arithmetic only; not full model parity',production_math_changed=False,
                  provenance=provenance(),threads=1,backend='CPU direct allocation; no scheduler/repack',
                  input_dtype='F32',rows_per_tile=32,results=[])
    with MultiShardTensorStore.open(MODEL) as store, tempfile.TemporaryDirectory(prefix='activation-0219-') as tmp:
        # Inspect metadata and descriptors before choosing or reading any payload.
        report['metadata'] = {k:v for k,v in store.metadata.items() if k.startswith('qwen4exp.') and not isinstance(v,list)}
        report['shards'] = [str(p) for p in store.paths]
        selected = ['blk.17.ffn_gate_exps.weight','blk.17.ffn_up_exps.weight',
                    'blk.0.ffn_gate_exps.weight','blk.1.ffn_gate_exps.weight',
                    'blk.0.ffn_down_exps.weight','blk.1.ffn_down_exps.weight',
                    'blk.0.attn_gate.weight','blk.0.attn_qkv.weight','blk.1.attn_qkv.weight',
                    'blk.0.ffn_up_shexp.weight','blk.0.ffn_down_shexp.weight',
                    'blk.4.ffn_up_shexp.weight','blk.0.ssm_out.weight',
                    'blk.0.hc_attn_down.weight','blk.0.hc_attn_up.weight']
        binary = compile_reference(tmp)
        total = 0
        for name in selected:
            d = store.tensors[name]
            expert = 511 if len(d.shape)==3 else None
            raw = store.rows(name,0,32,outer=() if expert is None else (expert,)).copy()
            total += raw.nbytes
            for source,x in inputs(d.shape[-1]):
                result = run_tile(binary,tmp,d.format.type_id,raw,x)
                report['results'].append(dict(tensor=name,format=d.quant_type,shape=d.shape,expert=expert,
                    row_range=[0,32],input_source=source,shard=d.shard_index,
                    packed_byte_offset=d.data_offset+(expert or 0)*d.byte_strides[0],**result))
        report['packed_bytes_read'] = total
    report['hc_audit'] = dict(source='src/models/qwen4exp.cpp:build_hc_mix; ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_dsv4_hc_pre_f32',
        chain='F32 grouped RMSNorm * gamma -> BF16 down matmul (activation BF16) -> scale 1/4 -> F32 SiLU -> BF16 up matmul (activation BF16) -> F32 fused sigmoid(gate)*xn stream sum /4',
        fused_mixing_dtype='F32; no BF16 cast inside fused HC pre',
        limitation='HC down/up are separate bounded row tiles on substitute inputs, not a composed HC replay. No retained pre-HC normalized streams/gates available; cannot attribute earliest HC read divergence from these tiles alone.',
        fp32_baseline='D(BF16 W) x in F32 and FP64; activation rounding isolated per matmul, not full HC mixed baseline')
    summary = dict(scope=report['scope'],hc_audit=report['hc_audit'],
        activation_types={r['format']:r['traits']['activation_type'] for r in report['results']},
        observations=[dict(tensor=r['tensor'],input_source=r['input_source'],
                           rounding_relative_l2=r['activation_rounding_vs_fp64']['relative_l2'],
                           graph_relative_l2=r['graph_vs_fp64']['relative_l2'],
                           kernel_relative_l2=r['graph_vs_rounded_oracle']['relative_l2']) for r in report['results']],
        limitations=['32 output rows only; no full-bank decode or full-model run',
                     'Descriptive measurements, no FP32-equals-GGML assertion or broad pass claim',
                     'Different-width inputs are substitutes, not real intermediate activations'])
    for path,data in zip(paths,[summary,report]):
        with path.open('x') as f: json.dump(data,f,indent=2)
    print(json.dumps(dict(cases=len(report['results']),packed_bytes_read=report['packed_bytes_read'],activation_types=summary['activation_types']),indent=2))

if __name__ == '__main__':
    main()
