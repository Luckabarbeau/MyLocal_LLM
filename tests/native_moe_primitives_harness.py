"""Standalone WP5 GPU primitives: temporary nvcc shim, unchanged Python oracles.

MINI_LLM_BACKEND=cupy .venv/bin/python tests/native_moe_primitives_harness.py
No runtime/bindings/CMake or model checkpoint required. --quick limits dimensions
and oracle repetitions for compute-sanitizer; still covers all seven triples.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import iq_packed_gpu as iq
from mini_llm.ops.quantized_linear import quantized_linear, packed_to_device
from mini_llm.ops.router import expert_topk_forward
from mini_llm.external.qwen4exp_model import Qwen4ExpMoE
from test_0231_grouped_iq_gpu import ACTUAL, handles
from test_0228_iq_gpu_integration import packed

NATIVE = ROOT / 'mini_llm/native_cpp'
TYPES = dict(Q8_0=8, Q4_K=12, Q5_K=13, Q6_K=14, IQ3_XXS=18,
             IQ4_NL=20, IQ3_S=21, IQ2_S=22, IQ4_XS=23, Q2_0=42)
MATRIX = np.dtype([('data', '<u8'), ('row_bytes', '<u8'),
                   ('type', '<u4'), ('reserved', '<u4')], align=True)
EXPERT = np.dtype([(k, MATRIX) for k in ('gate', 'up', 'down')], align=True)
SHIM = r'''
#include "moe.cu"
__global__ void probe_products(MLMoeMatrix m,int rows,int cols,const float* x,float* out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<rows*cols) out[i]=__fmul_rn(x[i%cols],weight_at(m,i/cols,i%cols));
}
extern "C" int products_probe(cudaStream_t s,const MLMoeMatrix* m,int rows,int cols,const float* x,float* out) {
 probe_products<<<(rows*cols+255)/256,256,0,s>>>(*m,rows,cols,x,out);
 return cudaGetLastError();
}
__global__ void probe_decode(MLMoeMatrix m,int rows,int cols,float* out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<rows*cols) out[i]=weight_at(m,i/cols,i%cols);
}
__global__ void probe_project(MLMoeMatrix m,int rows,int cols,const float* x,float* out) {
 int row=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32;
 if(row>=rows) return;
 float v=0;
 for(int j=lane;j<cols;j+=32) v=__fadd_rn(v,__fmul_rn(x[j],weight_at(m,row,j)));
 v=row_sum(v);
 if(!lane) out[row]=v;
}
extern "C" int project_probe(cudaStream_t s,const MLMoeMatrix* m,int rows,int cols,const float* x,float* out) {
 probe_project<<<(rows+3)/4,128,0,s>>>(*m,rows,cols,x,out);
 return cudaGetLastError();
}
extern "C" int decode_probe(cudaStream_t s,const MLMoeMatrix* m,int rows,int cols,float* out) {
 probe_decode<<<(rows*cols+255)/256,256,0,s>>>(*m,rows,cols,out);
 return cudaGetLastError();
}
static_assert(sizeof(MLMoeMatrix)==24 && sizeof(MLMoeExpert)==72,"descriptor ABI");
extern "C" int route(cudaStream_t s,const float* l,float scale,float* p,int* ids,float* w,int* st) {
 return ml_moe_route_launch(s,l,scale,p,ids,w,st);
}
extern "C" int experts(cudaStream_t s,int d,int f,const MLMoeExpert* e,const int* ids,
 const float* w,const float* x,float* h,float* t,float* out,int* st) {
 return ml_moe_experts_launch(s,d,f,e,ids,w,x,h,t,out,st);
}
extern "C" int shared(cudaStream_t s,int d,int f,const MLMoeExpert* e,const float* sg,
 const float* x,float* g,float* u,float* h,float* score,float* raw,float* out,int* st) {
 return ml_moe_shared_launch(s,d,f,*e,sg,x,g,u,h,score,raw,out,st);
}
extern "C" int combine(cudaStream_t s,int d,const float* a,const float* b,float* out,int* st) {
 return ml_moe_combine_launch(s,d,a,b,out,st);
}
'''
ERRORS = {}


def ptr(a):
    return a.data.ptr


def close(a, b, label, exact=False):
    a = cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)
    b = cp.asnumpy(b) if isinstance(b, cp.ndarray) else np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    diff = a.astype(np.float64) - b.astype(np.float64)
    absolute = float(np.max(np.abs(diff)))
    relative = float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30))
    ERRORS[label] = dict(max_absolute=absolute, relative_l2=relative)
    if exact:
        np.testing.assert_array_equal(a, b, err_msg=label)
    else:
        # Same bound as test_0237_warp_iq_gpu, plus 2e-6 for shared native
        # 128-thread vs Python projection reduction/BLAS scalar projection.
        assert relative < 1e-5 or absolute < 2e-6, (label, ERRORS[label])
        assert absolute < 2e-6 + 2e-5 * float(np.max(np.abs(b))), (label, ERRORS[label])


def ok(code):
    assert code == 0, ('CUDA launch', code)


def descriptor(weights, padded=False):
    """Test-only registration: GPU byte payloads and descriptor host record."""
    record = np.zeros(1, EXPERT)
    owners = []
    for key, weight in zip(('gate', 'up', 'down'), weights):
        rb = weight.nbytes // weight.shape[0]
        stride = rb + (7 if padded else 0)
        buf = cp.zeros((weight.shape[0], stride), cp.uint8)
        buf[:, :rb] = cp.asarray(weight.data.reshape(weight.shape[0], rb))
        owners.append(buf)
        record[key]['data'] = ptr(buf)
        record[key]['row_bytes'] = stride
        record[key]['type'] = TYPES[weight.quant_type]
    return record, owners


def route_case(lib, stream, logits, scale, label):
    logits = cp.asarray(logits, cp.float32)
    p, ids, w = cp.empty(512, cp.float32), cp.empty(10, cp.int32), cp.empty(10, cp.float32)
    status = cp.zeros(1, cp.int32)
    ok(lib.route(stream.ptr, ptr(logits), float(scale), ptr(p), ptr(ids), ptr(w), ptr(status)))
    rw, ri, cache = expert_topk_forward(logits, 10)
    rw = (rw / cp.maximum(cp.sum(rw), cp.float32(2**-14))) * cp.float32(scale)
    order = cp.argsort(ri)
    stream.synchronize()
    assert int(status.get()[0]) == 0
    close(ids, ri[order], label + '/ids', exact=True)
    close(p, cache[0], label + '/full512')
    close(w, rw[order], label + '/weights')
    # Selection probabilities retain full-softmax mass, never just top10 softmax.
    mass = float(cp.sum(p).get())
    # Serial FP32 accumulation has gamma_511 ~= 3.05e-5; 8e-6 is a
    # stricter bounded-fixture check, not a demand for pairwise reduction bits.
    assert abs(mass - 1) < 8e-6, (label, 'FP32 mass', mass, 'reference',
                                float(cp.sum(cache[0]).get()), 'tolerance', 8e-6,
                                'error', ERRORS[label + '/full512'])
    return ids, w


def routed_case(lib, stream, formats, d, f, index):
    hosts, device_experts = handles(formats, count=10, d=d, f=f)
    ids_h = np.array([0, 17, 29, 100, 200, 311, 399, 450, 510, 511], np.int32)
    logits = np.full(512, -12, np.float32)
    logits[ids_h] = np.linspace(0.7, 3, 10, dtype=np.float32)[::-1]
    ids, routes = route_case(lib, stream, logits, 1.7, f'triple{index}/router')
    table = np.zeros(512, EXPERT)
    owners = []
    for id_, ws in zip(ids_h, hosts):
        rec, buffers = descriptor(ws, padded=True)
        table[id_] = rec[0]
        owners.extend(buffers)
    gpu_table = cp.asarray(table.view(np.uint8))
    for key, host_weight in zip(('gate', 'up', 'down'), hosts[0]):
        decoded = cp.empty(host_weight.shape, cp.float32)
        m = np.array(table[0][key], dtype=MATRIX)
        ok(lib.decode_probe(stream.ptr, m.ctypes.data, *host_weight.shape, ptr(decoded)))
        close(decoded, host_weight.dequantize_reference(), f'triple{index}/decode/{key}', exact=True)
    x = cp.asarray(np.random.default_rng(590 + index).normal(0, .1, (1, d)).astype(np.float32))
    hidden, terms = cp.empty((10, f), cp.float32), cp.empty((10, d), cp.float32)
    out, status = cp.empty((1, d), cp.float32), cp.zeros(1, cp.int32)
    ok(lib.experts(stream.ptr, d, f, ptr(gpu_table), ptr(ids), ptr(routes), ptr(x),
                   ptr(hidden), ptr(terms), ptr(out), ptr(status)))
    # Unchanged grouped warp oracle: same lane decoder/arithmetic, separate
    # weighted multiplication and ordered accumulation. Host metadata is test-only.
    os.environ['MINI_LLM_IQ_GPU_GROUPED_KERNEL'] = 'warp'
    reference = iq.grouped(x, device_experts, routes)
    reference_terms = iq.grouped_terms(x, device_experts, routes)
    stream.synchronize()
    assert int(status.get()[0]) == 0
    x64 = cp.asnumpy(x).astype(np.float64)
    for key, ws in zip(('gate', 'up'), hosts[0][:2]):
        m = np.array(table[0][key], dtype=MATRIX)
        products = cp.empty((f, d), cp.float32)
        ok(lib.products_probe(stream.ptr, m.ctypes.data, f, d, ptr(x), ptr(products)))
        close(products, ws.dequantize_reference() * cp.asnumpy(x),
              f'triple{index}/products/{key}', exact=True)
        proj = cp.empty(f, cp.float32)
        ok(lib.project_probe(stream.ptr, m.ctypes.data, f, d, ptr(x), ptr(proj)))
        close(proj, (x64 @ ws.dequantize_reference().astype(np.float64).T).reshape(-1),
              f'triple{index}/project/{key}')
    for e in range(10):
        g64, u64 = [x64 @ ws.dequantize_reference().astype(np.float64).T for ws in hosts[e][:2]]
        h64 = (g64 / (1 + np.exp(-g64))) * u64
        close(hidden[e], h64.reshape(-1), f'triple{index}/hidden_dense{e}')
        h = iq.swiglu(x, device_experts[e].gate, device_experts[e].up)
        close(hidden[e], h.reshape(-1), f'triple{index}/hidden{e}')
    close(out, reference, f'triple{index}/warp', exact=True)
    close(terms, reference_terms.reshape(10, d), f'triple{index}/terms', exact=True)
    # Independent unchanged CPU dequantization oracle, bounded synthetic matrices.
    from test_0231_grouped_iq_gpu import dense_oracle
    dense = dense_oracle(cp.asnumpy(x), hosts, cp.asnumpy(routes))
    close(out, dense, f'triple{index}/dense')
    # Status is parent owned and is not cleared by a successful call.
    status.fill(64)
    ok(lib.experts(stream.ptr, d, f, ptr(gpu_table), ptr(ids), ptr(routes), ptr(x),
                   ptr(hidden), ptr(terms), ptr(out), ptr(status)))
    stream.synchronize()
    assert int(status.get()[0]) == 64
    # A selected missing descriptor is detected on GPU, not dereferenced.
    bad = table.copy()
    bad[0]['gate']['data'] = 0
    bad_gpu = cp.asarray(bad.view(np.uint8))
    status.fill(0)
    ok(lib.experts(stream.ptr, d, f, ptr(bad_gpu), ptr(ids), ptr(routes), ptr(x),
                   ptr(hidden), ptr(terms), ptr(out), ptr(status)))
    stream.synchronize()
    assert int(status.get()[0]) & 2
    if index == 0:
        for invalid_ids in ([-1] + ids_h[1:].tolist(), ids_h[:-1].tolist() + [512],
                            [0, 0] + ids_h[2:].tolist(), ids_h[::-1].tolist()):
            bad_ids = cp.asarray(invalid_ids, cp.int32)
            status.fill(0)
            ok(lib.experts(stream.ptr, d, f, ptr(gpu_table), ptr(bad_ids), ptr(routes), ptr(x),
                           ptr(hidden), ptr(terms), ptr(out), ptr(status)))
            stream.synchronize()
            assert int(status.get()[0]) & 2
    # Restore a valid result for shared/combination checks.
    status.fill(0)
    ok(lib.experts(stream.ptr, d, f, ptr(gpu_table), ptr(ids), ptr(routes), ptr(x),
                   ptr(hidden), ptr(terms), ptr(out), ptr(status)))
    return x, out, reference, (owners, gpu_table, ids, routes, hidden, terms, status)


def shared_case(lib, stream, x, routed, routed_ref, d, f, gu, down, score_kind):
    ws = [packed(fmt, n, k, seed=610 + p) for p, (fmt, n, k) in
          enumerate(zip((gu, gu, down), (f, f, d), (d, d, f)))]
    rec, owners = descriptor(ws)
    # Shared GU must remain contiguous for the existing helper. Only down
    # exercises the independently supported physical stride.
    rb = int(rec['down']['row_bytes'][0])
    padded_down = cp.zeros((d, rb + 7), cp.uint8)
    padded_down[:, :rb] = owners[2]
    owners.append(padded_down)
    rec['down']['data'] = ptr(padded_down)
    rec['down']['row_bytes'] = rb + 7
    device = [packed_to_device(w) for w in ws]
    sg = cp.asarray(np.random.default_rng(510).normal(0, .2, d).astype(np.float32))
    if score_kind == 'zero':
        sg.fill(0)
    elif score_kind in ('positive', 'negative'):
        sg = x.reshape(-1) * cp.float32(1e5 if score_kind == 'positive' else -1e5)
    gate, up, hidden = [cp.empty(f, cp.float32) for _ in range(3)]
    score, raw, out = cp.empty(1, cp.float32), cp.empty(d, cp.float32), cp.empty(d, cp.float32)
    status = cp.zeros(1, cp.int32)
    ok(lib.shared(stream.ptr, d, f, rec.ctypes.data, ptr(sg), ptr(x), ptr(gate), ptr(up),
                  ptr(hidden), ptr(score), ptr(raw), ptr(out), ptr(status)))
    # Use the unchanged model shared graph, not a rewritten sigmoid/FFN oracle.
    projections = [lambda a, w=w: quantized_linear(a.reshape(1, -1), w).reshape(-1)
                   for w in device]
    model = Qwen4ExpMoE(None, None, *projections, sg)
    ref_raw, ref_score = model._compute_shared(x.reshape(-1))
    combined_ref = model.combine_gpu_branch(routed_ref, (ref_raw, ref_score))
    combined = cp.empty(d, cp.float32)
    ok(lib.combine(stream.ptr, d, ptr(routed), ptr(out), ptr(combined), ptr(status)))
    stream.synchronize()
    assert int(status.get()[0]) == 0
    label = f'shared/{gu}/{down}/{score_kind}'
    close(raw, ref_raw, label + '/raw')
    close(score, ref_score.reshape(1), label + '/score')
    close(combined, combined_ref.reshape(-1), label + '/combined')
    # Explicit separate scalar gate checks, including exact saturated limits.
    expected_shared = cp.reciprocal(1 + cp.exp(-ref_score)) * ref_raw
    close(out, expected_shared, label + '/gated')
    if score_kind == 'positive':
        close(out, raw, label + '/saturated', exact=True)
    if score_kind == 'negative':
        close(out, cp.zeros_like(out), label + '/saturated', exact=True)
    if score_kind == 'zero':
        close(out, raw * cp.float32(.5), label + '/half', exact=True)
    # Allowed alias, exact no-FMA output addition.
    saved = combined.copy()
    ok(lib.combine(stream.ptr, d, ptr(routed), ptr(out), ptr(out), ptr(status)))
    stream.synchronize()
    close(out, saved, label + '/alias', exact=True)


def run(lib, quick):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        rng = np.random.default_rng(591)
        for name, logits in [
            ('random', rng.normal(0, 3, 512).astype(np.float32)),
            ('all_tied', np.zeros(512, np.float32)),
            ('boundary_ties', np.r_[np.ones(15), np.zeros(497)].astype(np.float32)),
            ('underflow', np.r_[1000, np.full(511, -1000)].astype(np.float32)),
        ]:
            for scale in (0., 1., 1.7):
                route_case(lib, stream, logits, scale, f'router/{name}/{scale}')
        d, f = (256, 256) if quick else (768, 512)
        for i, formats in enumerate(ACTUAL):
            x, out, ref, owners = routed_case(lib, stream, formats, d, f, i)
            stream.synchronize()
        for gu in ('Q4_K', 'Q5_K', 'Q6_K', 'IQ4_XS'):
            for down in ('IQ4_NL', 'Q8_0'):
                for kind in (('zero',) if quick else ('random', 'zero', 'positive', 'negative')):
                    shared_case(lib, stream, x, out, ref, d, f, gu, down, kind)
        # Nonfinite route logits signal device status, not silent valid output.
        p, w = cp.empty(512, cp.float32), cp.empty(10, cp.float32)
        ids, status = cp.empty(10, cp.int32), cp.zeros(1, cp.int32)
        logits = cp.zeros(512, cp.float32)
        logits[0] = cp.nan
        ok(lib.route(stream.ptr, ptr(logits), 1., ptr(p), ptr(ids), ptr(w), ptr(status)))
        stream.synchronize()
        assert int(status.get()[0]) & 1
        assert lib.route(stream.ptr, ptr(logits), float('nan'), ptr(p), ptr(ids), ptr(w), ptr(status)) != 0
        assert lib.experts(stream.ptr, 255, f, *([0] * 8)) != 0
        assert lib.combine(stream.ptr, d, 0, ptr(w), ptr(w), ptr(status)) != 0
    return dict(status='passed', triples=len(ACTUAL), dimensions=[d, f], quick=quick,
                gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
                comparisons=len(ERRORS), max_absolute=max(v['max_absolute'] for v in ERRORS.values()),
                max_relative_l2=max(v['relative_l2'] for v in ERRORS.values()), errors=ERRORS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    assert BACKEND_NAME == 'cupy', 'set MINI_LLM_BACKEND=cupy'
    # Prevent unnoticed drift of the copied, licensed source and codebooks.
    canonical = iq._cuda_source('constant').split('// One CTA per token/output pair.')[0]
    assert (NATIVE / 'moe_formats.cuh').read_text().split('#pragma once\n', 1)[1] == canonical
    assert MATRIX.itemsize == 24 and EXPERT.itemsize == 72
    with tempfile.TemporaryDirectory(prefix='wp5_moe_') as temp:
        path = Path(temp)
        (path / 'shim.cu').write_text(SHIM)
        command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '--fmad=false',
                   '-shared', '-Xcompiler=-fPIC', '-arch=sm_120', '-I', str(NATIVE),
                   str(NATIVE / 'qsa_projection.cu'),
                   str(path / 'shim.cu'), '-o', str(path / 'moe.so')]
        subprocess.run(command, check=True, timeout=120)
        lib = ct.CDLL(str(path / 'moe.so'))
        P, I = ct.c_void_p, ct.c_int
        lib.products_probe.argtypes = [P, P, I, I, P, P]
        lib.products_probe.restype = I
        lib.project_probe.argtypes = [P, P, I, I, P, P]
        lib.project_probe.restype = I
        lib.decode_probe.argtypes = [P, P, I, I, P]
        lib.decode_probe.restype = I
        lib.route.argtypes = [P, P, ct.c_float, P, P, P, P]
        lib.experts.argtypes = [P, I, I] + [P] * 8
        lib.shared.argtypes = [P, I, I] + [P] * 10
        lib.combine.argtypes = [P, I] + [P] * 4
        for name in ('route', 'experts', 'shared', 'combine'):
            getattr(lib, name).restype = I
        report = run(lib, args.quick)
        report['build_command'] = command
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'errors'}, indent=2))


if __name__ == '__main__':
    main()
