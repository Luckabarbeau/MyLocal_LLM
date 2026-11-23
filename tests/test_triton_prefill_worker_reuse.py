"""Native Triton orchestration: worker ownership, ring wrap and row accounting."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
triton = pytest.importorskip('triton')
import triton.language as tl
from mini_llm.ops.triton_packed_prefill import _weight_tile


@triton.jit
def _decode_tile_test(W,OUT,G2S,G3S,G3X,KSG,KV,K:tl.constexpr,QK:tl.constexpr,QB:tl.constexpr,FMT:tl.constexpr):
    row=tl.program_id(0)*8+tl.arange(0,8)
    col=tl.program_id(1)*64+tl.arange(0,64)
    value=_weight_tile(W,G2S,G3S,G3X,KSG,KV,row[None,:],col[:,None],K,QK,QB,FMT)
    tl.store(OUT+row[None,:]*K+col[:,None],value)

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy required')


@pytest.mark.parametrize('kind',['IQ2_S','IQ3_S','IQ3_XXS','IQ4_XS','IQ4_NL','Q2_0'])
@pytest.mark.parametrize('width',[256,2560])
def test_wide_dequant_tile_matches_reference_and_original(kind,width):
    from mini_llm.ops import triton_prefill_experts as module
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS
    from test_qwen4exp_chunked_prefill import gpu_packed
    weight=gpu_packed(kind,7,width,7010)
    source=xp.asarray(weight.data)
    results=[]
    qk,qb=BLOCK_FORMATS[kind]
    books=module._codebooks()
    for tile in (32,128):
        out=xp.empty((7,width),xp.float16)
        module._dequant_kernel[(7,width//tile,1)](module._ptr(source),module._ptr(out),
            *books,weight.nbytes,N=7,K=width,QK=qk,QB=qb,FMT=module._FORMAT_ID[kind],TILE=tile)
        results.append(asnumpy(out))
    expected=weight.dequantize_reference().astype(np.float16)
    np.testing.assert_array_equal(results[0],expected)
    np.testing.assert_array_equal(results[1],expected)


@pytest.mark.parametrize('arithmetic', ['fp16', 'native_int8'])
@pytest.mark.parametrize('gate_kind,down_kind', [('IQ3_S','IQ4_NL'),('IQ2_S','Q2_0'),
    ('IQ3_XXS','Q2_0'),('IQ4_XS','IQ4_NL')])
def test_existing_executor_workers_survive_chunk_dispatch(monkeypatch,gate_kind,down_kind,arithmetic):
    from mini_llm.ops import triton_prefill_experts as module
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    from test_qwen4exp_chunked_prefill import gpu_packed
    native_int8 = arithmetic == 'native_int8'
    if native_int8:
        monkeypatch.setenv('MINI_LLM_TRITON_ARITHMETIC', arithmetic)
    else:
        monkeypatch.delenv('MINI_LLM_TRITON_ARITHMETIC', raising=False)
    monkeypatch.setenv('MINI_LLM_TRITON_FUSED_DEQUANT','0')
    monkeypatch.setenv('MINI_LLM_TRITON_REUSE_WORKERS','1')
    monkeypatch.setenv('MINI_LLM_TRITON_WAVE', '2')
    monkeypatch.setenv('MINI_LLM_TRITON_THREADS', '2')
    monkeypatch.setenv('MINI_LLM_TRITON_BM', '16')
    # Native waves 1, 2 and 4 are empty; active waves still wrap the 3-slot ring.
    n_experts = 14 if native_int8 else 8
    sources = [(gpu_packed(gate_kind,640,2560,6100+i),
                gpu_packed(gate_kind,640,2560,6200+i),
                gpu_packed(down_kind,2560,640,6300+i)) for i in range(n_experts)]
    executor = object.__new__(HeterogeneousExpertExecutor)
    executor.model_id = 'triton-worker-test'
    executor.execution_mode = 'mixed'
    executor.routing_profiler = None
    executor.cache = SimpleNamespace(_admission_limit=0, idle_eviction_requests=False)
    executor.counters = Counter()
    executor.dispatch_scope = lambda *args: nullcontext()
    resolved = []
    def source_weights(key):
        resolved.append(key.expert_id)
        return sources[key.expert_id]
    executor.cpu_source_weights = source_weights
    rng = np.random.default_rng(5080)
    hidden = rng.normal(0,.01,(1,17,2560)).astype(np.float32)
    ids = np.array([[(i+j)%8 for j in range(4)] for i in range(17)],np.int64)
    if native_int8:
        # Non-expert-sorted router rank, including ranks split across waves.
        active_ids = np.array([13, 0, 7, 10, 1, 12])
        ids = active_ids[np.array([[(i+j) % 6 for j in range(4)] for i in range(17)])]
        routes = rng.uniform(.02, .7, (17, 4)).astype(np.float32) * np.float32(1.75)
    else:
        routes = np.full((17,4), .25,np.float32)
    experts = SimpleNamespace(expert_executor=executor,layer_id=0,n_experts=n_experts)
    x,w,i = xp.asarray(hidden),xp.asarray(routes[None]),xp.asarray(ids[None])
    def forbidden(*args,**kwargs):
        pytest.fail('Production dispatch created a per-layer thread pool')
    monkeypatch.setattr(module,'ThreadPoolExecutor',forbidden)
    quant_calls, linear_calls, uploads = [], [], []
    if native_int8:
        from mini_llm.ops import triton_native_prefill as native
        quantize, linear = native.quantize_native, native.native_linear
        upload = module._StageRing.upload
        def track_quantize(a):
            quant_calls.append(a.shape)
            return quantize(a)
        def track_linear(q, weights, counts, offsets, **kwargs):
            linear_calls.append((q.codes.shape[0], asnumpy(counts), asnumpy(offsets),
                                 kwargs['kind'], q.packed is None))
            return linear(q, weights, counts, offsets, **kwargs)
        def track_upload(ring, slot):
            uploads.append((slot, ring.device_used[slot]))
            return upload(ring, slot)
        monkeypatch.setattr(native, 'quantize_native', track_quantize)
        monkeypatch.setattr(native, 'native_linear', track_linear)
        monkeypatch.setattr(module._StageRing, 'upload', track_upload)
        empty = xp.empty
        def no_fp16_empty(shape, dtype=float, *args, **kwargs):
            assert np.dtype(dtype) != np.dtype(np.float16), 'native allocated FP16 storage'
            return empty(shape, dtype, *args, **kwargs)
        monkeypatch.setattr(xp, 'empty', no_fp16_empty)
        monkeypatch.setattr(module, '_dequant_kernel', forbidden)
        monkeypatch.setattr(module, '_gateup_kernel', forbidden)
        monkeypatch.setattr(module, '_down_kernel', forbidden)
    with ThreadPoolExecutor(max_workers=2) as pool:
        executor._pool = pool
        first = module.try_triton_prefill_experts(experts,x,w,i)
        monkeypatch.setenv('MINI_LLM_TRITON_FUSED_DEQUANT','1')
        second = module.try_triton_prefill_experts(experts,x,w,i)
        xp.cuda.get_current_stream().synchronize()
        a,b=asnumpy(first),asnumpy(second)
        print('materialized_vs_fused',gate_kind,down_kind,'dtype',a.dtype,
              'max_abs',float(np.max(np.abs(a-b))),
              'relative_l2',float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)))
        # Fusion changes compiler scheduling of FP32 dot/epilogue operations.
        # This is FP16-policy qualification, not a bitwise graph assertion:
        # require error below one FP16 epsilon at both global scales.
        delta=a.astype(np.float64)-b.astype(np.float64)
        epsilon=float(np.finfo(np.float16).eps)
        assert np.linalg.norm(delta) <= epsilon*max(np.linalg.norm(a.astype(np.float64)),1e-30)
        assert np.max(np.abs(delta)) <= epsilon*max(float(np.max(np.abs(a))),1e-30)
        assert pool.submit(lambda: 42).result()==42
    assert executor.counters['prefill_triton_rows']==2*17*4
    assert executor.counters['prefill_routed_rows']==2*17*4
    assert executor.counters['prefill_triton_waves']==8
    assert executor.counters['prefill_triton_layers']==2
    assert executor.counters['prefill_triton_experts']==2*len(np.unique(ids))
    assert executor.counters['prefill_triton_native_int8_layers']==(2 if native_int8 else 0)
    assert executor.counters['prefill_triton_streamed_bytes']==2*sum(
        weight.nbytes for e in np.unique(ids) for weight in sources[e])
    assert resolved == list(np.unique(ids))*2
    assert bool(xp.isfinite(first).all())
    if native_int8:
        assert quant_calls == [(17, 2560)]*2
        assert len(linear_calls) == 2*4*3  # three products per nonempty wave
        assert sum(call[0] for call in linear_calls) == 2*17*4*3
        for rows, counts, offsets, kind, unpacked in linear_calls:
            np.testing.assert_array_equal(offsets, np.r_[0, counts.cumsum()])
            assert offsets[-1] == rows and np.all(counts > 0)
        assert all(call[4] for j, call in enumerate(linear_calls) if j % 3 != 2)
        assert uploads == [(0, False), (1, False), (2, False),
                                   (0, True), (1, False), (2, False), (0, True)]*2
        monkeypatch.setattr(native, 'quantize_native', quantize)
        monkeypatch.setattr(native, 'native_linear', linear)
        expected = _native_perexpert_reference(hidden, ids, routes, sources)
        np.testing.assert_array_equal(a.view(np.uint32), asnumpy(expected[None]).view(np.uint32))
        np.testing.assert_array_equal(b.view(np.uint32), asnumpy(expected[None]).view(np.uint32))
        return
    # Independent policy reference: FP16 weights and activations, FP32 dot
    # accumulation, FP16 hidden between the two products. Routing adds by ID.
    reference = np.zeros((17,2560),np.float32)
    x16 = xp.asarray(hidden[0],dtype=xp.float16).astype(xp.float32)
    for expert_id,trio in enumerate(sources):
        rows,slots=np.nonzero(ids==expert_id)
        g,u,d=[xp.asarray(weight.dequantize_reference().astype(np.float16)).astype(xp.float32)
               for weight in trio]
        gate=x16[rows]@g.T
        up=x16[rows]@u.T
        h=(gate/(1+xp.exp(-gate))*up).astype(xp.float16).astype(xp.float32)
        value=asnumpy(h@d.T)
        reference[rows]+=routes[rows,slots,None]*value
    epsilon=float(np.finfo(np.float16).eps)
    ref64=reference.astype(np.float64)
    for result in (first[0],second[0]):
        delta=asnumpy(result).astype(np.float64)-ref64
        relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(ref64),1e-30))
        print('policy_reference_relative_l2',gate_kind,down_kind,relative_l2)
        assert relative_l2 <= epsilon
        assert np.max(np.abs(delta)) <= epsilon*max(float(np.max(np.abs(ref64))),1e-30)


@pytest.mark.parametrize('kind',['IQ2_S','IQ3_S','IQ3_XXS','IQ4_XS','IQ4_NL','Q2_0'])
def test_fused_weight_tile_matches_materialized_fp16(kind):
    from mini_llm.ops import triton_prefill_experts as module
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS
    from test_qwen4exp_chunked_prefill import gpu_packed
    weight=gpu_packed(kind,8,2560,7010)
    source=xp.asarray(weight.data)
    out=xp.empty((8,2560),xp.float16)
    qk,qb=BLOCK_FORMATS[kind]
    _decode_tile_test[(1,40)](module._ptr(source),module._ptr(out),*module._codebooks(),
        K=2560,QK=qk,QB=qb,FMT=module._FORMAT_ID[kind])
    np.testing.assert_array_equal(asnumpy(out),weight.dequantize_reference().astype(np.float16))


def test_copy_stream_waits_for_last_staged_weight_reader():
    from mini_llm.ops.triton_prefill_experts import _StageRing
    ring=_StageRing([1024,1024,1024],1,slots=1)
    stream=xp.cuda.get_current_stream()
    snapshots=[]
    try:
        for n in range(12):
            if ring.filled[0]:
                ring.h2d[0].synchronize()
            ring.host.fill(n)
            ring.filled[0]=True
            ring.upload(0)
            stream.wait_event(ring.h2d[0])
            snapshots.append(ring.device.copy())
            ring.record_consumed(0,stream)
        stream.synchronize()
        for n,snapshot in enumerate(snapshots):
            np.testing.assert_array_equal(asnumpy(snapshot),np.full(ring.total,n,np.uint8))
    finally:
        stream.synchronize()
        ring.device_stream.synchronize()


_COMBINE_ORACLE = r'''
extern "C" __global__ void combine(const float* values, const int* slots,
    const float* weights, float* out, int tokens, int width, int topk) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i >= tokens*width) return;
    int t=i/width, d=i%width;
    float sum=0.f;
    for(int k=0;k<topk;++k)
        sum=fmaf(weights[t*topk+k], values[slots[t*topk+k]*width+d], sum);
    out[i]=sum;
}
'''


def _rank_fma_reference(values, slots, weights):
    tokens, topk = weights.shape
    width = values.shape[-1]
    result = xp.empty((tokens, width), xp.float32)
    oracle = xp.RawKernel(_COMBINE_ORACLE, 'combine', options=('--std=c++17',))
    oracle(((tokens*width+127)//128,), (128,), (
        values, slots, weights, result, np.int32(tokens), np.int32(width), np.int32(topk)))
    return result


def _native_perexpert_reference(hidden, ids, routes, sources):
    # Independent orchestration: one expert at a time, no staged ring, compact
    # wave metadata, global grouped rows, or production route-map construction.
    from mini_llm.ops import triton_native_prefill as native
    tokens, topk = ids.shape
    rank_values = xp.empty((tokens, topk, hidden.shape[-1]), xp.float32)
    for expert_id in np.unique(ids):
        rows, ranks = np.nonzero(ids == expert_id)
        aq = native.quantize_native(xp.asarray(np.ascontiguousarray(hidden[0, rows])))
        count = xp.asarray(np.array([len(rows)], np.int32))
        offsets = xp.asarray(np.array([0, len(rows)], np.int32))
        def product(q, weight):
            return native.native_linear(q, xp.asarray(weight.data.reshape(-1)), count, offsets,
                N=weight.shape[0], K=weight.shape[1], kind=weight.quant_type,
                weight_stride=weight.nbytes, max_rows=len(rows))
        g, u, d = sources[expert_id]
        hq = native.native_swiglu_quantize(product(aq, g), product(aq, u))
        rank_values[xp.asarray(rows), xp.asarray(ranks)] = product(hq, d)
    values = rank_values.reshape(tokens*topk, -1)
    slots = xp.arange(tokens*topk, dtype=xp.int32).reshape(tokens, topk)
    return _rank_fma_reference(values, slots, xp.asarray(routes))


def test_native_combine_router_rank_fma_on_cupy_stream():
    from mini_llm.ops import triton_prefill_experts as module
    from mini_llm.ops import triton_native_prefill as native
    # In router order this is 1; in expert/slot order it is 2. Both could look
    # reasonable under relative tolerance, so require exact FP32 bit patterns.
    with xp.cuda.Stream(non_blocking=True) as stream:
        values = xp.asarray(np.repeat(np.array([[2**24], [-2**24], [1], [1]], np.float32), 259, axis=1))
        slots = xp.asarray(np.array([[0, 2, 1, 3]], np.int32))
        weights = xp.ones((1, 4), xp.float32)
        out = xp.empty((1, 259), xp.float32)
        native._launch(module._combine_native_kernel, (1, 2, 1),
            (module._ptr(values), module._ptr(out), module._ptr(slots), module._ptr(weights)),
            dict(D=259, K=4, DC=256), 4, xp)
        expected = _rank_fma_reference(values, slots, weights)
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(out).view(np.uint32), asnumpy(expected).view(np.uint32))
        np.testing.assert_array_equal(asnumpy(out), np.ones((1, 259), np.float32))
        # Non-exact products exercise fused multiplication, not just add order.
        rng = np.random.default_rng(103)
        values = xp.asarray(rng.normal(size=(32, 259)).astype(np.float32))
        slots = xp.asarray(rng.permutation(32).astype(np.int32).reshape(4, 8))
        weights = xp.asarray(rng.uniform(.1, 1.7, (4, 8)).astype(np.float32))
        out = xp.empty((4, 259), xp.float32)
        native._launch(module._combine_native_kernel, (4, 2, 1),
            (module._ptr(values), module._ptr(out), module._ptr(slots), module._ptr(weights)),
            dict(D=259, K=8, DC=256), 4, xp)
        expected = _rank_fma_reference(values, slots, weights)
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(out).view(np.uint32), asnumpy(expected).view(np.uint32))


@pytest.mark.parametrize('value', ['', 'native', 'FP16', 'native_int8 '])
def test_invalid_arithmetic_rejected_before_execution(monkeypatch, value):
    from mini_llm.ops import triton_prefill_experts as module
    monkeypatch.setenv('MINI_LLM_TRITON_ARITHMETIC', value)
    # Even eligibility and source resolution must not precede validation.
    with pytest.raises(ValueError, match='MINI_LLM_TRITON_ARITHMETIC'):
        module.try_triton_prefill_experts(None, None, None, None)


def test_default_arithmetic_is_fp16(monkeypatch):
    from mini_llm.ops import triton_prefill_experts as module
    monkeypatch.delenv('MINI_LLM_TRITON_ARITHMETIC', raising=False)
    # Invalid executor returns ineligible, not an invalid-arithmetic error.
    assert module.try_triton_prefill_experts(SimpleNamespace(), None, None, None) is None
    monkeypatch.setenv('MINI_LLM_TRITON_ARITHMETIC', 'fp16')
    assert module.try_triton_prefill_experts(SimpleNamespace(), None, None, None) is None
