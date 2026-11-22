"""Chunk primitives retain the native singleton arithmetic and stream ownership."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.ops.hyperconnection_prefill_gpu import read_chunk
from mini_llm.ops.prefill_experts import _PackedStageRing, _gpu_rows
from mini_llm.ops.quantized_linear import packed_to_device
from test_qwen4exp_chunked_prefill import gpu_packed

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy required')]


@pytest.mark.parametrize('rows,cut', [(1,1), (17,3), (128,31), (1024,513)])
def test_chunk_delta_is_exact_across_boundaries_and_following_decode(rows, cut):
    from mini_llm.ops.gated_delta_inference import GatedDeltaMixerInference
    from mini_llm.ops.gated_delta_prefill import try_chunk
    rng = np.random.default_rng(5300)
    mixer = GatedDeltaMixerInference(1, 2, 128, 128, 4)
    mixer.set_weights(xp.asarray(rng.normal(0, .1, (mixer.channels,4)), dtype=xp.float32),
                      xp.ones(128, xp.float32))
    qkv = xp.asarray(rng.normal(0, .1, (1,rows+1,mixer.channels)), dtype=xp.float32)
    gate = xp.asarray(rng.normal(0, .1, (1,rows+1,2,128)), dtype=xp.float32)
    decay = xp.full((1,rows+1,2), -.1, xp.float32)
    beta = xp.full((1,rows+1,2), .3, xp.float32)
    a, b = mixer.create_state(1), mixer.create_state(1)
    expected = xp.concatenate([mixer.forward(qkv[:,t:t+1], gate[:,t:t+1],
        decay[:,t:t+1], beta[:,t:t+1], a) for t in range(rows)], axis=1)
    results = []
    for begin, end in ((0,cut), (cut,rows)):
        if begin == end:
            continue
        result = try_chunk(mixer, xp.ascontiguousarray(qkv[:,begin:end]),
            xp.ascontiguousarray(gate[:,begin:end]), xp.ascontiguousarray(decay[:,begin:end]),
            xp.ascontiguousarray(beta[:,begin:end]), b)
        assert result is not None
        results.append(result)
    np.testing.assert_array_equal(asnumpy(xp.concatenate(results,axis=1)), asnumpy(expected))
    np.testing.assert_array_equal(asnumpy(a.conv.history), asnumpy(b.conv.history))
    np.testing.assert_array_equal(asnumpy(a.delta.memory), asnumpy(b.delta.memory))
    assert a.length == b.length == rows
    left = mixer.forward(qkv[:,-1:],gate[:,-1:],decay[:,-1:],beta[:,-1:],a)
    right = mixer.forward(qkv[:,-1:],gate[:,-1:],decay[:,-1:],beta[:,-1:],b)
    np.testing.assert_array_equal(asnumpy(left), asnumpy(right))


@pytest.mark.parametrize('rows', [1, 17, 256])
@pytest.mark.parametrize('parallel', ['0', '1'])
def test_native_hc_batch_is_bitwise_singleton(monkeypatch, rows, parallel):
    monkeypatch.setenv('MINI_LLM_HC_PARALLEL_READ', parallel)
    rng = np.random.default_rng(5090)
    width, rank = 257, 33
    def bits(shape):
        a = rng.normal(0, .04, shape).astype(np.float32)
        return xp.asarray((a.view(np.uint32)>>16).astype(np.uint16))
    hc = Qwen4ExpHyperConnection(xp.ones((4,width), xp.float32), bits((rank,4*width)),
        bits((4*width,rank)), bits((4,4*width)), epsilon=1e-6,
        fast_decode=True, native_read=True, packed_bf16=True)
    x = xp.asarray(rng.normal(size=(rows,4,width)), dtype=xp.float32)
    expected = [hc.read(row) for row in x]
    actual = read_chunk(hc, x)
    np.testing.assert_array_equal(asnumpy(actual[0]), asnumpy(xp.stack([r[0] for r in expected])))
    np.testing.assert_array_equal(asnumpy(actual[1]), asnumpy(xp.stack([r[1] for r in expected])))


def test_packed_ring_wraparound_preserves_inflight_weight_bytes(monkeypatch):
    monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID', '1')
    from mini_llm.ops.q8_1_gpu import quantize_q8_1
    sources = [(gpu_packed('IQ3_S', 256,256, 5000+i),
                gpu_packed('IQ3_S', 256,256, 5100+i),
                gpu_packed('Q2_0', 256,256, 5200+i)) for i in range(4)]
    reference = [tuple(packed_to_device(w) for w in weights) for weights in sources]
    packed = quantize_q8_1(xp.asarray(np.random.default_rng(1).normal(size=(17,256)), dtype=xp.float32))
    rows = xp.arange(17, dtype=xp.int32)
    stream = xp.cuda.get_current_stream()
    ring = _PackedStageRing(max(sum(w.nbytes for w in weights) for weights in sources), slots=2)
    expected = [_gpu_rows(packed, rows, weights) for weights in reference]
    actual = []
    try:
        for i in range(19):
            slot, weights = ring.stage(sources[i%4], stream)
            actual.append(_gpu_rows(packed, rows, weights))
            ring.consumed(slot, stream)
        stream.synchronize()
        for i, result in enumerate(actual):
            np.testing.assert_array_equal(asnumpy(result), asnumpy(expected[i%4]))
    finally:
        stream.synchronize()
        ring.fence()


def test_broadcast_gemv_is_singleton_reduction():
    from mini_llm.ops.inference_matmul import matmul_inference
    rng = np.random.default_rng(5290)
    x = xp.asarray(rng.normal(size=(65,2560)), dtype=xp.float32)
    weight = xp.asarray(rng.normal(size=(2560,32)), dtype=xp.float32)
    expected = xp.concatenate([matmul_inference(row[None], weight) for row in x])
    actual = matmul_inference(x[:,None,:], weight)[:,0]
    np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
