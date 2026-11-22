"""Whole-query-chunk QSA must use each query's causal prefix, not future KV."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.qsa_decode_gpu import qsa_decode_gpu
from mini_llm.ops.qsa_prefill_gpu import attention_chunk, try_qsa_chunk
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
from test_qwen4exp_chunked_prefill import gpu_packed

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy required')]


@pytest.mark.parametrize('start,rows', [(0,1), (0,17), (7,17), (127,32)])
def test_batched_attention_is_bitwise_causal_singleton(monkeypatch, start, rows):
    monkeypatch.setenv('MINI_LLM_QSA_WARP_DOTS','1')
    rng = np.random.default_rng(5400)
    def random(shape):
        return xp.asarray(rng.normal(size=shape),dtype=xp.float32)
    q = random((rows,4,256))
    qg = random((rows,4,512))
    k,v = random((rows,2,256)),random((rows,2,256))
    keys,values = random((start+rows,2,256)),random((start+rows,2,256))
    keys[start:],values[start:] = k,v
    expected = xp.stack([qsa_decode_gpu(q[i],qg[i],k[i],v[i],keys,values,start+i+1,.0625).reshape(4,256)
                         for i in range(rows)])
    actual = attention_chunk(q,qg,k,v,keys,values,start,.0625)
    np.testing.assert_array_equal(asnumpy(actual),asnumpy(expected))


@pytest.mark.parametrize('topk',[14,2048])
def test_packed_qsa_chunk_pooling_state_and_following_decode(monkeypatch,topk):
    monkeypatch.setenv('MINI_LLM_QSA_WARP_DOTS','1')
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8','1')
    c = QSAConfig(256,n_q_heads=4,n_kv_heads=2,head_dim=256,index_topk=topk)
    shapes = dict(query_gate=(2048,256),key=(512,256),value=(512,256),
                  output=(256,1024),index_query=(512,256),index_key=(128,256))
    callbacks = {}
    for i,(name,(n,k)) in enumerate(shapes.items()):
        device = packed_to_device(gpu_packed('IQ3_S',n,k,5500+i))
        def project(x,device=device):
            return quantized_linear(xp.ascontiguousarray(x.reshape(-1,x.shape[-1])),device).reshape(x.shape[:-1]+(device.shape[0],))
        project.packed_device_weight = device
        callbacks[name] = project
    weights = {**callbacks, 'gamma_q':xp.ones(256,xp.float32),'gamma_k':xp.ones(256,xp.float32),
               'gamma_index_q':xp.ones(128,xp.float32),'gamma_index_k':xp.ones(128,xp.float32)}
    attention = Qwen4ExpAttention(c,**weights,fast_decode=True,qsa_state_storage='host')
    model = SimpleNamespace(weights=None)
    x = xp.asarray(np.random.default_rng(5401).normal(0,.1,(34,256)),dtype=xp.float32)
    a,b = attention.create_state(64),attention.create_state(64)
    expected = xp.stack([attention.step(row,a,position=i) for i,row in enumerate(x[:33])])
    pieces = []
    for begin,end in ((0,7),(7,19),(19,33)):
        from mini_llm.external.qwen4exp_prefill import _qsa
        result = _qsa(model,0,attention,x[begin:end],b)
        assert result is not None
        pieces.append(result)
    np.testing.assert_array_equal(asnumpy(xp.concatenate(pieces)),asnumpy(expected))
    assert a.length == b.length == 33
    for name in ('keys','values','raw_index','pooled'):
        np.testing.assert_array_equal(asnumpy(getattr(a,name)),asnumpy(getattr(b,name)))
    np.testing.assert_array_equal(asnumpy(attention.step(x[33],b,position=33)),
                                  asnumpy(attention.step(x[33],a,position=33)))
