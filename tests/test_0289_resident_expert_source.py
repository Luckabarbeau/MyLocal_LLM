"""Bounded packed anonymous RAM source, paced registration, immutable fallback."""
import gc
from pathlib import Path
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import resident_expert_source as rs
from mini_llm.ops.expert_residency import ExpertKey
from test_0238_packed_expert_slots import weights
from test_0256_ggml_packed_cpu import selected_library


def bank():
    result={ExpertKey('289',0,i):weights(seed=289+i) for i in range(8)}
    for triple in result.values():
        for w in triple: w.data.flags.writeable=False
    return result


def test_compact_complete_bytes_readonly_owner_and_export(monkeypatch):
    sources=bank()
    monkeypatch.setattr(rs,'host_available_bytes',lambda:1024**3)
    from mini_llm.ops.quantized_weight import PackedGGUFWeight
    monkeypatch.setattr(PackedGGUFWeight,'dequantize_reference',lambda *a:pytest.fail('No weight expansion'))
    source=rs.ResidentPackedExpertSource(sources.__getitem__,sources,budget_bytes=2*1024**2,
        reserve_bytes=0,chunk_bytes=64*1024)
    assert source.statistics()['complete_complement']
    assert source.statistics()['pinned_expert_source_bytes']==0
    assert source.nbytes<=2*1024**2
    for key,original in sources.items():
        for a,b in zip(original,source.weights(key)):
            assert a.shape==b.shape and a.quant_type==b.quant_type and not b.data.flags.writeable
            assert not np.shares_memory(a.data,b.data)
            np.testing.assert_array_equal(a.data,b.data)
    exported=source.weights(next(iter(sources)))[0].data
    snapshot=exported.copy()
    source.close(); gc.collect()
    np.testing.assert_array_equal(exported,snapshot)
    with pytest.raises(RuntimeError,match='closed'): source.weights(next(iter(sources)))


def test_partial_budget_and_pressure_failure(monkeypatch):
    sources=bank(); available=[1024**3]
    monkeypatch.setattr(rs,'host_available_bytes',lambda:available[0])
    source=rs.ResidentPackedExpertSource(sources.__getitem__,sources,budget_bytes=256*1024,reserve_bytes=0)
    assert 0<len(source.bundles)<len(sources)
    assert not source.statistics()['complete_complement']
    source.close()
    with pytest.raises(MemoryError,match='one complete'): 
        rs.ResidentPackedExpertSource(sources.__getitem__,sources,budget_bytes=1,reserve_bytes=0)
    available[0]=0
    with pytest.raises(MemoryError,match='headroom'):
        rs.ResidentPackedExpertSource(sources.__getitem__,sources,budget_bytes=2*1024**2)
    available[0]=1024**3
    def pressure(*args): available[0]=0
    with pytest.raises(MemoryError,match='reserve'):
        rs.ResidentPackedExpertSource(sources.__getitem__,sources,budget_bytes=2*1024**2,
            reserve_bytes=1,chunk_bytes=4096,progress=pressure)


def test_cgroup_available_bounds(tmp_path,monkeypatch):
    monkeypatch.setattr(rs,'memory_info',lambda:dict(MemAvailable=1000))
    membership=tmp_path/'membership'; membership.write_text('0::/parent/child\n')
    root=tmp_path/'cg'; (root/'parent'/'child').mkdir(parents=True)
    (root/'parent'/'memory.max').write_text('400')
    (root/'parent'/'memory.current').write_text('100')
    (root/'parent'/'child'/'memory.max').write_text('max')
    assert rs.host_available_bytes(cgroup_root=root,membership=membership)==300


@pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA registration')
def test_paced_pinned_prefix_and_dma(monkeypatch):
    import cupy as cp
    sources=bank()
    monkeypatch.setattr(rs,'host_available_bytes',lambda:1024**3)
    source=rs.ResidentPackedExpertSource(sources.__getitem__,sources,budget_bytes=2*1024**2,
        reserve_bytes=0,pin=True,pin_limit_bytes=256*1024,chunk_bytes=64*1024)
    try:
        assert source.pinned_bytes==256*1024
        for key,(start,end) in source.offsets.items():
            pointer=source.dma_pointer(key)
            if end<=source.pinned_bytes:
                assert pointer
                out=cp.empty(end-start,cp.uint8)
                cp.cuda.runtime.memcpyAsync(out.data.ptr,pointer,out.nbytes,
                    cp.cuda.runtime.memcpyHostToDevice,cp.cuda.get_current_stream().ptr)
                np.testing.assert_array_equal(out.get(),source.raw[start:end])
            else: assert not pointer
    finally:
        cp.cuda.get_current_stream().synchronize(); source.close()


@pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Captured heterogeneous source registry')
def test_native_plan_uses_anonymous_source_and_clears_old_registry(selected_library,monkeypatch):
    import cupy as cp
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_HANDOFF','1')
    monkeypatch.setenv('MINI_LLM_NATIVE_MOE_GRAPH','1')
    sources=bank()
    managed=HeterogeneousExpertExecutor(sources.__getitem__,model_id='289',d_model=256,d_ff=256,
        compute_dtype='float32',device_budget_bytes=0,cpu_workers=4,cpu_packed_backend='ggml_pool',
        device_expert_plan=True)
    x=np.random.default_rng(289).normal(0,.1,(1,256)).astype(np.float32)
    routes=np.linspace(.01,.08,8,dtype=np.float32)
    expected=np.zeros_like(x)
    for i,(key,ws) in enumerate(sources.items()):
        expected+=routes[i]*managed._ggml_cpu.expert(x,ws)
    try:
        managed._ggml_row_pool.register(next(iter(sources)),sources[next(iter(sources))])
        setup=managed.prepare_resident_host_source(sources,budget_bytes=2*1024**2,pin=True,reserve_bytes=0)
        source=managed._resident_source
        assert not managed._ggml_row_pool._registered and setup['complete_complement']
        managed.cache.set_admission_budget(0)
        output=managed.try_forward_device_plan(0,cp.asarray(x),cp.asarray(routes),cp.arange(8,dtype=cp.int64),8)
        np.testing.assert_array_equal(output.get(),expected)
        assert managed.statistics()['native_handoffs'][0]['resident_host_registered_experts']==8
        # Neither the immutable fallback pointer nor its bytes were replaced.
        for key in sources:
            assert managed._host_weights(key)[0] is sources[key][0]
        with pytest.raises(ValueError,match='before device plans'):
            managed.prepare_resident_host_source(sources,budget_bytes=2*1024**2)
    finally:
        managed.close()
    assert source.closed


@pytest.mark.skipif(BACKEND_NAME!='numpy',reason='Small CPU model same-arithmetic comparison')
def test_model_host_budget_and_request_parity(selected_library):
    from test_0209_qwen4exp_model import TinyStore,factory
    from mini_llm.external.qwen4exp_weights import Qwen4ExpWeights
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    store=TinyStore()
    weights_owner=Qwen4ExpWeights(Qwen4ExpGGUFIndex(store,model_id='tiny'))
    base=weights_owner.memory_plan(16)
    extra=weights_owner.memory_plan(16,host_expert_bytes=1024**2)
    assert extra['host_reserved_bytes']==base['host_reserved_bytes']+1024**2
    assert not store.reads
    with pytest.raises(MemoryError,match='host budget'):
        weights_owner.memory_plan(16,host_expert_bytes=9*1024**3)
    with factory(store,expert_cpu_backend='ggml_pool',host_expert_bytes=1024**2) as model:
        before=model.forward(np.array([1,2,3,4]))
        stats=model.prepare_host_experts(reserve_bytes=0)
        assert stats['complete_complement']
        after=model.forward(np.array([1,2,3,4]))
        np.testing.assert_array_equal(after,before)
