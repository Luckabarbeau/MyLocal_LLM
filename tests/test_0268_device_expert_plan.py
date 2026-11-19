"""Persistent device route plan versus retained warp-GPU/GGML-CPU arithmetic."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor, ExpertKey
from mini_llm.ops.experts_inference import ExpertsInference
from test_0256_ggml_packed_cpu import selected_library, packed

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Device-side packed routing')


@pytest.fixture(autouse=True)
def same_warp_arithmetic(monkeypatch):
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'warp')


def managed(selected_library):
    sources = {}
    def provider(key):
        e = key.expert_id
        if e not in sources:
            sources[e] = (packed('IQ3_S', 320, 256, 500+e),
                          packed('IQ3_S', 320, 256, 1500+e),
                          packed('IQ4_NL', 256, 320, 2500+e))
        return sources[e]
    return HeterogeneousExpertExecutor(provider, model_id='0268', d_model=256, d_ff=320,
        compute_dtype='float32', device_budget_bytes=32*1024**2, cpu_workers=4,
        cpu_packed_backend='ggml_pool', promote_after=1, device_expert_plan=True)


def oracle(executor, x, ids, routes):
    from mini_llm.ops.iq_packed_gpu import grouped_terms
    pairs = sorted(zip(ids, routes))
    gpu = [(e, weight) for e, weight in pairs
           if ExpertKey('0268', 0, e) in executor.cache.resident]
    terms = grouped_terms(x, [executor.cache.resident[ExpertKey('0268',0,e)][0] for e, _ in gpu],
                          xp.asarray([r for _, r in gpu], dtype=xp.float32)) if gpu else None
    result = xp.zeros_like(x)
    index = 0
    for e, weight in pairs:
        if ExpertKey('0268', 0, e) in executor.cache.resident:
            term = terms[index:index+1]
            index += 1
        else:
            raw = executor._ggml_cpu.expert(x.get(), executor._host_weights(ExpertKey('0268',0,e)))
            term = xp.asarray(raw)*xp.float32(weight)
        result += term
    return result


@pytest.mark.parametrize('hits', [0, 3, 10])
def test_exact_mixed_order_and_retained_output(selected_library, hits):
    executor = managed(selected_library)
    x = xp.asarray(np.random.default_rng(268).normal(0, .1, (1, 256)).astype(np.float32))
    ids = [19, 0, 511, 4, 7, 11, 3, 1, 13, 9]
    routes = np.linspace(.01, .1, 10, dtype=np.float32)
    try:
        for e in ids[:hits]:
            executor._prepare(0, e, x)
        executor.cache.set_admission_budget(0)
        expected = oracle(executor, x, ids, routes)
        output = executor.try_forward_device_plan(0, x, xp.asarray(routes), xp.asarray(ids,dtype=xp.int64), 512)
        assert output is not None
        np.testing.assert_array_equal(output.get(), expected.get())
        saved = output.get().copy()
        executor.try_forward_device_plan(0, x*2, xp.asarray(routes), xp.asarray(ids,dtype=xp.int64), 512)
        np.testing.assert_array_equal(output.get(), saved)
        assert executor.counters['device_plan_dispatches'] == 2
        assert len(executor._device_expert_plans) == 1
    finally:
        executor.close()


def test_adapter_invalid_routes_and_resume_placement(selected_library):
    executor = managed(selected_library)
    layer = ExpertsInference(256, 320, 512, expert_executor=executor, layer_id=0)
    x = xp.ones((1,1,256), xp.float32)
    routes = xp.full((1,1,10), .1, xp.float32)
    ids = xp.arange(10, dtype=xp.int64).reshape(1,1,10)
    try:
        executor._prepare(0, 0, x.reshape(1,256))
        executor.cache.set_admission_budget(0)
        result = layer.forward(x, routes, ids)
        assert result.shape == x.shape and executor.counters['device_plan_dispatches'] == 1
        plan = executor._device_expert_plans[0]
        bad = ids.copy()
        bad[..., 1] = bad[..., 0]
        assert executor.try_forward_device_plan(0, x.reshape(1,256), routes.reshape(-1), bad.reshape(-1),512) is None
        bad[...,1] = 2**40
        assert executor.try_forward_device_plan(0, x.reshape(1,256), routes.reshape(-1), bad.reshape(-1),512) is None
        executor.cache.set_admission_budget(None)
        assert executor.try_forward_device_plan(0,x.reshape(1,256),routes.reshape(-1),ids.reshape(-1),512) is None
        assert not executor._device_expert_plans
        executor._prepare(0, 11, x.reshape(1,256))
        executor.cache.set_admission_budget(0)
        layer.forward(x, routes, ids)
        assert executor._device_expert_plans[0] is not plan
    finally:
        executor.close()


def test_block_vs_warp_roundoff_is_separate(selected_library, monkeypatch):
    executor = managed(selected_library)
    x = xp.asarray(np.random.default_rng(268).normal(0, .1, (1,256)).astype(np.float32))
    ids = [19, 0, 511, 4, 7, 11, 3, 1, 13, 9]
    routes = np.linspace(.01, .1, 10, dtype=np.float32)
    try:
        for e in ids:
            executor._prepare(0,e,x)
        executor.cache.set_admission_budget(0)
        warp = executor.try_forward_device_plan(0,x,xp.asarray(routes),xp.asarray(ids,dtype=xp.int64),512).get()
        monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'block')
        block = oracle(executor,x,ids,routes).get()
        delta = warp.astype(np.float64)-block
        max_abs = float(np.abs(delta).max())
        relative_l2 = float(np.linalg.norm(delta)/np.linalg.norm(block.astype(np.float64)))
        print('0268 FP32 block/warp: max_abs=', max_abs, 'relative_l2=', relative_l2)
        assert relative_l2 < 5e-6
        assert max_abs < 5e-6*float(np.max(np.abs(block)))
    finally:
        executor.close()


def test_gpu_failure_drains_native_batch(selected_library):
    executor = managed(selected_library)
    x = xp.ones((1,256),xp.float32)
    weights, ids = xp.full(10,.1,xp.float32), xp.arange(10,dtype=xp.int64)
    try:
        executor._prepare(0,0,x)
        executor.cache.set_admission_budget(0)
        executor.try_forward_device_plan(0,x,weights,ids,512)
        plan = executor._device_expert_plans[0]
        def fail(*args):
            raise RuntimeError('injected launch failure')
        plan.gate_up = fail
        with pytest.raises(RuntimeError,match='injected launch failure'):
            executor.try_forward_device_plan(0,x,weights,ids,512)
        assert executor._ggml_row_pool._pending is None
    finally:
        executor.close()
