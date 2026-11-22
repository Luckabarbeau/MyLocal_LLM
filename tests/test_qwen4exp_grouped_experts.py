"""Chunk expert execution versus the same singleton arithmetic policies."""
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.prefill_experts import try_prefill_experts
from test_qwen4exp_chunked_prefill import gpu_packed

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy required')]


def setup(monkeypatch, service):
    from scripts.serve_qwen4exp import GGML_CPU
    if not Path(GGML_CPU).is_file():
        pytest.skip('Pinned GGML CPU library unavailable')
    for name, value in {
        'MINI_LLM_GGML_CPU_LIBRARY': GGML_CPU,
        'MINI_LLM_EXPERT_GPU_Q8': '1',
        'MINI_LLM_IQ_GPU_CODEBOOK_STORAGE': 'global',
        'MINI_LLM_IQ_GPU_GROUPED_KERNEL': 'warp',
        'MINI_LLM_Q8_SHARED_GRID': '1',
        'MINI_LLM_NATIVE_MOE_HANDOFF': service,
        'MINI_LLM_NATIVE_MOE_GRAPH': service,
        'MINI_LLM_NATIVE_MOE_SERVICE': service,
        'MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH': service,
        'MINI_LLM_NATIVE_MOE_STAGE_LIMIT': '0',
    }.items():
        monkeypatch.setenv(name, value)
    sources = [(gpu_packed('IQ3_S', 256, 256, 3800+e),
                gpu_packed('IQ3_S', 256, 256, 3900+e),
                gpu_packed('Q2_0', 256, 256, 4000+e)) for e in range(4)]
    return sources


def executor(sources, resident):
    result = HeterogeneousExpertExecutor(lambda key: sources[key.expert_id],
        model_id='grouped-prefill', d_model=256, d_ff=256, compute_dtype='float32',
        device_budget_bytes=2*1024**2, cpu_workers=2, cpu_packed_backend='ggml_pool',
        promote_after=1, device_expert_plan=True)
    row = xp.ones((1, 256), xp.float32)
    for e in resident:
        result._prepare(0, e, row)
    result.cache.set_admission_budget(0)
    return result


def inputs(rows):
    x = xp.asarray(np.random.default_rng(5080).normal(0, .1, (1, rows, 256)), dtype=xp.float32)
    ids = xp.asarray(np.array([[2, 0], [1, 3], [0, 1], [3, 2], [1, 2]])[np.arange(rows)%5][None], dtype=xp.int64)
    weights = xp.asarray(np.array([[.3, .7], [.6, .4], [.2, .8], [.4, .6], [.9, .1]], np.float32)[np.arange(rows)%5][None])
    return x, weights, ids


def singleton(experts, x, weights, ids):
    result = xp.concatenate([experts.forward(x[:, t:t+1], weights[:, t:t+1], ids[:, t:t+1]).copy()
                             for t in range(x.shape[1])], axis=1)
    xp.cuda.get_current_stream().synchronize()
    return result


@pytest.mark.parametrize('rows', [1, 5, 64])
@pytest.mark.parametrize('service', ['0', '1'])
def test_grouped_mixed_is_exact_and_not_singleton_dispatch(monkeypatch, rows, service):
    sources = setup(monkeypatch, service)
    managed = executor(sources, (0, 1))
    experts = ExpertsInference(256, 256, 4, expert_executor=managed)
    x, weights, ids = inputs(rows)
    try:
        expected = singleton(experts, x, weights, ids)
        before = managed.counters['device_plan_dispatches']
        actual = try_prefill_experts(experts, x, weights, ids)
        assert actual is not None
        xp.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
        assert managed.counters['device_plan_dispatches'] == before
        assert managed.counters['prefill_grouped_layers'] == 1
        assert managed.counters['prefill_routed_rows'] == rows*2
        assert managed.counters['prefill_cpu_rows'] > 0
        assert managed.counters['prefill_gpu_rows'] > 0
    finally:
        managed.close()


@pytest.mark.parametrize('rows', [1, 5, 64])
def test_stream_matches_all_gpu_reference_without_residency_changes(monkeypatch, rows):
    sources = setup(monkeypatch, '0')
    reference = executor(sources, (0, 1, 2, 3))
    streamed = executor(sources, (0, 1))
    x, weights, ids = inputs(rows)
    try:
        expected = singleton(ExpertsInference(256, 256, 4, expert_executor=reference), x, weights, ids)
        before = dict(streamed.cache.resident)
        actual = try_prefill_experts(ExpertsInference(256, 256, 4, expert_executor=streamed),
                                     x, weights, ids, mode='stream')
        xp.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
        assert dict(streamed.cache.resident) == before
        assert streamed.counters['prefill_streamed_bytes'] > 0
        assert streamed.counters['prefill_cpu_rows'] == 0
    finally:
        streamed.close()
        reference.close()
