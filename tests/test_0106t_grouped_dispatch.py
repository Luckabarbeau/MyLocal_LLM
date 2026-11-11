"""Grouped resident dispatch integrates with tracing, ownership and hybrid fallback."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference


def test_unscoped_or_cpu_group_probe_is_observational():
    executor = HeterogeneousExpertExecutor(lambda _: pytest.fail('provider called'),
        model_id='group-test', d_model=256, d_ff=256, compute_dtype='float32')
    before = executor.statistics()
    assert executor.try_forward_grouped(0, (0,), xp.ones((1, 256), xp.float32),
                                       xp.ones(1, xp.float32), (0,)) is None
    assert executor.statistics() == before
    executor.close()


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires real GPU packed handles')
@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('mode', ['summary', 'detailed'])
def test_grouped_dispatch_exact_weight_order_trace_and_stream_ownership(monkeypatch, workers, mode):
    from test_0106t_grouped_quantized_experts import _hosts
    import mini_llm.ops.grouped_quantized_experts as grouped

    hosts = _hosts(('Q4_K', 'Q5_K', 'Q6_K'))
    bank = {ExpertKey('group-test', 0, i): (h.gate, h.up, h.down) for i, h in enumerate(hosts)}
    size = sum(w.nbytes for w in bank[ExpertKey('group-test', 0, 0)])
    executor = HeterogeneousExpertExecutor(bank.__getitem__, model_id='group-test',
        d_model=256, d_ff=256, compute_dtype='float32', device_budget_bytes=3 * size,
        promote_after=1, cpu_workers=workers, cpu_packed_backend='native')
    layer = ExpertsInference(256, 256, 3, expert_executor=executor)
    x = xp.asarray(np.random.default_rng(10620).normal(0, .05, (1, 1, 256)).astype(np.float32))
    weights = xp.asarray([[[.3, .2, .5]]], dtype=xp.float32)
    ids = xp.asarray([[[2, 0, 1]]], dtype=xp.int32)
    try:
        for i in range(3):
            executor.forward(0, i, x.reshape(1, 256))
        # Existing non-grouped singleton implementation is the matched GPU oracle.
        with monkeypatch.context() as ordinary:
            ordinary.setattr(executor, 'try_forward_grouped', lambda *args: None)
            reference = layer.forward(x, weights, ids)
        trace = RoutingTrace(model_id='group-test', quantization='mixed',
            expert_mapping='identity', workload='unit', mode=mode)
        executor.routing_profiler = trace
        before = executor.statistics()
        original = grouped.grouped_quantized_experts_forward
        launched = []
        def checked(x_in, handles, routes, *, routing_slots=None):
            assert executor._dispatch_depth == 1
            assert all(executor.cache._protected[k] == 1 for k in bank)
            launched.append(asnumpy(routes)[list(routing_slots)].tolist())
            return original(x_in, handles, routes, routing_slots=routing_slots)
        monkeypatch.setattr(grouped, 'grouped_quantized_experts_forward', checked)
        monkeypatch.setattr(executor, 'submit', lambda *args: pytest.fail('per-expert submit used'))
        ready = xp.cuda.Event(disable_timing=True)
        ready.record(xp.cuda.get_current_stream())
        with xp.cuda.Stream(non_blocking=True) as stream:
            stream.wait_event(ready)
            # Produce operands on this stream; no legacy-default-stream assumption.
            actual = layer.forward(x.copy(), weights.copy(), ids.copy())
            stream.synchronize()
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(reference))
        np.testing.assert_allclose(launched, [[.2, .5, .3]], rtol=1e-7)
        stats = executor.statistics()
        assert stats['execution']['grouped_gpu_dispatches'] == 1
        assert stats['execution']['gpu_experts'] - before['execution'].get('gpu_experts', 0) == 3
        assert stats['device']['requests'] - before['device']['requests'] == 3
        assert stats['device']['hits'] - before['device'].get('hits', 0) == 3
        assert stats['device']['protected_experts'] == 0
        snap = trace.snapshot()
        assert snap['tokens_by_layer'] == {'0': 1}
        assert snap['counters']['gpu_hit_experts'] == 3
        assert snap['counters']['routed_assignments'] == 3
        assert snap['counters']['promoted_experts'] == 0
        assert snap['counters']['transferred_bytes'] == 0
        assert 'grouped_gpu_compute_host_ms' in snap['timings_ms']
        assert snap['pending_gpu_timers'] == 0
    finally:
        executor.close()
        executor.cache.clear()


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires real GPU packed handles')
@pytest.mark.parametrize('fallback', ['miss', 'unfused', 'mixed', 'idle'])
def test_group_probe_does_not_mutate_cache_or_counters_on_fallback(fallback):
    from test_0106t_grouped_quantized_experts import _hosts
    from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device

    hosts = _hosts(('Q8_0', 'Q8_0', 'Q8_0'))
    executor = HeterogeneousExpertExecutor(lambda _: pytest.fail('provider called'),
        model_id='group-test', d_model=256, d_ff=256, compute_dtype='float32',
        device_budget_bytes=1024**2)
    for i, h in enumerate(hosts):
        handle = PackedExpertFFNInference(*(packed_to_device(w) for w in (h.gate, h.up, h.down)))
        executor.cache.resident[ExpertKey('group-test', 0, i)] = (handle, 100)
    if fallback == 'miss':
        del executor.cache.resident[ExpertKey('group-test', 0, 2)]
    elif fallback == 'unfused':
        executor.cache.resident[ExpertKey('group-test', 0, 1)][0].fuse_gate_up = False
    elif fallback == 'mixed':
        h = _hosts(('Q4_K', 'Q5_K', 'Q6_K'), count=1)[0]
        executor.cache.resident[ExpertKey('group-test', 0, 1)] = (
            PackedExpertFFNInference(*(packed_to_device(w) for w in (h.gate, h.up, h.down))), 100)
    else:
        executor.cache.idle_eviction_requests = 1
    before = executor.statistics()
    try:
        with executor.dispatch_scope(0, (0, 1, 2)):
            assert executor.try_forward_grouped(0, (0, 1, 2), xp.ones((1, 256), xp.float32),
                xp.ones(3, xp.float32), (2, 0, 1)) is None
            assert executor.cache.counters == {}
        after = executor.statistics()
        assert after['execution'] == before['execution']
        assert after['device'].get('requests', 0) == 0
    finally:
        executor.close()
        executor.cache.clear()
