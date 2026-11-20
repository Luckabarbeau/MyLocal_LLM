"""Whole-layer CUDA capture, explicit recurrence and real mapped CPU/GPU work.

No checkpoint/model runs. The native cases reuse the D256/F320 packed fixture.
"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp as _xp
from mini_llm.external.qwen4exp_recurrent_block_graph import Qwen4ExpRecurrentBlockGraph
from test_0256_ggml_packed_cpu import selected_library


xp: Any = _xp
gpu = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Actual CUDA recurrent block graph')


def test_import_and_subclass_contract():
    from mini_llm.external.qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph
    assert issubclass(Qwen4ExpRecurrentBlockGraph, Qwen4ExpPreMoEGraph)
    assert Qwen4ExpRecurrentBlockGraph.complete_block is True


def test_plan_validation_never_reads_back_native_service(monkeypatch):
    from mini_llm.external import qwen4exp_recurrent_block_graph as module
    def forbidden():
        pytest.fail('Plan validation must not perform a blocking service readback')
    fingerprint = (0, 0, 1024)
    executor = SimpleNamespace(cache=SimpleNamespace(_admission_limit=0,
        idle_eviction_requests=0), _device_expert_plans={})
    plan = SimpleNamespace(executor=executor, layer=0, fingerprint=fingerprint,
        cache_fingerprint=lambda e: fingerprint, stream_ptr=7, device=0)
    handoff = SimpleNamespace(plan=plan, capture_enabled=True,
        service=SimpleNamespace(check=forbidden), stage=None, handle=1, enqueue=lambda: None)
    plan.native_handoff = handoff
    executor._device_expert_plans[0] = plan
    graph: Any = object.__new__(Qwen4ExpRecurrentBlockGraph)
    graph.plan, graph.handoff, graph.fingerprint = plan, handoff, fingerprint
    graph.block = SimpleNamespace(moe=SimpleNamespace(
        experts=SimpleNamespace(expert_executor=executor, layer_id=0),
        enqueue_gpu_branch=lambda: None, combine_gpu_branch=lambda: None))
    stream = SimpleNamespace(ptr=7, is_capturing=lambda: False)
    monkeypatch.setattr(module, 'xp', SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: stream, runtime=SimpleNamespace(getDevice=lambda: 0))))
    graph._validate_plan()
    graph._validate_plan()


@pytest.mark.parametrize('outcome', ['success', 'failure', 'invalid_capture'])
def test_forward_records_external_retirement_unless_capture_is_active(monkeypatch, outcome):
    from mini_llm.external import qwen4exp_recurrent_block_graph as module
    from mini_llm.external.qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph
    calls = []
    capturing = False
    stream = SimpleNamespace(is_capturing=lambda: capturing)
    def record(actual_stream):
        assert actual_stream is stream and not capturing
        calls.append('external retirement')
    graph: Any = object.__new__(Qwen4ExpRecurrentBlockGraph)
    graph.closed, graph.bind_state = False, False
    graph.graph = SimpleNamespace(calls=1)
    graph.plan = SimpleNamespace(retired=SimpleNamespace(record=record), pending=False)
    graph._validate_plan = lambda: None
    state = SimpleNamespace(conv=SimpleNamespace(history=None), delta=SimpleNamespace(memory=None))
    result = object()
    def forward(self, streams, actual_state):
        nonlocal capturing
        assert self is graph and actual_state is state
        calls.append('base forward')
        if outcome != 'success':
            capturing = outcome == 'invalid_capture'
            raise RuntimeError('injected forward failure')
        return result
    monkeypatch.setattr(Qwen4ExpPreMoEGraph, 'forward', forward)
    monkeypatch.setattr(module, 'xp', SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: stream)))
    if outcome == 'success':
        assert graph.forward(None, state) is result
    else:
        with pytest.raises(RuntimeError, match='injected forward failure'):
            graph.forward(None, state)
    expected = ['base forward']
    if outcome != 'invalid_capture':
        expected.append('external retirement')
    assert calls == expected
    assert graph.plan.pending == (outcome != 'invalid_capture')


@pytest.fixture
def configuration(monkeypatch):
    for name in ('MINI_LLM_NATIVE_MOE_SERVICE', 'MINI_LLM_NATIVE_MOE_HANDOFF',
                 'MINI_LLM_NATIVE_MOE_GRAPH', 'MINI_LLM_NATIVE_MOE_FUSED_SHARED',
                 'MINI_LLM_NATIVE_MOE_ROUTE_PROFILE', 'MINI_LLM_NATIVE_MOE_TIMELINE',
                 'MINI_LLM_GGML_POOL_HOST_HELP'):
        monkeypatch.setenv(name, '1')
    for name in ('MINI_LLM_NATIVE_MOE_STAGE_LIMIT', 'MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH',
                 'MINI_LLM_BOUND_RECURRENT_STATE', 'MINI_LLM_EXPERT_GPU_Q8'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('MINI_LLM_GGML_POOL_SCHEDULER', 'tasks')
    monkeypatch.setenv('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'warp')


def components(executor) -> tuple[Any, Any, Any, list[int]]:
    """0265 projection arithmetic, adapted to the actual native plan width."""
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec, Qwen4ExpRecurrentMixerInference
    from mini_llm.external.qwen4exp_model import Qwen4ExpLayer, Qwen4ExpMoE
    from mini_llm.external.qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph
    from mini_llm.ops.experts_inference import ExpertsInference
    from mini_llm.ops.inference_matmul import matmul_inference
    from mini_llm.ops.router_inference import RouterInference

    rng = np.random.default_rng(313)
    def rand(shape):
        return xp.asarray(rng.normal(0, .03, shape).astype(np.float32))
    spec = Qwen4ExpMixerSpec(256, 2, 4, 16, 4, 1e-5)
    weights: dict[str, Any] = dict(qkv=rand((spec.qkv_channels, 256)), z=rand((spec.inner_size, 256)),
        beta=rand((4, 256)), alpha=rand((4, 256)), out=rand((256, spec.inner_size)),
        conv_weight=rand((spec.qkv_channels, 4)), norm_weight=xp.ones(16, xp.float32),
        ssm_a=-xp.exp(rand((4,))), dt_bias=rand((4,)), fast_coefficients=True)
    for name in ('qkv', 'z', 'beta', 'alpha', 'out'):
        weight = weights[name]
        weights[name] = lambda x, weight=weight: matmul_inference(x, weight.T)
    mixer = Qwen4ExpRecurrentMixerInference(spec, **weights)
    mixer.mixer._decode_fastpath_enabled = True
    def hc():
        return Qwen4ExpHyperConnection(xp.ones((4, 256), xp.float32), rand((16, 1024)),
            rand((1024, 16)), rand((4, 1024)), epsilon=1e-5,
            fast_decode=True, native_read=True)
    attention, ffn = hc(), hc()
    router = RouterInference(256, 512, 10, renormalize_selected=True,
                             normalization_floor=2**-14)
    router.W_router = rand((256, 512))*xp.float32(.01)
    bias = np.full(512, -10, np.float32)
    # Stable known hit counts, but genuinely input-dependent logits and weights.
    ids = [19, 0, 511, 4, 7, 11, 3, 1, 13, 9]
    bias[ids] = np.linspace(2, 1, 10, dtype=np.float32)
    router.b_router = xp.asarray(bias)
    router._decode_fastpath_enabled = True
    setattr(router, '_capturable_logits', True)
    experts = ExpertsInference(256, 320, 512, expert_executor=executor, layer_id=0)
    moe = Qwen4ExpMoE(router, experts, lambda x: x*xp.float32(.3),
        lambda x: x*xp.float32(-.7), lambda x: x*xp.float32(.2), rand((256,)),
        scale=.7, fast_elementwise=True, shared_graph_decode=True)
    segment = Qwen4ExpPreMoEGraph(attention, mixer, ffn, router)
    return segment, Qwen4ExpLayer(attention, mixer, ffn, moe, True), rand, ids


def state_arrays(state):
    return state.conv.history, state.delta.memory


def snapshot(state):
    return tuple(a.get().copy() for a in state_arrays(state))


def assert_state(state, values, length):
    assert state.conv.length == state.delta.length == state.length == length
    for got, expected in zip(state_arrays(state), values):
        np.testing.assert_array_equal(got.get().view(np.uint32), expected.view(np.uint32))


@pytest.fixture
def fake_plan(configuration):
    """GPU-only geometry/lifetime fixture, not a native arithmetic oracle."""
    from mini_llm.ops.device_expert_plan import DeviceExpertPlan
    cache = SimpleNamespace(_admission_limit=0, idle_eviction_requests=0,
        device_budget_bytes=32*1024**2, counters=dict(promotions=0, evictions=0))
    executor = SimpleNamespace(cache=cache, _device_expert_plans={})
    segment, block, rand, _ = components(executor)
    plan = SimpleNamespace(executor=executor, layer=0, q8=None,
        stream_ptr=xp.cuda.get_current_stream().ptr, device=xp.cuda.runtime.getDevice(),
        fingerprint=DeviceExpertPlan.cache_fingerprint(executor),
        cache_fingerprint=DeviceExpertPlan.cache_fingerprint,
        retired=xp.cuda.Event(disable_timing=True), pending=False)
    totals = np.zeros(6, np.uint64)
    handoff = SimpleNamespace(plan=plan, capture_enabled=True, service=SimpleNamespace(check=lambda: None),
        stage=None, handle=1, graph=None, graph_branch=None, stats=np.zeros(6, np.uint64),
        accounted=np.zeros(6, np.uint64), route_heat=None, timeline=None)
    def enqueue(x, routes, ids, callback, *, gpu_branch):
        assert callback is None and gpu_branch is block.moe
        if not xp.cuda.get_current_stream().is_capturing():
            totals[0] += 1
        return gpu_branch.combine_gpu_branch(x*xp.float32(.25), gpu_branch.enqueue_gpu_branch(x))
    def stats(handle, address):
        handoff.stats[:] = totals
    def flush():
        stats(1, 0)
        if totals[3] != handoff.accounted[3]:
            raise RuntimeError('fake native error')
        handoff.accounted[:] = totals
    handoff.enqueue = enqueue
    handoff.bridge = SimpleNamespace(mylocal_ggml_handoff_stats=stats)
    handoff.flush_statistics = flush
    plan.native_handoff = handoff
    executor._device_expert_plans[0] = plan
    yield segment, block, plan, rand
    for owner, attr in ((segment, 'graph'), (handoff, 'graph'), (block.moe, '_shared_graph')):
        graph = getattr(owner, attr)
        if graph is not None:
            graph.close()
            setattr(owner, attr, None)


@gpu
def test_upgrade_retires_all_old_pools_and_preserves_bound_policy(fake_plan, monkeypatch):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    segment, block, plan, rand = fake_plan
    segment.bind_state = True
    segment.forward(rand((4, 256)), block.mixer.create_state(1))
    hidden = rand((256,))
    block.moe._run_shared(hidden)
    plan.native_handoff.graph = CapturedArrayInference(lambda x, arrays: x+1, hidden, ())
    old = (segment.graph, plan.native_handoff.graph, block.moe._shared_graph)
    monkeypatch.setenv('MINI_LLM_BOUND_RECURRENT_STATE', '0')
    graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    try:
        assert graph.bind_state
        assert graph.attention_hc is segment.attention_hc and graph.router is segment.router
        assert segment.graph is plan.native_handoff.graph is block.moe._shared_graph is None
        for retired in old:
            assert retired.closed and retired.statistics()['pool_reserved_bytes'] == 0
        state = block.mixer.create_state(1)
        graph.forward(rand((4, 256)), state)
        assert state.length == 1
        graph.close()  # Fence an outstanding replay, not just the native join.
        assert graph.closed and graph.graph is None and plan.native_handoff.handle == 1
        graph.close()
        with pytest.raises(ValueError, match='closed'):
            graph.forward(rand((4, 256)), state)
    finally:
        graph.close()


@gpu
@pytest.mark.parametrize('bad', ['nonrecurrent', 'ple', 'operators', 'unfrozen', 'idle',
    'fingerprint', 'identity', 'handoff', 'capture', 'service', 'stage', 'handle'])
def test_constructor_rejects_ineligible_plan_before_retiring_segment(fake_plan, bad):
    segment, block, plan, rand = fake_plan
    segment.forward(rand((4, 256)), block.mixer.create_state(1))
    old = segment.graph
    if bad == 'nonrecurrent': block.recurrent = False
    elif bad == 'ple': block.ple = object()
    elif bad == 'operators': block.attention_hc = object()
    elif bad == 'unfrozen': plan.executor.cache._admission_limit = None
    elif bad == 'idle': plan.executor.cache.idle_eviction_requests = 1
    elif bad == 'fingerprint': plan.executor.cache.counters['evictions'] += 1
    elif bad == 'identity': plan.executor._device_expert_plans[0] = object()
    elif bad == 'handoff': plan.native_handoff = None
    elif bad == 'capture': plan.native_handoff.capture_enabled = False
    elif bad == 'service': plan.native_handoff.service = None
    elif bad == 'stage': plan.native_handoff.stage = object()
    elif bad == 'handle': plan.native_handoff.handle = None
    with pytest.raises(ValueError):
        Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    assert segment.graph is old and not old.closed


@gpu
@pytest.mark.parametrize('bad', ['shape', 'dtype', 'history', 'memory', 'layout', 'length', 'negative'])
def test_base_geometry_contract_does_not_advance_state(fake_plan, bad):
    segment, block, plan, rand = fake_plan
    graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    state, streams = block.mixer.create_state(1), rand((4, 256))
    if bad == 'shape': streams = streams[:3]
    elif bad == 'dtype': streams = streams.astype(xp.float64)
    elif bad == 'history': state.conv.history = state.conv.history[:, :1]
    elif bad == 'memory': state.delta.memory = state.delta.memory[:, :1]
    elif bad == 'layout':
        state.conv.history = xp.zeros((1, 3, 256), xp.float32)[..., ::2]
    elif bad == 'length': state.conv.length = 1
    elif bad == 'negative': state.conv.length = state.delta.length = -1
    before = snapshot(state)
    lengths = state.conv.length, state.delta.length
    try:
        with pytest.raises(ValueError, match='geometry|positions|contiguous'):
            graph.forward(streams, state)
        for array, saved in zip(state_arrays(state), before):
            np.testing.assert_array_equal(array.get(), saved)
        assert (state.conv.length, state.delta.length) == lengths
        assert graph.calls == graph.captures == 0 and graph.graph is None
    finally:
        graph.close()


@gpu
@pytest.mark.parametrize('bad', ['fingerprint', 'plan_fingerprint', 'identity', 'handoff', 'unfrozen', 'stream'])
def test_replay_revalidates_frozen_live_plan_before_state_changes(fake_plan, bad):
    segment, block, plan, rand = fake_plan
    graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    state, streams = block.mixer.create_state(1), rand((4, 256))
    try:
        graph.forward(streams, state)
        before, calls = snapshot(state), graph.calls
        if bad == 'fingerprint': plan.executor.cache.device_budget_bytes += 1
        elif bad == 'plan_fingerprint': plan.fingerprint = (-1, -1, -1)
        elif bad == 'identity': plan.executor._device_expert_plans[0] = object()
        elif bad == 'handoff': plan.native_handoff = object()
        elif bad == 'unfrozen': plan.executor.cache._admission_limit = None
        with pytest.raises(ValueError, match='fingerprint|identity|stream'):
            if bad == 'stream':
                with xp.cuda.Stream(non_blocking=True): graph.forward(streams, state)
            else:
                graph.forward(streams, state)
        assert_state(state, before, 1)
        assert graph.calls == calls
    finally:
        graph.close()


@gpu
@pytest.mark.parametrize('bound', [False, True])
def test_private_warmup_and_capture_failure_leave_caller_recurrence_untouched(fake_plan, bound, monkeypatch):
    segment, block, plan, rand = fake_plan
    segment.bind_state = bound
    graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    state, streams = block.mixer.create_state(1), rand((4, 256))
    state.conv.history.fill(.1)
    state.delta.memory.fill(.2)
    before = snapshot(state)
    original = block.mixer.projections['out']
    def fail_during_capture(x):
        if xp.cuda.get_current_stream().is_capturing():
            raise RuntimeError('injected capture failure after private warmup')
        return original(x)
    monkeypatch.setitem(block.mixer.projections, 'out', fail_during_capture)
    try:
        with pytest.raises(RuntimeError, match='injected capture failure'):
            graph.forward(streams, state)
        assert_state(state, before, 0)
        assert graph.graph is None and graph.calls == graph.captures == 0
        assert plan.native_handoff.accounted[0] == 1  # Completed private batch only.
    finally:
        graph.close()


def native_case(selected_library, hits):
    from test_0268_device_expert_plan import managed
    executor = managed(selected_library)
    try:
        segment, block, rand, ids = components(executor)
        streams = rand((4, 256))
        _, hidden, _, weights, indices = segment.forward(streams, block.mixer.create_state(1))
        np.testing.assert_array_equal(indices.get().reshape(-1), ids)
        for expert in ids[:hits]:
            executor._prepare(0, expert, hidden.reshape(1, -1))
        executor.cache.set_admission_budget(0)
        block.moe._run_shared(hidden)  # Exercise retirement of a real shared pool too.
        block.moe(hidden, routing=(weights, indices))  # Build existing frozen plan first.
        plan = executor._device_expert_plans[0]
        executor.check_native_service()
        plan.native_handoff.flush_statistics()
        return executor, segment, block, plan, rand
    except BaseException:
        executor.close()
        raise


def uncaptured_block(segment, block, handoff, streams, state) -> Any:
    """Same primitives/Q8 paths, no captured segment or omitted experts."""
    hidden, gains = segment.attention_hc.read(streams)
    mixed = segment.mixer.forward(hidden[None, None, :], state)[0, 0]
    residual = segment.attention_hc.write(streams, mixed, gains)
    hidden, gains = segment.ffn_hc.read(residual)
    weights, ids = segment.router.forward(hidden.reshape(1, 1, -1))
    routed = handoff.enqueue(hidden.reshape(1, -1),
        (weights*block.moe.scale).reshape(-1), ids.reshape(-1), None, gpu_branch=block.moe)
    return segment.ffn_hc.write(residual, routed.reshape(-1), gains)


@gpu
@pytest.mark.parametrize('hits', [0, 3, 10], ids=['cpu_only', 'mixed', 'all_gpu'])
@pytest.mark.parametrize('q8', [False, True], ids=['warp_fp32', 'q8'])
@pytest.mark.parametrize('bound', [False, True], ids=['copied_state', 'bound_recapture'])
def test_real_native_service_exact_layer_sequence_and_warmup_statistics(
        configuration, selected_library, monkeypatch, hits, q8, bound):
    monkeypatch.setenv('MINI_LLM_EXPERT_GPU_Q8', str(int(q8)))
    monkeypatch.setenv('MINI_LLM_BOUND_RECURRENT_STATE', str(int(bound)))
    executor, segment, block, plan, rand = native_case(selected_library, hits)
    handoff = plan.native_handoff
    old = (segment.graph, handoff.graph, block.moe._shared_graph)
    graph = None
    try:
        raw_before = int(handoff.stats[0])
        warm_before = len(handoff.timeline_warmup_ordinals)
        production_before = handoff.production.copy()
        graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
        assert all(g is not None and g.closed for g in old)
        assert segment.graph is handoff.graph is block.moe._shared_graph is None
        def forbidden(*args, **kwargs):
            pytest.fail('Whole-layer enqueue must not enter Python dispatch/host callbacks')
        monkeypatch.setattr(handoff, '_enqueue_callback', forbidden)
        monkeypatch.setattr(handoff, 'forward', forbidden)
        monkeypatch.setattr(plan, 'forward', forbidden)
        monkeypatch.setattr(block.moe, '_run_shared', forbidden)
        reference = [block.mixer.create_state(1) for _ in range(2)]
        captured = [block.mixer.create_state(1) for _ in range(2)]
        for step in range(6):
            session = step % 2
            streams = rand((4, 256))
            expected = uncaptured_block(segment, block, handoff, streams, reference[session]).get()
            actual = graph.forward(streams, captured[session])
            executor.check_native_service()  # Model-level safety boundary, once per token.
            np.testing.assert_array_equal(actual.get().view(np.uint32), expected.view(np.uint32))
            assert_state(captured[session], snapshot(reference[session]), step//2+1)
        handoff.flush_statistics()
        captures = 6 if bound else 1
        assert graph.calls == 6 and graph.captures == captures
        assert int(handoff.stats[0])-raw_before == 12+captures
        assert len(handoff.timeline_warmup_ordinals)-warm_before == captures
        assert int(handoff.production[0]-production_before[0]) == 12
        assert int(handoff.production[1]-production_before[1]) == 12*(10-hits)
        assert int(handoff.production[2]-production_before[2]) == 12*hits
        assert handoff.production[3] == 0
        assert sum(handoff.statistics()['route_heat']) == int(handoff.production[0])*10
        stats = graph.graph_statistics()
        assert stats['scope'] == 'recurrent_decoder_block' and stats['complete_block']
        assert stats['bound_state'] == bound and stats['native_handoff']
        assert stats['gpu_arithmetic'] == ('q8_1' if q8 else 'fp32')
        assert stats['pool_reserved_bytes'] <= graph.pool_limit_bytes
        assert executor.statistics()['execution']['native_handoff_dispatches'] == 13
        if bound:
            assert graph.graph is not None
            assert all(a is b for a, b in zip(graph.graph.bound_arrays(), state_arrays(captured[1])))
        graph.close()
        assert graph.graph is None and handoff.handle is not None
    finally:
        if graph is not None: graph.close()
        executor.close()
    assert handoff.handle is None and not executor.cache._protected


@gpu
def test_service_readback_is_warmup_only_not_per_replay(fake_plan, monkeypatch):
    segment, block, plan, rand = fake_plan
    segment.bind_state = True
    checks = []
    monkeypatch.setattr(plan.native_handoff.service, 'check', lambda: checks.append('warmup'))
    graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    try:
        assert checks == []
        state, streams = block.mixer.create_state(1), rand((4, 256))
        graph.forward(streams, state)
        assert checks == ['warmup']
        graph.forward(streams, state)
        assert checks == ['warmup']
        graph.forward(streams, block.mixer.create_state(1))
        assert checks == ['warmup', 'warmup']
        assert graph.captures == 2 and graph.calls == 3
    finally:
        graph.close()


@gpu
def test_original_plan_stream_required_even_before_first_replay(fake_plan):
    segment, block, plan, rand = fake_plan
    state, streams = block.mixer.create_state(1), rand((4, 256))
    graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
    try:
        with xp.cuda.Stream(non_blocking=True):
            with pytest.raises(ValueError, match='original plan replay stream'):
                Qwen4ExpRecurrentBlockGraph(segment, block, plan)
            with pytest.raises(ValueError, match='original plan replay stream'):
                graph.forward(streams, state)
        assert graph.graph is None and graph.calls == 0 and state.length == 0
    finally:
        graph.close()


@gpu
def test_real_gpu_branch_failure_drains_private_native_batch_and_excludes_warmup(
        configuration, selected_library, monkeypatch):
    executor, segment, block, plan, rand = native_case(selected_library, 3)
    handoff, graph = plan.native_handoff, None
    try:
        graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
        state, streams = block.mixer.create_state(1), rand((4, 256))
        before, production = snapshot(state), handoff.production.copy()
        ordinal = int(handoff.accounted[0])
        def fail(hidden):
            raise RuntimeError('injected GPU branch launch failure after native publication')
        monkeypatch.setattr(block.moe, 'enqueue_gpu_branch', fail)
        with pytest.raises(RuntimeError, match='injected GPU branch launch failure'):
            graph.forward(streams, state)
        assert_state(state, before, 0)
        assert graph.graph is None
        assert int(handoff.accounted[0]) == ordinal+1
        flags = handoff.service_flags.host
        assert flags[0] == flags[16]  # The native batch was drained before returning.
        executor.check_native_service()
        handoff.flush_statistics()
        np.testing.assert_array_equal(handoff.production, production)
    finally:
        if graph is not None: graph.close()
        executor.close()


@gpu
@pytest.mark.parametrize('bad', ['duplicate', 'negative', 'range', 'wide'])
def test_real_native_invalid_warmup_route_errors_are_not_discarded(
        configuration, selected_library, monkeypatch, bad):
    executor, segment, block, plan, rand = native_case(selected_library, 0)
    handoff, graph = plan.native_handoff, None
    try:
        graph = Qwen4ExpRecurrentBlockGraph(segment, block, plan)
        state, streams = block.mixer.create_state(1), rand((4, 256))
        state.conv.history.fill(.1)
        state.delta.memory.fill(.2)
        before, errors = snapshot(state), int(handoff.accounted[3])
        original = segment.router.forward
        def invalid(x):
            weights, ids = original(x)
            ids = ids.copy()
            ids[..., 1] = ids[..., 0] if bad == 'duplicate' else {
                'negative': -1, 'range': 512, 'wide': 2**40}.get(bad, -1)
            return weights, ids
        monkeypatch.setattr(segment.router, 'forward', invalid)
        with pytest.raises(RuntimeError, match='warmup failed|service failed'):
            graph.forward(streams, state)
        assert_state(state, before, 0)
        assert graph.graph is None and graph.calls == 0
        assert int(handoff.stats[3]) > errors and int(handoff.accounted[3]) == errors
        # Explicitly observe, rather than erase, the unaccounted native failure.
        with pytest.raises(RuntimeError, match='invalid/failed dispatches'):
            handoff.flush_statistics()
        assert int(handoff.production[3]) > 0
    finally:
        if graph is not None: graph.close()
        executor.close()
