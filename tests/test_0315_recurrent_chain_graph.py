"""CPU contract tests and opt-in real-GPU/native two/three-block parity.

No model/checkpoint runs. GPU cases reuse 0313 operators and the multilayer-capable
0268 packed provider; each layer has its own actual native plan and handoff.
"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp as _xp
from mini_llm.external import qwen4exp_recurrent_chain_graph as module
from mini_llm.external.qwen4exp_recurrent_block_graph import Qwen4ExpRecurrentBlockGraph
from mini_llm.external.qwen4exp_recurrent_chain_graph import Qwen4ExpRecurrentChainGraph
from mini_llm.ops.causal_conv_inference import CausalConvState
from mini_llm.ops.gated_delta_inference import DeltaMixerState, DeltaRuleState
from test_0256_ggml_packed_cpu import selected_library
from test_0313_recurrent_block_graph import configuration

xp: Any = _xp
gpu = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Actual CUDA/native recurrent chain')


class CPUArray(np.ndarray):
    """NumPy storage with just the device metadata inspected by the adapter."""
    device_id = 0

    def __array_finalize__(self, original):
        self.device_id = getattr(original, 'device_id', 0)

    @property
    def device(self):
        return SimpleNamespace(id=self.device_id)


def cpu_array(shape):
    return np.zeros(shape, np.float32).view(CPUArray)


def cpu_state():
    return DeltaMixerState(CausalConvState(cpu_array((1, 2, 6))),
                           DeltaRuleState(cpu_array((1, 2, 2, 2))))


@pytest.fixture
def rig(monkeypatch):
    from mini_llm.external import qwen4exp_recurrent_block_graph as block_module
    log = []
    stream = SimpleNamespace(ptr=7, capturing=False)
    stream.is_capturing = lambda: stream.capturing
    runtime = SimpleNamespace(log=log, stream=stream, fail=None, graphs=[])
    fake_xp = SimpleNamespace(ndarray=CPUArray, float32=np.float32,
        cuda=SimpleNamespace(get_current_stream=lambda: stream,
                             runtime=SimpleNamespace(getDevice=lambda: 0)))
    monkeypatch.setattr(module, 'xp', fake_xp)
    monkeypatch.setattr(block_module, 'xp', fake_xp)

    class Capture:
        def __init__(self, enqueue, x, arrays, *, pool_limit_bytes, bound_mutable):
            self.calls, self.closed = 0, False
            self.enqueue, self.limit = enqueue, pool_limit_bytes
            self.bound_mutable = bound_mutable
            self.bound = tuple(arrays[i] for i in bound_mutable)
            self.calls_at_close = None
            self.pointers = tuple(a.ctypes.data for a in self.bound)
            self.mutable = tuple(arrays)
            self.output = x.copy()
            runtime.graphs.append(self)
            # Same ordering as CapturedArrayInference: private warmup, then
            # nonexecuting capture. Simulated capture uses disposable arrays.
            enqueue(x.copy(), tuple(a.copy() for a in arrays))
            stream.capturing = True
            try:
                enqueue(x.copy(), tuple(a.copy() for a in arrays))
                if runtime.fail == 'constructor':
                    raise RuntimeError('injected constructor failure')
            finally:
                stream.capturing = False

        def matches(self, x, arrays):
            return (not self.closed and
                    tuple(arrays[i].ctypes.data for i in self.bound_mutable) == self.pointers)

        def forward(self, x, arrays, *, borrow_output):
            assert borrow_output
            if runtime.fail in ('replay', 'active_capture'):
                stream.capturing = runtime.fail == 'active_capture'
                raise RuntimeError('injected replay failure')
            # Replaying GPU nodes does not execute Python native accounting.
            stream.capturing = True
            try:
                self.output[:] = self.enqueue(x, arrays)
            finally:
                stream.capturing = False
            log.append('replay')
            self.calls += 1
            return self.output

        def close(self):
            self.calls_at_close = self.calls
            log.append('graph fence')
            self.closed = True

        def statistics(self):
            return dict(pool_reserved_bytes=128, pool_used_bytes=64,
                        pool_limit_bytes=self.limit)

        def bound_arrays(self):
            return self.bound

    monkeypatch.setattr(module, 'CapturedArrayInference', Capture)
    executor = SimpleNamespace(cache=SimpleNamespace(_admission_limit=0,
        idle_eviction_requests=0), _device_expert_plans={})
    members = []
    for layer in range(3):
        member: Any = object.__new__(Qwen4ExpRecurrentBlockGraph)
        p = SimpleNamespace(executor=executor, layer=layer, device=0, stream_ptr=7,
            fingerprint=(0, 0, 1), cache_fingerprint=lambda e: (0, 0, 1), q8=None,
            retired=SimpleNamespace(record=lambda s, layer=layer:
                log.append(('retired', layer, s.capturing))), pending=False)
        executor._device_expert_plans[layer] = p
        h = SimpleNamespace(plan=p, capture_enabled=True, stage=None, handle=1,
            enqueue=lambda: None, route_heat=True, profile_warmup=np.zeros(1),
            service=SimpleNamespace(check=lambda: pytest.fail('per-token service readback')),
            flush_statistics=lambda layer=layer: log.append(('flush', layer)),
            read_profile=lambda: np.array([17]))
        p.native_handoff = h
        member.plan, member.handoff, member.fingerprint = p, h, p.fingerprint
        member.block = SimpleNamespace(recurrent=True, ple=None, moe=SimpleNamespace(
            experts=SimpleNamespace(expert_executor=executor, layer_id=layer),
            enqueue_gpu_branch=lambda: None, combine_gpu_branch=lambda: None))
        member.mixer = SimpleNamespace(spec=SimpleNamespace(d_model=4,
            conv_kernel_size=3, qkv_channels=6, n_value_heads=2, head_dim=2))
        member.graph, member.closed, member.bind_state = None, False, False
        member._warm_profile_before = None
        def enqueue(x, arrays, layer=layer, member=member):
            assert len(arrays) == 2
            if not stream.capturing:
                assert member._warm_profile_before[0] == 17
                log.append(('private warm', layer))
            arrays[0][...] += np.float32(layer + 1)
            arrays[1][...] += np.float32(layer + 2)
            return x + arrays[0].sum(dtype=np.float32) + arrays[1].sum(dtype=np.float32)
        member._enqueue = enqueue
        members.append(member)
    runtime.members = tuple(members)
    runtime.states = tuple(cpu_state() for _ in members)
    runtime.x = cpu_array((4, 4))
    return runtime


def test_import_and_constructor_retires_pools_without_closing_members(rig):
    assert Qwen4ExpRecurrentChainGraph.complete_block
    old = SimpleNamespace(close=lambda: rig.log.append('old pool fence'))
    for member in rig.members:
        member.graph = old  # Shared owners must not double-retire a pool.
    chain = Qwen4ExpRecurrentChainGraph(rig.members)
    assert chain.members == rig.members and isinstance(chain.members, tuple)
    assert chain.plan is rig.members[0].plan
    assert rig.log == ['old pool fence']
    assert all(m.graph is None and not m.closed for m in rig.members)
    assert chain.pool_limit_bytes == 3 * 16 * 1024**2
    stats = chain.graph_statistics()
    assert stats['scope'] == 'recurrent_decoder_chain'
    assert stats['chain_length'] == 3 and stats['complete_block']
    assert stats['pool_limit_bytes'] == chain.pool_limit_bytes


@pytest.mark.parametrize('bad', ['empty', 'single', 'four', 'type', 'closed', 'incomplete',
    'qsa', 'ple', 'executor', 'gap', 'device', 'stream', 'policy', 'unfrozen',
    'fingerprint', 'identity', 'handoff', 'stage', 'service', 'handle', 'capturing'])
def test_eligibility_rejects_entire_chain_before_retiring_any_pool(rig, bad):
    members = list(rig.members[:2])
    tail = members[-1]
    old = SimpleNamespace(close=lambda: pytest.fail('retired an ineligible chain'))
    for m in members:
        m.graph = old
    if bad == 'empty': members = []
    elif bad == 'single': members = members[:1]
    elif bad == 'four': members = members * 2
    elif bad == 'type': members[-1] = object()
    elif bad == 'closed': tail.closed = True
    elif bad == 'incomplete': tail.complete_block = False
    elif bad == 'qsa': tail.block.recurrent = False
    elif bad == 'ple': tail.block.ple = object()
    elif bad == 'executor': tail.plan.executor = object()
    elif bad == 'gap': tail.plan.layer = 3
    elif bad == 'device': tail.plan.device = 1
    elif bad == 'stream': tail.plan.stream_ptr = 8
    elif bad == 'policy': tail.bind_state = True
    elif bad == 'unfrozen': tail.plan.executor.cache._admission_limit = None
    elif bad == 'fingerprint': tail.plan.fingerprint = (-1,)
    elif bad == 'identity': tail.plan.executor._device_expert_plans[1] = object()
    elif bad == 'handoff': tail.plan.native_handoff = object()
    elif bad == 'stage': tail.handoff.stage = object()
    elif bad == 'service': tail.handoff.service = None
    elif bad == 'handle': tail.handoff.handle = None
    elif bad == 'capturing': rig.stream.capturing = True
    with pytest.raises(ValueError):
        Qwen4ExpRecurrentChainGraph(members)
    assert not rig.log


@pytest.mark.parametrize('bad', ['count', 'type', 'shape', 'dtype', 'input_layout',
    'input_device', 'history', 'memory', 'layout', 'state_dtype', 'state_device',
    'length', 'negative', 'different_positions'])
def test_all_geometry_is_validated_before_any_enqueue(rig, bad):
    chain = Qwen4ExpRecurrentChainGraph(rig.members)
    states, x = list(rig.states), rig.x
    tail = states[-1]
    if bad == 'count': states.pop()
    elif bad == 'type': states[-1] = object()
    elif bad == 'shape': x = x[:3]
    elif bad == 'dtype': x = x.astype(np.float64)
    elif bad == 'input_layout': x = cpu_array((4, 8))[:, ::2]
    elif bad == 'input_device': x.device_id = 1
    elif bad == 'history': tail.conv.history = tail.conv.history[:, :1]
    elif bad == 'memory': tail.delta.memory = tail.delta.memory[:, :1]
    elif bad == 'layout': tail.conv.history = cpu_array((1, 2, 12))[..., ::2]
    elif bad == 'state_dtype': tail.delta.memory = tail.delta.memory.astype(np.float64)
    elif bad == 'state_device': tail.delta.memory.device_id = 1
    elif bad == 'length': tail.conv.length = 1
    elif bad == 'negative': tail.conv.length = tail.delta.length = -1
    elif bad == 'different_positions': tail.conv.length = tail.delta.length = 1
    before = [(s.conv.history.copy(), s.delta.memory.copy(), s.conv.length, s.delta.length)
              for s in rig.states]
    with pytest.raises(ValueError, match='geometry|positions|contiguous|device|count'):
        chain.forward(x, states)
    assert not rig.log and not rig.graphs
    assert chain.calls == chain.captures == 0 and chain.graph is None
    for s, (conv, delta, cl, dl) in zip(rig.states, before):
        np.testing.assert_array_equal(s.conv.history, conv)
        np.testing.assert_array_equal(s.delta.memory, delta)
        assert (s.conv.length, s.delta.length) == (cl, dl)


@pytest.mark.parametrize('bound', [False, True])
@pytest.mark.parametrize('count', [2, 3])
def test_replay_bound_recapture_warmup_and_close_contract(rig, bound, count):
    members, states = rig.members[:count], rig.states[:count]
    for m in members:
        m.bind_state = bound
    chain = Qwen4ExpRecurrentChainGraph(members)
    first = chain.forward(rig.x, states)
    saved = first.copy()
    assert all(s.length == 1 for s in states)
    assert all(m.plan.pending for m in members)
    assert [v for v in rig.log if isinstance(v, tuple) and v[0] == 'retired'] == [
        ('retired', j, False) for j in range(count)]
    first_graph = chain.graph
    expected_bound = tuple(a for s in states for a in (s.conv.history, s.delta.memory))
    assert all(a is b for a, b in zip(first_graph.bound_arrays(), expected_bound))
    assert len(first_graph.bound_arrays()) == (2 * count if bound else 0)
    assert chain.forward(rig.x, states) is first
    assert np.all(first != saved)  # Borrowed output, not fresh caller ownership.
    assert all(s.length == 2 for s in states)
    assert rig.log.count(('flush', 0)) == 1
    assert rig.log.count(('private warm', 0)) == 1
    other = tuple(cpu_state() for _ in members)
    chain.forward(rig.x, other)
    assert chain.calls == 3 and chain.captures == (2 if bound else 1)
    assert all(s.length == 2 for s in states) and all(s.length == 1 for s in other)
    if bound:
        assert first_graph.closed and first_graph.calls_at_close == 2
        assert rig.log.count(('flush', 0)) == 2
        assert rig.log.count(('private warm', 0)) == 2
    stats = chain.graph_statistics()
    assert stats['pool_reserved_bytes'] <= 16 * 1024**2 * count
    rig.log.clear()
    chain.close()
    assert rig.log == ['graph fence', *(('flush', j) for j in range(count))]
    assert chain.closed and chain.graph is None
    assert all(not m.closed and m.handoff.handle == 1 for m in members)
    chain.close()
    with pytest.raises(ValueError, match='closed'):
        chain.forward(rig.x, states)


@pytest.mark.parametrize('bad', ['fingerprint', 'identity', 'policy', 'ple', 'stream'])
def test_each_live_plan_is_revalidated_without_flushing_or_readback(rig, bad):
    chain = Qwen4ExpRecurrentChainGraph(rig.members)
    chain.forward(rig.x, rig.states)
    rig.log.clear()
    tail = rig.members[-1]
    if bad == 'fingerprint': tail.plan.fingerprint = (-1,)
    elif bad == 'identity': tail.plan.executor._device_expert_plans[2] = object()
    elif bad == 'policy': tail.bind_state = True
    elif bad == 'ple': tail.block.ple = object()
    elif bad == 'stream': rig.stream.ptr = 8
    with pytest.raises(ValueError):
        chain.forward(rig.x, rig.states)
    assert rig.log == [] and chain.calls == 1
    assert all(s.length == 1 for s in rig.states)


@pytest.mark.parametrize('failure', ['constructor', 'replay', 'active_capture'])
def test_failure_preserves_caller_state_and_records_all_external_retirements(rig, failure):
    chain = Qwen4ExpRecurrentChainGraph(rig.members)
    rig.fail = failure
    with pytest.raises(RuntimeError, match='injected'):
        chain.forward(rig.x, rig.states)
    assert chain.graph is None and chain.calls == 0
    for s in rig.states:
        assert s.length == 0
        assert not s.conv.history.any() and not s.delta.memory.any()
    assert rig.log.count(('private warm', 0)) == 1
    events = [v for v in rig.log if isinstance(v, tuple) and v[0] == 'retired']
    assert events == ([] if failure == 'active_capture' else
                      [('retired', j, False) for j in range(3)])
    assert all(m.plan.pending == (failure != 'active_capture') for m in rig.members)
    if failure != 'constructor':
        assert rig.graphs[0].closed
    rig.stream.capturing = False
    chain.close()


def test_close_rejects_capture_without_releasing_live_graph(rig):
    chain = Qwen4ExpRecurrentChainGraph(rig.members)
    chain.forward(rig.x, rig.states)
    rig.stream.capturing = True
    with pytest.raises(ValueError, match='outside capture'):
        chain.close()
    assert not chain.closed and not chain.graph.closed
    rig.stream.capturing = False
    chain.close()


def build_native_chain(selected_library, count, hits, bound):
    """Independent real plans per layer, frozen together before graph upgrades."""
    from test_0268_device_expert_plan import managed
    from test_0313_recurrent_block_graph import components
    executor = managed(selected_library)  # Provider accepts every layer key.
    members, prepared = [], []
    try:
        for layer in range(count):
            segment, block, rand, ids = components(executor)
            block.moe.experts.layer_id = layer
            streams = rand((4, 256))
            _, hidden, _, weights, indices = segment.forward(streams, block.mixer.create_state(1))
            np.testing.assert_array_equal(indices.get().reshape(-1), ids)
            for expert in ids[:hits]:
                executor._prepare(layer, expert, hidden.reshape(1, -1))
            prepared.append((segment, block, hidden, weights, indices))
        executor.cache.set_admission_budget(0)
        for layer, (segment, block, hidden, weights, indices) in enumerate(prepared):
            block.moe._run_shared(hidden)
            block.moe(hidden, routing=(weights, indices))
            plan = executor._device_expert_plans[layer]
            segment.bind_state = bound
            members.append(Qwen4ExpRecurrentBlockGraph(segment, block, plan))
        executor.check_native_service()
        return executor, tuple(members), rand
    except BaseException:
        for member in members:
            member.close()
        for segment, block, *_ in prepared:
            if segment.graph is not None:
                segment.graph.close()
            if block.moe._shared_graph is not None:
                block.moe._shared_graph.close()
        executor.close()
        raise


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize('count', [2, 3])
@pytest.mark.parametrize('bound', [False, True], ids=['copied', 'bound'])
@pytest.mark.parametrize('hits', [0, 3, 10], ids=['cpu', 'mixed', 'gpu'])
@pytest.mark.parametrize('q8', [False, True], ids=['fp32', 'q8'])
def test_native_chain_exact_sequential_members_and_warm_accounting(
        configuration, selected_library, monkeypatch, count, bound, hits, q8):
    from test_0313_recurrent_block_graph import snapshot, assert_state
    monkeypatch.setenv('MINI_LLM_EXPERT_GPU_Q8', str(int(q8)))
    executor, members, rand = build_native_chain(selected_library, count, hits, bound)
    chain = None
    try:
        reference = [[m.mixer.create_state(1) for m in members] for _ in range(2)]
        captured = [[m.mixer.create_state(1) for m in members] for _ in range(2)]
        inputs = [rand((4, 256)) for _ in range(4)]
        outputs, values = [], []
        # Exact reference is sequential member.forward, not an alternate equation
        # or expert reduction. Finish it before transferring pools to the chain.
        for step, streams in enumerate(inputs):
            session = step % 2
            expected = streams
            for member, state in zip(members, reference[session]):
                expected = member.forward(expected, state)
            executor.check_native_service()  # One model boundary per token.
            outputs.append(expected.get().copy())
            values.append([snapshot(s) for s in reference[session]])
        old_pools = [m.graph for m in members]
        for m in members:
            m.handoff.flush_statistics()
        production = [m.handoff.production.copy() for m in members]
        raw = [int(m.handoff.stats[0]) for m in members]
        warm = [len(m.handoff.timeline_warmup_ordinals) for m in members]
        chain = Qwen4ExpRecurrentChainGraph(members)
        assert all(g.closed for g in old_pools)
        assert all(not m.closed and m.graph is None for m in members)
        def forbidden(*args, **kwargs):
            pytest.fail('Chain capture/replay must not use host dispatch')
        for m in members:
            monkeypatch.setattr(m, 'forward', forbidden)
            monkeypatch.setattr(m.plan, 'forward', forbidden)
            monkeypatch.setattr(m.handoff, 'forward', forbidden)
            monkeypatch.setattr(m.handoff, '_enqueue_callback', forbidden)
            monkeypatch.setattr(m.block.moe, '_run_shared', forbidden)
        for step, streams in enumerate(inputs):
            session = step % 2
            actual = chain.forward(streams, captured[session])
            executor.check_native_service()
            np.testing.assert_array_equal(actual.get().view(np.uint32),
                                          outputs[step].view(np.uint32))
            for state, saved in zip(captured[session], values[step]):
                assert_state(state, saved, step // 2 + 1)
            assert all(m.plan.pending for m in members)
        captures = len(inputs) if bound else 1
        assert chain.calls == len(inputs) and chain.captures == captures
        for j, m in enumerate(members):
            h = m.handoff
            h.flush_statistics()
            assert int(h.stats[0]) - raw[j] == len(inputs) + captures
            assert len(h.timeline_warmup_ordinals) - warm[j] == captures
            assert int(h.production[0] - production[j][0]) == len(inputs)
            assert int(h.production[1] - production[j][1]) == len(inputs) * (10 - hits)
            assert int(h.production[2] - production[j][2]) == len(inputs) * hits
            assert h.production[3] == 0
            assert sum(h.statistics()['route_heat']) == int(h.production[0]) * 10
        if bound:
            bound_arrays = tuple(a for s in captured[1] for a in (s.conv.history, s.delta.memory))
            assert all(a is b for a, b in zip(chain.graph.bound_arrays(), bound_arrays))
            assert len(chain.graph.bound_arrays()) == 2 * count
        stats = chain.graph_statistics()
        assert stats['complete_block'] and stats['chain_length'] == count
        assert stats['pool_reserved_bytes'] <= stats['pool_limit_bytes'] == count * 16 * 1024**2
        chain.close()
        assert all(m.handoff.handle is not None and not m.closed for m in members)
    finally:
        if chain is not None:
            chain.close()
        for m in members:
            m.close()
        executor.close()


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize('bound', [False, True])
def test_native_later_block_capture_failure_keeps_all_caller_states_private(
        configuration, selected_library, monkeypatch, bound):
    from test_0313_recurrent_block_graph import snapshot, assert_state
    executor, members, rand = build_native_chain(selected_library, 3, 3, bound)
    chain = None
    try:
        chain = Qwen4ExpRecurrentChainGraph(members)
        states = [m.mixer.create_state(1) for m in members]
        for s in states:
            s.conv.history.fill(.1)
            s.delta.memory.fill(.2)
        before = [snapshot(s) for s in states]
        for m in members:
            m.handoff.flush_statistics()
        production = [m.handoff.production.copy() for m in members]
        original = members[-1].mixer.projections['out']
        def fail(x):
            if xp.cuda.get_current_stream().is_capturing():
                raise RuntimeError('injected last-block capture failure')
            return original(x)
        monkeypatch.setitem(members[-1].mixer.projections, 'out', fail)
        with pytest.raises(RuntimeError, match='last-block capture failure'):
            chain.forward(rand((4, 256)), states)
        assert chain.graph is None and chain.calls == chain.captures == 0
        executor.check_native_service()
        for m, s, saved, totals in zip(members, states, before, production):
            assert_state(s, saved, 0)
            assert m.plan.pending
            m.plan.retired.synchronize()  # Must be a valid external event, not a capture node.
            m.handoff.flush_statistics()
            np.testing.assert_array_equal(m.handoff.production, totals)
    finally:
        if chain is not None:
            chain.close()
        for m in members:
            m.close()
        executor.close()
