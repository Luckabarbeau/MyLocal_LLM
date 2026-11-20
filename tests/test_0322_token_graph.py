"""Whole-token CPU lifecycle contracts and opt-in tiny real-CuPy math tests.

No checkpoints, native model runs, or dispatch integration. GPU tests reuse the
0313/0317 GPU-only fake services; they do not validate native expert arithmetic.
"""
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp as _xp
from mini_llm.external import qwen4exp_token_graph as module
from mini_llm.external.qwen4exp_ple import PLEState
from mini_llm.external.qwen4exp_qsa_block_graph import Qwen4ExpQSABlockGraph
from mini_llm.external.qwen4exp_token_graph import Qwen4ExpTokenGraph
from mini_llm.ops.causal_conv_inference import CausalConvState
from mini_llm.ops.gated_delta_inference import DeltaMixerState, DeltaRuleState


xp: Any = _xp


class DeviceArray(np.ndarray):
    """Host storage with device metadata, never a GPU arithmetic substitute."""
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    @property
    def device(self) -> Any:
        return NS(id=0)

    @property
    def data(self) -> Any:
        return NS(ptr=self.__array_interface__['data'][0])


class HostBackend:
    ndarray = DeviceArray
    cuda: Any

    def __getattr__(self, name):
        return getattr(np, name)

    def stack(self, values):
        return DeviceArray(np.stack(values))

    def empty(self, shape, dtype):
        return DeviceArray(np.empty(shape, dtype))

    def broadcast_to(self, value, shape):
        return DeviceArray(np.broadcast_to(value, shape))


def array(shape, dtype: Any = np.float32):
    return DeviceArray(np.zeros(shape, dtype))


@pytest.fixture
def host(monkeypatch):
    from mini_llm.external import qwen4exp_recurrent_block_graph as lifecycle
    from mini_llm.external.qwen4exp_attention import QSAConfig
    backend = HostBackend()
    events, checks, position_calls = [], [], []
    stream = NS(ptr=7, capturing=False, is_capturing=lambda: stream.capturing,
                synchronize=lambda: events.append('sync'))
    backend.cuda = NS(get_current_stream=lambda: stream, runtime=NS(getDevice=lambda: 0),
                      memory=NS(OutOfMemoryError=type('PoolOOM', (MemoryError,), {})))
    monkeypatch.setattr(module, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(module, 'xp', backend)
    monkeypatch.setattr(lifecycle, 'xp', backend)

    def validate(name, a, dtype, shape=None):
        if (not isinstance(a, DeviceArray) or a.dtype != dtype or
                (shape is not None and a.shape != shape) or not a.flags.c_contiguous):
            raise ValueError(f'{name} shape/device/contiguous layout mismatch')
        return a
    monkeypatch.setattr(module, '_array', validate)

    def position(a, t, bound):
        assert 0 <= t <= bound
        position_calls.append((t, bound))
        a[...] = t
    monkeypatch.setattr(module, 'set_position_gpu', position)

    def hc(final=False):
        return NS(native_read=True, fast_decode=True, graph_decode=False, width=4,
            inject=None if final else array((4, 16)),
            initialize=lambda x: DeviceArray(np.repeat(x[None], 4, axis=0)),
            collapse=lambda x: x.mean(axis=0),
            read=lambda x: (x.mean(axis=0), DeviceArray(np.ones(4, np.float32))),
            write=lambda x, y, gains: x + gains[:, None] * y[None])

    class Pre:
        def __init__(self, attention_hc, mixer, ffn_hc, router):
            self.graph = None
        def _enqueue(self, streams, arrays):
            arrays[0][...] += np.float32(1)
            arrays[1][...] += np.float32(2)
            residual = streams + arrays[0].sum() * np.float32(.01)
            hidden = residual.mean(axis=0)
            return residual, hidden, DeviceArray(np.ones(4, np.float32)), array((1,)), array((1,), np.int32)
    monkeypatch.setattr(module, 'Qwen4ExpPreMoEGraph', Pre)
    executor = NS(cache=NS(_admission_limit=0, idle_eviction_requests=0), _device_expert_plans={})
    blocks = []
    for i, recurrent in enumerate((False, True, False)):
        b = NS(recurrent=recurrent, ple=None, attention_hc=hc(), ffn_hc=hc())
        b.moe = NS(router=NS(_decode_fastpath_enabled=True, _capturable_logits=True),
            _shared_graph_enabled=True, _shared_graph=None, scale=np.float32(.7),
            experts=NS(expert_executor=executor, layer_id=i),
            enqueue_gpu_branch=lambda x: x * np.float32(.125),
            combine_gpu_branch=lambda r, s: r + s)
        if recurrent:
            b.mixer = NS(graph_decode=False, spec=NS(d_model=4, conv_kernel_size=2,
                qkv_channels=4, n_value_heads=1, head_dim=2), mixer=NS(_decode_fastpath_enabled=True))
        else:
            b.mixer = NS(config=QSAConfig(4, n_q_heads=2, n_kv_heads=1, head_dim=4,
                rotary_dim=4, index_heads=1, index_dim=4, index_topk=5, block_size=4,
                sections=(1, 1, 0, 0)), fast_decode=True, _frontend_capturable=True,
                _frontend_graph=None, _state=lambda s: None)
        blocks.append(b)
        p = NS(executor=executor, layer=i, fingerprint=(0, 0, 1024),
            cache_fingerprint=lambda e: (0, 0, 1024), stream_ptr=7, device=0,
            retired=NS(record=lambda s, i=i: events.append(('retire', i))), pending=False)
        h = NS(plan=p, capture_enabled=True, service=NS(check=lambda: checks.append('private')),
            stage=None, handle=i + 1, graph=None, graph_branch=None,
            route_heat=None, timeline=None, stats=np.zeros(6, np.uint64),
            accounted=np.zeros(6, np.uint64), totals=np.zeros(6, np.uint64))
        def enqueue(x, weights, ids, callback, *, gpu_branch, h=h):
            assert callback is None
            if not stream.capturing:
                h.totals[0] += 1
            return gpu_branch.combine_gpu_branch(x * np.float32(.25), gpu_branch.enqueue_gpu_branch(x))
        h.enqueue = enqueue
        h.bridge = NS(mylocal_ggml_handoff_stats=lambda *args, h=h: h.stats.__setitem__(slice(None), h.totals))
        h.flush_statistics = lambda i=i: events.append(('flush', i))
        p.native_handoff = h
        executor._device_expert_plans[i] = p

    ple = NS(width=4, _closed=False)
    ple.embedding = lambda token, predecessors: DeviceArray(np.full(4, token + sum(predecessors), np.float32))
    def apply(streams, embedding, history):
        z = streams * np.float32(.1) + embedding[None]
        return streams + z + history[0], DeviceArray(np.concatenate((history[1:], z[None])))
    ple.apply_embedding = apply
    blocks[1].ple = ple
    model = NS(_closed=False, blocks=tuple(blocks), diagnostic_callback=None, final_hc=hc(True),
        config=NS(hidden_size=4, vocab_size=4), weights=None, _pre_moe_segments={},
        _qsa_block_segments={}, embedding=lambda token: DeviceArray(np.full(4, token, np.float32)),
        output=lambda x: x * np.float32(.3))
    def validate_state(s):
        if s.owner is not model or s.failed or s.length >= s.max_length:
            raise ValueError('bad generation state')
    model._validate = validate_state
    for i in (0, 2):
        a: Any = object.__new__(Qwen4ExpQSABlockGraph)
        a.block, a.attention = blocks[i], blocks[i].mixer
        a.plan = executor._device_expert_plans[i]
        a.handoff, a.fingerprint = a.plan.native_handoff, a.plan.fingerprint
        a.closed, a.graph = False, None
        a.frontend = NS(graph=None)
        a.validation_status, a.position = array((), np.uint32), array((), np.int64)
        a._warm_profile_before = None
        def operands(streams, state, t, a=a):
            validate('streams', streams, np.float32, (4, 4))
            factors = tuple(DeviceArray(np.full((1, 2), t + j, np.float32)) for j in range(4))
            return t, (state.keys, state.values, state.raw_index, state.pooled,
                       a.validation_status, a.position, *factors)
        a._operands = operands
        def qsa_enqueue(streams, arrays, a=a):
            warming = not stream.capturing
            try:
                for cache in arrays[:4]:
                    cache[...] += np.float32(1) + arrays[5].astype(np.float32)
                hidden = streams.mean(axis=0) + arrays[6].sum() * np.float32(.01)
                return streams + a.handoff.enqueue(hidden[None], None, None, None, gpu_branch=a.block.moe)
            finally:
                if warming:
                    stream.synchronize()
                    a._account_warmup()
        a._enqueue = qsa_enqueue
        model._qsa_block_segments[i] = a

    def state():
        return NS(owner=model, failed=False, length=0, max_length=8,
            recurrent={1: DeltaMixerState(CausalConvState(array((1, 1, 4))), DeltaRuleState(array((1, 1, 2, 2))))},
            qsa={i: NS(length=0, max_length=8, keys=array((8, 1, 4)), values=array((8, 1, 4)),
                       raw_index=array((8, 4)), pooled=array((3, 4))) for i in (0, 2)},
            ple={1: PLEState((-1, -1), array((9, 4, 4)))})

    class Capture:
        fail = None
        def __init__(self, enqueue, x, arrays, **kwargs):
            events.append('new_pool')
            self.enqueue, self.bound, self.calls = enqueue, kwargs['bound_mutable'], 0
            self.closed = False
            self.mutable = arrays
            self.pointers = tuple(arrays[i].data.ptr for i in self.bound)
            assert kwargs['copy_back_mutable'] is False
            assert kwargs['pool_limit_bytes'] == 256 * 1024**2
            enqueue(x.copy(), tuple(a.copy() for a in arrays))
            if self.fail == 'warmup':
                raise RuntimeError('injected capture failure')
            stream.capturing = True
            try:
                enqueue(x.copy(), tuple(a.copy() for a in arrays))
            finally:
                stream.capturing = False
            if self.fail == 'capture':
                raise RuntimeError('injected capture failure')
        def matches(self, x, arrays):
            return self.pointers == tuple(arrays[i].data.ptr for i in self.bound)
        def forward(self, x, arrays, *, borrow_output):
            assert borrow_output is False
            stream.capturing = True
            try:
                result = self.enqueue(x, arrays)
            finally:
                stream.capturing = False
            self.calls += 1
            return tuple(a.copy() for a in result) if isinstance(result, tuple) else result.copy()
        def close(self):
            self.closed = True
            events.append('close_pool')
        def bound_arrays(self):
            return tuple(self.mutable[i] for i in self.bound)
        def statistics(self):
            return dict(pool_used_bytes=64, pool_reserved_bytes=128)
    monkeypatch.setattr(module, 'CapturedArrayInference', Capture)
    return NS(model=model, state=state, events=events, checks=checks, positions=position_calls,
              capture=Capture, stream=stream, executor=executor)


def all_arrays(state):
    return (state.recurrent[1].conv.history, state.recurrent[1].delta.memory,
            *(a for s in state.qsa.values() for a in (s.keys, s.values, s.raw_index, s.pooled)),
            *(s.normalized_history for s in state.ple.values()))


def test_cpu_backend_rejected():
    if BACKEND_NAME != 'cupy':
        assert not Qwen4ExpTokenGraph.is_eligible(None, None)
        with pytest.raises(RuntimeError, match='CuPy'):
            Qwen4ExpTokenGraph(None)


def test_owned_outputs_history_lengths_readonly_runtime_and_recapture(host):
    m, s = host.model, host.state()
    assert Qwen4ExpTokenGraph.is_eligible(m, s)
    assert host.events == host.checks == host.positions == []
    g: Any = Qwen4ExpTokenGraph(m)
    old = s.ple[1]
    old_bits = old.normalized_history.copy()
    logits = g.forward(2, s)
    first_logits, first_history = logits.copy(), s.ple[1].normalized_history.copy()
    assert logits.shape == (1, 4)
    assert s.length == 0  # The parent's post-validation increment is not ours.
    assert s.recurrent[1].conv.length == s.recurrent[1].delta.length == 1
    assert all(q.length == 1 for q in s.qsa.values())
    assert s.ple[1] is not old and s.ple[1].predecessors == (-1, 2)
    np.testing.assert_array_equal(old.normalized_history, old_bits)
    assert len(host.checks) == 3  # Only private warmup, including both QSA layers.
    assert g._history_index not in g.graph.bound
    assert all(i not in g.graph.bound for i in range(g._runtime_index, g._runtime_index + 5))
    old_graph = g.graph
    s.length += 1
    g.forward(3, s)
    assert g.captures == 1 and len(host.positions) == 2 and len(host.checks) == 3
    np.testing.assert_array_equal(logits, first_logits)
    np.testing.assert_array_equal(old.normalized_history, old_bits)
    assert not np.shares_memory(s.ple[1].normalized_history, first_history)
    other = host.state()
    untouched = [a.copy() for a in all_arrays(s)]
    g.forward(1, other)
    assert old_graph.closed and g.captures == 2 and len(host.checks) == 6
    for a, before in zip(all_arrays(s), untouched):
        np.testing.assert_array_equal(a, before)
    assert g.graph_statistics()['scope'] == 'whole_token'
    assert len(g.graph.bound_arrays()) == 12
    assert sum(isinstance(e, tuple) and e[0] == 'retire' for e in host.events) == 9
    g.close()
    g.close()
    assert g.graph is None and g.closed
    assert all(p.native_handoff.handle for p in host.executor._device_expert_plans.values())


@pytest.mark.parametrize('phase', ['warmup', 'capture'])
def test_capture_failure_never_advances_caller_state(host, phase):
    s, g = host.state(), Qwen4ExpTokenGraph(host.model)
    before = [a.copy() for a in all_arrays(s)]
    old = s.ple[1]
    host.capture.fail = phase
    with pytest.raises(RuntimeError, match='injected capture failure'):
        g.forward(1, s)
    assert g.graph is None and g.calls == g.captures == s.length == 0
    assert s.ple[1] is old
    assert s.recurrent[1].conv.length == s.recurrent[1].delta.length == 0
    assert all(q.length == 0 for q in s.qsa.values())
    for a, snapshot in zip(all_arrays(s), before):
        np.testing.assert_array_equal(a, snapshot)
    assert all(p.pending for p in host.executor._device_expert_plans.values())
    g.close()


@pytest.mark.parametrize('defect', ['diagnostic', 'plan', 'stage', 'admission', 'qsa', 'factors',
    'ple_api', 'ple_layer', 'shared', 'context', 'payload', 'contiguous', 'position', 'alias'])
def test_preflight_not_ready_is_metadata_only(host, defect):
    m, s = host.model, host.state()
    p = host.executor._device_expert_plans[1]
    if defect == 'diagnostic': m.diagnostic_callback = lambda *args: None
    elif defect == 'plan': del host.executor._device_expert_plans[1]
    elif defect == 'stage': p.native_handoff.stage = object()
    elif defect == 'admission': host.executor.cache._admission_limit = 1
    elif defect == 'qsa': del m._qsa_block_segments[2]
    elif defect == 'factors':
        from dataclasses import replace
        m.blocks[2].mixer.config = replace(m.blocks[2].mixer.config, rope_base=100.)
    elif defect == 'ple_api': m.blocks[1].ple.apply_embedding = None
    elif defect == 'ple_layer': m.blocks[0].ple = m.blocks[1].ple
    elif defect == 'shared': m.blocks[1].moe._shared_graph_enabled = False
    elif defect == 'context': s.length = s.qsa[0].length = s.qsa[2].length = 8
    elif defect == 'payload': s.qsa[0].pooled = array((2 * 1024**2,))
    elif defect == 'contiguous': s.recurrent[1].conv.history = array((1, 1, 8))[..., ::2]
    elif defect == 'position': s.recurrent[1].conv.length = 1
    elif defect == 'alias': s.qsa[2].keys = s.qsa[0].keys
    assert not Qwen4ExpTokenGraph.is_eligible(m, s)
    assert host.events == host.checks == host.positions == []


@pytest.fixture
def weighted_host(host):
    """Canonical-sized metadata only; no packed payloads or model execution."""
    m, executor = host.model, host.executor
    recurrent, attention = m.blocks[1], m.blocks[0]
    template = m._qsa_block_segments[0]
    base_plan = executor._device_expert_plans[0]
    blocks, adapters, tensors = [], {}, {}
    executor._device_expert_plans.clear()
    for i in range(48):
        b = NS(**vars(recurrent if (i + 1) % 4 else attention))
        b.ple = recurrent.ple if i == 1 else None
        b.moe = NS(**vars(b.moe))
        b.moe.experts = NS(expert_executor=executor, layer_id=i)
        plan = NS(**vars(base_plan))
        plan.layer = i
        plan.native_handoff = NS(**vars(base_plan.native_handoff))
        plan.native_handoff.plan = plan
        executor._device_expert_plans[i] = plan
        if not b.recurrent:
            a: Any = object.__new__(Qwen4ExpQSABlockGraph)
            a.__dict__.update(vars(template))
            a.block, a.attention, a.plan, a.handoff = b, b.mixer, plan, plan.native_handoff
            a.validation_status = array((), np.uint32)
            adapters[i] = a
        names = (('attn_qkv', 'attn_gate', 'ssm_beta', 'ssm_alpha', 'ssm_out') if b.recurrent else
                 ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'indexer.k_proj'))
        names += ('ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp')
        if b.ple is not None:
            names += ('ple_key', 'ple_value')
        tensors.update({f'blk.{i}.{name}.weight': NS(quant_type='F32') for name in names})
        blocks.append(b)
    tensors['output.weight'] = NS(quant_type='Q6_K')
    m.blocks, m._qsa_block_segments = tuple(blocks), adapters
    m.weights = NS(closed=False, plan=object(), store=NS(closed=False, tensors=tensors),
        headers=NS(recurrent=tuple(b.recurrent for b in blocks), attention=attention.mixer.config),
        device={'output.weight': NS(quant_type='Q6_K')})
    return host


@pytest.mark.parametrize('quant_type', ['Q6_K', 'Q5_K', 'Q8_0', 'IQ4_XS'])
def test_resident_head_uses_supported_checkpoint_encoding_with_q8_activations(weighted_host, monkeypatch, quant_type):
    from mini_llm.ops.mixed_q8_gpu import FORMATS
    assert quant_type in FORMATS
    m = weighted_host.model
    head = m.weights.device['output.weight']
    head.quant_type = m.weights.store.tensors['output.weight'].quant_type = quant_type
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', '1')
    native, qsa, ple = Qwen4ExpTokenGraph._configuration(m)
    assert len(native) == 48 and len(qsa) == 12 and ple == 1
    assert m.weights.device['output.weight'] is head and head.quant_type == quant_type
    assert weighted_host.events == weighted_host.checks == weighted_host.positions == []


@pytest.mark.parametrize('defect', ['unsupported', 'not_resident', 'q8_disabled', 'q8_unset', 'q8_not_exact'])
def test_head_metadata_rejects_unsupported_unresident_or_non_q8_backend(weighted_host, monkeypatch, defect):
    m = weighted_host.model
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', '1')
    if defect == 'unsupported':
        m.weights.device['output.weight'].quant_type = 'F32'
        m.weights.store.tensors['output.weight'].quant_type = 'F32'
        message = 'supported packed GPU'
    elif defect == 'not_resident':
        del m.weights.device['output.weight']
        message = 'GPU resident'
    else:
        if defect == 'q8_unset':
            monkeypatch.delenv('MINI_LLM_LINEAR_GPU_Q8', raising=False)
        else:
            monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', '0' if defect == 'q8_disabled' else 'true')
        message = 'MINI_LLM_LINEAR_GPU_Q8=1'
    with pytest.raises(ValueError, match=message):
        Qwen4ExpTokenGraph._configuration(m)
    assert weighted_host.events == weighted_host.checks == weighted_host.positions == []


@pytest.mark.parametrize('overlap', ['partial_cache', 'nested_ple', 'status'])
def test_sorted_ranges_reject_all_operand_overlap_kinds(host, overlap):
    m, s = host.model, host.state()
    if overlap == 'partial_cache':
        storage = array((48,))
        s.qsa[0].keys = storage[:32].reshape(8, 1, 4)
        s.qsa[2].values = storage[16:].reshape(8, 1, 4)
    elif overlap == 'nested_ple':
        s.qsa[0].keys = s.ple[1].normalized_history.reshape(-1)[16:48].reshape(8, 1, 4)
    else:
        m._qsa_block_segments[2].validation_status = s.qsa[0].keys.reshape(-1)[:1].view(np.uint32).reshape(())
    native, qsa, ple = Qwen4ExpTokenGraph._configuration(m)
    with pytest.raises(ValueError, match='must not alias'):
        Qwen4ExpTokenGraph._validate_states(m, s, native, qsa, ple)
    assert host.events == host.checks == host.positions == []


def test_sorted_ranges_read_each_pointer_once_accept_adjacency_and_ignore_empty(host, monkeypatch):
    m, s = host.model, host.state()
    storage = array((64,))
    s.qsa[0].keys = storage[:32].reshape(8, 1, 4)
    s.qsa[2].keys = storage[32:].reshape(8, 1, 4)
    s.qsa[0].pooled = array((0, 4))  # Metadata fixture permits an empty operand.
    native, qsa, ple = Qwen4ExpTokenGraph._configuration(m)
    reads = {}
    def pointer(a):
        reads[id(a)] = reads.get(id(a), 0) + 1
        return NS(ptr=a.__array_interface__['data'][0])
    monkeypatch.setattr(DeviceArray, 'data', property(pointer))
    arrays = Qwen4ExpTokenGraph._validate_states(m, s, native, qsa, ple)
    assert len(reads) == sum(bool(a.nbytes) for a in arrays)
    assert all(count == 1 for count in reads.values())
    assert host.events == host.checks == host.positions == []


def test_live_fingerprint_is_error_not_fallback_and_validates_before_launch(host):
    m, s = host.model, host.state()
    g = m._token_graph = Qwen4ExpTokenGraph(m)
    g.forward(1, s)
    s.length += 1
    host.events.clear()
    host.executor._device_expert_plans[1].fingerprint = ()
    with pytest.raises(ValueError, match='fingerprint'):
        Qwen4ExpTokenGraph.is_eligible(m, s)
    with pytest.raises(ValueError, match='fingerprint'):
        g.forward(1, s)
    assert host.events == [] and s.length == 1
    g.close()


def test_invalid_last_state_cannot_launch_or_retire_old_pools(host):
    m, s = host.model, host.state()
    g = Qwen4ExpTokenGraph(m)
    s.qsa[2].length = 1
    with pytest.raises(ValueError, match='contiguous'):
        g.forward(1, s)
    assert g.graph is None and host.events == host.positions == []
    g.close()


def test_old_pools_deduplicated_chain_members_wrappers_borrowed(host):
    m = host.model
    old = NS(closed=False)
    def close():
        assert not old.closed
        old.closed = True
        host.events.append('old_pool')
    old.close = close
    member = NS(graph=old, closed=False)
    chain = NS(graph=old, members=(member,), closed=False)
    m._pre_moe_segments = {1: chain}
    for a in m._qsa_block_segments.values():
        a.graph = a.frontend.graph = old
        a.attention._frontend_graph = NS(graph=old)
    for b, p in zip(m.blocks, host.executor._device_expert_plans.values()):
        b.moe._shared_graph = p.native_handoff.graph = old
    g = Qwen4ExpTokenGraph(m)
    g.forward(1, host.state())
    assert host.events.count('old_pool') == 1
    assert host.events.index('old_pool') < host.events.index('new_pool')
    assert member.graph is chain.graph is None and not member.closed and not chain.closed
    assert not any(a.closed for a in m._qsa_block_segments.values())
    g.close()


def test_qsa_and_gdn_warm_profile_snapshots_precede_private_batches(host):
    g = Qwen4ExpTokenGraph(host.model)
    for c in g._native:
        h = c.handoff
        h.route_heat = True
        h.profile_warmup = np.zeros(1, np.uint64)
        h.read_profile = lambda h=h: h.totals[:1].copy()
    g.forward(1, host.state())
    for i, c in enumerate(g._native):
        np.testing.assert_array_equal(c.handoff.profile_warmup, [1])
        if i in g._qsa:
            np.testing.assert_array_equal(g._qsa[i]._warm_profile_before, [0])
    g.close()


def test_no_ple_is_supported(host):
    host.model.blocks[1].ple = None
    s = host.state()
    s.ple.clear()
    g = Qwen4ExpTokenGraph(host.model)
    result = g.forward(2, s)
    assert result.shape == (1, 4) and g._history_index is None and g.graph_statistics()['ple_layers'] == 0
    g.close()


# Parent-only actual CUDA tests. Import the fixtures, not any model/checkpoint.
from test_0317_qsa_block_graph import math_block  # noqa: E402,F401
from test_0313_recurrent_block_graph import configuration, fake_plan  # noqa: E402,F401


gpu = pytest.mark.gpu
cuda = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Parent isolated GPU run only')


@pytest.fixture
def tiny_model(math_block, fake_plan):
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from mini_llm.ops.inference_matmul import matmul_inference
    qsa, attention, rand, checks = math_block
    pre, recurrent, plan, _ = fake_plan
    qsa.block.moe._shared_graph_enabled = True
    e = qsa.plan.executor
    plan.executor = recurrent.moe.experts.expert_executor = e
    plan.layer = recurrent.moe.experts.layer_id = 1
    plan.fingerprint = qsa.plan.fingerprint
    plan.cache_fingerprint = qsa.plan.cache_fingerprint
    e._device_expert_plans[1] = plan
    c = qsa.block.attention_hc
    final = Qwen4ExpHyperConnection(c.gamma, c.down, c.up, epsilon=float(c.epsilon),
        fast_decode=True, native_read=True)
    embeddings, head = rand((11, 256)), rand((11, 256))
    model = Qwen4ExpGGUFModel.from_components(width=256, vocab=11,
        blocks=(qsa.block, recurrent), final_hc=final,
        embedding=lambda token: embeddings[token].copy(),
        output=lambda x: matmul_inference(x[None], head.T).reshape(-1), max_length=16)
    model._pre_moe_segments = {1: pre}
    model._qsa_block_segments = {0: qsa}
    return model, qsa, rand, checks


def reference_token(model, token, state):
    streams = model.final_hc.initialize(model.embedding(token))
    for i, b in enumerate(model.blocks):
        if b.ple is not None:
            streams, state.ple[i] = b.ple.step(token, streams, state.ple[i])
        hidden, gains = b.attention_hc.read(streams)
        mixed = (b.mixer.forward(hidden[None, None], state.recurrent[i])[0, 0] if b.recurrent else
                 b.mixer.step(hidden, state.qsa[i], position=state.length))
        streams = b.attention_hc.write(streams, mixed, gains)
        hidden, gains = b.ffn_hc.read(streams)
        weights, ids = b.moe.router.forward(hidden.reshape(1, 1, -1))
        handoff = b.moe.experts.expert_executor._device_expert_plans[i].native_handoff
        mixed = handoff.enqueue(hidden[None], (weights * b.moe.scale).reshape(-1), ids.reshape(-1),
                                None, gpu_branch=b.moe)
        streams = b.ffn_hc.write(streams, mixed.reshape(-1), gains)
    state.length += 1
    return model.output(model.final_hc.collapse(streams)).reshape(1, -1)


@gpu
@cuda
@pytest.mark.parametrize('with_ple', [False, True])
def test_tiny_real_math_exact_state_dynamic_positions_owned_outputs(tiny_model, with_ple):
    from mini_llm.external.qwen4exp_ple import Qwen4ExpPLE
    from mini_llm.ops.inference_matmul import matmul_inference
    model, qsa, rand, checks = tiny_model
    if with_ple:
        if not hasattr(Qwen4ExpPLE, 'apply_embedding'):
            pytest.skip('Parent must provide the validated PLE.apply_embedding math adapter')
        ple: Any = object.__new__(Qwen4ExpPLE)
        ple.width, ple._closed, ple.epsilon = 256, False, xp.float32(1e-5)
        for name in ('gamma_key', 'gamma_query', 'gamma_conv'):
            setattr(ple, name, xp.ones((4, 256), xp.float32))
        ple.conv = rand((1024, 4))
        key, value, table = rand((1024, 256)), rand((256, 256)), rand((11, 256))
        ple.key = lambda x: matmul_inference(x[None], key.T).reshape(-1)
        ple.value = lambda x: matmul_inference(x[None], value.T).reshape(-1)
        ple.embedding = lambda token, predecessors: table[token].copy()
        model.blocks[1].ple = ple
    actual = model.create_generation_state(max_length=16)
    expected = model.create_generation_state(max_length=16)
    graph = Qwen4ExpTokenGraph(model)
    saved = []
    try:
        for token in (2, 7, 4, 3, 9, 1, 6, 5, 8):
            old = actual.ple[1] if with_ple else None
            old_history = old.normalized_history.get().copy() if old is not None else None
            logits = graph.forward(token, actual)
            assert actual.length == expected.length
            actual.length += 1  # Parent's post-validation commit.
            oracle = reference_token(model, token, expected)
            np.testing.assert_array_equal(logits.get().view(np.uint32), oracle.get().view(np.uint32))
            assert int(qsa.validation_status.get()) == 0
            for name in ('keys', 'values', 'raw_index', 'pooled'):
                np.testing.assert_array_equal(getattr(actual.qsa[0], name).get().view(np.uint32),
                                              getattr(expected.qsa[0], name).get().view(np.uint32))
            for a, b in ((actual.recurrent[1].conv.history, expected.recurrent[1].conv.history),
                         (actual.recurrent[1].delta.memory, expected.recurrent[1].delta.memory)):
                np.testing.assert_array_equal(a.get().view(np.uint32), b.get().view(np.uint32))
            assert actual.recurrent[1].length == actual.qsa[0].length == actual.length
            if with_ple:
                assert old is not None
                np.testing.assert_array_equal(old.normalized_history.get(), old_history)
                np.testing.assert_array_equal(actual.ple[1].normalized_history.get().view(np.uint32),
                                              expected.ple[1].normalized_history.get().view(np.uint32))
            saved.append((logits, logits.get().copy()))
        assert graph.captures == 1 and len(checks) == 1
        replacement = model.create_generation_state(max_length=16)
        graph.forward(2, replacement)
        assert graph.captures == 2 and len(checks) == 2
        assert graph.graph_statistics()['pool_reserved_bytes'] <= graph.pool_limit_bytes
        graph.close()
        for output, snapshot in saved:
            np.testing.assert_array_equal(output.get(), snapshot)
    finally:
        graph.close()


@gpu
@cuda
def test_tiny_capture_failure_keeps_all_bound_state_private(tiny_model):
    model, qsa, _, _ = tiny_model
    state = model.create_generation_state(max_length=16)
    arrays = (state.recurrent[1].conv.history, state.recurrent[1].delta.memory,
              state.qsa[0].keys, state.qsa[0].values, state.qsa[0].raw_index, state.qsa[0].pooled)
    before = [a.get().copy() for a in arrays]
    graph = Qwen4ExpTokenGraph(model)
    output = model.output
    def fail(x):
        result = output(x)
        if xp.cuda.get_current_stream().is_capturing():
            raise RuntimeError('injected whole-token capture failure')
        return result
    model.output = fail
    # Establish the injected callback as the immutable source before capture.
    graph._source = graph._source_identity()
    try:
        with pytest.raises(RuntimeError, match='injected whole-token'):
            graph.forward(2, state)
        assert state.length == state.recurrent[1].length == state.qsa[0].length == 0
        assert graph.graph is None
        for a, bits in zip(arrays, before):
            np.testing.assert_array_equal(a.get(), bits)
        for b in model.blocks:
            plan = b.moe.experts.expert_executor._device_expert_plans[b.moe.experts.layer_id]
            assert plan.pending
            plan.retired.synchronize()
    finally:
        graph.close()
