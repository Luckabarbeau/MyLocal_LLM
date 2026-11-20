"""CPU contracts plus actual CUDA math/capture tests for the parent's isolated run.

No checkpoints or model executions. GPU cases use real QSA/HC/router arithmetic
and a GPU-only fake native service, not a native CPU expert arithmetic oracle.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.external import qwen4exp_qsa_block_graph as module
from mini_llm.external.qwen4exp_qsa_block_graph import Qwen4ExpQSABlockGraph
from mini_llm.ops import qsa_graph_gpu as kernels
from mini_llm.ops.qsa_decode_gpu import _SOURCE as SERIAL_SOURCE


gpu = pytest.mark.gpu
cuda = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Actual CUDA capture; parent isolated run only')
CACHE_NAMES = ('keys', 'values', 'raw_index', 'pooled')


def test_import_and_cpu_backend_rejection():
    assert Qwen4ExpQSABlockGraph.complete_block
    assert Qwen4ExpQSABlockGraph.pool_limit_bytes == 8*1024**2
    if BACKEND_NAME != 'cupy':
        with pytest.raises(RuntimeError, match='CuPy'):
            Qwen4ExpQSABlockGraph(None, None, None)


def test_serial_source_changes_only_entry_signature_and_runtime_n():
    source = kernels._SOURCE.replace('void qsa_decode_graph(', 'void qsa_decode(')
    source = source.replace('const long long* position, int dim, int groups, float scale,',
                            'int n, int dim, int groups, float scale,')
    source = source.replace('    const int n = (int)(position[0] + 1);\n', '')
    assert source == SERIAL_SOURCE  # Includes every serial dot/softmax/gate operation.
    assert 'token == n - 1 ? v[' in kernels._SOURCE
    assert 'cell == n - 1 ? k[' in kernels._SOURCE


@pytest.mark.parametrize('old', ['void qsa_decode(', '    int n, int dim, int groups, float scale,',
    '    const int h = blockIdx.x, kh = h / groups, lane = threadIdx.x;'])
def test_serial_source_adapter_fails_closed_on_source_drift(old):
    with pytest.raises(ValueError, match='contract changed'):
        kernels._runtime_source(SERIAL_SOURCE.replace(old, '', 1))
    with pytest.raises(ValueError, match='contract changed'):
        kernels._runtime_source(SERIAL_SOURCE + old)


class MetadataArray:
    def __init__(self, shape, dtype=np.float32, device=0, contiguous=True):
        self.shape, self.ndim = shape, len(shape)
        self.dtype, self.device = np.dtype(dtype), NS(id=device)
        self.flags = NS(c_contiguous=contiguous)
        self.size = int(np.prod(shape))


@pytest.fixture
def metadata_backend(monkeypatch):
    backend = NS(ndarray=MetadataArray, float32=np.float32, uint32=np.uint32, int64=np.int64,
                 cuda=NS(runtime=NS(getDevice=lambda: 0)))
    monkeypatch.setattr(kernels, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(kernels, 'xp', backend)
    monkeypatch.setattr(kernels, '_kernel', lambda *args: pytest.fail('Invalid metadata launched a kernel'))
    return backend


@pytest.mark.parametrize('bad', ['dtype', 'device', 'shape', 'layout', 'host'])
def test_device_operand_validation_without_cuda(metadata_backend, bad):
    a = MetadataArray((), np.int64)
    if bad == 'dtype': a.dtype = np.dtype(np.float64)
    if bad == 'device': a.device.id = 1
    if bad == 'shape': a.shape = (1,)
    if bad == 'layout': a.flags.c_contiguous = False
    if bad == 'host': a = np.zeros((), np.int64)
    with pytest.raises((TypeError, ValueError)):
        kernels._position(a)


@pytest.mark.parametrize('t,maximum', [(-1, 10), (True, 10), (1.5, 10), (11, 10),
                                       (0, 2**31-1), (0, -1)])
def test_runtime_position_bound_before_launch(metadata_backend, t, maximum):
    with pytest.raises(ValueError, match='runtime'):
        kernels.set_position_gpu(MetadataArray((), np.int64), t, maximum)


@pytest.mark.parametrize('bad', ['qshape', 'qgshape', 'groups', 'capacity', 'scale', 'dtype'])
def test_decode_metadata_rejects_invalid_inputs(metadata_backend, bad):
    args = [MetadataArray(s) for s in ((4, 8), (4, 16), (2, 8), (2, 8), (20, 2, 8), (20, 2, 8))]
    scale = .37
    if bad == 'qshape': args[0] = MetadataArray((32,))
    if bad == 'qgshape': args[1] = MetadataArray((4, 8))
    if bad == 'groups': args[2] = MetadataArray((3, 8))
    if bad == 'capacity': args[4] = MetadataArray((0, 2, 8))
    if bad == 'scale': scale = np.nan
    if bad == 'dtype': args[2].dtype = np.dtype(np.float16)
    with pytest.raises((ValueError, TypeError)):
        kernels.qsa_decode_graph_gpu(*args, MetadataArray((), np.int64), scale)


def test_plan_checks_are_metadata_only(monkeypatch):
    from mini_llm.external import qwen4exp_recurrent_block_graph as lifecycle
    fingerprint = (0, 0, 1024)
    executor = NS(cache=NS(_admission_limit=0, idle_eviction_requests=0), _device_expert_plans={})
    plan = NS(executor=executor, layer=0, fingerprint=fingerprint,
              cache_fingerprint=lambda e: fingerprint, stream_ptr=7, device=0)
    handoff = NS(plan=plan, capture_enabled=True, service=NS(check=lambda: pytest.fail('blocking check')),
                 stage=None, handle=1, enqueue=lambda: None)
    plan.native_handoff = handoff
    executor._device_expert_plans[0] = plan
    graph = object.__new__(Qwen4ExpQSABlockGraph)
    graph.plan, graph.handoff, graph.fingerprint = plan, handoff, fingerprint
    graph.block = NS(moe=NS(experts=NS(expert_executor=executor, layer_id=0),
        enqueue_gpu_branch=lambda: None, combine_gpu_branch=lambda: None))
    stream = NS(ptr=7, is_capturing=lambda: False)
    monkeypatch.setattr(lifecycle, 'xp', NS(cuda=NS(get_current_stream=lambda: stream,
                        runtime=NS(getDevice=lambda: 0))))
    graph._validate_plan()
    plan.fingerprint = ()
    with pytest.raises(ValueError, match='fingerprint'):
        graph._validate_plan()
    plan.fingerprint = fingerprint
    stream.ptr = 8
    with pytest.raises(ValueError, match='stream'):
        graph._validate_plan()


def test_upgrade_retires_old_pools_only_after_plan_validation(monkeypatch):
    class Attention:
        pass
    class OldGraph:
        def __init__(self): self.closed = False
        def close(self): self.closed = True
    old = [OldGraph() for _ in range(3)]
    a = Attention()
    a.fast_decode = a._frontend_capturable = True
    a._frontend_graph = NS(graph=old[2])
    h = NS(graph=old[0], graph_branch=object(), flush_statistics=lambda: None)
    block = NS(recurrent=False, ple=None, mixer=a, attention_hc=NS(native_read=True),
        ffn_hc=NS(native_read=True), moe=NS(_shared_graph=old[1], router=NS(
            _decode_fastpath_enabled=True, _capturable_logits=True)))
    plan = NS(native_handoff=h, fingerprint=())
    monkeypatch.setattr(module, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(module, 'Qwen4ExpAttention', Attention)
    monkeypatch.setattr(module, '_array', lambda *args: None)
    monkeypatch.setattr(module, 'xp', NS(uint32=np.uint32, int64=np.int64,
                                      empty=lambda *args, **kwargs: object()))
    def reject(self): raise ValueError('ineligible frozen plan')
    monkeypatch.setattr(Qwen4ExpQSABlockGraph, '_validate_plan', reject)
    with pytest.raises(ValueError, match='ineligible'):
        Qwen4ExpQSABlockGraph(block, plan, object())
    assert not any(g.closed for g in old)
    monkeypatch.setattr(Qwen4ExpQSABlockGraph, '_validate_plan', lambda self: None)
    Qwen4ExpQSABlockGraph(block, plan, object())
    assert all(g.closed for g in old)
    assert h.graph is h.graph_branch is block.moe._shared_graph is a._frontend_graph.graph is None


@pytest.mark.parametrize('outcome', ['success', 'failure', 'invalid_capture'])
def test_forward_uses_readonly_factors_once_and_external_retirement(monkeypatch, outcome):
    graph = object.__new__(Qwen4ExpQSABlockGraph)
    graph.closed, graph.graph, graph.calls, graph.captures = False, None, 0, 0
    graph._validate_plan = lambda: None
    graph.attention = NS(config=NS(index_topk=13, block_size=4))
    graph.position = object()
    arrays = tuple(object() for _ in range(10))
    graph._operands = lambda streams, state, position: (position, arrays)
    capturing = [False]
    stream = NS(is_capturing=lambda: capturing[0])
    records, positions = [], []
    graph.plan = NS(retired=NS(record=lambda s: records.append(s)), pending=False)
    graph.handoff = NS(flush_statistics=lambda: None, route_heat=None)
    graph._enqueue = object()
    result = object()
    class FakeCapture:
        def __init__(self, enqueue, streams, actual, **kwargs):
            assert actual is arrays and enqueue is graph._enqueue
            assert kwargs == dict(pool_limit_bytes=8*1024**2, copy_back_mutable=False,
                                  bound_mutable=(0, 1, 2, 3, 4))
            self.calls = 0
        def matches(self, streams, actual): return actual is arrays
        def forward(self, streams, actual, *, borrow_output):
            assert borrow_output
            if outcome != 'success':
                capturing[0] = outcome == 'invalid_capture'
                raise RuntimeError('injected replay failure')
            self.calls += 1
            return result
        def close(self): pass
    monkeypatch.setattr(module, 'CapturedArrayInference', FakeCapture)
    monkeypatch.setattr(module, 'set_position_gpu', lambda p, t, bound: positions.append((p, t, bound)))
    monkeypatch.setattr(module, 'xp', NS(cuda=NS(get_current_stream=lambda: stream)))
    state = NS(length=3, max_length=20)
    if outcome == 'success':
        assert graph.forward(None, state, 3) is result
        assert state.length == 4 and graph.calls == 1
        assert graph.forward(None, state, 4) is result
        assert state.length == 5 and graph.calls == 2 and graph.captures == 1
    else:
        with pytest.raises(RuntimeError, match='injected replay'):
            graph.forward(None, state, 3)
        assert state.length == 3 and graph.calls == 0 and graph.graph is None
    assert all(bound == 15 for _, _, bound in positions)
    assert len(records) == (len(positions) if outcome != 'invalid_capture' else 0)
    assert graph.plan.pending == (outcome != 'invalid_capture')


def test_warmup_errors_remain_unaccounted():
    graph = object.__new__(Qwen4ExpQSABlockGraph)
    h = NS(stats=np.zeros(6, np.uint64), accounted=np.zeros(6, np.uint64), handle=1)
    h.bridge = NS(mylocal_ggml_handoff_stats=lambda *args: h.stats.__setitem__(3, 1))
    h.service = NS(check=lambda: pytest.fail('error must remain visible before check'))
    graph.handoff = h
    with pytest.raises(RuntimeError, match='warmup failed'):
        graph._account_warmup()
    assert h.accounted[3] == 0 and h.stats[3] == 1


@pytest.fixture
def math_block(monkeypatch):
    if BACKEND_NAME != 'cupy':
        pytest.skip('Actual CUDA math graph')
    from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    from mini_llm.ops.inference_matmul import matmul_inference
    from mini_llm.ops.router_inference import RouterInference
    monkeypatch.setenv('MINI_LLM_QSA_COOPERATIVE', '0')
    rng = np.random.default_rng(317)
    def rand(shape): return xp.asarray(rng.normal(0, .03, shape).astype(np.float32))
    def hc():
        return Qwen4ExpHyperConnection(xp.ones((4, 256), xp.float32), rand((16, 1024)),
            rand((1024, 16)), rand((4, 1024)), epsilon=1e-5, fast_decode=True, native_read=True)
    config = QSAConfig(256, n_q_heads=4, n_kv_heads=2, head_dim=256, rotary_dim=4,
        index_heads=2, index_dim=6, index_topk=13, block_size=4, sections=(1, 1, 0, 0))
    shapes = dict(query_gate=(2048, 256), key=(512, 256), value=(512, 256),
                  output=(256, 1024), index_query=(12, 256), index_key=(6, 256))
    projections = {}
    for name, shape in shapes.items():
        weight = rand(shape)
        projections[name] = lambda x, w=weight: matmul_inference(x.reshape(1, -1), w.T).reshape(-1)
    a = Qwen4ExpAttention(config, **projections, gamma_q=xp.ones(256, xp.float32),
        gamma_k=xp.ones(256, xp.float32), gamma_index_q=xp.ones(6, xp.float32),
        gamma_index_k=xp.ones(6, xp.float32), fast_decode=True)
    a._frontend_capturable = True
    a._frontend_graph_enabled = False
    router = RouterInference(256, 512, 10, renormalize_selected=True, normalization_floor=2**-14)
    router.W_router, router.b_router = rand((256, 512)), rand((512,))
    router._decode_fastpath_enabled = router._capturable_logits = True
    executor = NS(cache=NS(_admission_limit=0, idle_eviction_requests=0), _device_expert_plans={})
    moe = NS(router=router, experts=NS(expert_executor=executor, layer_id=0), scale=.7,
        _shared_graph=None, enqueue_gpu_branch=lambda x: x*xp.float32(.125),
        combine_gpu_branch=lambda routed, shared: routed+shared)
    block = NS(recurrent=False, ple=None, mixer=a, attention_hc=hc(), ffn_hc=hc(), moe=moe)
    fingerprint = (0, 0, 1024)
    plan = NS(executor=executor, layer=0, q8=None, fingerprint=fingerprint,
        cache_fingerprint=lambda e: fingerprint, stream_ptr=xp.cuda.get_current_stream().ptr,
        device=xp.cuda.runtime.getDevice(), retired=xp.cuda.Event(disable_timing=True), pending=False)
    totals, checks = np.zeros(6, np.uint64), []
    h = NS(plan=plan, capture_enabled=True, service=NS(check=lambda: checks.append('warmup')),
        stage=None, handle=1, graph=None, graph_branch=None, stats=np.zeros(6, np.uint64),
        accounted=np.zeros(6, np.uint64), route_heat=None, timeline=None)
    def enqueue(x, routes, ids, callback, *, gpu_branch):
        assert callback is None and gpu_branch is moe
        if not xp.cuda.get_current_stream().is_capturing(): totals[0] += 1
        out = gpu_branch.combine_gpu_branch(x*routes[0]+ids[0].astype(xp.float32)*xp.float32(.001),
                                            gpu_branch.enqueue_gpu_branch(x))
        plan.retired.record(xp.cuda.get_current_stream())
        plan.pending = True
        return out
    def stats(*args): h.stats[:] = totals
    def flush():
        stats()
        if h.stats[3] != h.accounted[3]: raise RuntimeError('fake native error')
        h.accounted[:] = h.stats
    h.enqueue, h.flush_statistics = enqueue, flush
    h.bridge = NS(mylocal_ggml_handoff_stats=stats)
    plan.native_handoff = h
    executor._device_expert_plans[0] = plan
    status = xp.zeros((), xp.uint32)
    graph = Qwen4ExpQSABlockGraph(block, plan, status)
    try:
        yield graph, a, rand, checks
    finally:
        graph.close()


def uncaptured(graph, streams, state):
    b = graph.block
    hidden, gains = b.attention_hc.read(streams)
    mixed = b.mixer.step(hidden, state, position=state.length)
    residual = b.attention_hc.write(streams, mixed, gains)
    hidden, gains = b.ffn_hc.read(residual)
    weights, ids = b.moe.router.forward(hidden.reshape(1, 1, -1))
    mixed = graph.handoff.enqueue(hidden.reshape(1, -1), (weights*b.moe.scale).reshape(-1),
                                  ids.reshape(-1), None, gpu_branch=b.moe)
    return b.ffn_hc.write(residual, mixed.reshape(-1), gains)


def snapshot(state):
    return tuple(getattr(state, name).get().view(np.uint32).copy() for name in CACHE_NAMES)


def assert_cache(state, expected):
    for name, bits in zip(CACHE_NAMES, expected):
        np.testing.assert_array_equal(getattr(state, name).get().view(np.uint32), bits)


@gpu
@cuda
def test_complete_block_dynamic_positions_exact_caches_reset_and_pool_boundary(math_block):
    graph, a, rand, checks = math_block
    candidate, reference = a.create_state(20), a.create_state(20)
    # Current/future cache slots are deliberately invalid; projections are the
    # only legal source of the current K/V/raw, including each pool boundary.
    for state in (candidate, reference):
        for name in CACHE_NAMES: getattr(state, name).fill(xp.nan)
    for repeat in range(2):
        for t in range(16):
            streams = rand((4, 256))
            expected = uncaptured(graph, streams, reference)
            actual = graph.forward(streams, candidate, t).copy()
            np.testing.assert_array_equal(actual.get(), expected.get())
            assert np.isfinite(actual.get()).all()
            assert int(graph.validation_status.get()) == 0
            assert candidate.length == reference.length == t+1
            assert_cache(candidate, snapshot(reference))
        assert graph.captures == 1 and checks == ['warmup']
        a.reset(candidate)
        a.reset(reference)
    assert graph.calls == 32
    assert graph.graph_statistics()['pool_reserved_bytes'] <= 8*1024**2
    assert graph.graph.copy_back_mutable is False
    assert graph.graph.bound_arrays() == (candidate.keys, candidate.values, candidate.raw_index,
                                         candidate.pooled, graph.validation_status)


@gpu
@cuda
def test_interleaved_request_recapture_and_external_event_can_retire(math_block):
    graph, a, rand, checks = math_block
    first, second = a.create_state(20), a.create_state(20)
    refs = a.create_state(20), a.create_state(20)
    old = None
    for index in (0, 1, 0):
        state = (first, second)[index]
        streams = rand((4, 256))
        expected = uncaptured(graph, streams, refs[index])
        actual = graph.forward(streams, state, state.length).copy()
        np.testing.assert_array_equal(actual.get(), expected.get())
        assert_cache(state, snapshot(refs[index]))
        if old is not None: assert old.closed
        old = graph.graph
        # A captured-only event can be invalid on later graph destruction.
        graph.plan.retired.synchronize()
    assert graph.captures == 3 and len(checks) == 3
    graph.close()
    assert graph.handoff.handle == 1  # Protected native ownership remains borrowed.


@gpu
@cuda
def test_nonfinite_validation_does_not_commit_any_cache(math_block):
    graph, a, rand, _ = math_block
    state = a.create_state(20)
    for name in CACHE_NAMES: getattr(state, name).fill(xp.float32(.375))
    before = snapshot(state)
    streams = rand((4, 256))
    streams.fill(xp.nan)
    graph.forward(streams, state, 0)
    assert int(graph.validation_status.get()) != 0
    assert state.length == 1  # Failed request is not resumable (original contract).
    assert_cache(state, before)


@gpu
@cuda
@pytest.mark.parametrize('bad', ['owner', 'position', 'capacity', 'width', 'layout', 'dtype', 'stream', 'fingerprint'])
def test_original_state_and_frozen_plan_contract(math_block, bad):
    graph, a, rand, _ = math_block
    state, streams = a.create_state(20), rand((4, 256))
    t = 0
    if bad == 'owner': state.owner = object()
    if bad == 'position': t = 1
    if bad == 'capacity': t = state.length = 20
    if bad == 'width': t = state.length = 16
    if bad == 'layout': state.keys = xp.zeros((20, 2, 512), xp.float32)[..., ::2]
    if bad == 'dtype': streams = streams.astype(xp.float64)
    if bad == 'fingerprint': graph.plan.fingerprint = ()
    before, length = snapshot(state), state.length
    with pytest.raises((ValueError, TypeError)):
        if bad == 'stream':
            with xp.cuda.Stream(non_blocking=True): graph.forward(streams, state, t)
        else: graph.forward(streams, state, t)
    assert state.length == length and graph.calls == 0
    assert_cache(state, before)


@gpu
@cuda
def test_capture_failure_only_touches_private_warmup_state(math_block, monkeypatch):
    graph, a, rand, checks = math_block
    state = a.create_state(20)
    before = snapshot(state)
    project, size = a.projections['output']
    def fail_in_capture(x):
        if xp.cuda.get_current_stream().is_capturing():
            raise RuntimeError('injected capture failure')
        return project(x)
    monkeypatch.setitem(a.projections, 'output', (fail_in_capture, size))
    with pytest.raises(RuntimeError, match='injected capture failure'):
        graph.forward(rand((4, 256)), state, 0)
    assert state.length == 0 and graph.graph is None and graph.calls == 0
    assert_cache(state, before)
    assert checks == ['warmup'] and graph.handoff.accounted[0] == 1
    graph.plan.retired.synchronize()


@gpu
@cuda
def test_serial_math_graph_capture_once_replays_runtime_n_across_tiles(monkeypatch):
    from mini_llm.ops.qsa_decode_gpu import qsa_decode_gpu
    monkeypatch.setenv('MINI_LLM_QSA_COOPERATIVE', '0')
    rng = np.random.default_rng(317)
    def array(shape): return xp.asarray(rng.normal(size=shape).astype(np.float32))
    with xp.cuda.Stream(non_blocking=True) as stream:
        q, qg, k, v = [array(s) for s in ((4, 256), (4, 512), (2, 256), (2, 256))]
        source_k, source_v = array((264, 2, 256)), array((264, 2, 256))
        keys, values = source_k.copy(), source_v.copy()
        position = xp.empty((), xp.int64)
        kernels.set_position_gpu(position, 0, 259)
        kernels.qsa_decode_graph_gpu(q, qg, k, v, keys, values, position, .0625)
        stream.synchronize()
        stream.begin_capture()
        out = kernels.qsa_decode_graph_gpu(q, qg, k, v, keys, values, position, .0625)
        captured = stream.end_capture()
        try:
            for n in (1, 2, 127, 128, 129, 131, 260, 3):
                xp.copyto(keys, source_k)
                xp.copyto(values, source_v)
                keys[n-1:], values[n-1:] = xp.nan, xp.nan
                kernels.set_position_gpu(position, n-1, 259)
                captured.launch(stream=stream)
                actual = out.copy()
                expected = qsa_decode_gpu(q, qg, k, v, keys, values, n, .0625)
                np.testing.assert_array_equal(actual.get().view(np.uint32), expected.get().view(np.uint32))
                assert np.isfinite(actual.get()).all()
        finally:
            stream.synchronize()
