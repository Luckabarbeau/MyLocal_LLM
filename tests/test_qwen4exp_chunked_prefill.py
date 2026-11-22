"""Focused CPU/CuPy synthetic parity; no full-checkpoint performance claims."""
from contextlib import nullcontext
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec, Qwen4ExpRecurrentMixerInference
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel, Qwen4ExpLayer, Qwen4ExpMoE
from mini_llm.external.qwen4exp_ple import PLEConfig, Qwen4ExpPLE
from mini_llm.external.qwen4exp_prefill import _hc_read, _project, _routed, prefill_chunked
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.router_inference import RouterInference

@pytest.fixture(autouse=True)
def selected_test_backend(request):
    gpu = request.node.get_closest_marker('gpu') is not None
    if gpu and BACKEND_NAME != 'cupy':
        pytest.skip('Actual CuPy required')
    if not gpu and BACKEND_NAME != 'numpy':
        pytest.skip('CPU synthetic test')


class TinyPLE(Qwen4ExpPLE):
    """Real PLE math/state with a small deterministic hash-table lookup fixture."""
    width = 8

    def __init__(self, matrix):
        self.key = matrix(32, 8)
        self.value = matrix(8, 8)
        self.gamma_key = self.gamma_query = self.gamma_conv = xp.ones((4, 8), xp.float32)
        self.conv = matrix(32, 4)
        self.epsilon = np.float32(1e-5)
        self._closed = False
        self.config = PLEConfig((17, 31, 43), (0,) * 16, (23,) * 16, 2, 23)
        self.table = matrix(23, 8)

    def embedding(self, token, predecessors):
        rows = self.config.rows(token, predecessors)
        return self.table[np.asarray(rows)].mean(axis=0, dtype=np.float32)


class RecordingExecutor:
    overlaps_cpu_gpu = False
    routing_profiler = None

    def __init__(self, matrices):
        self.matrices = matrices
        self.calls = []

    def dispatch_scope(self, layer, active):
        return nullcontext()

    def forward(self, layer, expert, hidden):
        self.calls.append((layer, expert, hidden.copy()))
        gate, up, down = self.matrices[expert]
        g = hidden @ gate
        return ((g / (1 + xp.exp(-g))) * (hidden @ up)) @ down


def make_model():
    rng = np.random.default_rng(839)

    def matrix(*shape):
        return xp.asarray(rng.normal(0, .12, shape), dtype=xp.float32)

    def hc(inject=True):
        return Qwen4ExpHyperConnection(xp.ones((4, 8), xp.float32), matrix(3, 32),
            matrix(32, 3), matrix(4, 32) if inject else None, epsilon=1e-5)

    blocks, executors, calls, routes = [], [], [], []

    def projection(weight, label):
        def apply(x):
            calls.append((label, x.shape))
            return x @ weight.T
        return apply

    for i in range(3):
        if i != 1:
            spec = Qwen4ExpMixerSpec(8, 1, 2, 2, 4, 1e-5)
            mixer = Qwen4ExpRecurrentMixerInference(spec,
                qkv=projection(matrix(8, 8), f'{i}.qkv'),
                z=projection(matrix(4, 8), f'{i}.z'),
                beta=matrix(2, 8), alpha=matrix(2, 8), out=matrix(8, 4),
                conv_weight=matrix(8, 4), norm_weight=xp.ones(2, xp.float32),
                ssm_a=-xp.ones(2, xp.float32), dt_bias=matrix(2))
        else:
            c = QSAConfig(8, n_q_heads=2, n_kv_heads=1, head_dim=4, rotary_dim=4,
                index_heads=2, index_dim=4, index_topk=2, block_size=3,
                sections=(1, 1, 0, 0), epsilon=1e-5)
            mixer = Qwen4ExpAttention(c, query_gate=matrix(16, 8), key=matrix(4, 8),
                value=matrix(4, 8), output=matrix(8, 8), index_query=matrix(8, 8),
                index_key=matrix(4, 8), gamma_q=xp.ones(4, xp.float32),
                gamma_k=xp.ones(4, xp.float32), gamma_index_q=xp.ones(4, xp.float32),
                gamma_index_k=xp.ones(4, xp.float32))
        router = RouterInference(8, 4, 2, renormalize_selected=True)
        router.W_router, router.b_router = matrix(8, 4), matrix(4)
        original = router.forward

        def record(x, original=original, layer=i):
            w, ids = original(x)
            routes.append((layer, w.copy(), ids.copy()))
            return w, ids

        router.forward = record
        executor = RecordingExecutor([(matrix(8, 6), matrix(8, 6), matrix(6, 8))
                                      for _ in range(4)])
        experts = ExpertsInference(8, 6, 4, expert_executor=executor, layer_id=i)
        moe = Qwen4ExpMoE(router, experts,
            projection(matrix(6, 8), f'{i}.shared_gate'),
            projection(matrix(6, 8), f'{i}.shared_up'),
            projection(matrix(8, 6), f'{i}.shared_down'), matrix(8), scale=.75)
        blocks.append(Qwen4ExpLayer(hc(), mixer, hc(), moe, i != 1, TinyPLE(matrix)))
        executors.append(executor)
    embeddings, head = matrix(11, 8), matrix(11, 8)
    model = Qwen4ExpGGUFModel.from_components(width=8, vocab=11, blocks=blocks,
        final_hc=hc(False), embedding=lambda token: embeddings[token],
        output=projection(head, 'head'), max_length=32)
    return model, executors, calls, routes


def assert_state(a, b):
    assert a.length == b.length
    assert not a.failed and not b.failed
    for i in a.recurrent:
        x, y = a.recurrent[i], b.recurrent[i]
        assert x.conv.length == y.conv.length == a.length
        assert x.delta.length == y.delta.length == a.length
        np.testing.assert_allclose(x.conv.history, y.conv.history, atol=3e-6, rtol=3e-5)
        np.testing.assert_allclose(x.delta.memory, y.delta.memory, atol=3e-6, rtol=3e-5)
    for i in a.qsa:
        x, y = a.qsa[i], b.qsa[i]
        assert x.length == y.length == a.length
        for name in ('keys', 'values', 'raw_index', 'pooled'):
            np.testing.assert_allclose(getattr(x, name), getattr(y, name), atol=3e-6, rtol=3e-5)
    for i in a.ple:
        assert a.ple[i].predecessors == b.ple[i].predecessors
        np.testing.assert_allclose(a.ple[i].normalized_history, b.ple[i].normalized_history,
                                   atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 4, 7, 32])
@pytest.mark.parametrize('prefix', [[], [4, 2, 6]])
def test_equivalence_final_state_decode_boundaries_and_routing(chunk_size, prefix):
    model, executors, calls, routes = make_model()
    sequential = model.create_generation_state(max_length=32)
    chunked = model.create_generation_state(max_length=32)
    for token in prefix:
        model.step(token, sequential)
        model.step(token, chunked)
    tokens = [1, 2, 3, 8, 1, 5, 2, 7, 9, 0, 6]
    routes.clear()
    for token in tokens:
        expected = model.step(token, sequential)
    expected_routes = {i: (np.concatenate([w for j, w, _ in routes if j == i], axis=1),
                           np.concatenate([ids for j, _, ids in routes if j == i], axis=1))
                       for i in range(3)}
    calls.clear()
    routes.clear()
    for executor in executors:
        executor.calls.clear()
    actual = prefill_chunked(model, np.asarray([tokens]), chunked, chunk_size)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    assert_state(sequential, chunked)
    assert sum(label == 'head' for label, _ in calls) == 1
    for i in range(3):
        weights = np.concatenate([w for j, w, _ in routes if j == i], axis=1)
        ids = np.concatenate([ids for j, _, ids in routes if j == i], axis=1)
        np.testing.assert_allclose(weights, expected_routes[i][0], atol=3e-6, rtol=3e-5)
        np.testing.assert_array_equal(ids, expected_routes[i][1])
    if chunk_size > 1:
        assert any(label == '0.qkv' and shape[1] > 1 for label, shape in calls)
        assert any(label == '0.shared_gate' and shape[0] > 1 for label, shape in calls)
        assert any(x.shape[0] > 1 for e in executors for _, _, x in e.calls)
    for token in [3, 2, 10]:
        np.testing.assert_allclose(model.decode_one([token], chunked),
                                   model.decode_one([token], sequential), atol=3e-6, rtol=3e-5)
        assert_state(sequential, chunked)


@pytest.mark.parametrize('tokens,size', [([], 2), ([1.5], 2), ([-1], 2), ([11], 2),
                                       ([1], 0), ([1], True), ([1], 1.5)])
def test_invalid_input_does_not_mutate(tokens, size):
    model, _, _, _ = make_model()
    state = model.create_generation_state(max_length=8)
    with pytest.raises((ValueError, TypeError)):
        prefill_chunked(model, tokens, state, size)
    assert state.length == 0 and not state.failed
    assert all(s.conv.length == 0 for s in state.recurrent.values())


def test_capacity_owner_and_model_lock():
    model, _, _, _ = make_model()
    state = model.create_generation_state(max_length=2)
    with pytest.raises(ValueError, match='capacity'):
        prefill_chunked(model, [1, 2, 3], state, 2)
    other, _, _, _ = make_model()
    with pytest.raises(ValueError, match='owner'):
        prefill_chunked(other, [1], state, 1)
    with model._call_lock:
        with pytest.raises(RuntimeError, match='serially'):
            prefill_chunked(model, [1], state, 1)
    assert not state.failed and state.length == 0
    prefill_chunked(model, [1, 2], state, 2)


def test_partial_failure_poison_releases_lock_and_no_retry():
    model, executors, _, _ = make_model()
    state = model.create_generation_state(max_length=8)

    def fail(x):
        raise RuntimeError('injected shared failure')

    model.blocks[0].moe.gate = fail
    with pytest.raises(RuntimeError, match='injected'):
        prefill_chunked(model, [1, 2, 3], state, 2)
    assert state.failed and state.length == 0
    assert state.recurrent[0].length == 2
    assert executors[0].calls
    assert not model._call_lock.locked()
    with pytest.raises(RuntimeError, match='discard'):
        model.decode_one([1], state)


def test_packed_bf16_hc_is_bit_expanded_not_numeric():
    model, _, _, _ = make_model()
    hc = model.blocks[0].attention_hc
    packed = SimpleNamespace(width=hc.width, gamma=hc.gamma, epsilon=hc.epsilon,
                             packed_bf16=True)
    decoded = {}
    for name in ('down', 'up', 'inject'):
        bits = (getattr(hc, name).view(np.uint32) >> 16).astype(np.uint16)
        setattr(packed, name, bits)
        decoded[name] = (bits.astype(np.uint32) << 16).view(np.float32)
    reference = Qwen4ExpHyperConnection(hc.gamma, **decoded, epsilon=float(hc.epsilon))
    streams = np.random.default_rng(9).normal(size=(5, 4, 8)).astype(np.float32)
    for actual, expected in zip(_hc_read(packed, streams), reference.read(streams)):
        np.testing.assert_array_equal(actual, expected)


def test_vector_wrapper_bypassed_for_batched_packed_projection(monkeypatch):
    from mini_llm.ops import quantized_linear as module
    weight = SimpleNamespace(shape=(3, 8))

    def callback(x):
        raise AssertionError('packed callback must not receive chunk')
    callback.packed_device_weight = weight
    calls = []

    def linear(x, w):
        assert x.shape == (5, 8) and w is weight
        calls.append(x.copy())
        return x[:, :3].copy()

    monkeypatch.setattr(module, 'quantized_linear', linear)
    model = SimpleNamespace(weights=None)
    hidden = np.arange(40, dtype=np.float32).reshape(1, 5, 8)
    np.testing.assert_array_equal(_project(model, callback, 'unused', hidden), hidden[..., :3])
    assert len(calls) == 1


def test_gpu_device_plan_policy_preserves_routes_and_borrowed_outputs(monkeypatch):
    from mini_llm.external import qwen4exp_prefill as module
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    # Policy-only test: no CuPy import, allocation, or GPU execution. Device
    # plans return reused workspaces, so each token must be copied before reuse.
    executor = object.__new__(HeterogeneousExpertExecutor)
    executor.device_expert_plan = True
    calls = []
    borrowed = np.empty((1, 1, 4), np.float32)

    def forward(x, weights, ids):
        assert x.shape == (1, 1, 4)
        calls.append((x.copy(), weights.copy(), ids.copy()))
        borrowed[:] = x * weights.sum(axis=-1, keepdims=True)
        return borrowed

    experts = SimpleNamespace(expert_executor=executor, forward=forward)
    x = np.arange(20, dtype=np.float32).reshape(1, 5, 4)
    weights = np.arange(10, dtype=np.float32).reshape(1, 5, 2) / 10
    ids = np.array([[[3, 1], [0, 2], [1, 0], [2, 3], [3, 2]]])
    monkeypatch.setattr(module, 'BACKEND_NAME', 'cupy')
    actual = _routed(experts, x, weights, ids)
    np.testing.assert_array_equal(actual, x * weights.sum(axis=-1, keepdims=True))
    np.testing.assert_array_equal(np.concatenate([w for _, w, _ in calls], axis=1), weights)
    np.testing.assert_array_equal(np.concatenate([i for _, _, i in calls], axis=1), ids)
    assert len(calls) == 5
    executor.device_expert_plan = False
    experts.forward = lambda a, w, i: a * w.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(_routed(experts, x, weights, ids), actual)


def test_real_heterogeneous_async_executor_prefill_and_decode():
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    model, records, _, _ = make_model()
    managed = []
    batch_shapes = []
    try:
        for layer, record in enumerate(records):
            executor = HeterogeneousExpertExecutor(
                lambda key, matrices=record.matrices: matrices[key.expert_id],
                model_id='chunk-test', d_model=8, d_ff=6, cpu_workers=2,
                device_budget_bytes=0, compute_dtype='float32')
            managed.append(executor)
            submit = executor.submit

            def track(layer, expert, x, submit=submit):
                batch_shapes.append(x.shape)
                return submit(layer, expert, x)

            executor.submit = track
            model.blocks[layer].moe.experts.expert_executor = executor
        a = model.create_generation_state(max_length=32)
        b = model.create_generation_state(max_length=32)
        tokens = [1, 2, 3, 8, 1, 5, 2, 7, 9, 0, 6]
        expected = model.prefill(tokens, a)
        batch_shapes.clear()
        actual = prefill_chunked(model, tokens, b, 4)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        assert_state(a, b)
        assert any(shape[0] > 1 for shape in batch_shapes)
        np.testing.assert_allclose(model.decode_one([4], b), model.decode_one([4], a),
                                   atol=3e-6, rtol=3e-5)
        assert_state(a, b)
    finally:
        for executor in managed:
            executor.close()


@pytest.mark.parametrize('backend', ['native', 'ggml_pool'])
def test_packed_heterogeneous_grouping_without_dequantization(monkeypatch, backend):
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
    if backend == 'ggml_pool' and not os.environ.get('MINI_LLM_GGML_CPU_LIBRARY'):
        pytest.skip('GGML requires an explicitly selected CPU library')
    rng = np.random.default_rng(2249)

    def packed(kind):
        qk, qb = BLOCK_FORMATS[kind]
        raw = rng.integers(0, 256, (256 * 256 // qk, qb), dtype=np.uint8)
        raw[:, :2] = np.frombuffer(np.float16(1 / 1024).tobytes(), np.uint8)
        raw.flags.writeable = False
        return PackedGGUFWeight(raw, (256, 256), kind)

    matrices = [(packed('IQ3_S'), packed('IQ3_S'), packed('Q2_0')) for _ in range(3)]

    def forbidden(*args, **kwargs):
        pytest.fail('Packed prefill must not dequantize a full expert matrix')

    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', forbidden)
    executor = HeterogeneousExpertExecutor(lambda key: matrices[key.expert_id],
        model_id='packed-chunk', d_model=256, d_ff=256, cpu_workers=2,
        device_budget_bytes=0, host_dequant_budget_bytes=0, cpu_packed_backend=backend)
    experts = ExpertsInference(256, 256, 3, expert_executor=executor)
    x = rng.normal(0, .1, (1, 5, 256)).astype(np.float32)
    weights = np.array([[[.3, .7], [.6, .4], [.2, .8], [.4, .6], [.9, .1]]], np.float32)
    ids = np.array([[[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]]])
    shapes = []
    original = executor._cpu_forward_numpy

    def record(key, ws, hidden):
        shapes.append(hidden.shape)
        return original(key, ws, hidden)

    executor._cpu_forward_numpy = record
    try:
        expected = np.concatenate([experts.forward(x[:, t:t+1], weights[:, t:t+1], ids[:, t:t+1])
                                   for t in range(5)], axis=1)
        shapes.clear()
        actual = _routed(experts, x, weights, ids)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        assert len(shapes) == 3 and all(shape[0] > 1 for shape in shapes)
        assert executor.host_cache.used_bytes == 0
    finally:
        executor.close()


def test_diagnostics_still_final_only_head():
    model, _, calls, _ = make_model()
    events = []
    model.diagnostic_callback = lambda i, stage, value: events.append((i, stage, value))
    state = model.create_generation_state(max_length=8)
    prefill_chunked(model, [1, 2, 3, 4, 5], state, 3)
    assert sum(label == 'head' for label, _ in calls) == 1
    assert sum(stage == 'post_moe' for _, stage, _ in events) == 6


def gpu_close(actual, expected, *, atol=3e-6, rtol=3e-5):
    a, b = asnumpy(actual), asnumpy(expected)
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print('max_abs=', float(difference.max(initial=0)),
          'relative_l2=', float(np.linalg.norm(difference) /
                                max(np.linalg.norm(b.astype(np.float64)), 1e-30)),
          'dtype=', a.dtype, 'atol=', atol, 'rtol=', rtol)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def gpu_packed(kind, n, k, seed):
    from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
    qk, qb = BLOCK_FORMATS[kind]
    raw = np.random.default_rng(seed).integers(0, 256, (n * k // qk, qb), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1 / 1024).tobytes(), np.uint8)
    raw.flags.writeable = False
    return PackedGGUFWeight(raw, (n, k), kind)


@pytest.mark.gpu
@pytest.mark.parametrize('inject', [False, True])
@pytest.mark.parametrize('rows', [1, 5])
def test_cupy_packed_bf16_hc_batch_vs_native_singleton(inject, rows):
    rng = np.random.default_rng(3349)
    width, rank = 257, 33

    def packed(shape):
        value = rng.normal(0, .04, shape).astype(np.float32)
        return xp.asarray((value.view(np.uint32) >> 16).astype(np.uint16))

    hc = Qwen4ExpHyperConnection(xp.ones((4, width), xp.float32),
        packed((rank, 4 * width)), packed((4 * width, rank)),
        packed((4, 4 * width)) if inject else None, epsilon=1e-5,
        fast_decode=True, native_read=True, packed_bf16=True)
    streams = xp.asarray(rng.normal(size=(rows, 4, width)), dtype=xp.float32)
    expected = [hc.read(row) for row in streams]
    hidden, gains = _hc_read(hc, streams)
    gpu_close(hidden, xp.stack([h for h, _ in expected]), atol=0, rtol=0)
    if inject:
        gpu_close(gains, xp.stack([g for _, g in expected]), atol=0, rtol=0)
        gpu_close(hc.write(streams, hidden, gains), xp.stack([
            hc.write(row, h, g) for row, (h, g) in zip(streams, expected)]))
    else:
        assert gains is None


@pytest.mark.gpu
@pytest.mark.parametrize('rows', [16, 32])
def test_cupy_native_recurrence_with_batched_packed_projections(monkeypatch, rows):
    from mini_llm.external.qwen4exp_prefill import _recurrent
    from mini_llm.ops.inference_matmul import matmul_inference
    from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', '1')
    rng = np.random.default_rng(3510)

    def packed_projection(kind, output, seed):
        device = packed_to_device(gpu_packed(kind, output, 256, seed))

        def project(x):
            return quantized_linear(xp.ascontiguousarray(x), device)

        project.packed_device_weight = device
        return project

    def dense_coefficients():
        weight = xp.asarray(rng.normal(0, .1, (256, 2)), dtype=xp.float32)
        return lambda x: matmul_inference(x, weight)

    spec = Qwen4ExpMixerSpec(256, 1, 2, 128, 4, 1e-5)
    mixer = Qwen4ExpRecurrentMixerInference(spec,
        qkv=packed_projection('IQ3_S', 512, 3511),
        z=packed_projection('IQ3_S', 256, 3512),
        beta=dense_coefficients(), alpha=dense_coefficients(),
        out=packed_projection('IQ4_NL', 256, 3513),
        conv_weight=xp.asarray(rng.normal(0, .1, (512, 4)), dtype=xp.float32),
        norm_weight=xp.ones(128, xp.float32), ssm_a=-xp.ones(2, xp.float32),
        dt_bias=xp.zeros(2, xp.float32), fast_coefficients=True)
    mixer.mixer._decode_fastpath_enabled = True
    hidden = xp.asarray(rng.normal(0, .1, (rows, 256)), dtype=xp.float32)
    a, b = mixer.create_state(1), mixer.create_state(1)
    original_projections, original_core = dict(mixer.projections), mixer.mixer
    expected = xp.stack([mixer.forward(row[None, None], a)[0, 0] for row in hidden])
    actual = _recurrent(SimpleNamespace(weights=None), 0, mixer, hidden, b)
    gpu_close(actual, expected, atol=0, rtol=0)
    gpu_close(b.conv.history, a.conv.history, atol=0, rtol=0)
    gpu_close(b.delta.memory, a.delta.memory, atol=0, rtol=0)
    assert b.length == a.length == rows
    assert mixer.mixer is original_core and mixer.projections == original_projections


@pytest.mark.gpu
@pytest.mark.parametrize('kind', ['IQ3_S', 'IQ4_NL', 'Q2_0'])
@pytest.mark.parametrize('q8', ['0', '1'])
def test_cupy_packed_multirow_projection_preserves_activation_policy(monkeypatch, kind, q8):
    from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', q8)
    monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID', '1')
    device = packed_to_device(gpu_packed(kind, 37, 256, 3350))
    hidden = xp.asarray(np.random.default_rng(3350).normal(size=(1, 5, 256)), dtype=xp.float32)

    def vector_wrapper(x):
        assert x.ndim == 1
        return quantized_linear(x[None], device)[0]

    # Exercise canonical model handle lookup, not callback closure inspection.
    model = SimpleNamespace(weights=SimpleNamespace(device={'test.weight': device}))
    expected = xp.stack([vector_wrapper(row) for row in hidden[0]])[None]
    actual = _project(model, vector_wrapper, 'test.weight', hidden)
    gpu_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('q8,service', [('0', '0'), ('1', '0'), ('1', '1')])
def test_cupy_real_device_plan_routing_singleton(monkeypatch, q8, service):
    from pathlib import Path
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    library = os.environ.get('MINI_LLM_GGML_CPU_LIBRARY')
    if not library or not Path(library).is_file():
        pytest.skip('Explicit pinned GGML CPU library required')
    for name, value in {
        'MINI_LLM_EXPERT_GPU_Q8': q8,
        'MINI_LLM_IQ_GPU_CODEBOOK_STORAGE': 'global',
        'MINI_LLM_IQ_GPU_GROUPED_KERNEL': 'warp',
        'MINI_LLM_Q8_SHARED_GRID': '1',
        'MINI_LLM_NATIVE_MOE_HANDOFF': service,
        'MINI_LLM_NATIVE_MOE_GRAPH': service,
        'MINI_LLM_NATIVE_MOE_SERVICE': service,
        'MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH': service,
        'MINI_LLM_NATIVE_MOE_STAGE_LIMIT': '0',
        'MINI_LLM_GGML_POOL_SCHEDULER': 'tasks',
        'MINI_LLM_GGML_POOL_HOST_HELP': '1',
    }.items():
        monkeypatch.setenv(name, value)
    sources = [(gpu_packed('IQ3_S', 256, 256, 3360 + e),
                gpu_packed('IQ3_S', 256, 256, 3370 + e),
                gpu_packed('Q2_0', 256, 256, 3380 + e)) for e in range(4)]
    executor = HeterogeneousExpertExecutor(lambda key: sources[key.expert_id],
        model_id='chunk-gpu-plan', d_model=256, d_ff=256, compute_dtype='float32',
        device_budget_bytes=2 * 1024**2, cpu_workers=2, cpu_packed_backend='ggml_pool',
        promote_after=1, device_expert_plan=True)
    experts = ExpertsInference(256, 256, 4, expert_executor=executor)
    hidden = xp.asarray(np.random.default_rng(3351).normal(0, .1, (1, 5, 256)), dtype=xp.float32)
    ids = xp.asarray([[[2, 0], [1, 3], [0, 1], [3, 2], [1, 2]]], dtype=xp.int64)
    routes = xp.asarray([[[.3, .7], [.6, .4], [.2, .8], [.4, .6], [.9, .1]]], dtype=xp.float32)
    try:
        # Two resident GPU experts and two GGML CPU misses; freeze admission as
        # the production device-plan contract requires.
        for e in (0, 1):
            executor._prepare(0, e, hidden.reshape(5, 256)[:1])
        assert len(executor.cache.resident) == 2
        executor.cache.set_admission_budget(0)
        expected = xp.concatenate([experts.forward(hidden[:, t:t+1], routes[:, t:t+1], ids[:, t:t+1]).copy()
                                   for t in range(5)], axis=1)
        plan = executor._device_expert_plans[0]
        assert (plan.q8 is not None) == (q8 == '1')
        plan.flush_statistics()
        before = executor.counters['device_plan_dispatches']
        actual = _routed(experts, hidden, routes, ids)
        executor.check_native_service()
        xp.cuda.get_current_stream().synchronize()
        plan.flush_statistics()
        assert executor.counters['device_plan_dispatches'] - before == 5
        gpu_close(actual, expected, atol=0, rtol=0)
        saved = asnumpy(actual).copy()
        experts.forward(hidden[:, :1] * xp.float32(2), routes[:, :1], ids[:, :1])
        executor.check_native_service()
        np.testing.assert_array_equal(asnumpy(actual), saved)
        assert executor.host_cache.used_bytes == 0
        if service == '1':
            assert plan.native_handoff.service is not None
    finally:
        executor.close()


@pytest.mark.gpu
def test_cupy_mixed_grouped_multirow_experts(monkeypatch):
    from pathlib import Path
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    library = os.environ.get('MINI_LLM_GGML_CPU_LIBRARY')
    if not library or not Path(library).is_file():
        pytest.skip('Explicit pinned GGML CPU library required')
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8', '1')
    monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID', '1')
    sources = [(gpu_packed('IQ3_S', 256, 256, 3400 + e),
                gpu_packed('IQ3_S', 256, 256, 3410 + e),
                gpu_packed('Q2_0', 256, 256, 3420 + e)) for e in range(4)]
    executor = HeterogeneousExpertExecutor(lambda key: sources[key.expert_id],
        model_id='chunk-gpu-grouped', d_model=256, d_ff=256, compute_dtype='float32',
        device_budget_bytes=2 * 1024**2, cpu_workers=2, cpu_packed_backend='ggml_pool',
        promote_after=1, device_expert_plan=False)
    experts = ExpertsInference(256, 256, 4, expert_executor=executor)
    hidden = xp.asarray(np.random.default_rng(3400).normal(0, .1, (1, 5, 256)), dtype=xp.float32)
    ids_host = np.array([[[2, 0], [1, 3], [0, 1], [3, 2], [1, 2]]])
    ids = xp.asarray(ids_host, dtype=xp.int64)
    routes = xp.asarray([[[.3, .7], [.6, .4], [.2, .8], [.4, .6], [.9, .1]]], dtype=xp.float32)
    try:
        for e in (0, 1):
            executor._prepare(0, e, hidden.reshape(5, 256)[:1])
        assert len(executor.cache.resident) == 2
        executor.cache.set_admission_budget(0)
        expected = xp.zeros_like(hidden)
        for t in range(5):
            for slot in np.argsort(ids_host[0, t]):
                e = int(ids_host[0, t, slot])
                term = executor.forward(0, e, hidden[0, t:t+1])
                expected[0, t:t+1] += routes[0, t, slot] * term
        submitted = []
        submit = executor.submit

        def record(layer, expert, x):
            submitted.append((expert, x.shape))
            return submit(layer, expert, x)

        executor.submit = record
        actual = _routed(experts, hidden, routes, ids)
        gpu_close(actual, expected, atol=0, rtol=0)
        assert [e for e, _ in submitted] == [0, 1, 2, 3]
        assert all(shape[0] > 1 for _, shape in submitted)
        assert executor.host_cache.used_bytes == 0
    finally:
        executor.close()


@pytest.mark.gpu
@pytest.mark.parametrize('chunk_size', [2, 4])
def test_cupy_tiny_chunked_graph_state_routing_and_decode(monkeypatch, chunk_size):
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_SELECT', '1')
    monkeypatch.setenv('MINI_LLM_ROUTER_WARP_REDUCE', '1')
    model, _, calls, routes = make_model()
    sequential = model.create_generation_state(max_length=32)
    chunked = model.create_generation_state(max_length=32)
    for token in (4, 2, 6):
        model.step(token, sequential)
        model.step(token, chunked)
    tokens = [1, 2, 3, 8, 1, 5, 2, 7, 9, 0, 6]
    routes.clear()
    for token in tokens:
        expected = model.step(token, sequential)
    wanted = {i: (xp.concatenate([w for j, w, _ in routes if j == i], axis=1),
                  xp.concatenate([ids for j, _, ids in routes if j == i], axis=1)) for i in range(3)}
    routes.clear()
    calls.clear()
    actual = prefill_chunked(model, tokens, chunked, chunk_size)
    gpu_close(actual, expected)
    assert sum(label == 'head' for label, _ in calls) == 1
    for i in range(3):
        np.testing.assert_array_equal(asnumpy(xp.concatenate([ids for j, _, ids in routes if j == i], axis=1)),
                                      asnumpy(wanted[i][1]))
        gpu_close(xp.concatenate([w for j, w, _ in routes if j == i], axis=1), wanted[i][0])

    def compare_state():
        assert sequential.length == chunked.length
        assert not sequential.failed and not chunked.failed
        for i in sequential.recurrent:
            a, b = sequential.recurrent[i], chunked.recurrent[i]
            assert a.length == b.length == sequential.length
            gpu_close(a.conv.history, b.conv.history)
            gpu_close(a.delta.memory, b.delta.memory)
        for i in sequential.qsa:
            a, b = sequential.qsa[i], chunked.qsa[i]
            assert a.length == b.length == sequential.length
            for name in ('keys', 'values', 'raw_index', 'pooled'):
                gpu_close(getattr(a, name), getattr(b, name))
        for i in sequential.ple:
            a, b = sequential.ple[i], chunked.ple[i]
            assert a.predecessors == b.predecessors
            gpu_close(a.normalized_history, b.normalized_history)

    compare_state()
    for token in (3, 2):
        gpu_close(model.decode_one([token], chunked), model.decode_one([token], sequential))
        compare_state()
