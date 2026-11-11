"""Shared FFN integration only; CUDA cases require MINI_LLM_BACKEND=cupy."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external import gguf_projection as projection
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.external.qwen3next_moe import Qwen3NextMoEInference
from mini_llm.ops import quantized_linear as ql
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from test_0106c_fused_packed_expert import make_quant


SHARED = ('ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'ffn_down_shexp.weight')


def _packed(fmt, n, k, seed):
    weight = make_quant(fmt, n, k, seed)
    raw = weight.data.reshape(-1, BLOCK_FORMATS[fmt][1])
    offset = {'Q3_K': 108, 'Q6_K': 208}.get(fmt, 0)
    raw[:, offset:offset + 2] = np.frombuffer(np.float16(1 / 1024).tobytes(), np.uint8)
    if fmt in ('Q4_K', 'Q5_K'):
        raw[:, 2:4] = np.frombuffer(np.float16(1 / 2048).tobytes(), np.uint8)
    return weight


class _Tensor:
    def __init__(self, name, shape, data, reads, kind='F32'):
        self.name, self.shape = name, tuple(reversed(shape))
        self.tensor_type = SimpleNamespace(name=kind)
        self._data, self.reads = data, reads

    @property
    def data(self):
        self.reads.append(self.name)
        return self._data


def _pair(kinds):
    reads, tensors, dense = [], {}, {}
    for i, kind in enumerate(kinds):
        name = ('gate', 'up')[i]
        if kind in BLOCK_FORMATS:
            weight = _packed(kind, 7, 256, 10620 + i)
            raw, value = weight.data, weight.dequantize_reference()
        else:
            raw = np.random.default_rng(i).normal(0, .03, (7, 256)).astype(
                            {'F32': np.float32, 'F16': np.float16}[kind])
            value = raw.astype(np.float32)
        tensors[name] = _Tensor(name, (7, 256), raw, reads, kind)
        dense[name] = value

    def non_expert_weight(name):
        assert tensors[name].tensor_type.name not in BLOCK_FORMATS
        return tensors[name].data

    return SimpleNamespace(tensors=tensors, non_expert_weight=non_expert_weight), dense, reads


@pytest.fixture
def cpu_projection(monkeypatch):
    """CPU-only dispatch/injection tests, even in a CuPy pytest process."""
    monkeypatch.setattr(projection, 'BACKEND_NAME', 'numpy')
    monkeypatch.setattr(projection, 'xp', np)
    monkeypatch.setattr(projection, 'silu', lambda x: x * (1.0 / (1.0 + np.exp(-x))))
    return monkeypatch


def _simulate_device(monkeypatch):
    # Real DevicePacked validation over CPU bytes; no CUDA context or kernel.
    monkeypatch.setattr(projection, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(ql, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(ql, 'xp', np)

    def upload(weight):
        return ql.DevicePackedGGUFWeight(weight.data.copy(), weight.shape, weight.quant_type)

    return upload


def _forbidden(*args, **kwargs):
    raise AssertionError('unexpected fused call, upload, decode, or fallback')


@pytest.mark.parametrize('kind', ['Q8_0', 'F16'])
def test_resolve_is_lazy_cached_and_does_not_execute(cpu_projection, kind):
    index, _, reads = _pair((kind, 'F32'))
    host, device, key = {}, {}, ('adapter-owned', 3)
    cpu_projection.setattr(projection, 'quantized_linear', _forbidden)
    cpu_projection.setattr(projection, 'packed_to_device', _forbidden)
    assert reads == []
    weight = projection.resolve_gguf_weight(index, 'gate', host, device, key)
    assert projection.resolve_gguf_weight(index, 'gate', host, device, key) is weight
    assert list(host) == [key] and device == {} and reads == ['gate']
    if kind in BLOCK_FORMATS:
        assert isinstance(weight, PackedGGUFWeight)
        assert np.shares_memory(weight.data, index.tensors['gate']._data)
    else:
        assert weight.dtype == np.float32
        weight.flags.writeable = False
        assert index.tensors['gate']._data.flags.writeable


@pytest.mark.parametrize('opaque_upload', [False, True])
def test_existing_linear_api_retains_injected_upload_linear_and_cache_keys(cpu_projection, opaque_upload):
    index, dense, reads = _pair(('Q8_0', 'Q4_0'))
    upload_device = _simulate_device(cpu_projection)
    uploads, calls, host, device = [], [], {}, {}
    key = ('mixer', 'gate')

    def upload(weight):
        uploads.append(weight)
        return object() if opaque_upload else upload_device(weight)

    def linear(hidden, weight, *, cpu_packed_backend):
        calls.append((weight, cpu_packed_backend))
        assert hidden.flags.c_contiguous
        return hidden @ dense['gate'].T

    cpu_projection.setattr(projection, 'packed_to_device', _forbidden)
    cpu_projection.setattr(projection, 'quantized_linear', _forbidden)
    cpu_projection.setattr(PackedGGUFWeight, 'dequantize_reference', _forbidden)
    hidden = np.ones((1, 4, 256), np.float32)[:, ::2]
    for _ in range(2):
        actual = projection.project_gguf_linear(
            hidden, index, 'gate', host, device, key,
            cpu_packed_backend='native', linear=linear, upload=upload)
        np.testing.assert_array_equal(actual, hidden @ dense['gate'].T)
    assert reads == ['gate'] and len(uploads) == 1
    assert uploads[0] is host[key] and list(device) == [key]
    assert all(w is device[key] and backend == 'native' for w, backend in calls)


@pytest.mark.parametrize('kinds', [('F32', 'F16'), ('Q8_0', 'F32'),
                                  ('F16', 'Q4_0'), ('Q8_0', 'Q4_0')])
@pytest.mark.parametrize('simulated_device,dtype', [(False, 'float32'), (True, 'float16'),
                                                   (True, 'float64')])
def test_fallback_exact_graph_and_injection(cpu_projection, kinds, simulated_device, dtype):
    index, dense, reads = _pair(kinds)
    host, device, calls, uploads = {}, {}, [], []
    upload_device = _simulate_device(cpu_projection) if simulated_device else _forbidden

    def upload(weight):
        uploads.append(weight)
        return upload_device(weight)

    def linear(hidden, weight, *, cpu_packed_backend):
        name = next(name for name, value in host.items()
                    if value is weight or device.get(name) is weight)
        assert hidden.flags.c_contiguous and cpu_packed_backend == 'native'
        calls.append(name)
        # Match the existing packed-linear FP32 accumulation/output contract.
        return (hidden.astype(np.float32) @ dense[name].T).astype(hidden.dtype)

    cpu_projection.setattr(projection, 'quantized_swiglu', _forbidden)
    hidden = np.random.default_rng(4).normal(0, .1, (1, 4, 256)).astype(dtype)[:, ::2]
    values = []
    for name, kind in zip(('gate', 'up'), kinds):
        values.append((hidden.astype(np.float32) @ dense[name].T).astype(hidden.dtype)
                      if kind in BLOCK_FORMATS else hidden @ dense[name].T)
    gate, up = values
    expected = (gate * (1.0 / (1.0 + np.exp(-gate)))) * up
    for _ in range(2):
        actual = projection.project_gguf_swiglu(
            hidden, index, 'gate', 'up', host, device, 'gate', 'up',
            cpu_packed_backend='native', linear=linear, upload=upload)
        np.testing.assert_array_equal(actual, expected)
    packed_names = [name for name, kind in zip(('gate', 'up'), kinds) if kind in BLOCK_FORMATS]
    assert calls == packed_names * 2 and reads == ['gate', 'up']
    assert len(uploads) == (len(packed_names) if simulated_device else 0)
    assert set(device) == (set(packed_names) if simulated_device else set())


def test_fused_dispatch_mixed_quants_contiguous_and_upload_once(cpu_projection):
    index, _, reads = _pair(('Q8_0', 'Q4_0'))
    upload_device = _simulate_device(cpu_projection)
    host, device, uploads, calls = {}, {}, [], []
    keys = (('shared', 'gate'), ('shared', 'up'))

    def upload(weight):
        uploads.append(weight)
        return upload_device(weight)

    def swiglu(hidden, gate, up):
        assert hidden.flags.c_contiguous and hidden.dtype == np.float32
        assert gate is device[keys[0]] and up is device[keys[1]]
        assert (gate.quant_type, up.quant_type) == ('Q8_0', 'Q4_0')
        calls.append(hidden)
        return np.zeros(hidden.shape[:-1] + (7,), np.float32)

    cpu_projection.setattr(projection, 'quantized_swiglu', swiglu)
    cpu_projection.setattr(PackedGGUFWeight, 'dequantize_reference', _forbidden)
    hidden = np.ones((1, 4, 256), np.float32)[:, ::2]
    for _ in range(2):
        actual = projection.project_gguf_swiglu(
            hidden, index, 'gate', 'up', host, device, *keys,
            linear=_forbidden, upload=upload)
        assert actual.shape == (1, 2, 7)
    assert reads == ['gate', 'up'] and len(uploads) == len(calls) == 2
    assert list(host) == list(device) == list(keys)


@pytest.mark.parametrize('error', [ValueError('gate/up shape mismatch'), RuntimeError('kernel failed')])
def test_fused_errors_propagate_without_retry(cpu_projection, error):
    index, _, _ = _pair(('Q8_0', 'Q4_0'))
    upload = _simulate_device(cpu_projection)

    def fail(*args):
        raise error

    cpu_projection.setattr(projection, 'quantized_swiglu', fail)
    with pytest.raises(type(error), match=str(error)) as caught:
        projection.project_gguf_swiglu(
            np.ones((1, 256), np.float32), index, 'gate', 'up', {}, {}, 'g', 'u',
            upload=upload, linear=_forbidden)
    assert caught.value is error


def test_host_packed_on_cupy_backend_still_falls_back(cpu_projection):
    index, dense, _ = _pair(('Q8_0', 'Q4_0'))
    cpu_projection.setattr(projection, 'BACKEND_NAME', 'cupy')
    cpu_projection.setattr(projection, 'quantized_swiglu', _forbidden)
    hidden = np.ones((1, 256), np.float32)
    calls = []

    def linear(x, w, *, cpu_packed_backend):
        assert isinstance(w, PackedGGUFWeight)
        calls.append(w.quant_type)
        return x @ dense['gate' if w.quant_type == 'Q8_0' else 'up'].T

    projection.project_gguf_swiglu(
        hidden, index, 'gate', 'up', {}, {}, 'g', 'u',
        upload=lambda weight: weight, linear=linear)
    assert calls == ['Q8_0', 'Q4_0']


def _moe_fixture(kinds):
    """Real GGUF index/provider, real routed executor, observable lazy shared reads."""
    rng = np.random.default_rng(10621)
    d = f = 256
    reads, tensors, weights = [], [], {}

    def add(suffix, shape, kind, seed):
        if kind in BLOCK_FORMATS:
            weight = _packed(kind, int(np.prod(shape[:-1])), shape[-1], seed)
            raw, value = weight.data, weight.dequantize_reference().reshape(shape)
        else:
            raw = rng.normal(0, .03, shape).astype(np.float32)
            value = raw.copy()
        tensors.append(_Tensor('blk.0.' + suffix, shape, raw, reads, kind))
        weights[suffix] = value.astype(np.float64)

    add('ffn_gate_inp.weight', (3, d), 'F32', 0)
    add('ffn_gate_inp_shexp.weight', (d,), 'F32', 0)
    for i, (suffix, kind) in enumerate(zip(SHARED, kinds)):
        add(suffix, (f, d) if i < 2 else (d, f), kind, 10622 + i)
    for i, suffix in enumerate(('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')):
        add(suffix, (3, f, d), 'Q8_0', 10630 + i)
    fields = {
        'general.architecture': 'qwen3next', 'qwen3next.block_count': 1,
        'qwen3next.embedding_length': d, 'qwen3next.expert_count': 3,
        'qwen3next.expert_used_count': 2,
        'qwen3next.expert_shared_feed_forward_length': f,
    }
    reader = SimpleNamespace(tensors=tensors, fields={
        name: SimpleNamespace(contents=lambda _, value=value: value) for name, value in fields.items()})
    index = Qwen3NextGGUFIndex(reader)
    assert reads == []
    return index, weights, reads


def _reference_moe(hidden, weights):
    """Independent FP64 routing, per-expert FFN, shared FFN and admission gate."""
    x = hidden.astype(np.float64)
    logits = x @ weights['ffn_gate_inp.weight'].T
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    ids = np.argsort(-logits, axis=-1)[..., :2]
    selected = np.take_along_axis(probs, ids, -1)
    selected /= selected.sum(axis=-1, keepdims=True)
    result = np.zeros_like(x)
    for expert in range(3):
        g = x @ weights['ffn_gate_exps.weight'][expert].T
        u = x @ weights['ffn_up_exps.weight'][expert].T
        y = (g / (1 + np.exp(-g)) * u) @ weights['ffn_down_exps.weight'][expert].T
        route = np.sum(np.where(ids == expert, selected, 0), axis=-1)
        result += y * route[..., None]
    gate = x @ weights[SHARED[0]].T
    up = x @ weights[SHARED[1]].T
    shared = (gate / (1 + np.exp(-gate)) * up) @ weights[SHARED[2]].T
    admission = x @ weights['ffn_gate_inp_shexp.weight']
    return result + shared * np.exp(-np.logaddexp(0, -admission))[..., None]


@pytest.fixture
def cuda():
    if BACKEND_NAME != 'cupy':
        pytest.skip('Run with MINI_LLM_BACKEND=cupy for actual CUDA integration')
    pytest.importorskip('triton')
    if xp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    return xp


def _check_full_moe(monkeypatch, kinds, *, stream=None):
    index, weights, reads = _moe_fixture(kinds)
    executor = HeterogeneousExpertExecutor(
        index.provider, model_id='0106u', d_model=256, d_ff=256,
        device_budget_bytes=0, compute_dtype='float32', cpu_packed_backend='reference')
    uploads, fused_calls, linear_calls = [], [], []
    original_upload = projection.packed_to_device
    original_fused = projection.quantized_swiglu
    original_linear = projection.quantized_linear
    original_decode = index.non_expert_weight

    def upload(weight):
        uploads.append(weight)
        return original_upload(weight)

    def fused(hidden, gate, up):
        assert hidden.flags.c_contiguous and hidden.dtype == xp.float32
        if stream is not None:
            assert xp.cuda.get_current_stream().ptr == stream.ptr
        fused_calls.append((gate, up))
        return original_fused(hidden, gate, up)

    def linear(hidden, weight, **kwargs):
        linear_calls.append(weight)
        return original_linear(hidden, weight, **kwargs)

    def decode(name):
        assert not (name.endswith(SHARED) and index.tensors[name].tensor_type.name in BLOCK_FORMATS)
        return original_decode(name)

    monkeypatch.setattr(projection, 'packed_to_device', upload)
    monkeypatch.setattr(projection, 'quantized_swiglu', fused)
    monkeypatch.setattr(projection, 'quantized_linear', linear)
    monkeypatch.setattr(index, 'non_expert_weight', decode)
    host_hidden = np.random.default_rng(10640).normal(0, .1, (2, 4, 256)).astype(np.float32)[:, ::2]
    expected = _reference_moe(host_hidden, weights)
    try:
        with stream if stream is not None else nullcontext():
            moe = Qwen3NextMoEInference.from_index(index, 0, expert_executor=executor)
            assert reads == ['blk.0.ffn_gate_inp.weight', 'blk.0.ffn_gate_inp_shexp.weight']
            assert not moe.projection_weights and not moe._device_projections and not uploads
            hidden = xp.asarray(np.ascontiguousarray(host_hidden).repeat(2, axis=1))[:, ::2]
            assert not hidden.flags.c_contiguous
            before = hidden.copy()
            cold = moe.forward(hidden)
            cached = dict(moe._device_projections)
            for _ in range(2):
                warm = moe.forward(hidden)
            assert all(moe._device_projections[key] is value for key, value in cached.items())
        if stream is not None:
            stream.synchronize()
        actual = asnumpy(warm)
        np.testing.assert_array_equal(actual, asnumpy(cold))
        np.testing.assert_array_equal(asnumpy(hidden), asnumpy(before))
        error = actual.astype(np.float64) - expected
        max_abs = float(np.max(np.abs(error)))
        relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(expected))
        print(f'{BACKEND_NAME} {kinds}: max_abs={max_abs:.9g}, relative_l2={relative_l2:.9g}')
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-6)
        packed_keys = {key for key, kind in zip(SHARED, kinds) if kind in BLOCK_FORMATS}
        on_gpu = BACKEND_NAME == 'cupy'
        assert set(moe.projection_weights) == set(SHARED)
        assert set(moe._device_projections) == (packed_keys if on_gpu else set())
        assert len(uploads) == (len(packed_keys) if on_gpu else 0)
        use_fused = on_gpu and all(kind in BLOCK_FORMATS for kind in kinds[:2])
        assert len(fused_calls) == (3 if use_fused else 0)
        assert len(linear_calls) == 3 * (len(packed_keys) - (2 if use_fused else 0))
        for key in SHARED:
            assert reads.count('blk.0.' + key) == 1
            if key in packed_keys:
                assert np.shares_memory(moe.projection_weights[key].data, index.tensors['blk.0.' + key]._data)
    finally:
        executor.close()


FORMATS = [('F32', 'F32', 'F32'), ('Q8_0', 'F32', 'Q8_0'),
           ('F32', 'Q4_0', 'Q8_0'), ('Q8_0', 'Q4_0', 'Q8_0'), ('Q4_K', 'Q5_K', 'Q6_K')]


@pytest.mark.parametrize('kinds', FORMATS)
def test_cpu_full_moe_fp64_and_warm_cache(monkeypatch, kinds):
    if BACKEND_NAME != 'numpy':
        pytest.skip('CPU full-MoE reference requires NumPy backend')
    _check_full_moe(monkeypatch, kinds)


@pytest.mark.gpu
@pytest.mark.parametrize('kinds', FORMATS)
def test_cuda_full_moe_fp64_nondefault_stream_upload_once(cuda, monkeypatch, kinds):
    _check_full_moe(monkeypatch, kinds, stream=cuda.cuda.Stream(non_blocking=True))


@pytest.mark.gpu
def test_cuda_fused_shape_error_is_not_masked(cuda):
    index, _, _ = _pair(('Q8_0', 'Q4_0'))
    with pytest.raises(ValueError, match='Input K'):
        projection.project_gguf_swiglu(
            cuda.ones((1, 128), cuda.float32), index, 'gate', 'up', {}, {}, 'g', 'u')
