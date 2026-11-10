"""CPU-only Qwen3-Next attention adapter checks; no checkpoint/CUDA runs.

MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q tests/test_0106n_qwen3next_attention.py
Runtime checks skip explicitly until the parent projection/attention helpers exist.
"""

import importlib.util
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'numpy':
    pytest.skip('Qwen3-Next attention tests are CPU-only', allow_module_level=True)

from mini_llm.external import qwen3next_attention as adapter
from mini_llm.external.qwen3next_attention import (
    Qwen3NextAttentionInference, Qwen3NextAttentionSpec,
)
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.inference_state import GenerationState
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from mini_llm.ops.rope import prepare_rope_positions


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self, index):
        return self.value


class Tensor:
    def __init__(self, name, shape, data, reads, kind='F32'):
        self.name = name
        self.shape = tuple(reversed(shape))
        self.tensor_type = SimpleNamespace(name=kind)
        self._data = data
        self.reads = reads

    @property
    def data(self):
        self.reads.append(self.name)
        return self._data


def _metadata(d=8, hq=4, hkv=2, dk=6, dv=6, rotary=4, layers=4):
    return {k: Field(v) for k, v in {
        'general.architecture': 'qwen3next',
        'qwen3next.embedding_length': d,
        'qwen3next.block_count': layers,
        'qwen3next.expert_count': 2,
        'qwen3next.expert_used_count': 1,
        'qwen3next.attention.head_count': hq,
        'qwen3next.attention.head_count_kv': hkv,
        'qwen3next.attention.key_length': dk,
        'qwen3next.attention.value_length': dv,
        'qwen3next.rope.dimension_count': rotary,
        'qwen3next.rope.freq_base': 5e6,
        'qwen3next.attention.layer_norm_rms_epsilon': 1e-6,
        'qwen3next.full_attention_interval': 4,
    }.items()}


def _pack_q8(shape, rng):
    quants = rng.integers(-7, 8, (int(np.prod(shape)) // 32, 32), dtype=np.int8)
    scales = np.full((len(quants), 1), 0.0625, dtype='<f2')
    raw = np.concatenate((scales.view(np.uint8), quants.view(np.uint8)), axis=1)
    return raw.reshape(-1), (scales.astype(np.float32) * quants).reshape(shape)


def _fixture(*, packed=False, header_only=False):
    rng = np.random.default_rng(10614)
    reads = []
    fields = (_metadata(2048, 16, 2, 256, 256, 64, 48) if header_only else
              _metadata(32, 4, 2, 32, 32, 8) if packed else _metadata())
    reader = SimpleNamespace(fields=fields, tensors=[])
    d = fields['qwen3next.embedding_length'].value
    layers = fields['qwen3next.block_count'].value
    # Actual index, including its required expert headers. These payloads are
    # never used, and a header-only fixture allocates no checkpoint-sized array.
    for layer in range(layers):
        for suffix, shape in {
            'ffn_down_exps.weight': (2, d, 2),
            'ffn_gate_up_exps.weight': (2, 4, d),
            'ffn_gate_inp.weight': (2, d),
        }.items():
            reader.tensors.append(Tensor(f'blk.{layer}.' + suffix, shape, None, reads))
    index = Qwen3NextGGUFIndex(reader)
    spec = Qwen3NextAttentionSpec.from_index(index)
    weights = {}
    for layer in (range(3, layers, 4) if header_only else (3,)):
        for suffix, shape in spec.tensor_shapes().items():
            kind = 'Q8_0' if (packed or header_only) and suffix in adapter._PROJECTIONS else 'F32'
            data = None
            if not header_only:
                value = rng.normal(0, 0.35, shape).astype(np.float32)
                if suffix.endswith('_norm.weight'):
                    value = np.linspace(0.4, 1.6, spec.key_dim, dtype=np.float32)
                data = value
                if kind == 'Q8_0':
                    data, value = _pack_q8(shape, rng)
                weights[suffix] = value
            tensor = Tensor(f'blk.{layer}.' + suffix, shape, data, reads, kind)
            reader.tensors.append(tensor)
            index.tensors[tensor.name] = tensor
    assert reads == []
    return index, weights, reads


@pytest.fixture
def runtime():
    modules = ('mini_llm.ops.projected_attention_inference',
               'mini_llm.ops.rotary_inference', 'mini_llm.external.gguf_projection')
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip('Parent helpers not ready: ' + ', '.join(missing))


def _reference(hidden, weights, spec, *, prefix=None, positions=None,
               global_split=False, scale=None, gamma_offset=0):
    """Independent FP64 scalar graph, including native-head cached history."""
    bsz, time, _ = hidden.shape
    hq, hkv, dk, dv = spec.n_q_heads, spec.n_kv_heads, spec.key_dim, spec.value_dim
    hidden = hidden.astype(np.float64)
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    raw = hidden @ w['attn_q.weight'].T
    if global_split:
        q, gate = np.split(raw, 2, axis=-1)
        q, gate = q.reshape(bsz, time, hq, dk), gate.reshape(bsz, time, hq, dk)
    else:
        raw = raw.reshape(bsz, time, hq, 2 * dk)
        q, gate = raw[..., :dk].copy(), raw[..., dk:]
    k = (hidden @ w['attn_k.weight'].T).reshape(bsz, time, hkv, dk)
    v = (hidden @ w['attn_v.weight'].T).reshape(bsz, time, hkv, dv)
    offset = 0 if prefix is None else prefix[0].shape[2]
    if positions is None:
        positions = np.broadcast_to(np.arange(offset, offset + time), (bsz, time))
    elif np.ndim(positions) == 1:
        positions = np.broadcast_to(positions, (bsz, time))
    for x, suffix in ((q, 'attn_q_norm.weight'), (k, 'attn_k_norm.weight')):
        for b in range(bsz):
            for t in range(time):
                for h in range(x.shape[2]):
                    row = x[b, t, h]
                    inv = 1 / math.sqrt(sum(float(z) ** 2 for z in row) / dk + spec.rms_eps)
                    row *= inv * (w[suffix] + gamma_offset)
                    original = row.copy()
                    half = spec.rotary_dim // 2
                    for j in range(half):
                        angle = float(positions[b, t]) / spec.rope_base ** (2 * j / spec.rotary_dim)
                        c, s = math.cos(angle), math.sin(angle)
                        row[j] = original[j] * c - original[j + half] * s
                        row[j + half] = original[j + half] * c + original[j] * s
    kh, vh = k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
    if prefix is not None:
        kh = np.concatenate((prefix[0].astype(np.float64), kh), axis=2)
        vh = np.concatenate((prefix[1].astype(np.float64), vh), axis=2)
    scale = 1 / math.sqrt(dk) if scale is None else scale
    out = np.empty((bsz, time, hq, dv), np.float64)
    for b in range(bsz):
        for t in range(time):
            for h in range(hq):
                kvh = h // (hq // hkv)
                scores = np.array([
                    sum(float(q[b, t, h, j]) * kh[b, kvh, p, j] for j in range(dk)) * scale
                    for p in range(offset + t + 1)])
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                for j in range(dv):
                    context = sum(probs[p] * vh[b, kvh, p, j] for p in range(len(probs)))
                    g = float(gate[b, t, h, j])
                    sigmoid = 1 / (1 + math.exp(-g)) if g >= 0 else math.exp(g) / (1 + math.exp(g))
                    out[b, t, h, j] = context * sigmoid
    return out.reshape(bsz, time, hq * dv) @ w['attn_output.weight'].T, kh, vh


def _prefill_state(model, batch=2, prefix_length=2, capacity=9):
    rng = np.random.default_rng(88)
    state = model.create_state(batch, capacity)
    spec = model.spec
    k = rng.normal(size=(batch, spec.n_kv_heads, prefix_length, spec.key_dim)).astype(np.float32)
    v = rng.normal(size=(batch, spec.n_kv_heads, prefix_length, spec.value_dim)).astype(np.float32)
    state.k_cache.fill(0)
    state.v_cache.fill(0)
    state.k_cache[:, :, :prefix_length] = k
    state.v_cache[:, :, :prefix_length] = v
    state.length = prefix_length
    return state, (k.copy(), v.copy())


def test_actual_checkpoint_header_metadata_only():
    index, _, reads = _fixture(header_only=True)
    spec = Qwen3NextAttentionSpec.from_index(index)
    assert (spec.d_model, spec.layers, spec.n_q_heads, spec.n_kv_heads) == (2048, 48, 16, 2)
    assert (spec.key_dim, spec.value_dim, spec.rotary_dim, spec.rope_base) == (256, 256, 64, 5e6)
    assert spec.rms_eps == 1e-6
    assert spec.tensor_shapes()['attn_q.weight'] == (8192, 2048)
    full = [layer for layer in range(spec.layers) if (layer + 1) % spec.full_attention_interval == 0]
    assert full == list(range(3, 48, 4))
    for layer in full:
        for suffix, shape in spec.tensor_shapes().items():
            tensor = index.tensors[f'blk.{layer}.' + suffix]
            assert tuple(reversed(tensor.shape)) == shape
            if suffix in adapter._PROJECTIONS:
                assert tensor.tensor_type.name == 'Q8_0'
    assert reads == []


def test_architecture_alias_and_identity_scaling():
    index, _, reads = _fixture()
    fields = index.reader.fields
    index.reader.fields = {name.replace('qwen3next', 'qwen3_next'): field for name, field in fields.items()}
    index.architecture = 'qwen3_next'
    index.reader.fields['qwen3_next.rope.scaling.type'] = Field('linear')
    index.reader.fields['qwen3_next.rope.scaling.factor'] = Field(1.0)
    assert Qwen3NextAttentionSpec.from_index(index).key_dim == 6
    assert reads == []


@pytest.mark.parametrize('packed', [False, True])
def test_fp64_scalar_reference_full_chunks_steps_and_native_cache(runtime, packed):
    index, weights, _ = _fixture(packed=packed)
    model = Qwen3NextAttentionInference.from_index(index, 3)
    spec = model.spec
    hidden = np.random.default_rng(59).normal(size=(2, 5, spec.d_model)).astype(np.float32)
    original = hidden.copy()
    state, prefix = _prefill_state(model)
    assert isinstance(state, GenerationState)
    expected, k, v = _reference(hidden, weights, spec, prefix=prefix)
    outputs = []
    for chunks in ((5,), (2, 1, 2), (1, 1, 1, 1, 1)):
        current = model.attention.copy_state(state)
        assert not np.shares_memory(current.k_cache, state.k_cache)
        assert not np.shares_memory(current.v_cache, state.v_cache)
        parts, start = [], 0
        for count in chunks:
            part = hidden[:, start:start + count]
            parts.append(model.step(part[:, 0], current)[:, None] if count == 1 else model.forward(part, current))
            start += count
        actual = np.concatenate(parts, axis=1)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=6e-6)
        np.testing.assert_allclose(current.k_cache[:, :, :7], k, rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(current.v_cache[:, :, :7], v, rtol=2e-5, atol=3e-6)
        assert current.length == 7
        assert current.k_cache.shape == (2, spec.n_kv_heads, 9, spec.key_dim)
        assert current.v_cache.shape == (2, spec.n_kv_heads, 9, spec.value_dim)
        np.testing.assert_array_equal(current.k_cache[:, :, 7:], 0)
        outputs.append(actual)
    np.testing.assert_array_equal(hidden, original)
    assert state.length == 2
    for actual in outputs[1:]:
        np.testing.assert_allclose(actual, outputs[0], rtol=3e-5, atol=6e-6)
    if not packed:
        assert spec.d_model != spec.n_q_heads * spec.key_dim
        for kwargs in ({'global_split': True}, {'scale': 1 / math.sqrt(spec.rotary_dim)},
                       {'gamma_offset': 1}):
            wrong, _, _ = _reference(hidden, weights, spec, prefix=prefix, **kwargs)
            assert np.max(np.abs(wrong - expected)) > 1e-3


def test_explicit_positions_prepared_once_and_not_causal_chronology(runtime, monkeypatch):
    index, weights, _ = _fixture()
    model = Qwen3NextAttentionInference.from_index(index, 3)
    hidden = np.random.default_rng(9).normal(size=(2, 3, 8)).astype(np.float32)
    positions = np.array([[19, 2, 7], [4, 11, 0]], np.int64)
    state, prefix = _prefill_state(model)
    calls = []

    def prepare(*args, **kwargs):
        result = prepare_rope_positions(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(adapter, 'prepare_rope_positions', prepare)
    actual = model.forward(hidden, state, position_ids=positions)
    expected, k, v = _reference(hidden, weights, model.spec, prefix=prefix, positions=positions)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=6e-6)
    np.testing.assert_allclose(state.k_cache[:, :, :5], k, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(state.v_cache[:, :, :5], v, rtol=2e-5, atol=3e-6)
    assert len(calls) == 1 and state.length == 5
    other, _ = _prefill_state(model)
    np.testing.assert_allclose(model.forward(hidden, other, position_ids=calls[0]), actual)
    assert len(calls) == 1
    stepped, _ = _prefill_state(model)
    parts = [model.step(hidden[:, t], stepped, position_ids=positions[:, t:t + 1]) for t in range(3)]
    np.testing.assert_allclose(np.stack(parts, axis=1), expected, rtol=3e-5, atol=6e-6)


@pytest.mark.parametrize('suffix', tuple(Qwen3NextAttentionSpec(8, 4, 4, 2, 6, 6, 4, 5e6, 1e-6, 4).tensor_shapes()))
@pytest.mark.parametrize('missing', [False, True])
def test_all_headers_before_payload(suffix, missing):
    index, _, reads = _fixture(packed=False)
    if missing:
        del index.tensors['blk.3.' + suffix]
    else:
        tensor = index.tensors['blk.3.' + suffix]
        shape = list(tensor.shape)
        shape[-1] += 1
        tensor.shape = tuple(shape)
    with pytest.raises(ValueError, match='Missing full-attention tensor|expected shape'):
        Qwen3NextAttentionInference.from_index(index, 3)
    assert reads == []


@pytest.mark.parametrize('suffix,value', [
    ('attention.head_count', 3), ('attention.head_count_kv', True),
    ('attention.key_length', 0), ('attention.value_length', 5),
    ('rope.dimension_count', 7), ('rope.dimension_count', 8),
    ('rope.freq_base', float('nan')), ('full_attention_interval', 0),
    ('attention.layer_norm_rms_epsilon', -1), ('rope.scaling.type', 'yarn'),
    ('rope.scaling.type', 'linear'), ('rope.scaling.factor', 2),
    ('rope.scaling.factor', True), ('rope.scaling.factor', float('inf')),
    ('rope.scaling.original_context_length', 32768), ('rope.freq_scale', 1),
])
def test_invalid_metadata_before_payload(suffix, value):
    index, _, reads = _fixture()
    index.reader.fields['qwen3next.' + suffix] = Field(value)
    with pytest.raises((ValueError, NotImplementedError)):
        Qwen3NextAttentionInference.from_index(index, 3)
    assert reads == []


@pytest.mark.parametrize('suffix', ['attention.head_count', 'attention.head_count_kv',
    'attention.key_length', 'attention.value_length', 'rope.dimension_count',
    'rope.freq_base', 'attention.layer_norm_rms_epsilon', 'full_attention_interval'])
def test_exact_metadata_required(suffix):
    index, _, reads = _fixture()
    index.reader.fields.pop('qwen3next.' + suffix)
    with pytest.raises(ValueError):
        Qwen3NextAttentionInference.from_index(index, 3)
    assert reads == []


@pytest.mark.parametrize('layer,error', [(0, ValueError), (2, ValueError), (-1, IndexError),
                                          (4, IndexError), (True, TypeError), (3.0, TypeError)])
def test_layer_rejections(layer, error):
    index, _, reads = _fixture()
    with pytest.raises(error):
        Qwen3NextAttentionInference.from_index(index, layer)
    assert reads == []


@pytest.mark.parametrize('suffix', ['attn_q.bias', 'attn_output.bias', 'attn_q_norm.bias',
                                  'attn_qkv.weight', 'attn_gate.weight', 'rope_freqs.weight'])
def test_unsupported_layouts(suffix):
    index, _, reads = _fixture()
    name = 'blk.3.' + suffix
    index.tensors[name] = Tensor(name, (8,), None, reads)
    with pytest.raises(NotImplementedError, match='layout'):
        Qwen3NextAttentionInference.from_index(index, 3)
    assert reads == []


def test_unsupported_format_alignment_and_backend_before_payload():
    index, _, reads = _fixture()
    tensor = index.tensors['blk.3.attn_output.weight']
    tensor.tensor_type.name = 'IQ4_XS'
    with pytest.raises(NotImplementedError, match='IQ4_XS'):
        Qwen3NextAttentionInference.from_index(index, 3)
    tensor.tensor_type.name = 'Q8_0'
    with pytest.raises(ValueError, match='block-aligned'):
        Qwen3NextAttentionInference.from_index(index, 3)
    with pytest.raises(ValueError, match='cpu_packed_backend'):
        Qwen3NextAttentionInference.from_index(index, 3, 'auto')
    assert reads == []


def test_capacity_overflow_before_projection_or_state_mutation(runtime):
    index, _, reads = _fixture()
    model = Qwen3NextAttentionInference.from_index(index, 3)
    state, _ = _prefill_state(model, capacity=3)
    k, v, before = state.k_cache.copy(), state.v_cache.copy(), reads.copy()
    with pytest.raises(ValueError, match='capacity'):
        model.forward(np.ones((2, 2, 8), np.float32), state)
    assert state.length == 2 and reads == before
    np.testing.assert_array_equal(state.k_cache, k)
    np.testing.assert_array_equal(state.v_cache, v)


@pytest.mark.parametrize('dtype', [np.float16, np.float64, np.int32])
def test_hidden_dtype_rejected_without_projection_reads(runtime, dtype):
    index, _, reads = _fixture()
    model = Qwen3NextAttentionInference.from_index(index, 3)
    state = model.create_state(1, 4)
    before = reads.copy()
    with pytest.raises(TypeError, match='float32 hidden only'):
        model.forward(np.ones((1, 2, 8), dtype), state)
    with pytest.raises(TypeError, match='float32 hidden only'):
        model.step(np.ones((1, 8), dtype), state)
    assert state.length == 0 and reads == before


@pytest.mark.parametrize('kind', ['F32', 'F16', 'BF16'])
def test_dense_loading_lazy_and_weight_owner_unchanged(runtime, kind):
    index, weights, reads = _fixture()
    for suffix, value in weights.items():
        tensor = index.tensors['blk.3.' + suffix]
        tensor.tensor_type.name = kind
        if kind == 'F16':
            tensor._data = value.astype(np.float16)
            weights[suffix] = tensor._data.astype(np.float32)
        elif kind == 'BF16':
            words = (value.view(np.uint32) >> 16).astype('<u2')
            tensor._data = words.view(np.uint8).reshape(-1)
            weights[suffix] = (words.astype(np.uint32) << 16).view(np.float32).reshape(value.shape)
    originals = {s: index.tensors['blk.3.' + s]._data.copy() for s in weights}
    owner_writeable = {s: index.tensors['blk.3.' + s]._data.flags.writeable for s in weights}
    model = Qwen3NextAttentionInference.from_index(index, 3)
    assert model.index is index
    assert sorted(reads) == ['blk.3.attn_k_norm.weight', 'blk.3.attn_q_norm.weight']
    assert not model.projection_weights and not model._device_projections
    hidden = np.random.default_rng(8).normal(size=(1, 3, 8)).astype(np.float32)
    expected, _, _ = _reference(hidden, weights, model.spec)
    np.testing.assert_allclose(model.forward(hidden, model.create_state(1, 4)), expected,
                               rtol=3e-5, atol=6e-6)
    assert len(reads) == 6
    model.forward(hidden, model.create_state(1, 4))
    assert len(reads) == 6
    for suffix in weights:
        owner = index.tensors['blk.3.' + suffix]._data
        np.testing.assert_array_equal(owner, originals[suffix])
        assert owner.flags.writeable == owner_writeable[suffix]
    assert all(w.dtype == np.float32 and not w.flags.writeable for w in model.projection_weights.values())


@pytest.mark.parametrize('backend', ['reference', 'native'])
def test_packed_view_dispatch_without_eager_decode_or_upload(runtime, monkeypatch, backend):
    from mini_llm.external import gguf_projection

    index, weights, reads = _fixture(packed=True)
    originals = {s: index.tensors['blk.3.' + s]._data.copy() for s in adapter._PROJECTIONS}

    def forbidden(*args, **kwargs):
        raise AssertionError('No eager packed decode or CPU device upload')

    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', forbidden)
    monkeypatch.setattr(gguf_projection, 'packed_to_device', forbidden)
    model = Qwen3NextAttentionInference.from_index(index, 3, backend)
    calls = []

    def linear(hidden, weight, *, cpu_packed_backend):
        suffix = next(s for s, cached in model.projection_weights.items() if cached is weight)
        calls.append((suffix, cpu_packed_backend))
        assert hidden.flags.c_contiguous and hidden.dtype == np.float32
        return hidden @ weights[suffix].T

    monkeypatch.setattr(gguf_projection, 'quantized_linear', linear)
    for suffix, weight in model.projection_weights.items():
        assert isinstance(weight, PackedGGUFWeight)
        assert np.shares_memory(weight.data, index.tensors['blk.3.' + suffix]._data)
        assert not weight.data.flags.writeable
    hidden = np.random.default_rng(4).normal(size=(1, 3, 32)).astype(np.float32)
    expected, _, _ = _reference(hidden, weights, model.spec)
    np.testing.assert_allclose(model.forward(hidden, model.create_state(1, 4)), expected,
                               rtol=3e-5, atol=6e-6)
    assert calls == [(s, backend) for s in adapter._PROJECTIONS]
    assert len(reads) == 6 and not model._device_projections
    for suffix in adapter._PROJECTIONS:
        np.testing.assert_array_equal(index.tensors['blk.3.' + suffix]._data, originals[suffix])


def test_extreme_raw_gates_are_exact_stable_sigmoid(runtime):
    index, _, _ = _fixture()
    tensor = index.tensors['blk.3.attn_q.weight']
    qg = tensor._data.reshape(4, 12, 8)
    qg[:, 6:, :] = 0
    qg[:, 6:, 0] = np.array([-1000, 1000, -100, 100, 0, 2], np.float32)
    weights = {s: index.tensors['blk.3.' + s]._data for s in Qwen3NextAttentionSpec.from_index(index).tensor_shapes()}
    model = Qwen3NextAttentionInference.from_index(index, 3)
    hidden = np.ones((1, 2, 8), np.float32)
    expected, _, _ = _reference(hidden, weights, model.spec)
    with np.errstate(over='raise', invalid='raise'):
        actual = model.forward(hidden, model.create_state(1, 3))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=6e-6)
