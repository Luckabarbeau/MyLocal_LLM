"""CPU-only synthetic GGUF mixer checks; no model downloads or CUDA work.

Run after the parent-provided ops/gated_delta_inference.py is ready:
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q tests/test_0106m_qwen3next_mixer.py
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'numpy':
    pytest.skip('Qwen3-Next adapter tests are NumPy CPU-only', allow_module_level=True)

from mini_llm.external import qwen3next_mixer as adapter
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.external.qwen3next_mixer import (
    Qwen3NextMixerSpec,
    Qwen3NextRecurrentMixerInference,
)
from mini_llm.ops.quantized_weight import PackedGGUFWeight


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self, index):
        return self.value


class Tensor:
    """Real reader orientation and observable payload access, including views."""

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


def _metadata(*, d_model=8, hk=2, hv=6, dk=3, dv=2):
    return {k: Field(v) for k, v in {
        'general.architecture': 'qwen3next',
        'qwen3next.block_count': 4,
        'qwen3next.embedding_length': d_model,
        'qwen3next.expert_count': 2,
        'qwen3next.expert_used_count': 1,
        'qwen3next.ssm.group_count': hk,
        'qwen3next.ssm.time_step_rank': hv,
        'qwen3next.ssm.state_size': dk,
        'qwen3next.ssm.inner_size': hv * dv,
        'qwen3next.ssm.conv_kernel': 4,
        'qwen3next.full_attention_interval': 4,
        'qwen3next.attention.layer_norm_rms_epsilon': 1e-6,
    }.items()}


def _pack_q8(shape, rng):
    quants = rng.integers(-7, 8, size=(int(np.prod(shape)) // 32, 32), dtype=np.int8)
    scales = np.full((len(quants), 1), 0.0625, dtype='<f2')
    raw = np.concatenate((scales.view(np.uint8), quants.view(np.uint8)), axis=1)
    dense = (scales.astype(np.float32) * quants).reshape(shape)
    return raw.reshape(-1), dense


def _fixture(*, packed=False):
    rng = np.random.default_rng(106)
    reads = []
    fields = _metadata(d_model=32, dv=32) if packed else _metadata()
    reader = SimpleNamespace(fields=fields, tensors=[])
    # Use the actual index (including its required expert headers), not a
    # substitute non_expert_weight implementation. No routed bank is accessed.
    d_model = fields['qwen3next.embedding_length'].value
    for layer in range(4):
        for suffix, shape in {
            'ffn_down_exps.weight': (2, d_model, 2),
            'ffn_gate_up_exps.weight': (2, 4, d_model),
            'ffn_gate_inp.weight': (2, d_model),
        }.items():
            reader.tensors.append(Tensor(f'blk.{layer}.' + suffix, shape,
                                         np.zeros(shape, np.float32), reads))
    index = Qwen3NextGGUFIndex(reader)
    spec = Qwen3NextMixerSpec.from_index(index)
    weights = {}
    for suffix, shape in spec.tensor_shapes().items():
        value = rng.normal(0, 0.28, shape).astype(np.float32)
        if suffix == 'ssm_a':
            value = -np.linspace(0.15, 1.45, spec.n_value_heads, dtype=np.float32)
        elif suffix == 'ssm_dt.bias':
            value = np.linspace(-0.8, 0.9, spec.n_value_heads, dtype=np.float32)
        elif suffix == 'ssm_norm.weight':
            value = np.linspace(0.4, 1.6, spec.value_dim, dtype=np.float32)
        kind = 'F32'
        payload = value
        if packed and suffix in adapter._PROJECTIONS:
            payload, value = _pack_q8(shape, rng)
            kind = 'Q8_0'
        tensor = Tensor('blk.0.' + suffix, shape, payload, reads, kind)
        reader.tensors.append(tensor)
        index.tensors[tensor.name] = tensor
        weights[suffix] = value
    assert reads == []
    return index, weights, reads


def _reference(hidden, weights, spec, history=None, memory=None, *,
               ba_layout='group', transform_a_again=False, reverse_conv=False,
               norm_offset=False):
    """Independent float64 graph with scalar conv and affine state transform.

    S_t = exp(g_t) (I - beta_t k_t k_t^T) S_(t-1)
          + beta_t k_t v_t^T; output reads the UPDATED state.
    No production convolution, delta, normalization or projection is called.
    """
    x = hidden.astype(np.float64)
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    batch, time = x.shape[:2]
    hk, hv, dk, dv = (spec.n_key_heads, spec.n_value_heads,
                       spec.key_dim, spec.value_dim)
    channels, kernel = spec.qkv_channels, spec.conv_kernel_size
    history = (np.zeros((batch, kernel - 1, channels)) if history is None
               else history.astype(np.float64).copy())
    memory = (np.zeros((batch, hv, dk, dv)) if memory is None
              else memory.astype(np.float64).copy())
    qkv = x @ w['attn_qkv.weight'].T
    gate = (x @ w['attn_gate.weight'].T).reshape(batch, time, hv, dv)
    raw_ba = x @ w['ssm_ba.weight'].T
    ratio = hv // hk
    b = np.empty((batch, time, hv))
    a = np.empty_like(b)
    # Index heads explicitly so this oracle does not mirror adapter reshapes.
    for head in range(hv):
        group, local = divmod(head, ratio)
        if ba_layout == 'group':
            b_row, a_row = group * 2 * ratio + local, group * 2 * ratio + ratio + local
        elif ba_layout == 'global':
            b_row, a_row = head, hv + head
        elif ba_layout == 'pairs':
            b_row, a_row = 2 * head, 2 * head + 1
        else:
            raise AssertionError(ba_layout)
        b[..., head], a[..., head] = raw_ba[..., b_row], raw_ba[..., a_row]
    beta = np.exp(-np.logaddexp(0, -b))
    ssm_a = w['ssm_a']
    if transform_a_again:
        ssm_a = -np.exp(ssm_a)
    log_decay = ssm_a * np.logaddexp(0, a + w['ssm_dt.bias'])
    output = np.empty((batch, time, hv, dv))
    taps = w['ssm_conv1d.weight']
    if reverse_conv:
        taps = taps[:, ::-1]
    for t in range(time):
        window = np.concatenate((history, qkv[:, t:t + 1]), axis=1)
        convolved = np.zeros((batch, channels))
        for n in range(batch):
            for c in range(channels):
                for j in range(kernel):
                    convolved[n, c] += window[n, j, c] * taps[c, j]
        convolved *= np.exp(-np.logaddexp(0, -convolved))
        history = window[:, 1:].copy()
        q = convolved[:, :hk * dk].reshape(batch, hk, dk)
        k = convolved[:, hk * dk:2 * hk * dk].reshape(batch, hk, dk)
        v = convolved[:, 2 * hk * dk:].reshape(batch, hv, dv)
        for n in range(batch):
            for head in range(hv):
                kh = head // ratio
                query = q[n, kh] / np.sqrt(np.dot(q[n, kh], q[n, kh]) + spec.l2_eps)
                key = k[n, kh] / np.sqrt(np.dot(k[n, kh], k[n, kh]) + spec.l2_eps)
                rate = beta[n, t, head]
                transform = np.exp(log_decay[n, t, head]) * (
                    np.eye(dk) - rate * np.outer(key, key))
                memory[n, head] = (transform @ memory[n, head]
                                   + rate * np.outer(key, v[n, head]))
                value = (query / np.sqrt(dk)) @ memory[n, head]
                value /= np.sqrt(np.mean(value * value) + spec.rms_eps)
                gamma = w['ssm_norm.weight'] + (1 if norm_offset else 0)
                z = gate[n, t, head]
                output[n, t, head] = value * gamma * z * np.exp(-np.logaddexp(0, -z))
    return (output.reshape(batch, time, hv * dv) @ w['ssm_out.weight'].T,
            history, memory)


def test_spec_exact_metadata_and_real_checkpoint_dimensions():
    index, _, reads = _fixture()
    spec = Qwen3NextMixerSpec.from_index(index)
    assert (spec.n_key_heads, spec.n_value_heads, spec.key_dim, spec.value_dim) == (2, 6, 3, 2)
    assert spec.full_attention_interval == 4
    assert reads == []
    actual = SimpleNamespace(
        architecture='qwen3next', d_model=2048, layers=48,
        reader=SimpleNamespace(fields=_metadata(d_model=2048, hk=16, hv=32, dk=128, dv=128)),
    )
    actual_spec = Qwen3NextMixerSpec.from_index(actual)
    assert actual_spec.qkv_channels == 8192
    assert actual_spec.inner_size == 4096
    assert actual_spec.layers == 48
    assert actual_spec.tensor_shapes()['ssm_out.weight'] == (2048, 4096)


def test_float32_matches_independent_float64_graph_from_nonzero_state():
    index, weights, _ = _fixture()
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    state = mixer.create_state(2)
    rng = np.random.default_rng(19)
    prefix = rng.normal(size=(2, 3, mixer.spec.d_model)).astype(np.float32)
    hidden = rng.normal(size=(2, 7, mixer.spec.d_model)).astype(np.float32)
    prefix_expected, history, memory = _reference(prefix, weights, mixer.spec)
    np.testing.assert_allclose(mixer.forward(prefix, state), prefix_expected, rtol=3e-4, atol=5e-6)
    assert np.linalg.norm(history) > 0 and np.linalg.norm(memory) > 0
    assert np.any(state.conv.history != 0)
    expected, final_history, _ = _reference(hidden, weights, mixer.spec, history, memory)
    actual = mixer.forward(hidden, state)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=5e-6)
    np.testing.assert_allclose(state.conv.history, final_history, rtol=2e-6, atol=2e-6)
    assert state.length == 10
    # Demonstrate the fixture can detect the easy-to-miss model mappings.
    for layout, again, reverse, offset in (
            ('global', False, False, False), ('pairs', False, False, False),
            ('group', True, False, False), ('group', False, True, False),
            ('group', False, False, True)):
        wrong, _, _ = _reference(hidden, weights, mixer.spec, history, memory,
                                 ba_layout=layout, transform_a_again=again,
                                 reverse_conv=reverse, norm_offset=offset)
        assert np.max(np.abs(expected - wrong)) > 1e-4, (layout, again, reverse, offset)


def test_group_interleaved_ba_and_pretransformed_a_are_passed_to_generic(monkeypatch):
    index, weights, _ = _fixture()
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    hidden = np.random.default_rng(2).normal(size=(2, 3, mixer.spec.d_model)).astype(np.float32)
    captured = {}

    def core(qkv, gate, log_decay, beta, state):
        captured.update(qkv=qkv, gate=gate, log_decay=log_decay, beta=beta, state=state)
        return np.zeros(gate.shape, np.float32)

    monkeypatch.setattr(mixer.mixer, 'forward', core)
    state = mixer.create_state(2)
    mixer.forward(hidden, state)
    raw = hidden.astype(np.float64) @ weights['ssm_ba.weight'].astype(np.float64).T
    # Ratio=3: groups are [b0,b1,b2,a0,a1,a2], [b3,b4,b5,a3,a4,a5].
    b = raw[..., [0, 1, 2, 6, 7, 8]]
    a = raw[..., [3, 4, 5, 9, 10, 11]]
    np.testing.assert_allclose(captured['beta'], np.exp(-np.logaddexp(0, -b)), rtol=2e-6)
    np.testing.assert_allclose(captured['log_decay'], weights['ssm_a'] * np.logaddexp(
        0, a + weights['ssm_dt.bias']), rtol=2e-6)
    np.testing.assert_allclose(captured['qkv'], hidden @ weights['attn_qkv.weight'].T)
    np.testing.assert_allclose(captured['gate'].reshape(2, 3, -1),
                               hidden @ weights['attn_gate.weight'].T)
    assert captured['qkv'].flags.c_contiguous
    assert captured['state'] is state


def test_extreme_ba_uses_stable_sigmoid_and_softplus(monkeypatch):
    index, weights, _ = _fixture()
    ba = weights['ssm_ba.weight']
    ba.fill(0)
    ba[:, 0] = [-1000, 1000, 0, -1000, 1000, 0] * 2
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    captured = {}

    def core(qkv, gate, log_decay, beta, state):
        captured.update(log_decay=log_decay, beta=beta)
        return np.zeros(gate.shape, np.float32)

    monkeypatch.setattr(mixer.mixer, 'forward', core)
    hidden = np.zeros((1, 1, mixer.spec.d_model), np.float32)
    hidden[..., 0] = 1
    with np.errstate(over='raise', invalid='raise'):
        mixer.forward(hidden, mixer.create_state(1))
    np.testing.assert_array_equal(captured['beta'], [[[0, 1, 0.5, 0, 1, 0.5]]])
    assert np.isfinite(captured['log_decay']).all()
    assert (captured['log_decay'] <= 0).all()


def test_chunk_step_copy_reset_empty_and_independent_sessions():
    index, _, _ = _fixture()
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    rng = np.random.default_rng(4)
    prefix = rng.normal(size=(2, 3, mixer.spec.d_model)).astype(np.float32)
    hidden = rng.normal(size=(2, 7, mixer.spec.d_model)).astype(np.float32)
    initial = mixer.create_state(2)
    mixer.forward(prefix, initial)
    full, chunks, steps = (initial.copy() for _ in range(3))
    expected = mixer.forward(hidden, full)
    chunked = np.concatenate([mixer.forward(hidden[:, :2], chunks),
                              mixer.forward(hidden[:, 2:5], chunks),
                              mixer.forward(hidden[:, 5:], chunks)], axis=1)
    stepped = np.stack([mixer.step(hidden[:, t], steps) for t in range(7)], axis=1)
    np.testing.assert_allclose(chunked, expected, rtol=3e-4, atol=5e-6)
    np.testing.assert_allclose(stepped, expected, rtol=3e-4, atol=5e-6)
    assert initial.length == 3
    assert full.length == chunks.length == steps.length == 10
    assert not np.shares_memory(initial.conv.history, full.conv.history)
    history = full.conv.history.copy()
    empty = mixer.forward(hidden[:, :0], full)
    assert empty.shape == (2, 0, mixer.spec.d_model)
    assert full.length == 10
    np.testing.assert_array_equal(full.conv.history, history)
    # Empty chunks must preserve delta state too, as observed by a later read.
    np.testing.assert_allclose(mixer.step(hidden[:, 0], full),
                               mixer.step(hidden[:, 0], chunks), rtol=3e-4, atol=5e-6)
    full.reset()
    assert full.length == 0
    np.testing.assert_array_equal(full.conv.history, 0)
    fresh = mixer.create_state(2)
    np.testing.assert_allclose(mixer.forward(hidden, full), mixer.forward(hidden, fresh),
                               rtol=3e-4, atol=5e-6)


@pytest.mark.parametrize('suffix', list(Qwen3NextMixerSpec(8, 4, 2, 6, 3, 2, 4, 4, 1e-6).tensor_shapes()))
def test_all_shapes_validated_before_any_payload_access(suffix):
    index, _, reads = _fixture()
    tensor = index.tensors['blk.0.' + suffix]
    shape = list(reversed(tensor.shape))
    shape[0] += 1
    tensor.shape = tuple(reversed(shape))
    with pytest.raises(ValueError, match='expected shape'):
        Qwen3NextRecurrentMixerInference.from_index(index, 0)
    assert reads == []


@pytest.mark.parametrize('suffix', list(Qwen3NextMixerSpec(8, 4, 2, 6, 3, 2, 4, 4, 1e-6).tensor_shapes()))
def test_missing_tensors_rejected_before_payload_access(suffix):
    index, _, reads = _fixture()
    del index.tensors['blk.0.' + suffix]
    with pytest.raises(ValueError, match='Missing recurrent mixer tensor'):
        Qwen3NextRecurrentMixerInference.from_index(index, 0)
    assert reads == []


@pytest.mark.parametrize('suffix,value', [
    ('ssm.group_count', 4), ('ssm.group_count', True),
    ('ssm.time_step_rank', 5), ('ssm.inner_size', 13),
    ('ssm.state_size', 0), ('ssm.conv_kernel', -1),
    ('full_attention_interval', 0), ('full_attention_interval', 4.5),
    ('attention.layer_norm_rms_epsilon', float('nan')),
    ('attention.layer_norm_rms_epsilon', 0),
])
def test_incompatible_metadata_rejected_before_payload(suffix, value):
    index, _, reads = _fixture()
    index.reader.fields['qwen3next.' + suffix] = Field(value)
    with pytest.raises(ValueError):
        Qwen3NextRecurrentMixerInference.from_index(index, 0)
    assert reads == []


@pytest.mark.parametrize('suffix', [
    'ssm.group_count', 'ssm.time_step_rank', 'ssm.inner_size', 'ssm.state_size',
    'ssm.conv_kernel', 'full_attention_interval', 'attention.layer_norm_rms_epsilon',
])
def test_exact_metadata_required_without_alias_fallback(suffix):
    index, _, reads = _fixture()
    index.reader.fields['qwen3next.not_the_right_key.' + suffix] = index.reader.fields.pop('qwen3next.' + suffix)
    with pytest.raises(ValueError):
        Qwen3NextRecurrentMixerInference.from_index(index, 0)
    assert reads == []


@pytest.mark.parametrize('layer,exception', [(3, ValueError), (-1, IndexError),
                                            (4, IndexError), (0.5, TypeError), (True, TypeError)])
def test_full_attention_and_invalid_layers_rejected_before_payload(layer, exception):
    index, _, reads = _fixture()
    with pytest.raises(exception):
        Qwen3NextRecurrentMixerInference.from_index(index, layer)
    assert reads == []


def test_unsupported_format_alignment_and_backend_rejected_before_payload():
    index, _, reads = _fixture()
    tensor = index.tensors['blk.0.ssm_out.weight']
    tensor.tensor_type.name = 'IQ4_XS'
    with pytest.raises(NotImplementedError, match='IQ4_XS'):
        Qwen3NextRecurrentMixerInference.from_index(index, 0)
    tensor.tensor_type.name = 'Q8_0'
    with pytest.raises(ValueError, match='block-aligned'):
        Qwen3NextRecurrentMixerInference.from_index(index, 0)
    with pytest.raises(ValueError, match='cpu_packed_backend'):
        Qwen3NextRecurrentMixerInference.from_index(index, 0, 'auto')
    assert reads == []


@pytest.mark.parametrize('dtype', [np.float16, np.float64, np.int32])
def test_hidden_dtype_fails_clearly_before_projection_reads(dtype):
    index, _, reads = _fixture()
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    before = reads.copy()
    hidden = np.ones((1, 2, mixer.spec.d_model), dtype=dtype)
    state = mixer.create_state(1)
    with pytest.raises(TypeError, match='float32 hidden only'):
        mixer.forward(hidden, state)
    with pytest.raises(TypeError, match='float32 hidden only'):
        mixer.step(hidden[:, 0], state)
    assert reads == before
    assert state.length == 0


@pytest.mark.parametrize('kind', ['F32', 'F16', 'BF16'])
def test_dense_projection_loading_is_lazy_and_converts_to_fp32(kind):
    index, weights, reads = _fixture()
    for suffix, value in weights.items():
        tensor = index.tensors['blk.0.' + suffix]
        tensor.tensor_type.name = kind
        if kind == 'F16':
            tensor._data = value.astype(np.float16)
            weights[suffix] = tensor._data.astype(np.float32)
        elif kind == 'BF16':
            words = (value.view(np.uint32) >> 16).astype('<u2')
            tensor._data = words.view(np.uint8).reshape(-1)
            weights[suffix] = (words.astype(np.uint32) << 16).view(np.float32).reshape(value.shape)
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    assert mixer.index is index
    assert set(reads) == {'blk.0.' + s for s in (
        'ssm_a', 'ssm_dt.bias', 'ssm_conv1d.weight', 'ssm_norm.weight')}
    assert len(reads) == 4
    assert all(w is None for w in mixer.projection_weights.values())
    assert mixer.ssm_a.dtype == mixer.dt_bias.dtype == np.float32
    hidden = np.random.default_rng(8).normal(size=(1, 3, mixer.spec.d_model)).astype(np.float32)
    expected, _, _ = _reference(hidden, weights, mixer.spec)
    actual = mixer.forward(hidden, mixer.create_state(1))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=5e-6)
    assert len(reads) == 8
    assert all(w.dtype == np.float32 for w in mixer.projection_weights.values())
    mixer.forward(hidden, mixer.create_state(1))
    assert len(reads) == 8


@pytest.mark.parametrize('backend', ['reference', 'native'])
def test_packed_weights_remain_views_and_dispatch_without_full_decode(monkeypatch, backend):
    index, weights, reads = _fixture(packed=True)

    def no_decode(*args, **kwargs):
        raise AssertionError('Adapter must not expand packed projections')

    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', no_decode)
    monkeypatch.setattr(adapter, 'packed_to_device', no_decode)
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0, backend)
    assert mixer.index is index
    for suffix, weight in mixer.projection_weights.items():
        assert isinstance(weight, PackedGGUFWeight)
        assert np.shares_memory(weight.data, index.tensors['blk.0.' + suffix]._data)
    calls = []

    def packed_linear(hidden, weight, *, cpu_packed_backend):
        suffix = next(s for s, w in mixer.projection_weights.items() if w is weight)
        calls.append((suffix, cpu_packed_backend))
        assert hidden.flags.c_contiguous and hidden.dtype == np.float32
        return hidden @ weights[suffix].T

    monkeypatch.setattr(adapter, 'quantized_linear', packed_linear)
    hidden = np.random.default_rng(5).normal(size=(1, 3, mixer.spec.d_model)).astype(np.float32)
    expected, _, _ = _reference(hidden, weights, mixer.spec)
    actual = mixer.forward(hidden, mixer.create_state(1))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=5e-6)
    assert calls == [(suffix, backend) for suffix in adapter._PROJECTIONS]
    assert mixer._device_projections == {}
    assert len(reads) == 8  # Four packed view accesses plus four small decodes.


def test_real_reference_packed_projection_path():
    index, weights, _ = _fixture(packed=True)
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, 0)
    assert mixer.cpu_packed_backend == 'reference'
    hidden = np.random.default_rng(9).normal(size=(1, 2, mixer.spec.d_model)).astype(np.float32)
    expected, _, _ = _reference(hidden, weights, mixer.spec)
    np.testing.assert_allclose(mixer.forward(hidden, mixer.create_state(1)), expected,
                               rtol=3e-4, atol=5e-6)
