"""CPU FP64 full-layer oracles over synthetic GGUF; no checkpoint/CUDA required."""

import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() != 'numpy':
    pytest.skip('Full-layer oracle fixtures are CPU-only', allow_module_level=True)

from mini_llm.diagnostics.routing_trace import RoutingTrace
from mini_llm.external.qwen3next_decoder import Qwen3NextDecoderLayerInference
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.external.qwen3next_moe import Qwen3NextMoEInference, Qwen3NextMoESpec
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor

ROOT = Path(__file__).resolve().parents[1]
MIXER_REFERENCE = runpy.run_path(str(ROOT / 'tests/test_0106m_qwen3next_mixer.py'))
ATTENTION_REFERENCE = runpy.run_path(str(ROOT / 'tests/test_0106n_qwen3next_attention.py'))
Field, Tensor = ATTENTION_REFERENCE['Field'], ATTENTION_REFERENCE['Tensor']


def fixture(*, packed_shared=False, fused=False):
    """Complete four-layer hybrid headers, real packed routed banks, observable reads."""
    mi, mw, _ = MIXER_REFERENCE['_fixture'](packed=True)
    ai, aw, _ = ATTENTION_REFERENCE['_fixture'](packed=True)
    fields = dict(mi.reader.fields, **ai.reader.fields)
    fields.update({name: Field(value) for name, value in {
        'qwen3next.expert_count': 4, 'qwen3next.expert_used_count': 2,
        'qwen3next.expert_feed_forward_length': 32,
        'qwen3next.expert_shared_feed_forward_length': 64,
        # Deliberately different: this is NOT either individual expert width.
        'qwen3next.feed_forward_length': 128,
    }.items()})
    reader = SimpleNamespace(fields=fields, tensors=[])
    reads, weights = [], {}
    rng = np.random.default_rng(10615)

    def add(layer, suffix, value, packed=False):
        if packed:
            # A separate scalar decoder below checks outputs using these exact bytes.
            q = rng.integers(-7, 8, (value.size // 32, 32), dtype=np.int8)
            scales = np.full((len(q), 1), .0078125, dtype='<f2')
            raw = np.concatenate((scales.view(np.uint8), q.view(np.uint8)), axis=1).reshape(-1)
            value = (scales.astype(np.float32) * q).reshape(value.shape)
            payload, kind = raw, 'Q8_0'
        else:
            payload, kind = value.copy(), 'F32'
        reader.tensors.append(Tensor(f'blk.{layer}.' + suffix, value.shape, payload, reads, kind))
        return value

    for layer in range(4):
        # Earlier single-mixer fixtures have deliberately large packed weights.
        # Bound residual growth when composing four layers in FP32 vs FP64.
        mixer = {suffix: value * np.float32(1 / 16)
                 if suffix in ('ssm_out.weight', 'attn_output.weight') else value
                 for suffix, value in (aw if layer == 3 else mw).items()}
        for suffix, value in mixer.items():
            # Mixer reference weights were already decoded from the earlier fixture.
            add(layer, suffix, value)
        ff = {}
        shapes = {
            'attn_norm.weight': (32,), 'post_attention_norm.weight': (32,),
            'ffn_gate_inp.weight': (4, 32), 'ffn_gate_inp_shexp.weight': (32,),
            'ffn_gate_shexp.weight': (64, 32), 'ffn_up_shexp.weight': (64, 32),
            'ffn_down_shexp.weight': (32, 64), 'ffn_down_exps.weight': (4, 32, 32),
        }
        if fused:
            shapes['ffn_gate_up_exps.weight'] = (4, 64, 32)
        else:
            shapes['ffn_gate_exps.weight'] = shapes['ffn_up_exps.weight'] = (4, 32, 32)
        for suffix, shape in shapes.items():
            value = rng.normal(0, .12, shape).astype(np.float32)
            if suffix.endswith('norm.weight'):
                value = np.linspace(.65, 1.45, 32, dtype=np.float32)
            packed = '_exps.weight' in suffix or (packed_shared and '_shexp.weight' in suffix
                                                  and 'gate_inp' not in suffix)
            ff[suffix] = add(layer, suffix, value, packed)
        if fused:
            ff['ffn_gate_exps.weight'], ff['ffn_up_exps.weight'] = np.split(
                ff['ffn_gate_up_exps.weight'], 2, axis=1)
        weights[layer] = dict(mixer=mixer, ffn=ff)
    index = Qwen3NextGGUFIndex(reader)
    assert reads == []
    return index, weights, reads


def executor(index, *, backend='reference', workers=0, profiler=None):
    return HeterogeneousExpertExecutor(index.provider, model_id='0106o-synthetic',
        d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
        device_budget_bytes=0, cpu_packed_backend=backend, cpu_workers=workers,
        routing_profiler=profiler)


def rms(x, gamma, eps):
    value = x.astype(np.float64)
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps) * gamma


def reference_moe(hidden, weights, k=2):
    """Independent FP64 per-token/expert accumulation; no production dispatcher."""
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    x = hidden.astype(np.float64)
    logits = x @ w['ffn_gate_inp.weight'].T
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    ids = np.argsort(-logits, axis=-1)[..., :k]
    selected = np.take_along_axis(probs, ids, -1)
    selected /= selected.sum(axis=-1, keepdims=True)
    result = np.zeros_like(x)
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            row = x[b, t]
            for slot, expert_id in enumerate(ids[b, t]):
                g = w['ffn_gate_exps.weight'][expert_id] @ row
                u = w['ffn_up_exps.weight'][expert_id] @ row
                out = w['ffn_down_exps.weight'][expert_id] @ (g / (1 + np.exp(-g)) * u)
                result[b, t] += selected[b, t, slot] * out
            g = w['ffn_gate_shexp.weight'] @ row
            u = w['ffn_up_shexp.weight'] @ row
            out = w['ffn_down_shexp.weight'] @ (g / (1 + np.exp(-g)) * u)
            scalar = float(w['ffn_gate_inp_shexp.weight'] @ row)
            result[b, t] += out * np.exp(-np.logaddexp(0, -scalar))
    return result, ids, selected


def reference_layer(hidden, weights, layer):
    ff = weights['ffn']
    normalized = rms(hidden, ff['attn_norm.weight'], layer.mixer.spec.rms_eps)
    oracle = ATTENTION_REFERENCE['_reference'] if layer.is_attention else MIXER_REFERENCE['_reference']
    mixed, a, b = oracle(normalized, weights['mixer'], layer.mixer.spec)
    residual = hidden.astype(np.float64) + mixed
    normalized = rms(residual, ff['post_attention_norm.weight'], layer.mixer.spec.rms_eps)
    moe, _, _ = reference_moe(normalized, ff)
    return residual + moe, a, b


@pytest.mark.parametrize('layer_id', [0, 3])
@pytest.mark.parametrize('packed_shared', [False, True])
@pytest.mark.parametrize('backend', ['reference', 'native'])
def test_full_layer_independent_fp64_and_streaming_nonzero_state(layer_id, packed_shared, backend):
    index, weights, _ = fixture(packed_shared=packed_shared)
    run = executor(index, backend=backend)
    try:
        layer = Qwen3NextDecoderLayerInference.from_index(index, layer_id,
            expert_executor=run, cpu_packed_backend=backend)
        hidden = np.random.default_rng(30).normal(0, .2, (2, 7, 32)).astype(np.float32)
        expected, ref_a, ref_b = reference_layer(hidden, weights[layer_id], layer)
        initial = layer.create_state(2, 9)
        prefix_output = layer.forward(hidden[:, :2], initial)
        np.testing.assert_allclose(prefix_output, expected[:, :2], rtol=3e-4, atol=6e-6)
        for sizes in ((5,), (2, 0, 3), (1,) * 5):
            state = layer.copy_state(initial)
            chunks, start = [], 2
            for size in sizes:
                chunks.append(layer.forward(hidden[:, start:start + size], state))
                start += size
            actual = np.concatenate(chunks, axis=1)
            np.testing.assert_allclose(actual, expected[:, 2:], rtol=3e-4, atol=6e-6)
            native = state.mixer_state
            if layer.is_attention:
                np.testing.assert_allclose(native.k_cache[:, :, :7], ref_a, rtol=3e-4, atol=6e-6)
                np.testing.assert_allclose(native.v_cache[:, :, :7], ref_b, rtol=3e-4, atol=6e-6)
            else:
                np.testing.assert_allclose(native.conv.history, ref_a, rtol=3e-4, atol=6e-6)
                np.testing.assert_allclose(native.delta.memory, ref_b, rtol=3e-4, atol=6e-6)
            assert state.length == 7 and initial.length == 2
        np.testing.assert_array_equal(hidden, np.random.default_rng(30).normal(0, .2, (2, 7, 32)).astype(np.float32))
    finally:
        run.close()


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('workers', [0, 2])
def test_moe_exact_ids_normalization_shared_gate_and_existing_trace(fused, workers):
    index, weights, reads = fixture(packed_shared=True, fused=fused)
    trace = RoutingTrace(model_id='0106o-synthetic', quantization='Q8_0', expert_mapping='identity',
                         workload='synthetic-layer', mode='detailed')
    run = executor(index, workers=workers, profiler=trace)
    try:
        moe = Qwen3NextMoEInference.from_index(index, 0, expert_executor=run)
        assert not any('_exps.weight' in name for name in reads)
        assert not moe.projection_weights
        hidden = np.random.default_rng(77).normal(0, .3, (2, 5, 32)).astype(np.float32)
        expected, ids, weights_ref = reference_moe(hidden, weights[0]['ffn'])
        actual_weights, actual_ids = moe.router.forward(hidden)
        np.testing.assert_array_equal(actual_ids, ids)
        np.testing.assert_allclose(actual_weights, weights_ref, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(moe.forward(hidden), expected, rtol=3e-5, atol=2e-6)
        report = trace.snapshot()
        assert report['tokens_by_layer'] == {'0': 10}
        assert report['counters']['routed_assignments'] == 20
        np.testing.assert_array_equal(report['routes'][0]['expert_ids'], ids.reshape(-1, 2))
        np.testing.assert_allclose(report['routes'][0]['routing_weights'], weights_ref.reshape(-1, 2),
                                   rtol=2e-6, atol=2e-7)
        assert all(record['path'] == 'cpu' for record in report['executions'])
        # The shared scalar gate is neither SiLU nor an extra normalized route.
        moe.shared_gate.fill(0)
        weights[0]['ffn']['ffn_gate_inp_shexp.weight'] = np.zeros(32, np.float32)
        expected, _, _ = reference_moe(hidden, weights[0]['ffn'])
        np.testing.assert_allclose(moe.forward(hidden), expected, rtol=3e-5, atol=2e-6)
    finally:
        run.close()


def test_four_layer_hybrid_chain_full_chunk_step_and_fp64():
    index, weights, _ = fixture(packed_shared=True)
    run = executor(index, backend='native', workers=2)
    try:
        layers = [Qwen3NextDecoderLayerInference.from_index(index, i, expert_executor=run,
                  cpu_packed_backend='native') for i in range(4)]
        hidden = np.random.default_rng(91).normal(0, .2, (2, 5, 32)).astype(np.float32)
        expected = hidden
        for layer in layers:
            expected, _, _ = reference_layer(expected, weights[layer.layer_id], layer)
        for sizes in ((5,), (2, 3), (1,) * 5):
            states = [layer.create_state(2, 7) for layer in layers]
            output, start = [], 0
            for size in sizes:
                x = hidden[:, start:start + size]
                for layer, state in zip(layers, states):
                    x = layer.forward(x, state)
                output.append(x)
                start += size
            np.testing.assert_allclose(np.concatenate(output, axis=1), expected, rtol=4e-4, atol=8e-6)
            assert all(state.length == 5 for state in states)
    finally:
        run.close()


@pytest.mark.parametrize('layer_id', [0, 3])
def test_state_reset_copy_empty_capacity_and_layer_ownership(layer_id):
    index, _, reads = fixture()
    run = executor(index)
    try:
        layer = Qwen3NextDecoderLayerInference.from_index(index, layer_id, expert_executor=run)
        state = layer.create_state(1, 3)
        hidden = np.ones((1, 2, 32), np.float32) * .1
        output = layer.forward(hidden, state)
        copied = layer.copy_state(state)
        before = reads.copy()
        assert layer.forward(hidden[:, :0], state).shape == (1, 0, 32)
        with pytest.raises(ValueError, match='capacity'):
            layer.forward(hidden, state)
        assert state.length == copied.length == 2 and reads == before
        with pytest.raises(TypeError, match='float32'):
            layer.forward(hidden.astype(np.float16), state)
        with pytest.raises(ValueError, match='different decoder layer'):
            wrong = layer.create_state(1, 3)
            wrong.layer_id = 9
            layer.forward(hidden, wrong)
        state.reset()
        assert state.length == 0 and copied.length == 2
        np.testing.assert_array_equal(layer.forward(hidden, state), output)
        state.reset()
        np.testing.assert_allclose(layer.step(hidden[:, 0], state), output[:, 0], rtol=3e-4, atol=6e-6)
        if not layer.is_attention:
            with pytest.raises(ValueError, match='RoPE'):
                layer.forward(hidden[:, :1], state, position_ids=np.array([0]))
    finally:
        run.close()


@pytest.mark.parametrize('mutation,error', [
    ('missing_shared', ValueError), ('norm_shape', ValueError),
    ('shared_shape', ValueError), ('iq_format', NotImplementedError),
    ('router_bias', NotImplementedError), ('routing_metadata', NotImplementedError),
    ('expert_width', ValueError), ('missing_shared_metadata', ValueError),
])
def test_unsupported_contracts_fail_before_payload_reads(mutation, error):
    index, _, reads = fixture()
    if mutation == 'missing_shared':
        del index.tensors['blk.0.ffn_gate_inp_shexp.weight']
    elif mutation == 'norm_shape':
        index.tensors['blk.0.attn_norm.weight'].shape = (31,)
    elif mutation == 'shared_shape':
        index.tensors['blk.0.ffn_up_shexp.weight'].shape = (32, 63)
    elif mutation == 'iq_format':
        index.tensors['blk.0.ffn_down_exps.weight'].tensor_type.name = 'IQ1_M'
    elif mutation == 'router_bias':
        index.tensors['blk.0.ffn_gate_inp.bias'] = Tensor('blk.0.ffn_gate_inp.bias', (4,), None, reads)
    elif mutation == 'routing_metadata':
        index.reader.fields['qwen3next.expert_weights_scale'] = Field(2.)
    elif mutation == 'expert_width':
        index.reader.fields['qwen3next.expert_feed_forward_length'] = Field(128)
    else:
        del index.reader.fields['qwen3next.expert_shared_feed_forward_length']
    run = executor(index)
    try:
        with pytest.raises(error):
            Qwen3NextDecoderLayerInference.from_index(index, 0, expert_executor=run)
        assert reads == []
    finally:
        run.close()


@pytest.mark.parametrize('adapter,layer_id', [
    (Qwen3NextMoEInference, 0), (Qwen3NextDecoderLayerInference, 0),
    (Qwen3NextDecoderLayerInference, 3),
])
def test_executor_precision_and_dimensions_and_spec_width(adapter, layer_id):
    index, _, reads = fixture()
    spec = Qwen3NextMoESpec.from_index(index, 0)
    assert (spec.d_ff, spec.shared_d_ff) == (32, 64)
    run = executor(index)
    try:
        run.compute_dtype = 'float16'
        with pytest.raises(ValueError, match='float32 expert compute'):
            adapter.from_index(index, layer_id, expert_executor=run)
        run.compute_dtype, run.d_ff = 'float32', 128
        with pytest.raises(ValueError, match='dimensions'):
            adapter.from_index(index, layer_id, expert_executor=run)
        assert reads == []
    finally:
        run.close()
