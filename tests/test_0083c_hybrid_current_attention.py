"""0083C: hybrid full-current/local-512 attention atop 0083B history.

All checks run on NumPy; GPU BF16/GQA grouped kernels require separate smoke
runs on the production RTX 5080. Layer/head geometry remains production size.
"""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.config import (LocalAttentionConfig, ModelConfig,
                             RetrievalAttentionConfig)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.historical_schedule import historical_slots, sweep_coverage
from mini_llm.ops.indexed_attention import (local_window_attention_forward,
                                             local_window_attention_backward)
from train_model import model_config_from_name


PRESETS = (
    ('ensemble-deep-512-dense-historical-hybrid-64k', 32, 8, 105_800_488),
    ('ensemble-wide-768-dense-historical-hybrid-64k', 13, 12, 106_115_620),
)


@pytest.mark.parametrize('name,n_layers,n_heads,param_count', PRESETS)
def test_exact_new_preset_topology_and_parameter_count(name, n_layers, n_heads, param_count):
    c = model_config_from_name(name)
    assert (c.n_layers, c.n_q_heads) == (n_layers, n_heads)
    assert c.estimated_parameter_count() == param_count
    assert c.memory_context.integration_mode == 'historical_read'
    assert (c.memory_context.target_length, c.memory_context.distant_memory_length) == (4096, 61440)
    assert (c.memory_context.block_size, c.memory_context.top_k_blocks) == (32, 8)
    assert c.tie_embeddings and c.ffn_mode == 'dense'
    slots = historical_slots(n_layers, n_heads)
    assert sorted(sweep_coverage(n_layers, n_heads)) == list(range(61440))
    n_hist = 0
    for layer_idx, layer in enumerate(c.attention_layers):
        assert len(layer.heads) == n_heads
        assert all(isinstance(h, LocalAttentionConfig) for h in layer.heads)
        assert [h.window for h in layer.heads] == [4096] + [512] * (n_heads - 1)
        historical_heads = {s.head for s in slots[layer_idx]}
        assert 0 not in historical_heads, 'full-current head must never be replaced'
        assert len(historical_heads) == len(slots[layer_idx])
        assert (n_heads - len(historical_heads)) >= 1
        n_hist += len(historical_heads)
    assert n_hist == (40 if n_layers == 32 else 36)


@pytest.mark.parametrize('name,n_layers,n_heads,param_count', PRESETS)
def test_instantiated_attention_has_two_static_groups_and_no_current_retrieval(
        name, n_layers, n_heads, param_count):
    c = model_config_from_name(name)
    c = replace(c, tokenizer_vocab_size=128, d_ff=64, dtype='float32',
                memory_context=replace(c.memory_context,
                                       memory_length=264, recent_length=8,
                                       target_length=8, router_query_length=8))
    model = DecoderLanguageModel(c, rng_seed=47)
    assert sum(p.size for p in model.parameters()) == c.estimated_parameter_count()
    for layer_idx, block in enumerate(model.blocks):
        groups = block.attention._static_head_groups
        assert {(g['kind'], g['window']) for g in groups} == {
            ('local', 4096), ('local', 512)
        }
        full = next(g for g in groups if g['window'] == 4096)
        narrow = next(g for g in groups if g['window'] == 512)
        assert full['head_indices'] == (0,)
        assert narrow['head_indices'] == tuple(range(1, n_heads))
        assert block.attention._retrieval_groups == []
        assert all(slot.head != 0 for slot in model.historical_memory.slots[layer_idx])


def test_full_head_reaches_outside_local_512_and_backward_is_causal():
    # Shared single-KV head: only full head sees value at position 0 when
    # querying position 519. Local-512 head must ignore it exactly.
    length = 520
    q = np.zeros((1, length, 2, 8), dtype=np.float32)
    k = np.zeros((1, length, 1, 8), dtype=np.float32)
    v = np.zeros((1, length, 1, 8), dtype=np.float32)
    v[0, 0, 0, 0] = 1.0
    full, full_cache = local_window_attention_forward(
        q[:, :, :1, :], k, v, 4096, kv_head_indices=(0,), return_cache=True,
        query_chunk_size=128,
    )
    local, local_cache = local_window_attention_forward(
        q[:, :, 1:, :], k, v, 512, kv_head_indices=(0,), return_cache=True,
        query_chunk_size=128,
    )
    assert full[0, -1, 0, 0] == pytest.approx(1.0 / length, abs=1e-7)
    assert local[0, -1, 0, 0] == 0
    assert full[0, 0, 0, 0] == pytest.approx(1.0)
    d_out = np.zeros_like(full)
    d_out[0, -1, 0, 0] = 1.0
    _, _, d_v_full = local_window_attention_backward(d_out, full_cache)
    _, _, d_v_local = local_window_attention_backward(d_out, local_cache)
    assert d_v_full[0, 0, 0, 0] == pytest.approx(1.0 / length, abs=1e-7)
    assert d_v_local[0, 0, 0, 0] == 0


@pytest.mark.parametrize('name,n_layers,n_heads,param_count', PRESETS)
@pytest.mark.parametrize('history_recompute', (0, 1))
def test_both_models_forward_backward_with_history(monkeypatch, name, n_layers,
                                                    n_heads, param_count,
                                                    history_recompute):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', str(history_recompute))
    c = model_config_from_name(name)
    c = replace(c, tokenizer_vocab_size=128, d_ff=64, dtype='float32',
                memory_context=replace(c.memory_context,
                                       memory_length=264, recent_length=8,
                                       target_length=8, router_query_length=8))
    m = DecoderLanguageModel(c, rng_seed=103)
    tokens = np.random.default_rng(104).integers(0, 128, (1, 264))
    body, cache = m.forward_body(tokens, memory_metadata={
        'history_valid_starts': np.array([32]),
    })
    assert body.shape == (1, 8, c.d_model)
    # The full head remains a current local-attention group even when every
    # historical head slot is actively replaced by a historical read.
    for i in (2, 5, 6):
        attn_cache = cache['block_caches'][i]['attn_cache']
        local_groups = [g for g in attn_cache['pattern_caches'] if g['kind'] == 'local_window']
        assert any(tuple(g['head_indices']) == (0,) for g in local_groups)
    m.backward_body(np.ones_like(body) * 0.0001, cache)
    assert np.linalg.norm(m.historical_memory.key.grad) > 0
    assert np.linalg.norm(m.historical_memory.in_proj.grad) > 0
    assert np.linalg.norm(m.blocks[0].attention.Wq.grad) > 0
    assert np.isfinite(m.blocks[-1].attention.Wo.grad).all()


def test_future_perturbation_does_not_change_earlier_current_predictions():
    c = model_config_from_name(PRESETS[1][0])
    c = replace(c, tokenizer_vocab_size=128, d_ff=64, dtype='float32',
                memory_context=replace(c.memory_context,
                                       memory_length=264, recent_length=8,
                                       target_length=8, router_query_length=8))
    m = DecoderLanguageModel(c, rng_seed=10)
    tokens = np.random.default_rng(11).integers(0, 128, (1, 264))
    changed = tokens.copy()
    changed[:, -3:] = (changed[:, -3:] + 7) % 128
    meta = {'history_valid_starts': np.array([16])}
    a = m.forward_body(tokens, memory_metadata=meta, return_cache=False)
    b = m.forward_body(changed, memory_metadata=meta, return_cache=False)
    np.testing.assert_allclose(a[:, :-3], b[:, :-3], atol=1e-6, rtol=1e-6)


def test_0083b_presets_not_modified_by_new_preset_builders():
    # Important for existing checkpoints and direct 0083B comparisons.
    for old, hybrid in (
        (ModelConfig.ensemble_deep_512_dense_historical_64k(),
         ModelConfig.ensemble_deep_512_dense_historical_hybrid_64k()),
        (ModelConfig.ensemble_wide_768_dense_historical_64k(),
         ModelConfig.ensemble_wide_768_dense_historical_hybrid_64k()),
    ):
        assert old.attention_layers[0].heads != hybrid.attention_layers[0].heads
        assert any(isinstance(h, RetrievalAttentionConfig)
                   for h in old.attention_layers[0].heads)
        assert old.memory_context == hybrid.memory_context
        assert old.estimated_parameter_count() > hybrid.estimated_parameter_count()


def test_mixed_head_dispatch_full_and_512_share_kv_and_accumulate_gradients():
    """Exercise the actual two-group attention dispatch, not just each kernel."""
    from mini_llm.config import AttentionLayerConfig
    from mini_llm.ops.attention import GQAAttention
    from mini_llm.backend import RandomStream

    attn = GQAAttention(
        d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        input_std=0.02, output_std=0.02, rng=RandomStream(22),
        attention_config=AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=4096), LocalAttentionConfig(window=512),
        )),
    )
    n = 520
    x = np.zeros((1, n, 16), dtype=np.float32)
    q = np.zeros((1, n, 2, 8), dtype=np.float32)
    k = np.zeros((1, n, 1, 8), dtype=np.float32)
    v = np.zeros_like(k)
    v[0, 0, 0, 0] = 1
    context, groups, routes = attn._mixed_context_forward(
        x, q, k, v, return_cache=True,
    )
    assert routes == {}
    assert [g['head_indices'] for g in groups] == [(0,), (1,)]
    assert context[0, -1, 0, 0] == pytest.approx(1/n, abs=1e-7)
    assert context[0, -1, 1, 0] == 0
    dcontext = np.zeros_like(context)
    dcontext[:, -1, :, 0] = 1
    dq, dk, dv, dx = attn._mixed_context_backward(dcontext, {
        'q': q, 'k': k, 'v': v, 'x': x, 'pattern_caches': groups,
    })
    assert dv[0, 0, 0, 0] == pytest.approx(1/n, abs=1e-7)
    assert not np.any(dx)
    assert np.isfinite(dq).all() and np.isfinite(dk).all()


def test_local_probability_recompute_preserves_hybrid_model_gradients(monkeypatch):
    name = PRESETS[1][0]
    c = model_config_from_name(name)
    c = replace(c, tokenizer_vocab_size=128, d_ff=64, dtype='float32',
                memory_context=replace(c.memory_context,
                                       memory_length=264, recent_length=8,
                                       target_length=8, router_query_length=8))
    ids = np.random.default_rng(555).integers(0, 128, (1, 264))
    outputs, gradients = [], []
    for enabled in (0, 1):
        monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', str(enabled))
        m = DecoderLanguageModel(c, rng_seed=556)
        result, cache = m.forward_body(ids, memory_metadata={
            'history_valid_starts': np.array([32]),
        })
        outputs.append(result)
        m.backward_body(np.ones_like(result) * 0.0001, cache)
        gradients.append({p.name: p.grad.copy() for p in m.parameters()})
    np.testing.assert_array_equal(outputs[0], outputs[1])
    for name in gradients[0]:
        np.testing.assert_allclose(gradients[0][name], gradients[1][name],
                                   atol=1e-6, rtol=1e-6, err_msg=name)
