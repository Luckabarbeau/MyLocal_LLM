"""0083F: 1024-wide, depth-scaled read-only historical hybrid AR family.

Production geometry is tested without allocating 227–323M full-size weights.
Small NumPy variants exercise the actual historical reader, loss-facing hidden
states, causal attention, backward, and gradient checkpoint replay.
"""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.config import LocalAttentionConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.historical_schedule import historical_slots, source_indices, sweep_coverage
from train_model import model_config_from_name, parse_args


FAMILY = (
    (16, 226_977_060, (3, 5, 7, 9, 11, 12)),
    (20, 275_219_748, (4, 6, 9, 12, 14, 16)),
    (22, 299_341_092, (4, 7, 10, 13, 16, 18)),
    (24, 323_462_436, (4, 8, 12, 16, 18, 20)),
)


def preset_name(depth):
    return f'ensemble-1024-{depth}l-dense-historical-hybrid-64k'


def small_variant(depth):
    base = model_config_from_name(preset_name(depth))
    return replace(
        base, d_model=128, d_head=8, tokenizer_vocab_size=64,
        d_ff=64, dtype='float32',
        memory_context=replace(
            base.memory_context, memory_length=264, recent_length=8,
            target_length=8, router_query_length=8,
        ),
    )


@pytest.mark.parametrize('depth,expected,learned', FAMILY)
def test_exact_production_geometry_and_cli(monkeypatch, depth, expected, learned):
    name = preset_name(depth)
    monkeypatch.setattr('sys.argv', ['train_model.py', '--model', name])
    assert parse_args().model == name
    cfg = model_config_from_name(name)
    assert (cfg.n_layers, cfg.d_model, cfg.d_ff, cfg.n_q_heads,
            cfg.n_kv_heads, cfg.d_head) == (depth, 1024, 3072, 16, 4, 64)
    assert cfg.estimated_parameter_count() == expected
    assert cfg.ffn_mode == 'dense' and cfg.tie_embeddings
    assert cfg.n_experts == cfg.top_k == 1
    memory = cfg.memory_context
    assert memory.integration_mode == 'historical_read'
    assert (cfg.vocab_size, memory.memory_length, memory.target_length,
            memory.distant_memory_length) == (32768, 65536, 4096, 61440)
    assert (memory.block_size, memory.top_k_blocks) == (32, 8)
    assert len(cfg.attention_layers) == depth
    for layer in cfg.attention_layers:
        assert len(layer.heads) == 16
        assert all(isinstance(h, LocalAttentionConfig) for h in layer.heads)
        assert [h.window for h in layer.heads] == [4096] + [512] * 15


@pytest.mark.parametrize('depth,expected,learned', FAMILY)
def test_historical_schedule_exact_coverage_and_masks(depth, expected, learned):
    slots = historical_slots(depth, 16)
    assert len(slots) == depth
    assert sum(s.kind == 'block' for row in slots for s in row) == 12
    assert sum(s.kind == 'sparse' for row in slots for s in row) == 12
    assert sum(s.kind == 'learned' for row in slots for s in row) == 12
    assert {s.layer for row in slots for s in row if s.kind == 'learned'} == set(learned)
    assert max(learned) <= depth - 4  # three subsequent current integration layers
    assert {s.head for row in slots for s in row if s.kind == 'block'} == {10, 11}
    assert {s.head for row in slots for s in row if s.kind == 'sparse'} == {12, 13}
    assert {s.head for row in slots for s in row if s.kind == 'learned'} == {14, 15}
    for row in slots:
        assert len({s.head for s in row}) == len(row)
        assert all(0 < s.head < 16 for s in row)
        for slot in row:
            if slot.kind == 'block':
                assert slot.period == 384
                assert len(source_indices(slot, 61440)) == 5120
            elif slot.kind == 'sparse':
                assert len(source_indices(slot, 61440)) == 480
    seen = sweep_coverage(depth, 16)
    assert len(seen) == 61440
    assert sorted(seen) == list(range(61440))
    assert set(Counter(seen).values()) == {1}
    for first, last in ((0, 37), (3, 31), (37, 515), (190, 1025), (61430, 61440)):
        assert sorted(sweep_coverage(depth, 16, history_length=last, valid_start=first)) == list(range(first, last))


@pytest.mark.parametrize('depth,expected,learned', FAMILY)
def test_numpy_forward_backward_no_recomputation(monkeypatch, depth, expected, learned):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', '0')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '0')
    cfg = small_variant(depth)
    model = DecoderLanguageModel(cfg, rng_seed=200 + depth)
    assert sum(p.size for p in model.parameters()) == cfg.estimated_parameter_count()
    assert all(not b.attention._retrieval_groups for b in model.blocks)
    assert all({(g['kind'], g['window']) for g in b.attention._static_head_groups}
               == {('local', 4096), ('local', 512)} for b in model.blocks)
    ids = np.random.default_rng(300 + depth).integers(0, 64, size=(1, 264))
    hidden, cache = model.forward_body(
        ids, activation_checkpoint=False,
        memory_metadata={'history_valid_starts': np.array([32])},
    )
    assert hidden.shape == (1, 8, 128)
    assert cache['activation_checkpoint'] == 'none'
    assert cache['memory_context_cache']['state'].encoder_cache is not None
    model.backward_body(np.ones_like(hidden) * 1e-4, cache)
    for p in (model.historical_memory.in_proj, model.historical_memory.router_key,
              model.blocks[0].attention.Wq, model.blocks[-1].moe.expert.W_gate):
        assert np.isfinite(p.grad).all() and np.linalg.norm(p.grad) > 0


@pytest.mark.parametrize('depth', (16, 24))
@pytest.mark.parametrize('history_recompute', (0, 1))
def test_block_checkpoint_backward_with_optional_history_replay(monkeypatch, depth, history_recompute):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', str(history_recompute))
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '1')
    model = DecoderLanguageModel(small_variant(depth), rng_seed=400 + depth)
    ids = np.random.default_rng(500 + depth).integers(0, 64, size=(1, 264))
    out, cache = model.forward_body(
        ids, activation_checkpoint=True,
        memory_metadata={'history_valid_starts': np.array([32])},
    )
    assert out.shape == (1, 8, 128)
    assert cache['activation_checkpoint'] == 'block'
    assert len(cache['block_checkpoints']) == depth
    assert (cache['memory_context_cache']['state'].encoder_cache is None) == bool(history_recompute)
    model.backward_body(np.ones_like(out) * 1e-4, cache)
    assert np.linalg.norm(model.historical_memory.in_proj.grad) > 0
    assert np.linalg.norm(model.blocks[-1].attention.Wq.grad) > 0


def test_future_token_invariance_20l():
    model = DecoderLanguageModel(small_variant(20), rng_seed=609)
    ids = np.random.default_rng(610).integers(0, 64, size=(1, 264))
    later = ids.copy()
    later[:, -3:] = (later[:, -3:] + 11) % 64
    meta = {'history_valid_starts': np.array([32])}
    a = model.forward_body(ids, memory_metadata=meta, return_cache=False)
    b = model.forward_body(later, memory_metadata=meta, return_cache=False)
    np.testing.assert_allclose(a[:, :-3], b[:, :-3], atol=1e-6, rtol=1e-6)


def test_checkpointed_noncheckpointed_gradient_parity_20l(monkeypatch):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', '1')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '1')
    ids = np.random.default_rng(611).integers(0, 64, size=(1, 264))
    outs = []
    grads = []
    for use_checkpoint in (False, True):
        model = DecoderLanguageModel(small_variant(20), rng_seed=612)
        out, cache = model.forward_body(
            ids, activation_checkpoint=use_checkpoint,
            memory_metadata={'history_valid_starts': np.array([32])},
        )
        model.backward_body(np.ones_like(out)*1e-4, cache)
        outs.append(out)
        grads.append(tuple(p.grad.copy() for p in (
            model.historical_memory.in_proj, model.historical_memory.router_key,
            model.blocks[8].attention.Wq, model.blocks[-1].moe.expert.W_gate,
        )))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-6, rtol=1e-6)
    for a, b in zip(grads[0], grads[1]):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-4)


def test_legacy_preset_counts_unchanged():
    assert ModelConfig.ensemble_intermediate_768_20l_dense_historical_hybrid_64k().estimated_parameter_count() == 161_274_916
    assert ModelConfig.ensemble_wide_768_dense_historical_hybrid_64k().estimated_parameter_count() == 106_115_620
    assert ModelConfig.ensemble_deep_512_dense_historical_hybrid_64k().estimated_parameter_count() == 105_800_488
    assert ModelConfig.ensemble_large_1536_36l_dense_historical_hybrid_64k().estimated_parameter_count() == 1_197_578_024
    assert ModelConfig.ensemble_max_1792_32l_dense_historical_hybrid_64k().estimated_parameter_count() == 1_373_251_624


def test_20l_optimizer_update_includes_historical_parameters(monkeypatch):
    from mini_llm.optim.hybrid_muon import HybridMuonAdamW
    monkeypatch.setenv('MINI_LLM_OPTIMIZER_OFFLOAD', 'none')
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', '0')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '0')
    model = DecoderLanguageModel(small_variant(20), rng_seed=725)
    ids = np.random.default_rng(726).integers(0, 64, size=(1, 264))
    out, cache = model.forward_body(
        ids, memory_metadata={'history_valid_starts': np.array([32])},
    )
    model.backward_body(np.ones_like(out)*1e-4, cache)
    hist = model.historical_memory.in_proj
    attn = model.blocks[0].attention.Wq
    before = hist.data.copy(), attn.data.copy()
    opt = HybridMuonAdamW(
        model.parameters(), lr=1e-3, beta1=0.9, beta2=0.95,
        eps=1e-8, weight_decay=0.01, muon_momentum=0.95, muon_ns_steps=5,
    )
    assert {hist.name, attn.name} <= {p.name for p in opt.parameters}
    opt.step()
    assert opt.step_index == 1
    assert not np.array_equal(hist.data, before[0])
    assert not np.array_equal(attn.data, before[1])


def test_22l_checkpoint_roundtrip(tmp_path):
    from mini_llm.checkpoint import save_checkpoint, load_checkpoint
    model = DecoderLanguageModel(small_variant(22), rng_seed=812)
    original = {p.name: p.data for p in model.parameters()}
    assert 'embedding.W' in original and 'output_proj.W' not in original
    save_checkpoint(tmp_path/'family1024', original)
    restored, _, _ = load_checkpoint(tmp_path/'family1024', skip_optimizer=True)
    assert set(original) == set(restored)
    for name, array in original.items():
        np.testing.assert_array_equal(array, restored[name])
