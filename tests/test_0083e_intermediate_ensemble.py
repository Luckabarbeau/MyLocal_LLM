"""0083E: intermediate 20x768 hybrid full/local-512 historical ensemble.

The real 161M parameter count is checked algebraically; small-width NumPy
variants exercise forward/backward without allocating the production model.
"""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.config import LocalAttentionConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.historical_schedule import historical_slots, source_indices, sweep_coverage
from train_model import model_config_from_name, parse_args


PRESET = "ensemble-intermediate-768-20l-dense-historical-hybrid-64k"
PARAMETERS = 161_274_916
LEARNED_LAYERS = (4, 6, 9, 12, 14, 16)


def _small():
    production = model_config_from_name(PRESET)
    memory = replace(production.memory_context, memory_length=264,
                     recent_length=8, target_length=8,
                     router_query_length=8)
    return replace(production, d_model=96, d_head=8,
                   tokenizer_vocab_size=64, d_ff=64, dtype='float32',
                   memory_context=memory)


def test_preset_cli_and_exact_geometry(monkeypatch):
    monkeypatch.setattr('sys.argv', ['train_model.py', '--model', PRESET])
    assert parse_args().model == PRESET
    c = model_config_from_name(PRESET)
    assert (c.n_layers, c.d_model, c.d_ff, c.n_q_heads, c.n_kv_heads) == (20,768,2304,12,3)
    assert c.estimated_parameter_count() == PARAMETERS
    assert c.tie_embeddings and c.ffn_mode == 'dense'
    assert (c.vocab_size, c.memory_context.memory_length,
            c.memory_context.target_length, c.memory_context.distant_memory_length) == (32768,65536,4096,61440)
    assert (c.memory_context.block_size, c.memory_context.top_k_blocks) == (32,8)
    assert c.memory_context.integration_mode == 'historical_read'
    assert len(c.attention_layers) == c.n_layers
    for layer in c.attention_layers:
        assert len(layer.heads) == 12
        assert all(isinstance(h, LocalAttentionConfig) for h in layer.heads)
        assert [h.window for h in layer.heads] == [4096] + [512]*11


def test_layer_head_roles_and_coverage():
    slots = historical_slots(20,12)
    assert len(slots) == 20
    assert sum(s.kind == 'block' for row in slots for s in row) == 12
    assert sum(s.kind == 'sparse' for row in slots for s in row) == 12
    assert sum(s.kind == 'learned' for row in slots for s in row) == 12
    assert {s.layer for row in slots for s in row if s.kind == 'learned'} == set(LEARNED_LAYERS)
    for row in slots:
        assert len({s.head for s in row}) == len(row)
        assert all(0 < s.head < 12 for s in row)
    full = sweep_coverage(20,12)
    assert len(full) == 61440
    assert sorted(full) == list(range(61440))
    assert set(Counter(full).values()) == {1}
    for row in slots:
        for s in row:
            if s.kind == 'block':
                assert s.period == 384
                assert len(source_indices(s, 61440)) == 5120
            if s.kind == 'sparse':
                assert len(source_indices(s, 61440)) == 480
    for start, end in ((0,37),(3,31),(37,515),(190,1025),(61430,61440)):
        assert sorted(sweep_coverage(20,12,history_length=end,valid_start=start)) == list(range(start,end))


def test_forward_and_backward_with_no_recomputation(monkeypatch):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE','0')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS','0')
    config = _small()
    model = DecoderLanguageModel(config, rng_seed=51)
    assert sum(p.size for p in model.parameters()) == config.estimated_parameter_count()
    assert not any(getattr(b.attention,'_retrieval_groups',[]) for b in model.blocks)
    for b in model.blocks:
        assert {(g['kind'],g['window']) for g in b.attention._static_head_groups} == {('local',4096),('local',512)}
    tokens=np.random.default_rng(52).integers(0,64,size=(2,264))
    meta={'history_valid_starts':np.array([32,64])}
    hidden,cache=model.forward_body(tokens, activation_checkpoint=False, memory_metadata=meta)
    assert hidden.shape == (2,8,96)
    assert cache['activation_checkpoint'] == 'none'
    assert cache['memory_context_cache']['state'].encoder_cache is not None
    for i in (2,4,6,9,16):
        local=[g for g in cache['block_caches'][i]['attn_cache']['pattern_caches'] if g['kind']=='local_window']
        assert any(tuple(g['head_indices'])==(0,) for g in local)
    model.backward_body(np.ones_like(hidden)*0.0001,cache)
    for p in (model.historical_memory.in_proj,model.historical_memory.key,
              model.blocks[0].attention.Wq,model.blocks[-1].moe.expert.W_gate):
        assert np.isfinite(p.grad).all() and np.linalg.norm(p.grad)>0


@pytest.mark.parametrize('history_recompute', (False, True))
@pytest.mark.parametrize('checkpoint', (False, True))
def test_recomputation_options_preserve_training_path(monkeypatch, history_recompute, checkpoint):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE',str(int(history_recompute)))
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS','1')
    model=DecoderLanguageModel(_small(),rng_seed=63)
    ids=np.random.default_rng(64).integers(0,64,size=(1,264))
    y,cache=model.forward_body(ids, activation_checkpoint=checkpoint,
                               memory_metadata={'history_valid_starts':np.array([32])})
    assert y.shape==(1,8,96)
    assert (cache['memory_context_cache']['state'].encoder_cache is None) == history_recompute
    assert cache['activation_checkpoint'] == ('block' if checkpoint else 'none')
    model.backward_body(np.ones_like(y)*1e-4,cache)
    assert np.linalg.norm(model.historical_memory.router_key.grad)>0
    assert np.isfinite(model.historical_memory.in_proj.grad).all()


def test_future_token_causality():
    model=DecoderLanguageModel(_small(),rng_seed=70)
    ids=np.random.default_rng(71).integers(0,64,size=(1,264))
    changed=ids.copy()
    changed[:,-3:]=(changed[:,-3:]+7)%64
    meta={'history_valid_starts':np.array([32])}
    y0=model.forward_body(ids, memory_metadata=meta, return_cache=False)
    y1=model.forward_body(changed, memory_metadata=meta, return_cache=False)
    np.testing.assert_allclose(y0[:,:-3],y1[:,:-3],rtol=1e-6,atol=1e-6)


def test_small_checkpoint_roundtrip(tmp_path):
    from mini_llm.checkpoint import save_checkpoint, load_checkpoint
    model=DecoderLanguageModel(_small(),rng_seed=91)
    values={p.name:p.data for p in model.parameters()}
    assert 'embedding.W' in values and 'output_proj.W' not in values
    save_checkpoint(tmp_path/'intermediate',values)
    loaded,_,_=load_checkpoint(tmp_path/'intermediate',skip_optimizer=True)
    assert set(values)==set(loaded)
    for name,arr in values.items():
        np.testing.assert_array_equal(arr,loaded[name])


def test_preexisting_preset_parameter_counts():
    assert ModelConfig.ensemble_wide_768_dense_historical_hybrid_64k().estimated_parameter_count() == 106115620
    assert ModelConfig.ensemble_deep_512_dense_historical_hybrid_64k().estimated_parameter_count() == 105800488
    assert ModelConfig.ensemble_large_1536_36l_dense_historical_hybrid_64k().estimated_parameter_count() == 1197578024
    assert ModelConfig.ensemble_max_1792_32l_dense_historical_hybrid_64k().estimated_parameter_count() == 1373251624


def test_hybrid_muon_optimizer_updates_historical_and_transformer_weights(monkeypatch):
    from mini_llm.optim.hybrid_muon import HybridMuonAdamW
    monkeypatch.setenv('MINI_LLM_OPTIMIZER_OFFLOAD', 'none')
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE', '0')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '0')
    model=DecoderLanguageModel(_small(),rng_seed=123)
    ids=np.random.default_rng(124).integers(0,64,size=(1,264))
    hidden,cache=model.forward_body(ids,memory_metadata={'history_valid_starts':np.array([32])})
    model.backward_body(np.ones_like(hidden)*1e-4,cache)
    hist=model.historical_memory.in_proj
    attn=model.blocks[0].attention.Wq
    h_before=hist.data.copy()
    a_before=attn.data.copy()
    optimizer=HybridMuonAdamW(model.parameters(),lr=1e-3,beta1=0.9,beta2=0.95,
                              eps=1e-8,weight_decay=0.01,
                              muon_momentum=0.95,muon_ns_steps=5)
    assert {hist.name,attn.name} <= {p.name for p in optimizer.parameters}
    optimizer.step()
    assert optimizer.step_index==1
    assert np.isfinite(hist.data).all() and np.isfinite(attn.data).all()
    assert not np.array_equal(hist.data,h_before)
    assert not np.array_equal(attn.data,a_before)


def test_block_checkpoint_output_and_gradient_parity(monkeypatch):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE','1')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS','1')
    inputs=np.random.default_rng(130).integers(0,64,size=(1,264))
    outputs=[]
    gradients=[]
    for checkpoint in (False,True):
        model=DecoderLanguageModel(_small(),rng_seed=131)
        y,cache=model.forward_body(inputs,activation_checkpoint=checkpoint,
                                   memory_metadata={'history_valid_starts':np.array([32])})
        model.backward_body(np.ones_like(y)*1e-4,cache)
        outputs.append(y)
        gradients.append((model.historical_memory.in_proj.grad.copy(),
                          model.historical_memory.router_key.grad.copy(),
                          model.blocks[6].attention.Wq.grad.copy(),
                          model.blocks[-1].moe.expert.W_gate.grad.copy()))
    np.testing.assert_allclose(outputs[0],outputs[1],rtol=1e-6,atol=1e-6)
    for a,b in zip(*gradients):
        np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
