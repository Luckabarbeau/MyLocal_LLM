"""0083D large ensemble configurations and historical-reader contracts.

The production shapes are checked without allocating >1B parameters. Small
widths with the same head/layer topology exercise true forward/backward and
block-checkpoint replay on the NumPy backend.
"""
from dataclasses import replace
from collections import Counter

import numpy as np
import pytest

from mini_llm.config import ModelConfig, LocalAttentionConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.historical_schedule import historical_slots, source_indices, sweep_coverage
from train_model import model_config_from_name


CONFIGS = (
    ("ensemble-large-1536-36l-dense-historical-hybrid-64k", 36, 1536, 24, 6, 5632, 1_197_578_024),
    ("ensemble-max-1792-32l-dense-historical-hybrid-64k", 32, 1792, 28, 7, 6144, 1_373_251_624),
)


@pytest.mark.parametrize('name,L,d,h,kv,ff,p', CONFIGS)
def test_large_preset_exact_geometry(name,L,d,h,kv,ff,p):
    c = model_config_from_name(name)
    assert (c.n_layers,c.d_model,c.n_q_heads,c.n_kv_heads,c.d_ff)==(L,d,h,kv,ff)
    assert c.estimated_parameter_count()==p
    assert c.tokenizer_vocab_size==32768
    assert c.tie_embeddings and c.ffn_mode=='dense'
    assert c.memory_context.integration_mode=='historical_read'
    assert (c.memory_context.memory_length, c.memory_context.target_length) == (65536,4096)
    assert (c.memory_context.block_size,c.memory_context.top_k_blocks)==(32,8)
    assert len(c.attention_layers)==L
    for layer in c.attention_layers:
        assert len(layer.heads)==h
        assert all(isinstance(head,LocalAttentionConfig) for head in layer.heads)
        assert [head.window for head in layer.heads]==[4096]+[512]*(h-1)


@pytest.mark.parametrize('name,L,d,h,kv,ff,p', CONFIGS)
def test_history_exact_coverage_and_reserved_full_head(name,L,d,h,kv,ff,p):
    slots=historical_slots(L,h)
    assert sum(len(layer) for layer in slots)==40
    assert sum(s.kind=='block' for layer in slots for s in layer)==16
    assert sum(s.kind=='sparse' for layer in slots for s in layer)==12
    assert sum(s.kind=='learned' for layer in slots for s in layer)==12
    assert all(s.head!=0 for layer in slots for s in layer)
    positions=sweep_coverage(L,h)
    assert len(positions)==61440
    assert sorted(positions)==list(range(61440))
    assert max(Counter(positions).values())==1
    for layer in slots:
        assert len(set(s.head for s in layer))==len(layer)
        for s in layer:
            if s.kind=='block':
                assert s.period==512
                assert len(source_indices(s,61440))==3840
                assert all(0<=j<61440 for j in source_indices(s,61440))
            if s.kind=='sparse':
                assert len(source_indices(s,61440))==480
    # First valid history position may bisect a 32-token block.
    short=sweep_coverage(L,h,history_length=515,valid_start=37)
    assert sorted(short)==list(range(37,515))


@pytest.mark.parametrize('name,L,d,h,kv,ff,p', CONFIGS)
@pytest.mark.parametrize('history_recompute', (0,1))
def test_small_forward_backward_with_checkpointed_history(monkeypatch,name,L,d,h,kv,ff,p,history_recompute):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE',str(history_recompute))
    base=model_config_from_name(name)
    small=replace(base, d_model=h*8,d_head=8, tokenizer_vocab_size=64,
                  d_ff=64,dtype='float32',
                  memory_context=replace(base.memory_context,
                      memory_length=264, recent_length=8, target_length=8,
                      router_query_length=8))
    model=DecoderLanguageModel(small,rng_seed=37)
    assert sum(param.size for param in model.parameters())==small.estimated_parameter_count()
    tokens=np.random.default_rng(38).integers(0,64,size=(1,264))
    meta={'history_valid_starts':np.array([32])}
    hidden,cache=model.forward_body(tokens,activation_checkpoint=True,memory_metadata=meta)
    assert cache['activation_checkpoint']=='block'
    assert len(cache['block_checkpoints'])==L
    assert hidden.shape==(1,8,h*8)
    model.backward_body(np.ones_like(hidden)*1e-5,cache)
    assert np.linalg.norm(model.historical_memory.in_proj.grad)>0
    assert np.linalg.norm(model.historical_memory.key.grad)>0
    assert np.linalg.norm(model.blocks[-1].attention.Wq.grad)>0
    assert np.isfinite(model.blocks[0].attention.Wo.grad).all()


def test_old_presets_unchanged():
    assert ModelConfig.ensemble_deep_512_dense_historical_hybrid_64k().estimated_parameter_count()==105800488
    assert ModelConfig.ensemble_wide_768_dense_historical_hybrid_64k().estimated_parameter_count()==106115620


@pytest.mark.parametrize('name,L,d,h,kv,ff,p', CONFIGS)
def test_full_history_block_checkpoint_gradient_parity(monkeypatch,name,L,d,h,kv,ff,p):
    monkeypatch.setenv('MINI_LLM_HISTORY_RECOMPUTE','1')
    base=model_config_from_name(name)
    small=replace(base,d_model=h*8,d_head=8,tokenizer_vocab_size=64,
                  d_ff=64,dtype='float32',
                  memory_context=replace(base.memory_context,
                      memory_length=264,recent_length=8,target_length=8,
                      router_query_length=8))
    tokens=np.random.default_rng(73).integers(0,64,size=(1,264))
    meta={'history_valid_starts':np.array([32])}
    outputs=[]
    gradients=[]
    for ckpt in (False,True):
        model=DecoderLanguageModel(small,rng_seed=72)
        y,cache=model.forward_body(tokens,activation_checkpoint=ckpt,memory_metadata=meta)
        model.backward_body(np.ones_like(y)*1e-5,cache)
        outputs.append(y)
        gradients.append((model.historical_memory.in_proj.grad.copy(),
                          model.historical_memory.key.grad.copy(),
                          model.blocks[6].attention.Wq.grad.copy(),
                          model.blocks[-1].moe.expert.W_gate.grad.copy()))
    np.testing.assert_allclose(outputs[0],outputs[1],rtol=1e-6,atol=1e-6)
    for a,b in zip(*gradients):
        np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
