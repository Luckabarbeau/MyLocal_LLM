"""P2: exact training/causality and shared-resident K/V gradient tests."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.config import (ModelConfig, AttentionLayerConfig,
                             LocalAttentionConfig, DilatedAttentionConfig)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.ops.shared_indexed_kv import SharedIndexedKV


def make_trainer(sparse=False, layers=2, block_size=4):
    cfg = ModelConfig(tokenizer_vocab_size=32, context_length=96, n_layers=layers,
                      d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
                      n_experts=2, top_k=1, d_ff=32, dtype='float64')
    if sparse:
        layer = AttentionLayerConfig(heads=(LocalAttentionConfig(window=8),
                                            DilatedAttentionConfig(window=48, dilation=3)))
        cfg = replace(cfg, attention_layers=tuple(layer for _ in range(layers)))
    return WindowRefinementTrainer(DecoderLanguageModel(cfg, rng_seed=912, dtype='float64'),
                                   RefinementReferenceConfig(block_size=block_size,
                                                             compact_recurrent_tape=True,
                                                             enable_shared_context_training=True),
                                   rng_seed=917)


def run(t, tokens, mask, *, resident, gradients=True, groups=3):
    t.zero_grad()
    metrics = t.train_window(xp.asarray(tokens), prefix_length=12,
                             valid_token_mask=xp.asarray(mask), group_size=groups,
                             max_window_tokens=96, compact_active_blocks=True,
                             compact_history=True, resident_history=resident,
                             max_history_keys=96, record_row_ce=True,
                             compute_gradients=gradients)
    return metrics, [asnumpy(p.grad).copy() for p in t.parameters()]


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('groups', [1, 3, 7])
def test_resident_cache_matches_gathered_loss_and_every_gradient(sparse, groups):
    t = make_trainer(sparse=sparse)
    n=32
    token_ids = (np.arange(2*n).reshape(2,n)*7 % 29 + 2).astype('int64')
    mask = np.arange(n)[None, :] < np.array([32, 27])[:, None]
    token_ids[~mask] = -100
    baseline, ref_grads = run(t, token_ids, mask, resident=False, groups=groups)
    resident, got_grads = run(t, token_ids, mask, resident=True, groups=groups)
    for label in ('total_loss', 'commitment_ce', 'initial_ce'):
        np.testing.assert_allclose(float(resident[label]), float(baseline[label]), atol=2e-10)
    for label in ('row_commitment_ce_sums','initial_position_ce_sums',
                  'commitment_position_ce_sums','position_counts'):
        np.testing.assert_allclose(asnumpy(resident[label]), asnumpy(baseline[label]), atol=2e-10)
    for ref,g,p in zip(ref_grads,got_grads,t.parameters()):
        np.testing.assert_allclose(g, ref, atol=2e-8, rtol=2e-8, err_msg=p.name)
    assert resident['resident_history']
    assert resident['avoided_group_history_copy_bytes'] > 0
    assert baseline['avoided_group_history_copy_bytes'] == 0
    assert resident['supervised_tokens'] == baseline['supervised_tokens']
    assert resident['refinement_passes'] == baseline['refinement_passes']


def test_forward_only_is_unchanged_and_does_not_touch_gradients():
    t = make_trainer()
    ids = (np.arange(2*28).reshape(2,28)*3%29+2).astype('int64')
    mask = np.ones(ids.shape, bool)
    m_ref, _ = run(t, ids, mask, resident=False, gradients=False)
    m_p2, grads = run(t, ids, mask, resident=True, gradients=False)
    np.testing.assert_allclose(float(m_ref['commitment_ce']),float(m_p2['commitment_ce']),atol=1e-12)
    assert all(np.all(grad==0) for grad in grads)


def test_shared_kv_gather_mixed_history_and_provisional_keys():
    h = xp.asarray(np.arange(2*5*1*3).reshape(2,5,1,3), dtype='float64')
    g = xp.zeros(h.shape, dtype='float64')
    row_doc = xp.asarray([1,0], dtype=xp.int64)
    history = xp.asarray([[2,4],[1,3]], dtype=xp.int64)
    active = xp.asarray(100+np.arange(2*2*1*3).reshape(2,2,1,3),dtype='float64')
    view = SharedIndexedKV(h,g,row_doc,history).with_active(active)
    indices = xp.asarray([[[[0,2,1,3]]],[[[3,0,2,1]]]],dtype=xp.int64)
    out=view.gather(xp.arange(2,dtype=xp.int64)[:,None,None,None],
                    indices,xp.zeros((1,1,1,1),dtype=xp.int64))
    np.testing.assert_array_equal(asnumpy(out[0,0,0]),
                                  np.stack((asnumpy(h[1,2,0]),asnumpy(active[0,0,0]),
                                            asnumpy(h[1,4,0]),asnumpy(active[0,1,0]))))
    np.testing.assert_array_equal(asnumpy(out[1,0,0]),
                                  np.stack((asnumpy(active[1,1,0]),asnumpy(h[0,1,0]),
                                            asnumpy(active[1,0,0]),asnumpy(h[0,3,0]))))
    assert view.historical is h


def test_resident_history_requires_compact_selection():
    t=make_trainer()
    tokens=xp.asarray(np.arange(24).reshape(1,24)%29+2,dtype=xp.int64)
    with pytest.raises(ValueError, match='requires compact_history'):
        t.train_window(tokens,prefix_length=12,resident_history=True)

@pytest.mark.skipif(__import__('mini_llm.backend', fromlist=['BACKEND_NAME']).BACKEND_NAME != 'cupy',
                    reason='requires CUDA BF16 runtime')
@pytest.mark.parametrize('history_width', [0, 7])
@pytest.mark.parametrize('bias', [False, True])
def test_cuda_shared_kv_forward_backward_matches_physical_gather(history_width, bias):
    """Direct FP32 atomic accumulation, GQA, repeated keys, masks, mixed sources."""
    from mini_llm.backend import resolve_dtype
    from mini_llm.ops.attention_selection import KeySelectionPlan
    from mini_llm.ops.indexed_attention import indexed_attention_forward, indexed_attention_backward
    rng=np.random.default_rng(801)
    bf16=resolve_dtype('bfloat16')
    rows, docs, length, queries, kvheads, qheads, width = 3,2,12,4,2,4,64
    history= xp.asarray(rng.normal(size=(docs,length,kvheads,width)),dtype=bf16)
    values = xp.asarray(rng.normal(size=(docs,length,kvheads,width)),dtype=bf16)
    grad_h=xp.zeros(history.shape,dtype=xp.float32)
    grad_v=xp.zeros(values.shape,dtype=xp.float32)
    doc=xp.asarray([0,1,0],dtype=xp.int64)
    index=xp.asarray(rng.integers(0,length,size=(rows,history_width)),dtype=xp.int64)
    active_k=xp.asarray(rng.normal(size=(rows,queries,kvheads,width)),dtype=bf16)
    active_v=xp.asarray(rng.normal(size=(rows,queries,kvheads,width)),dtype=bf16)
    q=xp.asarray(rng.normal(size=(rows,queries,qheads,width)),dtype=bf16)
    slot_count=history_width+queries
    idx=rng.integers(0,slot_count,size=(rows,1,queries,9))
    idx[...,2]=idx[...,1]
    valid=rng.random(size=idx.shape)>0.2
    valid[:,:,0,:]=False
    logit_bias=xp.asarray(rng.normal(size=idx.shape)*0.01,dtype=xp.float32) if bias else None
    plan=KeySelectionPlan(xp.asarray(idx,dtype=xp.int64),xp.asarray(valid),logit_bias)
    mapping=(0,0,1,1)
    ck=SharedIndexedKV(history,grad_h,doc,index).with_active(active_k)
    cv=SharedIndexedKV(values,grad_v,doc,index).with_active(active_v)
    if history_width:
        material_k=xp.concatenate((history[doc[:,None],index],active_k),axis=1)
        material_v=xp.concatenate((values[doc[:,None],index],active_v),axis=1)
    else:
        material_k,material_v=active_k,active_v
    pred_ref,c_ref=indexed_attention_forward(q,material_k,material_v,plan,
                kv_head_indices=mapping,query_chunk_size=4)
    pred_p2,c_p2=indexed_attention_forward(q,ck,cv,plan,
                kv_head_indices=mapping,query_chunk_size=4)
    np.testing.assert_allclose(asnumpy(pred_ref),asnumpy(pred_p2),atol=1e-4,rtol=1e-4)
    upstream=xp.asarray(rng.normal(size=q.shape),dtype=xp.float32)
    grad_ref=indexed_attention_backward(upstream,c_ref)
    grad_p2=indexed_attention_backward(upstream,c_p2)
    xp.cuda.Stream.null.synchronize()
    expected_k=xp.zeros_like(grad_h)
    expected_v=xp.zeros_like(grad_v)
    if history_width:
        xp.add.at(expected_k,(doc[:,None],index),grad_ref[1][:,:history_width])
        xp.add.at(expected_v,(doc[:,None],index),grad_ref[2][:,:history_width])
    for expected,actual in ((grad_ref[0],grad_p2[0]),
                            (grad_ref[1][:,history_width:],grad_p2[1]),
                            (grad_ref[2][:,history_width:],grad_p2[2]),
                            (expected_k,grad_h),(expected_v,grad_v)):
        np.testing.assert_allclose(asnumpy(actual),asnumpy(expected),rtol=2e-3,atol=8e-4)
    if bias:
        np.testing.assert_allclose(asnumpy(grad_p2[3]),asnumpy(grad_ref[3]),rtol=0,atol=0)

@pytest.mark.skipif(__import__('mini_llm.backend', fromlist=['BACKEND_NAME']).BACKEND_NAME != 'cupy',
                    reason='requires CUDA BF16 runtime')
def test_cuda_full_shared_context_gradient_equivalence():
    from mini_llm.backend import resolve_dtype
    cfg=ModelConfig(tokenizer_vocab_size=64,context_length=48,n_layers=1,
                    d_model=32,n_q_heads=4,n_kv_heads=2,d_head=8,
                    n_experts=2,top_k=1,d_ff=64,dtype='bfloat16')
    t=WindowRefinementTrainer(DecoderLanguageModel(cfg,rng_seed=908,dtype='bfloat16'),
        RefinementReferenceConfig(block_size=4,compact_recurrent_tape=True,
                                  enable_shared_context_training=True),rng_seed=910)
    token_ids=xp.asarray((np.arange(2*32).reshape(2,32)*11%61+2),dtype=xp.int64)
    mask=xp.asarray(np.arange(32)[None,:]<np.array([32,26])[:,None])
    def evaluate(resident):
        t.zero_grad()
        m=t.train_window(token_ids,prefix_length=12,valid_token_mask=mask,
             group_size=3,compact_active_blocks=True,compact_history=True,
             resident_history=resident,max_history_keys=48, max_window_tokens=48)
        return m,[asnumpy(p.grad).copy() for p in t.parameters()]
    ref,ref_grad=evaluate(False)
    opt,opt_grad=evaluate(True)
    np.testing.assert_allclose(float(ref['commitment_ce']),float(opt['commitment_ce']),
                               rtol=3e-3,atol=3e-3)
    for a,b,p in zip(ref_grad,opt_grad,t.parameters()):
        np.testing.assert_allclose(a,b,rtol=4e-2,atol=3e-3,err_msg=p.name)
