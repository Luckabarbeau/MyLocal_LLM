"""0080BK: tests for shared learned decoder QKV read-only 128-state context."""
import numpy as np
import pytest
from mini_llm.backend import xp, RandomStream
from mini_llm.ops.attention import GQAAttention
from mini_llm.block_proposal.decoder_context_kv import (
    decoder_context_forward, decoder_context_backward,
)


def build_attn(width=16, heads=2):
    return GQAAttention(width,heads,heads,width//heads,
                        .02,.02,RandomStream(505),name='0080bk.testing',dtype='float32')


def local_allow(k):
    # Local PLANS are bidirectional; DEC j sees PLANs and only DEC <= j.
    allow=np.zeros((2*k,2*k), dtype=bool)
    allow[:k,:k]=True
    allow[k:,:k]=True
    allow[k:,k:]=np.tril(np.ones((k,k),dtype=bool))
    return xp.asarray(allow)


def test_shapes_and_mask_invariant():
    rng=np.random.default_rng(81)
    attn=build_attn()
    local=xp.asarray(rng.standard_normal((3,8,16)).astype('float32'))
    mem=xp.asarray(rng.standard_normal((3,12,16)).astype('float32'))
    valid=xp.asarray([[0]*6+[1]*6,[0]*11+[1],[1]*12],dtype=xp.bool_)
    allow=local_allow(4)
    y,cache=decoder_context_forward(attn,local,mem,valid,allow)
    corrupt=mem.copy(); corrupt[~valid]=1e6
    y_corrupt,_=decoder_context_forward(attn,local,corrupt,valid,allow)
    np.testing.assert_allclose(np.asarray(y),np.asarray(y_corrupt),rtol=1e-6,atol=1e-6)
    assert y.shape==(3,8,16)
    dy=xp.asarray(rng.standard_normal((3,8,16)).astype('float32'))
    dx,dm=decoder_context_backward(dy,cache)
    assert dx.shape==local.shape and dm.shape==mem.shape
    np.testing.assert_allclose(np.asarray(dm)[~np.asarray(valid)],0.,atol=1e-6)


def test_previous_commits_change_later_logits():
    # DEC fourth position may read earlier DEC tokens; DEC first may not read them.
    rng=np.random.default_rng(9)
    attn=build_attn()
    local=xp.asarray(rng.standard_normal((1,8,16)).astype('float32'))
    mem=xp.asarray(rng.standard_normal((1,5,16)).astype('float32'))
    valid=xp.ones((1,5),dtype=xp.bool_)
    baseline,_=decoder_context_forward(attn,local,mem,valid,local_allow(4))
    alt=local.copy();alt[0,4,:]+=10
    changed,_=decoder_context_forward(attn,alt,mem,valid,local_allow(4))
    # DEC 1 index4 can see itself, but cannot influence plan outputs or DEC0
    np.testing.assert_allclose(np.asarray(changed)[0,:4],np.asarray(baseline)[0,:4],atol=1e-5)
    assert np.max(np.abs(np.asarray(changed)[0,7]-np.asarray(baseline)[0,7]))>1e-9


def test_gradients_wrt_decoder_queries_context_and_weights():
    rng=np.random.default_rng(11)
    attn=build_attn(width=8,heads=2)
    local=xp.asarray(rng.standard_normal((1,4,8)).astype('float32'))
    mem=xp.asarray(rng.standard_normal((1,3,8)).astype('float32'))
    valid=xp.asarray([[0,1,1]],dtype=xp.bool_)
    allow=local_allow(2)
    upstream=xp.asarray(rng.standard_normal((1,4,8)).astype('float32'))
    y,cache=decoder_context_forward(attn,local,mem,valid,allow)
    dx,dm=decoder_context_backward(upstream,cache)
    def scalar():
        y,_=decoder_context_forward(attn,local,mem,valid,allow)
        return float(xp.sum(y*upstream))
    eps=.003
    for a,d,coords in ((local,dx,[(0,0,0),(0,3,7)]),
                       (mem,dm,[(0,2,6),(0,1,1)]),
                       (attn.Wq.data,attn.Wq.grad,[(1,4)]),
                       (attn.Wk.data,attn.Wk.grad,[(2,5)]),
                       (attn.Wv.data,attn.Wv.grad,[(4,3)]),
                       (attn.Wo.data,attn.Wo.grad,[(5,4)])):
        for loc in coords:
            old=float(a[loc]);a[loc]=old+eps;plus=scalar()
            a[loc]=old-eps;minus=scalar();a[loc]=old
            actual=float(d[loc]);expect=(plus-minus)/(2*eps)
            assert np.isclose(actual,expect,rtol=.04,atol=.0006),(loc,actual,expect)


def test_prefix_attention_does_not_read_invalid_future_context():
    # User-provided gatherer ensures <= boundary; attention must preserve the
    # padded-invalid mask rather than treating index-clipped data as valid.
    from mini_llm.block_proposal.context_attention import gather_causal_working_context
    hidden=xp.asarray(np.arange(300*8,dtype='float32').reshape(1,300,8))
    rows=xp.asarray([0,0],dtype=xp.int64)
    bounds=xp.asarray([40,104],dtype=xp.int64)
    memory,valid,ids=gather_causal_working_context(hidden,rows,bounds,40,128)
    assert np.array_equal(np.asarray(ids)[:,-1],[40,104])
    assert np.all(np.asarray(ids)<=np.asarray(bounds)[:,None])
    assert np.array_equal(np.asarray(valid).sum(axis=1),[1,65])


def test_both_existing_decoder_layers_use_the_same_read_only_memory():
    from mini_llm.block_proposal.decoder import DenseDecoderLayer
    from mini_llm.backend import RandomStream
    rng=np.random.default_rng(82)
    a=DenseDecoderLayer(16,2,32,RandomStream(82),'0080bk.layer.a','float32')
    b=DenseDecoderLayer(16,2,32,RandomStream(83),'0080bk.layer.b','float32')
    tokens=xp.asarray(rng.standard_normal((2,8,16)).astype('float32'))
    context=xp.asarray(rng.standard_normal((2,128,16)).astype('float32'))
    valid=xp.asarray([[0]*95+[1]*33,[1]*128],dtype=xp.bool_)
    mid,c1=a.forward(tokens,local_allow(4),context_memory=context,context_valid=valid)
    pred,c2=b.forward(mid,local_allow(4),context_memory=context,context_valid=valid)
    dm,dmem_b=b.backward(xp.asarray(rng.standard_normal(pred.shape).astype('float32')),c2)
    dx,dmem_a=a.backward(dm,c1)
    assert dx.shape==tokens.shape
    assert dmem_a.shape==context.shape and dmem_b.shape==context.shape
    assert float(xp.linalg.norm(dmem_a))>0 and float(xp.linalg.norm(dmem_b))>0
    np.testing.assert_array_equal(np.asarray(dmem_a)[0,:95],0)
    np.testing.assert_array_equal(np.asarray(dmem_b)[0,:95],0)


def test_context_off_retains_old_decoder_execution_path():
    from mini_llm.block_proposal.decoder import DenseDecoderLayer
    from mini_llm.backend import RandomStream
    layer=DenseDecoderLayer(16,2,32,RandomStream(84),'0080bk.layer.off','float32')
    values=xp.asarray(np.random.default_rng(2).standard_normal((1,8,16)).astype('float32'))
    out,cache=layer.forward(values,local_allow(4))
    assert len(cache)==4
    grad=layer.backward(xp.ones_like(out),cache)
    assert grad.shape==values.shape


def test_v5_full_two_stage_backward_returns_context_gradient():
    """Check production's (d_boundary,d_context) return contract end-to-end."""
    from test_block_proposal_0080bd_direct_latents import _net
    net=_net()
    rng=np.random.default_rng(804)
    h=rng.normal(size=(2,16))
    targets=rng.integers(0,101,size=(2,4),dtype=np.int64)
    valid=np.ones_like(targets,dtype=bool)
    valid[:,3]=False
    memory=rng.normal(size=(2,8,16))
    memory_valid=np.asarray([[False]*6+[True]*2,[True]*8])
    # Same CE objective as production: final CE + lambda*proposal CE.
    net.zero_grad()
    values,cache=net.forward(h,targets,valid,context_memory=memory,
                             context_valid=memory_valid)
    grad_h,grad_memory=net.backward(cache)
    assert np.isfinite(values['total_loss'])
    assert grad_h.shape==h.shape and grad_memory.shape==memory.shape
    assert np.all(np.isfinite(grad_memory))
    np.testing.assert_array_equal(grad_memory[~memory_valid],0.)
    assert np.linalg.norm(grad_memory[memory_valid])>0
    # Both decoder-layer attention weights receive the contextual gradients.
    for layer in net.decoder.layers:
        assert np.linalg.norm(layer.attn.Wk.grad)>0
        assert np.linalg.norm(layer.attn.Wv.grad)>0
    # A finite difference to rule out a shape-correct but disconnected VJP.
    def loss():
        result,_=net.forward(h,targets,valid,context_memory=memory,
                             context_valid=memory_valid)
        return result['total_loss']
    j=(1,7,0); orig=memory[j]; eps=0.003
    memory[j]=orig+eps; plus=loss()
    memory[j]=orig-eps; minus=loss()
    memory[j]=orig
    np.testing.assert_allclose(grad_memory[j],(plus-minus)/(2*eps),rtol=.1,atol=.0003)


def test_context_disabled_matches_original_decoder_checkpoint_behavior():
    """No context must leave the 0080BD teacher-forced/cached parity intact."""
    from test_block_proposal_0080bd_direct_latents import _net
    net=_net()
    rng=np.random.default_rng(765)
    h=rng.normal(size=(1,16))
    targets=rng.integers(0,101,size=(1,4))
    mask=np.ones_like(targets,dtype=bool)
    logits, latents=net.proposal_head.forward(h,return_cache=False,return_latents=True)
    plain,cache=net.decoder.forward(logits,targets,mask,proposal_latents=latents)
    assert cache.context_attention_cache is None
    state=net.decoder.prepare_plan_cache(logits,proposal_latents=latents)
    for step in range(4):
        pred=net.decoder.cached_step(state,None if step==0 else targets[:,step-1])
        np.testing.assert_allclose(pred,plain[:,step,:],rtol=3e-11,atol=3e-12)


def test_production_single_pass_scatter_with_context(monkeypatch):
    """With BK enabled, grouped blocks still use one trunk backward."""
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.training import SinglePassBlockTrainer
    from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from test_block_proposal_0080l_baselines import _configuration, _window
    main=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    net=TwoStageBlockModel(
        HorizonProposalHead(main,4,hidden_width=12,rng_seed=8081),
        BlockPrefixDecoder(main,4,width=main.config.d_model,n_layers=2,n_heads=4,
            ff_multiplier=2,rng_seed=8082,
            soft_interface_config=SoftInterfaceConfig(mode='proposal_latents_direct'),
            experimental_cpu_full_vocab=True),
        lambda_proposal=0.5)
    src,y,valid,meta=_window()
    runner=SinglePassBlockTrainer(net)
    monkeypatch.setenv('MINI_LLM_BLOCK_DECODER_CONTEXT_TOKENS','32')
    runner.zero_grad()
    results=runner.train_window(src,y,valid_target_mask=valid,
               memory_metadata=meta,block_group_size=2,compute_gradients=True)
    assert results['main_forward_passes']==1
    assert results['main_backward_passes']==1
    assert results['unique_supervised_tokens']==int(valid.sum())
    assert np.isfinite(results['final_ce'])
    assert np.linalg.norm(net.decoder.layers[0].attn.Wk.grad)>0
    assert np.linalg.norm(net.decoder.layers[1].attn.Wk.grad)>0
    for layer in main.blocks:
        assert np.all(np.isfinite(np.asarray(layer.attention.Wq.grad)))


def test_bk_contextual_cached_inference_fails_closed(monkeypatch):
    """Training with context must not silently evaluate context-free generation."""
    from test_block_proposal_0080bd_direct_latents import _net
    net=_net()
    h=np.random.default_rng(89).normal(size=(1,16))
    p,z=net.proposal_head.forward(h,return_cache=False,return_latents=True)
    monkeypatch.setenv('MINI_LLM_BLOCK_DECODER_CONTEXT_TOKENS','128')
    with pytest.raises(NotImplementedError,match='0080BK contextual'):
        net.decoder.prepare_plan_cache(p,proposal_latents=z)
