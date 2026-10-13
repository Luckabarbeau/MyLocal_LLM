"""0080C: differentiable two-stage teacher forcing and no future-label leakage."""
import numpy as np
import pytest
from mini_llm.backend import xp,asnumpy
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.decoder import BlockPrefixDecoder,TwoStageBlockModel


def _joint():
    cfg=ModelConfig(tokenizer_vocab_size=19,context_length=16,n_layers=1,
        d_model=8,n_q_heads=2,n_kv_heads=1,d_head=4,d_ff=16,n_experts=2,top_k=1,dtype='float64')
    main=DecoderLanguageModel(cfg,rng_seed=11,dtype='float64')
    head=HorizonProposalHead(main,4,hidden_width=9,rng_seed=80)
    dec=BlockPrefixDecoder(main,4,width=12,n_heads=3,n_layers=2,rng_seed=81)
    net=TwoStageBlockModel(head,dec,lambda_proposal=0.5)
    context=xp.asarray([[1,2,3,4,5,6]])
    h=main.forward_body(context,return_cache=False)[:,2:3,:].reshape(1,8).copy()
    targets=xp.asarray([[7,8,9,10]])
    valid=xp.asarray([[1,1,1,1]],dtype=xp.bool_)
    return net,h,targets,valid


def test_joint_objective_has_unique_target_denominator_and_nonzero_gradients():
    model,h,targets,valid=_joint()
    model.zero_grad()
    metrics,tape=model.forward(h,targets,valid)
    assert metrics['unique_supervised_tokens']==4
    np.testing.assert_allclose(metrics['total_loss'],metrics['final_ce']+0.5*metrics['proposal_ce'],atol=1e-14)
    dh=model.backward(tape)
    assert np.linalg.norm(asnumpy(dh))>0
    assert np.linalg.norm(asnumpy(model.proposal_head.W_back.grad))>0
    assert np.linalg.norm(asnumpy(model.decoder.up.W.grad))>0
    assert np.linalg.norm(asnumpy(model.decoder.layers[0].attn.Wq.grad))>0
    assert np.linalg.norm(asnumpy(model.proposal_head.model.embedding.W.grad))>0
    assert np.linalg.norm(asnumpy(model.proposal_head.model.output_proj.W.grad))>0


def test_future_teacher_forcing_cannot_change_earlier_decision():
    model,h,targets,valid=_joint()
    lp=model.proposal_head.forward(h,return_cache=False)
    baseline,_=model.decoder.forward(lp,targets,valid)
    for prefix in (0,1,2):
        corrupted=targets.copy()
        corrupted[:,prefix:]=xp.asarray(np.mod(np.array([3,4,5,6])[prefix:]+prefix,19))
        final,_=model.decoder.forward(lp,corrupted,valid)
        # DEC_j input is y_(j-1), never y_j or later.
        np.testing.assert_allclose(asnumpy(final[:,:prefix+1]),asnumpy(baseline[:,:prefix+1]),atol=1e-14,rtol=1e-14)
    # Changing PLAN 3 can influence DEC 1, because all proposals are visible.
    changed=lp.copy(); changed[:,3,0]+=10
    result,_=model.decoder.forward(changed,targets,valid)
    assert not np.allclose(asnumpy(result[:,0]),asnumpy(baseline[:,0]))


def test_finite_difference_joint_gradient_through_both_paths():
    model,h,targets,valid=_joint()
    targets=xp.asarray([[7,8,-100,-100]])
    valid=xp.asarray([[1,1,0,0]],dtype=xp.bool_)
    model.zero_grad()
    metrics,tape=model.forward(h,targets,valid)
    dh=model.backward(tape)
    def loss_at():
        return model.forward(h,targets,valid)[0]['total_loss']
    eps=1e-5
    params=[(model.decoder.plan_projection.W,(2,3)),
            (model.decoder.token_projection.W,(3,4)),
            (model.decoder.up.W,(4,5)),
            (model.decoder.layers[0].attn.Wq,(2,4)),
            (model.decoder.bos,(5,)),
            (model.proposal_head.horizon_embedding,(2,1)),
            (model.proposal_head.model.embedding.W,(7,3)),
            (model.proposal_head.model.output_proj.W,(3,7))]
    for p,idx in params:
        base=float(p.data[idx]);p.data[idx]=base+eps
        lplus=loss_at();p.data[idx]=base-eps
        lminus=loss_at();p.data[idx]=base
        numerical=(lplus-lminus)/(2*eps)
        np.testing.assert_allclose(float(p.grad[idx]),numerical,atol=2e-7,rtol=7e-3,err_msg=p.name)
    for i in (0,5):
        state=h.copy();state[0,i]+=eps
        lplus=model.forward(state,targets,valid)[0]['total_loss']
        state[0,i]-=2*eps
        lminus=model.forward(state,targets,valid)[0]['total_loss']
        np.testing.assert_allclose(float(dh[0,i]),(lplus-lminus)/(2*eps),rtol=3e-3,atol=3e-7)


def test_invalid_teacher_forced_tail_never_enters_valid_logits():
    model,h,targets,valid=_joint()
    valid[:,2:]=False
    a=model.forward(h,targets,valid)[0]
    other=targets.copy();other[:,2:]=xp.asarray([1,17])
    b=model.forward(h,other,valid)[0]
    for key in ('final_ce','proposal_ce','total_loss'):
        np.testing.assert_allclose(a[key],b[key],atol=0,rtol=0)
