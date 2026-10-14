"""0080O10: fixed-shape BOS-safe embedding-gradient scatter parity."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy


def _model(*, low_precision=False):
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel

    dtype = 'bfloat16' if low_precision else 'float64'
    cfg=ModelConfig(tokenizer_vocab_size=37,context_length=16,n_layers=1,
        d_model=16,n_q_heads=2,n_kv_heads=1,d_head=8,d_ff=32,
        n_experts=2,top_k=1,dtype=dtype)
    main=DecoderLanguageModel(cfg,rng_seed=111,dtype=dtype)
    head=HorizonProposalHead(main,4,hidden_width=16,rng_seed=80)
    decoder=BlockPrefixDecoder(main,4,width=16,n_heads=4,n_layers=2,rng_seed=81)
    return TwoStageBlockModel(head,decoder,lambda_proposal=.5)


def _run(model,monkeypatch,fixed):
    monkeypatch.setenv('MINI_LLM_BLOCK_DECODER_FIXED_SCATTER',str(int(fixed)))
    model.zero_grad()
    # Valid token ID 0 occurs alongside BOS and several invalid tails.
    h=xp.asarray(np.random.default_rng(81010).normal(size=(4,16)).astype('float32'),
                 dtype=model.proposal_head.model.dtype)
    labels=xp.asarray([[0,3,4,3], [4,0,-100,-100], [0,-100,-100,-100],
                       [5,7,0,-100]],dtype=xp.int64)
    valid=xp.asarray([[1,1,1,1], [1,1,0,0], [1,0,0,0], [1,1,1,0]],dtype=xp.bool_)
    metrics,cache=model.forward(h,labels,valid)
    dx=model.backward(cache)
    grads={p.name:asnumpy(p.grad).copy() for p in model.parameters()}
    return metrics,asnumpy(dx).copy(),grads


@pytest.mark.skipif(BACKEND_NAME!='numpy',reason='CPU gradient-parity reference')
def test_full_two_stage_model_fixed_scatter_exact_cpu_parity(monkeypatch):
    model=_model()
    (a, da, ga)=_run(model,monkeypatch,False)
    (b, db, gb)=_run(model,monkeypatch,True)
    for name in ('proposal_ce','final_ce','total_loss'):
        np.testing.assert_array_equal(asnumpy(a[name]),asnumpy(b[name]))
    np.testing.assert_array_equal(da,db)
    assert ga.keys()==gb.keys()
    for key in ga:
        np.testing.assert_array_equal(ga[key],gb[key],err_msg=key)
    # The shared input embedding received a real gradient for vocabulary ID 0.
    assert np.any(ga['embedding.W'][0] != 0)


@pytest.mark.skipif(BACKEND_NAME!='numpy',reason='CPU finite-difference test')
def test_fixed_scatter_preserves_all_0080c_finite_difference_gradients(monkeypatch):
    monkeypatch.setenv('MINI_LLM_BLOCK_DECODER_FIXED_SCATTER','1')
    from test_block_proposal_0080c import test_finite_difference_joint_gradient_through_both_paths
    test_finite_difference_joint_gradient_through_both_paths()


@pytest.mark.skipif(BACKEND_NAME!='cupy',reason='requires CUDA')
def test_cuda_bf16_fixed_scatter_preserves_joint_gradients(monkeypatch):
    model=_model(low_precision=True)
    a,da,ga=_run(model,monkeypatch,False)
    b,db,gb=_run(model,monkeypatch,True)
    for name in ('proposal_ce','final_ce','total_loss'):
        np.testing.assert_allclose(asnumpy(a[name]),asnumpy(b[name]),atol=2e-3,rtol=1e-3)
    np.testing.assert_allclose(da,db,atol=1e-4,rtol=3e-2)
    assert ga.keys()==gb.keys()
    for name in ga:
        np.testing.assert_allclose(ga[name],gb[name],atol=2e-4,rtol=3e-2,err_msg=name)
