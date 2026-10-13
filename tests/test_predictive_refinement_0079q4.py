"""Q4: next-token alignment, unbiased short documents, masked routed cache."""
import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.predictive_refinement.routed_real_data import (
    align_next_token_targets, PackedRoutedRefinementDocuments,
)
from tests.test_predictive_refinement_0079q2 import make_trainer, _source


def test_source_to_labels_shift_keeps_first_conditioning_token():
    D=6; T=4
    # Real history h0..h5 then current inputs x0..x3.
    src=np.array([[1,2,3,4,5,6,7,8,9,10],
                  [99,99,99,99,99,20,21,22,23,99]],dtype=np.int64)
    labels=np.array([[8,9,10,11],[22,23,99,99]],dtype=np.int64)
    md={'history_valid_starts':np.array([0,5]),
        'target_valid_lengths':np.array([4,2])}
    out,mask,meta=align_next_token_targets(src,labels,md,memory_length=D,pad_token_id=99)
    np.testing.assert_array_equal(out[0],np.array([2,3,4,5,6,7,8,9,10,11]))
    np.testing.assert_array_equal(out[1],np.array([99,99,99,99,20,21,22,23,99,99]))
    np.testing.assert_array_equal(meta['history_valid_starts'],[0,4])
    np.testing.assert_array_equal(mask,[np.ones(T,bool),np.array([1,1,0,0],bool)])
    assert meta['causal_masked_target_refinement'] is True


def test_document_first_short_sample_retains_first_input_and_eos(tmp_path):
    # One 4-token document, so no size-based rejection is possible.
    np.asarray([10,11,12,99],dtype=np.uint16).tofile(tmp_path/'train_shard_000.bin')
    data=PackedRoutedRefinementDocuments(shard_dir=tmp_path,split='train',
                                         eos_token_id=99,token_dtype='uint16')
    src,valid,md,obs=data.sample_routed(np.random.default_rng(17),batch_size=3,
                                       memory_length=32,target_length=16,
                                       min_history_tokens=10)
    assert src.shape==(3,48)
    np.testing.assert_array_equal(md['history_valid_starts'],[31]*3)
    np.testing.assert_array_equal(md['target_valid_lengths'],[3]*3)
    assert (src[:,31]==10).all()
    np.testing.assert_array_equal(src[:,32:35],np.array([[11,12,99]]*3))
    assert valid[:,:3].all() and not valid[:,3:].any()
    assert [x['sampled_document_span_tokens'] for x in obs]==[4]*3


def _masked_eval(trainer,source,mask,meta,group):
    trainer.zero_grad()
    m=trainer.train_routed_window(source, memory_metadata=meta,
                                  valid_token_mask=mask,group_size=group,
                                  compact_history=True,compact_active_blocks=True,
                                  resident_history=True,max_window_tokens=16,
                                  max_history_keys=32,max_active_queries=32)
    return m,[(p.name,asnumpy(p.grad).copy()) for p in trainer.parameters()]


@pytest.mark.parametrize('learned',[False,True])
def test_short_window_masked_gradients_and_future_pad_independence(learned):
    trainer=make_trainer(k=8, learned=learned)
    src=_source(1)
    D=32
    mask=xp.asarray(np.array([[True]*11 +[False]*5]))
    # Only one genuinely historical token immediately before first label;
    # this checks we do not discard that token when target is short.
    src[:, :D-1]=0
    src[:, D-1]=37
    src[:, D+11:]=0
    meta=dict(history_valid_starts=xp.asarray([D-1]),
              target_valid_lengths=xp.asarray([11]),
              causal_masked_target_refinement=True,use_retrieval=True,
              router_stochastic=False)
    ref,grads=_masked_eval(trainer,src,mask,meta,1)
    other,othergrads=_masked_eval(trainer,src,mask,meta,2)
    assert ref['supervised_tokens']==11
    for key in ('commitment_ce','initial_ce','total_loss'):
        np.testing.assert_allclose(float(ref[key]),float(other[key]),rtol=1e-8,atol=1e-9)
    for (name,a),(name2,b) in zip(grads,othergrads):
        assert name==name2
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7,err_msg=name)
    changed=src.copy()
    changed[:, D+11:]=(changed[:,D+11:]+57)%99
    last,changed_grads=_masked_eval(trainer,changed,mask,meta,1)
    np.testing.assert_allclose(float(ref['commitment_ce']),float(last['commitment_ce']),rtol=0,atol=1e-12)
    for (name,a),(name2,b) in zip(grads,changed_grads):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7,err_msg=name)


def test_reject_inconsistent_mask_and_provenance():
    t=make_trainer(k=8)
    src=_source(1)
    kw=dict(valid_token_mask=xp.asarray([[True]*12+[False]*4]),
            group_size=1,compact_history=True,compact_active_blocks=True,
            resident_history=True,max_window_tokens=16,max_history_keys=32)
    with pytest.raises(ValueError,match='requires a fully valid'):
        t.train_routed_window(src,memory_metadata={'history_valid_starts':xp.asarray([0])},**kw)
    with pytest.raises(ValueError,match='disagree'):
        t.train_routed_window(src,memory_metadata=dict(history_valid_starts=xp.asarray([0]),
             target_valid_lengths=xp.asarray([13]),causal_masked_target_refinement=True),**kw)
    with pytest.raises(ValueError,match='causal_prefix_query'):
        t.model.forward_body(src,memory_metadata=dict(causal_masked_target_refinement=True,
                                                       causal_prefix_query=False),return_cache=True)


def test_q4_real_trainer_helper_optimizer_and_fixed_validation():
    """Execute actual forward/backward/AdamW on a small routed MoE model."""
    from argparse import Namespace
    from mini_llm.optim.adamw import AdamW
    from examples.train_predictive_refinement_routed_text import _run_window, _evaluate
    trainer=make_trainer(k=8,learned=True)
    x=_source(1)
    mask=np.asarray([[True]*11+[False]*5])
    x[:,43:]=0
    md=dict(history_valid_starts=np.asarray([0]),
            target_valid_lengths=np.asarray([11]),
            causal_masked_target_refinement=True,router_stochastic=False,
            causal_prefix_query=True,use_retrieval=True)
    sample=(asnumpy(x),mask,md,[{'source':'test','sampled_document_span_tokens':12}])
    opts=Namespace(refinement_group_size=1,max_active_queries=8,
                   max_history_keys=32,smoke=False,max_supervised_blocks=None,
                   block_size=8)
    opt=AdamW(trainer.parameters(),lr=1e-4,weight_decay=0.1)
    trainer.zero_grad()
    before=[asnumpy(p.data).copy() for p in trainer.parameters()]
    result=_run_window(trainer,sample,opts,normalization=11,compute_gradients=True)
    assert result['supervised_tokens']==11
    assert np.isfinite(float(result['commitment_ce']))
    opt.step()
    assert opt.step_index==1
    assert any(not np.array_equal(b,asnumpy(p.data)) for b,p in zip(before,trainer.parameters()))
    m=_evaluate(trainer,[sample],opts)
    assert m['supervised_tokens']==11
    assert np.isfinite(m['commitment_ce'])
    assert m['by_source']['test']['tokens']==11
