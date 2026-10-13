"""0080F first checkpoint milestone: exact AdamW resume on a small CPU model."""
import numpy as np
import pytest
from mini_llm.backend import xp,asnumpy
from mini_llm.optim.adamw import AdamW
from mini_llm.block_proposal.checkpoint import save_two_stage_checkpoint,load_two_stage_checkpoint
from tests.test_block_proposal_0080c import _joint


def test_two_stage_optimizer_resume_exact_after_one_step(tmp_path):
    one,h,targets,valid=_joint()
    other,h2,_,_=_joint()
    opt=AdamW(one.parameters(),lr=1e-4,weight_decay=0.1)
    opt2=AdamW(other.parameters(),lr=1e-4,weight_decay=0.1)
    def step(model,optimizer,states):
        model.zero_grad()
        loss,tape=model.forward(states,targets,valid)
        model.backward(tape)
        optimizer.step()
        model.proposal_head.model.refresh_compute_buffers()
        return loss
    step(one,opt,h)
    dest=tmp_path/'model'
    save_two_stage_checkpoint(dest,one,optimizer=opt,training_step=1,supervised_tokens=4,
        tokenizer_hash='test-tokenizer',data_provenance={'dataset':'tiny'},rng_state={'seed':8})
    restored=load_two_stage_checkpoint(dest,other,optimizer=opt2,
        expected_tokenizer_hash='test-tokenizer',expected_data_provenance={'dataset':'tiny'})
    assert restored['step']==1 and opt2.step_index==1
    a=step(one,opt,h)
    b=step(other,opt2,h2)
    np.testing.assert_allclose(a['total_loss'],b['total_loss'],rtol=0,atol=0)
    for x,y in zip(one.parameters(),other.parameters()):
        assert x.name==y.name
        np.testing.assert_allclose(asnumpy(x.data),asnumpy(y.data),rtol=0,atol=0,err_msg=x.name)
    with pytest.raises(ValueError,match='tokenizer hash'):
        load_two_stage_checkpoint(dest,other,expected_tokenizer_hash='incorrect')
