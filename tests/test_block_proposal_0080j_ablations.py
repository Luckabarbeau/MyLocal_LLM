"""0080J evaluation-only ablations preserve direct priors and labels."""
import numpy as np
import json
import pytest

from mini_llm.backend import asnumpy, xp
from tests.test_block_proposal_0080d import _build, _sample
from tests.test_block_proposal_0080i_learning import prepared, _argv
from examples import train_block_proposal_cpu_learning_0080i as learn
from examples import evaluate_block_proposal_ablations_0080j as ablate


def _metric(trainer, plan='normal', prefix='normal', group=2):
    src,y,valid,meta=_sample()
    return trainer.train_window(src,y,valid_target_mask=valid,
        memory_metadata=meta,block_group_size=group,
        compute_gradients=False,plan_condition=plan,prefix_condition=prefix)


def test_each_ablation_preserves_prior_ce_and_valid_counts():
    model=_build(learned=True)
    # Amplify the random residual correction so forward-only controls are
    # measurably different without needing a trained checkpoint in this test.
    model.net.decoder.up.W.data *= 100.
    reference=_metric(model)
    for plan in ('normal','neutral','shuffled'):
        for prefix in ('normal','masked'):
            got=_metric(model,plan,prefix)
            assert got['unique_supervised_tokens']==reference['unique_supervised_tokens']
            np.testing.assert_array_equal(asnumpy(got['position_counts']),asnumpy(reference['position_counts']))
            np.testing.assert_allclose(asnumpy(got['proposal_position_ce_sums']),
                                       asnumpy(reference['proposal_position_ce_sums']),atol=1e-11,rtol=0)
            assert np.isfinite(got['final_ce'])
    assert abs(_metric(model,'neutral')['final_ce']-reference['final_ce'])>1e-9
    assert abs(_metric(model,'normal','masked')['final_ce']-reference['final_ce'])>1e-9
    assert abs(_metric(model,'shuffled')['final_ce']-reference['final_ce'])>1e-9


def test_controls_are_chunk_size_invariant():
    model=_build(learned=True)
    model.net.decoder.up.W.data *= 100.
    for plan,prefix in ablate.CONDITIONS:
        a=_metric(model,plan,prefix,group=1)
        b=_metric(model,plan,prefix,group=9)
        np.testing.assert_allclose(a['final_ce'],b['final_ce'],rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(asnumpy(a['final_position_ce_sums']),
                                   asnumpy(b['final_position_ce_sums']),atol=3e-6,rtol=3e-7)


def test_ablation_is_read_only_and_needs_multiple_documents():
    model=_build()
    src,labels,valid,meta=_sample()
    with pytest.raises(ValueError,match='evaluation-only'):
        model.train_window(src,labels,valid_target_mask=valid,memory_metadata=meta,
                           plan_condition='neutral',compute_gradients=True)
    before=[asnumpy(p.data).copy() for p in model.parameters()]
    _metric(model,'shuffled','masked')
    for p,x in zip(model.parameters(),before):
        np.testing.assert_array_equal(asnumpy(p.data),x)
    with pytest.raises(ValueError,match='independent source rows'):
        model.train_window(src[:1],labels[:1],valid_target_mask=valid[:1],
            memory_metadata={k:(v[:1] if isinstance(v,np.ndarray) else v)
                             for k,v in meta.items()},
            plan_condition='shuffled',compute_gradients=False)


def test_load_saved_0080i_checkpoint_and_evaluate_all_controls(prepared,tmp_path,monkeypatch):
    output=tmp_path/'learning'
    learn.main(_argv(prepared,output,'--architecture','two_stage','--batch-size','2'))
    monkeypatch.setattr(ablate,'_load_tokenizer',lambda path:(__import__('tests.test_block_proposal_0080g_cpu',fromlist=['ToyTokenizer']).ToyTokenizer(),1))
    result=ablate.main(['--training-output',str(output),
                        '--mixed-shard-root',str(prepared)])
    assert result['training_step']==2
    assert len(result['conditions'])==6
    saved=json.loads((output/'two_stage_validation.jsonl').read_text().splitlines()[-1])
    assert result['conditions']['normal_plan__normal_prefix']['final_ce']==pytest.approx(
        saved['final_ce'],abs=2e-6)
    assert (output/'ablations_0080j.json').is_file()
    baseline=result['conditions']['normal_plan__normal_prefix']
    assert all(x['valid_tokens']==baseline['valid_tokens'] for x in result['conditions'].values())
    assert all(abs(x['proposal_ce']-baseline['proposal_ce'])<1e-10 for x in result['conditions'].values())
