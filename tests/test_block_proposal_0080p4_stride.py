"""0080P4: strict inference-only commitment stride and teacher-forced scoring."""
import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.inference_model import InferenceModel
from mini_llm.block_proposal.generation import SequentialCommitBlockGenerator
from examples.evaluate_block_proposal_stride_0080p4 import score_fixed_rows
from tests.test_block_proposal_0080c import _joint


def _fresh(net, prompt, length=20):
    model = net.proposal_head.model
    inf = InferenceModel(model.config, dtype=model.dtype)
    inf.set_weights(model)
    state = inf.create_generation_state(1,length)
    inf.prefill(xp.asarray([prompt],dtype=xp.int32),state)
    return inf,state


@pytest.mark.parametrize('stride,expected_plans',[(1,9),(2,5),(4,3)])
def test_stride_uses_full_plan_each_time_and_commits_exactly(stride,expected_plans):
    net,_,_,_=_joint()
    inf,state=_fresh(net,[1,2,3,4],20)
    calls=[]
    def choose(scores):
        calls.append(int(scores.shape[0]))
        return 3
    gen=SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
                                       chooser=choose,commit_stride=stride)
    ret=gen.generate(state,9)
    assert ret['tokens']==[3]*9
    assert ret['proposal_blocks']==expected_plans
    assert ret['causal_main_updates']==9
    assert ret['commit_stride']==stride and ret['proposal_horizon']==4
    assert state.length==13
    assert calls==[19]*9


def test_stride_one_matches_manual_per_token_replanning():
    net,_,_,_=_joint()
    a,sa=_fresh(net,[1,2,3,4],20)
    b,sb=_fresh(net,[1,2,3,4],20)
    gen=SequentialCommitBlockGenerator(a,net.proposal_head,net.decoder,
        chooser=lambda s:int(xp.argmax(s)),commit_stride=1)
    actual=gen.generate(sa,6)['tokens']
    expected=[]
    for _ in range(6):
        logits=net.proposal_head.forward(b.last_hidden,return_cache=False)
        cache=net.decoder.prepare_plan_cache(logits)
        first=net.decoder.cached_step(cache,None)
        token=int(xp.argmax(first[0]))
        expected.append(token)
        b.decode_one(xp.asarray([[token]],dtype=xp.int32),sb)
    assert actual==expected
    np.testing.assert_allclose(asnumpy(a.last_hidden),asnumpy(b.last_hidden),atol=1e-11)


@pytest.mark.parametrize('stride', [1,2,4])
def test_reference_nll_is_same_cached_logits_and_true_incremental_prefix(stride):
    net,_,_,_=_joint()
    inf,state=_fresh(net,[1,2,3,4],20)
    gen=SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
        chooser=lambda s: 1,commit_stride=stride)
    reference=[5,6,7,8,9,10]
    scored=gen.score_reference_tokens(state,reference)
    assert state.length==10
    assert scored['n_tokens']==6
    assert scored['proposal_blocks']==(6+stride-1)//stride
    assert np.isfinite(scored['nll_sum'])
    assert scored['mean_ce']==pytest.approx(np.mean(scored['per_token_nll']))
    # Replay with identical main inputs and direct explicit logits to verify NLL.
    inf2,state2=_fresh(net,[1,2,3,4],20)
    nll=[]
    for start in range(0,len(reference),stride):
        proposals=net.proposal_head.forward(inf2.last_hidden,return_cache=False)
        cache=net.decoder.prepare_plan_cache(proposals)
        for j,token in enumerate(reference[start:start+stride]):
            scores=net.decoder.cached_step(cache,None if j==0 else xp.asarray([reference[start+j-1]]))[0]
            from scipy.special import logsumexp
            nll.append(float(logsumexp(np.asarray(scores)) - scores[token]))
            inf2.decode_one(xp.asarray([[token]],dtype=xp.int32),state2)
    np.testing.assert_allclose(scored['per_token_nll'],nll,rtol=1e-12,atol=1e-12)


def test_teacher_forced_stride_four_matches_full_prefix_lm_forward():
    net,_,_,_=_joint()
    inf,state=_fresh(net,[1,2,3,4],20)
    logits=net.proposal_head.forward(inf.last_hidden,return_cache=False)
    reference=xp.asarray([[5,6,7,8]],dtype=xp.int64)
    all_valid=xp.ones((1,4),dtype=xp.bool_)
    expected,_=net.decoder.forward(logits,reference,all_valid)
    gen=SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
        chooser=lambda s: 1,commit_stride=4)
    score=gen.score_reference_tokens(state,reference[0])
    from scipy.special import logsumexp
    expected_nll=[float(logsumexp(np.asarray(expected[0,j])) - expected[0,j,int(reference[0,j])])
                  for j in range(4)]
    np.testing.assert_allclose(score['per_token_nll'],expected_nll,rtol=1e-10,atol=1e-10)


def test_eos_and_invalid_stride():
    net,_,_,_=_joint()
    inf,state=_fresh(net,[1,2,3,4],20)
    with pytest.raises(ValueError,match='commit_stride'):
        SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
            chooser=lambda s:1,commit_stride=0)
    with pytest.raises(ValueError,match='commit_stride'):
        SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
            chooser=lambda s:1,commit_stride=5)
    gen=SequentialCommitBlockGenerator(inf,net.proposal_head,net.decoder,
        chooser=lambda s:7,commit_stride=1,eos_token_id=7)
    ret=gen.generate(state,9)
    assert ret['eos'] and ret['tokens']==[7] and ret['proposal_blocks']==1


def test_heldout_row_scoring_returns_paired_stats():
    net,_,_,_=_joint()
    rows=[([1,2,3,4],[5,6,7,8]),([4,3,2,1],[9,10,11])]
    results=score_fixed_rows(net,rows,strides=[1,2,4],max_context=20,target_tokens=4)
    assert set(results)=={'1','2','4'}
    assert {results[str(i)]['tokens'] for i in (1,2,4)}=={7}
    assert results['1']['plans']==7
    assert results['2']['plans']==4
    assert results['4']['plans']==2


def test_generation_benchmark_cli_stride_manifest_resume_isolation(tmp_path, monkeypatch):
    import json
    from benchmarks import evaluate_generation as benchmark
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps(dict(
        prompts=[dict(id='math',domain='science',topic='math',length='short',
                      format='prose',prompt='four')],recommended_seeds=[42],
        recommended_control=dict(temperature=0.8,top_k=40),recommended_new_tokens=6)))
    class StubGenerator:
        def __init__(self, stride): self.stride=stride
        def generate(self, **kwargs):
            return 'made up example',dict(generated_token_ids=[1,2,3,4,5,6],
                       eos_token_id=0,total_time_s=1.,prefill_time_s=.1,
                       tokens_per_sec=6,commit_stride=self.stride,proposal_horizon=4,
                       proposal_blocks=(6+self.stride-1)//self.stride,
                       causal_main_updates=6)
    def loader(checkpoint,tokenizer_path,context,stride):
        return StubGenerator(stride),dict(step=2500,model_dtype='float32',context=256,
                                      backend='numpy',runtime_env={})
    monkeypatch.setattr(benchmark,'load_block_generator',loader)
    path=tmp_path/'out'
    params=['--architecture','block-proposal-0080i','--checkpoint',str(tmp_path/'ckpt'),
            '--output-dir',str(path),'--prompt-suite',str(suite),'--strategies','greedy',
            '--commit-stride','2']
    assert benchmark.main(params)==0
    row=json.loads((path/'generations.jsonl').read_text().splitlines()[0])
    assert row['timing']['commit_stride']==2 and row['timing']['proposal_blocks']==3
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['configuration']['commit_stride_request']==2
    assert benchmark.main(params+['--resume'])==0
    with pytest.raises(ValueError,match='Resume configuration differs'):
        benchmark.main([('1' if p=='2' else p) for p in params]+['--resume'])
    with pytest.raises(SystemExit):
        benchmark.main(['--architecture','autoregressive','--checkpoint',str(tmp_path/'ckpt'),
            '--output-dir',str(tmp_path/'new'),'--prompt-suite',str(suite),
            '--commit-stride','2','--dry-run'])


from tests.test_block_proposal_0080i_learning import prepared

def test_stride_evaluator_actual_prepared_checkpoint_provenance(prepared, tmp_path, monkeypatch):
    """Checkpoint+tokenizer reconstruction, fixed dataset, exact inference CE."""
    from examples import train_block_proposal_cpu_learning_0080i as train
    from examples import evaluate_block_proposal_stride_0080p4 as evaluator
    from tests.test_block_proposal_0080i_learning import _argv
    from tests.test_block_proposal_0080g_cpu import ToyTokenizer
    out = tmp_path / 'run'
    train.main(_argv(prepared,out,'--architecture','two_stage','--batch-size','2'))
    ckpt = out / 'checkpoints' / 'two_stage' / 'step_00000002'
    monkeypatch.setattr(evaluator,'_load_tokenizer',lambda path:(ToyTokenizer(),1))
    monkeypatch.setattr('mini_llm.block_proposal.benchmark_adapter.FastBPETokenizer.load',
                        lambda path:ToyTokenizer())
    args = ['--mixed-shard-root',str(prepared),'--training-output',str(out),
            '--checkpoint',str(ckpt),'--strides','4,2,1',
            '--validation-rows','2','--target-tokens','6',
            '--output',str(tmp_path/'stride_eval.json')]
    report = evaluator.main(args)
    assert report['checkpoint_step']==2
    assert report['selected_rows']==2
    for s in ['4','2','1']:
        entry = report['strides'][s]
        assert entry['tokens'] > 0 and np.isfinite(entry['final_ce'])
        assert entry['rows']==2
    with pytest.raises(FileExistsError,match='refusing to overwrite'):
        evaluator.main(args)
