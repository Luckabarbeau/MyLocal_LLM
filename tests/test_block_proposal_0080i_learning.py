"""0080I: learning curves, same-sample fairness, checkpoint/resume on CPU."""
import json
import numpy as np
import pytest

from examples import train_block_proposal_cpu_learning_0080i as learn
from examples import prepare_block_proposal_cpu_data as prep
from tests.test_block_proposal_0080g_cpu import ToyTokenizer


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(prep,'FastBPETokenizer',ToyTokenizer)
    sources=[]
    for name in ('fine_edu','cosmopedia_v2'):
        folder=tmp_path/name
        folder.mkdir()
        for i in range(16):
            (folder/f'{i:02d}.txt').write_text((f'{name} record {i}: hello world.\n')*6)
        sources.extend(['--source',f'{name}={folder}'])
    root=tmp_path/'shards'
    prep.prepare(prep.parse_args(['--output',str(root),'--max-documents-per-source','16',
                                  '--val-fraction','0.25',*sources]))
    monkeypatch.setattr(learn,'_load_tokenizer',lambda p:(ToyTokenizer(),1))
    return root


def _argv(root,out,*extra):
    return ['--mixed-shard-root',str(root),'--model-preset','cpu_tiny',
            '--steps','2','--warmup-steps','0','--eval-every','1',
            '--save-every','1','--log-every','1','--val-batches','2',
            '--block-size','4','--block-group-size','2',
            '--proposal-width','16','--decoder-width','16',
            '--decoder-layers','1','--decoder-heads','2',
            '--output',str(out),*extra]


def test_learning_both_matched_steps_validation_and_checkpoints(prepared,tmp_path):
    output=tmp_path/'learning'
    result=learn.main(_argv(prepared,output))
    ar,tw=result['standard_ar'],result['two_stage']
    assert ar['step_sample_digests']==tw['step_sample_digests']
    assert ar['main_initial_weights_sha256']==tw['main_initial_weights_sha256']
    assert tw['added_parameters']>0
    for name in ('standard_ar','two_stage'):
        history=[json.loads(x) for x in (output/f'{name}_validation.jsonl').read_text().splitlines()]
        assert [v['step'] for v in history]==[0,1,2]
        assert all(np.isfinite(v['final_ce']) and v['valid_tokens']>0 for v in history)
        assert (output/'checkpoints'/name/'step_00000002'/'training_state.json').exists()
    assert len(result['two_stage']['validation_records'][-1]['decoder_gain_by_horizon'])==4
    assert np.isfinite(result['comparison']['final_ce_difference_two_stage_minus_ar'])


@pytest.mark.parametrize('name',('standard_ar','two_stage'))
def test_resume_matches_uninterrupted_training(prepared,tmp_path,name):
    full=tmp_path/f'{name}_full'
    split=tmp_path/f'{name}_split'
    base=_argv(prepared,full,'--architecture',name)
    rfull=learn.main(base)[name]
    base2=_argv(prepared,split,'--architecture',name)
    short=list(base2)
    short[short.index('--steps')+1]='1'
    learn.main(short)
    rsplit=learn.main(_argv(prepared,split,'--architecture',name,'--resume'))[name]
    assert rfull['step_sample_digests']==rsplit['step_sample_digests']
    assert rfull['supervised_tokens']==rsplit['supervised_tokens']
    assert rfull['validation_final_ce']==pytest.approx(rsplit['validation_final_ce'],abs=1e-9)
    assert rfull['validation_initial_ce']==pytest.approx(rsplit['validation_initial_ce'],abs=1e-9)
    assert (full/'checkpoints'/name/'step_00000002'/'training_state.json').exists()
