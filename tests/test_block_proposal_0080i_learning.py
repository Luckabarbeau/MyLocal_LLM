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


def test_weight_only_warm_start_expanded_data_and_resume(prepared, tmp_path, monkeypatch):
    """Changed data rejects full resume, but exact-architecture weight loading works."""
    import shutil
    from mini_llm.optim.adamw import AdamW

    old_output = tmp_path / 'old_training'
    learn.main(_argv(prepared, old_output, '--architecture', 'two_stage'))
    source = old_output/'checkpoints'/'two_stage'/'step_00000002'
    expanded = tmp_path/'expanded_data'
    shutil.copytree(prepared, expanded)
    # Retain tokenizer bytes while extending a train shard and updating the
    # recorded token count; this is a different training corpus.
    import numpy as np
    arr = np.asarray([3, 7, 9, 1], dtype=np.uint16)
    with (expanded/'fine_edu'/'train_shard_00000.bin').open('ab') as fd:
        fd.write(arr.tobytes())
    manifest_path = expanded/'mixture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sources']['fine_edu']['train_tokens'] += len(arr)
    manifest_path.write_text(json.dumps(manifest))

    new_output = tmp_path / 'warm_training'
    base = _argv(expanded, new_output, '--architecture','two_stage',
                 '--warm-start-checkpoint', str(source), '--learning-rate','0.0001')
    original_step = AdamW.step
    step_observations = []

    def check_fresh_master(opt, *args, **kwargs):
        if not step_observations:
            assert opt.step_index == 0
            for i, p in enumerate(opt.parameters):
                np.testing.assert_allclose(opt.master_weights[i], p.data, atol=1e-6, rtol=0)
            step_observations.append(True)
        return original_step(opt, *args, **kwargs)
    monkeypatch.setattr(AdamW, 'step', check_fresh_master)
    one = base.copy()
    one[one.index('--steps')+1]='1'
    learn.main(one)
    assert step_observations
    new_state = json.loads((new_output/'checkpoints'/'two_stage'/'step_00000001'/'training_state.json').read_text())
    assert new_state['data_provenance']['warm_start']['source_step'] == 2
    assert new_state['data_provenance']['warm_start']['source_supervised_tokens'] > 0
    assert new_state['optimizer_step']==1
    assert new_state['tokenizer_hash']==json.loads((source/'training_state.json').read_text())['tokenizer_hash']
    assert new_state['data_provenance']['mixture_root']==str(expanded.resolve())
    # Full resume from the new checkpoint works while keeping the warm-start identity.
    learn.main(base+['--resume'])
    assert (new_output/'checkpoints'/'two_stage'/'step_00000002'/'training_state.json').exists()
    assert source.is_dir()
    # Ordinary checkpoint resume from the old run is not permitted with the new dataset.
    with pytest.raises((ValueError, FileNotFoundError)):
        learn.main(_argv(expanded, old_output, '--architecture','two_stage','--resume'))


def test_weight_only_warm_start_rejects_tokenizer_mismatch(prepared, tmp_path):
    old_output = tmp_path/'old'
    learn.main(_argv(prepared, old_output, '--architecture','two_stage'))
    source = old_output/'checkpoints'/'two_stage'/'step_00000002'
    expanded = tmp_path/'different_tokenizer'
    import shutil
    shutil.copytree(prepared, expanded)
    (expanded/'tokenizer.json').write_text('{"toy":false}',encoding='utf8')
    # Make the expanded dataset internally consistent, but mismatched with
    # the original checkpoint tokenizer identity.
    from mini_llm.predictive_refinement.real_data import sha256
    manifest_path = expanded/'mixture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['tokenizer_sha256'] = sha256(expanded/'tokenizer.json')
    manifest_path.write_text(json.dumps(manifest))
    new_output = tmp_path/'warm'
    with pytest.raises(ValueError, match='tokenizer hash mismatch'):
        learn.main(_argv(expanded, new_output, '--architecture', 'two_stage',
                         '--warm-start-checkpoint', str(source)))
