"""Versioned v2 checkpoint & decoder-inclusive v1 weights-only migration."""
import json
from pathlib import Path
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.block_proposal.decoder import BlockPrefixDecoder,TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.block_proposal.checkpoint import (
    ARCHITECTURE,ARCHITECTURE_V2,architecture_metadata,
    save_two_stage_checkpoint,load_two_stage_checkpoint,warmstart_v2_from_v1)
from test_block_proposal_0080l_baselines import _configuration

pytestmark=pytest.mark.skipif(BACKEND_NAME!='numpy',reason='CPU v2 checkpoint contract')


def network(variant='v1', source='own'):
    main=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    proposal=HorizonProposalHead(main,4,hidden_width=12,rng_seed=8081)
    kwargs={}
    if variant!='v1':
        kwargs=dict(soft_interface_config=SoftInterfaceConfig(
            mode='full_vocab' if variant in ('full','both') else 'expectation',
            source_scope=source,read_heads=4,
            boundary_condition='dec_residual' if variant in ('boundary','both') else 'off'),
            experimental_cpu_full_vocab=True)
    decoder=BlockPrefixDecoder(main,4,width=12,n_layers=2,n_heads=3,
                               rng_seed=8082,**kwargs)
    return TwoStageBlockModel(proposal,decoder)


def dataset():
    rng=np.random.default_rng(902)
    hidden=rng.normal(size=(2,16))
    targets=rng.integers(0,101,size=(2,4),dtype=np.int64)
    return hidden,targets,np.ones((2,4),dtype=bool)


@pytest.mark.parametrize('variant',['v1','full','boundary','both'])
def test_v1_metadata_unchanged_and_v2_has_explicit_interface(variant):
    net=network(variant)
    meta=architecture_metadata(net)
    assert meta['type']==(ARCHITECTURE if variant=='v1' else ARCHITECTURE_V2)
    if variant=='v1':
        assert 'soft_interface' not in meta and 'vocab_read_execution' not in meta
    else:
        assert meta['soft_interface']['boundary_condition']==(
            'dec_residual' if variant in ('boundary','both') else 'off')
        assert ('vocab_read_execution' in meta)==(variant!='boundary')


@pytest.mark.parametrize('variant',['full','boundary','both'])
def test_v1_full_decoder_warmstart_and_output_identity(tmp_path,variant):
    old=network('v1')
    ckpt=tmp_path/'v1'
    save_two_stage_checkpoint(ckpt,old,tokenizer_hash='same-hash')
    fresh=network(variant)
    original_names={p.name for p in old.parameters()}
    old_weights={p.name:p.data.copy() for p in old.parameters()}
    info=warmstart_v2_from_v1(ckpt,fresh,expected_tokenizer_hash='same-hash')
    assert info['loaded']==len(original_names)
    for p in fresh.parameters():
        if p.name in old_weights:
            np.testing.assert_array_equal(p.data,old_weights[p.name])
    hidden,targets,valid=dataset()
    baseline,_=old.forward(hidden,targets,valid)
    extended,_=fresh.forward(hidden,targets,valid)
    np.testing.assert_array_equal(extended['final_position_ce_sums'],baseline['final_position_ce_sums'])
    assert np.isclose(extended['final_ce'],baseline['final_ce'],atol=0,rtol=0)
    assert info['initialized_v2']
    with pytest.raises(ValueError,match='tokenizer'):
        warmstart_v2_from_v1(ckpt,network(variant),expected_tokenizer_hash='wrong-hash')
    damaged=tmp_path/'damaged'
    import shutil
    shutil.copytree(ckpt,damaged)
    first=next((damaged/'model_params').glob('block_decoder.*.npy'))
    first.unlink()
    with pytest.raises(ValueError,match='incomplete/unexpected'):
        warmstart_v2_from_v1(damaged,network(variant),expected_tokenizer_hash='same-hash')


def test_updated_v2_roundtrip_parameters_optimizer_and_forward(tmp_path):
    model=network('both','all')
    optimizer=AdamW(model.parameters(),lr=.001,weight_decay=.1)
    hidden,targets,valid=dataset()
    model.zero_grad()
    output,tape=model.forward(hidden,targets,valid)
    model.backward(tape)
    optimizer.step()
    assert optimizer.step_index==1
    expected=dict((p.name,p.data.copy()) for p in model.parameters())
    output,_=model.forward(hidden,targets,valid)
    save_two_stage_checkpoint(tmp_path/'v2',model,optimizer=optimizer,training_step=1,
                              tokenizer_hash='same-hash')
    loaded=network('both','all')
    opt_loaded=AdamW(loaded.parameters(),lr=.001,weight_decay=.1)
    state=load_two_stage_checkpoint(tmp_path/'v2',loaded,optimizer=opt_loaded,
                                    expected_tokenizer_hash='same-hash')
    assert state['step']==1 and opt_loaded.step_index==optimizer.step_index
    for p in loaded.parameters():
        np.testing.assert_array_equal(p.data,expected[p.name])
    for a,b in zip(optimizer.m,opt_loaded.m):np.testing.assert_array_equal(a,b)
    for a,b in zip(optimizer.v,opt_loaded.v):np.testing.assert_array_equal(a,b)
    result,_=loaded.forward(hidden,targets,valid)
    np.testing.assert_array_equal(result['final_position_ce_sums'],output['final_position_ce_sums'])
    with pytest.raises(ValueError,match='incompatible 0080 architecture'):
        load_two_stage_checkpoint(tmp_path/'v2',network('full'),expected_tokenizer_hash='same-hash')


def test_extended_trainer_v2_real_window_optimizer_and_checkpoint(tmp_path):
    """A real packed-document production optimizer step, not just a head VJP."""
    from mini_llm.train_extended import ExtendedTrainer
    from mini_llm.block_proposal.objective import BlockProposalObjective
    from test_block_proposal_0080n_production import _docs, _close
    source=_docs(tmp_path)
    live=network('both','all')
    reloaded=network('both','all')
    def trainer(net, checkpoint_dir=None):
        return ExtendedTrainer(
            model=net.proposal_head.model,
            training_objective=BlockProposalObjective(net,block_group_size=2),
            train_shard_paths=[str(source)],val_shard_paths=[str(source)],
            batch_size=2,seq_length=24,grad_accum_steps=2,
            gradient_accumulation_weighting='supervised_tokens',
            total_steps=5,warmup_steps=0,peak_lr=1e-4,grad_clip=10.,
            eos_token_id=100,rng_seed=101,checkpoint_dir=checkpoint_dir)
    first=trainer(live,str(tmp_path/'v2production'))
    second=trainer(reloaded)
    try:
        loss,norm=first.train_step()
        assert np.isfinite(loss) and np.isfinite(norm)
        assert first.optimizer.step_index==1
        assert first._block_proposal_supervised_tokens>0
        first.save()
        metadata=load_two_stage_checkpoint(tmp_path/'v2production',reloaded,
                                           optimizer=second.optimizer)
        assert metadata['architecture']['type']==ARCHITECTURE_V2
        for a,b in zip(live.parameters(),reloaded.parameters()):
            np.testing.assert_array_equal(a.data,b.data)
        assert first.optimizer.step_index==second.optimizer.step_index
    finally:
        _close(first);_close(second)


def test_warmstart_resolves_exact_source_decoder_dimensions(tmp_path):
    from types import SimpleNamespace
    from train_model import populate_v1_warmstart_options
    source=network('v1')
    path=tmp_path/'v1'
    save_two_stage_checkpoint(path,source,tokenizer_hash='sha')
    args=SimpleNamespace(block_proposal_init_from_v1=str(path),
                         block_proposal_k=None,block_proposal_decoder_width=None)
    original=populate_v1_warmstart_options(args)
    assert args.block_proposal_k==4
    assert args.block_proposal_decoder_width==12
    assert args.block_proposal_decoder_layers==2
    assert args.block_proposal_head_width==12
    assert args.block_proposal_decoder_ff_multiplier==4
    assert original['type']==ARCHITECTURE
    args.block_proposal_decoder_width=512
    with pytest.raises(ValueError,match='warm-start architecture option mismatch'):
        populate_v1_warmstart_options(args)
