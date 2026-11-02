"""Small CPU tests of specialist updates, checkpoint/resume and freeze parity."""
from dataclasses import asdict
import json
import numpy as np
import pytest

from mini_llm.checkpoint import save_checkpoint
from mini_llm.config import ModelConfig, DualResolutionConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.specialists import (FrozenFoundation, ResidualSpecialist,
    SpecialistConfig, FrozenResidualComposition)
from mini_llm.specialists.training_run import SpecialistTrainingRun, SpecialistTrainingConfig
from mini_llm.specialists.document_sampler import SpecialistDocumentSampler
from test_specialist_sampler_0105d import mixture


def prepare(tmp_path):
    cfg = ModelConfig(tokenizer_vocab_size=64, context_length=32,
        n_layers=1, d_model=16, d_ff=32, n_q_heads=2, n_kv_heads=1,
        d_head=8, n_experts=1, top_k=1, ffn_mode='dense', dtype='float32',
        dual_resolution=DualResolutionConfig(enabled=True, block_size=4,
        local_window=8, compression_dim=8, compression_heads=2,
        compression_mode='matrix_pool', global_heads=1, global_head_dim=8,
        max_blocks=16, global_query_chunk=4))
    base = tmp_path/'foundation'; base.mkdir()
    (base/'config.json').write_text(json.dumps(asdict(cfg)))
    model = DecoderLanguageModel(cfg, rng_seed=2)
    save_checkpoint(base, {p.name: p.data for p in model.parameters()})
    foundation = FrozenFoundation.from_checkpoint(base)
    s = ResidualSpecialist(SpecialistConfig(
        specialist_id='general', n_layers=2, d_model=16, d_ff=32,
        n_q_heads=2, n_kv_heads=1, d_head=8,
        full_window=16, local_window=8, dtype='float32'), rng_seed=5)
    comp = FrozenResidualComposition(foundation, s)
    root = mixture(tmp_path)
    sampler = SpecialistDocumentSampler(root, context_length=32,batch_size=1,
                                        eos_token_id=63,block_size=4)
    tcfg = SpecialistTrainingConfig(total_steps=8,warmup_steps=1,
                                    grad_accum_steps=2, muon_matmul_precision='fp32')
    return foundation, comp, sampler, tcfg


def test_freeze_training_and_deterministic_resume(tmp_path):
    base, comp, sampler, cfg = prepare(tmp_path)
    fixed = {p.name: p.data.copy() for p in base.model.parameters()}
    run = SpecialistTrainingRun(comp,sampler,config=cfg,
                                checkpoint_dir=tmp_path/'specialist')
    a = run.train_step()
    assert a['supervised_tokens_step'] > 0
    assert a['step'] == 1
    assert a['loss'] > 0
    assert a['grad_norm'] > 0
    assert run.optimizer.step_index == 1
    assert sum(run.source_tokens.values()) == run.supervised_tokens
    assert all(p.grad is None for p in base.model.parameters())
    for p in base.model.parameters():
        np.testing.assert_array_equal(fixed[p.name],p.data)
    run.save_checkpoint()
    restarted = SpecialistTrainingRun.resume(tmp_path/'specialist',base,
        SpecialistDocumentSampler(sampler.root,context_length=32,batch_size=1,
                                   eos_token_id=63,block_size=4,seed=99), config=cfg)
    assert restarted.step==1
    assert restarted.supervised_tokens==run.supervised_tokens
    next_a = run.train_step()
    next_b = restarted.train_step()
    assert next_a['supervised_tokens']==next_b['supervised_tokens']
    np.testing.assert_allclose(next_a['loss'],next_b['loss'],atol=1e-7)
    for p,q in zip(run.parameters,restarted.parameters):
        np.testing.assert_array_equal(p.data,q.data)
    with pytest.raises(ValueError, match='foundation'):
        SpecialistTrainingRun(comp,sampler,config=cfg,checkpoint_dir=base.checkpoint_path)


def test_specialist_validation_does_not_update_parameters(tmp_path):
    base, comp, sampler, cfg = prepare(tmp_path)
    run = SpecialistTrainingRun(comp,sampler,config=cfg)
    before={p.name:p.data.copy() for p in run.parameters}
    state = sampler.checkpoint_state()
    paired = run.evaluate_paired(batches=2)
    assert paired['foundation_ce'] == pytest.approx(paired['specialist_ce'], abs=1e-7)
    assert paired['gain_ce'] == pytest.approx(0.0, abs=1e-7)
    assert sampler.checkpoint_state()['cursors'] == state['cursors']
    score=run.evaluate(batches=2)
    assert score>0
    for p in run.parameters:
        np.testing.assert_array_equal(before[p.name],p.data)
