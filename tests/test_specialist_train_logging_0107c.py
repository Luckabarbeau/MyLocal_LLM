"""0107C: paired on-training-set frozen baseline and production-style logging."""
import json

import numpy as np
import pytest

from mini_llm.specialists.training_run import SpecialistTrainingRun
from mini_llm.specialists.train_logging_0107c import SpecialistIntervalLogger
from test_specialist_training_run_0105e import prepare


def test_source_metrics_and_frozen_ce_identical_training_microbatches(tmp_path, monkeypatch):
    foundation, comp, sampler, cfg = prepare(tmp_path)
    run = SpecialistTrainingRun(comp, sampler, config=cfg)
    # A CE diagnostic must never trigger a second full foundation forward pass.
    calls = []
    original_hidden = type(foundation).hidden

    def count_hidden(self, *args, **kwargs):
        calls.append(1)
        return original_hidden(self, *args, **kwargs)

    monkeypatch.setattr(type(foundation), 'hidden', count_hidden)
    before = {p.name: p.data.copy() for p in foundation.model.parameters()}
    report = run.train_step(source_metrics=True, foundation_train_ce=True,
                            measure_step_time=True)
    assert len(calls) == cfg.grad_accum_steps
    assert report['foundation_ce'] == pytest.approx(report['loss'], abs=1e-6)
    assert report['foundation_gain_ce'] == pytest.approx(0.0, abs=1e-6)
    assert report['supervised_tokens_step'] > 0
    assert report['step_time_s'] > 0
    assert report['executed_tokens_step'] <= report['capacity_tokens_step']
    assert sum(v['supervised_tokens'] for v in report['per_source'].values()) == report['supervised_tokens_step']
    for source in report['per_source'].values():
        assert source['sampled_document_spans'] >= 1
        assert source['sampled_document_span_tokens_sum'] > 0
        assert source['target_capacity_tokens'] >= source['supervised_tokens']
        assert source['loss_nll_sum'] == pytest.approx(source['foundation_nll_sum'], abs=1e-5)
    for param in foundation.model.parameters():
        assert param.grad is None
        np.testing.assert_array_equal(before[param.name], param.data)
    logger = SpecialistIntervalLogger(log_interval=1, total_steps=cfg.total_steps)
    assert logger.observe(report)
    text = '\n'.join(logger.flush())
    for marker in ('Step 1/8:', 'loss=', 'lr=', 'grad_norm=',
                   'grad_weighting=supervised_tokens', 'target_fill=',
                   'supervised_total=', 'step_time=', 'supervised_tok/s=',
                   'executed_tok/s=', 'capacity_tok/s=',
                   'Per-source train CE', 'sampled_doc_mean=',
                   'foundation_train_ce=', 'foundation_CE='):
        assert marker in text
    json.dumps(report)


def test_paired_logging_does_not_modify_gradient_updates(tmp_path):
    (tmp_path/'reference').mkdir()
    (tmp_path/'paired').mkdir()
    _, comp_a, sampler_a, cfg_a = prepare(tmp_path/'reference')
    _, comp_b, sampler_b, cfg_b = prepare(tmp_path/'paired')
    a = SpecialistTrainingRun(comp_a, sampler_a, config=cfg_a)
    b = SpecialistTrainingRun(comp_b, sampler_b, config=cfg_b)
    for _ in range(2):
        normal = a.train_step()
        paired = b.train_step(source_metrics=True, foundation_train_ce=True)
        assert normal['supervised_tokens_step'] == paired['supervised_tokens_step']
        assert normal['loss'] == pytest.approx(paired['loss'], abs=1e-7)
    for a_param, b_param in zip(a.parameters, b.parameters):
        np.testing.assert_allclose(a_param.data, b_param.data, atol=1e-7, rtol=1e-7)


def test_source_interval_weighted_aggregation_and_partial_flush(tmp_path):
    _, comp, sampler, cfg = prepare(tmp_path)
    run = SpecialistTrainingRun(comp, sampler, config=cfg)
    logger = SpecialistIntervalLogger(log_interval=3, total_steps=cfg.total_steps)
    reports = run.train_steps(2, source_metrics=True, foundation_train_ce=True,
                              measure_step_time=True)
    assert not logger.observe(reports[0])
    assert not logger.observe(reports[1])
    lines = logger.flush()
    assert lines[0].startswith('Step 2/8:')
    for name in sorted({source for r in reports for source in r['per_source']}):
        assert any(f'{name}: CE=' in line for line in lines)
    assert logger.flush() == []
    # The log interval must be positive; zero could never flush.
    with pytest.raises(ValueError, match='log_interval'):
        SpecialistIntervalLogger(log_interval=0)


def test_cli_flags_parse_without_running():
    from scripts.train_0105_specialist import arguments
    args = arguments(['--foundation-checkpoint', '/tmp/foundation',
                      '--mixed-shard-root', '/tmp/mix',
                      '--specialist-checkpoint', '/tmp/specialist',
                      '--train-foundation-ce', '--log-interval', '3',
                      '--log-text-file', '/tmp/train.log'])
    assert args.train_foundation_ce is True
    assert args.log_interval == 3
    assert args.log_text_file.name == 'train.log'


def test_streaming_large_runs_do_not_retain_step_reports(tmp_path):
    _, comp, sampler, cfg = prepare(tmp_path)
    run = SpecialistTrainingRun(comp, sampler, config=cfg)
    seen = []
    returned = run.train_steps(2, source_metrics=True, measure_step_time=True,
                               on_step=lambda record: seen.append(record['step']),
                               retain_reports=False)
    assert returned == []
    assert seen == [1, 2]
