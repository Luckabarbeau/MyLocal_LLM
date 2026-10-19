"""0082O: instrumentation must not change loss/gradients or training semantics."""

import csv

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from examples.benchmark_probability_fusion_batches_0082m import (
    FusionStepTimeline, _summarize_stage_profiles, _write_profile_csv,
    run_batch,
)


def test_timeline_measurements_have_expected_stages():
    timeline = FusionStepTimeline(gpu=False)
    timeline.start()
    timeline.mark('optimizer.zero_grad')
    timeline.mark('forward.input_softmax')
    times, host, peak = timeline.snapshot()
    assert list(times) == ['optimizer.zero_grad', 'forward.input_softmax']
    assert list(host) == list(times)
    assert all(t >= 0 for t in times.values())
    assert peak == 0
    rows = _summarize_stage_profiles([(times, host, peak)] * 3,
                                      median_update_ms=50.0, batch_rows=32)
    assert len(rows) == 2
    assert rows[1]['stage'] == 'forward.input_softmax'
    assert rows[1]['share_pct'] >= 0.0


def test_profiled_and_unprofiled_identical_grads():
    if BACKEND_NAME != 'numpy':
        pytest.skip('numpy reference checks only')
    cfg = FusionConfig(vocab_size=31, max_models=3, seed=4)
    a = ProbabilityResidualCombiner(cfg)
    b = ProbabilityResidualCombiner(cfg)
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(5, 2, 31)).astype(np.float32)
    targets = rng.integers(0, 31, size=5, dtype=np.int64)
    mids = np.array([0, 1], dtype=np.int64)
    a.zero_grad()
    plain = a.loss_and_backward(inputs, mids, targets, return_weighted=True,
                                defer_metrics=True, check_finite=False)
    timeline = FusionStepTimeline(gpu=False)
    timeline.start()
    b.zero_grad()
    timeline.mark('optimizer.zero_grad')
    profiled = b.loss_and_backward(inputs, mids, targets, return_weighted=True,
                                   defer_metrics=True, check_finite=False,
                                   profile=timeline)
    times, _, _ = timeline.snapshot()
    assert list(times) == [
        'optimizer.zero_grad',
        'forward.input_softmax',
        'forward.reference_features',
        'forward.vocabulary_moments',
        'forward.member_encoder_aggregate',
        'forward.assemble_features',
        'forward.residual_decoder',
        'forward.output_softmax',
        'backward.loss_and_seed',
        'backward.mixture_weights',
        'backward.output_head',
        'backward.decoder',
        'backward.member_encoder',
        'backward.member_parameter_grads',
        'backward.vocabulary_embedding_grads',
    ]
    for x, y in zip(plain, profiled):
        np.testing.assert_array_equal(asnumpy(x), asnumpy(y))
    for pa, pb in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(asnumpy(pa.grad), asnumpy(pb.grad))


def test_profile_csv_and_run_batch(tmp_path):
    if BACKEND_NAME != 'numpy':
        pytest.skip('small NumPy-only regression; CUDA profiling tested on target GPU')
    result = run_batch(2, members=2, vocab=32, warmup=1, repeats=2,
                       clip=1.0, profile=True)
    assert len(result) == 4
    elapsed, tok_s, stages, peak = result
    assert elapsed > 0 and tok_s > 0 and peak == 0
    assert stages[0]['stage'] == 'optimizer.zero_grad'
    assert stages[-1]['stage'] == 'optimizer.adamw_step'
    assert any(row['stage'].startswith('forward.') for row in stages)
    assert any(row['stage'].startswith('backward.') for row in stages)
    path = tmp_path / 'nested' / 'profile.csv'
    _write_profile_csv(path, [{'batch_rows': 2, 'stages': stages,
                               'median_update_ms': elapsed * 1000.,
                               'peak_used_bytes': peak}])
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(stages)
    assert rows[0]['stage'] == 'optimizer.zero_grad'
    assert float(rows[0]['gpu_ms']) >= 0.0
    assert len(run_batch(2, members=2, vocab=32, warmup=0, repeats=1,
                         clip=1.0, profile=False)) == 2


def test_profile_stage_list_must_be_consistent():
    with pytest.raises(ValueError, match='stage names changed'):
        _summarize_stage_profiles([({'a': 1.}, {'a': 1.}, 0),
                                   ({'b': 2.}, {'b': 2.}, 0)],
                                  median_update_ms=10., batch_rows=2)


def test_context_attention_stages_do_not_change_gradients():
    if BACKEND_NAME != 'numpy':
        pytest.skip('numpy reference checks only')
    cfg = FusionConfig(vocab_size=27, max_models=2, context_heads=2, seed=7)
    plain_model = ProbabilityResidualCombiner(cfg)
    profiled_model = ProbabilityResidualCombiner(cfg)
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(3, 2, 27)).astype(np.float32)
    targets = rng.integers(0, 27, size=3, dtype=np.int64)
    tokens = rng.integers(0, 27, size=(3, 7), dtype=np.int64)
    lengths = np.array([7, 4, 6], dtype=np.int64)
    kwargs = {'return_weighted': True, 'defer_metrics': True,
              'context_tokens': tokens, 'context_lengths': lengths}
    plain_model.zero_grad()
    expected = plain_model.loss_and_backward(logits, [0, 1], targets, **kwargs)
    profiled_model.zero_grad()
    timeline = FusionStepTimeline(gpu=False)
    timeline.start()
    actual = profiled_model.loss_and_backward(logits, [0, 1], targets,
                                               profile=timeline, **kwargs)
    times, _, _ = timeline.snapshot()
    assert 'forward.model_context_attention' in times
    assert 'backward.model_context_attention' in times
    for x, y in zip(expected, actual):
        np.testing.assert_array_equal(asnumpy(x), asnumpy(y))
    for a, b in zip(plain_model.parameters(), profiled_model.parameters()):
        np.testing.assert_array_equal(asnumpy(a.grad), asnumpy(b.grad))
