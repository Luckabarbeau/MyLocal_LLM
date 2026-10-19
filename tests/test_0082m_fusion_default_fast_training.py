"""0082M: default streaming optimizer path stays numerically equivalent."""

import numpy as np
import pytest

from mini_llm.backend import asnumpy, scalar
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from examples.train_probability_fusion_0082a import _batch, _clip_grad_norm_
from examples.benchmark_probability_fusion_batches_0082m import main as benchmark_main
from examples.train_probability_fusion_streaming_0082f import main as streaming_main


def test_default_fast_metrics_and_clipping_match_original():
    rng = np.random.default_rng(50)
    logits = rng.normal(size=(4, 2, 11)).astype(np.float32)
    targets = np.array([1, 3, 2, 5], np.int64)
    ids = [0, 1]
    reference = ProbabilityResidualCombiner(FusionConfig(vocab_size=11))
    fast = ProbabilityResidualCombiner(FusionConfig(vocab_size=11))
    for model in (reference, fast):
        model.zero_grad()
    ce, arith, weighted = reference.loss_and_backward(
        logits, ids, targets, return_weighted=True)
    legacy_norm = _clip_grad_norm_(reference.parameters(), .1)
    ce_fast, arith_fast, weighted_fast = fast.loss_and_backward(
        logits, ids, targets, return_weighted=True, defer_metrics=True,
        check_finite=False)
    device_norm = _clip_grad_norm_(fast.parameters(), .1, device_only=True)
    np.testing.assert_allclose(
        [ce, arith, weighted, legacy_norm],
        [scalar(ce_fast), scalar(arith_fast), scalar(weighted_fast), scalar(device_norm)],
        rtol=1e-5, atol=1e-6)
    for p0, p1 in zip(reference.parameters(), fast.parameters()):
        np.testing.assert_allclose(asnumpy(p0.grad), asnumpy(p1.grad), rtol=1e-5, atol=1e-6)


def test_logits_validated_once_before_device_upload():
    arrays = [np.ones((2, 5), dtype=np.float32), np.zeros((2, 5), dtype=np.float32)]
    good = _batch(arrays, np.array([0, 1]), [0, 1], validate_finite=True)
    assert good.shape == (2, 2, 5)
    arrays[1][1, 2] = np.nan
    with pytest.raises(ValueError, match='finite'):
        _batch(arrays, np.array([0, 1]), [0, 1], validate_finite=True)


def test_no_opt_in_fast_gpu_flag_is_needed():
    with pytest.raises(SystemExit) as err:
        streaming_main(['--help'])
    assert err.value.code == 0


def test_batch_size_benchmark_runs_on_tiny_cpu(capsys):
    benchmark_main(['--batch-sizes', '2,4', '--members', '2', '--vocab', '7',
                    '--warmup', '0', '--repeats', '1'])
    out = capsys.readouterr().out
    assert 'supervised_tok/s' in out
    assert 'batch_rows' in out
