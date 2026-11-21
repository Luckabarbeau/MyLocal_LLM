"""CPU diagnostic bookkeeping tests; no model/native library is loaded here."""
import numpy as np
import pytest
from scripts.probe_0335_cpu_expert_ab import metrics, paired_samples, summaries


def test_metrics_uses_fp64_and_reference_denominator():
    got = metrics(np.array([4, 3], np.float32), np.array([3, 4], np.float32))
    assert got['max_abs'] == 1
    assert got['relative_l2'] == pytest.approx(np.sqrt(2) / 5)
    assert got['reference_norm'] == 5


@pytest.mark.parametrize('a,b', [([1, 2], [[1, 2]]), ([float('nan')], [0]),
                               ([0], [float('inf')])])
def test_metrics_rejects_incomparable_inputs(a, b):
    with pytest.raises(ValueError):
        metrics(a, b)


def test_zero_baseline_is_finite():
    assert metrics([0], [0])['relative_l2'] == 0
    assert np.isfinite(metrics([1], [0])['relative_l2'])


def test_paired_samples_alternates_order_and_separates_warmup():
    calls = []
    def execute(name, iterations):
        calls.append((name, iterations))
        return float(iterations) if name == 'a' else float(iterations) * 2
    result = paired_samples(execute, ('a', 'b'), 3, 10)
    assert calls[:8] == [('a', 1), ('b', 1)] * 4
    assert calls[8:] == [('a', 10), ('b', 10), ('b', 10), ('a', 10), ('a', 10), ('b', 10)]
    assert result == {'a': [10.] * 3, 'b': [20.] * 3}
    assert summaries(result)['a']['median_ms'] == 10


def test_three_arm_samples_preserve_all_outputs():
    calls = []
    result = paired_samples(lambda name, count: calls.append(name) or 1.,
                            ('local', 'actq', 'same_q8'), 4, 5)
    assert {name: len(values) for name, values in result.items()} == {
        'local': 4, 'actq': 4, 'same_q8': 4}
    assert calls[12:18] == ['local', 'actq', 'same_q8', 'same_q8', 'actq', 'local']
