"""0082L: fixed-width per-step logging, source grouping and unchanged counts."""
from collections import Counter

import numpy as np

from examples.train_probability_fusion_streaming_0082f import (
    _comparison_lines, _member_table, _print_step_block,
    _source_table, _step_header,
)


def test_left_step_header_and_relative_indentation(capsys):
    _print_step_block(1550, 4000, 3, 'TRAIN', [
        '    Train CE            fusion=3.41480',
        '    Fusion speed        interval=334.7 tok/s',
    ], members=2)
    text = capsys.readouterr().out
    assert text.startswith('Step 1550/4000 | Segment 3 | TRAIN | Members 2\n')
    lines = text.rstrip().splitlines()
    assert all(line.startswith('    ') for line in lines[1:])
    assert _step_header(9, 4000, 0, phase='VALIDATION').startswith('Step    9/4000')


def test_one_line_per_source_with_aligned_columns_and_correct_percentages():
    names = ['github', 'fineweb_edu', 'wikipedia']
    lines = _source_table(names, batch=Counter(github=22, fineweb_edu=36),
                          interval=Counter(github=262, fineweb_edu=407, wikipedia=229),
                          total=Counter(github=262, fineweb_edu=407, wikipedia=229),
                          total_rows=898, interval_seconds=5.354)
    assert len(lines) == len(names) + 1
    assert 'Batch' in lines[0] and 'Interval' in lines[0]
    assert 'tok/s' in lines[0] and 'Share' in lines[0]
    for name, row in zip(names, lines[1:]):
        assert name in row
        assert row.count(name) == 1
    assert 'fineweb_edu' in lines[2]
    assert '45.32%' in lines[2]  # 407 / 898
    assert '76.0' in lines[2]  # 407 / 5.354
    assert ' 0 ' in lines[3] or ' 0,' in lines[3] or '      0 ' in lines[3]


def test_producer_table_distinguishes_windows_scored_and_teacher_input():
    lines = _source_table(['github', 'fineweb'],
                          window_counts={'github': 18, 'fineweb': 25},
                          available_counts={'github': 9216, 'fineweb': 12800},
                          teacher_counts={'github': 18432, 'fineweb': 25600})
    assert len(lines) == 3
    assert 'Windows' in lines[0] and 'Available' in lines[0]
    assert '9,216' in lines[1] and '18,432' in lines[1]
    assert '12,800' in lines[2] and '25,600' in lines[2]


def test_model_table_identity_and_gain_signs():
    rows = _member_table(np.asarray([1, 0]), [2.9, 3.5],
                         global_weights=[.75, .25],
                         context_means=[.8, .2], context_stds=[.1, .1])
    assert 'Context mean' in rows[0]
    assert '2.90000' in rows[1] and '0.7500' in rows[1]
    gains = _comparison_lines(2.8, 3.1, 3.2, 3.0,
                              np.asarray([1, 0]), [2.9, 3.5])
    assert 'vs best model 1=+0.10000' in gains[1]
    assert 'vs best reference=+0.10000' in gains[1]
