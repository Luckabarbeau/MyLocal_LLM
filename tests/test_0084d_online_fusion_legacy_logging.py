"""0084D: online logging is schema-compatible with cached 0082F telemetry."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.train_probability_fusion_streaming_0082f import _metrics_csv_fields
from mini_llm.collective import online_training as online
from mini_llm.collective.online_logging import (OnlineFusionTelemetry,
        metrics_csv_fields, append_metrics_csv, reconcile_metrics_csv,
        comparison_lines)
from mini_llm.collective.probability_fusion import FusionConfig, ProbabilityResidualCombiner
from mini_llm.optim.adamw import AdamW


class FakeTeacher:
    def __init__(self, shift):
        self.shift = shift
        self.calls = 0

        class Projection:
            def __init__(self, shift):
                self.shift = shift
            def forward(self, x):
                x = np.asarray(x, dtype=np.float32)
                v = np.arange(11, dtype=np.float32)[None, None, :]
                return np.sin(v * .31 + x[..., :1] + self.shift), None

        self.output_proj = Projection(shift)

    def forward_body(self, ids, *, return_cache):
        assert return_cache is False
        self.calls += 1
        return np.asarray(ids, dtype=np.float32)[..., None] * .01 + self.shift


class FakeSampler:
    names = ['github', 'fineweb_edu']
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
    def batch(self, n):
        arr = np.tile(np.array([[1, 2, 3, 4, 5]], dtype=np.int64), (n, 1))
        return arr, ['github' if i % 2 == 0 else 'fineweb_edu' for i in range(n)]


def test_exact_0082f_csv_schema_and_numeric_segment():
    sources = ['fineweb_edu', 'github']
    ids = [0, 1]
    assert metrics_csv_fields(sources, ids) == _metrics_csv_fields(sources, ids)
    assert 'val_geometric_ce' in metrics_csv_fields(sources, ids)
    assert 'model_0_ce' in metrics_csv_fields(sources, ids)
    assert comparison_lines(1.1, 1.2, 1.3, 1.4, ids, [1.5, 1.6])[1].endswith('+0.10000')


def test_online_diagnostics_aligned_and_no_second_teacher_forward():
    teachers = [FakeTeacher(0.), FakeTeacher(.2)]
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=11, max_models=10,
                    context_heads=2, context_head_dim=4, context_hidden=6, seed=32))
    opt = AdamW(model.parameters(), lr=3e-3)
    results = online.train_online_step(model, opt, teachers, FakeSampler(2),
                    online.prediction_positions(4, 4, 1), rows_per_update=6,
                    teacher_batch_windows=1, batch_rows=3, lm_head_rows=2,
                    grad_clip=1.0, member_ids=[0, 1], diagnostics=True)
    assert [t.calls for t in teachers] == [2, 2]
    assert results['teacher_windows'] == 2
    assert sum(results['source_rows'].values()) == 6
    assert results['microbatches'] == 3
    assert len(results['member_ce']) == 2
    assert all(np.isfinite(x) for x in results['member_ce'])
    assert np.isfinite(results['geometric_ce'])
    np.testing.assert_allclose(sum(results['context_means']), 1., atol=1e-5)
    assert all(x >= 0 for x in results['context_stds'])
    val = online.evaluate_online(model, teachers, np.tile([[1, 2, 3, 4, 5]], (2, 1)),
                    online.prediction_positions(4, 4, 1), member_ids=[0, 1],
                    teacher_batch_windows=1, batch_rows=3, lm_head_rows=2)
    assert val['geometric_ce'] > 0
    assert len(val['context_stds']) == 2


def test_telemetry_rows_by_source_persistence_and_rates():
    meter = OnlineFusionTelemetry(['github', 'fineweb_edu'])
    meter.record_step(by_source={'github': 7, 'fineweb_edu': 1},
                      members=2, context_length=4, scored=4,
                      windows=2, wall_seconds=.10, teacher_seconds=.06)
    assert meter.total_rows == 8
    assert meter.teacher_input_tokens == 16
    assert meter.teacher_scored_tokens == 16
    assert meter.rates()['fusion_train_tok_s'] == pytest.approx(200)
    assert meter.rates()['fusion_end_to_end_tok_s'] == pytest.approx(80)
    meter.finish_interval()
    restored = OnlineFusionTelemetry(['github', 'fineweb_edu'], previous=meter.state())
    assert restored.total_rows == 8
    assert restored.source_rows['github'] == 7
    restored.record_step(by_source={'github': 4}, members=2, context_length=4,
                         scored=4, windows=1, wall_seconds=.1, teacher_seconds=.05)
    assert restored.total_rows == 12


def test_log_resume_reconciliation_and_schema_guard(tmp_path):
    fields = metrics_csv_fields(['github'], [0, 1])
    path = tmp_path / 'fusion.csv'
    append_metrics_csv(path, fields, {'step':1,'segment':0,'members':2})
    append_metrics_csv(path, fields, {'step':2,'segment':0,'members':2})
    reconcile_metrics_csv(path, fields, 1)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['step'] == '1'
    with pytest.raises(ValueError, match='incompatible columns'):
        append_metrics_csv(path, ['incorrect'], {'incorrect':1})


def test_entrypoint_legacy_style_csv_and_report(monkeypatch, tmp_path, capsys):
    teachers = [tmp_path / 'deep', tmp_path / 'wide']
    for path in teachers:
        path.mkdir()
    shard_root = tmp_path / 'shards'; shard_root.mkdir()
    (shard_root / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr(online, 'make_online_samplers', lambda *args: (
        {'train': FakeSampler(1), 'val': FakeSampler(4)},
        {'sources': {'github': {}, 'fineweb_edu': {}}}, {}))
    monkeypatch.setattr(online, 'validate_teacher_tokenizers',
                        lambda *args: ('hash', {'0': 'abc', '1': 'def'}))
    loaded = [FakeTeacher(0.), FakeTeacher(.2)]
    monkeypatch.setattr(online, 'load_frozen_teacher',
        lambda directory, length: (loaded[0] if directory.name == 'deep' else loaded[1],
                                   SimpleNamespace(vocab_size=11)))
    csv_path = tmp_path / 'fusion.csv'
    out = tmp_path / 'best.npz'
    cmd = ['--member',f'0:{teachers[0]}', '--member',f'1:{teachers[1]}',
           '--mixed-shard-root',str(shard_root), '--output',str(out),
           '--context-length','4','--score-last-k','4',
           '--batch-windows','1','--lm-head-rows','2',
           '--batch-rows','3','--grad-accum-steps','2',
           '--context-head-dim','4','--context-hidden','6',
           '--steps','3','--val-windows','2','--eval-interval','2',
           '--log-interval','2','--save-interval','1',
           '--log-file',str(csv_path)]
    online.main(cmd)
    report = capsys.readouterr().out
    assert 'Train CE' in report and 'Validation CE' in report
    assert 'Sources (train)' in report and 'Models' in report and 'Gains' in report
    assert 'Segment online' in report and 'Fusion speed' in report
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == _metrics_csv_fields(['fineweb_edu', 'github'], [0,1])
        rows = list(reader)
    assert [int(row['step']) for row in rows] == [1,2,3]
    assert sum(int(rows[-1][f'total_source_tokens_{s}']) for s in ['github','fineweb_edu']) == 18
    assert float(rows[-1]['val_geometric_ce']) > 0
    assert float(rows[-1]['model_1_ce']) > 0
    assert all(int(row['segment']) == 0 for row in rows)
    with np.load(tmp_path / 'best.last.npz', allow_pickle=False) as state:
        meta = json.loads(str(state['__meta_json__']))
    assert meta['telemetry']['total_rows'] == 18
    resumed = list(cmd)
    resumed[resumed.index('--steps') + 1] = '4'
    online.main(resumed + ['--resume'])
    with csv_path.open() as f:
        resumed_rows = list(csv.DictReader(f))
    assert [int(row['step']) for row in resumed_rows] == [1, 2, 3, 4]
    assert int(resumed_rows[-1]['total_supervised_tokens']) == 24
    assert int(resumed_rows[-1]['total_source_tokens_github']) + int(
        resumed_rows[-1]['total_source_tokens_fineweb_edu']) == 24
