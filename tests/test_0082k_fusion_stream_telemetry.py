"""0082K: teacher-token, fusion-supervision and source telemetry integrity."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples import train_probability_fusion_streaming_0082f as streaming


def test_window_major_rows_map_to_exact_source_and_not_vocabulary_count():
    trace = [('fineweb', 's1', 1), ('github', 's2', 10), ('fineweb', 's3', 20)]
    rows = [0, 2, 3, 5, 6, 6, 8]
    result = streaming._source_rows_for_batch(rows, trace, scored=3)
    assert dict(result) == {'fineweb': 5, 'github': 2}
    assert sum(result.values()) == len(rows)
    with pytest.raises(ValueError, match='mapped'):
        streaming._source_rows_for_batch([9], trace, scored=3)
    with pytest.raises(ValueError, match='positive'):
        streaming._source_rows_for_batch([0], trace, scored=0)


def test_metrics_separate_teacher_input_teacher_scored_and_fusion_labels():
    meter = streaming._FusionTelemetry(['fineweb', 'github'])
    meter.exported(n_windows=32, context_length=512, scored=128,
                   n_members=2, seconds=4.0)
    assert meter.teacher_input_tokens == 32768
    assert meter.teacher_scored_tokens == 8192
    meter.trained({'fineweb': 24, 'github': 8})
    meter.train_seconds(0.5)
    metrics = meter.rates(active_wall_seconds=6.0)
    assert meter.total_rows == 32  # one supervised label per row
    assert metrics['fusion_train_tok_s'] == pytest.approx(64.)
    assert metrics['fusion_end_to_end_tok_s'] == pytest.approx(32/6)
    assert meter.interval_source_rows['github'] == 8
    meter.finish_interval()
    meter.trained({'github': 4})
    meter.train_seconds(1)
    assert meter.rates(8)['fusion_train_tok_s'] == pytest.approx(4.)
    state = meter.state()
    resumed = streaming._FusionTelemetry(['fineweb', 'github'], previous=state)
    assert resumed.total_rows == 36
    assert resumed.source_rows['github'] == 12
    assert resumed.teacher_scored_tokens == 8192
    assert resumed.interval_rows == 0  # correct epoch boundary


def test_csv_resume_discards_non_checkpointed_rows(tmp_path):
    src = ['fineweb', 'github']
    fields = streaming._metrics_csv_fields(src, [0, 1])
    path = tmp_path/'logs.csv'
    for step in (2, 4, 6):
        streaming._append_metrics_csv(path, fields, {'step': step, 'ce': 1.2})
    streaming._reconcile_metrics_csv(path, fields, saved_step=4)
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r['step']) for r in rows] == [2, 4]
    with pytest.raises(ValueError, match='incompatible'):
        streaming._append_metrics_csv(path, fields + ['unexpected'], {'step': 8})


class _Sampler:
    def __init__(self, rng, context):
        self.rng = rng
        self.context = context
    def sample(self):
        x = self.rng.integers(1, 11, size=self.context+1, dtype=np.int32)
        return x, Path('sample.bin'), int(self.rng.integers(0, 1000))


def test_complete_csv_stream_and_resumption_track_exact_batch_sources(tmp_path, monkeypatch, capsys):
    root = tmp_path/'corpus'
    root.mkdir()
    manifest = tmp_path/'windows_metadata.json'
    manifest.write_text('{}')
    tokenizer = b'{"shared":"tokenizer"}'
    digest = hashlib.sha256(tokenizer).hexdigest()
    members = []
    val = []
    rng = np.random.default_rng(1)
    for mid in (0, 1):
        ck = tmp_path/f'ck{mid}'
        ck.mkdir()
        (ck/'tokenizer.json').write_bytes(tokenizer)
        members.append(f'{mid}:{ck}')
        vf = tmp_path/f'val_{mid}.npy'
        np.save(vf, rng.normal(size=(6, 11)).astype('float16'))
        val.append(f'{mid}:{vf}')
    y = tmp_path/'validation_targets.npy'
    np.save(y, rng.integers(0, 11, size=6))

    def fake_samplers(root, metadata, seed, context, rng):
        sources = [type('Source', (), {'name': n, 'weight': 0.5})()
                   for n in ('fineweb', 'github')]
        return sources, {'vocab_size':11, 'tokenizer_sha256':digest}, {
            n: _Sampler(rng, context) for n in ('fineweb', 'github')
        }, np.array([0.5, 0.5])

    def fake_export(argv):
        windows = np.load(argv[argv.index('--windows')+1])
        output = Path(argv[argv.index('--output-dir')+1])
        output.mkdir(parents=True, exist_ok=True)
        positions = streaming._scored_positions(4, score_last_k=4)
        labels = windows[:, positions+1].reshape(-1)
        np.save(output/'targets.npy', labels)
        for mid in (0, 1):
            logits = np.zeros((len(labels), 11), dtype='float16')
            logits[np.arange(len(labels)), labels] = 1+mid
            np.save(output/f'model_{mid}_logits.npy', logits)
        (output/'metadata.json').write_text(json.dumps({
            'rows':len(labels), 'tokenizer_sha256':digest,
            'token_windows_sha256':'synthetic'}))

    monkeypatch.setattr(streaming, '_stream_samplers', fake_samplers)
    monkeypatch.setattr(streaming, 'export_main', fake_export)
    csvfile = tmp_path/'logs'/'fusion.csv'
    args = ['--member', members[0], '--member', members[1],
            '--mixed-shard-root', str(root), '--windows-metadata', str(manifest),
            '--val-member', val[0], '--val-member', val[1], '--val-targets', str(y),
            '--output', str(tmp_path/'best.npz'),
            '--work-dir', str(tmp_path/'cache'),
            '--context-length', '4', '--score-last-k', '4',
            '--batch-windows', '2', '--segment-windows', '2',
            '--batch-rows', '2', '--max-cache-gib', '0.01',
            '--eval-interval', '2', '--log-interval', '1',
            '--early-stop-patience', '0', '--log-file', str(csvfile),
            '--steps', '5']
    streaming.main(args)
    out = capsys.readouterr().out
    for field in ('TEACHER EXPORT', 'Token positions', 'Fusion speed',
                  'Supervised tokens', 'Sources (train)', 'Sources (export)',
                  'Validation CE', 'fineweb', 'github', 'fusion_stream_done'):
        assert field in out
    with csvfile.open(newline='') as fh:
        logged = list(csv.DictReader(fh))
    assert len(logged) == 5
    assert int(logged[-1]['total_supervised_tokens']) == 10
    assert int(logged[-1]['teacher_input_tokens_total']) == 32
    assert int(logged[-1]['teacher_scored_tokens_total']) == 32
    for row in logged:
        assert int(row['batch_supervised_tokens']) == 2
        assert int(row['batch_source_tokens_fineweb']) + int(row['batch_source_tokens_github']) == 2
        assert float(row['fusion_train_tok_s']) > 0
        assert float(row['fusion_end_to_end_tok_s']) > 0
        if int(row['step']) % 2 == 0 or int(row['step']) == 1 or int(row['step']) == 5:
            assert row['val_ce'] != ''
    with np.load(tmp_path/'best_last.npz') as ck:
        meta = json.loads(str(ck['__meta_json__']))
    assert meta['telemetry']['total_rows'] == 10
    assert sum(meta['telemetry']['source_rows'].values()) == 10
    # Simulate CSV output produced after an interrupted, uncheckpointed segment.
    fields = list(logged[0])
    streaming._append_metrics_csv(csvfile, fields, {'step': 99, 'ce': 12})
    streaming.main(args[:-2] + ['--steps', '8', '--resume'])
    with csvfile.open(newline='') as fh:
        logged = list(csv.DictReader(fh))
    assert [int(r['step']) for r in logged] == list(range(1, 9))
    assert int(logged[-1]['total_supervised_tokens']) == 16
    assert sum(int(logged[-1][f'total_source_tokens_{s}']) for s in ('fineweb', 'github')) == 16
