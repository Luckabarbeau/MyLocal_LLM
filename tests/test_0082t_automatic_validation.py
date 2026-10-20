"""0082T: validation is produced by the same frozen teachers as training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples import train_probability_fusion_streaming_0082f as streaming


def _fixtures(tmp_path):
    windows_dir = tmp_path / 'windows'
    windows_dir.mkdir()
    windows = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=np.int32)
    win_file = windows_dir / 'fusion_val_windows.npy'
    np.save(win_file, windows)
    manifest = windows_dir / 'windows_metadata.json'
    manifest.write_text(json.dumps({
        'context_length': 4,
        'splits': {'val': {'count': 2, 'windows': str(win_file)}}
    }), encoding='utf-8')
    members = []
    fingerprints = {}
    for mid in (0, 1):
        ckpt = tmp_path / f'ckpt_{mid}'
        ckpt.mkdir()
        (ckpt / 'tokenizer.json').write_text('test')
        members.append((mid, ckpt.resolve()))
        fingerprints[str(mid)] = streaming._checkpoint_fingerprint(ckpt)
    args = argparse.Namespace(windows_metadata=str(manifest), context_length=4,
                              work_dir=str(tmp_path / 'scratch'),
                              val_max_rows=4, lm_head_rows=2,
                              batch_windows=2, context_heads=2)
    return args, members, fingerprints, windows


def test_manifest_val_window_resolution(tmp_path):
    args, _, _, windows = _fixtures(tmp_path)
    path, count = streaming._manifest_validation_windows(args.windows_metadata, 4)
    assert path.name == 'fusion_val_windows.npy' and count == len(windows)
    with pytest.raises(ValueError, match='context length'):
        streaming._manifest_validation_windows(args.windows_metadata, 8)


def test_automatic_teacher_validation_cache_and_context(tmp_path, monkeypatch):
    args, members, fingerprints, windows = _fixtures(tmp_path)
    tokenizer_hash = hashlib.sha256(b'test').hexdigest()
    exported = []

    def fake_export(argv):
        exported.append(list(argv))
        source = Path(argv[argv.index('--windows') + 1])
        out = Path(argv[argv.index('--output-dir') + 1])
        count = int(argv[argv.index('--context-length') + 1])
        last_k = int(argv[argv.index('--score-last-k') + 1])
        stride = int(argv[argv.index('--score-stride') + 1])
        positions = streaming._scored_positions(count, score_last_k=last_k,
                                                score_stride=stride)
        w = np.load(source)
        targets = w[:, positions + 1].reshape(-1)
        np.save(out / 'targets.npy', targets)
        provenance = []
        for mid, ckpt in members:
            logits = np.zeros((len(targets), 17), dtype=np.float16)
            logits[np.arange(len(targets)), targets] = 3.0 + mid
            np.save(out / f'model_{mid}_logits.npy', logits)
            provenance.append({'id': mid, 'checkpoint': str(ckpt),
                               'checkpoint_fingerprint': fingerprints[str(mid)]})
        (out / 'metadata.json').write_text(json.dumps({
            'models': provenance,
            'tokenizer_sha256': tokenizer_hash,
            'token_windows_sha256': streaming._file_sha256(source),
            'prediction_positions': positions.tolist(),
            'vocab_size': 17, 'rows': len(targets)
        }), encoding='utf-8')

    monkeypatch.setattr(streaming, 'export_main', fake_export)
    first, context, identity = streaming._automatic_validation(
        args, members, tokenizer_hash, fingerprints, 17)
    assert len(exported) == 1
    assert first[0].tolist() == [0, 1]
    assert context.rows == len(first[2]) == 4
    x, lengths = context.batch([0, 3])
    assert lengths.tolist() == [2, 4]
    np.testing.assert_array_equal(x[1], windows[1, :4])
    assert identity['validation_mode'] == 'automatic_frozen_teachers'

    second, _, _ = streaming._automatic_validation(args, members, tokenizer_hash, fingerprints, 17)
    assert len(exported) == 1  # reuse matching frozen teachers
    np.testing.assert_array_equal(first[2], second[2])

    # New weights must invalidate the validation cache.
    (members[0][1] / 'weights.npz').write_bytes(b'new snapshot')
    fingerprints['0'] = streaming._checkpoint_fingerprint(members[0][1])
    streaming._automatic_validation(args, members, tokenizer_hash, fingerprints, 17)
    assert len(exported) == 2

    # Changing val windows must also invalidate the validation cache.
    changed = windows.copy()
    changed[0, 0] = 8
    np.save(Path(identity['validation_windows']), changed)
    streaming._automatic_validation(args, members, tokenizer_hash, fingerprints, 17)
    assert len(exported) == 3


def test_cli_does_not_require_external_validation(monkeypatch):
    seen = []
    monkeypatch.setattr(streaming, '_run', lambda args: seen.append(args))
    streaming.main(['--member', '0:/dummy/deep', '--member', '1:/dummy/wide',
                    '--mixed-shard-root', 'dummy', '--windows-metadata', 'dummy',
                    '--output', 'new.npz', '--context-heads', '2'])
    assert seen and seen[0].val_member is None and seen[0].val_targets is None
    with pytest.raises(SystemExit):
        streaming.main(['--member', '0:/dummy/deep', '--mixed-shard-root', 'dummy',
                        '--windows-metadata', 'dummy', '--output', 'new.npz',
                        '--val-targets', 'legacy.npy'])


def test_real_streamed_context_validation_and_resume(tmp_path, monkeypatch):
    """Real tiny parents, disjoint validation shards, no external validation args."""
    import dataclasses
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.collective import ProbabilityResidualCombiner
    from mini_llm.checkpoint import save_checkpoint
    from examples.prepare_probability_fusion_windows_0082b import prepare_windows

    # Accommodate old local 0082Q source when this patch is tested in isolation;
    # on the real 0082R code, gradient_scale=1 is simply handled by the model.
    base_backward = ProbabilityResidualCombiner.loss_and_backward

    def compatible_backward(self, *args, gradient_scale=1.0, **kwargs):
        assert gradient_scale == 1.0
        return base_backward(self, *args, **kwargs)

    monkeypatch.setattr(ProbabilityResidualCombiner, 'loss_and_backward', compatible_backward)
    root = tmp_path / 'mix'
    (root / 'tiny').mkdir(parents=True)
    tokenizer = b'{"tiny":"shared"}'
    digest = hashlib.sha256(tokenizer).hexdigest()
    (root / 'tokenizer.json').write_bytes(tokenizer)
    seq = [1]
    for i in range(15):
        seq.extend([2 + ((i+j) % 15) for j in range(12)])
        seq.append(1)
    (root / 'tiny/train_shard_000.bin').write_bytes(np.asarray(seq, dtype=np.uint16).tobytes())
    for n in range(6):
        (root / f'tiny/val_shard_{n:03d}.bin').write_bytes(
            np.asarray(seq, dtype=np.uint16).tobytes())
    (root / 'mixture_manifest.json').write_text(json.dumps({
        'format_version': 'weighted_packed_mixture_v1', 'complete': True,
        'tokenizer_sha256': digest, 'vocab_size': 17, 'token_dtype': 'uint16',
        'eos_token_id': 1, 'sources': {'tiny': {'weight': 1.0,
        'relative_shard_dir': 'tiny', 'train_shards': 1, 'val_shards': 6}}
    }))
    wins = tmp_path / 'windows'
    prepare_windows(root, wins, 4, (3, 2, 2), 82)
    members = []
    for mid in (0, 1):
        ckpt = tmp_path / f'model_{mid}'
        ckpt.mkdir()
        (ckpt / 'tokenizer.json').write_bytes(tokenizer)
        cfg = ModelConfig(tokenizer_vocab_size=17, context_length=8, n_layers=2,
                          d_model=8, n_q_heads=1, n_kv_heads=1, d_head=8,
                          n_experts=1, top_k=1, d_ff=16, ffn_mode='dense')
        (ckpt / 'config.json').write_text(json.dumps(dataclasses.asdict(cfg)))
        m = DecoderLanguageModel(cfg, rng_seed=432 + mid, dtype='float32')
        save_checkpoint(ckpt, {p.name: p.data for p in m.parameters()})
        members.extend(['--member', f'{mid}:{ckpt}'])
    args = [*members, '--mixed-shard-root', str(root),
            '--windows-metadata', str(wins / 'windows_metadata.json'),
            '--output', str(tmp_path / 'context_fusion.npz'),
            '--work-dir', str(tmp_path / 'cache'), '--context-length', '8',
            '--score-last-k', '8', '--batch-rows', '2',
            '--batch-windows', '2', '--segment-windows', '2', '--lm-head-rows', '3',
            '--context-heads', '2', '--context-head-dim', '4', '--context-hidden', '8',
            '--steps', '2', '--eval-interval', '1', '--log-interval', '1',
            '--early-stop-patience', '0']
    streaming.main(args)
    assert (tmp_path / 'context_fusion.npz').is_file()
    val_dir = tmp_path / 'cache/validation_cache'
    assert (val_dir / 'metadata.json').is_file()
    assert (val_dir / 'targets.npy').is_file()
    regenerated = tmp_path / 'cache/validation_windows/fusion_val_context_8.npy'
    assert regenerated.is_file()
    assert np.load(regenerated).shape == (2, 9)
    cached_sha = streaming._file_sha256(regenerated)
    with np.load(tmp_path / 'context_fusion_last.npz') as state:
        meta = json.loads(str(state['__meta_json__']))
    assert meta['validation_mode'] == 'automatic_frozen_teachers'
    assert meta['step'] == 2
    streaming.main(args[0:-2] + ['--early-stop-patience', '0', '--steps', '3', '--resume'])
    with np.load(tmp_path / 'context_fusion_last.npz') as state:
        meta = json.loads(str(state['__meta_json__']))
    assert meta['step'] == 3
    assert streaming._file_sha256(regenerated) == cached_sha
