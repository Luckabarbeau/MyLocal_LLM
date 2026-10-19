"""0082B: EOS-safe and shard-disjoint fusion windows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.prepare_probability_fusion_windows_0082b import prepare_windows


def _fixture(root):
    root.mkdir()
    (root / 'tokenizer.json').write_text('{"test": true}')
    sha = hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest()
    sources = {}
    for name in ('alpha', 'beta'):
        directory = root / name
        directory.mkdir()
        for i in range(8):
            # Distinct shards, many complete 16-token docs, EOS=2.
            raw = np.concatenate([np.append(np.arange(12, 28, dtype=np.uint16)+i, 2)
                                  for _ in range(30)])
            raw.tofile(directory / f'val_shard_{i:05d}.bin')
        np.array([11, 12, 2], np.uint16).tofile(directory / 'train_shard_00000.bin')
        sources[name] = {'weight': 0.5, 'relative_shard_dir': name,
                         'train_shards': 1, 'val_shards': 8}
    manifest = {'format_version': 'weighted_packed_mixture_v1', 'complete': True,
                'tokenizer_sha256': sha, 'eos_token_id': 2,
                'token_dtype': 'uint16', 'sources': sources}
    (root/'mixture_manifest.json').write_text(json.dumps(manifest))


def test_validation_only_reproducible_eos_bounded_and_disjoint(tmp_path):
    root = tmp_path/'mixture'
    _fixture(root)
    a = prepare_windows(root, tmp_path/'a', 8, (40, 20, 20), 19)
    b = prepare_windows(root, tmp_path/'b', 8, (40, 20, 20), 19)
    sets = {}
    for split in ('train', 'val', 'test'):
        x = np.load(tmp_path/'a'/f'fusion_{split}_windows.npy')
        y = np.load(tmp_path/'b'/f'fusion_{split}_windows.npy')
        np.testing.assert_array_equal(x, y)
        assert x.shape[1] == 9 and not np.any(x[:, :-1] == 2)
        assert np.all(x < 32768)
        sets[split] = {p for paths in a['splits'][split]['validation_shards'].values()
                       for p in paths}
        assert all('val_shard_' in p for p in sets[split])
    assert not sets['train'] & sets['val']
    assert not sets['train'] & sets['test']
    assert not sets['val'] & sets['test']


def test_validation_shard_count_precondition(tmp_path):
    root = tmp_path/'mixture'
    _fixture(root)
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from examples.prepare_probability_fusion_windows_0082b import _partition_by_shard
    src, _ = load_weighted_shard_sources(root)
    from dataclasses import replace
    with pytest.raises(ValueError, match='>=3 val shards'):
        _partition_by_shard([replace(src[0], val_paths=src[0].val_paths[:2])], 17)


def test_heldout_eval_helper_matches_arithmetic_at_initialization(tmp_path, capsys):
    from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
    from examples.evaluate_probability_fusion_0082b import main
    rng = np.random.default_rng(21)
    a, b = (rng.normal(size=(5, 7)).astype(np.float16) for _ in range(2))
    np.save(tmp_path/'a.npy', a)
    np.save(tmp_path/'b.npy', b)
    np.save(tmp_path/'targets.npy', np.arange(5, dtype=np.int32))
    checkpoint = tmp_path/'fusion.npz'
    ProbabilityResidualCombiner(FusionConfig(vocab_size=7, max_models=10)).save(checkpoint)
    main(['--checkpoint', str(checkpoint), '--member', f'0:{tmp_path / "a.npy"}',
          '--member', f'1:{tmp_path / "b.npy"}',
          '--targets', str(tmp_path/'targets.npy'), '--batch-rows', '2'])
    output = capsys.readouterr().out
    assert 'model_0_ce=' in output and 'model_1_ce=' in output
    line = [x for x in output.splitlines() if x.startswith('fusion_minus_arithmetic=')][0]
    assert abs(float(line.split('=')[1])) < 1e-5
