"""Regression tests for the independent online-fusion source eligibility gate."""

from types import SimpleNamespace
import numpy as np
import pytest

from mini_llm.collective.online_data import (
    split_fusion_shards, make_online_samplers, OnlineShardSampler,
)


def _source(tmp_path, name, weight, shard_count, *, document_length=20):
    paths = []
    directory = tmp_path / name
    directory.mkdir()
    for i in range(shard_count):
        path = directory / f'val_shard_{i:03d}.bin'
        np.asarray([99] + list(range(1, document_length + 1)) + [99],
                   dtype=np.uint16).tofile(path)
        paths.append(path)
    return SimpleNamespace(name=name, weight=weight, val_paths=tuple(paths))


def test_tiny_source_is_skipped_without_dropping_good_data(tmp_path):
    good = _source(tmp_path, 'good', 0.2, 8)
    tiny = _source(tmp_path, 'c_edu', 0.8, 1)
    with pytest.warns(RuntimeWarning, match='skipping source.*c_edu'):
        partitions = split_fusion_shards([tiny, good], seed=82)
    assert list(partitions['train']) == ['good']
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert set(partitions[a]['good']).isdisjoint(partitions[b]['good'])


def test_too_few_sources_fails_clearly(tmp_path):
    tiny = _source(tmp_path, 'c_edu', 0.5, 2)
    with pytest.warns(RuntimeWarning, match='skipping source'):
        with pytest.raises(ValueError, match='no eligible fusion sources'):
            split_fusion_shards([tiny], seed=82)


def test_tiny_window_shards_skipped_and_weights_renormalized(monkeypatch, tmp_path):
    good = _source(tmp_path, 'good', 0.2, 8, document_length=25)
    tiny = _source(tmp_path, 'c_edu', 0.8, 3, document_length=2)
    manifest = {'token_dtype': 'uint16', 'eos_token_id': 99}
    monkeypatch.setattr('mini_llm.collective.online_data.load_weighted_shard_sources',
                        lambda root: ([good, tiny], manifest))
    with pytest.warns(RuntimeWarning, match='skipping source.*c_edu'):
        samplers, _, partitions = make_online_samplers(tmp_path, 8, 82, 82)
    assert samplers['train'].names == ['good']
    assert samplers['val'].names == ['good']
    assert samplers['train'].weights.tolist() == [1.0]
    assert 'c_edu' not in partitions['train']
    for name in ('train', 'val'):
        batch, chosen = samplers[name].batch(8)
        assert batch.shape == (8, 9)
        assert set(chosen) == {'good'}
        assert not np.any(batch[:, :-1] == 99)


def test_exhausted_source_is_no_longer_selected(tmp_path):
    good = _source(tmp_path, 'good', 0.2, 4, document_length=15)
    tiny = _source(tmp_path, 'c_edu', 0.8, 3, document_length=15)
    sampler = OnlineShardSampler(
        [tiny, good], {'c_edu': (), 'good': good.val_paths},
        {'token_dtype': 'uint16', 'eos_token_id': 99}, 5,
        np.random.default_rng(5))
    windows, names = sampler.batch(5)
    assert windows.shape == (5, 6)
    assert set(names) == {'good'}
