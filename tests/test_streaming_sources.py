"""0077A: deterministic bounded remote token streaming; no network needed."""
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.data.streaming_sources import (
    StreamSpec, StreamingTokenCache, load_registry,
)


class TinyTokenizer:
    eos_token = '<eos>'
    token_to_id = {'<eos>': 1}

    def encode(self, text):
        return [3 + (ord(ch) % 43) for ch in text]


class ResumableRows:
    def __init__(self):
        self.rows = [{'text': f'document_{i}_' + 'x'*75} for i in range(128)]
        self.pos = 0

    def load_state_dict(self, value):
        self.pos = value['pos']

    def state_dict(self):
        return {'pos': self.pos}

    def __iter__(self):
        while self.pos < len(self.rows):
            row = self.rows[self.pos]
            self.pos += 1
            yield row


def spec():
    return StreamSpec(name='fake', dataset='fake/name', revision='abc123',
                      weight=0.15, license='test-only')


def cache(tmp_path, *, reuse=3):
    return StreamingTokenCache([spec()], TinyTokenizer(), segment_tokens=1024,
                               reuse_batches=reuse, cache_dir=tmp_path,
                               factory=lambda _: ResumableRows())


def test_registry_requires_pinned_dataset_and_weights(tmp_path):
    conf = tmp_path/'registry.json'
    conf.write_text(json.dumps({'version': 1, 'sources': [spec().__dict__]}))
    assert load_registry(conf) == [spec()]
    conf.write_text(json.dumps({'version': 1, 'sources': [
        {**spec().__dict__, 'revision': 'main'}]}))
    with pytest.raises(ValueError, match='pin'):
        load_registry(conf)


def test_rotation_document_boundaries_and_bounded_files(tmp_path):
    with_cache = cache(tmp_path)
    try:
        with_cache.prefetch()
        first, gen, old = with_cache.next_shard('fake')
        assert gen == 0 and old == []
        ids = np.fromfile(first, dtype=np.uint16)
        assert (ids == 1).sum() >= 1
        for _ in range(2):
            assert with_cache.next_shard('fake')[0] == first
        second, gen, retired = with_cache.next_shard('fake')
        assert gen == 1 and retired == [first]
        assert with_cache.sources['fake'].history[0]['microbatches'] == 3
        assert 'start_cursor' not in with_cache.sources['fake'].history[0]
        assert len(list(Path(with_cache._temp.name).glob('*.bin'))) <= 3
        assert not np.array_equal(ids, np.fromfile(second, dtype=np.uint16))
    finally:
        with_cache.close()


def test_checkpoint_replays_active_segment_with_same_digest(tmp_path):
    original = cache(tmp_path)
    try:
        original.prefetch()
        active, _, _ = original.next_shard('fake')
        original.next_shard('fake')
        expected = np.fromfile(active, dtype=np.uint16)
        state = json.loads(json.dumps(original.state_dict()))
        restored = cache(tmp_path)
        try:
            restored.load_state_dict(state)
            restored.prefetch()
            path, generation, _ = restored.next_shard('fake')
            assert generation == 0
            np.testing.assert_array_equal(expected, np.fromfile(path, dtype=np.uint16))
            assert restored.sources['fake'].active_uses == 3
            # After the final resumed batch, rotation uses the correct cursor.
            next_path, generation, _ = restored.next_shard('fake')
            assert generation == 1
            assert next_path.exists()
        finally:
            restored.close()
    finally:
        original.close()


def test_checkpoint_at_segment_boundary(tmp_path):
    original = cache(tmp_path, reuse=2)
    try:
        original.next_shard('fake')
        original.next_shard('fake')
        saved = original.state_dict()
        restored = cache(tmp_path, reuse=2)
        try:
            restored.load_state_dict(saved)
            path, generation, _ = restored.next_shard('fake')
            assert generation == 1
            assert path.exists()
            assert len(restored.sources['fake'].history) == 1
        finally:
            restored.close()
    finally:
        original.close()


def test_registry_fingerprint_prevents_silent_source_changes(tmp_path):
    first = cache(tmp_path)
    try:
        saved = first.state_dict()
        other = StreamingTokenCache([
            StreamSpec(name='fake', dataset='different', revision='abc123',
                       weight=0.15, license='test-only')
        ], TinyTokenizer(), cache_dir=tmp_path, factory=lambda _: ResumableRows())
        try:
            with pytest.raises(ValueError, match='registry differs'):
                other.load_state_dict(saved)
        finally:
            other.close()
    finally:
        first.close()


def test_persisted_active_segment_survives_nondeterministic_remote_replay(tmp_path):
    """Legacy cursor replay may differ; new checkpoints never need it."""
    old = cache(tmp_path / 'cache_old', reuse=4)
    try:
        file, gen, retired = old.next_shard('fake')
        old.next_shard('fake')
        original = file.read_bytes()
        checkpoint = tmp_path / 'saved_checkpoint'
        state = old.checkpoint_state(checkpoint)
        entry = state['sources']['fake']['active']
        assert (checkpoint / entry['snapshot']).read_bytes() == original
    finally:
        old.close()

    class ChangingData(ResumableRows):
        def __init__(self):
            super().__init__()
            for row in self.rows:
                row['text'] += 'MUTATED'

    resumed = StreamingTokenCache(
        [spec()], TinyTokenizer(), segment_tokens=1024,
        reuse_batches=4, cache_dir=tmp_path / 'cache_resumed',
        factory=lambda _: ChangingData(),
    )
    try:
        resumed.load_state_dict(state, checkpoint_dir=checkpoint)
        resumed.prefetch()
        path, newgen, retired = resumed.next_shard('fake')
        assert newgen == gen
        assert path.read_bytes() == original
        assert resumed.sources['fake'].active_uses == 3
    finally:
        resumed.close()


def test_corrupt_or_missing_saved_segment_refuses_replay(tmp_path):
    old = cache(tmp_path / 'cache_old', reuse=4)
    try:
        old.next_shard('fake')
        checkpoint = tmp_path / 'checkpoint'
        state = old.checkpoint_state(checkpoint)
    finally:
        old.close()
    entry = state['sources']['fake']['active']
    snapshot = checkpoint / entry['snapshot']
    snapshot.write_bytes(snapshot.read_bytes()[:-2] + b'xx')
    restored = cache(tmp_path / 'cache_restored', reuse=4)
    try:
        restored.load_state_dict(state, checkpoint_dir=checkpoint)
        with pytest.raises(ValueError, match='corrupted'):
            restored.next_shard('fake')
    finally:
        restored.close()


def test_checkpoint_snapshot_prunes_old_only_after_second_checkpoint(tmp_path):
    c = cache(tmp_path / 'temp', reuse=2)
    folder = tmp_path / 'checkpoint'
    try:
        c.next_shard('fake')
        first = c.checkpoint_state(folder)
        original = folder / first['sources']['fake']['active']['snapshot']
        assert original.is_file()
        c.next_shard('fake')
        c.next_shard('fake')
        second = c.checkpoint_state(folder)
        replacement = folder / second['sources']['fake']['active']['snapshot']
        assert replacement.is_file()
        c.prune_checkpoint_snapshots(folder, second)
        assert replacement.is_file()
        assert not original.exists()
    finally:
        c.close()
