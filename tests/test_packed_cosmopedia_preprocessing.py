from pathlib import Path

import numpy as np

from generate_packed_cosmopedia_shards import (
    PackedShardWriter,
    is_validation_document,
)


def test_packed_writer_streams_and_rotates_without_padding(tmp_path):
    # Four uint16 tokens per shard.
    shard_size_mb = (4 * np.dtype(np.uint16).itemsize) / (1024 * 1024)
    writer = PackedShardWriter(tmp_path, "train", shard_size_mb, np.uint16)
    writer.write([1, 2, 3])
    writer.write([4, 5, 6, 7, 8, 9, 10])
    writer.close()

    paths = sorted(tmp_path.glob("train_shard_*.bin"))
    assert len(paths) == 3
    arrays = [np.fromfile(path, dtype=np.uint16) for path in paths]
    np.testing.assert_array_equal(arrays[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(arrays[1], [5, 6, 7, 8])
    np.testing.assert_array_equal(arrays[2], [9, 10])
    assert writer.total_tokens == 10


def test_validation_split_is_deterministic_and_seeded():
    texts = [f"document-{i}" for i in range(200)]
    first = [is_validation_document(text, 0.1, 42) for text in texts]
    second = [is_validation_document(text, 0.1, 42) for text in texts]
    changed = [is_validation_document(text, 0.1, 43) for text in texts]

    assert first == second
    assert first != changed
    assert 5 <= sum(first) <= 35
