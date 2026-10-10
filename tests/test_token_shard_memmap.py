import os
import tempfile
from pathlib import Path

import numpy as np

from mini_llm.data.token_shards import map_token_shard, create_minibatch


def test_map_packed_token_shard_without_copy():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "packed.bin"
        tokens = np.arange(1024, dtype=np.uint16)
        tokens.tofile(path)

        mapped = map_token_shard(str(path))
        assert isinstance(mapped, np.memmap)
        assert mapped.ndim == 1
        np.testing.assert_array_equal(mapped[:20], tokens[:20])

        x, y = create_minibatch(
            mapped, batch_size=2, seq_length=16,
            rng=np.random.default_rng(5),
        )
        assert x.shape == (2, 16)
        assert y.shape == (2, 16)


def test_map_legacy_rectangular_shard():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.bin"
        data = np.arange(4 * 33, dtype=np.uint16).reshape(4, 33)
        with open(path, "wb") as f:
            f.write((4).to_bytes(8, "little"))
            f.write((33).to_bytes(8, "little"))
            data.tofile(f)

        mapped = map_token_shard(str(path))
        assert isinstance(mapped, np.memmap)
        assert mapped.shape == (4, 33)
        np.testing.assert_array_equal(mapped, data)
