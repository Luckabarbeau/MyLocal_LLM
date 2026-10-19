"""Aligned causal-token inputs for a probability-fusion logit export.

The context window file and exporter metadata are a *pair*: row r must use
window r//scored, prediction position positions[r%scored]. Reject missing
or stale alignment, including a file with the same shape but different tokens.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_fingerprint(directory):
    """Detect any change to frozen checkpoint files (names, sizes, mtimes).

    This is a cheap mutation guard, not a cryptographic content checksum.
    """
    directory = Path(directory).resolve()
    files = []
    for file in sorted(directory.rglob('*')):
        if file.is_file():
            stat = file.stat()
            files.append((str(file.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    if not files:
        raise ValueError(f'checkpoint directory is empty: {directory}')
    return hashlib.sha256(json.dumps(files).encode('utf-8')).hexdigest()


class AlignedContextWindows:
    def __init__(self, windows_path, metadata_path, targets, *, tokenizer_sha256=None):
        self.windows_path = Path(windows_path)
        self.metadata_path = Path(metadata_path)
        self.windows = np.load(self.windows_path, mmap_mode='r', allow_pickle=False)
        meta = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        self.positions = np.asarray(meta['prediction_positions'], dtype=np.int64)
        if self.windows.ndim != 2 or not np.issubdtype(self.windows.dtype, np.integer):
            raise ValueError('context windows must be integer [windows,time+1]')
        if self.positions.ndim != 1 or not len(self.positions) or np.any(self.positions < 0):
            raise ValueError('invalid exported prediction positions')
        if np.any(self.positions >= self.windows.shape[1] - 1) or not np.all(np.diff(self.positions) > 0):
            raise ValueError('prediction positions outside input context or not ascending')
        if tokenizer_sha256 is not None and meta.get('tokenizer_sha256') != tokenizer_sha256:
            raise ValueError('context/tokenizer mismatch with teacher predictions')
        digest = hashlib.sha256()
        with self.windows_path.open('rb') as f:
            for part in iter(lambda: f.read(1 << 20), b''):
                digest.update(part)
        if meta.get('token_windows_sha256') != digest.hexdigest():
            raise ValueError('context window SHA256 does not match teacher predictions')
        total = self.windows.shape[0] * len(self.positions)
        if int(meta['rows']) != total or len(targets) != total:
            raise ValueError('context windows do not match teacher logit row count')
        expected = np.asarray(self.windows[:, self.positions + 1]).reshape(-1)
        if not np.array_equal(expected, np.asarray(targets)):
            raise ValueError('teacher prediction targets are misaligned with context tokens')
        self.rows = total
        self.scored = len(self.positions)

    def batch(self, row_indices):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1 or np.any(row_indices < 0) or np.any(row_indices >= self.rows):
            raise ValueError('context prediction row index out of bounds')
        lengths = self.positions[row_indices % self.scored] + 1
        width = int(lengths.max())
        context = np.asarray(self.windows[row_indices // self.scored, :width], dtype=np.int64)
        return context, lengths
