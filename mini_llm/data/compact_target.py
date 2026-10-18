"""0080BN: compact only provably trailing-pad AR working-window positions.

All checks use existing CPU document-first metadata. This never changes
sampling, source positions, or the full external history representation.
"""

import numpy as np


def ar_compact_metadata(metadata, *, target_length, batch_size, bucket_tokens=128):
    """Return a copy with compact AR working length, validating prefix masks.

    The full right-padded [D+T] input buffer remains unchanged. All genuine
    target positions, including the final token in a document, are retained.
    """
    if metadata is None:
        raise ValueError("compact AR requires document-aware target metadata")
    target_length = int(target_length)
    batch_size = int(batch_size)
    bucket_tokens = int(bucket_tokens)
    if target_length < 1 or batch_size < 1 or bucket_tokens < 1:
        raise ValueError("compact AR capacity, batch and bucket must be positive")
    lengths = np.asarray(metadata.get("target_valid_lengths"), dtype=np.int64)
    mask = np.asarray(metadata.get("target_loss_mask"), dtype=np.float32)
    if lengths.shape != (batch_size,) or mask.shape != (batch_size, target_length):
        raise ValueError("compact AR requires aligned target lengths and loss mask")
    if np.any(lengths < 1) or np.any(lengths > target_length):
        raise ValueError("compact AR valid target lengths outside working window")
    expected = np.arange(target_length)[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected.astype(np.float32)):
        raise ValueError("compact AR requires prefix-valid unmodified binary loss masks")
    max_valid = int(lengths.max())
    compact = min(target_length, ((max_valid + bucket_tokens - 1) // bucket_tokens) * bucket_tokens)
    result = dict(metadata)
    if compact < target_length:
        result["compact_working_length"] = compact
        result["compact_ar_target"] = True
    return result
