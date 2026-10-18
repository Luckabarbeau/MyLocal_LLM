"""0080BR: regroup *already sampled* AR examples by valid target length.

The sampling distribution, EOS boundaries, causal history tensors, target
masks, and per-source exposure are unchanged.  Only which samples share a
microbatch within a single optimizer step changes.
"""

from collections import OrderedDict
from typing import Sequence

import numpy as np


def _scalar_signature(metadata: dict) -> tuple:
    """Keep entire-batch retrieval decisions and sampler modes unchanged."""
    if not isinstance(metadata, dict):
        raise ValueError("length bucketing requires document-aware metadata")
    signature = []
    for key, value in sorted(metadata.items()):
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (list, dict)):
            raise ValueError(f"unsupported non-array metadata field: {key}")
        signature.append((key, type(value).__name__, repr(value)))
    return tuple(signature)


def regroup_ar_accumulation_batches(
    batches: Sequence[tuple], *, batch_size: int, target_length: int,
    bucket_tokens: int,
):
    """Return (repacked batches, CPU-only packing statistics).

    Input tuples: (source_name, source_inputs, target_labels, metadata).
    Source/mode strata are strictly separated, protecting source attribution
    and the sampler's pre-existing retrieval eligibility decision.
    """
    batch_size, target_length, bucket_tokens = (
        int(batch_size), int(target_length), int(bucket_tokens)
    )
    if min(batch_size, target_length, bucket_tokens) <= 0:
        raise ValueError("batch size, target length, and bucket size must be positive")
    if not batches:
        raise ValueError("cannot bucket an empty accumulation")

    strata = OrderedDict()
    original_rows = 0
    raw_cost = 0
    for source, inputs, targets, metadata in batches:
        if metadata is None or "target_valid_lengths" not in metadata:
            raise ValueError("length bucketing requires target_valid_lengths")
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if inputs.ndim != 2 or targets.shape != (batch_size, target_length):
            raise ValueError("unexpected source or target batch shapes")
        if inputs.shape[0] != batch_size:
            raise ValueError("unexpected source batch size")
        lengths = np.asarray(metadata["target_valid_lengths"], dtype=np.int64)
        mask = np.asarray(metadata.get("target_loss_mask"))
        if lengths.shape != (batch_size,) or mask.shape != (batch_size, target_length):
            raise ValueError("invalid sampler target lengths or loss mask")
        if np.any(lengths <= 0) or np.any(lengths > target_length):
            raise ValueError("length bucketing needs positive in-range target lengths")
        if not np.array_equal(mask, np.arange(target_length)[None, :] < lengths[:, None]):
            raise ValueError("length bucketing requires contiguous binary target masks")
        for key, val in metadata.items():
            if isinstance(val, np.ndarray) and val.shape[0] != batch_size:
                raise ValueError(f"unexpected metadata batch dimension: {key}")
        scalar = _scalar_signature(metadata)
        strata.setdefault((str(source), scalar), []).append((inputs, targets, metadata))
        original_rows += batch_size
        raw_cost += batch_size * min(
            target_length, ((int(lengths.max()) + bucket_tokens - 1) // bucket_tokens) * bucket_tokens
        )

    result = []
    packed_cost = 0
    for (source, _), group in strata.items():
        # Stable sort: ordering of equally long examples is unchanged.
        rows = []
        for inputs, targets, metadata in group:
            for idx, length in enumerate(metadata["target_valid_lengths"]):
                rows.append((int(length), inputs, targets, metadata, idx))
        # Largest first: expose peak-VRAM requirements early, and let later
        # smaller batches reuse CuPy's already allocated workspaces.
        rows.sort(key=lambda row: -row[0])
        if len(rows) % batch_size:
            raise AssertionError("every stratum contains complete source microbatches")
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start + batch_size]
            ins = np.stack([r[1][r[4]] for r in subset], axis=0)
            tgt = np.stack([r[2][r[4]] for r in subset], axis=0)
            first_metadata = subset[0][3]
            meta = {}
            for key, value in first_metadata.items():
                if isinstance(value, np.ndarray):
                    meta[key] = np.stack([r[3][key][r[4]] for r in subset], axis=0)
                else:
                    meta[key] = value
            packed_cost += batch_size * min(
                target_length,
                ((int(meta["target_valid_lengths"].max()) + bucket_tokens - 1) // bucket_tokens) * bucket_tokens,
            )
            result.append((source, ins, tgt, meta))

    if len(result) != len(batches):
        raise AssertionError("accumulation bucketing changed the microbatch count")
    return result, {
        "original_rows": raw_cost,
        "bucketed_rows": packed_cost,
        "target_capacity_rows": original_rows * target_length,
        "strata": len(strata),
    }
