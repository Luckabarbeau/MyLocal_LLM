"""0102X: re-pair already sampled EOS-bounded dual-resolution documents.

Only microbatch composition changes. No document is rejected, resampled,
truncated, or concatenated to another document. Source strata are retained so
production per-source CE reporting remains correct without modifying the loss.
"""

from collections import OrderedDict

import numpy as np

from .token_shards import _dual_resolution_geometric_length


def regroup_dual_resolution_document_batches(
    batches, *, batch_size, context_length, eos_token_id,
):
    """Return regrouped `(source, inputs, targets, sampler_metadata)` batches.

    Caller MUST rebuild dual-resolution attention metadata from the new inputs.
    The input batches were sampled in advance by the normal document-first
    sampler. Every original document window appears exactly once in the result.
    Grouping is deterministic (stable sort by descending valid length) and
    does not consume any random numbers. Different sources are never mixed.
    """
    batch_size = int(batch_size)
    context_length = int(context_length)
    if batch_size < 1 or context_length < 1 or not batches:
        raise ValueError("batch_size, context_length and batches must be positive")

    strata = OrderedDict()
    original_rows = 0
    genuine_rows = 0
    for source, inputs, targets, metadata in batches:
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if (inputs.ndim != 2 or targets.shape != inputs.shape
                or inputs.shape[0] != batch_size):
            raise ValueError("unexpected dual-resolution minibatch shape")
        if not isinstance(metadata, dict):
            raise ValueError("missing document-first sampling metadata")
        if metadata.get("document_sampling_policy") != "document_first_full_sequence":
            raise ValueError("requires dual-resolution document-first sampling")
        if int(metadata.get("context_capacity", -1)) != context_length:
            raise ValueError("source context capacity mismatch")
        shape_bucket = metadata.get("shape_bucket", "none")
        if shape_bucket not in ("none", "geometric"):
            raise ValueError("unsupported dual-resolution shape bucket")
        lengths = np.asarray(metadata.get("target_valid_lengths"), dtype=np.int64)
        loss_mask = np.asarray(metadata.get("target_loss_mask"))
        if lengths.shape != (batch_size,) or loss_mask.shape != inputs.shape:
            raise ValueError("invalid target valid lengths or loss mask")
        if np.any(lengths < 1) or np.any(lengths > inputs.shape[1]):
            raise ValueError("invalid document length")
        if np.any((loss_mask != 0) & (loss_mask != 1)):
            raise ValueError("loss mask must be binary")
        for row, length in enumerate(lengths):
            if np.any(loss_mask[row, int(length):] != 0):
                raise ValueError("padding must not be supervised")
        for key in ("document_starts", "document_ends", "target_source_starts"):
            if np.asarray(metadata.get(key)).shape != (batch_size,):
                raise ValueError("missing per-row document provenance: " + key)
        # Stratify by source AND shape policy; never distort per-source metrics.
        key = (str(source), shape_bucket)
        for row, length in enumerate(lengths):
            strata.setdefault(key, []).append(
                (int(length), inputs[row], targets[row], loss_mask[row], metadata, row)
            )
            genuine_rows += int(np.count_nonzero(loss_mask[row, :int(length)]))
        original_rows += int(inputs.size)

    result = []
    bucketed_rows = 0
    for (source, shape_bucket), rows in strata.items():
        if len(rows) % batch_size:
            raise AssertionError("source stratum lacks a complete microbatch")
        # Stable sorting keeps ties in original sampling order.
        rows.sort(key=lambda row: -row[0])
        for offset in range(0, len(rows), batch_size):
            subset = rows[offset:offset + batch_size]
            max_valid = max(row[0] for row in subset)
            width = (max_valid if shape_bucket == "none" else
                     _dual_resolution_geometric_length(max_valid, context_length))
            inp = np.full((batch_size, width), eos_token_id, dtype=subset[0][1].dtype)
            tgt = np.full((batch_size, width), eos_token_id, dtype=subset[0][2].dtype)
            mask = np.zeros((batch_size, width), dtype=np.float32)
            lengths = np.empty(batch_size, dtype=np.int64)
            metadata_out = {
                "document_sampling_policy": "document_first_full_sequence",
                "document_aware": True,
                "target_valid_lengths": lengths,
                "target_loss_mask": mask,
                "executed_sequence_length": width,
                "context_capacity": context_length,
                "shape_bucket": shape_bucket,
            }
            for key in ("document_starts", "document_ends", "target_source_starts"):
                metadata_out[key] = np.empty(batch_size, dtype=np.int64)
            for i, (length, original_x, original_y, original_mask, original_meta, row) in enumerate(subset):
                inp[i, :length] = original_x[:length]
                tgt[i, :length] = original_y[:length]
                mask[i, :length] = original_mask[:length]
                lengths[i] = length
                for key in ("document_starts", "document_ends", "target_source_starts"):
                    metadata_out[key][i] = original_meta[key][row]
            bucketed_rows += int(inp.size)
            result.append((source, inp, tgt, metadata_out))

    if len(result) != len(batches):
        raise AssertionError("repacking changed gradient accumulation length")
    if bucketed_rows > original_rows:
        raise AssertionError("length grouping unexpectedly increased padding")
    return result, {
        "original_rows": original_rows,
        "bucketed_rows": bucketed_rows,
        "genuine_rows": genuine_rows,
        "padding_saved_rows": original_rows - bucketed_rows,
        "strata": len(strata),
        "microbatches": len(result),
    }
