"""0078A: observational per-source training accounting.

Each observed loss is the unmodified *pre-update*, masked cross-entropy of one
microbatch.  The source mean is token-weighted rather than a mean of batch
means.  This module never changes optimizer weights or batch selection.

Document lengths refer to *sampled packed document fragments*, including the
EOS marker when present.  The sampling process is length-biased: these means
must NOT be reported as an unbiased corpus-wide mean document length.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

VERSION = 1
FIELDS = (
    "microbatches", "samples", "source_capacity_tokens", "valid_source_tokens",
    "history_capacity_tokens", "valid_history_tokens",
    "target_capacity_tokens", "valid_active_tokens", "supervised_tokens",
    "supervised_token_weight", "loss_nll_sum", "sampled_document_spans",
    "sampled_document_span_tokens_sum", "sampled_document_span_tokens_min",
    "sampled_document_span_tokens_max",
)


def empty_record():
    return {
        "microbatches": 0, "samples": 0,
        "source_capacity_tokens": 0, "valid_source_tokens": 0,
        "history_capacity_tokens": 0, "valid_history_tokens": 0,
        "target_capacity_tokens": 0, "valid_active_tokens": 0,
        "supervised_tokens": 0, "supervised_token_weight": 0.0,
        "loss_nll_sum": 0.0,
        "sampled_document_spans": 0,
        "sampled_document_span_tokens_sum": 0,
        "sampled_document_span_tokens_min": None,
        "sampled_document_span_tokens_max": None,
    }


def batch_coverage(*, batch_size, source_length, target_shape, loss_mask,
                   memory_length=0, target_length=None, metadata=None):
    """Pure CPU accounting, using exactly the mask consumed by the LM loss."""
    metadata = metadata or {}
    size = int(np.prod(target_shape))
    if loss_mask is None:
        supervised_tokens = size
        supervised_weight = float(size)
    else:
        if not isinstance(loss_mask, np.ndarray):
            raise TypeError("source coverage requires a CPU loss mask")
        supervised_tokens = int(np.count_nonzero(loss_mask))
        supervised_weight = float(np.sum(loss_mask, dtype=np.float64))
    if not np.isfinite(supervised_weight) or supervised_weight <= 0:
        raise ValueError("invalid supervised token weight")
    history_capacity = int(batch_size * memory_length)
    if memory_length:
        starts = np.asarray(metadata.get("history_valid_starts", [0] * batch_size))
        lengths = np.asarray(metadata.get("target_valid_lengths", [target_length] * batch_size))
        if starts.shape != (batch_size,) or lengths.shape != (batch_size,):
            raise ValueError("source-metric metadata must have batch_size entries")
        valid_history = int(np.clip(memory_length - starts, 0, memory_length).sum())
        valid_active = int(np.clip(lengths, 0, target_length).sum())
        valid_source = valid_history + valid_active
    else:
        valid_history = 0
        valid_active = int(batch_size * source_length)
        valid_source = valid_active

    record = empty_record()
    record.update({
        "microbatches": 1, "samples": int(batch_size),
        "source_capacity_tokens": int(batch_size * source_length),
        "valid_source_tokens": valid_source,
        "history_capacity_tokens": history_capacity,
        "valid_history_tokens": valid_history,
        "target_capacity_tokens": size,
        "valid_active_tokens": valid_active,
        "supervised_tokens": supervised_tokens,
        "supervised_token_weight": supervised_weight,
    })
    if "document_starts" in metadata and "document_ends" in metadata:
        lengths = (np.asarray(metadata["document_ends"], dtype=np.int64)
                   - np.asarray(metadata["document_starts"], dtype=np.int64))
        if lengths.shape != (batch_size,) or np.any(lengths <= 0):
            raise ValueError("invalid sampled document spans")
        record["sampled_document_spans"] = int(lengths.size)
        record["sampled_document_span_tokens_sum"] = int(lengths.sum())
        record["sampled_document_span_tokens_min"] = int(lengths.min())
        record["sampled_document_span_tokens_max"] = int(lengths.max())
    return record


def merge_record(dst, src):
    for key in FIELDS:
        if key in ("sampled_document_span_tokens_min", "sampled_document_span_tokens_max"):
            value = src.get(key)
            if value is not None:
                if dst[key] is None:
                    dst[key] = int(value)
                elif key.endswith("_min"):
                    dst[key] = min(dst[key], int(value))
                else:
                    dst[key] = max(dst[key], int(value))
        else:
            dst[key] += src.get(key, 0)


def row(source, record, step, scope, since_step):
    weight = record["supervised_token_weight"]
    docs = record["sampled_document_spans"]
    target_capacity = record["target_capacity_tokens"]
    return {
        "step": int(step), "since_step": since_step, "scope": scope,
        "source": source, **record,
        "mean_loss_nats_per_target": (record["loss_nll_sum"] / weight if weight else ""),
        "perplexity": (float(np.exp(record["loss_nll_sum"] / weight)) if weight else ""),
        "target_fill": (weight / target_capacity if target_capacity else ""),
        "mean_sampled_document_span_tokens": (
            record["sampled_document_span_tokens_sum"] / docs if docs else ""
        ),
    }


CSV_COLUMNS = ("step", "since_step", "scope", "source", *FIELDS,
               "mean_loss_nats_per_target", "perplexity", "target_fill",
               "mean_sampled_document_span_tokens")


def write_source_rows(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        for item in rows:
            writer.writerow(item)
