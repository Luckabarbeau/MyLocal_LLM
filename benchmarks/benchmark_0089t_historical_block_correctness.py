#!/usr/bin/env python3
"""Audit every production deterministic historical block read against CuPy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import profile_0087a_training as profile


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixed-shard-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    difference = actual - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    actual_norm = float(np.linalg.norm(actual.reshape(-1)))
    denominator = reference_norm * actual_norm
    return {
        "finite": bool(np.isfinite(actual).all()),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1))) / max(
            reference_norm, 1.0e-30
        ),
        "max_absolute": float(np.max(np.abs(difference), initial=0.0)),
        "mean_absolute": float(np.mean(np.abs(difference))),
        "cosine_similarity": (
            float(np.vdot(actual.reshape(-1), reference.reshape(-1))) / denominator
            if denominator > 0.0 else 1.0
        ),
    }


def main() -> int:
    args = _parse_args()
    for name, value in profile._PRODUCTION_BENCH_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_BACKEND", "triton")
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    os.environ.setdefault("MINI_LLM_FUSED_QKV_ROPE_FORWARD", "1")
    os.environ["MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD"] = "1"
    os.environ.setdefault("MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM", "1")

    import cupy as cp
    from mini_llm.backend import BACKEND_NAME, asnumpy
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.historical_read import HistoricalReadMemory
    from mini_llm.ops.historical_schedule import HistoricalSlot, source_indices
    import mini_llm.ops.triton_historical_block as block_kernel
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy":
        raise RuntimeError("0089T requires the CuPy backend")

    checkpoint = args.checkpoint.expanduser().resolve()
    config = ModelConfig(**json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    ))
    apply_runtime_defaults(
        "gpu-fast", backend_name=BACKEND_NAME, config=config,
        training=True, precision="bf16-mixed",
    )
    sources, _ = load_weighted_shard_sources(
        args.mixed_shard_root.expanduser().resolve()
    )
    batch = profile._sample_batch(
        sources=sources,
        config=config,
        batch_size=3,
        rng=np.random.default_rng(args.seed),
        desired_working_length=4096,
        max_attempts=10_000,
    )
    inputs, _, _, metadata, fixture = batch

    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    parameters = model.parameters()
    names = [parameter.name for parameter in parameters]
    loaded, _, _ = load_checkpoint(
        checkpoint, param_names=names, skip_optimizer=True, skip_training=True
    )
    for parameter in parameters:
        parameter.data[...] = loaded.pop(parameter.name)
    model.refresh_compute_buffers()

    original = block_kernel.deterministic_block_read_forward
    rows = []

    def audited(query, key, value, valid, *, phase, period,
                source_offset, source_length):
        candidate = original(
            query, key, value, valid,
            phase=phase,
            period=period,
            source_offset=source_offset,
            source_length=source_length,
        )
        slot = HistoricalSlot(-1, -1, "block", int(phase), int(period))
        selected_host = np.asarray(
            source_indices(
                slot, int(source_length), valid_start=int(source_offset)
            ),
            dtype=np.int64,
        ) - int(source_offset)
        selected = cp.asarray(selected_host)
        reference, _ = HistoricalReadMemory._read(
            model.historical_memory,
            query,
            key[:, selected, :],
            value[:, selected, :],
            valid[:, selected],
        )
        cp.cuda.runtime.deviceSynchronize()
        candidate_host = asnumpy(candidate)
        reference_host = asnumpy(reference)
        candidate_bf16 = candidate.astype(cp.dtype("bfloat16"))
        reference_bf16 = reference.astype(cp.dtype("bfloat16"))
        candidate_bf16_host = asnumpy(candidate_bf16.astype(cp.float32))
        reference_bf16_host = asnumpy(reference_bf16.astype(cp.float32))
        candidate_bits = asnumpy(candidate_bf16.view(cp.uint16))
        reference_bits = asnumpy(reference_bf16.view(cp.uint16))
        per_sample = [
            _metrics(candidate_host[index], reference_host[index])
            for index in range(int(query.shape[0]))
        ]
        mismatches = int(np.count_nonzero(candidate_bits != reference_bits))
        rows.append({
            "ordinal": len(rows),
            "phase": int(phase),
            "period": int(period),
            "source_offset": int(source_offset),
            "source_length": int(source_length),
            "selected_tokens": int(selected_host.size),
            "query_shape": list(query.shape),
            "query_dtype": str(query.dtype),
            "query_strides_elements": [
                int(stride // query.dtype.itemsize) for stride in query.strides
            ],
            "valid_tokens_per_sample": asnumpy(valid.sum(axis=1)).tolist(),
            "fp32": _metrics(candidate_host, reference_host),
            "bf16_quantized": {
                **_metrics(candidate_bf16_host, reference_bf16_host),
                "mismatched_elements": mismatches,
                "total_elements": int(candidate_bits.size),
                "mismatch_fraction": mismatches / int(candidate_bits.size),
            },
            "per_sample_fp32": per_sample,
        })
        return candidate

    block_kernel.deterministic_block_read_forward = audited
    try:
        body, _ = model.forward_body(
            inputs,
            finite_trace=None,
            activation_checkpoint=False,
            memory_metadata=metadata,
        )
        cp.cuda.runtime.deviceSynchronize()
        body_finite = bool(cp.isfinite(body.astype(cp.float32)).all().item())
    finally:
        block_kernel.deterministic_block_read_forward = original

    expected_calls = sum(
        slot.kind == "block"
        for layer in model.historical_memory.slots
        for slot in layer
    )
    fp32_max_relative = max(row["fp32"]["relative_l2"] for row in rows)
    fp32_max_absolute = max(row["fp32"]["max_absolute"] for row in rows)
    bf16_mismatches = sum(
        row["bf16_quantized"]["mismatched_elements"] for row in rows
    )
    bf16_elements = sum(
        row["bf16_quantized"]["total_elements"] for row in rows
    )
    all_invalid_rows = [
        sample
        for row in rows
        for valid_count, sample in zip(
            row["valid_tokens_per_sample"], row["per_sample_fp32"]
        )
        if valid_count == 0
    ]
    accepted = (
        len(rows) == expected_calls
        and body_finite
        and all(row["fp32"]["finite"] for row in rows)
        and fp32_max_relative <= 1.0e-4
        and all(sample["max_absolute"] == 0.0 for sample in all_invalid_rows)
    )
    result = {
        "schema_version": "0089T-historical-block-correctness-v1",
        "status": "passed" if accepted else "failed",
        "fixture": fixture,
        "expected_block_calls": expected_calls,
        "observed_block_calls": len(rows),
        "body_finite": body_finite,
        "summary": {
            "fp32_max_relative_l2": fp32_max_relative,
            "fp32_max_absolute": fp32_max_absolute,
            "bf16_mismatched_elements": bf16_mismatches,
            "bf16_total_elements": bf16_elements,
            "bf16_mismatch_fraction": bf16_mismatches / bf16_elements,
            "all_invalid_sample_comparisons": len(all_invalid_rows),
            "all_invalid_max_absolute": max(
                (sample["max_absolute"] for sample in all_invalid_rows),
                default=0.0,
            ),
        },
        "calls": rows,
        "effective_flags": {
            name: os.environ.get(name) for name in profile._REPORT_ENV_NAMES
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
