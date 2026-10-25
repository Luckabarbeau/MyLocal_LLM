#!/usr/bin/env python3
"""Strict full-model parity gate for the 0089R historical block forward."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import profile_0087a_training as profile


_THRESHOLDS = {
    "forward_relative_l2": 1.0e-4,
    "global_gradient_relative_l2": 5.0e-4,
    "per_parameter_cosine_similarity": 0.99999,
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixed-shard-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--working-length", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    difference = actual - reference
    actual_norm = float(np.linalg.norm(actual.reshape(-1)))
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    difference_norm = float(np.linalg.norm(difference.reshape(-1)))
    denominator = actual_norm * reference_norm
    return {
        "finite": bool(np.isfinite(actual).all()),
        "bitwise_equal": bool(np.array_equal(actual, reference)),
        "relative_l2": difference_norm / max(reference_norm, 1.0e-30),
        "max_absolute": float(np.max(np.abs(difference), initial=0.0)),
        "cosine_similarity": (
            float(np.vdot(actual.reshape(-1), reference.reshape(-1))) / denominator
            if denominator > 0.0 else 1.0
        ),
        "difference_l2_sq": difference_norm * difference_norm,
        "reference_l2_sq": reference_norm * reference_norm,
    }


def main() -> int:
    args = _parse_args()
    for name, value in profile._PRODUCTION_BENCH_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_BACKEND", "triton")
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    os.environ.setdefault("MINI_LLM_FUSED_QKV_ROPE_FORWARD", "1")
    os.environ["MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD"] = "0"

    import cupy as cp
    from mini_llm.backend import BACKEND_NAME, asnumpy
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy":
        raise RuntimeError("0089R requires the CuPy backend")

    checkpoint = args.checkpoint.expanduser().resolve()
    mixed_root = args.mixed_shard_root.expanduser().resolve()
    config = ModelConfig(**json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    ))
    apply_runtime_defaults(
        "gpu-fast", backend_name=BACKEND_NAME, config=config,
        training=True, precision="bf16-mixed",
    )
    sources, _ = load_weighted_shard_sources(mixed_root)
    batch = profile._sample_batch(
        sources=sources,
        config=config,
        batch_size=3,
        rng=np.random.default_rng(args.seed),
        desired_working_length=args.working_length,
        max_attempts=10_000,
    )
    inputs, targets, loss_mask, metadata, fixture = batch

    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    parameters = model.parameters()
    names = [parameter.name for parameter in parameters]
    loaded, _, _ = load_checkpoint(
        checkpoint, param_names=names, skip_optimizer=True, skip_training=True
    )
    missing = sorted(set(names) - set(loaded))
    if missing:
        raise RuntimeError(f"checkpoint missing parameters: {missing[:5]}")
    for parameter in parameters:
        parameter.data[...] = loaded.pop(parameter.name)
    del loaded
    model.refresh_compute_buffers()
    chunk_tokens = int(os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"])

    def run(*, candidate):
        model.historical_memory.triton_block_forward = candidate
        os.environ["MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD"] = (
            "1" if candidate else "0"
        )
        model.zero_grad()
        body_output, cache = model.forward_body(
            inputs,
            finite_trace=None,
            activation_checkpoint=False,
            memory_metadata=metadata,
        )
        body_host = asnumpy(body_output.astype(cp.float32))
        target_slice = cache.get("target_slice")
        loss, loss_cache = model.chunked_lm_head_loss_forward(
            body_output,
            targets,
            chunk_tokens=chunk_tokens,
            loss_mask=loss_mask,
            return_device_loss=True,
            finite_trace=None,
            target_slice=target_slice,
        )
        dx_head = model.chunked_lm_head_backward(loss_cache, grad_scale=1.0)
        model.backward_body(dx_head, cache)
        cp.cuda.runtime.deviceSynchronize()
        loss_value = float(loss.item())
        del body_output, cache, loss_cache, dx_head, loss
        return body_host, loss_value

    baseline_body, baseline_loss = run(candidate=False)
    baseline_gradients = {
        parameter.name: asnumpy(parameter.grad).copy()
        for parameter in parameters
    }
    gc.collect()

    candidate_body, candidate_loss = run(candidate=True)
    body_metrics = _metrics(candidate_body, baseline_body)
    del candidate_body, baseline_body

    gradient_rows = []
    global_difference_sq = 0.0
    global_reference_sq = 0.0
    bitwise_equal_count = 0
    for parameter in parameters:
        reference = baseline_gradients.pop(parameter.name)
        actual = asnumpy(parameter.grad)
        row = {
            "name": parameter.name,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            **_metrics(actual, reference),
        }
        gradient_rows.append(row)
        global_difference_sq += row["difference_l2_sq"]
        global_reference_sq += row["reference_l2_sq"]
        bitwise_equal_count += int(row["bitwise_equal"])
        del actual, reference

    global_relative_l2 = (
        global_difference_sq / max(global_reference_sq, 1.0e-60)
    ) ** 0.5
    finite = body_metrics["finite"] and all(row["finite"] for row in gradient_rows)
    worst_cosine = min(row["cosine_similarity"] for row in gradient_rows)
    worst_relative = max(gradient_rows, key=lambda row: row["relative_l2"])
    accepted = (
        finite
        and body_metrics["relative_l2"] <= _THRESHOLDS["forward_relative_l2"]
        and global_relative_l2 <= _THRESHOLDS["global_gradient_relative_l2"]
        and worst_cosine >= _THRESHOLDS["per_parameter_cosine_similarity"]
    )

    result = {
        "schema_version": "0089R-parity-v1",
        "status": "passed" if accepted else "failed",
        "flag": "MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD",
        "fixture": fixture,
        "thresholds": _THRESHOLDS,
        "forward_output": body_metrics,
        "loss": {
            "baseline": baseline_loss,
            "candidate": candidate_loss,
            "absolute_difference": abs(candidate_loss - baseline_loss),
            "relative_difference": abs(candidate_loss - baseline_loss) / max(
                abs(baseline_loss), 1.0e-30
            ),
        },
        "gradients": {
            "parameter_count": len(gradient_rows),
            "bitwise_equal_count": bitwise_equal_count,
            "global_relative_l2": global_relative_l2,
            "worst_cosine_similarity": worst_cosine,
            "worst_relative_l2": worst_relative,
            "top20_relative_l2": sorted(
                gradient_rows, key=lambda row: row["relative_l2"], reverse=True
            )[:20],
        },
        "effective_flags": {
            name: os.environ.get(name) for name in profile._REPORT_ENV_NAMES
        },
        "execution_order": ["baseline", "candidate"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
