#!/usr/bin/env python3
"""HIP full Model B FFN wgrad parity, with one model and CPU gradient snapshots.

Runs full supervision sequentially with transpose-wgrad disabled/enabled. No
checkpoint files, optimizer, source rewriting, or second GPU model are used.
Block activation recomputation follows the production-width 0102D benchmark.
GPU execution is explicit via --gpu; --help imports no GPU/backend modules.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OPTION = "MINI_LLM_HIP_FFN_TRANSPOSE_WGRAD"
TOLERANCE = 0.001


def report(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", required=True,
                        help="explicitly execute both full-model HIP passes")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--head-chunk-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if not 8192 <= args.context <= 65536 or args.context % 64:
        parser.error("context must be a multiple of 64 in [8192, 65536]")
    if args.head_chunk_tokens < 1:
        parser.error("head-chunk-tokens must be positive")
    return args


def run(args):
    # Keep all backend/model imports after argument parsing (including --help).
    os.environ["MINI_LLM_BACKEND"] = "cupy"
    from benchmarks.benchmark_dual_resolution_full_model_0102d import (
        build_config, parse_args as benchmark_args,
    )
    from mini_llm.backend import BACKEND_NAME, xp, synchronize
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy" or not xp.cuda.runtime.is_hip:
        raise RuntimeError("This check requires the production CuPy HIP backend")
    setup = benchmark_args([
        "--architecture", "matrix-pool", "--context", str(args.context),
        "--score-last-k", str(args.context), "--seed", str(args.seed),
        "--head-chunk-tokens", str(args.head_chunk_tokens),
        "--runtime-profile", "consumer-gpu",
    ])
    config = build_config(setup, backend_name=BACKEND_NAME)
    runtime_profile = apply_runtime_defaults(
        "consumer-gpu", backend_name=BACKEND_NAME, config=config,
        training=True, precision="bf16-mixed",
    )
    if args.context > config.dual_resolution.block_size * config.dual_resolution.max_blocks:
        raise ValueError("context exceeds Model B block capacity")
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata

    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    parameters = model.parameters()
    count = sum(int(p.size) for p in parameters)
    if count != config.estimated_parameter_count():
        raise AssertionError("parameter accounting mismatch")
    down_ids = {id(p) for p in parameters
                if p.name.endswith(".W_down") and p.data.shape == (4096, 1024)}
    if len(down_ids) != config.n_layers:
        raise AssertionError("expected one full-width FFN down weight per Model B layer")
    if os.environ.get("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1").strip().lower() in ("1", "true", "yes"):
        model.refresh_compute_buffers()
    rng = np.random.default_rng(args.seed + 1)
    eos_id = int(config.vocab_size - 1)
    raw = rng.integers(0, eos_id, size=(1, args.context + 1), dtype=np.int64)
    inputs = xp.asarray(raw[:, :-1], dtype=xp.int64)
    targets = xp.asarray(raw[:, 1:], dtype=xp.int64)
    metadata = build_dual_resolution_metadata(
        raw[:, :-1], eos_token_id=eos_id,
        block_size=config.dual_resolution.block_size,
    )
    report({"check": "setup", "context": args.context,
            "supervised_tokens": int(targets.size), "parameters": count,
            "parameter_tensors": len(parameters), "seed": args.seed,
            "runtime_profile": vars(runtime_profile), "relative_l2_tolerance": TOLERANCE})
    baseline = []
    losses = []
    failures = []
    previous = os.environ.get(OPTION)
    try:
        for enabled in (0, 1):
            os.environ[OPTION] = str(enabled)
            model.zero_grad()
            head, cache = model.forward_body(
                inputs, activation_checkpoint=True, dual_resolution_metadata=metadata,
            )
            loss, loss_cache = model.chunked_lm_head_loss_forward(
                head, targets, chunk_tokens=args.head_chunk_tokens,
                target_slice=(0, args.context), return_device_loss=True,
            )
            loss_value = float(loss.item())
            del head, loss
            if not np.isfinite(loss_value) or loss_value <= 0:
                raise AssertionError(f"flag={enabled}: invalid loss {loss_value}")
            losses.append(loss_value)
            dx = model.chunked_lm_head_backward(loss_cache)
            model.backward_body(dx, cache)
            synchronize()
            del dx, cache, loss_cache
            report({"check": "pass", "flag": enabled, "loss": loss_value})
            for index, parameter in enumerate(parameters):
                if parameter.grad is None:
                    raise AssertionError(f"missing gradient: {parameter.name}")
                # Only the current parameter crosses PCIe; no GPU gradient clone.
                host = xp.asnumpy(parameter.grad)
                if enabled == 0:
                    if not np.isfinite(host).all():
                        raise AssertionError(f"nonfinite baseline gradient: {parameter.name}")
                    baseline.append(host)
                    continue
                reference = baseline[index]
                actual64 = host.astype(np.float64)
                reference64 = reference.astype(np.float64)
                finite = bool(np.isfinite(actual64).all())
                reference_norm = float(np.linalg.norm(reference64.ravel()))
                actual_norm = float(np.linalg.norm(actual64.ravel()))
                delta = actual64 - reference64
                absolute = float(np.linalg.norm(delta.ravel()))
                relative = absolute / max(reference_norm, 1e-300)
                exact = bool(np.array_equal(host, reference))
                down = id(parameter) in down_ids
                # Unused embedding rows can be zero, but each trained tensor
                # must have a finite, nonzero total gradient under full supervision.
                passed = (finite and reference_norm > 0 and actual_norm > 0
                          and np.isfinite(relative) and relative <= TOLERANCE
                          and (down or exact))
                report({"check": "gradient", "index": index, "name": parameter.name,
                        "shape": list(host.shape), "down_weight": down,
                        "reference_l2": reference_norm if np.isfinite(reference_norm) else None,
                        "actual_l2": actual_norm if np.isfinite(actual_norm) else None,
                        "absolute_l2": absolute if np.isfinite(absolute) else None,
                        "relative_l2": relative if np.isfinite(relative) else None,
                        "finite": finite, "exact": exact, "passed": bool(passed)})
                if not passed:
                    failures.append(parameter.name)
                del host, actual64, reference64, delta
    finally:
        if previous is None:
            os.environ.pop(OPTION, None)
        else:
            os.environ[OPTION] = previous
    identical_loss = losses[0] == losses[1]
    report({"check": "summary", "losses": losses, "loss_identical": identical_loss,
            "compared_parameters": len(baseline), "failed_parameters": failures,
            "passed": identical_loss and not failures})
    if failures or not identical_loss:
        raise AssertionError("full Model B wgrad parity failed; see per-parameter reports")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
