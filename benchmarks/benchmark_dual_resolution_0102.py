#!/usr/bin/env python3
"""0102 dual-resolution correctness and scaling benchmark.

This benchmark exercises the repository's explicit NumPy/CuPy implementation.
It is intentionally correctness-first and unfused; it is not evidence of a
production-ready GPU path.  Attention scores are formed only for block-local,
summary, and per-block compressor rectangles--never as a dense token T x T
matrix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Support direct `python benchmarks/benchmark_dual_resolution_0102.py` from the
# repository root, as documented, without requiring PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
PARAMETER_DIMS = (256, 512, 1024)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=_csv_ints, default=DEFAULT_CONTEXTS,
        help="comma-separated contexts (default: 1024,2048,...,65536)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compression-dim", type=int, default=512)
    parser.add_argument("--compression-heads", type=int, default=8)
    parser.add_argument("--compression-depth", type=int, default=1)
    parser.add_argument("--compression-block-chunk", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--global-query-chunk", type=int, default=64)
    parser.add_argument(
        "--local-kernel", choices=("cupy", "triton"), default="cupy",
        help="0102B experimental Triton local attention only; global and compression remain CuPy",
    )
    parser.add_argument("--global-kernel", choices=("cupy", "triton"),
                        default="cupy", help="0102C opt-in global-summary Triton")
    parser.add_argument("--compression-kernel", choices=("cupy", "triton"),
                        default="cupy", help="0102C opt-in block-compressor Triton")
    parser.add_argument(
        "--retain-attention-probs", action="store_true",
        help="retain probability matrices for debugging (large VRAM cost); "
             "default recomputes them in backward",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default=None,
        help="default: float32 on NumPy, bfloat16 on CuPy",
    )
    parser.add_argument(
        "--skip-backward", action="store_true",
        help="omit backward and end-to-end forward+backward measurements",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="omit native one-token prefill/decode/cache measurements",
    )
    parser.add_argument(
        "--inference-decode-tokens", type=int, default=16,
        help="timed decode tokens after each sequential prefill (default: 16)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="run small finite-output, strict-causality, EOS, and loss-mask checks",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="run validation and estimator reporting without scaling measurements",
    )
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("0102 scaling contract currently requires --batch-size 1")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    if args.inference_decode_tokens < 1:
        parser.error("inference-decode-tokens must be positive")
    if args.compression_depth < 0:
        parser.error("compression-depth must be non-negative")
    if args.compression_block_chunk < 1:
        parser.error("compression-block-chunk must be positive")
    if args.compression_dim % args.compression_heads:
        parser.error("compression-dim must be divisible by compression-heads")
    if args.local_window < args.block_size or args.local_window % args.block_size:
        parser.error("local-window must be >= and divisible by block-size")
    for context in args.contexts:
        if context % args.block_size:
            parser.error(f"context {context} is not divisible by block-size")
        if context > args.block_size * 512:
            parser.error(f"context {context} exceeds block-size * max-blocks")
    if args.validate_only:
        args.validate = True
    return args


def _sync(backend_name, xp) -> None:
    if backend_name == "cupy":
        xp.cuda.get_current_stream().synchronize()


def _time_call(call, *, warmup: int, repeats: int, backend_name: str, xp):
    for _ in range(warmup):
        result = call()
        _sync(backend_name, xp)
        del result
    samples = []
    last = None
    for _ in range(repeats):
        if last is not None:
            del last
            gc.collect()
        _sync(backend_name, xp)
        started = time.perf_counter()
        last = call()
        _sync(backend_name, xp)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }, last


def _score_workload(
    context: int, *, heads: int, compressor_heads: int, block_size: int,
    local_window: int, query_chunk: int, compression_depth: int,
) -> dict:
    blocks = context // block_size
    local_rectangles = 0
    for query_start in range(0, context, query_chunk):
        query_stop = min(context, query_start + query_chunk)
        key_start = max(0, query_start - local_window + 1)
        local_rectangles += heads * (query_stop - query_start) * (query_stop - key_start)
    local_valid = heads * sum(min(local_window, position + 1) for position in range(context))
    global_rectangles = heads * context * blocks
    global_valid = heads * block_size * blocks * (blocks - 1) // 2
    compressor = compression_depth * blocks * compressor_heads * block_size * block_size
    pooling = blocks * compressor_heads * block_size
    split = {
        "local_chunk_rectangles": int(local_rectangles),
        "local_causally_valid": int(local_valid),
        "global_chunk_rectangles": int(global_rectangles),
        "global_strictly_valid": int(global_valid),
        "compressor_bidirectional": int(compressor),
        "compressor_pooling": int(pooling),
    }
    split["implementation_total"] = int(
        local_rectangles + global_rectangles + compressor + pooling
    )
    split["dense_token_square_not_allocated"] = int(heads * context * context)
    # Diagnostic: what the old retained-FP32 probability cache would cost.
    # In the default recompute mode this is NOT retained across backward.
    split["retained_probability_bytes_without_recompute"] = int(
        split["implementation_total"] * np.dtype(np.float32).itemsize
    )
    return split


def _parameter_counts(base, compression_heads: int) -> dict:
    counts = {}
    for dim in PARAMETER_DIMS:
        if dim % compression_heads:
            counts[str(dim)] = None
            continue
        dual = replace(
            base.dual_resolution,
            compression_dim=dim,
            compression_heads=compression_heads,
        )
        counts[str(dim)] = int(
            replace(base, dual_resolution=dual).estimated_parameter_count()
        )
    return counts


def _operator_array_accounting(estimator_counts: dict) -> dict:
    """Parameter estimator now includes summary K/V and both pool scalars."""
    return {
        "known_estimator_delta_per_layer": 0,
        "known_estimator_delta_20_layers": 0,
        "implied_trainable_array_count_by_compression_dim": dict(estimator_counts),
    }


def _pool_snapshot(backend_name: str, xp) -> dict | None:
    if backend_name != "cupy":
        return None
    pool = xp.get_default_memory_pool()
    return {
        "used_bytes": int(pool.used_bytes()),
        "reserved_bytes": int(pool.total_bytes()),
    }


def _random_array(xp, shape, dtype, seed: int):
    rng = xp.random.RandomState(seed)
    return rng.normal(0.0, 0.02, shape).astype(dtype)


def _make_attention(args, *, config, dtype, seed):
    from mini_llm.backend import RandomStream
    from mini_llm.ops.dual_resolution_attention import DualResolutionAttention

    return DualResolutionAttention(
        d_model=1024,
        n_q_heads=16,
        n_kv_heads=4,
        d_head=64,
        input_std=0.02,
        output_std=0.02,
        rope_base=1_000_000.0,
        rng=RandomStream(seed),
        dtype=dtype,
        config=config,
    )


def _validate_contract() -> dict:
    from mini_llm.backend import RandomStream, xp
    from mini_llm.config import DualResolutionConfig
    from mini_llm.ops.dual_resolution_attention import (
        DualResolutionAttention,
        build_dual_resolution_metadata,
        dual_resolution_loss_mask,
    )

    config = DualResolutionConfig(
        enabled=True, block_size=4, local_window=4, compression_dim=8,
        compression_heads=2, compression_depth=1, max_blocks=4,
        global_query_chunk=2,
    )
    attention = DualResolutionAttention(
        d_model=16, n_q_heads=4, n_kv_heads=2, d_head=4,
        input_std=0.02, output_std=0.02, rope_base=10_000.0,
        rng=RandomStream(102), dtype="float32", config=config,
    )
    x = xp.asarray(np.random.default_rng(103).normal(0, 0.1, (1, 8, 16)).astype(np.float32))
    regular = attention.forward(x, return_cache=False)
    future = x.copy()
    future[:, 7, :] += 3.0
    future_output = attention.forward(future, return_cache=False)
    regular_host = np.asarray(regular)
    future_host = np.asarray(future_output)
    causal_error = float(np.max(np.abs(regular_host[:, :7] - future_host[:, :7])))

    eos = 3
    token_ids = np.asarray([[10, 11, eos, 20, 21, 22, 23, 24]], dtype=np.int64)
    metadata = build_dual_resolution_metadata(token_ids, eos, block_size=4)
    packed = attention.forward(x, return_cache=False, dual_resolution_metadata=metadata)
    changed = x.copy()
    changed[:, :3, :] += 5.0
    packed_changed = attention.forward(
        changed, return_cache=False, dual_resolution_metadata=metadata
    )
    packed_host = np.asarray(packed)
    packed_changed_host = np.asarray(packed_changed)
    eos_error = float(np.max(np.abs(packed_host[:, 3:] - packed_changed_host[:, 3:])))
    loss_mask = dual_resolution_loss_mask(token_ids, eos)
    finite = bool(np.all(np.isfinite(regular_host)))
    checks = {
        "finite_output": finite,
        "future_perturbation_max_abs_before_future": causal_error,
        "prior_document_perturbation_max_abs_after_eos": eos_error,
        "document_ids": metadata["document_ids"].tolist(),
        "position_ids": metadata["position_ids"].tolist(),
        "loss_mask": loss_mask.tolist(),
    }
    if not finite or causal_error > 1e-6 or eos_error > 1e-6:
        raise AssertionError(f"dual-resolution validation failed: {checks}")
    expected_mask = [[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    if loss_mask.tolist() != expected_mask:
        raise AssertionError(f"unexpected EOS loss mask: {loss_mask.tolist()}")
    return checks


def _inference_measure(attention, *, context, decode_tokens, dtype, backend_name, xp):
    from mini_llm.ops.dual_resolution_attention_inference import (
        DualResolutionAttentionInference,
    )

    operator = DualResolutionAttentionInference(
        1024, 16, 4, 64, dual_config=attention.config,
        rope_base=attention.rope_base, dtype=dtype,
    )
    operator.set_weights(attention)
    block_size = attention.block_size
    local_window = attention.local_window
    max_blocks = int(attention.config.max_blocks)
    local_k = xp.empty((1, 4, local_window, 64), dtype=dtype)
    local_v = xp.empty_like(local_k)
    summary_k = xp.empty((1, 4, max_blocks, 64), dtype=dtype)
    summary_v = xp.empty_like(summary_k)
    incomplete = xp.empty((1, block_size, 1024), dtype=dtype)
    local_start = xp.zeros((1,), dtype=xp.int32)
    local_count = xp.zeros((1,), dtype=xp.int32)
    summary_start = xp.zeros((1,), dtype=xp.int32)
    summary_count = xp.zeros((1,), dtype=xp.int32)
    incomplete_count = xp.zeros((1,), dtype=xp.int32)
    positions = xp.zeros((1,), dtype=xp.int64)
    token = _random_array(xp, (1, 1, 1024), dtype, 104)

    def step():
        result = operator.decode_one(
            token, local_k, local_v, summary_k, summary_v, incomplete,
            local_start, local_count, summary_start, summary_count,
            incomplete_count, positions,
        )
        positions[...] += 1
        return result

    _sync(backend_name, xp)
    started = time.perf_counter()
    last = None
    for _ in range(context):
        last = step()
    _sync(backend_name, xp)
    prefill_ms = (time.perf_counter() - started) * 1000.0
    _sync(backend_name, xp)
    started = time.perf_counter()
    for _ in range(decode_tokens):
        last = step()
    _sync(backend_name, xp)
    decode_ms = (time.perf_counter() - started) * 1000.0
    del last
    arrays = (local_k, local_v, summary_k, summary_v, incomplete)
    cache_bytes = int(sum(array.nbytes for array in arrays))
    return {
        "scope": "single_attention_layer_sequential_native_operator",
        "prefill_tokens": context,
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": context * 1000.0 / prefill_ms,
        "decode_tokens": decode_tokens,
        "decode_ms": decode_ms,
        "decode_tokens_per_second": decode_tokens * 1000.0 / decode_ms,
        "allocated_tensor_cache_bytes_per_layer": cache_bytes,
        "estimated_20_layer_tensor_cache_bytes": cache_bytes * 20,
        "local_count": int(local_count[0].item()),
        "summary_count": int(summary_count[0].item()),
        "incomplete_count": int(incomplete_count[0].item()),
    }


def _measure_context(args, *, context, config, dtype, backend_name, xp):
    attention = _make_attention(args, config=config, dtype=dtype, seed=args.seed)
    x = _random_array(xp, (1, context, 1024), dtype, args.seed + context)
    upstream = None if args.skip_backward else xp.ones_like(x)
    baseline_memory = _pool_snapshot(backend_name, xp)

    def forward():
        return attention.forward(x, return_cache=True)

    forward_timing, forward_result = _time_call(
        forward, warmup=args.warmup, repeats=args.repeats,
        backend_name=backend_name, xp=xp,
    )
    after_forward_memory = _pool_snapshot(backend_name, xp)
    if forward_result is None:
        raise RuntimeError("forward timing returned no result")
    output, cache = forward_result
    result = {
        "context": context,
        "batch_size": 1,
        "tokens": context,
        "recompute_attention_probs": config.recompute_attention_probs,
        "score_workload": _score_workload(
            context, heads=16, compressor_heads=args.compression_heads,
            block_size=args.block_size, local_window=args.local_window,
            query_chunk=args.global_query_chunk,
            compression_depth=args.compression_depth,
        ),
        "forward": forward_timing,
        "forward_tokens_per_second": context * 1000.0 / forward_timing["median_ms"],
        "memory_before_forward": baseline_memory,
        "memory_after_forward_cache_live": after_forward_memory,
    }
    if not args.skip_backward:
        def backward():
            attention.zero_grad()
            return attention.backward(upstream, cache)

        backward_timing, dx = _time_call(
            backward, warmup=args.warmup, repeats=args.repeats,
            backend_name=backend_name, xp=xp,
        )
        del dx

        def end_to_end():
            attention.zero_grad()
            y, end_cache = attention.forward(x, return_cache=True)
            dx_value = attention.backward(upstream, end_cache)
            return y, dx_value

        end_timing, end_result = _time_call(
            end_to_end, warmup=args.warmup, repeats=args.repeats,
            backend_name=backend_name, xp=xp,
        )
        del end_result
        result.update({
            "backward": backward_timing,
            "end_to_end": end_timing,
            "end_to_end_tokens_per_second": context * 1000.0 / end_timing["median_ms"],
            "memory_after_backward": _pool_snapshot(backend_name, xp),
        })
    if not args.skip_inference:
        result["inference"] = _inference_measure(
            attention, context=context,
            decode_tokens=args.inference_decode_tokens, dtype=dtype,
            backend_name=backend_name, xp=xp,
        )
    del output, cache, x, upstream, attention
    gc.collect()
    if backend_name == "cupy":
        xp.get_default_memory_pool().free_all_blocks()
    return result


def _print_summary(report: dict) -> None:
    print("0102 dual-resolution benchmark")
    print(
        f"backend={report['backend']} dtype={report['dtype']} "
        f"local_kernel={report['local_kernel']} "
        f"global_kernel={report['global_kernel']} "
        f"compression_kernel={report['compression_kernel']}"
    )
    print("parameter counts (actual ModelConfig estimator):")
    for dim, count in report["parameter_counts_by_compression_dim"].items():
        print(f"  dc={dim}: {count:,}" if count is not None else f"  dc={dim}: incompatible")
    accounting = report["operator_array_accounting"]
    print(
        "known estimator/operator delta: "
        f"{accounting['known_estimator_delta_20_layers']:+,} arrays elements"
    )
    if report.get("validation") is not None:
        print("validation: PASS")
    for row in report["measurements"]:
        forward = row["forward"]["median_ms"]
        text = f"T={row['context']:>5} forward={forward:>10.3f} ms"
        if "backward" in row:
            text += (
                f" backward={row['backward']['median_ms']:>10.3f} ms"
                f" end-to-end={row['end_to_end']['median_ms']:>10.3f} ms"
                f" tokens/s={row['end_to_end_tokens_per_second']:>10.2f}"
            )
        else:
            text += f" tokens/s={row['forward_tokens_per_second']:>10.2f}"
        if "inference" in row:
            text += f" decode tok/s={row['inference']['decode_tokens_per_second']:>10.2f}"
        print(text)


def main() -> None:
    args = _parse_args()
    # Must be set before constructing the benchmarked attention operator.
    os.environ["MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL"] = args.local_kernel
    os.environ["MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL"] = args.global_kernel
    os.environ["MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL"] = args.compression_kernel
    from mini_llm.backend import BACKEND_NAME, xp
    if any(kernel == "triton" for kernel in
           (args.local_kernel, args.global_kernel, args.compression_kernel)) \
            and BACKEND_NAME != "cupy":
        raise SystemExit("Triton kernels require MINI_LLM_BACKEND=cupy")
    if args.compression_kernel == "triton" and args.compression_dim // args.compression_heads != 64:
        raise SystemExit("Triton compression requires compression head dimension 64")
    from mini_llm.config import ModelConfig

    dtype = args.dtype or ("bfloat16" if BACKEND_NAME == "cupy" else "float32")
    base = ModelConfig.dual_resolution_1024_20l()
    dual = replace(
        base.dual_resolution,
        block_size=args.block_size,
        local_window=args.local_window,
        compression_dim=args.compression_dim,
        compression_heads=args.compression_heads,
        compression_depth=args.compression_depth,
        compression_block_chunk=args.compression_block_chunk,
        global_query_chunk=args.global_query_chunk,
        recompute_attention_probs=not args.retain_attention_probs,
    )
    benchmark_config = replace(base, dual_resolution=dual)
    parameter_counts = _parameter_counts(base, args.compression_heads)
    report = {
        "schema": "dual-resolution-0102-benchmark-v1",
        "implementation_status": (
            "experimental-triton-partial-unvalidated"
            if any(kernel == "triton" for kernel in
                   (args.local_kernel, args.global_kernel, args.compression_kernel))
            else "correctness-first-unfused"
        ),
        "backend": BACKEND_NAME,
        "local_kernel": args.local_kernel,
        "global_kernel": args.global_kernel,
        "compression_kernel": args.compression_kernel,
        "dtype": dtype,
        "batch_size": args.batch_size,
        "architecture": {
            "layers": 20, "d_model": 1024, "d_ff": 4096,
            "query_heads": 16, "kv_heads": 4, "head_dim": 64,
            "block_size": args.block_size, "local_window": args.local_window,
            "compression_dim": args.compression_dim,
            "compression_heads": args.compression_heads,
            "compression_depth": args.compression_depth,
            "compression_block_chunk": args.compression_block_chunk,
            "max_blocks": dual.max_blocks,
            "global_query_chunk": args.global_query_chunk,
            "recompute_attention_probs": dual.recompute_attention_probs,
        },
        "selected_parameter_count": int(benchmark_config.estimated_parameter_count()),
        "parameter_counts_by_compression_dim": parameter_counts,
        "operator_array_accounting": _operator_array_accounting(parameter_counts),
        "validation": _validate_contract() if args.validate else None,
        "measurements": [],
    }
    if not args.validate_only:
        for context in args.contexts:
            print(f"measuring context {context:,}...", flush=True)
            report["measurements"].append(
                _measure_context(
                    args, context=context, config=dual, dtype=dtype,
                    backend_name=BACKEND_NAME, xp=xp,
                )
            )
    _print_summary(report)
    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
