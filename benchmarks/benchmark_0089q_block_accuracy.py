#!/usr/bin/env python3
"""Production-shape arithmetic sweep for deterministic historical block reads."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import profile_0087a_training as profile

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("0089Q requires Triton 3.8 or newer") from exc


_LOG2E = 1.4426950408889634


class _Pointer:
    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return self.array.data.ptr


def _ptr(array):
    name = str(array.dtype)
    if name == "bfloat16":
        dtype = tl.bfloat16
    elif name == "bool":
        dtype = tl.int1
    else:
        dtype = tl.float32
    return _Pointer(array, dtype)


def _strides(array):
    return tuple(int(value // array.dtype.itemsize) for value in array.strides)


@triton.jit
def _sweep_kernel(
    q,
    k,
    v,
    valid,
    output,
    q_sb: tl.constexpr,
    q_st: tl.constexpr,
    k_sb: tl.constexpr,
    k_st: tl.constexpr,
    v_sb: tl.constexpr,
    v_st: tl.constexpr,
    valid_sb: tl.constexpr,
    valid_st: tl.constexpr,
    out_sb: tl.constexpr,
    out_st: tl.constexpr,
    query_length: tl.constexpr,
    history_length: tl.constexpr,
    first_local: tl.constexpr,
    selected_tokens: tl.constexpr,
    period: tl.constexpr,
    scale_log2: tl.constexpr,
    use_ieee: tl.constexpr,
    two_pass: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, 64)
    query_mask = offs_m[:, None] < query_length
    q_block = tl.load(
        q + batch * q_sb + offs_m[:, None] * q_st + offs_d[None, :],
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)
    row_max = tl.full((block_m,), -float("inf"), tl.float32)

    if two_pass:
        for selected_start in tl.range(0, selected_tokens, block_n, num_stages=2):
            selected_offsets = selected_start + offs_n
            source_offsets = (
                first_local
                + (selected_offsets // 32) * period
                + selected_offsets % 32
            )
            key_mask = (
                (selected_offsets < selected_tokens)
                & (source_offsets < history_length)
            )
            key_valid = tl.load(
                valid + batch * valid_sb + source_offsets * valid_st,
                mask=key_mask,
                other=0,
            ).to(tl.int1)
            k_block = tl.load(
                k + batch * k_sb + source_offsets[:, None] * k_st + offs_d[None, :],
                mask=key_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if use_ieee:
                scores = tl.dot(
                    q_block, tl.trans(k_block), input_precision="ieee"
                ) * scale_log2
            else:
                scores = tl.dot(
                    q_block, tl.trans(k_block), input_precision="tf32x3"
                ) * scale_log2
            attention_mask = (
                (offs_m[:, None] < query_length)
                & key_mask[None, :]
                & key_valid[None, :]
            )
            scores = tl.where(attention_mask, scores, -float("inf"))
            tile_has_values = tl.sum(attention_mask.to(tl.int32), axis=1) > 0
            tile_max = tl.max(scores, axis=1)
            row_max = tl.where(
                tile_has_values, tl.maximum(row_max, tile_max), row_max
            )

        row_sum = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, 64), tl.float32)
        for selected_start in tl.range(0, selected_tokens, block_n, num_stages=2):
            selected_offsets = selected_start + offs_n
            source_offsets = (
                first_local
                + (selected_offsets // 32) * period
                + selected_offsets % 32
            )
            key_mask = (
                (selected_offsets < selected_tokens)
                & (source_offsets < history_length)
            )
            key_valid = tl.load(
                valid + batch * valid_sb + source_offsets * valid_st,
                mask=key_mask,
                other=0,
            ).to(tl.int1)
            k_block = tl.load(
                k + batch * k_sb + source_offsets[:, None] * k_st + offs_d[None, :],
                mask=key_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if use_ieee:
                scores = tl.dot(
                    q_block, tl.trans(k_block), input_precision="ieee"
                ) * scale_log2
            else:
                scores = tl.dot(
                    q_block, tl.trans(k_block), input_precision="tf32x3"
                ) * scale_log2
            attention_mask = (
                (offs_m[:, None] < query_length)
                & key_mask[None, :]
                & key_valid[None, :]
            )
            probabilities = tl.where(
                attention_mask, tl.exp2(scores - row_max[:, None]), 0.0
            )
            row_sum += tl.sum(probabilities, axis=1)
            v_block = tl.load(
                v + batch * v_sb + source_offsets[:, None] * v_st + offs_d[None, :],
                mask=key_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if use_ieee:
                accumulator = tl.dot(
                    probabilities,
                    v_block,
                    accumulator,
                    input_precision="ieee",
                )
            else:
                accumulator = tl.dot(
                    probabilities,
                    v_block,
                    accumulator,
                    input_precision="tf32x3",
                )
    else:
        row_sum = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, 64), tl.float32)
        for selected_start in tl.range(0, selected_tokens, block_n, num_stages=2):
            selected_offsets = selected_start + offs_n
            source_offsets = (
                first_local
                + (selected_offsets // 32) * period
                + selected_offsets % 32
            )
            key_mask = (
                (selected_offsets < selected_tokens)
                & (source_offsets < history_length)
            )
            key_valid = tl.load(
                valid + batch * valid_sb + source_offsets * valid_st,
                mask=key_mask,
                other=0,
            ).to(tl.int1)
            k_block = tl.load(
                k + batch * k_sb + source_offsets[:, None] * k_st + offs_d[None, :],
                mask=key_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if use_ieee:
                scores = tl.dot(
                    q_block, tl.trans(k_block), input_precision="ieee"
                ) * scale_log2
            else:
                scores = tl.dot(
                    q_block, tl.trans(k_block), input_precision="tf32x3"
                ) * scale_log2
            attention_mask = (
                (offs_m[:, None] < query_length)
                & key_mask[None, :]
                & key_valid[None, :]
            )
            scores = tl.where(attention_mask, scores, -float("inf"))
            tile_has_values = tl.sum(attention_mask.to(tl.int32), axis=1) > 0
            tile_max = tl.max(scores, axis=1)
            new_max = tl.where(
                tile_has_values, tl.maximum(row_max, tile_max), row_max
            )
            probabilities = tl.where(
                attention_mask, tl.exp2(scores - new_max[:, None]), 0.0
            )
            alpha = tl.where(
                tile_has_values, tl.exp2(row_max - new_max), 1.0
            )
            row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            accumulator *= alpha[:, None]
            v_block = tl.load(
                v + batch * v_sb + source_offsets[:, None] * v_st + offs_d[None, :],
                mask=key_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if use_ieee:
                accumulator = tl.dot(
                    probabilities,
                    v_block,
                    accumulator,
                    input_precision="ieee",
                )
            else:
                accumulator = tl.dot(
                    probabilities,
                    v_block,
                    accumulator,
                    input_precision="tf32x3",
                )
            row_max = new_max

    nonempty = row_sum > 0.0
    result = tl.where(nonempty[:, None], accumulator / row_sum[:, None], 0.0)
    tl.store(
        output + batch * out_sb + offs_m[:, None] * out_st + offs_d[None, :],
        result,
        mask=query_mask,
    )


def _geometry(phase: int, period: int, source_offset: int, source_length: int):
    starts = []
    for start in range(phase, source_length, period):
        begin = max(start, source_offset)
        end = min(start + 32, source_length)
        if end > begin:
            if begin != start or end - begin != 32:
                raise ValueError("benchmark requires complete selected blocks")
            starts.append(start)
    if not starts:
        raise ValueError("empty deterministic selection")
    return starts[0] - source_offset, len(starts) * 32


def _candidate(q, k, v, valid, *, phase, period, source_offset, source_length,
               use_ieee, two_pass):
    import cupy as cp

    first_local, selected_tokens = _geometry(
        phase, period, source_offset, source_length
    )
    output = cp.empty(q.shape, dtype=cp.float32)
    q_s, k_s, v_s, valid_s, out_s = map(_strides, (q, k, v, valid, output))
    block_m = block_n = 64
    _sweep_kernel[(triton.cdiv(int(q.shape[1]), block_m), int(q.shape[0]))](
        _ptr(q), _ptr(k), _ptr(v), _ptr(valid), _ptr(output),
        q_s[0], q_s[1], k_s[0], k_s[1], v_s[0], v_s[1],
        valid_s[0], valid_s[1], out_s[0], out_s[1],
        query_length=int(q.shape[1]),
        history_length=int(k.shape[1]),
        first_local=first_local,
        selected_tokens=selected_tokens,
        period=period,
        scale_log2=(64.0 ** -0.5) * _LOG2E,
        use_ieee=use_ieee,
        two_pass=two_pass,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=3,
    )
    return output


def _metrics(actual, expected) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    difference = actual - expected
    ref_norm = float(np.linalg.norm(expected))
    actual_norm = float(np.linalg.norm(actual))
    denominator = ref_norm * actual_norm
    cosine = (
        1.0 if denominator == 0.0 and not np.any(difference)
        else 0.0 if denominator == 0.0
        else float(np.vdot(expected.reshape(-1), actual.reshape(-1)).real / denominator)
    )
    return {
        "relative_l2": float(np.linalg.norm(difference)) / max(ref_norm, 1e-30),
        "cosine_similarity": cosine,
        "max_absolute": float(np.max(np.abs(difference))),
        "mean_absolute": float(np.mean(np.abs(difference))),
        "finite": bool(np.isfinite(actual).all()),
    }


def _timing(fn, *, warmup: int, repeats: int) -> dict:
    import cupy as cp

    for _ in range(warmup):
        result = fn()
        del result
    cp.cuda.runtime.deviceSynchronize()
    samples = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        del result
    median = statistics.median(samples)
    return {
        "count": len(samples),
        "median_ms": median,
        "p10_ms": profile._percentile(samples, 0.1),
        "p90_ms": profile._percentile(samples, 0.9),
        "mad_ms": statistics.median(abs(value - median) for value in samples),
        "samples_ms": samples,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixed-shard-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name, value in profile._PRODUCTION_BENCH_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_BACKEND", "triton")
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    os.environ.setdefault("MINI_LLM_FUSED_QKV_ROPE_FORWARD", "1")

    import cupy as cp
    from mini_llm.backend import BACKEND_NAME, asnumpy
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.historical_read import HistoricalReadMemory
    from mini_llm.ops.historical_schedule import source_indices
    from mini_llm.ops.triton_historical_block import (
        deterministic_block_read_forward,
    )
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy":
        raise RuntimeError("0089Q requires the CuPy backend")
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

    captured = {}
    original_apply = model.historical_memory.apply

    class CaptureComplete(Exception):
        pass

    def capture_apply(context, q, layer_idx, state, *, return_cache=True):
        if int(layer_idx) == args.layer:
            slot = next(
                item for item in model.historical_memory.slots[layer_idx]
                if item.kind == "block"
            )
            captured.update({
                "q": cp.array(q[:, :, slot.head, :], copy=True),
                "k": cp.array(state.k, copy=True),
                "v": cp.array(state.v, copy=True),
                "valid": cp.array(state.valid, copy=True),
                "slot": slot,
                "source_offset": int(state.source_offset),
                "source_length": int(state.source_length),
            })
            raise CaptureComplete
        return original_apply(
            context, q, layer_idx, state, return_cache=return_cache
        )

    model.historical_memory.apply = capture_apply
    try:
        model.forward_body(
            inputs,
            finite_trace=None,
            activation_checkpoint=False,
            memory_metadata=metadata,
        )
    except CaptureComplete:
        pass
    else:
        raise RuntimeError(f"layer {args.layer} historical Q was not captured")
    finally:
        model.historical_memory.apply = original_apply
    del model, parameters, loaded, inputs, batch
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    q = captured["q"]
    k = captured["k"]
    v = captured["v"]
    valid = captured["valid"]
    slot = captured["slot"]
    source_offset = captured["source_offset"]
    source_length = captured["source_length"]
    selected_host = np.asarray(
        source_indices(
            slot, source_length, valid_start=source_offset, block_size=32
        ),
        dtype=np.int64,
    ) - source_offset
    selected = cp.asarray(selected_host)
    reader = SimpleNamespace(chunk=1024)

    def reference_call():
        result, _ = HistoricalReadMemory._read(
            reader,
            q,
            k[:, selected, :],
            v[:, selected, :],
            valid[:, selected],
        )
        return result

    expected = reference_call()
    cp.cuda.runtime.deviceSynchronize()
    expected_host = asnumpy(expected)
    expected_bf16 = expected.astype(cp.dtype("bfloat16"))
    expected_bf16_f32 = asnumpy(expected_bf16.astype(cp.float32))
    reference_timing = _timing(
        reference_call, warmup=args.warmup, repeats=args.repeats
    )

    variants = [
        ("ieee_online", True, False),
        ("tf32x3_two_pass", False, True),
        ("ieee_two_pass", True, True),
    ]
    rows = []
    for name, use_ieee, two_pass in variants:
        if name == "tf32x3_two_pass":
            call = lambda: deterministic_block_read_forward(
                q, k, v, valid,
                phase=int(slot.phase),
                period=int(slot.period),
                source_offset=source_offset,
                source_length=source_length,
            )
        else:
            call = lambda use_ieee=use_ieee, two_pass=two_pass: _candidate(
                q, k, v, valid,
                phase=int(slot.phase),
                period=int(slot.period),
                source_offset=source_offset,
                source_length=source_length,
                use_ieee=use_ieee,
                two_pass=two_pass,
            )
        actual = call()
        cp.cuda.runtime.deviceSynchronize()
        actual_host = asnumpy(actual)
        actual_bf16 = actual.astype(cp.dtype("bfloat16"))
        actual_bf16_f32 = asnumpy(actual_bf16.astype(cp.float32))
        actual_bits = asnumpy(actual_bf16.view(cp.uint16))
        expected_bits = asnumpy(expected_bf16.view(cp.uint16))
        timing = _timing(call, warmup=args.warmup, repeats=args.repeats)
        rows.append({
            "name": name,
            "use_ieee": use_ieee,
            "two_pass": two_pass,
            "fp32": _metrics(actual_host, expected_host),
            "bf16_quantized": {
                **_metrics(actual_bf16_f32, expected_bf16_f32),
                "mismatched_elements": int(np.count_nonzero(actual_bits != expected_bits)),
                "total_elements": int(actual_bits.size),
                "mismatch_fraction": float(np.count_nonzero(
                    actual_bits != expected_bits
                )) / actual_bits.size,
            },
            "timing": timing,
            "speedup_over_reference": (
                reference_timing["median_ms"] / timing["median_ms"]
            ),
        })
        del actual, actual_bf16

    result = {
        "schema_version": "0089Q-block-accuracy-v1",
        "fixture": fixture,
        "geometry": {
            "layer": args.layer,
            "head": int(slot.head),
            "phase": int(slot.phase),
            "period": int(slot.period),
            "batch": int(q.shape[0]),
            "query_length": int(q.shape[1]),
            "history_length": int(k.shape[1]),
            "selected_tokens": int(selected.size),
            "head_dim": int(q.shape[2]),
            "valid_tokens_per_sample": asnumpy(valid.sum(axis=1)).tolist(),
        },
        "reference_timing": reference_timing,
        "variants": rows,
        "effective_flags": {
            name: os.environ.get(name) for name in profile._REPORT_ENV_NAMES
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
