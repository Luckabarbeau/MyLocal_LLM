#!/usr/bin/env python3
"""Detailed cached-inference cost benchmark.

The normal text-generation timer answers "how many tokens/s?" but not *why*.
This benchmark is deliberately decode-centric and has three phases:

1. Uninstrumented throughput at several cache positions.  This is the number to
   compare against normal generation because CUDA-event instrumentation is not
   present in the hot path.
2. A short instrumented decode trace using deferred CUDA events.  It separates
   embedding, Transformer blocks, RMSNorm, sparse attention, MoE, final norm,
   and the vocabulary projection while also exposing host submission time.
3. Isolated microbenchmarks for suspected single-token bottlenecks, including
   the full 65k LM head, a 16k-vocabulary slice, current MoE decode, a fixed
   two-expert BF16 math path, RMSNorm, and sampling.

This script is diagnostic only.  It does not change model weights or inference
semantics and deliberately feeds deterministic synthetic token IDs so tokenizer
or text-decoding overhead cannot pollute the model timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow `python benchmarks/benchmark_inference_decode_components.py ...` from
# the repository root without requiring PYTHONPATH=.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.sparse_decode_attention import (
    fused_sparse_decode_enabled, fused_sparse_decode_failure_reason,
)
from mini_llm.ops.decode_qkv_fusion import (
    fused_decode_qkv_enabled, fused_decode_qkv_failure_reason,
)

# Reuse the exact production sampling helpers rather than duplicating them.
from inference import greedy_decode, temperature_sample, top_k_sample


class RegionRecorder:
    """Deferred host/CUDA timing for temporarily wrapped instance methods."""

    def __init__(self):
        self.host_total = defaultdict(float)
        self.calls = defaultdict(int)
        self.gpu_total_ms = defaultdict(float)
        self._pending = []
        self._restore = []

    def _call(self, name, fn, *args, **kwargs):
        host_start = time.perf_counter()
        if BACKEND_NAME == "cupy":
            start = xp.cuda.Event()
            end = xp.cuda.Event()
            start.record()
        else:
            start = end = None
        try:
            return fn(*args, **kwargs)
        finally:
            if BACKEND_NAME == "cupy":
                end.record()
                self._pending.append((name, start, end))
            else:
                self.gpu_total_ms[name] += (time.perf_counter() - host_start) * 1e3
            self.host_total[name] += time.perf_counter() - host_start
            self.calls[name] += 1

    def wrap(self, obj, method_name, region_name):
        original = getattr(obj, method_name)

        def wrapped(*args, **kwargs):
            return self._call(region_name, original, *args, **kwargs)

        setattr(obj, method_name, wrapped)
        self._restore.append((obj, method_name, original))

    def wrap_attention(self, attention, layer_idx):
        """Wrap sparse-attention internals and label single-query group calls."""
        base = f"layer.{layer_idx}.attention"
        static_kinds = [str(g["kind"]) for g in attention._static_head_groups]
        call_state = {"single": 0}

        original_decode = attention.decode_one
        original_single = attention._single_query_attention
        original_indices = attention._decode_indices
        original_route = attention._retrieval_route_for_position

        def decode_wrapped(*args, **kwargs):
            call_state["single"] = 0
            return self._call(f"{base}.total", original_decode, *args, **kwargs)

        def single_wrapped(*args, **kwargs):
            idx = call_state["single"]
            call_state["single"] += 1
            kind = static_kinds[idx] if idx < len(static_kinds) else "retrieval"
            return self._call(f"{base}.group.{kind}", original_single, *args, **kwargs)

        def indices_wrapped(group, *args, **kwargs):
            kind = str(group.get("kind", "unknown"))
            return self._call(
                f"{base}.indices.{kind}", original_indices, group, *args, **kwargs
            )

        def route_wrapped(*args, **kwargs):
            return self._call(f"{base}.router", original_route, *args, **kwargs)

        attention.decode_one = decode_wrapped
        attention._single_query_attention = single_wrapped
        attention._decode_indices = indices_wrapped
        attention._retrieval_route_for_position = route_wrapped
        self._restore.extend(
            [
                (attention, "decode_one", original_decode),
                (attention, "_single_query_attention", original_single),
                (attention, "_decode_indices", original_indices),
                (attention, "_retrieval_route_for_position", original_route),
            ]
        )

        self.wrap(attention, "_project_and_store", f"{base}.project_store")
        self.wrap(attention, "_apply_rope_absolute", f"{base}.rope")
        if hasattr(attention, "_fused_sparse_decode_context"):
            self.wrap(
                attention,
                "_fused_sparse_decode_context",
                f"{base}.fused_sparse_decode",
            )

    def resolve(self):
        if BACKEND_NAME == "cupy" and self._pending:
            synchronize()
            pending = self._pending
            self._pending = []
            for name, start, end in pending:
                self.gpu_total_ms[name] += float(xp.cuda.get_elapsed_time(start, end))

    def restore(self):
        for obj, name, original in reversed(self._restore):
            setattr(obj, name, original)
        self._restore = []

    def avg_gpu_ms(self, name):
        calls = self.calls.get(name, 0)
        return self.gpu_total_ms.get(name, 0.0) / max(calls, 1)

    def avg_host_ms(self, name):
        calls = self.calls.get(name, 0)
        return self.host_total.get(name, 0.0) * 1e3 / max(calls, 1)



def parse_positions(text):
    positions = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("positions must be positive")
        positions.append(value)
    if not positions:
        raise ValueError("at least one position is required")
    return positions



def load_models(checkpoint: Path):
    with open(checkpoint / "config.json", "r") as f:
        config = ModelConfig(**json.load(f))

    training_model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    names = [p.name for p in training_model.parameters()]
    loaded, _, _ = load_checkpoint(checkpoint, param_names=names, skip_optimizer=True)
    for p in training_model.parameters():
        if p.name in loaded:
            p.data[...] = loaded[p.name]

    inference_model = InferenceModel(config, dtype=config.dtype)
    inference_model.set_weights(training_model)
    return config, training_model, inference_model, len(loaded)



def synthetic_ids(vocab_size, length, seed):
    rng = np.random.default_rng(int(seed))
    # Avoid a special-token-heavy synthetic prompt while retaining a realistic
    # broad token distribution.  Cost is insensitive to the exact IDs.
    low = 8 if vocab_size > 16 else 0
    ids = rng.integers(low, vocab_size, size=(1, int(length)), dtype=np.int32)
    return xp.asarray(ids, dtype=xp.int32)



def timed_call_loop(fn, iterations, warmup=5):
    for _ in range(int(warmup)):
        fn()
    synchronize()

    host_start = time.perf_counter()
    if BACKEND_NAME == "cupy":
        start = xp.cuda.Event()
        end = xp.cuda.Event()
        start.record()
    else:
        start = end = None

    for _ in range(int(iterations)):
        fn()

    if BACKEND_NAME == "cupy":
        end.record()
    synchronize()
    host_elapsed = time.perf_counter() - host_start

    if BACKEND_NAME == "cupy":
        gpu_ms = float(xp.cuda.get_elapsed_time(start, end)) / max(iterations, 1)
    else:
        gpu_ms = host_elapsed * 1e3 / max(iterations, 1)
    host_ms = host_elapsed * 1e3 / max(iterations, 1)
    return gpu_ms, host_ms



def make_state(model, config, position, max_context, seed):
    state = model.create_generation_state(batch_size=1, max_length=max_context)
    ids = synthetic_ids(config.vocab_size, position, seed)
    synchronize()
    if BACKEND_NAME == "cupy":
        start = xp.cuda.Event()
        end = xp.cuda.Event()
        start.record()
    else:
        start = end = None
    host_start = time.perf_counter()
    logits = model.prefill(ids, state)
    if BACKEND_NAME == "cupy":
        end.record()
    synchronize()
    host_ms = (time.perf_counter() - host_start) * 1e3
    gpu_ms = (
        float(xp.cuda.get_elapsed_time(start, end))
        if BACKEND_NAME == "cupy"
        else host_ms
    )
    return state, logits, gpu_ms, host_ms



def throughput_sweep(model, config, positions, max_context, steps, warmup, seed):
    rows = []
    token = xp.asarray([[min(123, config.vocab_size - 1)]], dtype=xp.int32)

    for position in positions:
        needed = position + warmup + steps
        if needed > max_context:
            raise ValueError(
                f"position {position} + warmup {warmup} + steps {steps} "
                f"exceeds --max-context {max_context}"
            )
        state, _, prefill_gpu, prefill_host = make_state(
            model, config, position, max_context, seed + position
        )
        for _ in range(warmup):
            model.decode_one(token, state)
        synchronize()

        if BACKEND_NAME == "cupy":
            start = xp.cuda.Event()
            end = xp.cuda.Event()
            start.record()
        else:
            start = end = None
        host_start = time.perf_counter()
        for _ in range(steps):
            model.decode_one(token, state)
        if BACKEND_NAME == "cupy":
            end.record()
        synchronize()
        host_s = time.perf_counter() - host_start
        gpu_ms = (
            float(xp.cuda.get_elapsed_time(start, end)) / steps
            if BACKEND_NAME == "cupy"
            else host_s * 1e3 / steps
        )
        host_ms = host_s * 1e3 / steps
        rows.append(
            {
                "position": position,
                "measured_start": position + warmup,
                "prefill_gpu_ms": prefill_gpu,
                "prefill_host_ms": prefill_host,
                "decode_gpu_ms": gpu_ms,
                "decode_host_ms": host_ms,
                "gpu_tok_s": 1000.0 / gpu_ms if gpu_ms > 0 else 0.0,
                "host_tok_s": 1000.0 / host_ms if host_ms > 0 else 0.0,
            }
        )
    return rows



def current_greedy_loop(model, config, position, max_context, steps, seed):
    if position + steps > max_context:
        raise ValueError("greedy-loop benchmark exceeds cache capacity")
    state, logits, _, _ = make_state(model, config, position, max_context, seed)
    synchronize()
    host_start = time.perf_counter()
    if BACKEND_NAME == "cupy":
        start = xp.cuda.Event()
        end = xp.cuda.Event()
        start.record()
    else:
        start = end = None

    for _ in range(steps):
        next_id = greedy_decode(logits[0])
        token = xp.array([[next_id]], dtype=xp.int32)
        logits = model.decode_one(token, state)

    if BACKEND_NAME == "cupy":
        end.record()
    synchronize()
    host_s = time.perf_counter() - host_start
    gpu_ms = (
        float(xp.cuda.get_elapsed_time(start, end)) / steps
        if BACKEND_NAME == "cupy"
        else host_s * 1e3 / steps
    )
    host_ms = host_s * 1e3 / steps
    return {
        "gpu_ms": gpu_ms,
        "host_ms": host_ms,
        "gpu_tok_s": 1000.0 / gpu_ms,
        "host_tok_s": 1000.0 / host_ms,
    }



def install_profile_wrappers(model, recorder):
    recorder.wrap(model, "decode_one", "model.total")
    recorder.wrap(model, "_embedding_lookup", "model.embedding")
    recorder.wrap(model, "_lm_head_forward", "model.lm_head")
    recorder.wrap(model.final_norm, "forward", "model.final_norm")

    for i, block in enumerate(model.blocks):
        recorder.wrap(block, "decode_one", f"layer.{i}.block")
        recorder.wrap(block.norm1, "forward", f"layer.{i}.norm1")
        recorder.wrap(block.norm2, "forward", f"layer.{i}.norm2")
        recorder.wrap_attention(block.attention, i)
        recorder.wrap(block.moe, "decode_one", f"layer.{i}.moe.total")
        recorder.wrap(block.moe.router, "forward", f"layer.{i}.moe.router")
        recorder.wrap(block.moe.experts, "forward", f"layer.{i}.moe.experts")



def instrumented_profile(model, config, position, max_context, steps, warmup, seed):
    if position + warmup + steps > max_context:
        raise ValueError("instrumented profile exceeds cache capacity")
    token = xp.asarray([[min(123, config.vocab_size - 1)]], dtype=xp.int32)
    state, _, _, _ = make_state(model, config, position, max_context, seed)
    for _ in range(warmup):
        model.decode_one(token, state)
    synchronize()

    recorder = RegionRecorder()
    install_profile_wrappers(model, recorder)
    try:
        for _ in range(steps):
            model.decode_one(token, state)
        recorder.resolve()
    finally:
        recorder.restore()
    return recorder



def sum_region(rec, suffix):
    names = [name for name in rec.calls if name.endswith(suffix)]
    gpu = sum(rec.gpu_total_ms[name] for name in names)
    host = sum(rec.host_total[name] * 1e3 for name in names)
    calls = sum(rec.calls[name] for name in names)
    # These are totals over the complete profiled run, not avg/call.  Divide by
    # model.total calls to obtain a per-generated-token contribution.
    tokens = max(rec.calls.get("model.total", 1), 1)
    return gpu / tokens, host / tokens, calls



def exact_region_per_token(rec, name):
    tokens = max(rec.calls.get("model.total", 1), 1)
    return (
        rec.gpu_total_ms.get(name, 0.0) / tokens,
        rec.host_total.get(name, 0.0) * 1e3 / tokens,
    )



def print_breakdown(rec, n_layers):
    total_gpu, total_host = exact_region_per_token(rec, "model.total")
    print("\n=== Instrumented decode breakdown ===")
    print(
        "CUDA events add diagnostic overhead; use the uninstrumented sweep for "
        "the true tok/s baseline. Percentages below use inclusive GPU regions."
    )
    print(f"Profiled model total: {total_gpu:.3f} ms/token GPU, {total_host:.3f} ms/token host")

    def row(label, gpu, host, parent=total_gpu):
        pct = 100.0 * gpu / parent if parent > 0 else 0.0
        print(f"  {label:30s} {gpu:9.3f} ms GPU  {host:9.3f} ms host  {pct:6.1f}%")

    embed = exact_region_per_token(rec, "model.embedding")
    blocks_gpu = sum(rec.gpu_total_ms.get(f"layer.{i}.block", 0.0) for i in range(n_layers)) / max(rec.calls.get("model.total", 1), 1)
    blocks_host = sum(rec.host_total.get(f"layer.{i}.block", 0.0) * 1e3 for i in range(n_layers)) / max(rec.calls.get("model.total", 1), 1)
    final_norm = exact_region_per_token(rec, "model.final_norm")
    lm_head = exact_region_per_token(rec, "model.lm_head")
    known = embed[0] + blocks_gpu + final_norm[0] + lm_head[0]
    known_host = embed[1] + blocks_host + final_norm[1] + lm_head[1]
    row("embedding", *embed)
    row("all Transformer blocks", blocks_gpu, blocks_host)
    row("final RMSNorm", *final_norm)
    row("LM head / vocab projection", *lm_head)
    row("model residual/dispatch", max(0.0, total_gpu - known), max(0.0, total_host - known_host))

    norm1 = sum_region(rec, ".norm1")
    attn = sum_region(rec, ".attention.total")
    norm2 = sum_region(rec, ".norm2")
    moe = sum_region(rec, ".moe.total")
    block_known = norm1[0] + attn[0] + norm2[0] + moe[0]
    print("\nTransformer-block decomposition (summed over all layers):")
    row("RMSNorm 1", norm1[0], norm1[1], blocks_gpu)
    row("sparse attention", attn[0], attn[1], blocks_gpu)
    row("RMSNorm 2", norm2[0], norm2[1], blocks_gpu)
    row("MoE", moe[0], moe[1], blocks_gpu)
    row("residual/casts/other", max(0.0, blocks_gpu - block_known), 0.0, blocks_gpu)

    router = sum_region(rec, ".moe.router")
    experts = sum_region(rec, ".moe.experts")
    print("\nMoE decomposition:")
    row("MoE router/top-k", router[0], router[1], moe[0])
    row("expert computation", experts[0], experts[1], moe[0])
    row("combine/other", max(0.0, moe[0] - router[0] - experts[0]), 0.0, moe[0])

    project = sum_region(rec, ".attention.project_store")
    rope = sum_region(rec, ".attention.rope")
    indices = sum(
        rec.gpu_total_ms[name] for name in rec.calls if ".attention.indices." in name
    ) / max(rec.calls.get("model.total", 1), 1)
    indices_host = sum(
        rec.host_total[name] * 1e3 for name in rec.calls if ".attention.indices." in name
    ) / max(rec.calls.get("model.total", 1), 1)
    route = sum_region(rec, ".attention.router")
    fused_decode = sum_region(rec, ".attention.fused_sparse_decode")
    group_kinds = sorted(
        {
            name.split(".group.", 1)[1]
            for name in rec.calls
            if ".attention.group." in name
        }
    )
    group_rows = []
    for kind in group_kinds:
        gpu = sum(
            rec.gpu_total_ms[name]
            for name in rec.calls
            if name.endswith(f".attention.group.{kind}")
        ) / max(rec.calls.get("model.total", 1), 1)
        host = sum(
            rec.host_total[name] * 1e3
            for name in rec.calls
            if name.endswith(f".attention.group.{kind}")
        ) / max(rec.calls.get("model.total", 1), 1)
        group_rows.append((kind, gpu, host))

    print("\nSparse-attention diagnostics (nested regions; do not sum RoPE twice):")
    row("QKV + RoPE + cache store", project[0], project[1], attn[0])
    if fused_decode_qkv_enabled() and rope[0] == 0.0:
        row("  legacy RoPE subset", rope[0], rope[1], project[0])
    else:
        row("  RoPE subset", rope[0], rope[1], project[0])
    if fused_decode[0] > 0.0 or fused_decode[1] > 0.0:
        row("fused sparse cache read", fused_decode[0], fused_decode[1], attn[0])
    row("index construction", indices, indices_host, attn[0])
    for kind, gpu, host in group_rows:
        row(f"group attention: {kind}", gpu, host, attn[0])
    row("retrieval router", route[0], route[1], attn[0])

    print("\nPer-layer inclusive block time:")
    for i in range(n_layers):
        gpu, host = exact_region_per_token(rec, f"layer.{i}.block")
        print(f"  layer {i:2d}: {gpu:8.3f} ms GPU  {host:8.3f} ms host")



def microbenchmarks(model, config, iterations, warmup):
    print("\n=== Isolated single-token microbenchmarks ===")
    d = config.d_model
    compute_dtype = model.blocks[0].compute_dtype
    hidden = xp.ones((1, d), dtype=compute_dtype)
    residual = xp.ones((1, 1, d), dtype=xp.float32)

    def show(name, fn, iters=iterations):
        gpu, host = timed_call_loop(fn, iters, warmup=warmup)
        print(
            f"  {name:34s} {gpu:9.3f} ms GPU  {host:9.3f} ms host  "
            f"({1000.0/gpu:8.1f}/s GPU)"
        )
        return gpu, host

    full_logits = None
    show("LM head full vocabulary", lambda: model._lm_head_forward(hidden))
    slice_vocab = min(16_384, config.vocab_size)
    lm_slice = model.lm_head[:, :slice_vocab]
    show(f"LM head first {slice_vocab:,}", lambda: hidden @ lm_slice)

    # Materialize representative logits once; sampling helpers intentionally
    # include their current device->host scalar synchronization.
    full_logits = model._lm_head_forward(hidden)[0]
    synchronize()
    show("greedy sampling current", lambda: greedy_decode(full_logits), max(10, iterations // 4))
    show(
        "temperature sampling current",
        lambda: temperature_sample(full_logits, 1.0),
        max(10, iterations // 4),
    )
    show(
        "top-k=20 sampling current",
        lambda: top_k_sample(full_logits, 20, 1.0),
        max(10, iterations // 4),
    )

    block = model.blocks[0]
    branch = xp.ones((1, 1, d), dtype=compute_dtype)
    show("RMSNorm inference (FP32 residual)", lambda: block.norm1.forward(residual))
    show("MoE router only", lambda: block.moe.router.forward(branch))
    weights, expert_indices = block.moe.router.forward(branch)
    synchronize()
    show(
        f"MoE experts current ({config.n_experts} experts)",
        lambda: block.moe.experts.forward(branch, weights, expert_indices),
    )

    # Cost proxy for the desired sparse decode math: two known BF16 experts,
    # without the dynamic-GPU expert-id gather problem.  This is not a proposed
    # implementation; it tells us how much room exists between current all-E
    # FP32 decode and actual top-2 expert arithmetic.
    experts = block.moe.experts.experts
    if len(experts) >= 2:
        x2 = branch.reshape(1, d)

        def two_fixed():
            y0 = experts[0].forward(x2)
            y1 = experts[1].forward(x2)
            return (y0 + y1) * xp.asarray(0.5, dtype=y0.dtype)

        show("two fixed experts BF16 proxy", two_fixed)

    if BACKEND_NAME == "cupy":
        pool = xp.get_default_memory_pool()
        free_b, total_b = xp.cuda.runtime.memGetInfo()
        print(
            f"\nGPU pool after benchmark: used={pool.used_bytes()/2**30:.2f} GiB, "
            f"reserved={pool.total_bytes()/2**30:.2f} GiB, "
            f"device_free={free_b/2**30:.2f}/{total_b/2**30:.2f} GiB"
        )





def print_static_cost_estimate(model, config, max_context):
    """Print decode-relevant weight/cache byte counts from the actual model."""
    lm_bytes = int(getattr(model.lm_head, "nbytes", 0))
    qkv_out_bytes = 0
    moe_decode_fp32_bytes = 0
    moe_native_bytes = 0
    for block in model.blocks:
        attn = block.attention
        for w in (attn.Wqkv, attn.Wo):
            if w is not None:
                qkv_out_bytes += int(w.nbytes)
        experts = block.moe.experts
        for w in (experts.W_gate_stack, experts.W_up_stack, experts.W_down_stack):
            if w is not None:
                moe_native_bytes += int(w.nbytes)
        for w in (
            experts.W_gate_stack_f32, experts.W_up_stack_f32,
            experts.W_down_stack_f32,
        ):
            if w is not None:
                moe_decode_fp32_bytes += int(w.nbytes)

    model_itemsize = int(model.lm_head.dtype.itemsize)
    selected_topk_native = (
        int(config.n_layers) * 3 * int(config.top_k) * int(config.d_model)
        * int(config.d_ff) * model_itemsize
    )
    cache_bytes = (
        int(config.n_layers) * 2 * int(config.n_kv_heads) * int(max_context)
        * int(config.d_head) * model_itemsize
    )
    router_cache_bytes = 0
    if config.attention_layers is not None:
        router_cache_bytes = (
            int(config.n_layers) * int(max_context) * int(config.d_model)
            * model_itemsize
        )

    mib = 2**20
    print("\n=== Static decode cost/traffic proxies ===")
    print(f"LM-head weight matrix:             {lm_bytes/mib:9.2f} MiB")
    print(f"All-layer packed QKV + Wo:         {qkv_out_bytes/mib:9.2f} MiB")
    print(f"Native all-expert weights:         {moe_native_bytes/mib:9.2f} MiB")
    if moe_decode_fp32_bytes:
        print(f"Current BF16 decode FP32 experts:  {moe_decode_fp32_bytes/mib:9.2f} MiB/token weight set")
        print(f"Ideal top-{config.top_k} native expert set:   {selected_topk_native/mib:9.2f} MiB/token weight set")
        if selected_topk_native > 0:
            print(
                "Expert weight-traffic ratio (current/target): "
                f"{moe_decode_fp32_bytes/selected_topk_native:.2f}x"
            )
    print(f"KV cache @ {max_context:,}:               {cache_bytes/mib:9.2f} MiB")
    if router_cache_bytes:
        print(f"Sparse-router input cache:          {router_cache_bytes/mib:9.2f} MiB")
    print(
        "These are byte-count proxies, not measured bandwidth. They expose "
        "where single-token decode repeatedly touches large matrices."
    )


def write_csv(path, sweep_rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)



def main():
    parser = argparse.ArgumentParser(description="Profile cached inference cost by component")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument(
        "--positions", default="128,512,1536,3072",
        help="comma-separated cache positions for uninstrumented decode sweep",
    )
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--profile-position", type=int, default=512)
    parser.add_argument("--profile-steps", type=int, default=24)
    parser.add_argument("--micro-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    positions = parse_positions(args.positions)
    if args.max_context <= 0:
        raise ValueError("--max-context must be positive")

    print("=== Cached inference component benchmark ===")
    print(f"Backend: {BACKEND_NAME}")
    print(f"Checkpoint: {checkpoint}")
    load_start = time.perf_counter()
    config, _, model, loaded_count = load_models(checkpoint)
    synchronize()
    print(
        f"Model: {config.d_model}d, {config.n_layers} layers, "
        f"{config.n_experts} experts/top-{config.top_k}, vocab={config.vocab_size:,}"
    )
    print(
        f"Heads: {config.n_q_heads}Q/{config.n_kv_heads}KV, "
        f"dtype={config.dtype}, loaded arrays={loaded_count}"
    )
    print(f"Model setup: {time.perf_counter()-load_start:.3f}s")
    print(f"KV-cache horizon: {args.max_context:,}")
    print_static_cost_estimate(model, config, args.max_context)

    # Compile/warm the most important kernels once before timing.
    warm_state, _, _, _ = make_state(
        model, config, min(16, args.max_context - 2), args.max_context, args.seed
    )
    model.decode_one(xp.asarray([[1]], dtype=xp.int32), warm_state)
    synchronize()
    if model.sparse_attention:
        ready = sum(
            bool(getattr(block.attention, "_fused_decode_ready", False))
            for block in model.blocks
        )
        mode = "enabled" if fused_sparse_decode_enabled() else "disabled/legacy baseline"
        print(
            f"Fused single-token sparse decode: {mode}; "
            f"{ready}/{len(model.blocks)} layers active"
        )
        failure = fused_sparse_decode_failure_reason()
        if failure:
            print(f"Fused sparse decode rejection: {failure}")

        qkv_ready = sum(
            bool(getattr(block.attention, "_fused_qkv_decode_active", False))
            for block in model.blocks
        )
        qkv_mode = "enabled" if fused_decode_qkv_enabled() else "disabled/legacy baseline"
        print(
            f"Fused decode QKV/RoPE/cache store: {qkv_mode}; "
            f"{qkv_ready}/{len(model.blocks)} layers active"
        )
        qkv_failure = fused_decode_qkv_failure_reason()
        if qkv_failure:
            print(f"Fused QKV/RoPE/cache rejection: {qkv_failure}")

    sweep = throughput_sweep(
        model, config, positions, args.max_context, args.decode_steps,
        args.warmup_steps, args.seed,
    )
    print("\n=== Uninstrumented cached-decode sweep ===")
    print(
        f"{'cache pos':>10s} {'prefill GPU':>12s} {'decode GPU':>12s} "
        f"{'GPU tok/s':>11s} {'host tok/s':>11s}"
    )
    for row in sweep:
        print(
            f"{row['measured_start']:10d} "
            f"{row['prefill_gpu_ms']:10.2f} ms "
            f"{row['decode_gpu_ms']:10.3f} ms "
            f"{row['gpu_tok_s']:11.1f} {row['host_tok_s']:11.1f}"
        )
    write_csv(args.csv_out, sweep)

    realistic_pos = min(positions[0], args.max_context - args.decode_steps)
    greedy = current_greedy_loop(
        model, config, realistic_pos, args.max_context,
        min(args.decode_steps, args.max_context - realistic_pos), args.seed + 77,
    )
    print("\nCurrent Python greedy-loop cost (argmax + host scalar + token allocation + model):")
    print(
        f"  {greedy['gpu_ms']:.3f} ms/token GPU, "
        f"{greedy['host_ms']:.3f} ms/token wall, "
        f"{greedy['host_tok_s']:.1f} effective tok/s"
    )

    rec = instrumented_profile(
        model, config, args.profile_position, args.max_context,
        args.profile_steps, min(2, args.warmup_steps), args.seed + 999,
    )
    print_breakdown(rec, config.n_layers)

    microbenchmarks(model, config, args.micro_iters, warmup=5)

    print("\nInterpretation guide:")
    print("  * Trust the uninstrumented sweep for the real decode baseline.")
    print("  * If LM-head time grows strongly from 16k -> full vocab, optimize vocab projection/sampling.")
    print("  * If MoE experts dominate and the top-2 BF16 proxy is much cheaper, replace all-expert FP32 decode.")
    print("  * If RMSNorm is material, reuse/fuse the existing training BF16 residual->BF16 RMSNorm kernel.")
    print("  * If attention host time is much larger than GPU time, reduce index allocations/kernel launches.")
    print("  * Re-run with --profile-position 1536 or higher to include active learned retrieval routing.")


if __name__ == "__main__":
    main()
