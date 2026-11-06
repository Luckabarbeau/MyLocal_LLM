"""Fixed-shape end-to-end PyTorch benchmark for sparse-retrieval Model B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import statistics
import time

import numpy as np


def _load_eos_token_id(tokenizer_path: Path) -> int:
    from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer

    try:
        tokenizer = FastBPETokenizer.load(str(tokenizer_path))
    except Exception:
        tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))
    token = getattr(tokenizer, "eos_token", None)
    mapping = getattr(tokenizer, "token_to_id", None)
    eos = mapping.get(token) if isinstance(mapping, dict) and token is not None else None
    native = getattr(tokenizer, "_tokenizer", None)
    if eos is None and native is not None and token is not None:
        eos = native.token_to_id(token)
    if eos is None:
        raise ValueError(f"tokenizer has no EOS token ID: {tokenizer_path}")
    return int(eos)


def _find_fixed_document_window(
    shard_root: Path, *, eos_token_id: int, context_length: int,
) -> tuple[Path, int, np.ndarray, np.ndarray]:
    from mini_llm.data.token_shards import build_packed_document_index, map_token_shard

    required = int(context_length) + 1
    for shard_path in sorted(shard_root.rglob("train_shard_*.bin")):
        shard = map_token_shard(str(shard_path), context_length)
        if shard.ndim != 1:
            continue
        index = build_packed_document_index(shard, eos_token_id)
        previous = 0
        for end_value in index.ends:
            end = int(end_value)
            if end - previous >= required:
                window = np.asarray(shard[previous:previous + required], dtype=np.int64)
                return shard_path, previous, window[:-1].copy(), window[1:].copy()
            previous = end
    raise ValueError(
        f"no training document under {shard_root} contains {required} tokens"
    )


def _synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_stage(torch, operation):
    _synchronize(torch)
    started = time.perf_counter()
    value = operation()
    _synchronize(torch)
    return value, time.perf_counter() - started


def _training_microbatch(torch, trainer, batch, supervised_tokens: float) -> dict:
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    (loss_sum, count, executed), forward_s = _time_stage(
        torch, lambda: trainer._loss_sum(batch)
    )
    _, backward_s = _time_stage(torch, loss_sum.backward)

    def normalize_and_clip():
        for parameter in trainer.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(supervised_tokens)
        return torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), trainer.max_grad_norm,
            error_if_nonfinite=True,
        )

    grad_norm, clip_s = _time_stage(torch, normalize_and_clip)
    _, optimizer_s = _time_stage(torch, trainer.optimizer.step)
    trainer.optimizer.zero_grad(set_to_none=True)
    total_s = forward_s + backward_s + clip_s + optimizer_s
    count_value = float(count.item())
    if count_value != supervised_tokens:
        raise RuntimeError(
            f"fixed benchmark expected {supervised_tokens} supervised tokens, got {count_value}"
        )
    return {
        "loss": float(loss_sum.item() / count_value),
        "grad_norm": float(grad_norm.item()),
        "executed_tokens": int(executed),
        "supervised_tokens": count_value,
        "forward_s": forward_s,
        "backward_s": backward_s,
        "clip_s": clip_s,
        "optimizer_s": optimizer_s,
        "microbatch_s": forward_s + backward_s,
        "complete_step_s": total_s,
        "executed_tokens_per_s": float(executed) / total_s,
        "supervised_tokens_per_s": count_value / total_s,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-root", type=Path,
        default=Path("token_shards_ensemble_32k_mixed"),
    )
    parser.add_argument(
        "--tokenizer-path", type=Path,
        default=Path("token_shards_ensemble_32k_mixed/tokenizer.json"),
    )
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--lm-head-chunk-tokens", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument(
        "--optimizer-offload", choices=("none", "moments", "full"), default="full",
    )
    parser.add_argument(
        "--activation-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--checkpointed-layers", type=int,
        help="checkpoint this many leading layers; defaults to all enabled layers",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.context_length <= 0 or args.lm_head_chunk_tokens <= 0:
        raise ValueError("context and LM-head chunk lengths must be positive")
    if args.warmup_steps < 1 or args.timed_steps < 1:
        raise ValueError("at least one warm-up and timed step are required")
    if not args.shard_root.is_dir():
        raise FileNotFoundError(args.shard_root)
    if not args.tokenizer_path.is_file():
        raise FileNotFoundError(args.tokenizer_path)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the fixed Model B benchmark requires CUDA")
    # FlexAttention is compiled with dynamic dimensions (local and summary lengths).
    # Dynamic CUDA-graph replay currently violates Inductor input-lifetime invariants
    # after an optimizer step, so retain compiled kernels but skip graph capture.
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    from mini_llm.backends.pytorch.model import TorchDecoderLanguageModel
    from mini_llm.backends.pytorch.training import (
        TorchTrainer,
        prepare_dual_resolution_metadata,
    )
    from mini_llm.model_registry import registry

    torch.manual_seed(42)
    device = torch.device("cuda")
    eos_token_id = _load_eos_token_id(args.tokenizer_path)
    shard_path, source_start, inputs, targets = _find_fixed_document_window(
        args.shard_root,
        eos_token_id=eos_token_id,
        context_length=args.context_length,
    )
    config = registry.config("model_b_348m_sparse16")
    dual_metadata = prepare_dual_resolution_metadata(
        inputs[None, :], np.asarray([args.context_length]),
        config.dual_resolution.block_size,
    )
    batch = {
        "input_ids": inputs[None, :],
        "targets": targets[None, :],
        "loss_mask": np.ones((1, args.context_length), dtype=np.float32),
        "dual_resolution_metadata": dual_metadata,
    }

    model = TorchDecoderLanguageModel(
        config, dtype=torch.bfloat16, device=device,
        activation_checkpointing=args.activation_checkpointing,
        checkpointed_layers=args.checkpointed_layers,
    )
    trainer = TorchTrainer(
        model,
        lr=1.5e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_matmul_precision="bf16",
        grad_accum_steps=1,
        max_grad_norm=1.0,
        optimizer_offload=args.optimizer_offload,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
    )

    print(
        f"model=model_b_348m_sparse16 parameters={sum(p.numel() for p in model.parameters()):,} "
        f"device={torch.cuda.get_device_name()} torch={torch.__version__} dtype=bfloat16"
    )
    print(
        f"shard_root={args.shard_root} tokenizer={args.tokenizer_path} "
        f"eos_token_id={eos_token_id} source_shard={shard_path} "
        f"source_start={source_start} context={args.context_length} "
        f"sparse_stride={config.dual_resolution.sparse_retrieval_stride} "
        f"lm_head_chunk={args.lm_head_chunk_tokens} "
        f"optimizer_offload={args.optimizer_offload} "
        f"checkpointed_layers={model.checkpointed_layers}"
    )

    _synchronize(torch)
    warmup_started = time.perf_counter()
    for index in range(args.warmup_steps):
        row = _training_microbatch(
            torch, trainer, batch, float(args.context_length)
        )
        print(
            f"warmup={index + 1} loss={row['loss']:.6f} "
            f"time_s={row['complete_step_s']:.3f} "
            f"peak_vram_gib={row['peak_vram_bytes'] / 2**30:.3f}"
        )
    warmup_s = time.perf_counter() - warmup_started

    rows = []
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
    for index in range(args.timed_steps):
        if profiler is not None and index == 0:
            with profiler:
                row = _training_microbatch(
                    torch, trainer, batch, float(args.context_length)
                )
        else:
            row = _training_microbatch(
                torch, trainer, batch, float(args.context_length)
            )
        rows.append(row)
        print(
            f"timed={index + 1} loss={row['loss']:.6f} "
            f"forward_s={row['forward_s']:.3f} backward_s={row['backward_s']:.3f} "
            f"clip_s={row['clip_s']:.3f} optimizer_s={row['optimizer_s']:.3f} "
            f"complete_s={row['complete_step_s']:.3f} "
            f"supervised_tok_s={row['supervised_tokens_per_s']:.1f} "
            f"peak_vram_gib={row['peak_vram_bytes'] / 2**30:.3f}"
        )

    if profiler is not None:
        print(profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40,
        ))

    summary = {
        "schema": "pytorch-model-b-fixed-benchmark-v1",
        "model": "model_b_348m_sparse16",
        "backend": "pytorch",
        "attention": "torch.compile(max-autotune,dynamic=True) FlexAttention",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_length": args.context_length,
        "sparse_retrieval_stride": config.dual_resolution.sparse_retrieval_stride,
        "optimizer_offload": args.optimizer_offload,
        "activation_checkpointing": args.activation_checkpointing,
        "checkpointed_layers": model.checkpointed_layers,
        "lm_head_chunk_tokens": args.lm_head_chunk_tokens,
        "shard_root": str(args.shard_root),
        "tokenizer_path": str(args.tokenizer_path),
        "source_shard": str(shard_path),
        "source_start": source_start,
        "warmup_steps": args.warmup_steps,
        "warmup_s": warmup_s,
        "timed_steps": args.timed_steps,
        "mean_forward_s": _mean(rows, "forward_s"),
        "mean_backward_s": _mean(rows, "backward_s"),
        "mean_clip_s": _mean(rows, "clip_s"),
        "mean_optimizer_s": _mean(rows, "optimizer_s"),
        "mean_complete_step_s": _mean(rows, "complete_step_s"),
        "mean_executed_tokens_per_s": _mean(rows, "executed_tokens_per_s"),
        "mean_supervised_tokens_per_s": _mean(rows, "supervised_tokens_per_s"),
        "peak_vram_bytes": max(int(row["peak_vram_bytes"]) for row in rows),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "rows": rows,
    }
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
