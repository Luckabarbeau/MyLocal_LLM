#!/usr/bin/env python3
"""Read-only diagnostic for independent additive full-history attention heads."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import profile_0087a_training as profile


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--mixed-shard-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--min-history-tokens", type=int, default=8192)
    parser.add_argument("--max-sampling-attempts", type=int, default=10000)
    parser.add_argument("--attention-key-chunk", type=int, default=1024)
    parser.add_argument("--lm-head-chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=78078)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def _sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sample_validation_batches(args, config, sources):
    from mini_llm.data.token_shards import (
        create_hierarchical_memory_minibatch,
        load_or_build_packed_document_index,
        map_token_shard,
    )

    eligible = [source for source in sources if source.val_paths]
    if not eligible:
        raise ValueError("mixture has no validation shards")
    weights = np.asarray([source.weight for source in eligible], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(args.seed)
    memory_length = int(config.memory_context.distant_memory_length)
    target_length = int(config.memory_context.target_length)
    cache = {}
    batches = []

    for attempt in range(1, args.max_sampling_attempts + 1):
        source = eligible[int(rng.choice(len(eligible), p=weights))]
        path = source.val_paths[int(rng.integers(0, len(source.val_paths)))]
        key = str(path)
        if key not in cache:
            tokens = map_token_shard(key)
            index = load_or_build_packed_document_index(
                path, tokens, 1, write_sidecar=False
            )
            cache[key] = (tokens, index)
        tokens, document_index = cache[key]
        inputs, targets, metadata = create_hierarchical_memory_minibatch(
            tokens,
            1,
            memory_length,
            target_length,
            rng=rng,
            eos_token_id=1,
            document_index=document_index,
            document_aware=True,
            return_metadata=True,
            min_history_tokens=0,
            sampling_policy="document_first",
        )
        history_tokens = memory_length - int(metadata["history_valid_starts"][0])
        scored_tokens = int(metadata["target_loss_mask"].sum())
        if history_tokens < args.min_history_tokens or scored_tokens != target_length:
            continue
        loss_mask = np.asarray(metadata["target_loss_mask"], dtype=np.float32)
        fingerprint = _sha256_arrays(
            inputs,
            targets,
            loss_mask,
            metadata["history_valid_starts"],
            metadata["target_valid_lengths"],
        )
        batches.append({
            "inputs": np.asarray(inputs),
            "targets": np.asarray(targets),
            "loss_mask": loss_mask,
            "metadata": metadata,
            "source": source.name,
            "shard": key,
            "history_tokens": history_tokens,
            "scored_tokens": scored_tokens,
            "document_span": int(
                metadata["document_ends"][0] - metadata["document_starts"][0]
            ),
            "target_source_start": int(metadata["target_source_starts"][0]),
            "fingerprint": fingerprint,
            "sampling_attempt": attempt,
        })
        if len(batches) == args.samples:
            return batches

    raise RuntimeError(
        f"found only {len(batches)} validation samples with a full {target_length}-token "
        f"target and >= {args.min_history_tokens} history tokens after "
        f"{args.max_sampling_attempts} attempts"
    )


def _sumsq_and_count(cp, array, valid_mask):
    value = array.astype(cp.float32, copy=False)
    if valid_mask is None:
        return float(cp.sum(value * value).item()), int(value.size)
    valid = cp.asarray(valid_mask, dtype=cp.bool_)
    return (
        float(cp.sum((value * value) * valid[..., None]).item()),
        int(valid.sum().item()) * int(value.shape[-1]),
    )


def _attention_history_fraction(cp, cache, key_chunk):
    q = cache.query.astype(cp.float32, copy=False)
    k = cache.key
    lse = cache.lse_log2.astype(cp.float32, copy=False)
    query_length = int(q.shape[0])
    history_length = int(cache.history_length)
    key_length = int(k.shape[0])
    history_mass = cp.zeros((query_length,), dtype=cp.float32)
    current_mass = cp.zeros_like(history_mass)
    query_indices = cp.arange(query_length, dtype=cp.int64)[:, None]
    scale_log2 = (int(q.shape[1]) ** -0.5) * math.log2(math.e)

    for start in range(0, key_length, key_chunk):
        stop = min(start + key_chunk, key_length)
        key_indices = cp.arange(start, stop, dtype=cp.int64)
        scores_log2 = (q @ k[start:stop].astype(cp.float32, copy=False).T) * scale_log2
        visible = ((key_indices[None, :] < history_length) |
                   (key_indices[None, :] <= history_length + query_indices))
        mass = cp.where(
            visible,
            cp.exp2(cp.clip(scores_log2 - lse[:, None], -126.0, 30.0)),
            0.0,
        )
        history_keys = key_indices < history_length
        if bool(cp.any(history_keys).item()):
            history_mass += cp.sum(mass[:, history_keys], axis=1)
        if bool(cp.any(~history_keys).item()):
            current_mass += cp.sum(mass[:, ~history_keys], axis=1)
        del scores_log2, visible, mass

    denominator = history_mass + current_mass
    valid = denominator > 0
    fractions = cp.where(valid, history_mass / cp.maximum(denominator, 1.0e-30), 0.0)
    return float(cp.sum(fractions).item()), int(valid.sum().item())


def _empty_history(cp, state):
    return SimpleNamespace(
        ids=state.ids,
        positions=state.positions,
        valid=cp.zeros_like(state.valid, dtype=cp.bool_),
    )


@contextlib.contextmanager
def _capture_head_statistics(model, cp, records, key_chunk):
    originals = []
    for layer in model.config.full_history_layers:
        block = model.blocks[layer]
        head = block.full_history_head
        if head is None:
            raise RuntimeError(f"configured full-history layer {layer} has no attached head")
        state = {}
        original_block = block.forward
        original_head = head.forward
        record = records[layer]

        def block_forward(x, *positional, _original=original_block, _state=state,
                          **keywords):
            mask = keywords.get("full_history_current_valid_mask")
            _state["residual"] = _sumsq_and_count(cp, x, mask)
            return _original(x, *positional, **keywords)

        def head_forward(current, historical_state, positions, *positional,
                         _original=original_head, _state=state, _record=record,
                         _block=block, **keywords):
            requested_cache = bool(keywords.get("return_cache", True))
            keywords["return_cache"] = True
            output, cache = _original(
                current, historical_state, positions, *positional, **keywords
            )
            output_sumsq, output_count = _sumsq_and_count(
                cp, output, keywords.get("current_valid_mask")
            )
            gate = 1.0
            if getattr(_block, "residual_gates", False):
                gate = float(cp.asarray(_block.alpha_attn.data).item())
            residual_sumsq, residual_count = _state["residual"]
            _record["calls"] += 1
            _record["output_sumsq"] += output_sumsq * gate * gate
            _record["output_count"] += output_count
            _record["residual_sumsq"] += residual_sumsq
            _record["residual_count"] += residual_count
            for row in cache[1]:
                attention_cache = row[6]
                if attention_cache is None:
                    continue
                fraction_sum, query_count = _attention_history_fraction(
                    cp, attention_cache, key_chunk
                )
                _record["history_fraction_sum"] += fraction_sum
                _record["attention_queries"] += query_count
            return output, (cache if requested_cache else None)

        block.forward = block_forward
        head.forward = head_forward
        originals.append((block, original_block, head, original_head))
    try:
        yield
    finally:
        for block, original_block, head, original_head in originals:
            block.forward = original_block
            head.forward = original_head


@contextlib.contextmanager
def _remove_head_history(model, cp):
    originals = []
    for layer in model.config.full_history_layers:
        head = model.blocks[layer].full_history_head
        original = head.forward

        def forward(current, historical_state, positions, *positional,
                    _original=original, **keywords):
            empty = None if historical_state is None else _empty_history(cp, historical_state)
            return _original(current, empty, positions, *positional, **keywords)

        head.forward = forward
        originals.append((head, original))
    try:
        yield
    finally:
        for head, original in originals:
            head.forward = original


@contextlib.contextmanager
def _disable_heads(model, layers):
    saved = {}
    for layer in layers:
        saved[layer] = model.blocks[layer].full_history_head
        model.blocks[layer].full_history_head = None
    try:
        yield
    finally:
        for layer, head in saved.items():
            model.blocks[layer].full_history_head = head


def _score_condition(model, cp, batches, lm_head_chunk):
    weighted_loss = 0.0
    total_tokens = 0
    sample_losses = []
    for batch in batches:
        head_input, target_slice = model.forward_body(
            batch["inputs"],
            return_cache=False,
            return_target_slice=True,
            memory_metadata=batch["metadata"],
        )
        loss, loss_cache = model.chunked_lm_head_loss_forward(
            head_input,
            batch["targets"],
            chunk_tokens=lm_head_chunk,
            loss_mask=batch["loss_mask"],
            target_slice=target_slice,
        )
        count = int(batch["scored_tokens"])
        sample_losses.append(float(loss))
        weighted_loss += float(loss) * count
        total_tokens += count
        del head_input, loss_cache
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "ce": weighted_loss / total_tokens,
        "scored_tokens": total_tokens,
        "sample_ce": sample_losses,
    }


def _parameter_statistics(model, cp, checkpoint, initial_checkpoint):
    from mini_llm.checkpoint import load_checkpoint

    head_names = []
    parameters = {}
    for layer in model.config.full_history_layers:
        head = model.blocks[layer].full_history_head
        for projection, parameter in (("q", head.q), ("k", head.k),
                                      ("v", head.v), ("o", head.o)):
            head_names.append(parameter.name)
            parameters[(layer, projection)] = parameter

    initial = None
    if initial_checkpoint is not None:
        initial, _, _ = load_checkpoint(
            initial_checkpoint,
            param_names=head_names,
            skip_optimizer=True,
            skip_training=True,
        )
        missing = sorted(set(head_names) - set(initial))
        if missing:
            raise ValueError(f"initial checkpoint is missing head tensors: {missing[:4]}")

    result = {}
    for (layer, projection), parameter in parameters.items():
        data32 = parameter.data.astype(cp.float32, copy=False)
        row = {
            "weight_l2": float(cp.linalg.norm(data32).item()),
            "weight_rms": float(cp.sqrt(cp.mean(data32 * data32)).item()),
            "nonzero_fraction": float(cp.mean(data32 != 0).item()),
            "gradient_l2": None,
        }
        if initial is not None:
            difference = data32 - initial[parameter.name].astype(cp.float32, copy=False)
            row["delta_from_initial_l2"] = float(cp.linalg.norm(difference).item())
        momentum_path = (
            checkpoint / "optimizer_state" / "muon_m" / f"{parameter.name}.npy"
        )
        row["muon_momentum_l2"] = (
            float(np.linalg.norm(np.load(momentum_path).astype(np.float64).reshape(-1)))
            if momentum_path.is_file() else None
        )
        result.setdefault(layer, {})[projection] = row
    if initial is not None:
        del initial
        cp.get_default_memory_pool().free_all_blocks()
    return result


def _print_report(report):
    print("\nValidation samples")
    print("idx  source              history  scored  document  fingerprint")
    for index, sample in enumerate(report["samples"]):
        print(f"{index:>3}  {sample['source']:<18} {sample['history_tokens']:>7} "
              f"{sample['scored_tokens']:>7} {sample['document_span']:>9}  "
              f"{sample['fingerprint'][:16]}")

    print("\nPer-head behavior")
    print("layer  calls  output_rms  residual_rms  ratio_%  history_attn_%  current_attn_%")
    for row in report["per_head"]:
        print(f"{row['layer']:>5} {row['calls']:>6} {row['output_rms']:>11.6f} "
              f"{row['residual_rms']:>13.6f} {row['output_to_residual_percent']:>8.4f} "
              f"{row['history_attention_percent']:>15.3f} "
              f"{row['current_attention_percent']:>15.3f}")

    print("\nProjection L2 norms (and delta from migrated initialization)")
    print("layer    Q        K        V        O      dQ       dK       dV       dO")
    for row in report["per_head"]:
        stats = row["projections"]
        values = [stats[key]["weight_l2"] for key in ("q", "k", "v", "o")]
        deltas = [stats[key].get("delta_from_initial_l2") for key in ("q", "k", "v", "o")]
        delta_text = ["n/a" if value is None else f"{value:.4f}" for value in deltas]
        print(f"{row['layer']:>5} " + " ".join(f"{value:>8.4f}" for value in values) +
              " " + " ".join(f"{value:>8}" for value in delta_text))
    print("Gradient norms: unavailable (gradients are not serialized in checkpoints).")

    enabled_ce = report["ablations"]["all_enabled"]["ce"]
    print("\nAblation CE")
    print("condition                     CE         delta_vs_enabled")
    for name, row in report["ablations"].items():
        print(f"{name:<28} {row['ce']:>10.6f} {row['ce'] - enabled_ce:>+17.6f}")


def main():
    args = _parse_args()
    if args.samples < 1 or args.min_history_tokens <= 4096:
        raise ValueError("samples must be positive and min-history-tokens must exceed 4096")
    if args.attention_key_chunk < 1 or args.lm_head_chunk < 1:
        raise ValueError("chunk sizes must be positive")

    for name, value in profile._PRODUCTION_BENCH_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_BACKEND", "triton")
    os.environ.setdefault("MINI_LLM_TRAIN_ATTN_STRICT", "1")
    os.environ.setdefault("MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD", "1")
    os.environ.setdefault("MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM", "1")

    import cupy as cp
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy":
        raise RuntimeError("full-history diagnostics require MINI_LLM_BACKEND=cupy")

    checkpoint = args.checkpoint.expanduser().resolve()
    initial_checkpoint = (
        args.initial_checkpoint.expanduser().resolve()
        if args.initial_checkpoint is not None else None
    )
    mixed_root = args.mixed_shard_root.expanduser().resolve()
    config = ModelConfig(**json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    ))
    layers = tuple(config.full_history_layers)
    if len(layers) != 8:
        raise ValueError(f"expected eight full-history layers, found {layers}")
    apply_runtime_defaults(
        "gpu-fast", backend_name=BACKEND_NAME, config=config,
        training=True, precision="bf16-mixed",
    )

    sources, _ = load_weighted_shard_sources(mixed_root)
    batches = _sample_validation_batches(args, config, sources)
    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    parameters = model.parameters()
    parameter_names = [parameter.name for parameter in parameters]
    loaded, _, checkpoint_state = load_checkpoint(
        checkpoint,
        param_names=parameter_names,
        skip_optimizer=True,
    )
    missing = sorted(set(parameter_names) - set(loaded))
    if missing:
        raise ValueError(f"checkpoint is missing model tensors: {missing[:8]}")
    expected_head_names = {
        f"blocks.{layer}.full_history.W{projection}"
        for layer in layers for projection in ("q", "k", "v", "o")
    }
    if not expected_head_names <= set(loaded):
        raise ValueError("not all 32 full-history projection tensors were loaded")
    for parameter in parameters:
        parameter.data[...] = loaded.pop(parameter.name)
    if loaded:
        raise ValueError(f"unexpected loaded tensors remain: {sorted(loaded)[:8]}")
    model.refresh_compute_buffers()

    projection_stats = _parameter_statistics(
        model, cp, checkpoint, initial_checkpoint
    )
    capture_records = {
        layer: {
            "calls": 0,
            "output_sumsq": 0.0,
            "output_count": 0,
            "residual_sumsq": 0.0,
            "residual_count": 0,
            "history_fraction_sum": 0.0,
            "attention_queries": 0,
        }
        for layer in layers
    }

    ablations = {}
    with _capture_head_statistics(
        model, cp, capture_records, args.attention_key_chunk
    ):
        ablations["all_enabled"] = _score_condition(
            model, cp, batches, args.lm_head_chunk
        )
    with _remove_head_history(model, cp):
        ablations["heads_current_only"] = _score_condition(
            model, cp, batches, args.lm_head_chunk
        )
    with _disable_heads(model, layers):
        ablations["all_heads_disabled"] = _score_condition(
            model, cp, batches, args.lm_head_chunk
        )
    for layer in layers:
        with _disable_heads(model, (layer,)):
            ablations[f"layer_{layer + 1}_disabled"] = _score_condition(
                model, cp, batches, args.lm_head_chunk
            )

    per_head = []
    for layer in layers:
        record = capture_records[layer]
        if record["calls"] != len(batches):
            raise RuntimeError(
                f"layer {layer} executed {record['calls']} times; expected {len(batches)}"
            )
        output_rms = math.sqrt(record["output_sumsq"] / record["output_count"])
        residual_rms = math.sqrt(record["residual_sumsq"] / record["residual_count"])
        history_fraction = (
            record["history_fraction_sum"] / record["attention_queries"]
        )
        per_head.append({
            "layer": layer + 1,
            "internal_layer": layer,
            "calls": record["calls"],
            "output_rms": output_rms,
            "residual_rms": residual_rms,
            "output_to_residual_percent": 100.0 * output_rms / residual_rms,
            "history_attention_percent": 100.0 * history_fraction,
            "current_attention_percent": 100.0 * (1.0 - history_fraction),
            "attention_queries": record["attention_queries"],
            "projections": projection_stats[layer],
        })

    sample_report = [
        {key: batch[key] for key in (
            "source", "shard", "history_tokens", "scored_tokens",
            "document_span", "target_source_start", "fingerprint",
            "sampling_attempt",
        )}
        for batch in batches
    ]
    report = {
        "checkpoint": str(checkpoint),
        "initial_checkpoint": (
            None if initial_checkpoint is None else str(initial_checkpoint)
        ),
        "step": int(checkpoint_state.get("step", -1)),
        "full_history_layers_internal": list(layers),
        "full_history_layers_user_facing": [layer + 1 for layer in layers],
        "strict_model_tensor_count": len(parameter_names),
        "full_history_tensor_count": len(expected_head_names),
        "gradient_norms": {
            "available": False,
            "reason": "parameter gradients are not serialized in the checkpoint; runtime buffers start at zero",
        },
        "samples": sample_report,
        "per_head": per_head,
        "ablations": ablations,
    }
    _print_report(report)
    if args.out is not None:
        output = args.out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
