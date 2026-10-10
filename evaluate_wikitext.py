#!/usr/bin/env python3
"""Evaluate a Mini-LLM checkpoint on WikiText-2 raw text.

Reports token cross-entropy, perplexity, bits per byte, token/byte ratio, and
throughput. The benchmark is intentionally simple and reproducible: the raw
WikiText-2 test text is concatenated, tokenized with the checkpoint tokenizer,
and evaluated left-to-right in windows no larger than the model context.

Examples:
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/cosmopedia_medium_bf16 \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw

    # Overlapping windows: preserve more context while scoring each token once.
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/cosmopedia_medium_bf16 \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw \
        --stride 256

    # Routed-prefix evaluation: score the same current-window targets while
    # allowing the checkpoint router to select historical blocks.
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw \
        --memory-mode routed --context-length 4096 --stride 4096

    # Article-local routed evaluation (history resets at every article).
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw \
        --memory-mode routed --document-mode article

    # True teacher-forced generation: score one next token, feed the real token,
    # then decode the next token through the production routed KV cache.
    MINI_LLM_BACKEND=cupy python evaluate_wikitext.py \
        --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b \
        --data ./benchmarks/wikitext-2-raw/wiki.test.raw \
        --memory-mode routed --document-mode article --eval-mode autoregressive
"""

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.runtime_defaults import apply_runtime_defaults, format_runtime_profile
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer



_WIKITEXT_TOP_LEVEL_HEADING = re.compile(r"^\s*=\s+[^=\n].*?\s+=\s*$")


def split_wikitext_articles(raw_text: str):
    """Split WikiText raw text at top-level article headings.

    WikiText section headings use multiple equals (for example ``= = Section = =``),
    while a new article starts with one equal sign on each side.  Splitting the
    raw text before tokenization prevents BPE context and routed memory from
    crossing article boundaries.  All source characters are preserved exactly.
    """
    lines = raw_text.splitlines(keepends=True)
    if not lines:
        return []

    articles = []
    prefix = []
    current = []
    seen_heading = False
    for line in lines:
        if _WIKITEXT_TOP_LEVEL_HEADING.match(line):
            if seen_heading:
                if current:
                    articles.append("".join(current))
                current = [line]
            else:
                seen_heading = True
                current = prefix + [line]
                prefix = []
        elif seen_heading:
            current.append(line)
        else:
            prefix.append(line)

    if seen_heading:
        if current:
            articles.append("".join(current))
    elif prefix:
        articles.append("".join(prefix))

    return articles


def tokenize_eval_streams(raw_text, tokenizer, *, document_mode, max_tokens=None):
    """Tokenize either one continuous stream or each WikiText article separately."""
    if document_mode == "continuous":
        documents = [raw_text]
    elif document_mode == "article":
        documents = split_wikitext_articles(raw_text)
    else:
        raise ValueError(f"unknown document mode: {document_mode!r}")

    limit = None if max_tokens is None else max(2, int(max_tokens))
    streams = []
    total = 0
    for document in documents:
        if limit is not None and total >= limit:
            break
        ids = tokenizer.encode(document)
        if limit is not None:
            ids = ids[: max(0, limit - total)]
        if ids:
            streams.append(ids)
            total += len(ids)

    return streams, total, len(documents)


def routed_inference_mode_from_state(state):
    """Classify the prefix used for the next autoregressive prediction."""
    route = getattr(state, "routed_prefix_route", None)
    if route is None:
        return "none"
    selected = getattr(route, "selected_blocks", None)
    if selected is not None and getattr(selected, "ndim", 0) >= 2:
        if int(selected.shape[1]) == 0:
            return "direct_history"
    return "routed_blocks"


def single_token_nll(logits, target_id, *, return_device=False):
    """Stable next-token NLL for inference logits shaped [B,V] or [V]."""
    row = xp.asarray(logits).reshape(-1, int(logits.shape[-1]))[-1]
    work = row.astype(xp.float32, copy=False)
    maximum = xp.max(work)
    log_normalizer = maximum + xp.log(xp.sum(xp.exp(work - maximum)))
    nll = log_normalizer - work[int(target_id)]
    if return_device:
        return nll
    if hasattr(nll, "get"):
        return float(nll.get())
    return float(nll)

def load_tokenizer(checkpoint: Path):
    tokenizer_path = checkpoint / "tokenizer.json"
    if not tokenizer_path.exists():
        fallback = checkpoint.parent / "tokenizer.json"
        if fallback.exists():
            tokenizer_path = fallback
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {checkpoint / 'tokenizer.json'} "
                f"or {fallback}"
            )

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    is_simple = (
        "vocab_size" in tokenizer_config
        and "token_to_id" in tokenizer_config
    )
    tokenizer = (
        SimpleBPETokenizer.load(str(tokenizer_path))
        if is_simple
        else FastBPETokenizer.load(str(tokenizer_path))
    )
    return tokenizer, tokenizer_path


def load_model(checkpoint: Path, *, context_length=None, memory_mode="auto"):
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = ModelConfig(**json.load(f))

    # Evaluation-only architecture overrides. They deliberately do not modify
    # the checkpoint on disk. ``plain`` measures the underlying Transformer;
    # ``routed`` keeps the checkpoint's routed-prefix architecture intact.
    if memory_mode not in {"auto", "plain", "routed"}:
        raise ValueError(f"unknown memory evaluation mode: {memory_mode!r}")

    memory_cfg = getattr(config, "memory_context", None)
    if memory_mode == "plain":
        if memory_cfg is not None:
            config = dataclasses.replace(
                config,
                memory_context=dataclasses.replace(memory_cfg, enabled=False),
            )
        if context_length is not None:
            context_length = int(context_length)
            if context_length <= 0:
                raise ValueError("--context-length must be positive")
            config = dataclasses.replace(config, context_length=context_length)
    elif memory_mode == "routed":
        if (
            memory_cfg is None
            or not memory_cfg.enabled
            or memory_cfg.integration_mode != "routed_prefix"
        ):
            raise ValueError(
                "--memory-mode routed requires an enabled routed_prefix checkpoint"
            )
        # In routed mode the checkpoint's config.context_length is the bounded
        # deep active length (normally 6144), not the continuous target window.
        # A CLI context override therefore specifies/validates the current
        # window only; it must not rewrite the model's active-length config.
        if context_length is not None:
            requested = int(context_length)
            if requested != int(memory_cfg.target_length):
                raise ValueError(
                    "routed evaluation --context-length must equal the checkpoint "
                    f"target/current length {int(memory_cfg.target_length)}, got {requested}"
                )
    elif context_length is not None:
        context_length = int(context_length)
        if context_length <= 0:
            raise ValueError("--context-length must be positive")
        if memory_cfg is not None and memory_cfg.enabled:
            raise ValueError(
                "--context-length with an enabled memory checkpoint is ambiguous; "
                "use --memory-mode plain or --memory-mode routed"
            )
        config = dataclasses.replace(config, context_length=context_length)

    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    names = [p.name for p in model.parameters()]
    loaded, _, _ = load_checkpoint(
        checkpoint,
        param_names=names,
        skip_optimizer=True,
        skip_training=True,
    )

    missing = []
    for p in model.parameters():
        value = loaded.get(p.name)
        if value is None:
            missing.append(p.name)
        else:
            p.data[...] = value

    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} parameter arrays; "
            f"first missing parameter: {missing[0]}"
        )

    return model, config


def resolve_memory_mode(memory_mode: str, disable_memory: bool):
    """Resolve the new explicit mode while preserving the 0065 compatibility flag.

    An explicit ``--memory-mode`` wins over the legacy ``--disable-memory`` flag.
    This intentionally lets an old plain-eval command be converted to routed
    evaluation by appending ``--memory-mode routed``.
    """
    mode = str(memory_mode)
    if mode not in {"auto", "plain", "routed"}:
        raise ValueError(f"unknown memory evaluation mode: {mode!r}")
    overridden_disable = bool(disable_memory and mode == "routed")
    if mode == "auto":
        mode = "plain" if disable_memory else "auto"
    elif mode == "plain":
        mode = "plain"
    return mode, overridden_disable


def build_routed_eval_window(token_ids, *, begin, target_end, memory_cfg):
    """Build one fixed-width routed-prefix source plus the ordinary targets.

    The current window is exactly the same contiguous input used by the plain
    WikiText evaluator. Older tokens are right-aligned in the checkpoint's
    searchable history store and ``history_valid_starts`` marks the real suffix.
    The model then applies its normal direct-history / learned-router policy.
    """
    history_length = int(memory_cfg.distant_memory_length)
    working_length = int(memory_cfg.target_length)
    source_length = int(memory_cfg.source_input_length)
    if source_length != history_length + working_length:
        raise ValueError(
            "routed-prefix source length must equal history + current window"
        )

    seq = token_ids[begin:target_end]
    working = np.asarray(seq[:-1], dtype=np.int64)
    targets = np.asarray(seq[1:], dtype=np.int64)
    if working.size == 0 or working.size > working_length:
        raise ValueError(
            f"invalid routed working length {working.size}; expected 1..{working_length}"
        )

    history_begin = max(0, int(begin) - history_length)
    history = np.asarray(token_ids[history_begin:begin], dtype=np.int64)
    history_valid_start = history_length - int(history.size)

    # Invalid history rows and future current rows may contain any in-vocabulary
    # ID: the former are masked by history_valid_starts and the latter are after
    # every scored position in the causal current window. Zero is convenient.
    source = np.zeros((1, source_length), dtype=np.int64)
    if history.size:
        source[0, history_valid_start:history_length] = history
    source[0, history_length:history_length + working.size] = working

    metadata = {
        "history_valid_starts": np.asarray([history_valid_start], dtype=np.int64),
        "target_valid_lengths": np.asarray([working.size], dtype=np.int64),
        "use_retrieval": True,
        "router_stochastic": False,
        "router_temperature": float(memory_cfg.router_temperature_min),
    }
    return source, targets[None, :], metadata


def routed_prefix_mode_from_cache(cache):
    """Best-effort diagnostic of which top-level historical-prefix path ran."""
    if not isinstance(cache, dict):
        return "unknown"
    memory_cache = cache.get("memory_context_cache")
    if not isinstance(memory_cache, dict):
        return "unknown"
    return str(memory_cache.get("prefix_mode", "unknown"))


def suffix_nll(logits, targets, suffix_start: int):
    """Return summed NLL for targets[suffix_start:] using stable FP32 math."""
    vocab = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab)[suffix_start:]
    flat_targets = targets.reshape(-1)[suffix_start:]
    n = int(flat_targets.size)
    if n == 0:
        return 0.0, 0

    work = flat_logits.astype(xp.float32, copy=True)
    max_logits = xp.max(work, axis=-1, keepdims=True)
    work -= max_logits

    rows = xp.arange(n)
    target_shifted = work[rows, flat_targets]
    xp.exp(work, out=work)
    log_normalizer = xp.log(xp.sum(work, axis=-1))
    nll = xp.sum(log_normalizer - target_shifted)

    if hasattr(nll, "get"):
        nll = float(nll.get())
    else:
        nll = float(nll)
    return nll, n


def run_windowed_evaluation(
    model,
    streams,
    *,
    context,
    stride,
    routed_eval,
    memory_cfg,
    progress_every,
):
    total_nll = 0.0
    total_scored = 0
    windows = 0
    routed_modes = {
        "routed_blocks": 0,
        "direct_history": 0,
        "none": 0,
        "unknown": 0,
    }

    synchronize()
    start_time = time.perf_counter()

    for stream in streams:
        if len(stream) < 2:
            continue
        # Each stream is independent. In article mode this reset is what keeps
        # both the current context and routed history inside one WikiText article.
        for target_start in range(1, len(stream), stride):
            target_end = min(target_start + stride, len(stream))
            begin = max(0, target_end - (context + 1))
            if routed_eval:
                inputs_np, targets_np, memory_metadata = build_routed_eval_window(
                    stream,
                    begin=begin,
                    target_end=target_end,
                    memory_cfg=memory_cfg,
                )
            else:
                seq = stream[begin:target_end]
                inputs_np = np.asarray(seq[:-1], dtype=np.int64)[None, :]
                targets_np = np.asarray(seq[1:], dtype=np.int64)[None, :]
                memory_metadata = None

            inputs = xp.asarray(inputs_np)
            targets = xp.asarray(targets_np)
            suffix_start = target_start - (begin + 1)
            if suffix_start < 0 or suffix_start >= targets_np.shape[1]:
                raise RuntimeError(
                    f"Internal windowing error: suffix_start={suffix_start}, "
                    f"targets={targets_np.shape[1]}"
                )

            if routed_eval:
                logits, cache = model.forward(
                    inputs, memory_metadata=memory_metadata
                )
                logits = logits[:, :targets_np.shape[1], :]
                prefix_mode = routed_prefix_mode_from_cache(cache)
                routed_modes[
                    prefix_mode if prefix_mode in routed_modes else "unknown"
                ] += 1
            else:
                logits, cache = model.forward(inputs)

            nll, count = suffix_nll(logits, targets, suffix_start)
            total_nll += nll
            total_scored += count
            windows += 1
            del logits, cache, inputs, targets

            if progress_every > 0 and windows % progress_every == 0:
                running_loss = total_nll / total_scored
                message = (
                    f"window {windows:5d} | scored {total_scored:9,d} tokens | "
                    f"loss {running_loss:.4f} | ppl {math.exp(running_loss):.2f}"
                )
                if routed_eval:
                    message += (
                        " | memory "
                        f"routed={routed_modes['routed_blocks']} "
                        f"direct={routed_modes['direct_history']} "
                        f"none={routed_modes['none']}"
                    )
                print(message)

    synchronize()
    elapsed = time.perf_counter() - start_time
    return {
        "total_nll": total_nll,
        "total_scored": total_scored,
        "steps": windows,
        "elapsed": elapsed,
        "routed_modes": routed_modes,
        "route_refreshes": None,
    }


def run_autoregressive_evaluation(
    training_model,
    config,
    streams,
    *,
    memory_cfg,
    progress_every,
):
    """Teacher-force WikiText one token at a time through production inference."""
    if (
        memory_cfg is None
        or not memory_cfg.enabled
        or memory_cfg.integration_mode != "routed_prefix"
    ):
        raise ValueError(
            "--eval-mode autoregressive currently requires --memory-mode routed "
            "with an enabled routed_prefix checkpoint"
        )

    runtime_profile = apply_runtime_defaults(
        None,
        backend_name=BACKEND_NAME,
        config=config,
        training=False,
        precision=getattr(config, "dtype", None),
    )
    print(format_runtime_profile(runtime_profile))

    inference_model = InferenceModel(config, dtype=config.dtype)
    inference_model.set_weights(training_model)
    horizon = int(memory_cfg.memory_length)

    total_nll = 0.0
    pending_nll = xp.asarray(0.0, dtype=xp.float64)
    pending_count = 0
    total_scored = 0
    route_refreshes = 0
    routed_modes = {
        "routed_blocks": 0,
        "direct_history": 0,
        "none": 0,
        "unknown": 0,
    }

    def flush_pending():
        nonlocal total_nll, pending_nll, pending_count
        if pending_count <= 0:
            return
        if hasattr(pending_nll, "get"):
            total_nll += float(pending_nll.get())
        else:
            total_nll += float(pending_nll)
        pending_nll = xp.asarray(0.0, dtype=xp.float64)
        pending_count = 0

    synchronize()
    start_time = time.perf_counter()

    for article_index, stream in enumerate(streams, start=1):
        if len(stream) < 2:
            continue

        # A fresh state per stream is the article-boundary reset in article mode.
        # With continuous mode there is only one stream, so the state persists
        # through the complete tokenized test file.
        state = inference_model.create_generation_state(
            batch_size=1, max_length=horizon
        )
        first = xp.asarray([[int(stream[0])]], dtype=xp.int32)
        logits = inference_model.prefill(first, state)

        for target_index in range(1, len(stream)):
            mode = routed_inference_mode_from_state(state)
            routed_modes[mode if mode in routed_modes else "unknown"] += 1

            nll = single_token_nll(
                logits, int(stream[target_index]), return_device=True
            )
            pending_nll = pending_nll + nll.astype(xp.float64, copy=False)
            pending_count += 1
            total_scored += 1

            if progress_every > 0 and total_scored % progress_every == 0:
                flush_pending()
                running_loss = total_nll / total_scored
                print(
                    f"token {total_scored:9,d} | article {article_index:3d} | "
                    f"loss {running_loss:.4f} | ppl {math.exp(running_loss):.2f} | "
                    f"memory routed={routed_modes['routed_blocks']} "
                    f"direct={routed_modes['direct_history']} "
                    f"none={routed_modes['none']}"
                )

            if target_index + 1 < len(stream):
                actual = xp.asarray(
                    [[int(stream[target_index])]], dtype=xp.int32
                )
                logits = inference_model.decode_one(actual, state)

        route_refreshes += int(state.routed_prefix_refresh_count)
        flush_pending()
        del state, logits

    synchronize()
    elapsed = time.perf_counter() - start_time
    return {
        "total_nll": total_nll,
        "total_scored": total_scored,
        "steps": total_scored,
        "elapsed": elapsed,
        "routed_modes": routed_modes,
        "route_refreshes": route_refreshes,
        "refresh_tokens": int(inference_model.routed_prefix_refresh_tokens),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the WikiText-2 raw test set."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to wiki.test.raw")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Number of new target tokens scored per window. Defaults to the "
            "model context length. Only valid for --eval-mode windowed."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Optional evaluation-only current/context length. In routed mode it "
            "must equal the checkpoint target/current length."
        ),
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help=(
            "Compatibility alias for --memory-mode plain. If an explicit "
            "--memory-mode routed is also supplied, routed mode wins."
        ),
    )
    parser.add_argument(
        "--memory-mode",
        choices=["auto", "plain", "routed"],
        default="auto",
        help=(
            "Evaluation memory policy. auto preserves checkpoint behavior, plain "
            "disables hierarchical memory, routed evaluates a routed_prefix "
            "checkpoint with its real searchable history (default: auto)."
        ),
    )
    parser.add_argument(
        "--document-mode",
        choices=["continuous", "article"],
        default="continuous",
        help=(
            "continuous preserves the historical concatenated WikiText stream; "
            "article splits at top-level WikiText headings, tokenizes each article "
            "independently, and resets context/memory at every boundary."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        choices=["windowed", "autoregressive"],
        default="windowed",
        help=(
            "windowed runs the existing batched evaluator. autoregressive scores "
            "one next token and then teacher-forces that real token through the "
            "production KV-cache/routed-memory inference path."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional aggregate tokenizer-token limit for a quick test.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help=(
            "Progress interval. Defaults to 50 windows in windowed mode and "
            "4096 scored tokens in autoregressive mode. Set 0 to disable."
        ),
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"WikiText file not found: {args.data}")
    if args.eval_mode == "autoregressive" and args.stride is not None:
        raise ValueError("--stride is only valid with --eval-mode windowed")

    checkpoint = args.checkpoint
    tokenizer, tokenizer_path = load_tokenizer(checkpoint)
    memory_mode, disable_overridden = resolve_memory_mode(
        args.memory_mode, args.disable_memory
    )
    if disable_overridden:
        print(
            "Note: --memory-mode routed overrides the legacy --disable-memory flag."
        )

    model, config = load_model(
        checkpoint,
        context_length=args.context_length,
        memory_mode=memory_mode,
    )
    memory_cfg = getattr(config, "memory_context", None)
    routed_eval = bool(
        memory_cfg is not None
        and memory_cfg.enabled
        and memory_cfg.integration_mode == "routed_prefix"
        and memory_mode in {"auto", "routed"}
    )
    if args.eval_mode == "autoregressive" and not routed_eval:
        raise ValueError(
            "--eval-mode autoregressive currently requires --memory-mode routed "
            "with a routed_prefix checkpoint"
        )

    context = (
        int(memory_cfg.target_length)
        if routed_eval
        else int(config.context_length)
    )
    stride = context if args.stride is None else int(args.stride)
    if args.eval_mode == "windowed" and not (1 <= stride <= context):
        raise ValueError(f"--stride must be in [1, {context}], got {stride}")

    raw_text = args.data.read_text(encoding="utf-8")
    raw_bytes = len(raw_text.encode("utf-8"))
    print("Tokenizing WikiText-2 test set...")
    streams, tokenizer_tokens, detected_documents = tokenize_eval_streams(
        raw_text,
        tokenizer,
        document_mode=args.document_mode,
        max_tokens=args.max_tokens,
    )
    if sum(max(0, len(stream) - 1) for stream in streams) <= 0:
        raise RuntimeError("WikiText input produced no scoreable next-token pairs")

    progress_every = args.progress_every
    if progress_every is None:
        progress_every = 50 if args.eval_mode == "windowed" else 4096
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    print("=" * 60)
    print("WikiText-2 Benchmark")
    print("=" * 60)
    print(f"Checkpoint:       {checkpoint}")
    print(f"Tokenizer:        {tokenizer_path}")
    print(f"Model dtype:      {config.dtype}")
    print(f"Eval mode:        {args.eval_mode}")
    print(f"Document mode:    {args.document_mode}")
    if args.document_mode == "article":
        print(f"Articles:         {len(streams):,} tokenized / {detected_documents:,} detected")
    print(f"Context length:   {context}")
    if routed_eval:
        print("Memory mode:      routed_prefix")
        print(f"Memory horizon:   {int(memory_cfg.memory_length):,}")
        print(f"Search history:   {int(memory_cfg.distant_memory_length):,}")
        print(f"Routed prefix:    {int(memory_cfg.retrieved_length):,}")
        history_policy = (
            "reset at WikiText article boundaries"
            if args.document_mode == "article"
            else "continuous WikiText token stream"
        )
        print(f"History policy:   {history_policy}")
    elif memory_cfg is not None:
        display_memory_mode = (
            getattr(memory_cfg, "integration_mode", "enabled")
            if memory_cfg.enabled else "disabled"
        )
        print(f"Memory mode:      {display_memory_mode}")
    if args.eval_mode == "windowed":
        print(f"Stride:           {stride}")
    else:
        print("Teacher forcing:  one token at a time")
        print(
            "Route refresh:    "
            f"{int(memory_cfg.inference_route_refresh_tokens):,} tokens checkpoint default; "
            "MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS may override"
        )
    print(f"Raw bytes:        {raw_bytes:,}")
    print(f"Tokenizer tokens: {tokenizer_tokens:,}")
    print()

    if args.eval_mode == "windowed":
        result = run_windowed_evaluation(
            model,
            streams,
            context=context,
            stride=stride,
            routed_eval=routed_eval,
            memory_cfg=memory_cfg,
            progress_every=progress_every,
        )
    else:
        result = run_autoregressive_evaluation(
            model,
            config,
            streams,
            memory_cfg=memory_cfg,
            progress_every=progress_every,
        )

    total_nll = float(result["total_nll"])
    total_scored = int(result["total_scored"])
    if total_scored <= 0:
        raise RuntimeError("evaluation scored zero tokens")
    loss = total_nll / total_scored
    ppl = math.exp(loss)
    elapsed = float(result["elapsed"])

    bpb = None
    if args.max_tokens is None:
        bpb = total_nll / (raw_bytes * math.log(2.0))

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Scored tokens:    {total_scored:,}")
    print(f"Cross-entropy:    {loss:.6f} nats/token")
    print(f"Perplexity:       {ppl:.4f}")
    if bpb is not None:
        print(f"Bits per byte:    {bpb:.6f}")
        print(f"Tokens per byte:  {tokenizer_tokens / raw_bytes:.6f}")
    else:
        print("Bits per byte:    n/a (--max-tokens was used)")

    routed_modes = result["routed_modes"]
    if args.eval_mode == "windowed":
        print(f"Eval windows:     {int(result['steps']):,}")
        if routed_eval:
            print(
                "Memory windows:   "
                f"routed={routed_modes['routed_blocks']}, "
                f"direct={routed_modes['direct_history']}, "
                f"none={routed_modes['none']}, "
                f"unknown={routed_modes['unknown']}"
            )
    else:
        print(f"Inference steps:  {int(result['steps']):,}")
        print(
            "Memory tokens:    "
            f"routed={routed_modes['routed_blocks']}, "
            f"direct={routed_modes['direct_history']}, "
            f"none={routed_modes['none']}, "
            f"unknown={routed_modes['unknown']}"
        )
        print(f"Route refreshes:  {int(result['route_refreshes']):,}")
        print(f"Refresh interval: {int(result['refresh_tokens']):,} tokens")
    print(f"Elapsed:          {elapsed:.2f} s")
    print(f"Scored tok/s:     {total_scored / elapsed:,.1f}")


if __name__ == "__main__":
    main()
