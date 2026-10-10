#!/usr/bin/env python3
"""Generic CE/perplexity evaluator reusing the proven WikiText evaluator path.

This intentionally delegates model loading, routed-prefix window construction,
NLL math, and autoregressive teacher forcing to ``evaluate_wikitext.py``. The
only new responsibility here is loading arbitrary text/document corpora.
"""

import argparse
import json
import math
from pathlib import Path

from evaluate_wikitext import (
    load_model,
    load_tokenizer,
    run_autoregressive_evaluation,
    run_windowed_evaluation,
    split_wikitext_articles,
)


def iter_documents(path: Path, data_format: str, text_field: str):
    if data_format == "text":
        yield path.read_text(encoding="utf-8")
        return

    if data_format == "wikitext":
        raw = path.read_text(encoding="utf-8")
        yield from split_wikitext_articles(raw)
        return

    if data_format == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if text_field not in row:
                    raise KeyError(
                        f"{path}:{line_number}: missing text field {text_field!r}"
                    )
                text = row[text_field]
                if not isinstance(text, str):
                    raise TypeError(
                        f"{path}:{line_number}: field {text_field!r} is not a string"
                    )
                yield text
        return

    raise ValueError(f"unknown data format: {data_format!r}")


def tokenize_documents(
    path,
    tokenizer,
    *,
    data_format,
    text_field,
    max_documents=None,
    max_tokens=None,
):
    streams = []
    source_documents = 0
    tokenized_documents = 0
    tokenizer_tokens = 0
    raw_bytes = 0
    limit = None if max_tokens is None else max(2, int(max_tokens))

    for document in iter_documents(path, data_format, text_field):
        if max_documents is not None and source_documents >= int(max_documents):
            break
        source_documents += 1
        raw_bytes += len(document.encode("utf-8"))

        if limit is not None and tokenizer_tokens >= limit:
            break
        ids = tokenizer.encode(document)
        if limit is not None:
            ids = ids[: max(0, limit - tokenizer_tokens)]
        if ids:
            streams.append(ids)
            tokenizer_tokens += len(ids)
            tokenized_documents += 1

    return streams, tokenizer_tokens, source_documents, tokenized_documents, raw_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a generic text/document corpus."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-name", default="External corpus")
    parser.add_argument(
        "--format",
        choices=["text", "wikitext", "jsonl"],
        required=True,
        dest="data_format",
        help=(
            "text = one continuous document; wikitext = reset at top-level "
            "WikiText articles; jsonl = one independent document per record"
        ),
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--memory-mode",
        choices=["plain", "routed"],
        default="plain",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--eval-mode",
        choices=["windowed", "autoregressive"],
        default="windowed",
    )
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if args.max_documents is not None and args.max_documents <= 0:
        raise ValueError("--max-documents must be positive")
    if args.max_tokens is not None and args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
    if args.eval_mode == "autoregressive" and args.stride is not None:
        raise ValueError("--stride is only valid with --eval-mode windowed")

    tokenizer, tokenizer_path = load_tokenizer(args.checkpoint)
    model, config = load_model(
        args.checkpoint,
        context_length=args.context_length,
        memory_mode=args.memory_mode,
    )

    memory_cfg = getattr(config, "memory_context", None)
    routed_eval = bool(
        args.memory_mode == "routed"
        and memory_cfg is not None
        and memory_cfg.enabled
        and memory_cfg.integration_mode == "routed_prefix"
    )
    if args.eval_mode == "autoregressive" and not routed_eval:
        raise ValueError(
            "autoregressive generic evaluation currently requires --memory-mode routed"
        )

    context = (
        int(memory_cfg.target_length) if routed_eval else int(config.context_length)
    )
    stride = context if args.stride is None else int(args.stride)
    if args.eval_mode == "windowed" and not (1 <= stride <= context):
        raise ValueError(f"--stride must be in [1, {context}], got {stride}")

    print(f"Tokenizing {args.dataset_name}...")
    (
        streams,
        tokenizer_tokens,
        source_documents,
        tokenized_documents,
        raw_bytes,
    ) = tokenize_documents(
        args.data,
        tokenizer,
        data_format=args.data_format,
        text_field=args.text_field,
        max_documents=args.max_documents,
        max_tokens=args.max_tokens,
    )
    scoreable = sum(max(0, len(stream) - 1) for stream in streams)
    if scoreable <= 0:
        raise RuntimeError("input produced no scoreable next-token pairs")

    progress_every = args.progress_every
    if progress_every is None:
        progress_every = 50 if args.eval_mode == "windowed" else 4096
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    print("=" * 64)
    print(f"{args.dataset_name} Benchmark")
    print("=" * 64)
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Tokenizer:           {tokenizer_path}")
    print(f"Model dtype:         {config.dtype}")
    print(f"Input format:        {args.data_format}")
    print(f"Documents:           {tokenized_documents:,} tokenized / {source_documents:,} read")
    print(f"Eval mode:           {args.eval_mode}")
    print(f"Context length:      {context:,}")
    print(f"Memory mode:         {'routed_prefix' if routed_eval else 'disabled'}")
    if routed_eval:
        print(f"Memory horizon:      {int(memory_cfg.memory_length):,}")
        print(f"Search history:      {int(memory_cfg.distant_memory_length):,}")
        print(f"Routed prefix:       {int(memory_cfg.retrieved_length):,}")
    if args.eval_mode == "windowed":
        print(f"Stride:              {stride:,}")
    print(f"Text bytes read:     {raw_bytes:,}")
    print(f"Tokenizer tokens:    {tokenizer_tokens:,}")
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
    elapsed = float(result["elapsed"])
    if total_scored <= 0:
        raise RuntimeError("evaluation scored zero tokens")

    loss = total_nll / total_scored
    ppl = math.exp(loss)
    complete = args.max_tokens is None and args.max_documents is None

    print()
    print("=" * 64)
    print("Results")
    print("=" * 64)
    print(f"Scored tokens:       {total_scored:,}")
    print(f"Cross-entropy:       {loss:.6f} nats/token")
    print(f"Perplexity:          {ppl:.4f}")
    if complete and raw_bytes > 0:
        bpb = total_nll / (raw_bytes * math.log(2.0))
        print(f"Bits per byte:       {bpb:.6f}")
        print(f"Tokens per byte:     {tokenizer_tokens / raw_bytes:.6f}")
    else:
        print("Bits per byte:       n/a (subset evaluation)")

    routed_modes = result["routed_modes"]
    if args.eval_mode == "windowed":
        print(f"Eval windows:        {int(result['steps']):,}")
        if routed_eval:
            print(
                "Memory windows:      "
                f"routed={routed_modes['routed_blocks']}, "
                f"direct={routed_modes['direct_history']}, "
                f"none={routed_modes['none']}, "
                f"unknown={routed_modes['unknown']}"
            )
    else:
        print(f"Inference steps:     {int(result['steps']):,}")
        print(
            "Memory tokens:       "
            f"routed={routed_modes['routed_blocks']}, "
            f"direct={routed_modes['direct_history']}, "
            f"none={routed_modes['none']}, "
            f"unknown={routed_modes['unknown']}"
        )
        print(f"Route refreshes:     {int(result['route_refreshes']):,}")
        print(f"Refresh interval:    {int(result['refresh_tokens']):,} tokens")
    print(f"Elapsed:             {elapsed:.2f} s")
    print(f"Scored tok/s:        {total_scored / elapsed:,.1f}")


if __name__ == "__main__":
    main()
