#!/usr/bin/env python3
"""Evaluate a routed checkpoint on the OpenAI-preprocessed LAMBADA test set.

Metrics are tokenizer-native:
  * target-token cross entropy / perplexity on the final whitespace-delimited word;
  * exact target-word sequence accuracy: every tokenizer token covering that final
    word must be the greedy top-1 prediction in sequence.

For a multi-token target this exact criterion is equivalent to greedy generation
matching the complete target token sequence: once the first token differs, the
example is already incorrect.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

from evaluate_wikitext import load_model, load_tokenizer, single_token_nll
from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.inference_model import InferenceModel
from mini_llm.runtime_defaults import apply_runtime_defaults, format_runtime_profile


_LAST_NONSPACE = re.compile(r"\S+$")


def split_lambada_target(tokenizer, text):
    """Return (context_ids, target_ids, target_text) without retokenizing the target.

    We find the last whitespace-delimited word in text, tokenize the *complete*
    string once, then choose the latest tokenizer boundary that does not cross the
    target word's character start. If a BPE token spans the whitespace/word
    boundary, that token correctly remains part of the predicted target suffix.
    """
    text = text.rstrip()
    match = _LAST_NONSPACE.search(text)
    if match is None:
        raise ValueError("LAMBADA sample contains no non-whitespace target")

    full_ids = tokenizer.encode(text)
    if len(full_ids) < 2:
        raise ValueError("LAMBADA sample tokenized to fewer than two tokens")

    boundary = match.start()
    lo, hi = 0, len(full_ids)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        decoded = tokenizer.decode(full_ids[:mid])
        if text.startswith(decoded) and len(decoded) <= boundary:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    context_ids = full_ids[:best]
    target_ids = full_ids[best:]
    if not context_ids or not target_ids:
        raise ValueError("could not create non-empty LAMBADA context/target")
    return context_ids, target_ids, text[match.start():]


def iter_lambada(path: Path, text_field: str):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if text_field not in row:
                raise KeyError(f"{path}:{line_number}: missing field {text_field!r}")
            yield row[text_field]


def _to_int(x):
    if hasattr(x, "get"):
        x = x.get()
    return int(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")

    tokenizer, tokenizer_path = load_tokenizer(args.checkpoint)
    training_model, config = load_model(
        args.checkpoint,
        context_length=args.context_length,
        memory_mode="routed",
    )
    memory_cfg = getattr(config, "memory_context", None)
    if (
        memory_cfg is None
        or not memory_cfg.enabled
        or memory_cfg.integration_mode != "routed_prefix"
    ):
        raise ValueError("LAMBADA evaluator currently expects a routed_prefix checkpoint")

    runtime_profile = apply_runtime_defaults(
        None,
        backend_name=BACKEND_NAME,
        config=config,
        training=False,
        precision=getattr(config, "dtype", None),
    )
    print(format_runtime_profile(runtime_profile))

    model = InferenceModel(config, dtype=config.dtype)
    model.set_weights(training_model)
    horizon = int(memory_cfg.memory_length)

    examples = 0
    exact = 0
    skipped = 0
    target_token_count = 0
    total_target_nll = 0.0
    target_word_nll_sum = 0.0
    target_lengths = []

    synchronize()
    start = time.perf_counter()

    for text in iter_lambada(args.data, args.text_field):
        if args.max_examples is not None and examples >= args.max_examples:
            break
        try:
            context_ids, target_ids, _ = split_lambada_target(tokenizer, text)
        except ValueError:
            skipped += 1
            continue

        if len(context_ids) + len(target_ids) > horizon:
            skipped += 1
            continue

        state = model.create_generation_state(batch_size=1, max_length=horizon)
        context = xp.asarray([context_ids], dtype=xp.int32)
        logits = model.prefill(context, state)

        example_exact = True
        example_nll = 0.0
        for target_index, target_id in enumerate(target_ids):
            row = xp.asarray(logits).reshape(-1, int(logits.shape[-1]))[-1]
            predicted = _to_int(xp.argmax(row))
            if predicted != int(target_id):
                example_exact = False

            nll = single_token_nll(logits, int(target_id))
            example_nll += nll
            total_target_nll += nll
            target_token_count += 1

            if target_index + 1 < len(target_ids):
                actual = xp.asarray([[int(target_id)]], dtype=xp.int32)
                logits = model.decode_one(actual, state)

        examples += 1
        exact += int(example_exact)
        target_word_nll_sum += example_nll
        target_lengths.append(len(target_ids))
        del state, logits, context

        if args.progress_every > 0 and examples % args.progress_every == 0:
            print(
                f"example {examples:5,d} | exact {exact / examples:7.3%} | "
                f"target CE {total_target_nll / target_token_count:.4f}"
            )

    synchronize()
    elapsed = time.perf_counter() - start
    if examples == 0 or target_token_count == 0:
        raise RuntimeError("LAMBADA evaluation produced no scoreable examples")

    token_ce = total_target_nll / target_token_count
    token_ppl = math.exp(token_ce)
    exact_accuracy = exact / examples
    mean_word_nll = target_word_nll_sum / examples
    mean_target_tokens = sum(target_lengths) / len(target_lengths)

    print()
    print("=" * 64)
    print("LAMBADA OpenAI English Benchmark")
    print("=" * 64)
    print(f"Checkpoint:             {args.checkpoint}")
    print(f"Tokenizer:              {tokenizer_path}")
    print(f"Examples scored:        {examples:,}")
    print(f"Examples skipped:       {skipped:,}")
    print(f"Target tokenizer tokens:{target_token_count:9,d}")
    print(f"Mean target tokens/word:{mean_target_tokens:9.3f}")
    print(f"Target-token CE:        {token_ce:9.6f} nats/token")
    print(f"Target-token PPL:       {token_ppl:9.4f}")
    print(f"Mean target-word NLL:   {mean_word_nll:9.6f} nats/word")
    print(f"Exact target accuracy:  {exact_accuracy:9.3%}")
    print(f"Elapsed:                {elapsed:9.2f} s")
    print(f"Examples/s:             {examples / elapsed:9.2f}")


if __name__ == "__main__":
    main()
