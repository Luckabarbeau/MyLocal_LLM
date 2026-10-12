#!/usr/bin/env python3
"""0078B: fixed-prompt multi-decoder generation and repetition benchmark.

Load a checkpoint once, reuse the production TextGenerator/KV-cache path for
all prompts, and save every result as a line of JSON before proceeding. A run
can be resumed without regenerating already completed (prompt, seed, strategy) cases.

Run from the repository root, for example:

    MINI_LLM_BACKEND=cupy python benchmarks/evaluate_generation.py \\
        --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b_muon_run_baseline \\
        --output-dir ./benchmarks/results/generation_45600

Inspect short samples quickly:

    MINI_LLM_BACKEND=cupy python benchmarks/evaluate_generation.py \\
        --checkpoint ./checkpoints/wide-500m-memory-64k_0060_30b_muon_run_baseline \\
        --output-dir ./benchmarks/results/generation_smoke \\
        --prompt-ids math_general_short,code_python_short --seeds 42 \\
        --max-new-tokens 64

By default each (prompt, seed) is tested with top-k=40 and top-p=1.0, both
at temperature=0.8. An interrupted run resumes with the *same arguments*
and --resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generation_metrics import grouped_summary, repetition_metrics


DEFAULT_SUITE = Path(__file__).with_name("generation_prompts_v1.json")


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must be nonempty with no duplicates")
    return seeds


def parse_strategies(value: str) -> list[str]:
    strategies = [part.strip() for part in value.split(",")]
    allowed = {"greedy", "temperature", "top-k", "top-p"}
    if not strategies or any(name not in allowed for name in strategies):
        raise argparse.ArgumentTypeError(
            "--strategies must be a comma-separated subset of greedy,temperature,top-k,top-p"
        )
    if len(set(strategies)) != len(strategies):
        raise argparse.ArgumentTypeError("--strategies must not contain duplicates")
    return strategies


def result_key(row: dict) -> tuple[str, int, str]:
    """A checkpoint can have multiple decoders per prompt/seed."""
    return (row["prompt"]["id"], row["seed"], row["decoding"]["strategy"])


def select_prompts(suite: dict, ids: str | None, domains: str | None,
                   lengths: str | None) -> list[dict]:
    prompts = suite.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt suite must contain a nonempty 'prompts' list")
    seen = set()
    for item in prompts:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k), str) and item[k] for k in
            ("id", "domain", "topic", "length", "format", "prompt")
        ):
            raise ValueError(f"Invalid prompt entry: {item!r}")
        if item["id"] in seen:
            raise ValueError(f"Duplicate prompt ID: {item['id']}")
        seen.add(item["id"])
    selected = prompts
    for option, field in ((ids, "id"), (domains, "domain"), (lengths, "length")):
        if option:
            allowed = set(option.split(","))
            selected = [item for item in selected if item[field] in allowed]
    if not selected:
        raise ValueError("No prompts selected; check --prompt-ids/--domains/--lengths")
    return selected


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_records(path: Path) -> list[dict]:
    """Reject torn/corrupt JSONL rather than silently losing benchmark data."""
    if not path.exists():
        return []
    output = []
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = result_key(row)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid result at {path}:{line_number}; inspect the JSONL file"
                ) from exc
            if key in keys:
                raise ValueError(f"Duplicate generation in results: {key}")
            keys.add(key)
            output.append(row)
    return output


def write_summaries(output_dir: Path, records: list[dict], expected: int) -> None:
    result = grouped_summary(records)
    result.update({
        "completed": len(records),
        "expected": expected,
        "complete": len(records) == expected,
        "scope": "Exact token-ID repetition in generated text only; prompt and EOS excluded.",
        "limitations": (
            "These automatic metrics do not assess semantic repetition, topic drift, "
            "factuality, story coherence, or code correctness."
        ),
        "metric_definitions": {
            "distinct_n": "Unique n-grams divided by generated n-gram positions.",
            "repeated_ngram_rate": "Fraction of n-gram positions seen previously in the same output.",
            "first_repeated_8gram_end": "1-based end position of first repeated 8-gram, or null.",
            "longest_repeated_span_tokens": "Longest repeated contiguous substring; overlaps allowed.",
            "repeat_8gram_onset_fraction": "Fraction of generations with an exact repeated 8-gram.",
            "mean_repeat_8gram_onset_when_present": "Average onset among generations with repetition only.",
        },
    })
    write_json_atomic(output_dir / "summary.json", result)
    columns = ["group_type", "group", "count", "generated_tokens",
               "mean_distinct_3", "mean_distinct_4", "mean_distinct_5",
               "mean_repeated_5gram_rate", "mean_repeated_8gram_rate",
               "mean_longest_repeated_span_tokens", "repeat_8gram_onset_fraction",
               "mean_repeat_8gram_onset_when_present"]
    csv_path = output_dir / "summary.csv"
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        def add(group_type: str, name: str, stats: dict):
            writer.writerow({"group_type": group_type, "group": name,
                             **{k: v for k, v in stats.items() if k in columns}})
        add("overall", "all", result["overall"])
        for group_type in (
            "by_strategy", "by_strategy_and_length", "by_strategy_and_domain",
            "by_length", "by_domain", "by_domain_and_length",
        ):
            for name, stats in result[group_type].items():
                add(group_type, name, stats)
    tmp.replace(csv_path)


def load_generator(checkpoint: Path, requested_context: int | None,
                   runtime_profile: str | None):
    """Use the same model/tokenizer initialization and TextGenerator as inference.py."""
    from inference import TextGenerator
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.runtime_defaults import apply_runtime_defaults, format_runtime_profile
    from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    config = ModelConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    profile = apply_runtime_defaults(runtime_profile, backend_name=BACKEND_NAME,
                                     config=config, training=False,
                                     precision=getattr(config, "dtype", None))
    print(format_runtime_profile(profile), flush=True)

    memory_cfg = getattr(config, "memory_context", None)
    if requested_context is not None:
        max_context = requested_context
    elif memory_cfg is not None and memory_cfg.enabled and memory_cfg.integration_mode in {
        "terminal_landmark", "routed_prefix"
    }:
        max_context = int(memory_cfg.memory_length)
    else:
        max_context = int(config.context_length)
    if max_context <= 0:
        raise ValueError("--max-context must be positive")
    if (memory_cfg is not None and memory_cfg.enabled and
        memory_cfg.integration_mode in {"terminal_landmark", "routed_prefix"} and
        max_context > int(memory_cfg.memory_length)):
        raise ValueError("Requested context exceeds checkpoint memory horizon")

    tokenizer_path = checkpoint / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    token_json = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if "vocab_size" in token_json and "token_to_id" in token_json:
        tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))
    else:
        tokenizer = FastBPETokenizer.load(str(tokenizer_path))

    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    names = [p.name for p in model.parameters()]
    loaded, _, state = load_checkpoint(checkpoint, param_names=names, skip_optimizer=True)
    for p in model.parameters():
        if p.name in loaded:
            p.data[...] = loaded[p.name]
    if getattr(config, "progressive_depth", False):
        active = int((state or {}).get("progressive_active_layers", config.progressive_initial_layers))
        model.set_active_layers(active)
    step = (state or {}).get("step", None)
    print(f"Loaded {len(loaded)} parameter arrays; step={step}; context={max_context}; "
          f"backend={BACKEND_NAME}; tokenizer={len(tokenizer)}", flush=True)
    return TextGenerator(model, tokenizer, max_context=max_context), {
        "step": step,
        "context": max_context,
        "model_dtype": str(config.dtype),
        "backend": BACKEND_NAME,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "runtime_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("MINI_LLM_")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--prompt-ids", help="Comma-separated IDs; default is all 24")
    parser.add_argument("--domains", help="Comma-separated domains, e.g. coding,story")
    parser.add_argument("--lengths", help="Comma-separated lengths, e.g. short,long")
    parser.add_argument("--seeds", type=parse_seeds, default=None,
                        help="Comma-separated seeds; default from prompt suite")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    decoding = parser.add_mutually_exclusive_group()
    decoding.add_argument("--strategy", choices=("greedy", "temperature", "top-k", "top-p"),
                          help="Single-decoder compatibility mode")
    decoding.add_argument("--strategies", type=parse_strategies,
                          help="Comma-separated decoding strategies (default: top-k,top-p)")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--backend", choices=("kv-cache", "reference"), default="kv-cache")
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--runtime-profile", choices=("auto", "gpu-fast", "consumer-gpu", "reference"))
    parser.add_argument("--resume", action="store_true", help="Continue a matching interrupted benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Show selection without loading model")
    args = parser.parse_args(argv)

    raw_suite = args.prompt_suite.read_bytes()
    suite = json.loads(raw_suite)
    prompts = select_prompts(suite, args.prompt_ids, args.domains, args.lengths)
    seeds = args.seeds if args.seeds is not None else suite.get("recommended_seeds", [42])
    defaults = suite.get("recommended_control", {})
    strategies = args.strategies or ([args.strategy] if args.strategy else ["top-k", "top-p"])
    temperature = args.temperature if args.temperature is not None else defaults.get("temperature", 0.8)
    top_k = args.top_k if args.top_k is not None else defaults.get("top_k", 40)
    top_p = args.top_p if args.top_p is not None else 1.0
    tokens = args.max_new_tokens if args.max_new_tokens is not None else suite.get("recommended_new_tokens", 512)
    if tokens <= 0 or temperature <= 0:
        parser.error("--max-new-tokens and --temperature must be positive")
    if "top-k" in strategies and top_k <= 0:
        parser.error("--top-k must be positive")
    if "top-p" in strategies and not 0 < top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_context is not None and args.max_context <= 0:
        parser.error("--max-context must be positive")

    variants = [
        {"strategy": strategy, "temperature": temperature,
         "top_k": top_k if strategy == "top-k" else None,
         "top_p": top_p if strategy == "top-p" else None}
        for strategy in strategies
    ]
    expected = len(prompts) * len(seeds) * len(variants)
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_suite": str(args.prompt_suite.resolve()),
        "prompt_suite_sha256": hashlib.sha256(raw_suite).hexdigest(),
        "prompt_ids": [p["id"] for p in prompts],
        "seeds": seeds,
        "max_new_tokens": tokens,
        "decoding_variants": variants,
        "backend": args.backend,
        "max_context_request": args.max_context,
        "runtime_profile_request": args.runtime_profile,
    }
    print(f"Generation suite: {len(prompts)} prompts x {len(seeds)} seeds "
          f"x {len(variants)} strategies = {expected} samples", flush=True)
    for variant in variants:
        print(f"  Decoder: {variant}", flush=True)
    print(f"Max new tokens: {tokens}", flush=True)
    if args.dry_run:
        for p in prompts:
            print(f"  {p['id']:27s} {p['domain']:12s} {p['length']:7s} {p['prompt'][:75]!r}")
        return 0

    out = args.output_dir
    manifest_path = out / "manifest.json"
    results_path = out / "generations.jsonl"
    if args.resume:
        if not manifest_path.exists():
            raise ValueError("--resume requires an existing manifest.json")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["configuration"] != config:
            raise ValueError("Resume configuration differs from original run; use a new --output-dir")
    else:
        if manifest_path.exists() or results_path.exists():
            raise ValueError("Output exists. Choose a new directory or pass --resume")
    existing = read_records(results_path) if args.resume else []
    completed = {result_key(row) for row in existing}
    valid_keys = {(p["id"], seed, v["strategy"])
                  for p in prompts for seed in seeds for v in variants}
    if not completed <= valid_keys:
        raise ValueError("Result file contains prompt/seed/strategy cases not present in this run")
    if len(completed) == expected:
        write_summaries(out, existing, expected)
        print("Already complete; summaries refreshed.")
        return 0

    generator, meta = load_generator(args.checkpoint, args.max_context, args.runtime_profile)
    if args.resume:
        previous = old["checkpoint_metadata"]
        if (previous.get("step") != meta["step"] or
            previous.get("model_dtype") != meta["model_dtype"] or
            previous.get("context") != meta["context"]):
            raise ValueError("Checkpoint changed since original run; refusing to mix results")
        if previous.get("runtime_env") != meta["runtime_env"]:
            raise ValueError("Runtime environment changed since original run; use matching flags")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_path, {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config,
            "checkpoint_metadata": meta,
        })
    records = existing[:]
    with results_path.open("a", encoding="utf-8", buffering=1) as handle:
        for prompt in prompts:
            for seed in seeds:
                for variant in variants:
                    strategy = variant["strategy"]
                    key = (prompt["id"], seed, strategy)
                    if key in completed:
                        continue
                    print(f"[{len(records)+1}/{expected}] {prompt['id']} "
                          f"seed={seed} strategy={strategy}", flush=True)
                    generated_text, timing = generator.generate(
                        prompt=prompt["prompt"], max_new_tokens=tokens,
                        strategy=strategy, temperature=variant["temperature"],
                        top_k=variant["top_k"], top_p=variant["top_p"],
                        seed=seed, log_speed=True,
                        use_kv_cache=args.backend == "kv-cache",
                        return_token_ids=True,
                    )
                    raw_ids = timing["generated_token_ids"]
                    eos = timing["eos_token_id"]
                    ended_eos = bool(raw_ids and raw_ids[-1] == eos)
                    token_ids = raw_ids[:-1] if ended_eos else raw_ids
                    measured = repetition_metrics(token_ids)
                    row = {
                        "prompt": prompt,
                        "seed": seed,
                        "decoding": variant,
                        "generated_text": generated_text,
                        "generated_token_ids": token_ids,
                        "stopped_on_eos": ended_eos,
                        "metrics": measured,
                        "timing": {"total_time_s": timing.get("total_time_s"),
                                   "prefill_time_s": timing.get("prefill_time_s"),
                                   "tokens_per_sec": timing.get("tokens_per_sec")},
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    records.append(row)
                    completed.add(key)
                    print(f"    distinct-4={measured['distinct_4']!s} "
                          f"repeat-8={measured['repeated_8gram_rate']!s} "
                          f"onset={measured['first_repeated_8gram_end']} "
                          f"longest={measured['longest_repeated_span_tokens']}", flush=True)
                    # Partial summaries survive interruptions and do not count skipped work.
                    write_summaries(out, records, expected)
    print(f"Completed {len(records)}/{expected}. Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
