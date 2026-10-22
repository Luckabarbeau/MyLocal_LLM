#!/usr/bin/env python3
"""WikiText-2 raw-text evaluation for an external GGUF Llama checkpoint.

Reuse the native benchmark's tokenization-stream and sliding-window NLL
implementation, so the native and external results have the same window/stride
and document-boundary semantics. Tokens are ALWAYS obtained from the GGUF's
own embedded tokenizer; model weights are executed through MyLocal_LLM.

Example (repository root):
  MINI_LLM_BACKEND=cupy python -m benchmarks.evaluate_external_wikitext \\
      --model ../external_models/smollm2-360m-instruct-q8_0.gguf \\
      --data ../benchmarks/wikitext-2-raw/wiki.test.raw \\
      --context-length 256 --stride 256 --max-tokens 2048
"""

import argparse
import json
import math
from pathlib import Path

from mini_llm.backend import BACKEND_NAME
from benchmarks.evaluate_wikitext import (
    run_windowed_evaluation,
    tokenize_eval_streams,
)


class _GGUFTokenIDs:
    """Adapt tokenizers.Encoding to the native evaluator's list-of-IDs API."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        # Text-completion benchmark: no ChatML framing or appended EOS tokens.
        # The external tokenizer's add_special_tokens=False is deliberate.
        ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
        spec = self.tokenizer.spec
        if spec.add_bos:
            if spec.bos_id is None:
                raise ValueError("GGUF requests an automatic BOS but supplies no BOS token ID")
            ids.insert(0, spec.bos_id)
        return ids


class _ExternalWindowModel:
    """Return the two-item forward signature used by the native WikiText scorer."""

    def __init__(self, external_model):
        self.external_model = external_model

    def forward(self, token_ids):
        return self.external_model.forward(token_ids), None


def calculate_summary(results, *, raw_bytes, tokenizer_tokens, truncated):
    n = int(results["total_scored"])
    if n <= 0:
        raise ValueError("No next-token targets were scored")
    nll = float(results["total_nll"])
    elapsed = float(results["elapsed"])
    ce = nll / n
    # Match the native script's BPB definition: divide by original UTF-8 file
    # bytes for an untruncated evaluation. Initial non-BOS tokens are unscored;
    # document-mode, tokenizer, and boundary policy must be compared together.
    bpb = None if truncated else nll / (raw_bytes * math.log(2.0))
    return {
        "total_nll": nll,
        "scored_tokens": n,
        "cross_entropy_nats_per_token": ce,
        "perplexity": math.exp(ce),
        "bits_per_byte": bpb,
        "raw_utf8_bytes": int(raw_bytes),
        "tokenizer_tokens": int(tokenizer_tokens),
        "tokens_per_byte": (tokenizer_tokens / raw_bytes) if not truncated else None,
        "evaluation_windows": int(results["steps"]),
        "elapsed_seconds": elapsed,
        "scored_tokens_per_second": n / elapsed if elapsed > 0 else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Local SmolLM2/Llama-compatible GGUF checkpoint")
    parser.add_argument("--data", type=Path, required=True,
                        help="WikiText-2 wiki.test.raw (the original UTF-8 text)")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--context-length", type=int, default=256,
                        help="Full-causal attention window (default 256)")
    parser.add_argument("--stride", type=int, default=None,
                        help="New targets per window (default: context length)")
    parser.add_argument("--document-mode", choices=("continuous", "article"),
                        default="continuous",
                        help="Match the native script's WikiText article-boundary policy")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Optional quick-run token limit; disables full-corpus BPB")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Windows between progress reports (0 disables)")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional machine-readable results path")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"GGUF file not found: {args.model}")
    if not args.data.is_file():
        parser.error(f"WikiText-2 text file not found: {args.data}")
    if args.context_length < 1:
        parser.error("--context-length must be positive")
    stride = args.stride if args.stride is not None else args.context_length
    if not 1 <= stride <= args.context_length:
        parser.error("--stride must be in [1, --context-length]")
    if args.max_tokens is not None and args.max_tokens < 2:
        parser.error("--max-tokens must be >=2")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")

    from gguf import GGUFReader
    from mini_llm.external.gguf_loader import GGUFWeightMap
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    from mini_llm.external import load_gguf_model

    # Metadata-only GGUFReader first, so bad metadata/unsupported tokenizers
    # fail before the expensive GPU weight import.
    inspection = GGUFWeightMap(GGUFReader(str(args.model)))
    if args.context_length > inspection.config.max_position_embeddings:
        parser.error(
            f"--context-length {args.context_length} exceeds model limit "
            f"{inspection.config.max_position_embeddings}"
        )
    tokenizer = load_gguf_tokenizer(
        inspection.reader, vocab_size=inspection.config.vocab_size
    )
    text = args.data.read_text(encoding="utf-8")
    original_bytes = len(text.encode("utf-8"))
    streams, token_count, document_count = tokenize_eval_streams(
        text,
        _GGUFTokenIDs(tokenizer),
        document_mode=args.document_mode,
        max_tokens=args.max_tokens,
    )
    scoreable = sum(max(0, len(s) - 1) for s in streams)
    if scoreable <= 0:
        raise RuntimeError("No scoreable target tokens in WikiText input")
    if not all(0 <= i < inspection.config.vocab_size for s in streams for i in s):
        raise RuntimeError("GGUF tokenizer token ID outside model vocabulary")
    # --max-tokens changes the evaluated text; BPB against full source bytes
    # would be misleading. Report n/a even if this particular file is shorter.
    truncated = args.max_tokens is not None

    print("=" * 60)
    print("WikiText-2 — External GGUF / native MyLocal_LLM inference")
    print("=" * 60)
    print(f"Model:           {args.model.resolve()}")
    print(f"Backend:         {BACKEND_NAME}")
    print(f"Runtime dtype:   {args.dtype}")
    print(f"GGUF tensors:    {', '.join(inspection.tensor_types)}")
    print(f"Tokenizer:       GGUF embedded ({tokenizer.spec.pre}, no network)")
    print(f"Document mode:   {args.document_mode} ({document_count} detected)")
    print(f"Context length:  {args.context_length}")
    print(f"Stride:          {stride}")
    print(f"Raw bytes:       {original_bytes:,}")
    print(f"Tokenizer tokens:{token_count:>10,d}")
    print(f"Scorable tokens: {scoreable:,}")
    print("Loading pretrained weights...")
    model, provenance = load_gguf_model(args.model, dtype=args.dtype)
    if model.config != inspection.config:
        raise RuntimeError("GGUF model configuration changed between inspection and loading")
    print("Running teacher-forced scoring with native full-causal inference...")
    results = run_windowed_evaluation(
        _ExternalWindowModel(model),
        streams,
        context=args.context_length,
        stride=stride,
        routed_eval=False,
        memory_cfg=None,
        progress_every=args.progress_every,
    )
    metrics = calculate_summary(
        results, raw_bytes=original_bytes, tokenizer_tokens=token_count,
        truncated=truncated,
    )
    print("\n" + "=" * 60)
    print("WikiText-2 results")
    print("=" * 60)
    print(f"Cross-entropy:   {metrics['cross_entropy_nats_per_token']:.6f} nats/token")
    print(f"Perplexity:      {metrics['perplexity']:.4f}")
    print("Bits per byte:   " + (f"{metrics['bits_per_byte']:.6f}"
                                  if metrics['bits_per_byte'] is not None else "n/a (--max-tokens)"))
    print(f"Scored tokens:   {metrics['scored_tokens']:,}")
    print(f"Eval windows:    {metrics['evaluation_windows']:,}")
    print(f"Time:            {metrics['elapsed_seconds']:.2f} s")
    print(f"Throughput:      {metrics['scored_tokens_per_second']:.1f} scored tok/s")
    print("Note: as in the native evaluator, first tokens without a preceding "
          "context are not scored; BPB uses total original file bytes.")
    print("Compare against native runs using the SAME dataset, document mode, "
          "context length and stride. This file is an INSTRUCT model.")
    if args.output_json:
        report = {
            "format": "external-gguf-wikitext-v1",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "backend": BACKEND_NAME,
            "runtime_dtype": args.dtype,
            "gguf_tensor_types": inspection.tensor_types,
            "tokenizer_source": "embedded-gguf",
            "document_mode": args.document_mode,
            "context_length": args.context_length,
            "stride": stride,
            "max_tokens": args.max_tokens,
            "provenance": {k: str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict))
                           else v for k, v in provenance.items()},
            "metrics": metrics,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Results JSON:    {args.output_json}")
    return metrics


if __name__ == "__main__":
    main()
