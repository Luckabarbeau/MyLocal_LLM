"""Inspect or greedily decode a Llama-compatible GGUF through MyLocal_LLM.

Only GGUF file parsing/dequantization uses `gguf`. All Transformer
computations are the repository's native NumPy/CuPy inference operations.

Examples:
  python -m benchmarks.try_external_gguf --model ./model.gguf --inspect
  MINI_LLM_BACKEND=cupy python -m benchmarks.try_external_gguf \\
      --model ./model.gguf \\
      --prompt 'Explain numerical diffusion.' --chat --max-new-tokens 64
"""

import argparse
from pathlib import Path
import time


def _external_tokenizer(args, model_dir):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install tokenizer support: pip install tokenizers") from exc
    if args.tokenizer:
        path = Path(args.tokenizer).expanduser()
    elif args.tokenizer_repo:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=args.tokenizer_repo, filename="tokenizer.json"))
    else:
        path = model_dir / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"No external tokenizer.json at {path}")
    return Tokenizer.from_file(str(path)), path


def _tokenizer(args, reader, model_dir, vocab_size):
    """Prefer metadata-only reconstruction; external tokenizer is opt-in."""
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    if (args.tokenizer or args.tokenizer_repo) and not args.validate_tokenizer:
        return _external_tokenizer(args, model_dir)
    try:
        return load_gguf_tokenizer(reader, vocab_size=vocab_size), "GGUF embedded metadata (offline)"
    except ValueError:
        # Fallback to a colocated JSON, never silently download from the web.
        if not args.validate_tokenizer and (model_dir / "tokenizer.json").is_file():
            return _external_tokenizer(args, model_dir)
        raise


def _validate_embedded_tokenizer(tokenizer, reference):
    """Require exact token ID equivalence for a deliberately varied sample."""
    examples = [
        "Explain why mesh convergence is important in CFD simulations.",
        "We'LL use 123456789 and 3.14159 in the test!", "é café 🚀",
        "def f(x):\n    return x**2 + 123\n", "  \n\n \t  ",
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    ]
    for line in examples:
        native_ids = tokenizer.encode(line, add_special_tokens=False).ids
        reference_ids = reference.encode(line, add_special_tokens=False).ids
        if native_ids != reference_ids:
            offset = next((i for i, (x, y) in enumerate(zip(native_ids, reference_ids))
                           if x != y), min(len(native_ids), len(reference_ids)))
            raise RuntimeError(
                f"Embedded GGUF tokenizer parity mismatch on {line!r}, "
                f"first differing index={offset}; native={native_ids}, "
                f"reference={reference_ids}"
            )
    print(f"Tokenizer parity passed: {len(examples)} varied inputs")


def _sync_gpu():
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME == "cupy":
        import cupy as cp
        cp.cuda.get_current_stream().synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local GGUF checkpoint file")
    parser.add_argument("--inspect", action="store_true", help="Check config/tensor directory only")
    parser.add_argument("--tokenizer", help="Original model tokenizer.json")
    parser.add_argument("--tokenizer-repo", help="Optional HF tokenizer reference/override; requests a download")
    parser.add_argument("--validate-tokenizer", action="store_true",
                        help="Compare GGUF tokenizer against --tokenizer or --tokenizer-repo")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Validate GGUF tokenizer without loading model weights")
    parser.add_argument("--prompt", default="Explain the finite element method in one paragraph.")
    parser.add_argument("--chat", action="store_true", help="Wrap prompt as SmolLM2-Instruct chat")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--check-cache", action="store_true", help="Compare prefill vs full logits")
    args = parser.parse_args()

    from mini_llm.external.gguf_loader import GGUFWeightMap
    from mini_llm.backend import xp, BACKEND_NAME
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("Install GGUF metadata support: pip install gguf") from exc
    path = Path(args.model).expanduser().resolve()
    if not path.is_file():
        parser.error(f"Cannot find GGUF file: {path}")
    inspected = GGUFWeightMap(GGUFReader(str(path)))
    print("Model:", path)
    print("Backend:", BACKEND_NAME)
    print("Configuration:", inspected.config)
    print("Tensor dtypes:", inspected.tensor_types)
    print("Mapped tensors:", len(inspected))
    if args.inspect:
        return
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    if args.validate_tokenizer and not (args.tokenizer or args.tokenizer_repo):
        parser.error("--validate-tokenizer requires --tokenizer or --tokenizer-repo")
    tokenizer, tokenizer_file = _tokenizer(
        args, inspected.reader, path.parent, inspected.config.vocab_size
    )
    print("Tokenizer:", tokenizer_file)
    if args.validate_tokenizer:
        reference, reference_file = _external_tokenizer(args, path.parent)
        print("Tokenizer parity reference:", reference_file)
        _validate_embedded_tokenizer(tokenizer, reference)
    if args.tokenizer_only:
        print("Tokenizer metadata validated; no model weights loaded")
        return
    text = args.prompt
    if args.chat:
        if hasattr(tokenizer, "chat_prompt"):
            text = tokenizer.chat_prompt(text)
        else:
            text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids
    if not ids:
        parser.error("Tokenizer produced no prompt tokens")
    if len(ids) + args.max_new_tokens > inspected.config.max_position_embeddings:
        parser.error("Prompt + new tokens exceeds model's context length")
    if max(ids) >= inspected.config.vocab_size or min(ids) < 0:
        parser.error("Tokenizer token IDs exceed GGUF vocabulary; wrong tokenizer.json")

    from mini_llm.external import load_gguf_model
    start = time.perf_counter()
    model, provenance = load_gguf_model(path, dtype=args.dtype)
    _sync_gpu()
    load_seconds = time.perf_counter() - start
    print(f"Load time: {load_seconds:.2f}s; tensor types: {provenance['tensor_types']}")
    print(f"Prompt tokens: {len(ids)}")
    array = xp.asarray([ids], dtype=xp.int32)
    state = model.create_generation_state(1, len(ids) + args.max_new_tokens)

    start = time.perf_counter()
    logits = model.prefill(array, state)
    _sync_gpu()
    print(f"Prefill time: {time.perf_counter() - start:.3f}s")
    if args.check_cache:
        # Full causal validation deliberately limited to short prompts.
        if len(ids) > 128:
            parser.error("--check-cache requires <=128 input tokens")
        full_last = model.forward(array)[:, -1, :]
        difference = float(xp.max(xp.abs(full_last - logits)).item())
        print(f"Full forward vs prefill maximum logit error: {difference:.6g}")
        max_allowed = 0.02 if args.dtype == "float16" else 1e-3
        if difference > max_allowed:
            raise RuntimeError(f"Full forward vs prefill logit mismatch: {difference:.6g}")

    stop = {idx for idx in [tokenizer.token_to_id("<|im_end|>"),
                            tokenizer.token_to_id("<|endoftext|>")] if idx is not None}
    generated = []
    start = time.perf_counter()
    for index in range(args.max_new_tokens):
        # Next-token logits are [batch, vocab]; argmax enforces deterministic greedy decoding.
        next_id = int(xp.argmax(logits[0]).item())
        if next_id in stop:
            break
        generated.append(next_id)
        if index + 1 < args.max_new_tokens:
            logits = model.decode_one(xp.asarray([next_id], dtype=xp.int32), state)
    _sync_gpu()
    decode_seconds = time.perf_counter() - start
    print("\n=== GENERATED TEXT ===")
    print(tokenizer.decode(generated, skip_special_tokens=False))
    print("======================")
    print("Generated token IDs:", generated)
    print(f"Generated {len(generated)} tokens in {decode_seconds:.3f}s "
          f"({len(generated) / max(decode_seconds, 1e-9):.2f} token/s, "
          "including initial next-token selection)")
    if BACKEND_NAME == "cupy":
        import cupy as cp
        print(f"CuPy allocated: {cp.get_default_memory_pool().used_bytes()/1024**2:.1f} MiB; "
              f"reserved: {cp.get_default_memory_pool().total_bytes()/1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
