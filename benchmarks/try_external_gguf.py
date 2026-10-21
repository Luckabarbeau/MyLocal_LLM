"""Inspect or greedily decode a Llama-compatible GGUF through MyLocal_LLM.

Only GGUF file parsing/dequantization uses `gguf`. All Transformer
computations are the repository's native NumPy/CuPy inference operations.

Examples:
  python -m benchmarks.try_external_gguf --model ./model.gguf --inspect
  MINI_LLM_BACKEND=cupy python -m benchmarks.try_external_gguf \\
      --model ./model.gguf --tokenizer-repo HuggingFaceTB/SmolLM2-360M-Instruct \\
      --prompt 'Explain numerical diffusion.' --chat --max-new-tokens 64
"""

import argparse
from pathlib import Path
import time


def _tokenizer(args, model_dir):
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
        raise FileNotFoundError(
            f"Missing tokenizer.json at {path}; either supply --tokenizer PATH "
            "or --tokenizer-repo HF_MODEL_ID to download only tokenizer.json"
        )
    return Tokenizer.from_file(str(path)), path


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
    parser.add_argument("--tokenizer-repo", help="HF repository to fetch tokenizer.json if absent")
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

    tokenizer, tokenizer_file = _tokenizer(args, path.parent)
    print("Tokenizer:", tokenizer_file)
    text = args.prompt
    if args.chat:
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
