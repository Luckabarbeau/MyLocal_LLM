#!/usr/bin/env python3
"""Text generation inference script.

This script loads a trained model and generates text from prompts.
Supports multiple decoding strategies:
- Greedy: always pick highest probability token
- Temperature sampling: softmax with temperature
- Top-k sampling: sample from top-k tokens
- Top-p (nucleus) sampling: sample from cumulative probability threshold

Usage:
    # Basic generation with KV cache (fast)
    python inference.py \\
        --checkpoint ./checkpoints/mini_10shards \\
        --prompt "The sky is" \\
        --max-new-tokens 50
        --backend kv-cache
    
    # Reference full-prefix generation (slow, for verification)
    python inference.py \\
        --checkpoint ./checkpoints/mini_10shards \\
        --prompt "The sky is" \\
        --max-new-tokens 50
        --backend reference
    
    # Temperature sampling
    python inference.py \\
        --checkpoint ./checkpoints/mini_10shards \\
        --prompt "Once upon a time" \\
        --temperature 0.7 \\
        --max-new-tokens 100
"""

import argparse
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, synchronize
from mini_llm.ops.attention import GQAAttention
from mini_llm.runtime_defaults import format_runtime_profile


# Backward-compatible CLI imports, now backed by the library implementation.
from mini_llm.legacy_generation import (
    TextGenerator, softmax, greedy_decode, temperature_sample,
    top_k_sample, top_p_sample,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained model"
    )
    
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        default="The sky is",
        help="Initial prompt for generation",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "Read the complete prompt from a UTF-8 text file. Useful for "
            "terminal-memory checkpoints with 16k-64k contexts where a shell "
            "command-line argument would be impractical."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "temperature", "top-k", "top-p"],
        default="greedy",
        help="Decoding strategy",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--backend",
        choices=["reference", "kv-cache"],
        default="kv-cache",
        help="Inference backend: 'reference' uses full prefix, 'kv-cache' uses preallocated cache (default)",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["auto", "reference", "gpu-fast", "consumer-gpu"],
        default=None,
        help=(
            "Runtime defaults profile. Default: MINI_LLM_RUNTIME_PROFILE or auto. "
            "For CuPy inference, auto enables the validated GPU fast paths. "
            "Explicit MINI_LLM_* environment variables always win."
        ),
    )
    parser.add_argument(
        "--execution-policy", choices=["auto", "reference", "balanced", "fast", "strict-fast"],
        default="auto", help="0103D inference policy; fast selects eligible CUDA Graph decode automatically",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - keep model loaded for multiple prompts",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=128,
        help="Maximum tokens to keep in context for interactive mode (default: 128)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help=(
            "Inference KV-cache horizon. Defaults to the checkpoint context "
            "length. Sparse-attention checkpoints may use a larger value to "
            "exercise their learned/dilated long-context structure."
        ),
    )
    
    args = parser.parse_args()
    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file)
        args.prompt = prompt_path.read_text(encoding="utf-8")
    
    # 0103C: CLI, benchmarks, facade and serving use one native loader.
    from mini_llm.model_loading import load_native_checkpoint
    checkpoint_path = Path(args.checkpoint)
    import time as time_mod
    load_start = time_mod.time()
    loaded = load_native_checkpoint(checkpoint_path,
                                    requested_context=args.max_context,
                                    runtime_profile=args.runtime_profile,
                                    apply_profile=args.execution_policy != "reference",
                                    execution_policy=args.execution_policy)
    model, tokenizer, config = loaded.model, loaded.tokenizer, loaded.config
    memory_cfg = getattr(config, "memory_context", None)
    print(f"Loaded config: {config.d_model}d model, {config.n_layers} layers")
    print(format_runtime_profile(loaded.runtime_profile) if loaded.runtime_profile else
          "Runtime profile: reference (no GPU optimization defaults)")
    print(f"Loaded tokenizer: {len(tokenizer)} tokens")
    print(f"Inference backend: {args.backend}; execution policy: {args.execution_policy}")
    print(f"Loaded {len(model.parameters())} parameter arrays")
    load_time = time_mod.time() - load_start
    print(f"Model loading time: {load_time:.3f}s")
    inference_context = loaded.context_length
    print(f"Inference context/cache horizon: {inference_context:,} tokens")
    if getattr(config, "progressive_depth", False):
        print(f"Progressive checkpoint active depth: {model.active_layers}/{model.max_layers}")
    if memory_cfg is not None and memory_cfg.enabled:
        print(f"History integration: {memory_cfg.integration_mode}; "
              f"addressable horizon {inference_context:,} tokens")
    generator = TextGenerator(model, tokenizer, max_context=inference_context)

    if args.interactive:
        # Interactive mode - keep model loaded for multiple prompts
        print()
        print("=" * 60)
        print("Interactive Mode")
        print("=" * 60)
        print("Enter prompts. Type 'quit' or 'exit' to stop.")
        print(f"Max new tokens per generation: {args.max_new_tokens}")
        print(f"Strategy: {args.strategy}")
        if args.strategy in ["temperature", "top-k", "top-p"]:
            print(f"Temperature: {args.temperature}")
        if args.strategy == "top-k":
            print(f"Top-k: {args.top_k}")
        if args.strategy == "top-p":
            print(f"Top-p: {args.top_p}")
        print()
        
        # Keep context for interactive mode
        full_sequence_ids = []  # Track full sequence across turns
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Exiting interactive mode.")
                    break
                
                # Generate response (the generator handles context internally)
                output_text, timing = generator.generate(
                    prompt=user_input,
                    max_new_tokens=args.max_new_tokens,
                    strategy=args.strategy,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=args.seed,
                    log_speed=True,
                    use_kv_cache=(args.backend == "kv-cache"),
                )
                
                print(f"\nResponse: {output_text}")
                if generator.last_execution_report:
                    print(f"Execution: {generator.last_execution_report}")
                    if args.execution_policy == "strict-fast":
                        generator.last_execution_report.require_fast()
                
                # Calculate forward-only time
                total_forward_time = timing.get('total_forward_time_s', 0)
                total_other_time = timing.get('total_other_time_s', 0)
                
                # Log timing
                print(f"\n--- Prompt Timing ---")
                print(f"Generation speed: {timing['tokens_per_sec']:.2f} tokens/sec ({timing['total_tokens']} tokens)")
                if timing.get("breakdown_is_async_approx", False):
                    print(f"Forward host-enqueue timing: {total_forward_time:.3f}s (CUDA async; diagnostic only)")
                    print(f"Sampling/host timing:        {total_other_time:.3f}s (may include waiting on prior GPU work)")
                else:
                    print(f"Model Forward Pass: {total_forward_time:.3f}s ({timing['avg_forward_time_s']*1000:.1f}ms/token)")
                    print(f"Other operations:   {total_other_time:.3f}s")
                
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except EOFError:
                print("\nEOF reached.")
                break
    else:
        # Single generation mode
        print()
        print("=" * 60)
        print("Text Generation")
        print("=" * 60)
        if args.prompt_file is not None:
            print(f"Prompt file: {args.prompt_file} ({len(args.prompt):,} characters)")
        else:
            print(f"Prompt: {args.prompt}")
        print(f"Strategy: {args.strategy}")
        if args.strategy in ["temperature", "top-k", "top-p"]:
            print(f"Temperature: {args.temperature}")
        if args.strategy == "top-k":
            print(f"Top-k: {args.top_k}")
        if args.strategy == "top-p":
            print(f"Top-p: {args.top_p}")
        print()
        
        prompt_start = time_mod.time()
        output, timing = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            log_speed=True,
            use_kv_cache=(args.backend == "kv-cache"),
        )
        prompt_time = time_mod.time() - prompt_start
    
        print(f"Generated: {output}")
        if generator.last_execution_report:
            print(f"Execution: {generator.last_execution_report}")
            if args.execution_policy == "strict-fast":
                generator.last_execution_report.require_fast()
        # Calculate forward-only time (model inference)
        total_forward_time = timing.get('total_forward_time_s', 0)
        total_other_time = timing.get('total_other_time_s', 0)
        
        print(f"\n--- Timing ---")
        print(f"Model loading: {load_time:.3f}s")
        print(f"Prompt processing: {prompt_time:.3f}s")
        print(f"Generation speed: {timing['tokens_per_sec']:.2f} tokens/sec ({timing['total_tokens']} tokens)")
        if args.backend == "kv-cache":
            print(f"KV prefill:       {timing.get('prefill_time_s', 0.0)*1000:.2f} ms")
            print(f"Cached decode:    {timing.get('decode_tokens_per_sec', 0.0):.2f} tokens/sec")
        if timing.get("breakdown_is_async_approx", False):
            print(f"\nForward host-enqueue timing: {total_forward_time:.3f}s (CUDA async; diagnostic only)")
            print(f"Sampling/host timing:        {total_other_time:.3f}s (may include waiting on prior GPU work)")
        else:
            print(f"\nModel Forward Pass: {total_forward_time:.3f}s ({timing['avg_forward_time_s']*1000:.1f}ms/token)")
            print(f"Other operations:   {total_other_time:.3f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
