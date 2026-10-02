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
import json
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, synchronize
from mini_llm.ops.attention import GQAAttention
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature scaling in FP32.

    CuPy's sorting/partitioning stack does not support BF16, and probability
    calculations are numerically safer in FP32 anyway.  Sampling operates on
    only one vocabulary vector per generated token, so this promotion is
    negligible compared with the model forward pass.
    """
    logits = logits.astype(xp.float32, copy=False)
    # Subtract max for numerical stability
    logits = logits / temperature
    logits = logits - xp.max(logits, axis=-1, keepdims=True)
    exp_logits = xp.exp(logits)
    return exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)


def greedy_decode(logits: np.ndarray) -> int:
    """Greedy decoding: pick highest probability token."""
    logits = logits.astype(xp.float32, copy=False)
    return int(xp.argmax(logits, axis=-1))


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from softmax with temperature."""
    probs = softmax(logits, temperature)
    # Sample from distribution
    cumprobs = xp.cumsum(probs)
    r = xp.random.random()
    return int(xp.searchsorted(cumprobs, r))


def top_k_sample(logits, k: int, temperature: float = 1.0) -> int:
    """Sample from the top-k logits without a full-vocabulary softmax/sort.

    Softmax is monotonic, so the top-k probabilities are exactly the top-k
    logits.  Select those logits in O(V) with ``argpartition`` and perform the
    expensive exp/cumsum work only on the k retained values.

    This intentionally returns a Python ``int`` because the current generation
    loop performs EOS handling and builds a Python token list.  Consequently
    there is still one GPU->CPU synchronization per generated token, but no
    full-vocabulary sort, mask, normalization, or cumulative sum.
    """
    if k is None:
        raise ValueError("top_k must be provided for top-k sampling")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0 for top-k sampling")

    # The generation path supplies a single [vocab] vector.  Flattening keeps
    # the helper robust to an accidental [1, vocab] input without making a
    # copy.
    logits = logits.reshape(-1).astype(xp.float32, copy=False)
    vocab_size = int(logits.shape[0])
    k = max(1, min(int(k), vocab_size))

    if k == vocab_size:
        indices = xp.arange(vocab_size, dtype=xp.int32)
    else:
        # We do not need the top-k entries sorted; sampling only requires the
        # corresponding weights.  argpartition avoids the previous O(V log V)
        # full argsort.
        partition = vocab_size - k
        indices = xp.argpartition(logits, partition)[partition:]

    # Advanced indexing already creates a compact k-element array.  Convert
    # that small array to FP32 and do all probability work there.
    top_logits = logits[indices].astype(xp.float32, copy=False)
    if temperature != 1.0:
        top_logits /= xp.float32(temperature)

    # Stable unnormalised softmax weights.  Explicit normalisation is not
    # required for inverse-CDF sampling: drawing on [0, sum(weights)) is
    # exactly equivalent and removes another divide.
    top_logits -= xp.max(top_logits)
    xp.exp(top_logits, out=top_logits)
    cdf = xp.cumsum(top_logits)

    r = xp.asarray(xp.random.random(), dtype=xp.float32) * cdf[-1]
    selected_slot = xp.searchsorted(cdf, r)
    selected_id = indices[selected_slot]

    # Exactly one device->host scalar transfer remains in this implementation.
    return int(selected_id)


def top_p_sample(logits: np.ndarray, p: float, temperature: float = 1.0) -> int:
    """Sample from top-p (nucleus) tokens."""
    probs = softmax(logits, temperature)
    
    # Sort by probability
    sorted_indices = xp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Find cutoff for cumulative probability
    cumprobs = xp.cumsum(sorted_probs)
    cutoff_idx = xp.searchsorted(cumprobs, p) + 1
    
    # Mask out tokens below cutoff
    mask = xp.zeros_like(probs)
    mask[sorted_indices[:cutoff_idx]] = 1.0
    
    masked_probs = probs * mask
    masked_probs = masked_probs / xp.sum(masked_probs)
    
    # Sample from masked distribution
    cumprobs = xp.cumsum(masked_probs)
    r = xp.random.random()
    return int(xp.searchsorted(cumprobs, r))


class TextGenerator:
    """Text generation wrapper for the language model."""

    def __init__(self, model, tokenizer, max_context: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context

    def _get_eos_id(self) -> int:
        """Get EOS token ID from tokenizer or use default."""
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            return self.tokenizer.encode(self.tokenizer.eos_token)[0]
        # Default: 2 is commonly used as EOS
        return 2

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        log_speed: bool = False,
        use_kv_cache: bool = True,
    ) -> tuple[str, dict]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Initial text prompt
            max_new_tokens: Maximum tokens to generate
            strategy: Decoding strategy ('greedy', 'temperature', 'top-k', 'top-p')
            temperature: Temperature for sampling (used if strategy supports it)
            top_k: Top-k parameter (used if strategy == 'top-k')
            top_p: Top-p parameter (used if strategy == 'top-p')
            seed: Random seed for reproducibility
            log_speed: If True, return timing information
            use_kv_cache: If True, use KV cache (faster)
            
        Returns:
            Tuple of (generated_text, timing_info)
        """
        timing = {}
        
        if seed is not None:
            np.random.seed(seed)
            xp.random.seed(seed)
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        
        # Ensure we don't exceed context length
        if len(input_ids) > self.max_context:
            input_ids = input_ids[-self.max_context:]
        
        generated_ids = list(input_ids)
        eos_id = self._get_eos_id()
        
        # Start timing.  Synchronize only at the outer boundary so total
        # throughput reflects completed GPU work without adding a new barrier
        # between decode and sampling on every token.
        if log_speed:
            synchronize()
            total_start_time = time.perf_counter()
            forward_times = []
            other_times = []
        
        if use_kv_cache:
            # Use KV cache with proper prefill-once pattern
            # Reset state for this new prompt generation
            self._ensure_inference_model(reset=True)
            
            # 1. PREFILL ONCE with the entire prompt
            input_tensor = xp.array([input_ids], dtype=xp.uint16)
            input_tensor = xp.asarray(input_tensor, dtype=xp.int32)
            
            if log_speed:
                forward_start = time.perf_counter()
            logits = self._inference_model.prefill(input_tensor, self._state)
            if log_speed:
                forward_end = time.perf_counter()
                forward_times.append(forward_end - forward_start)
            
            # 2. AUTOREGRESSIVE DECODE for each new token
            for step in range(max_new_tokens):
                # Get logits for last position and sample
                if log_speed:
                    sample_start = time.perf_counter()
                
                last_logits = logits[0]  # [vocab]
                
                if strategy == "greedy":
                    next_id = greedy_decode(last_logits)
                elif strategy == "temperature":
                    next_id = temperature_sample(last_logits, temperature)
                elif strategy == "top-k":
                    next_id = top_k_sample(last_logits, top_k, temperature)
                elif strategy == "top-p":
                    next_id = top_p_sample(last_logits, top_p, temperature)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                if log_speed:
                    sample_end = time.perf_counter()
                    other_times.append(sample_end - sample_start)
                
                generated_ids.append(next_id)
                
                # Stop at EOS
                if next_id == eos_id:
                    break
                
                # Check capacity before decoding next token
                if not self._state.has_capacity(1):
                    break
                
                # Decode one token
                if log_speed:
                    forward_start = time.perf_counter()
                
                next_tensor = xp.array([[next_id]], dtype=xp.int32)
                logits = self._inference_model.decode_one(next_tensor, self._state)
                
                if log_speed:
                    forward_end = time.perf_counter()
                    forward_times.append(forward_end - forward_start)
        else:
            # Full prefix forward (no KV cache)
            for _ in range(max_new_tokens):
                # Prepare input (last max_context tokens)
                context = generated_ids[-self.max_context:]
                input_tensor = xp.array([context], dtype=xp.uint16)
                
                if log_speed:
                    forward_start = time.perf_counter()
                
                # Reference inference intentionally recomputes the full prefix,
                # but it must not retain training/backward activations.
                logits = self.model.forward(input_tensor, return_cache=False)
                
                if log_speed:
                    forward_end = time.perf_counter()
                    forward_times.append(forward_end - forward_start)
                
                # Get logits for last position and sample
                last_logits = logits[0, -1, :]
                
                if strategy == "greedy":
                    next_id = greedy_decode(last_logits)
                elif strategy == "temperature":
                    next_id = temperature_sample(last_logits, temperature)
                elif strategy == "top-k":
                    next_id = top_k_sample(last_logits, top_k, temperature)
                elif strategy == "top-p":
                    next_id = top_p_sample(last_logits, top_p, temperature)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                generated_ids.append(next_id)

                # Full-prefix reference decoding visits a new sequence length
                # every step.  Drop the large logits/prefix tensors and clear
                # length-dependent attention/RoPE caches before the next pass.
                # CuPy's pool otherwise keeps many differently-sized blocks
                # reserved, eventually exhausting VRAM during long generation.
                del last_logits, logits, input_tensor
                GQAAttention.clear_caches()
                if BACKEND_NAME == "cupy":
                    xp.get_default_memory_pool().free_all_blocks()
                    xp.get_default_pinned_memory_pool().free_all_blocks()
                
                # Stop at EOS
                if next_id == eos_id:
                    break
        
        # Decode to text (just the new tokens)
        output_text = self.tokenizer.decode(generated_ids[len(input_ids):])
        
        # Calculate timing info
        if log_speed:
            synchronize()
            total_time = time.perf_counter() - total_start_time
            new_tokens = len(generated_ids) - len(input_ids)
            
            total_forward_time = sum(forward_times) if forward_times else 0
            total_other_time = sum(other_times) if other_times else 0
            avg_forward_time = total_forward_time / len(forward_times) if forward_times else 0
            avg_other_time = total_other_time / len(other_times) if other_times else 0
            
            timing = {
                "tokens_per_sec": new_tokens / total_time if total_time > 0 else 0,
                "total_tokens": new_tokens,
                "total_time_s": total_time,
                "forward_times": forward_times,
                "other_times": other_times,
                "total_forward_time_s": total_forward_time,
                "total_other_time_s": total_other_time,
                "avg_forward_time_s": avg_forward_time,
                "avg_other_time_s": avg_other_time,
                "breakdown_is_async_approx": BACKEND_NAME == "cupy",
            }
        
        return output_text, timing

    def _ensure_inference_model(self, reset: bool = False):
        """Ensure inference model and state are initialized for KV cache.
        
        Args:
            reset: If True, create a fresh state (used for new prompts)
        """
        if not hasattr(self, '_inference_model'):
            from mini_llm.inference_model import InferenceModel
            
            self._inference_model = InferenceModel(
                self.model.config,
                dtype=self.model.dtype
            )
            self._inference_model.set_weights(self.model)
            
            self._state = self._inference_model.create_generation_state(
                batch_size=1,
                max_length=self.max_context
            )
        elif reset:
            # Reset state for new prompt
            self._state = self._inference_model.create_generation_state(
                batch_size=1,
                max_length=self.max_context
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
    
    args = parser.parse_args()
    
    # Load model config
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = ModelConfig(**config_dict)
    print(f"Loaded config: {config.d_model}d model, {config.n_layers} layers")
    
    # Load tokenizer
    tokenizer_path = checkpoint_path / "tokenizer.json"
    if not tokenizer_path.exists():
        # Try parent directory
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Detect tokenizer format by reading the file
    with open(tokenizer_path) as f:
        tokenizer_config = json.load(f)
    
    # Check if it's in SimpleBPETokenizer format (has vocab_size, token_to_id fields)
    # vs HuggingFace format (has version, model.type fields)
    is_simple_format = "vocab_size" in tokenizer_config and "token_to_id" in tokenizer_config
    
    if is_simple_format:
        tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))
    else:
        # Use FastBPETokenizer for HuggingFace format
        tokenizer = FastBPETokenizer.load(str(tokenizer_path))
    
    print(f"Loaded tokenizer: {len(tokenizer)} tokens")
    print(f"Inference backend: {args.backend}")
    
    # Load model with timing
    import time as time_mod
    load_start = time_mod.time()
    
    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    
    # Load checkpoint (skip optimizer state for inference - it's 12GB and not needed)
    param_names = [p.name for p in model.parameters()]
    loaded_params, _, _ = load_checkpoint(checkpoint_path, param_names=param_names, skip_optimizer=True)
    
    # Apply loaded parameters
    for p in model.parameters():
        if p.name in loaded_params:
            p.data[...] = loaded_params[p.name]
    
    load_time = time_mod.time() - load_start
    print(f"Loaded {len(loaded_params)} parameter arrays")
    print(f"Model loading time: {load_time:.3f}s")
    
    # Create generator
    generator = TextGenerator(model, tokenizer, max_context=config.context_length)
    
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
        # Calculate forward-only time (model inference)
        total_forward_time = timing.get('total_forward_time_s', 0)
        total_other_time = timing.get('total_other_time_s', 0)
        
        print(f"\n--- Timing ---")
        print(f"Model loading: {load_time:.3f}s")
        print(f"Prompt processing: {prompt_time:.3f}s")
        print(f"Generation speed: {timing['tokens_per_sec']:.2f} tokens/sec ({timing['total_tokens']} tokens)")
        if timing.get("breakdown_is_async_approx", False):
            print(f"\nForward host-enqueue timing: {total_forward_time:.3f}s (CUDA async; diagnostic only)")
            print(f"Sampling/host timing:        {total_other_time:.3f}s (may include waiting on prior GPU work)")
        else:
            print(f"\nModel Forward Pass: {total_forward_time:.3f}s ({timing['avg_forward_time_s']*1000:.1f}ms/token)")
            print(f"Other operations:   {total_other_time:.3f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
