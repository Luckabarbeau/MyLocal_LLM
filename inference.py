#!/usr/bin/env python3
"""Text generation inference script.

This script loads a trained model and generates text from prompts.
Supports multiple decoding strategies:
- Greedy: always pick highest probability token
- Temperature sampling: softmax with temperature
- Top-k sampling: sample from top-k tokens
- Top-p (nucleus) sampling: sample from cumulative probability threshold

Usage:
    # Basic generation
    python inference.py \\
        --checkpoint ./checkpoints/mini_10shards \\
        --prompt "The sky is" \\
        --max-new-tokens 50
    
    # Temperature sampling
    python inference.py \\
        --checkpoint ./checkpoints/mini_10shards \\
        --prompt "Once upon a time" \\
        --temperature 0.7 \\
        --max-new-tokens 100
    
    # Top-p sampling
    python inference.py \\
        --checkpoint ./checkpoints/mini_10shards \\
        --prompt "In the future" \\
        --top-p 0.9 \\
        --max-new-tokens 75
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mini_llm.backend import xp
from mini_llm.checkpoint import load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature scaling."""
    # Subtract max for numerical stability
    logits = logits / temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def greedy_decode(logits: np.ndarray) -> int:
    """Greedy decoding: pick highest probability token."""
    return int(np.argmax(logits, axis=-1))


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from softmax with temperature."""
    probs = softmax(logits, temperature)
    # Sample from distribution
    cumprobs = np.cumsum(probs)
    r = np.random.random()
    return int(np.searchsorted(cumprobs, r))


def top_k_sample(logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
    """Sample from top-k tokens."""
    probs = softmax(logits, temperature)
    
    # Zero out all but top-k
    indices = np.argsort(probs)[-k:]
    mask = np.zeros_like(probs)
    mask[indices] = 1.0
    
    masked_probs = probs * mask
    masked_probs = masked_probs / np.sum(masked_probs)
    
    # Sample from masked distribution
    cumprobs = np.cumsum(masked_probs)
    r = np.random.random()
    return int(np.searchsorted(cumprobs, r))


def top_p_sample(logits: np.ndarray, p: float, temperature: float = 1.0) -> int:
    """Sample from top-p (nucleus) tokens."""
    probs = softmax(logits, temperature)
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Find cutoff for cumulative probability
    cumprobs = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumprobs, p) + 1
    
    # Mask out tokens below cutoff
    mask = np.zeros_like(probs)
    mask[sorted_indices[:cutoff_idx]] = 1.0
    
    masked_probs = probs * mask
    masked_probs = masked_probs / np.sum(masked_probs)
    
    # Sample from masked distribution
    cumprobs = np.cumsum(masked_probs)
    r = np.random.random()
    return int(np.searchsorted(cumprobs, r))


class TextGenerator:
    """Text generation wrapper for the language model."""
    
    def __init__(
        self,
        model: DecoderLanguageModel,
        tokenizer: SimpleBPETokenizer,
        max_context: int = 512,
    ):
        """
        Initialize text generator.
        
        Args:
            model: Trained DecoderLanguageModel
            tokenizer: Tokenizer for encoding/decoding
            max_context: Maximum context length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        seed: int = None,
    ) -> str:
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
            
        Returns:
            Generated text (prompt + completion)
        """
        if seed is not None:
            np.random.seed(seed)
            xp.random.seed(seed)
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        
        # Ensure we don't exceed context length
        if len(input_ids) > self.max_context:
            input_ids = input_ids[-self.max_context:]
        
        # Generate tokens
        generated_ids = list(input_ids)
        
        for _ in range(max_new_tokens):
            # Prepare input (last max_context tokens)
            context = generated_ids[-self.max_context:]
            input_tensor = np.array([context], dtype=np.uint16)
            
            # Forward pass
            logits, _ = self.model.forward(input_tensor)
            
            # Get logits for last position
            last_logits = logits[0, -1, :]
            
            # Sample next token
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
            
            # Stop at EOS
            if next_id == self.tokenizer.token_to_id.get(self.tokenizer.eos_token, 0):
                break
        
        # Decode to text
        output_text = self.tokenizer.decode(generated_ids[len(input_ids):])
        
        return output_text


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
    
    tokenizer = SimpleBPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(str(tokenizer_path))
    print(f"Loaded tokenizer: {len(tokenizer)} tokens")
    
    # Load model
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float16")
    
    # Load checkpoint
    param_names = [p.name for p in model.parameters()]
    loaded_params, _, _ = load_checkpoint(checkpoint_path, param_names=param_names)
    
    # Apply loaded parameters
    for p in model.parameters():
        if p.name in loaded_params:
            p.data[...] = loaded_params[p.name]
    
    print(f"Loaded {len(loaded_params)} parameter arrays")
    
    # Create generator
    generator = TextGenerator(model, tokenizer, max_context=config.context_length)
    
    # Generate text
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
    
    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    
    print(f"Generated: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
