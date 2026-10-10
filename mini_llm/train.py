"""Small-scale training utilities for testing.

This module provides utilities for small-scale training experiments,
particularly useful for validating the model implementation with synthetic
or small real datasets.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mini_llm.data.token_shards import TokenShardWriter
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm
from mini_llm.optim.schedule import WarmupCosineSchedule


def load_token_shard(shard_path: str) -> np.ndarray:
    """
    Load a token shard from binary file.
    
    Args:
        shard_path: Path to .bin file
        
    Returns:
        Array of shape (num_documents, context_length)
    """
    import mmap
    
    with open(shard_path, "rb") as f:
        # Read header: (num_documents, context_length) as int64
        header = f.read(16)
        num_docs = int.from_bytes(header[0:8], byteorder="little")
        context_len = int.from_bytes(header[8:16], byteorder="little")
        
        # Read data using mmap for large files or direct read for small ones
        # Use mmap with size to avoid permission issues
        f.seek(16)
        total_bytes = num_docs * context_len * 2  # uint16 = 2 bytes
        data = f.read(total_bytes)
        
        arr = np.frombuffer(data, dtype=np.uint16)
        return arr.reshape(num_docs, context_len)


def create_minibatch(
    shard_data: np.ndarray,
    batch_size: int,
    seq_length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random minibatch from shard data.
    
    Args:
        shard_data: Array of shape (num_docs, context_length)
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence (defaults to full context)
        rng: Random generator for reproducibility (uses global by default)
        
    Returns:
        Tuple of (inputs, targets) where targets are shifted by 1 position
    """
    # Defer to the canonical implementation in token_shards
    from mini_llm.data.token_shards import create_minibatch as canonical_create_minibatch
    return canonical_create_minibatch(shard_data, batch_size, seq_length, rng)


class MiniTrainer:
    """
    Small-scale trainer for testing and debugging.
    
    This is NOT production-ready. For real training, you'd want:
    - Distributed data loading
    - Gradient accumulation
    - Checkpointing
    - Logging/metrics
    
    But this keeps the implementation visible and transparent.
    """
    
    def __init__(
        self,
        model: DecoderLanguageModel,
        shard_paths: List[str],
        batch_size: int = 8,
        seq_length: int = 512,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        peak_lr: float = 3e-4,
        grad_clip: float = 1.0,
        loss_scale: float = 1.0,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: DecoderLanguageModel instance
            shard_paths: List of paths to token shard files
            batch_size: Batch size for training
            seq_length: Sequence length for training
            warmup_steps: Warmup steps for learning rate schedule
            total_steps: Total training steps
            peak_lr: Peak learning rate
            grad_clip: Gradient clipping norm
            loss_scale: Static loss scale factor for mixed precision (default 1.0)
        """
        self.model = model
        self.shard_paths = [Path(p) for p in shard_paths]
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.grad_clip = grad_clip
        self.loss_scale = loss_scale
        
        # Optimizer - loss scaling is handled by trainer
        self.optimizer = AdamW(model.parameters(), lr=peak_lr)
        self.scheduler = WarmupCosineSchedule(
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        
        # Current state
        self.step = 0
        self.shards = {}
        self.current_shard_idx = 0
        
    def load_shard(self, path: str) -> np.ndarray:
        """Load a shard into memory."""
        return load_token_shard(path)
    
    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a random training batch."""
        # Lazy load shards
        if self.current_shard_idx not in self.shards:
            self.shards[self.current_shard_idx] = self.load_shard(
                str(self.shard_paths[self.current_shard_idx])
            )
        
        shard_data = self.shards[self.current_shard_idx]
        inputs, targets = create_minibatch(shard_data, self.batch_size, self.seq_length)
        
        # Cycle through shards
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
        
        return inputs, targets
    
    def train_step(self) -> float:
        """
        Perform one training step with optional loss scaling.
        
        With loss scaling:
        1. Multiply d_logits by loss_scale before backward
        2. Backward computes scaled gradients
        3. Divide gradients by loss_scale after backward
        
        Returns:
            Loss value
        """
        # Get learning rate for this step
        lr = self.scheduler(self.step)
        self.optimizer.lr = lr
        
        # Get batch
        inputs, targets = self.get_batch()
        
        # Forward pass
        logits, cache = self.model.forward(inputs)
        loss, loss_cache = self.model.compute_loss(logits, targets)
        
        # Check for NaN in loss
        if np.isnan(loss):
            print(f"WARNING: NaN loss detected!")
            print(f"  logits range: [{np.min(logits):.4f}, {np.max(logits):.4f}]")
            print(f"  logits has NaN: {np.any(np.isnan(logits))}")
            print(f"  logits has Inf: {np.any(np.isinf(logits))}")
        
        # Backward pass - apply loss scaling if enabled
        d_logits = self.model.backward_loss(loss_cache)
        
        # Scale gradient by loss_scale before backward
        if self.loss_scale != 1.0:
            d_logits = d_logits * self.loss_scale
        
        self.model.backward(d_logits, cache)
        
        # Scale down gradients by loss_scale to cancel out the scaling
        if self.loss_scale != 1.0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad[...] = p.grad / self.loss_scale
        
        # Global gradient clipping in float32
        clip_grad_global_norm(self.model.parameters(), max_norm=self.grad_clip)
        
        # Update parameters
        self.optimizer.step(lr=lr)
        if hasattr(self.model, "refresh_compute_buffers"):
            self.model.refresh_compute_buffers()
        self.optimizer.zero_grad()
        
        self.step += 1
        
        return float(loss)
    
    def train(
        self,
        num_steps: int,
        log_interval: int = 10,
    ) -> List[float]:
        """
        Train for specified number of steps.
        
        Args:
            num_steps: Number of training steps
            log_interval: Steps between logging
            
        Returns:
            List of loss values
        """
        losses = []
        
        print(f"Starting training for {num_steps} steps...")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Learning rate: {self.optimizer.lr}")
        
        start_time = time.time()
        
        for step in range(num_steps):
            loss = self.train_step()
            losses.append(loss)
            
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                
                print(
                    f"Step {step + 1}/{num_steps}: "
                    f"loss={avg_loss:.4f}, "
                    f"lr={self.optimizer.lr:.6f}, "
                    f"{steps_per_sec:.2f} steps/sec"
                )
        
        return losses
