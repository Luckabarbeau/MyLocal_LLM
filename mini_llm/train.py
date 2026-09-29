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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random minibatch from shard data.
    
    Args:
        shard_data: Array of shape (num_docs, context_length)
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence (defaults to full context)
        
    Returns:
        Tuple of (inputs, targets) where targets are shifted by 1 position
    """
    if seq_length is None:
        seq_length = shard_data.shape[1]
    
    num_docs, context_len = shard_data.shape
    
    # Ensure we can sample valid sequences
    max_start = context_len - seq_length
    if max_start <= 0:
        raise ValueError(
            f"context_len ({context_len}) must be > seq_length ({seq_length})."
        )
    
    # Sample random starting positions
    start_pos = np.random.randint(0, max_start, batch_size)
    
    # Extract sequences
    indices = np.arange(seq_length)
    inputs = np.zeros((batch_size, seq_length), dtype=np.uint16)
    
    for i, pos in enumerate(start_pos):
        inputs[i] = shard_data[i % num_docs, pos : pos + seq_length]
    
    # Targets are shifted by 1
    targets = np.zeros_like(inputs)
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = 0  # No target for last position
    
    return inputs, targets


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
        """
        self.model = model
        self.shard_paths = [Path(p) for p in shard_paths]
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.grad_clip = grad_clip
        
        # Optimizer and scheduler
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
        Perform one training step.
        
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
        
        # Backward pass
        d_logits = self.model.backward_loss(loss_cache)
        self.model.backward(d_logits, cache)
        
        # Gradient clipping
        for p in self.model.parameters():
            norm = np.sqrt(np.sum(p.grad * p.grad))
            if norm > self.grad_clip:
                scale = self.grad_clip / (norm + 1e-8)
                p.grad *= scale
        
        # Update parameters
        self.optimizer.step(lr=lr)
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
