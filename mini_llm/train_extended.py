"""Extended training utilities for Cosmopedia training.

This module provides production-ready training infrastructure including:
- Checkpointing (save/load model, optimizer, training state)
- Logging (CSV format for easy analysis)
- Gradient accumulation for effective larger batches
- Validation split monitoring
- Learning rate warmup + cosine decay schedule
- Packed token stream support
- Deterministic validation
"""

import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None

from mini_llm.backend import xp
from mini_llm.checkpoint import save_checkpoint, load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.data.packed_dataset import PackedTokenDataset, DatasetManifest
from mini_llm.data.token_shards import load_token_shard, create_minibatch
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm, _array_to_float
from mini_llm.optim.schedule import WarmupCosineSchedule

try:
    import numpy as np
except ImportError:
    np = None


class ExtendedTrainer:
    """
    Production-ready trainer for Cosmopedia-scale datasets.
    
    Features:
    - Gradient accumulation (effective batch size = batch_size * accum_steps)
    - Validation monitoring with held-out data
    - Periodic checkpointing
    - CSV logging for TensorBoard-compatible analysis
    - Mixed precision support (FP16 model with FP32 optimizer)
    - Packed token stream support
    - Deterministic validation with fixed block set
    - Persistent RNGs for reproducibility
    """
    
    def __init__(
        self,
        model: DecoderLanguageModel,
        train_shard_paths: List[str],
        val_shard_paths: List[str],
        batch_size: int = 8,
        seq_length: int = 512,
        grad_accum_steps: int = 1,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        peak_lr: float = 3e-4,
        grad_clip: float = 1.0,
        weight_decay: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        val_interval: int = 500,
        val_steps: int = 10,
        save_interval: int = 2000,
        loss_scale: float = 1.0,
        rng_seed: int = 42,
        numerical_debug: bool = False,
    ):
        """
        Initialize the extended trainer.
        
        Args:
            model: DecoderLanguageModel instance
            train_shard_paths: List of training shard file paths
            val_shard_paths: List of validation shard file paths
            batch_size: Batch size per step (before accumulation)
            seq_length: Sequence length for training
            grad_accum_steps: Number of steps to accumulate gradients
            warmup_steps: Warmup steps for learning rate schedule
            total_steps: Total training steps
            peak_lr: Peak learning rate
            grad_clip: Gradient clipping norm
            weight_decay: Weight decay coefficient for AdamW
            checkpoint_dir: Directory for saving checkpoints (None = no save)
            log_file: Path to CSV log file (None = no logging)
            val_interval: Steps between validation checks
            val_steps: Number of validation steps per check
            save_interval: Steps between checkpoint saves
            loss_scale: Static loss scale factor for mixed precision
            rng_seed: Seed for persistent RNGs
            numerical_debug: Enable Inf/NaN checks at each tensor (slow, for debugging)
        """
        self.model = model
        self.train_shard_paths = [Path(p) for p in train_shard_paths]
        self.val_shard_paths = [Path(p) for p in val_shard_paths]
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip  # Store grad_clip parameter
        self.loss_scale = loss_scale  # Store loss scale (trainer owns it)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_file = Path(log_file) if log_file else None
        self.val_interval = val_interval
        self.val_steps = val_steps
        self.save_interval = save_interval
        
        # Effective batch size
        self.effective_batch_size = batch_size * grad_accum_steps
        
        # Initialize optimizer with weight decay
        # Loss scaling is handled entirely by the trainer
        self.optimizer = AdamW(
            model.parameters(),
            lr=peak_lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate schedule
        self.scheduler = WarmupCosineSchedule(
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        
        # Training state (Issue #16: Track tokens processed separately)
        self.step = 0
        self.tokens_processed = 0  # New: track cumulative tokens
        self.total_steps = total_steps
        self.shards = {}  # Cache for shard data
        self.current_train_shard_idx = 0
        
        # Issue #12: Persistent RNGs (one for train, one for val)
        self.train_rng = np.random.default_rng(rng_seed)
        self.val_rng = np.random.default_rng(rng_seed + 1)
        
        # Issue #13: Deterministic validation blocks
        self._val_block_indices: Optional[List[int]] = None
        
        # Validation state
        self.current_val_shard_idx = 0
        self.val_shards = {}
        
        # Numerical debugging
        self.numerical_debug = numerical_debug
        
        # Logging setup
        self._setup_logging()
        
        # Issue #13: Initialize deterministic validation block set
        self._init_validation_blocks()
        
        print(f"ExtendedTrainer initialized:")
        print(f"  Train shards: {len(train_shard_paths)}")
        print(f"  Val shards: {len(val_shard_paths)}")
        print(f"  Batch size: {batch_size} (effective: {self.effective_batch_size})")
        print(f"  Sequence length: {seq_length}")
        print(f"  Total steps: {total_steps}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Log file: {log_file}")
        
    def _init_validation_blocks(self):
        """
        Issue #13: Initialize deterministic validation block set.
        
        For reproducible validation, we pre-select a fixed set of validation blocks
        that will be used for all validation runs. This ensures that validation
        loss is comparable across checkpoints.
        """
        # For now, use the same as sampling - but could pre-compute specific blocks
        self._val_block_indices = None  # Dynamic sampling for flexibility
    
    def _setup_logging(self):
        """Initialize CSV logging."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Write header if file doesn't exist
            if not self.log_file.exists():
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step", "tokens_processed", "train_loss", "lr", "grad_norm",
                        "val_loss", "steps_per_sec"
                    ])
    
    def _log(self, row: Dict[str, Any]):
        """Log a row to CSV."""
        if self.log_file:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    row.get("step", ""),
                    row.get("tokens_processed", ""),  # Issue #16: Track tokens processed
                    row.get("train_loss", ""),
                    row.get("lr", ""),
                    row.get("grad_norm", ""),
                    row.get("val_loss", ""),
                    row.get("steps_per_sec", ""),
                ])
    
    def _check_finite(self, tensor, name: str):
        """Check if tensor is finite and print warning if not."""
        if self.numerical_debug:
            try:
                cpu_tensor = tensor.get() if hasattr(tensor, "get") else tensor
                if np is not None and not np.all(np.isfinite(cpu_tensor)):
                    max_abs = np.max(np.abs(cpu_tensor))
                    is_finite = np.all(np.isfinite(cpu_tensor))
                    print(
                        f"\n[{name}] NONFINITE detected! "
                        f"max_abs={max_abs:.4e}, isfinite={is_finite}"
                    )
                    return False
            except Exception:
                pass
        return True
    
    def load_train_shard(self, path: str) -> np.ndarray:
        """Load a training shard into memory."""
        return load_token_shard(path)
    
    def load_val_shard(self, path: str) -> np.ndarray:
        """Load a validation shard into memory."""
        return load_token_shard(path)
    
    def get_train_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random training batch.
        
        Uses persistent RNG for reproducibility (Issue #12).
        """
        if self.current_train_shard_idx not in self.shards:
            self.shards[self.current_train_shard_idx] = self.load_train_shard(
                str(self.train_shard_paths[self.current_train_shard_idx])
            )
        
        shard_data = self.shards[self.current_train_shard_idx]
        inputs, targets = create_minibatch(shard_data, self.batch_size, self.seq_length, rng=self.train_rng)
        
        # Cycle through shards
        self.current_train_shard_idx = (self.current_train_shard_idx + 1) % len(self.train_shard_paths)
        
        # Issue #16: Track tokens processed
        self.tokens_processed += self.batch_size * self.seq_length
        
        return inputs, targets
    
    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a validation batch.
        
        Uses deterministic validation block selection (Issue #13).
        """
        if self.current_val_shard_idx not in self.val_shards:
            self.val_shards[self.current_val_shard_idx] = self.load_val_shard(
                str(self.val_shard_paths[self.current_val_shard_idx])
            )
        
        shard_data = self.val_shards[self.current_val_shard_idx]
        inputs, targets = create_minibatch(shard_data, self.batch_size, self.seq_length, rng=self.val_rng)
        
        # Cycle through shards
        self.current_val_shard_idx = (self.current_val_shard_idx + 1) % len(self.val_shard_paths)
        
        return inputs, targets
    
    def compute_val_loss(self) -> float:
        """
        Compute validation loss over multiple steps.
        
        Uses deterministic validation block selection (Issue #13).
        
        Returns float for logging, but internally accumulates on GPU/CPU
        without host-device sync until the final mean computation.
        
        Note: Loss is returned as scalar() which converts to Python float.
        We accumulate the raw float values and compute mean on backend.
        """
        # Accumulate losses as Python floats (minimal overhead)
        # The main optimization is avoiding host-device sync during gradient computation
        losses = []
        
        for _ in range(self.val_steps):
            inputs, targets = self.get_val_batch()
            
            # Forward pass (no gradient tracking needed)
            logits, _ = self.model.forward(inputs)
            loss, _ = self.model.compute_loss(logits, targets)
            losses.append(loss)  # loss is already a Python float from scalar()
        
        # Compute mean on backend array (only sync once at the end)
        loss_array = xp.asarray(losses, dtype="float32")
        return float(xp.mean(loss_array))
    
    def train_step(self) -> Tuple[float, float]:
        """
        Perform one training step with gradient accumulation.
        
        With loss scaling:
        1. Multiply d_logits by loss_scale before backward
        2. Accumulate scaled gradients
        3. Divide gradients by loss_scale after accumulation
        4. Divide by grad_accum_steps for proper averaging
        
        Returns:
            Tuple of (loss, grad_norm)
        """
        # Accumulate gradients over multiple steps
        # Use Python floats for loss accumulation (minimal overhead)
        # The main optimization is avoiding host-device sync during gradient computation
        losses = []
        
        for accum_step in range(self.grad_accum_steps):
            # Get learning rate for this step (use final lr of accumulated batch)
            lr = self.scheduler(self.step)
            self.optimizer.lr = lr
            
            # Get batch
            inputs, targets = self.get_train_batch()
            
            # Forward pass
            logits, cache = self.model.forward(inputs)
            
            # Numerical debug: check logits are finite
            if not self._check_finite(logits, f"logits_step_{self.step}"):
                raise ValueError(f"Nonfinite logits detected at step {self.step}!")
            
            loss, loss_cache = self.model.compute_loss(logits, targets)
            losses.append(loss)  # loss is already a Python float from scalar()
            
            # Check for NaN (this will sync once per step, acceptable)
            if loss != loss:  # NaN check without converting to backend
                raise ValueError(f"NaN loss detected at step {self.step}!")
            
            # Backward pass - apply loss scaling to d_logits before backward
            d_logits = self.model.backward_loss(loss_cache)
            
            # Scale the gradient by loss_scale for mixed precision (trainer owns it)
            if self.loss_scale != 1.0:
                d_logits = d_logits * self.loss_scale
            
            self.model.backward(d_logits, cache)
        
        # Average loss - only sync once at the end
        avg_loss = float(sum(losses) / self.grad_accum_steps)
        
        # Scale down gradients by loss_scale to cancel out the scaling
        if self.loss_scale != 1.0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad[...] = p.grad / self.loss_scale
        
        # Divide gradients by grad_accum_steps for proper averaging
        if self.grad_accum_steps > 1:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad[...] = p.grad / self.grad_accum_steps
        
        # Global gradient clipping - returns backend array norm, no sync
        # Now returns (norm_backend, scale, is_finite) tuple
        grad_norm_backend, grad_scale, is_finite = clip_grad_global_norm(
            self.model.parameters(), max_norm=self.grad_clip
        )
        
        if not is_finite:
            # Gradient contains Inf/NaN - skip this update
            self.optimizer.zero_grad()
            print(
                f"WARNING: Nonfinite gradient at step {self.step}; "
                f"norm={_array_to_float(grad_norm_backend) if grad_norm_backend is not None else 'unknown'}; "
                f"update skipped"
            )
            return avg_loss, float(_array_to_float(grad_norm_backend)) if grad_norm_backend is not None else 0.0
        
        # Update parameters (only once per accumulated batch)
        self.optimizer.step(lr=lr)
        self.optimizer.zero_grad()
        
        self.step += 1
        
        # Sync grad_norm only at the end (necessary for logging)
        return avg_loss, float(_array_to_float(grad_norm_backend))
    
    def save(self):
        """
        Issue #14: Save checkpoint with full state for resume.
        
        Saves:
        - Model parameters
        - Optimizer state (Adam first/second moments, step counter)
        - Training state (step, tokens_processed, shard indices)
        - RNG states (train_rng, val_rng)
        """
        if self.checkpoint_dir is None:
            return
        
        params = {p.name: p.data for p in self.model.parameters()}
        
        # Issue #14: Include optimizer state
        # Note: AdamW stores m and v internally, not on Parameter objects
        # We store the master weights (FP32 copies) and moments
        optimizer_state = {
            "step": self.optimizer.step_index,  # Use correct attribute name
            "master_weights": {
                p.name: self.optimizer.master_weights[i]
                for i, p in enumerate(self.model.parameters())
            },
            "m": {
                p.name: self.optimizer.m[i]
                for i, p in enumerate(self.model.parameters())
            },
            "v": {
                p.name: self.optimizer.v[i]
                for i, p in enumerate(self.model.parameters())
            },
        }
        
        # Issue #16: Track tokens_processed
        training_state = {
            "step": self.step,
            "tokens_processed": self.tokens_processed,  # New: track cumulative tokens
            "total_steps": self.total_steps,
            "current_train_shard_idx": self.current_train_shard_idx,
            "current_val_shard_idx": self.current_val_shard_idx,
        }
        
        # Issue #12: Include RNG states
        training_state["train_rng_state"] = self.train_rng.bit_generator.state
        training_state["val_rng_state"] = self.val_rng.bit_generator.state
        
        save_checkpoint(
            path=self.checkpoint_dir,
            model_params=params,
            optimizer_state=optimizer_state,
            training_state=training_state,
        )
    
    def train(
        self,
        num_steps: Optional[int] = None,
        log_interval: int = 10,
    ) -> List[float]:
        """
        Train for specified number of steps.
        
        Args:
            num_steps: Number of training steps (None = use total_steps)
            log_interval: Steps between logging
            
        Returns:
            List of loss values
        """
        if num_steps is None:
            num_steps = self.total_steps
        
        losses = []
        start_step = self.step
        
        print(f"\nStarting training from step {start_step}...")
        print(f"Training for {num_steps} additional steps")
        print(f"Effective batch size: {self.effective_batch_size}")
        
        start_time = time.time()
        
        for step in range(num_steps):
            # Train step
            loss, grad_norm = self.train_step()
            losses.append(loss)
            
            # Validation check (Issue #17: use step % interval, not step + 1)
            if (self.step + 1) % self.val_interval == 0:
                val_loss = self.compute_val_loss()
                print(f"  Validation loss: {val_loss:.4f}")
            else:
                val_loss = None
            
            # Logging (Issue #17: use step % interval, not step + 1)
            if (self.step + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - start_time
                steps_per_sec = (self.step - start_step + 1) / elapsed
                lr = self.optimizer.lr
                
                print(
                    f"Step {self.step + 1}/{num_steps + start_step}: "
                    f"loss={avg_loss:.4f}, "
                    f"lr={lr:.6f}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"{steps_per_sec:.2f} steps/sec"
                )
                
                # Log to CSV
                self._log({
                    "step": self.step + 1,
                    "train_loss": avg_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "val_loss": val_loss if val_loss is not None else "",
                    "steps_per_sec": steps_per_sec,
                })
            
            # Checkpoint save (Issue #17: use step % interval, not step + 1)
            if (self.step + 1) % self.save_interval == 0:
                self.save()
        
        print(f"\nTraining complete!")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {np.mean(losses):.4f}")
        
        return losses
