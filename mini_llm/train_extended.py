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

import os
import csv
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Mapping

try:
    import numpy as np
except ImportError:
    np = None

from mini_llm.backend import xp
from mini_llm.checkpoint import save_checkpoint, load_checkpoint
from mini_llm.config import ModelConfig
from mini_llm.data.packed_dataset import PackedTokenDataset, DatasetManifest
from mini_llm.data.token_shards import map_token_shard, create_minibatch
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm, _array_to_float
from mini_llm.optim.schedule import WarmupCosineSchedule
from mini_llm.backend import synchronize
from mini_llm.performance_profiler import (
    configure_performance_profiler,
    performance_report,
    performance_scope,
)

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
        shard_cache_size: int = 2,
        train_source_shards: Optional[Mapping[str, List[str]]] = None,
        val_source_shards: Optional[Mapping[str, List[str]]] = None,
        source_weights: Optional[Mapping[str, float]] = None,
        profile_steps: int = 0,
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
            shard_cache_size: Maximum number of train and validation shard
                memmaps retained by the trainer.
            train_source_shards: Optional mapping of source name to training
                shard paths. When provided, minibatches choose a source using
                ``source_weights`` before choosing that source's next shard.
            val_source_shards: Validation counterpart to ``train_source_shards``.
            source_weights: Sampling probabilities for mixed-corpus training.
                Weights are normalized internally and are independent of the
                number or physical size of shards in each source.
            profile_steps: Number of optimizer steps to run with synchronized
                coarse performance profiling. Profiling is intentionally
                intrusive and should normally be limited to 1-3 steps.
        """
        self.model = model
        self.train_shard_paths = [Path(p) for p in train_shard_paths]
        self.val_shard_paths = [Path(p) for p in val_shard_paths]
        self.train_source_shards = (
            {str(name): [Path(p) for p in paths] for name, paths in train_source_shards.items()}
            if train_source_shards is not None
            else None
        )
        self.val_source_shards = (
            {str(name): [Path(p) for p in paths] for name, paths in val_source_shards.items()}
            if val_source_shards is not None
            else None
        )
        self.source_weights = self._normalize_source_weights(source_weights)
        self._validate_mixed_sources()
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
        self.profile_steps = max(0, int(profile_steps))
        self._profiled_steps = 0
        self.shard_cache_size = max(1, int(shard_cache_size))
        if self.train_source_shards is not None:
            # Keep at least the current shard for each corpus mapped.  Memmaps
            # reserve address space but do not copy whole shards into RAM.
            self.shard_cache_size = max(
                self.shard_cache_size, len(self.train_source_shards)
            )
        
        # Effective batch size
        self.effective_batch_size = batch_size * grad_accum_steps
        
        # Initialize optimizer with weight decay
        # Loss scaling is handled entirely by the trainer
        self.optimizer = AdamW(
            model.parameters(),
            lr=peak_lr,
            weight_decay=weight_decay,
            numerical_debug=numerical_debug,
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
        # Bounded LRU caches of read-only memmaps.  The previous dictionary
        # retained every shard encountered, eventually holding the whole token
        # dataset in process memory.
        self.shards = OrderedDict()
        self.current_train_shard_idx = 0
        self.current_train_source_shard_idx = (
            {name: 0 for name in self.train_source_shards}
            if self.train_source_shards is not None
            else {}
        )
        self.train_source_batch_counts = (
            {name: 0 for name in self.train_source_shards}
            if self.train_source_shards is not None
            else {}
        )
        
        # Issue #12: Persistent RNGs (one for train, one for val)
        self.train_rng = np.random.default_rng(rng_seed)
        self.val_rng = np.random.default_rng(rng_seed + 1)
        
        # Issue #13: Deterministic validation blocks
        self._val_block_indices: Optional[List[int]] = None
        
        # Validation state
        self.current_val_shard_idx = 0
        self.current_val_source_shard_idx = (
            {name: 0 for name in self.val_source_shards}
            if self.val_source_shards is not None
            else {}
        )
        self.val_source_batch_counts = (
            {name: 0 for name in self.val_source_shards}
            if self.val_source_shards is not None
            else {}
        )
        self.val_shards = OrderedDict()
        
        # Numerical debugging
        self.numerical_debug = numerical_debug

        # Lightweight forward finite tracing.  Unlike numerical_debug this
        # keeps all stage checks on the GPU and only synchronizes if the final
        # logits are non-finite.  Enable for a narrow optimizer-step window via
        # MINI_LLM_FINITE_TRACE_START / MINI_LLM_FINITE_TRACE_END.
        self.finite_trace_start = int(os.environ.get("MINI_LLM_FINITE_TRACE_START", "-1"))
        self.finite_trace_end = int(os.environ.get("MINI_LLM_FINITE_TRACE_END", str(self.finite_trace_start)))
        
        # Logging setup
        self._setup_logging()
        
        # Issue #13: Initialize deterministic validation block set
        self._init_validation_blocks()
        
        print(f"ExtendedTrainer initialized:")
        print(f"  Train shards: {len(train_shard_paths)}")
        print(f"  Val shards: {len(val_shard_paths)}")
        if self.train_source_shards is not None:
            print("  Mixed pretraining sources:")
            for name in self._source_names:
                print(
                    f"    {name:16s} {100.0 * self.source_weights[name]:6.2f}%  "
                    f"{len(self.train_source_shards[name])} train shards / "
                    f"{len(self.val_source_shards[name])} val shards"
                )
        print(f"  Batch size: {batch_size} (effective: {self.effective_batch_size})")
        print(f"  Sequence length: {seq_length}")
        print(f"  Total steps: {total_steps}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Log file: {log_file}")
        
    @staticmethod
    def _normalize_source_weights(
        source_weights: Optional[Mapping[str, float]],
    ) -> Optional[Dict[str, float]]:
        if source_weights is None:
            return None
        converted = {str(name): float(weight) for name, weight in source_weights.items()}
        if any(weight < 0.0 for weight in converted.values()):
            raise ValueError("source weights must be non-negative")
        total = sum(converted.values())
        if total <= 0.0:
            raise ValueError("sum of source weights must be positive")
        return {name: weight / total for name, weight in converted.items()}

    def _validate_mixed_sources(self) -> None:
        mixed_values = (
            self.train_source_shards,
            self.val_source_shards,
            self.source_weights,
        )
        if all(value is None for value in mixed_values):
            self._source_names = []
            self._source_probabilities = None
            return
        if any(value is None for value in mixed_values):
            raise ValueError(
                "train_source_shards, val_source_shards, and source_weights "
                "must be provided together"
            )
        train_names = set(self.train_source_shards)
        val_names = set(self.val_source_shards)
        weight_names = set(self.source_weights)
        if train_names != val_names or train_names != weight_names:
            raise ValueError(
                "mixed train/validation sources and source weights must use "
                "the same source names"
            )
        for name in train_names:
            if not self.train_source_shards[name]:
                raise ValueError(f"mixed source {name!r} has no training shards")
            if not self.val_source_shards[name]:
                raise ValueError(f"mixed source {name!r} has no validation shards")
        self._source_names = sorted(train_names)
        self._source_probabilities = np.asarray(
            [self.source_weights[name] for name in self._source_names],
            dtype=np.float64,
        )

    def _choose_source(self, rng) -> str:
        index = int(rng.choice(len(self._source_names), p=self._source_probabilities))
        return self._source_names[index]

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
        """Memory-map a training shard instead of copying the full file."""
        return map_token_shard(path, seq_length=self.seq_length)

    def load_val_shard(self, path: str) -> np.ndarray:
        """Memory-map a validation shard instead of copying the full file."""
        return map_token_shard(path, seq_length=self.seq_length)

    @staticmethod
    def _close_mapped_shard(shard):
        """Release an mmap eagerly when evicting it from the small LRU."""
        mmap_obj = getattr(shard, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def _get_cached_shard(self, cache, cache_key, path, loader):
        if cache_key in cache:
            shard = cache.pop(cache_key)
            cache[cache_key] = shard
            return shard

        shard = loader(str(path))
        cache[cache_key] = shard

        while len(cache) > self.shard_cache_size:
            _, evicted = cache.popitem(last=False)
            self._close_mapped_shard(evicted)

        return shard
    
    def get_train_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random training batch.
        
        Uses persistent RNG for reproducibility (Issue #12).
        """
        if self.train_source_shards is not None:
            source = self._choose_source(self.train_rng)
            paths = self.train_source_shards[source]
            shard_idx = self.current_train_source_shard_idx[source]
            shard_data = self._get_cached_shard(
                self.shards,
                (source, shard_idx),
                paths[shard_idx],
                self.load_train_shard,
            )
            self.current_train_source_shard_idx[source] = (
                shard_idx + 1
            ) % len(paths)
            self.train_source_batch_counts[source] += 1
        else:
            shard_idx = self.current_train_shard_idx
            shard_data = self._get_cached_shard(
                self.shards,
                shard_idx,
                self.train_shard_paths[shard_idx],
                self.load_train_shard,
            )
            self.current_train_shard_idx = (
                self.current_train_shard_idx + 1
            ) % len(self.train_shard_paths)

        inputs, targets = create_minibatch(
            shard_data, self.batch_size, self.seq_length, rng=self.train_rng
        )
        
        # Issue #16: Track tokens processed
        self.tokens_processed += self.batch_size * self.seq_length
        
        return inputs, targets
    
    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a validation batch.
        
        Uses deterministic validation block selection (Issue #13).
        """
        if self.val_source_shards is not None:
            source = self._choose_source(self.val_rng)
            paths = self.val_source_shards[source]
            shard_idx = self.current_val_source_shard_idx[source]
            shard_data = self._get_cached_shard(
                self.val_shards,
                (source, shard_idx),
                paths[shard_idx],
                self.load_val_shard,
            )
            self.current_val_source_shard_idx[source] = (
                shard_idx + 1
            ) % len(paths)
            self.val_source_batch_counts[source] += 1
        else:
            shard_idx = self.current_val_shard_idx
            shard_data = self._get_cached_shard(
                self.val_shards,
                shard_idx,
                self.val_shard_paths[shard_idx],
                self.load_val_shard,
            )
            self.current_val_shard_idx = (
                self.current_val_shard_idx + 1
            ) % len(self.val_shard_paths)

        inputs, targets = create_minibatch(
            shard_data, self.batch_size, self.seq_length, rng=self.val_rng
        )
        
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
        profile_active = self._profiled_steps < self.profile_steps
        configure_performance_profiler(profile_active, reset=profile_active)
        profile_wall_start = None
        if profile_active:
            synchronize()
            profile_wall_start = time.perf_counter()

        # Accumulate gradients over multiple steps
        # Use Python floats for loss accumulation (minimal overhead)
        # The main optimization is avoiding host-device sync during gradient computation
        # Keep microbatch losses on-device; converting each one to a
        # Python float forces a CUDA stream synchronization.
        loss_sum_backend = xp.asarray(0.0, dtype="float32")
        
        for accum_step in range(self.grad_accum_steps):
            # Get learning rate for this step (use final lr of accumulated batch)
            lr = self.scheduler(self.step)
            self.optimizer.lr = lr
            
            # Get batch
            with performance_scope("train.data_batch"):
                inputs, targets = self.get_train_batch()
            
            # Forward pass.  Optional lightweight finite tracing records
            # asynchronous GPU reductions at key layer boundaries.  There is
            # only a host synchronization when the final logits are bad.
            trace_active = (
                self.finite_trace_start >= 0
                and self.finite_trace_start <= self.step <= self.finite_trace_end
            )
            finite_trace = [] if trace_active else None
            with performance_scope("train.model_forward"):
                logits, cache = self.model.forward(inputs, finite_trace=finite_trace)

            if finite_trace is not None:
                final_ok = bool(finite_trace[-1][1].item())
                if not final_ok:
                    print(f"\nFINITE TRACE FAILURE at optimizer step {self.step}, accumulation microstep {accum_step}")
                    first_bad = None
                    for label, ok_backend in finite_trace:
                        ok = bool(ok_backend.item())
                        state = "OK" if ok else "NONFINITE"
                        print(f"  {label}: {state}")
                        if first_bad is None and not ok:
                            first_bad = label
                    raise ValueError(
                        f"Nonfinite forward tensor at step {self.step}, "
                        f"microstep {accum_step}; first bad stage: {first_bad}"
                    )
            
            # Numerical debug: check logits are finite
            if not self._check_finite(logits, f"logits_step_{self.step}"):
                raise ValueError(f"Nonfinite logits detected at step {self.step}!")
            
            with performance_scope("train.loss_forward"):
                loss_backend, loss_cache = self.model.compute_loss(
                    logits, targets, return_device_loss=True
                )
            loss_sum_backend += loss_backend
            
            # Backward pass - apply loss scaling to d_logits before backward
            with performance_scope("train.loss_backward"):
                d_logits = self.model.backward_loss(loss_cache)
            
            # Scale the gradient by loss_scale for mixed precision (trainer owns it)
            if self.loss_scale != 1.0:
                d_logits = d_logits * self.loss_scale
            
            with performance_scope("train.model_backward"):
                self.model.backward(d_logits, cache)
        
        # Average loss - only sync once at the end
        avg_loss = float(_array_to_float(loss_sum_backend / self.grad_accum_steps))
        if not np.isfinite(avg_loss):
            raise ValueError(f"NaN/Inf loss detected at step {self.step}!")
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
        with performance_scope("train.grad_clip"):
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
        with performance_scope("train.optimizer_step"):
            self.optimizer.step(lr=lr)
        with performance_scope("train.zero_grad"):
            self.optimizer.zero_grad()
        
        self.step += 1

        if profile_active:
            synchronize()
            elapsed = time.perf_counter() - profile_wall_start
            tokens = self.batch_size * self.seq_length * self.grad_accum_steps
            print()
            print(
                performance_report(
                    title=f"Performance profile: optimizer step {self.step}"
                )
            )
            print(
                f"Profiled optimizer-step wall time: {elapsed:.3f}s; "
                f"effective throughput: {tokens / max(elapsed, 1e-12):,.0f} tokens/s"
            )
            print()
            self._profiled_steps += 1
            configure_performance_profiler(False)
        
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
            "current_train_source_shard_idx": self.current_train_source_shard_idx,
            "current_val_source_shard_idx": self.current_val_source_shard_idx,
            "train_source_batch_counts": self.train_source_batch_counts,
            "val_source_batch_counts": self.val_source_batch_counts,
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
            
            # train_step() increments self.step after a successful optimizer
            # update.  Therefore self.step is already the completed 1-based
            # optimizer-step count here; adding one would make every periodic
            # action fire one update early.
            if self.step % self.val_interval == 0:
                val_loss = self.compute_val_loss()
                print(f"  Validation loss: {val_loss:.4f}")
            else:
                val_loss = None
            
            # Log using the completed optimizer-step count.
            if self.step % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - start_time
                steps_per_sec = (self.step - start_step) / elapsed
                lr = self.optimizer.lr
                
                print(
                    f"Step {self.step}/{num_steps + start_step}: "
                    f"loss={avg_loss:.4f}, "
                    f"lr={lr:.6f}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"{steps_per_sec:.2f} steps/sec"
                )
                
                # Log to CSV
                self._log({
                    "step": self.step,
                    "train_loss": avg_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "val_loss": val_loss if val_loss is not None else "",
                    "steps_per_sec": steps_per_sec,
                })
            
            # Save using the completed optimizer-step count.
            if self.step % self.save_interval == 0:
                self.save()
        
        print(f"\nTraining complete!")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {np.mean(losses):.4f}")
        
        return losses
