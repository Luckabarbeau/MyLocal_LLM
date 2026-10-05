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
from mini_llm.data.token_shards import (
    map_token_shard,
    create_minibatch,
    create_hierarchical_memory_minibatch,
    load_or_build_packed_document_index,
)
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
        eos_token_id: Optional[int] = None,
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
            eos_token_id: Packed-shard EOS marker. Hierarchical-memory training
                uses it to keep every sample inside one coherent document.
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
        self.seq_length = int(seq_length)
        self.memory_context = (
            model.config.memory_context if model.config.memory_context.enabled else None
        )
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        if self.memory_context is not None:
            if self.eos_token_id is None:
                raise ValueError(
                    "hierarchical-memory training requires eos_token_id for "
                    "document-aware packed sampling"
                )
            expected_source = int(self.memory_context.source_input_length)
            if self.seq_length != expected_source:
                raise ValueError(
                    "hierarchical-memory trainer seq_length must equal "
                    f"the configured source_input_length ({expected_source}), got "
                    f"{self.seq_length}"
                )
            self.active_seq_length = int(self.memory_context.active_length)
            self.target_seq_length = int(self.memory_context.target_length)
        else:
            self.active_seq_length = self.seq_length
            self.target_seq_length = self.seq_length
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

        # 0055A: optional synchronized VRAM accounting.  This is deliberately
        # opt-in because memGetInfo()/synchronize() perturb timing.  When
        # enabled, record the maximum live device/pool usage observed at each
        # major training boundary across all accumulation microsteps.
        memory_profile_raw = os.environ.get("MINI_LLM_MEMORY_PROFILE", "0").strip().lower()
        memory_profile_enabled = memory_profile_raw not in {"0", "false", "off", "no", ""}
        memory_profile_steps_raw = os.environ.get("MINI_LLM_MEMORY_PROFILE_STEPS")
        if memory_profile_steps_raw is None:
            self.memory_profile_steps = 1 if memory_profile_enabled else 0
        else:
            self.memory_profile_steps = max(0, int(memory_profile_steps_raw))
        self._memory_profiled_steps = 0

        # 0056: training-only chunked/recomputed vocabulary head.  A value of
        # zero preserves the historical full-logits path used by inference and
        # reference tests.  Positive values bound the number of token rows whose
        # [tokens, vocab] logits exist at once.
        self.lm_head_chunk_tokens = max(
            0, int(os.environ.get("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "0"))
        )

        # 0057: block-level activation checkpoint/recompute.  ``none`` keeps
        # the established full backward caches.  ``block`` stores only each
        # Transformer block input and replays that block once during backward.
        # This is the intended long-context mode for 16k/32k/64k training.
        self.activation_checkpoint = os.environ.get(
            "MINI_LLM_ACTIVATION_CHECKPOINT", "none"
        ).strip().lower()
        if self.activation_checkpoint in {"0", "false", "off", "no", ""}:
            self.activation_checkpoint = "none"
        elif self.activation_checkpoint in {"1", "true", "on", "yes"}:
            self.activation_checkpoint = "block"
        if self.activation_checkpoint not in {"none", "block"}:
            raise ValueError(
                "MINI_LLM_ACTIVATION_CHECKPOINT must be 'none' or 'block'"
            )

        router_diag_raw = os.environ.get(
            "MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS", "0"
        ).strip().lower()
        self.memory_router_diagnostics = router_diag_raw not in {
            "0", "false", "off", "no", ""
        }
        self.memory_router_diagnostics_interval = max(
            1, int(os.environ.get("MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS_INTERVAL", "100"))
        )
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
        self.optimization_parameters = list(model.optimization_parameters())
        optimized_ids = {id(p) for p in self.optimization_parameters}
        self.frozen_parameters = [
            p for p in model.parameters() if id(p) not in optimized_ids
        ]
        # Only a subset of frozen tensors participates in router-only backward.
        # Earlier frozen blocks run inference-style, output-projection backward
        # skips W-grad, and embedding backward is disabled. Clearing every ~500M
        # frozen gradient buffer each step would therefore be a multi-GiB memory
        # write for no mathematical benefit.
        self.frozen_backward_parameters = self.frozen_parameters
        if (
            self.memory_context is not None
            and self.memory_context.integration_mode == "terminal_landmark"
            and self.memory_context.memory_training == "router_only"
        ):
            layer = int(model.memory_attention_layer)
            relevant = []
            for block in model.blocks[layer:]:
                relevant.extend(block.parameters())
            relevant.extend(model.final_norm.parameters())
            self.frozen_backward_parameters = [
                p for p in relevant if id(p) not in optimized_ids
            ]
        self.optimizer = AdamW(
            self.optimization_parameters,
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
        self.document_indices = OrderedDict()
        self.val_document_indices = OrderedDict()
        self.document_index_cache_size = max(8, self.shard_cache_size * 4)
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
        self._setup_memory_metrics_logging()
        self._last_step_memory_token_stats = None
        
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
        if self.memory_context is None:
            print(f"  Sequence length: {self.seq_length}")
        else:
            cfg = self.memory_context
            print("  Causal external memory routing: enabled")
            print("  Document-aware sampling:       enabled (EOS-bounded)")
            if cfg.integration_mode == "routed_prefix":
                print(f"  Total causal horizon:          {cfg.memory_length:,} tokens")
                print(f"  Searchable history:            {cfg.distant_memory_length:,} tokens")
                print(f"  Dense working window:          {cfg.target_length:,} tokens")
                print(f"  Routed historical prefix:      {cfg.retrieved_length:,} tokens")
                print(f"  Active Transformer length:     {cfg.active_length:,}")
                print(f"  Memory block size:             {cfg.block_size:,}")
                print(f"  Searchable memory blocks:      {cfg.searchable_blocks:,}")
                print(f"  Routed blocks:                 {cfg.top_k_blocks:,}")
                print(f"  Router query length:           {cfg.router_query_length:,}")
                print(f"  Router dimension:              {cfg.router_dim:,}")
                print(
                    f"  Retrieval batch probability:  "
                    f"{100.0 * cfg.retrieval_batch_probability:.1f}%"
                )
                print(
                    f"  Router temperature:            {cfg.router_temperature:g} -> "
                    f"{cfg.router_temperature_min:g} over "
                    f"{cfg.router_temperature_anneal_steps:,} steps"
                )
                print(f"  Gumbel exploration:            {cfg.router_gumbel_noise}")
                print(f"  ST surrogate scale:            {cfg.router_surrogate_scale:g}")
            elif cfg.integration_mode == "terminal_landmark":
                print(f"  Total causal horizon:          {cfg.memory_length:,} tokens")
                print(f"  External searchable history:   {cfg.distant_memory_length:,} tokens")
                print(f"  Dense working window:          {cfg.target_length:,} tokens")
                print(f"  Active Transformer length:     {cfg.active_length:,}")
                print(f"  Memory block size:             {cfg.block_size:,}")
                print(f"  Searchable memory blocks:      {cfg.searchable_blocks:,}")
                print(f"  Terminal routed blocks:        {cfg.top_k_blocks:,}")
                print(f"  Terminal exact K/V tokens:     {cfg.retrieved_length:,}")
                print(f"  Router query length:           {cfg.router_query_length:,}")
                print(f"  Router dimension:              {cfg.router_dim:,}")
                print(f"  Memory-aware layer:            {cfg.memory_attention_layer}")
                print(f"  Memory retrieval heads/KV:     {cfg.read_heads}/{cfg.read_kv_heads}")
                print("  Router-supervised rows/sample: 1 (terminal only)")
            else:
                print(f"  Historical memory store:       {cfg.memory_length:,} tokens")
                print(f"  Memory block size:             {cfg.block_size:,}")
                print(f"  Searchable memory blocks:      {cfg.searchable_blocks:,}")
                print(f"  Retrieved blocks/row:          {cfg.top_k_blocks:,}")
                print(f"  Retrieved exact tokens/row:    {cfg.retrieved_length:,}")
                print(f"  Recent router history:         {cfg.recent_length:,}")
                print(f"  Dense training window:         {cfg.target_length:,}")
                print(f"  Active Transformer length:     {cfg.active_length:,}")
                print(f"  Router query length:           {cfg.router_query_length:,}")
                print(f"  Router dimension:              {cfg.router_dim:,}")
                print(f"  Memory reader heads/KV:        {cfg.read_heads}/{cfg.read_kv_heads}")
            print(f"  Source sample input length:    {cfg.source_input_length:,}")
            print(f"  Memory training mode:          {cfg.memory_training}")
            if cfg.memory_training == "router_only":
                print(
                    f"  Optimized memory params:     "
                    f"{sum(int(p.data.size) for p in self.optimization_parameters):,}"
                )
                print("  Backbone backward:           truncated at memory layer")
        if self.lm_head_chunk_tokens > 0:
            print(
                f"  LM head: chunked/recomputed "
                f"({self.lm_head_chunk_tokens} token rows/tile)"
            )
        if self.activation_checkpoint == "block":
            n_blocks = len(self.model.blocks)
            rows = int(batch_size) * int(self.active_seq_length)
            model_itemsize = int(self.model.embedding.W.data.dtype.itemsize)
            residual_itemsize = (
                4
                if n_blocks > 0 and getattr(self.model.blocks[0], "use_fp32_residual", False)
                else model_itemsize
            )
            checkpoint_bytes = rows * int(self.model.config.d_model) * (
                model_itemsize + max(0, n_blocks - 1) * residual_itemsize
            )
            print(
                "  Activations: block checkpoint/recompute "
                f"({n_blocks} block inputs, ~{checkpoint_bytes / (1024 ** 3):.2f} GiB retained)"
            )
        if getattr(self.optimizer, "moments_offloaded", False):
            label = (
                "Optimizer state"
                if getattr(self.optimizer, "master_weights_offloaded", False)
                else "Optimizer moments"
            )
            print(
                f"  {label}: pinned RAM "
                f"({self.optimizer.offload_host_bytes / (1024 ** 3):.2f} GiB host, "
                f"{self.optimizer.offload_stage_bytes / (1024 ** 2):.0f} MiB GPU staging)"
            )
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
    
    @staticmethod
    def _history_diagnostic_edges(history_capacity: int) -> List[int]:
        """Return stable history-length bins clipped to the configured horizon."""
        history_capacity = max(0, int(history_capacity))
        if history_capacity == 0:
            return [0, 1]
        anchors = (0, 4096, 8192, 16384, 32768, 49152, history_capacity)
        edges = sorted({min(history_capacity, max(0, int(value))) for value in anchors})
        if edges[0] != 0:
            edges.insert(0, 0)
        if edges[-1] != history_capacity:
            edges.append(history_capacity)
        if len(edges) == 1:
            edges.append(history_capacity + 1)
        return edges

    def _empty_memory_token_stats(self) -> Optional[Dict[str, Any]]:
        """Create a CPU-only accumulator for document-aware token accounting."""
        if self.memory_context is None:
            return None
        history_capacity = int(self.memory_context.distant_memory_length)
        edges = self._history_diagnostic_edges(history_capacity)
        return {
            "microbatches": 0,
            "samples": 0,
            "valid_history_tokens": 0,
            "valid_source_tokens": 0,
            "valid_active_tokens": 0,
            "supervised_tokens": 0,
            "history_sum": 0,
            "history_min": None,
            "history_max": 0,
            "history_edges": edges,
            "history_hist": [0] * (len(edges) - 1),
        }

    def _accumulate_memory_batch_token_stats(
        self,
        stats: Optional[Dict[str, Any]],
        batch_metadata: Optional[Dict[str, Any]],
        effective_loss_mask,
    ) -> None:
        """Account real same-document tokens without touching the GPU stream.

        Hierarchical-memory tensors have a fixed padded capacity.  The sampler's
        ``history_valid_starts`` and ``target_valid_lengths`` arrays are CPU
        metadata and therefore let us measure semantic token coverage without
        introducing a CUDA synchronization into training.
        """
        if stats is None or self.memory_context is None:
            return

        history_capacity = int(self.memory_context.distant_memory_length)
        target_capacity = int(self.memory_context.target_length)
        metadata = batch_metadata or {}

        starts = np.asarray(
            metadata.get(
                "history_valid_starts",
                np.zeros(self.batch_size, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        target_lengths = np.asarray(
            metadata.get(
                "target_valid_lengths",
                np.full(self.batch_size, target_capacity, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        if starts.shape != (self.batch_size,):
            raise ValueError("history_valid_starts must have shape (batch_size,)")
        if target_lengths.shape != (self.batch_size,):
            raise ValueError("target_valid_lengths must have shape (batch_size,)")

        history_lengths = np.clip(history_capacity - starts, 0, history_capacity)
        target_lengths = np.clip(target_lengths, 0, target_capacity)

        stats["microbatches"] += 1
        stats["samples"] += int(self.batch_size)
        stats["valid_history_tokens"] += int(history_lengths.sum())
        stats["valid_active_tokens"] += int(target_lengths.sum())
        stats["valid_source_tokens"] += int(
            history_lengths.sum() + target_lengths.sum()
        )
        stats["history_sum"] += int(history_lengths.sum())
        batch_min = int(history_lengths.min()) if history_lengths.size else 0
        batch_max = int(history_lengths.max()) if history_lengths.size else 0
        stats["history_min"] = (
            batch_min
            if stats["history_min"] is None
            else min(int(stats["history_min"]), batch_min)
        )
        stats["history_max"] = max(int(stats["history_max"]), batch_max)

        if effective_loss_mask is None:
            supervised = int(target_lengths.sum())
        else:
            supervised = int(np.count_nonzero(np.asarray(effective_loss_mask)))
        stats["supervised_tokens"] += supervised

        edges = np.asarray(stats["history_edges"], dtype=np.int64)
        hist, _ = np.histogram(history_lengths, bins=edges)
        for idx, count in enumerate(hist.tolist()):
            stats["history_hist"][idx] += int(count)

    @staticmethod
    def _merge_memory_token_stats(
        destination: Optional[Dict[str, Any]],
        source: Optional[Dict[str, Any]],
    ) -> None:
        if destination is None or source is None:
            return
        for key in (
            "microbatches",
            "samples",
            "valid_history_tokens",
            "valid_source_tokens",
            "valid_active_tokens",
            "supervised_tokens",
            "history_sum",
        ):
            destination[key] += int(source[key])
        if source["history_min"] is not None:
            destination["history_min"] = (
                int(source["history_min"])
                if destination["history_min"] is None
                else min(int(destination["history_min"]), int(source["history_min"]))
            )
        destination["history_max"] = max(
            int(destination["history_max"]), int(source["history_max"])
        )
        if destination["history_edges"] != source["history_edges"]:
            raise ValueError("cannot merge memory token stats with different bins")
        for idx, count in enumerate(source["history_hist"]):
            destination["history_hist"][idx] += int(count)

    @staticmethod
    def _format_history_bin_label(lower: int, upper: int, is_last: bool) -> str:
        def compact(value: int) -> str:
            if value >= 1024 and value % 1024 == 0:
                return f"{value // 1024}k"
            return f"{value:,}"
        closing = "]" if is_last else ")"
        return f"{compact(lower)}-{compact(upper)}{closing}"

    def _memory_metrics_log_path(self) -> Optional[Path]:
        if self.log_file is None or self.memory_context is None:
            return None
        return self.log_file.with_name(
            f"{self.log_file.stem}.memory{self.log_file.suffix or '.csv'}"
        )

    def _setup_memory_metrics_logging(self) -> None:
        path = self._memory_metrics_log_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        edges = self._history_diagnostic_edges(
            int(self.memory_context.distant_memory_length)
        )
        bin_headers = [
            f"history_bin_{edges[i]}_{edges[i + 1]}"
            for i in range(len(edges) - 1)
        ]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "steps_in_interval", "samples",
                "source_capacity_tokens", "valid_source_tokens",
                "history_capacity_tokens", "valid_history_tokens",
                "active_capacity_tokens", "valid_active_tokens",
                "supervised_tokens", "history_utilization",
                "history_mean_per_sample", "history_min_per_sample",
                "history_max_per_sample", "source_capacity_tok_per_sec",
                "valid_source_tok_per_sec", "valid_history_tok_per_sec",
                "valid_active_tok_per_sec", "supervised_tok_per_sec",
                *bin_headers,
            ])

    def _log_memory_metrics(
        self,
        stats: Optional[Dict[str, Any]],
        *,
        steps_in_interval: int,
        elapsed_seconds: float,
    ) -> None:
        path = self._memory_metrics_log_path()
        if path is None or stats is None or stats["samples"] <= 0:
            return
        history_capacity = int(self.memory_context.distant_memory_length)
        active_capacity = int(self.memory_context.active_length)
        source_capacity = int(self.seq_length)
        samples = int(stats["samples"])
        source_capacity_tokens = samples * source_capacity
        history_capacity_tokens = samples * history_capacity
        active_capacity_tokens = samples * active_capacity
        denom = max(float(elapsed_seconds), 1e-12)
        history_util = (
            float(stats["valid_history_tokens"]) / history_capacity_tokens
            if history_capacity_tokens > 0 else 0.0
        )
        history_mean = float(stats["history_sum"]) / samples
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step, int(steps_in_interval), samples,
                source_capacity_tokens, int(stats["valid_source_tokens"]),
                history_capacity_tokens, int(stats["valid_history_tokens"]),
                active_capacity_tokens, int(stats["valid_active_tokens"]),
                int(stats["supervised_tokens"]), history_util, history_mean,
                int(stats["history_min"] or 0), int(stats["history_max"]),
                source_capacity_tokens / denom,
                int(stats["valid_source_tokens"]) / denom,
                int(stats["valid_history_tokens"]) / denom,
                int(stats["valid_active_tokens"]) / denom,
                int(stats["supervised_tokens"]) / denom,
                *[int(value) for value in stats["history_hist"]],
            ])

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

    def _get_document_index(self, cache, cache_key, path, shard_data):
        if shard_data.ndim != 1:
            return None
        if cache_key in cache:
            index = cache.pop(cache_key)
            cache[cache_key] = index
            return index
        index = load_or_build_packed_document_index(
            path, shard_data, self.eos_token_id, write_sidecar=True
        )
        cache[cache_key] = index
        while len(cache) > self.document_index_cache_size:
            cache.popitem(last=False)
        return index
    
    def get_train_batch(self, return_metadata: bool = False):
        """Get one reproducible training batch.

        Hierarchical-memory batches are EOS/document bounded.  A compact
        sidecar index is built lazily for existing packed shards, so no
        retokenization or dataset regeneration is required.
        """
        if self.train_source_shards is not None:
            source = self._choose_source(self.train_rng)
            paths = self.train_source_shards[source]
            shard_idx = self.current_train_source_shard_idx[source]
            cache_key = (source, shard_idx)
            shard_path = paths[shard_idx]
            shard_data = self._get_cached_shard(
                self.shards, cache_key, shard_path, self.load_train_shard,
            )
            self.current_train_source_shard_idx[source] = (
                shard_idx + 1
            ) % len(paths)
            self.train_source_batch_counts[source] += 1
        else:
            shard_idx = self.current_train_shard_idx
            cache_key = shard_idx
            shard_path = self.train_shard_paths[shard_idx]
            shard_data = self._get_cached_shard(
                self.shards, cache_key, shard_path, self.load_train_shard,
            )
            self.current_train_shard_idx = (
                self.current_train_shard_idx + 1
            ) % len(self.train_shard_paths)

        metadata = None
        if self.memory_context is None:
            inputs, targets = create_minibatch(
                shard_data, self.batch_size, self.seq_length, rng=self.train_rng
            )
        else:
            document_index = self._get_document_index(
                self.document_indices, cache_key, shard_path, shard_data
            )
            routed_prefix_retrieval = None
            min_history_tokens = 0
            if self.memory_context.integration_mode == "routed_prefix":
                routed_prefix_retrieval = bool(
                    self.train_rng.random()
                    < float(self.memory_context.retrieval_batch_probability)
                )
                if routed_prefix_retrieval:
                    # Right-aligned history can begin mid-block. K*B + (B-1)
                    # tokens guarantees K complete block-aligned candidates.
                    min_history_tokens = (
                        self.memory_context.top_k_blocks
                        * self.memory_context.block_size
                        + self.memory_context.block_size
                        - 1
                    )
            elif (
                self.memory_context.integration_mode == "terminal_landmark"
                and self.memory_context.memory_training == "router_only"
            ):
                min_history_tokens = (
                    self.memory_context.min_router_history_blocks
                    * self.memory_context.block_size
                )

            inputs, targets, metadata = create_hierarchical_memory_minibatch(
                shard_data,
                self.batch_size,
                self.memory_context.distant_memory_length,
                self.memory_context.target_length,
                rng=self.train_rng,
                eos_token_id=self.eos_token_id,
                document_index=document_index,
                document_aware=(shard_data.ndim == 1),
                return_metadata=True,
                min_history_tokens=min_history_tokens,
            )
            if routed_prefix_retrieval is not None:
                history_available = (
                    int(self.memory_context.distant_memory_length)
                    - np.asarray(metadata["history_valid_starts"], dtype=np.int64)
                )
                retrieval_eligible = bool(
                    np.all(history_available > 0)
                    and np.all(
                        metadata["target_valid_lengths"]
                        == self.memory_context.target_length
                    )
                )
                metadata["retrieval_requested"] = routed_prefix_retrieval
                metadata["use_retrieval"] = bool(
                    routed_prefix_retrieval and retrieval_eligible
                )

        self.tokens_processed += self.batch_size * self.seq_length
        if return_metadata:
            return inputs, targets, metadata
        return inputs, targets
    
    def get_val_batch(self, return_metadata: bool = False):
        """Get one validation batch using the same document-boundary semantics."""
        if self.val_source_shards is not None:
            source = self._choose_source(self.val_rng)
            paths = self.val_source_shards[source]
            shard_idx = self.current_val_source_shard_idx[source]
            cache_key = (source, shard_idx)
            shard_path = paths[shard_idx]
            shard_data = self._get_cached_shard(
                self.val_shards, cache_key, shard_path, self.load_val_shard,
            )
            self.current_val_source_shard_idx[source] = (
                shard_idx + 1
            ) % len(paths)
            self.val_source_batch_counts[source] += 1
        else:
            shard_idx = self.current_val_shard_idx
            cache_key = shard_idx
            shard_path = self.val_shard_paths[shard_idx]
            shard_data = self._get_cached_shard(
                self.val_shards, cache_key, shard_path, self.load_val_shard,
            )
            self.current_val_shard_idx = (
                self.current_val_shard_idx + 1
            ) % len(self.val_shard_paths)

        metadata = None
        if self.memory_context is None:
            inputs, targets = create_minibatch(
                shard_data, self.batch_size, self.seq_length, rng=self.val_rng
            )
        else:
            document_index = self._get_document_index(
                self.val_document_indices, cache_key, shard_path, shard_data
            )
            inputs, targets, metadata = create_hierarchical_memory_minibatch(
                shard_data,
                self.batch_size,
                self.memory_context.distant_memory_length,
                self.memory_context.target_length,
                rng=self.val_rng,
                eos_token_id=self.eos_token_id,
                document_index=document_index,
                document_aware=(shard_data.ndim == 1),
                return_metadata=True,
                min_history_tokens=(
                    (
                        self.memory_context.top_k_blocks
                        * self.memory_context.block_size
                        + self.memory_context.block_size
                        - 1
                    )
                    if self.memory_context.integration_mode == "routed_prefix"
                    else (
                        self.memory_context.min_router_history_blocks
                        * self.memory_context.block_size
                        if self.memory_context.integration_mode == "terminal_landmark"
                        and self.memory_context.memory_training == "router_only"
                        else 0
                    )
                ),
            )
            if self.memory_context.integration_mode == "routed_prefix":
                history_available = (
                    int(self.memory_context.distant_memory_length)
                    - np.asarray(metadata["history_valid_starts"], dtype=np.int64)
                )
                retrieval_eligible = bool(
                    np.all(history_available > 0)
                    and np.all(
                        metadata["target_valid_lengths"]
                        == self.memory_context.target_length
                    )
                )
                metadata["retrieval_requested"] = True
                metadata["use_retrieval"] = retrieval_eligible

        if return_metadata:
            return inputs, targets, metadata
        return inputs, targets
    
    def _record_memory_snapshot(self, records, label):
        """Record one synchronized CuPy/device memory snapshot for 0055A."""
        if self._memory_profiled_steps >= self.memory_profile_steps:
            return
        if not hasattr(xp, "cuda"):
            return
        synchronize()
        free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
        pool = xp.get_default_memory_pool()
        snapshot = {
            "device_used": int(total_bytes - free_bytes),
            "device_free": int(free_bytes),
            "device_total": int(total_bytes),
            "pool_used": int(pool.used_bytes()),
            "pool_reserved": int(pool.total_bytes()),
        }
        # Keep the highest-device-used occurrence for each boundary.  This
        # naturally captures the worst gradient-accumulation microstep.
        previous = records.get(label)
        if previous is None or snapshot["device_used"] > previous["device_used"]:
            records[label] = snapshot

    @staticmethod
    def _format_memory_gib(value):
        return float(value) / float(1024 ** 3)

    def _print_memory_report(self, records):
        if not records:
            return
        print()
        print(f"VRAM profile: optimizer step {self.step}")
        print("-" * 96)
        print(
            f"{'boundary':34s} {'device used':>12s} {'pool live':>12s} "
            f"{'pool reserved':>14s} {'device free':>12s} {'non-pool':>10s}"
        )
        peak_label = None
        peak_used = -1
        for label, snap in records.items():
            non_pool = max(0, snap["device_used"] - snap["pool_reserved"])
            print(
                f"{label:34s} "
                f"{self._format_memory_gib(snap['device_used']):11.2f}G "
                f"{self._format_memory_gib(snap['pool_used']):11.2f}G "
                f"{self._format_memory_gib(snap['pool_reserved']):13.2f}G "
                f"{self._format_memory_gib(snap['device_free']):11.2f}G "
                f"{self._format_memory_gib(non_pool):9.2f}G"
            )
            if snap["device_used"] > peak_used:
                peak_label = label
                peak_used = snap["device_used"]
        print(
            f"Observed boundary peak: {self._format_memory_gib(peak_used):.2f} GiB "
            f"at {peak_label}. (Boundary sampling; short-lived kernel workspaces "
            "between boundaries can be higher.)"
        )
        print()

    def _configure_routed_prefix_metadata(self, metadata, *, training):
        """Attach 0060A routing mode without changing the sampler contract."""
        if (
            metadata is None
            or self.memory_context is None
            or self.memory_context.integration_mode != "routed_prefix"
        ):
            return metadata
        cfg = self.memory_context
        metadata = dict(metadata)
        if training:
            if "use_retrieval" not in metadata:
                probability = float(cfg.retrieval_batch_probability)
                metadata["use_retrieval"] = bool(
                    self.train_rng.random() < probability
                )
            metadata["router_stochastic"] = True
            metadata["router_gumbel_seed"] = int(
                self.train_rng.integers(0, 2**31 - 1)
            )
            anneal_steps = max(1, int(cfg.router_temperature_anneal_steps))
            progress = min(max(float(self.step) / anneal_steps, 0.0), 1.0)
            metadata["router_temperature"] = (
                float(cfg.router_temperature)
                + progress
                * (float(cfg.router_temperature_min) - float(cfg.router_temperature))
            )
        else:
            # Validation measures the actual inference policy: deterministic
            # retrieval with no Gumbel exploration.  Preserve sampler-derived
            # eligibility so exact-4k/short fragments do not materialize a
            # useless padded 60k history just to discover there is no prefix.
            metadata.setdefault("use_retrieval", True)
            metadata["router_stochastic"] = False
            metadata["router_temperature"] = float(cfg.router_temperature_min)
        return metadata

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
            inputs, targets, batch_metadata = self.get_val_batch(
                return_metadata=True
            )
            batch_metadata = self._configure_routed_prefix_metadata(
                batch_metadata, training=False
            )
            loss_mask = (
                None if batch_metadata is None
                else batch_metadata.get("target_loss_mask")
            )
            loss_mask = self._effective_loss_mask(batch_metadata, loss_mask)
            targets, loss_mask = self._loss_targets_and_mask(targets, loss_mask)
            
            # Forward pass (no gradient tracking needed)
            logits, _ = self.model.forward(
                inputs, memory_metadata=batch_metadata
            )
            loss, _ = self.model.compute_loss(
                logits, targets, loss_mask=loss_mask
            )
            losses.append(loss)  # loss is already a Python float from scalar()
        
        # Compute mean on backend array (only sync once at the end)
        loss_array = xp.asarray(losses, dtype="float32")
        return float(xp.mean(loss_array))
    
    def _effective_loss_mask(self, batch_metadata, loss_mask):
        """Restrict post-training router optimization to one terminal label."""
        if (
            self.memory_context is None
            or self.memory_context.memory_training != "router_only"
        ):
            return loss_mask
        lengths = batch_metadata.get("target_valid_lengths") if batch_metadata else None
        if lengths is None:
            lengths = np.full(self.batch_size, self.target_seq_length, dtype=np.int64)
        mask = np.zeros((self.batch_size, self.target_seq_length), dtype=np.float32)
        for b, length in enumerate(np.asarray(lengths, dtype=np.int64)):
            if length > 0:
                mask[b, int(length) - 1] = 1.0
        return mask

    def _loss_targets_and_mask(self, targets, loss_mask):
        """Match labels to the terminal-only LM-head slice in router-only mode."""
        if (
            self.memory_context is None
            or self.memory_context.memory_training != "router_only"
        ):
            return targets, loss_mask
        # Router-only sampling requires a complete 4k working window, so the
        # causally correct router target is always the final next-token label.
        targets = targets[:, -1:]
        if loss_mask is not None:
            loss_mask = loss_mask[:, -1:]
        return targets, loss_mask

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

        memory_records = {}
        memory_active = self._memory_profiled_steps < self.memory_profile_steps
        step_memory_token_stats = self._empty_memory_token_stats()
        if memory_active:
            self._record_memory_snapshot(memory_records, "step_start")

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
                inputs, targets, batch_metadata = self.get_train_batch(
                    return_metadata=True
                )
            batch_metadata = self._configure_routed_prefix_metadata(
                batch_metadata, training=True
            )
            loss_mask = (
                None if batch_metadata is None
                else batch_metadata.get("target_loss_mask")
            )
            loss_mask = self._effective_loss_mask(batch_metadata, loss_mask)
            targets, loss_mask = self._loss_targets_and_mask(targets, loss_mask)
            self._accumulate_memory_batch_token_stats(
                step_memory_token_stats, batch_metadata, loss_mask
            )
            
            # Forward pass.  0056 can stop before the vocabulary projection so
            # the LM head is evaluated in bounded token tiles instead of one
            # batch-sized [B,T,V] allocation.
            trace_active = (
                self.finite_trace_start >= 0
                and self.finite_trace_start <= self.step <= self.finite_trace_end
            )
            finite_trace = [] if trace_active else None
            chunked_head = self.lm_head_chunk_tokens > 0
            if chunked_head:
                with performance_scope("train.model_forward"):
                    head_input, cache = self.model.forward_body(
                        inputs, finite_trace=finite_trace,
                        activation_checkpoint=(
                            self.activation_checkpoint == "block"
                        ),
                        memory_metadata=batch_metadata,
                    )
            else:
                with performance_scope("train.model_forward"):
                    logits, cache = self.model.forward(
                        inputs, finite_trace=finite_trace,
                        activation_checkpoint=(
                            self.activation_checkpoint == "block"
                        ),
                        memory_metadata=batch_metadata,
                    )
            if memory_active:
                self._record_memory_snapshot(memory_records, "after_model_forward")

            if (
                self.memory_router_diagnostics
                and self.memory_context is not None
                and accum_step == 0
                and self.step % self.memory_router_diagnostics_interval == 0
            ):
                diag = self.model.memory_routing_diagnostics(cache)
                if diag is not None:
                    unique_blocks = int(diag["unique_blocks"])
                    extra = ""
                    if "history_attention_mass_mean" in diag:
                        extra += (
                            f", history_mass="
                            f"{_array_to_float(diag['history_attention_mass_mean']):.3f}"
                        )
                    if "max_block_attention_mean" in diag:
                        extra += (
                            f", max_block_mass="
                            f"{_array_to_float(diag['max_block_attention_mean']):.3f}"
                        )
                    print(
                        "Memory router: "
                        f"unique_blocks={unique_blocks}, "
                        f"valid_routes="
                        f"{100.0 * _array_to_float(diag['valid_route_fraction']):.1f}%, "
                        f"entropy={_array_to_float(diag['entropy_mean']):.3f}, "
                        f"mean_source_distance="
                        f"{_array_to_float(diag['source_distance_mean']):,.0f} tokens"
                        f"{extra}"
                    )

            if not chunked_head:
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
                if not self._check_finite(logits, f"logits_step_{self.step}"):
                    raise ValueError(f"Nonfinite logits detected at step {self.step}!")

                with performance_scope("train.loss_forward"):
                    loss_backend, loss_cache = self.model.compute_loss(
                        logits, targets, loss_mask=loss_mask,
                        return_device_loss=True
                    )
                loss_sum_backend += loss_backend
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_forward")
                del logits
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_logits_release")

                with performance_scope("train.loss_backward"):
                    d_logits = self.model.backward_loss(loss_cache)
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_backward")
                if self.loss_scale != 1.0:
                    d_logits = d_logits * self.loss_scale
                with performance_scope("train.model_backward"):
                    self.model.backward(d_logits, cache)
                del d_logits
            else:
                # The chunked head owns projection+CE.  No full logits tensor or
                # persistent probability cache is created.  Only the scalar loss
                # and compact hidden representation survive forward.
                with performance_scope("train.loss_forward"):
                    target_slice = cache.get("target_slice")
                    loss_backend, loss_cache = self.model.chunked_lm_head_loss_forward(
                        head_input,
                        targets,
                        chunk_tokens=self.lm_head_chunk_tokens,
                        loss_mask=loss_mask,
                        return_device_loss=True,
                        finite_trace=finite_trace,
                        target_slice=target_slice,
                    )
                loss_sum_backend += loss_backend
                if target_slice is not None:
                    # The loss cache owns a compact copy of target hidden rows;
                    # release the full active final-normalized output now.
                    del head_input
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_forward")
                    self._record_memory_snapshot(memory_records, "after_logits_release")

                if finite_trace is not None:
                    final_ok = bool(finite_trace[-1][1].item())
                    if not final_ok:
                        first_bad = None
                        print(f"\nFINITE TRACE FAILURE at optimizer step {self.step}, accumulation microstep {accum_step}")
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

                with performance_scope("train.loss_backward"):
                    dx_head = self.model.chunked_lm_head_backward(
                        loss_cache, grad_scale=self.loss_scale
                    )
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_backward")
                with performance_scope("train.model_backward"):
                    self.model.backward_body(dx_head, cache)
                del dx_head
                if target_slice is None:
                    del head_input

            if memory_active:
                self._record_memory_snapshot(memory_records, "after_model_backward")

            # Critical 0055A lifetime cleanup: Python evaluates the RHS of the
            # next ``logits, cache = model.forward(...)`` before replacing the
            # old locals.  Without these deletes, the previous microbatch's
            # entire backward cache and CE gradient can stay alive during the
            # next microbatch forward, artificially doubling activation peak.
            del cache, loss_cache, inputs, targets, batch_metadata, loss_mask
            if memory_active:
                self._record_memory_snapshot(memory_records, "after_microbatch_release")

        if memory_active:
            self._record_memory_snapshot(memory_records, "after_grad_accum")

        # Preserve exact CPU-side document coverage for train() diagnostics.
        self._last_step_memory_token_stats = step_memory_token_stats

        # Average loss - only sync once at the end
        avg_loss = float(_array_to_float(loss_sum_backend / self.grad_accum_steps))
        if not np.isfinite(avg_loss):
            raise ValueError(f"NaN/Inf loss detected at step {self.step}!")
        # Scale down gradients by loss_scale to cancel out the scaling
        if self.loss_scale != 1.0:
            for p in self.optimization_parameters:
                if p.grad is not None:
                    p.grad[...] = p.grad / self.loss_scale
        
        # Divide gradients by grad_accum_steps for proper averaging
        if self.grad_accum_steps > 1:
            for p in self.optimization_parameters:
                if p.grad is not None:
                    p.grad[...] = p.grad / self.grad_accum_steps
        
        # Global gradient clipping - returns backend array norm, no sync
        # Now returns (norm_backend, scale, is_finite) tuple
        with performance_scope("train.grad_clip"):
            grad_norm_backend, grad_scale, is_finite = clip_grad_global_norm(
                self.optimization_parameters, max_norm=self.grad_clip
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
        if memory_active:
            self._record_memory_snapshot(memory_records, "after_optimizer_step")
        refresh_backbone_buffers = not (
            self.memory_context is not None
            and self.memory_context.memory_training == "router_only"
        )
        if refresh_backbone_buffers and hasattr(self.model, "refresh_compute_buffers"):
            with performance_scope("train.refresh_compute_buffers"):
                self.model.refresh_compute_buffers()
        with performance_scope("train.zero_grad"):
            self.optimizer.zero_grad()
            # Handwritten backward still computes some input gradients through
            # the frozen terminal block in router-only mode. Clear any incidental
            # frozen parameter accumulation once per optimizer step so it cannot
            # grow across the post-training run.
            if self.frozen_backward_parameters:
                for p in self.frozen_backward_parameters:
                    if p.grad is not None:
                        p.grad[...] = 0
        if memory_active:
            self._record_memory_snapshot(memory_records, "after_zero_grad")
        
        self.step += 1

        if memory_active:
            self._print_memory_report(memory_records)
            self._memory_profiled_steps += 1

        if profile_active:
            synchronize()
            elapsed = time.perf_counter() - profile_wall_start
            source_tokens = self.batch_size * self.seq_length * self.grad_accum_steps
            active_tokens = (
                self.batch_size * self.active_seq_length * self.grad_accum_steps
            )
            supervised_per_sequence = (
                1
                if self.memory_context is not None
                and self.memory_context.memory_training == "router_only"
                else self.target_seq_length
            )
            target_tokens = (
                self.batch_size * supervised_per_sequence * self.grad_accum_steps
            )
            print()
            print(
                performance_report(
                    title=f"Performance profile: optimizer step {self.step}"
                )
            )
            if self.memory_context is None:
                print(
                    f"Profiled optimizer-step wall time: {elapsed:.3f}s; "
                    f"effective throughput: "
                    f"{source_tokens / max(elapsed, 1e-12):,.0f} tokens/s"
                )
            else:
                denom = max(elapsed, 1e-12)
                print(f"Profiled optimizer-step wall time: {elapsed:.3f}s")
                print(f"  target capacity tokens/s: {target_tokens / denom:,.0f}")
                print(f"  active capacity tokens/s: {active_tokens / denom:,.0f}")
                print(f"  source capacity tokens/s: {source_tokens / denom:,.0f}")
                if step_memory_token_stats is not None:
                    print(
                        f"  valid source tokens/s:    "
                        f"{step_memory_token_stats['valid_source_tokens'] / denom:,.0f}"
                    )
                    print(
                        f"  valid history tokens/s:   "
                        f"{step_memory_token_stats['valid_history_tokens'] / denom:,.0f}"
                    )
                    print(
                        f"  supervised tokens/s:      "
                        f"{step_memory_token_stats['supervised_tokens'] / denom:,.0f}"
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
                for i, p in enumerate(self.optimizer.parameters)
            },
            "m": {
                p.name: self.optimizer.m[i]
                for i, p in enumerate(self.optimizer.parameters)
            },
            "v": {
                p.name: self.optimizer.v[i]
                for i, p in enumerate(self.optimizer.parameters)
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
        # 0053F: precise per-optimizer-step timing. train_step() returns only
        # after grad_norm is materialized on the host, so the CUDA stream has
        # already been synchronized at this boundary; perf_counter therefore
        # measures the completed optimizer step without adding another sync.
        recent_step_seconds = []
        memory_interval_stats = self._empty_memory_token_stats()
        memory_interval_seconds = 0.0
        memory_interval_steps = 0
        source_tokens_per_optimizer_step = (
            self.batch_size * self.seq_length * self.grad_accum_steps
        )
        if self.memory_context is not None:
            active_tokens_per_optimizer_step = (
                self.batch_size
                * int(self.memory_context.active_length)
                * self.grad_accum_steps
            )
            supervised_rows = (
                1
                if self.memory_context.memory_training == "router_only"
                else int(self.memory_context.target_length)
            )
            target_tokens_per_optimizer_step = (
                self.batch_size * supervised_rows * self.grad_accum_steps
            )
        else:
            active_tokens_per_optimizer_step = source_tokens_per_optimizer_step
            target_tokens_per_optimizer_step = source_tokens_per_optimizer_step
        
        for step in range(num_steps):
            # Train step
            precise_step_start = time.perf_counter()
            loss, grad_norm = self.train_step()
            precise_step_seconds = time.perf_counter() - precise_step_start
            recent_step_seconds.append(precise_step_seconds)
            if self.memory_context is not None:
                self._merge_memory_token_stats(
                    memory_interval_stats, self._last_step_memory_token_stats
                )
                memory_interval_seconds += precise_step_seconds
                memory_interval_steps += 1
            if len(recent_step_seconds) > 10:
                del recent_step_seconds[0]
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
                precise_source_tokens_per_sec = (
                    source_tokens_per_optimizer_step
                    / max(precise_step_seconds, 1e-12)
                )
                rolling_step_seconds = float(np.mean(recent_step_seconds))
                rolling_source_tokens_per_sec = (
                    source_tokens_per_optimizer_step
                    / max(rolling_step_seconds, 1e-12)
                )
                precise_active_tokens_per_sec = (
                    active_tokens_per_optimizer_step
                    / max(precise_step_seconds, 1e-12)
                )
                precise_target_tokens_per_sec = (
                    target_tokens_per_optimizer_step
                    / max(precise_step_seconds, 1e-12)
                )

                if self.memory_context is not None:
                    step_stats = self._last_step_memory_token_stats
                    if step_stats is not None and step_stats["samples"] > 0:
                        precise_valid_source_tokens_per_sec = (
                            step_stats["valid_source_tokens"]
                            / max(precise_step_seconds, 1e-12)
                        )
                        precise_valid_history_tokens_per_sec = (
                            step_stats["valid_history_tokens"]
                            / max(precise_step_seconds, 1e-12)
                        )
                        precise_valid_active_tokens_per_sec = (
                            step_stats["valid_active_tokens"]
                            / max(precise_step_seconds, 1e-12)
                        )
                        precise_supervised_tokens_per_sec = (
                            step_stats["supervised_tokens"]
                            / max(precise_step_seconds, 1e-12)
                        )
                        history_capacity_tokens = (
                            step_stats["samples"]
                            * int(self.memory_context.distant_memory_length)
                        )
                        history_utilization = (
                            step_stats["valid_history_tokens"]
                            / max(history_capacity_tokens, 1)
                        )
                        history_mean = (
                            step_stats["history_sum"] / step_stats["samples"]
                        )
                    else:
                        precise_valid_source_tokens_per_sec = 0.0
                        precise_valid_history_tokens_per_sec = 0.0
                        precise_valid_active_tokens_per_sec = 0.0
                        precise_supervised_tokens_per_sec = 0.0
                        history_utilization = 0.0
                        history_mean = 0.0
                    throughput_text = (
                        # ``source_tok/s`` is intentionally retained as the
                        # historical fixed-capacity metric for compatibility.
                        f"source_tok/s={precise_source_tokens_per_sec:,.0f}, "
                        f"valid_source_tok/s={precise_valid_source_tokens_per_sec:,.0f}, "
                        f"valid_history_tok/s={precise_valid_history_tokens_per_sec:,.0f}, "
                        f"active_tok/s={precise_active_tokens_per_sec:,.0f}, "
                        f"valid_active_tok/s={precise_valid_active_tokens_per_sec:,.0f}, "
                        f"supervised_tok/s={precise_supervised_tokens_per_sec:,.0f}, "
                        f"history/sample={history_mean:,.0f}, "
                        f"history_util={100.0 * history_utilization:.1f}%, "
                        f"source_tok/s_10={rolling_source_tokens_per_sec:,.0f}"
                    )
                else:
                    throughput_text = (
                        f"tok/s={precise_source_tokens_per_sec:,.0f}, "
                        f"tok/s_10={rolling_source_tokens_per_sec:,.0f}"
                    )

                print(
                    f"Step {self.step}/{num_steps + start_step}: "
                    f"loss={avg_loss:.4f}, "
                    f"lr={lr:.6f}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"{steps_per_sec:.2f} steps/sec, "
                    f"step_time={precise_step_seconds * 1000.0:.1f} ms, "
                    f"{throughput_text}"
                )

                if (
                    self.memory_context is not None
                    and memory_interval_stats is not None
                    and memory_interval_stats["samples"] > 0
                ):
                    samples = int(memory_interval_stats["samples"])
                    history_capacity_tokens = (
                        samples * int(self.memory_context.distant_memory_length)
                    )
                    interval_history_util = (
                        memory_interval_stats["valid_history_tokens"]
                        / max(history_capacity_tokens, 1)
                    )
                    interval_history_mean = (
                        memory_interval_stats["history_sum"] / samples
                    )
                    edges = memory_interval_stats["history_edges"]
                    hist_total = max(sum(memory_interval_stats["history_hist"]), 1)
                    bin_parts = []
                    for idx, count in enumerate(memory_interval_stats["history_hist"]):
                        label = self._format_history_bin_label(
                            int(edges[idx]), int(edges[idx + 1]),
                            idx == len(memory_interval_stats["history_hist"]) - 1,
                        )
                        bin_parts.append(
                            f"{label}={100.0 * int(count) / hist_total:.1f}%"
                        )
                    print(
                        "  Memory coverage "
                        f"({memory_interval_steps} steps/{samples} samples): "
                        f"history/sample mean={interval_history_mean:,.0f}, "
                        f"min={int(memory_interval_stats['history_min'] or 0):,}, "
                        f"max={int(memory_interval_stats['history_max']):,}, "
                        f"util={100.0 * interval_history_util:.1f}%"
                    )
                    print("  History distribution: " + ", ".join(bin_parts))
                    self._log_memory_metrics(
                        memory_interval_stats,
                        steps_in_interval=memory_interval_steps,
                        elapsed_seconds=memory_interval_seconds,
                    )

                # Log to CSV
                self._log({
                    "step": self.step,
                    "train_loss": avg_loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "val_loss": val_loss if val_loss is not None else "",
                    "steps_per_sec": steps_per_sec,
                    "step_time_ms": precise_step_seconds * 1000.0,
                    # Keep the historical CSV/API key source-based for backward
                    # compatibility.  Dense-memory runs print all three rates.
                    "tokens_per_sec": precise_source_tokens_per_sec,
                    "tokens_per_sec_rolling_10": rolling_source_tokens_per_sec,
                })
                if self.memory_context is not None:
                    memory_interval_stats = self._empty_memory_token_stats()
                    memory_interval_seconds = 0.0
                    memory_interval_steps = 0
            
            # Save using the completed optimizer-step count.
            if self.step % self.save_interval == 0:
                self.save()
        
        print(f"\nTraining complete!")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {np.mean(losses):.4f}")
        
        return losses
