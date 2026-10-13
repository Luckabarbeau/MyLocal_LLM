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
import hashlib
import csv
import gc
import math
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
from mini_llm.data.streaming_sources import StreamingTokenCache
from mini_llm.data.token_shards import (
    map_token_shard,
    create_minibatch,
    create_hierarchical_memory_minibatch,
    load_or_build_packed_document_index,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import (
    AdamW, DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2, DEFAULT_ADAM_EPS,
)
from mini_llm.optim.hybrid_muon import HybridMuonAdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm, _array_to_float
from mini_llm.optim.schedule import WarmupCosineSchedule
from mini_llm.backend import synchronize
from mini_llm.diagnostics.gradient_noise import GradientNoiseMonitor
from mini_llm.diagnostics.source_metrics import (
    batch_coverage, empty_record, merge_record, row as source_metrics_row,
    write_source_rows,
)
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
        gradient_accumulation_weighting: str = "microbatch",
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
        progressive_growth_steps: Optional[List[int]] = None,
        gradient_noise_monitor: bool = False,
        gradient_noise_interval: int = 100,
        gradient_noise_output: str = "./logs/gradient_noise.csv",
        gradient_noise_accum_points: str = "8,16,32",
        gradient_noise_parameter_groups: str = "attention,moe,moe_router,memory,head",
        adamw_update_monitor: bool = False,
        adamw_update_output: str = "./logs/adamw_update.csv",
        adam_beta1: float = DEFAULT_ADAM_BETA1,
        adam_beta2: float = DEFAULT_ADAM_BETA2,
        adam_eps: float = DEFAULT_ADAM_EPS,
        optimizer_kind: str = "adamw",
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_matmul_precision: str = "bf16",
        stream_specs=None,
        stream_tokenizer=None,
        stream_segment_tokens: int = 524288,
        stream_reuse_batches: int = 64,
        stream_cache_dir: Optional[str] = None,
        local_mixture_manifest_sha256: Optional[str] = None,
        val_per_source_steps: int = 0,
        val_source_seed: int = 78078,
        val_representative_steps: int = 0,
        val_representative_seed: int = 78078,
        val_fixed_mixed_steps: int = 0,
        val_fixed_mixed_seed: int = 78078,
        training_objective=None,
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
            gradient_accumulation_weighting: microbatch or supervised_tokens
            warmup_steps: Warmup steps for learning rate schedule
            total_steps: Total training steps
            peak_lr: Peak learning rate
            grad_clip: Gradient clipping norm
            weight_decay: Weight decay coefficient for AdamW
            checkpoint_dir: Directory for saving checkpoints (None = no save)
            log_file: Path to CSV log file (None = no logging)
            val_interval: Steps between validation checks
            val_steps: Number of validation steps per check
            val_per_source_steps: Optional fixed held-out batches per local source; zero disables
            val_source_seed: Fixed validation seed (independent of training RNG)
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
            progressive_growth_steps: Completed optimizer-step counts at which
                the next Transformer block becomes active before the following
                optimizer update. Example ``[1000, 3000]`` trains the initial
                depth for 1000 updates, then grows twice deterministically.
        """
        self.model = model
        # None is the historical/optimized AR objective, with identical calls
        # and runtime defaults. Non-AR objective must share this model object.
        if training_objective is not None:
            from mini_llm.block_proposal.objective import BlockProposalObjective
            if (not isinstance(training_objective, BlockProposalObjective)
                    or training_objective.model is not model):
                raise ValueError("training_objective must be a BlockProposalObjective bound to model")
        self.training_objective = training_objective
        self._last_step_block_proposal_metrics = None
        self._last_block_validation_metrics = None
        self._block_proposal_supervised_tokens = 0
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
        self.local_mixture_manifest_sha256 = local_mixture_manifest_sha256
        self.streaming_cache = None
        self.train_source_supervised_tokens = {}
        # 0078A: new counters start when this patch first runs; historical
        # train_source_supervised_tokens above remains backward compatible.
        self.source_metrics = {}
        self.source_metrics_since_step = None
        self._source_metrics_interval = {}
        self._last_step_source_metrics = {}
        self._last_step_supervised_tokens = 0.0
        self._last_step_supervised_utilization = 0.0
        # 0078F: actual execution modes, read from the forward cache for every
        # accumulation microbatch (not one router-diagnostic snapshot).
        self.routing_modes_since_step = None
        self.routing_modes_total = self._new_routing_mode_stats()
        self._routing_modes_interval = self._new_routing_mode_stats()
        self._last_step_routing_modes = self._new_routing_mode_stats()
        # Compact cryptographic chain over the *actual* sampled input/target
        # windows. Records their order without retaining private raw text.
        self.sample_trace_sha256 = "00" * 32
        self.sample_trace_microbatches = 0
        if stream_specs:
            if self.train_source_shards is None:
                raise ValueError("streaming requires existing mixed local training shards")
            if stream_tokenizer is None:
                raise ValueError("streaming requires the checkpoint tokenizer")
            overlap = set(self.train_source_shards) & {spec.name for spec in stream_specs}
            if overlap:
                raise ValueError("stream and local source names overlap: " + str(overlap))
            self.streaming_cache = StreamingTokenCache(
                stream_specs, stream_tokenizer, segment_tokens=stream_segment_tokens,
                reuse_batches=stream_reuse_batches, cache_dir=stream_cache_dir,
                checkpoint_dir=checkpoint_dir,
            )
            local_fraction = 1.0 - sum(spec.weight for spec in stream_specs)
            self._train_source_names = self._source_names + [spec.name for spec in stream_specs]
            self._train_source_probabilities = np.asarray(
                [float(self.source_weights[name]) * local_fraction for name in self._source_names]
                + [spec.weight for spec in stream_specs], dtype=np.float64,
            )
            self._train_source_probabilities /= self._train_source_probabilities.sum()
        else:
            self._train_source_names = self._source_names
            self._train_source_probabilities = self._source_probabilities
        self.batch_size = batch_size
        self.seq_length = int(seq_length)
        self.memory_context = (
            model.config.memory_context if model.config.memory_context.enabled else None
        )
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        raw_growth_steps = [] if progressive_growth_steps is None else list(progressive_growth_steps)
        self.progressive_growth_steps = tuple(int(x) for x in raw_growth_steps)
        if any(x <= 0 for x in self.progressive_growth_steps):
            raise ValueError("progressive growth steps must be positive")
        if tuple(sorted(set(self.progressive_growth_steps))) != self.progressive_growth_steps:
            raise ValueError("progressive growth steps must be unique and strictly increasing")
        if self.progressive_growth_steps and not getattr(
            model, "progressive_depth_enabled", False
        ):
            raise ValueError(
                "progressive growth steps require config.progressive_depth=True"
            )
        if getattr(model, "progressive_depth_enabled", False):
            remaining = int(model.max_layers) - int(model.config.progressive_initial_layers)
            if len(self.progressive_growth_steps) > remaining:
                raise ValueError(
                    "progressive growth schedule contains more activations than "
                    "the final model depth permits"
                )
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
        if gradient_accumulation_weighting not in {"microbatch", "supervised_tokens"}:
            raise ValueError("gradient_accumulation_weighting must be microbatch or supervised_tokens")
        if (self.training_objective is not None
                and gradient_accumulation_weighting != "supervised_tokens"):
            raise ValueError(
                "0080 requires gradient_accumulation_weighting='supervised_tokens' "
                "so every unique valid target has equal accumulated gradient weight"
            )
        self.gradient_accumulation_weighting = gradient_accumulation_weighting
        self.grad_clip = grad_clip  # Store grad_clip parameter
        self.loss_scale = loss_scale  # Store loss scale (trainer owns it)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_file = Path(log_file) if log_file else None
        self.val_interval = val_interval
        self.val_steps = val_steps
        self.val_per_source_steps = int(val_per_source_steps)
        self.val_source_seed = int(val_source_seed)
        self.val_representative_steps = int(val_representative_steps)
        self.val_representative_seed = int(val_representative_seed)
        self.val_fixed_mixed_steps = int(val_fixed_mixed_steps)
        self.val_fixed_mixed_seed = int(val_fixed_mixed_seed)
        if self.val_fixed_mixed_steps < 0:
            raise ValueError("val_fixed_mixed_steps must be nonnegative")
        if self.val_fixed_mixed_steps and self.val_source_shards is None:
            raise ValueError("fixed mixed validation requires local validation sources")
        if self.val_representative_steps < 0:
            raise ValueError("val_representative_steps must be nonnegative")
        if self.val_representative_steps and self.val_source_shards is None:
            raise ValueError("representative validation requires mixed local validation shards")
        if self.val_per_source_steps < 0:
            raise ValueError("val_per_source_steps must be nonnegative")
        if self.val_per_source_steps and self.val_source_shards is None:
            raise ValueError("per-source validation requires mixed local validation shards")
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
        # 0078I: profile every N optimizer updates, rather than profiling only
        # the first N updates of a run. 1 retains the original behavior.
        self.memory_profile_interval = int(
            os.environ.get("MINI_LLM_MEMORY_PROFILE_INTERVAL", "1")
        )
        if self.memory_profile_interval < 1:
            raise ValueError("MINI_LLM_MEMORY_PROFILE_INTERVAL must be >= 1")
        # Frequent inexpensive stage snapshots, separate from the heavier
        # synchronized per-microbatch 0055A profiler.
        self.gpu_memory_audit_train_interval = int(
            os.environ.get("MINI_LLM_GPU_MEMORY_AUDIT_TRAIN_INTERVAL", "0")
        )
        if self.gpu_memory_audit_train_interval < 0:
            raise ValueError("MINI_LLM_GPU_MEMORY_AUDIT_TRAIN_INTERVAL must be >= 0")
        self._gpu_memory_history_shapes = {}
        # 0078H: read-only CUDA memory accounting at validation/checkpoint
        # boundaries.  Optional post-validation trimming releases only idle
        # CuPy pool blocks; it never touches live arrays or changes training.
        self.gpu_memory_audit = os.environ.get(
            "MINI_LLM_GPU_MEMORY_AUDIT", "0"
        ).strip().lower() in {"1", "true", "on", "yes"}
        self.gpu_memory_trim_after_validation = os.environ.get(
            "MINI_LLM_GPU_MEMORY_TRIM_AFTER_VALIDATION", "0"
        ).strip().lower() in {"1", "true", "on", "yes"}
        self.gpu_memory_trim_after_checkpoint = os.environ.get(
            "MINI_LLM_GPU_MEMORY_TRIM_AFTER_CHECKPOINT", "0"
        ).strip().lower() in {"1", "true", "on", "yes"}
        # 0078J: opt-in GC/allocator isolation probe, executed only after
        # validation has returned and all training microbatches have completed.
        # Do not run cyclic GC inside forward/backward or accumulation loops.
        self.gpu_memory_gc_probe = os.environ.get(
            "MINI_LLM_GPU_MEMORY_GC_PROBE", "0"
        ).strip().lower() in {"1", "true", "on", "yes"}
        # Read-only inventory of known GPU caches (RoPE, masks, router metadata,
        # packed QKV). Separate CSV preserves the schema of the 0078H audit.
        self.gpu_memory_cache_audit = os.environ.get(
            "MINI_LLM_GPU_MEMORY_CACHE_AUDIT", "0"
        ).strip().lower() in {"1", "true", "on", "yes"}

        # 0056/0060B3: chunked/recomputed vocabulary head for both training
        # and validation.  A value of zero preserves the historical full-logits
        # path used by inference and reference tests.  Positive values bound the
        # number of token rows whose [tokens, vocab] logits exist at once.
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
        self.optimization_parameters = list(
            model.optimization_parameters() if self.training_objective is None
            else self.training_objective.parameters()
        )
        if self.training_objective is not None and optimizer_kind != "adamw":
            raise ValueError("0080M currently validates shared AdamW only")
        if optimizer_kind not in {"adamw", "hybrid-muon"}:
            raise ValueError("optimizer_kind must be adamw or hybrid-muon")
        if optimizer_kind == "hybrid-muon" and adamw_update_monitor:
            raise ValueError("0074B AdamW counterfactual diagnostics require pure AdamW; "
                             "disable --adamw-update-monitor for hybrid Muon")
        self.optimizer = (
            HybridMuonAdamW(
                self.optimization_parameters,
                lr=peak_lr, beta1=adam_beta1, beta2=adam_beta2, eps=adam_eps,
                weight_decay=weight_decay, numerical_debug=numerical_debug,
                muon_momentum=muon_momentum, muon_ns_steps=muon_ns_steps,
                muon_matmul_precision=muon_matmul_precision,
            ) if optimizer_kind == "hybrid-muon" else AdamW(
                self.optimization_parameters,
                lr=peak_lr, beta1=adam_beta1, beta2=adam_beta2,
                eps=adam_eps, weight_decay=weight_decay,
                numerical_debug=numerical_debug,
            )
        )
        self._refresh_parameter_partitions()
        self.gradient_noise = (
            GradientNoiseMonitor(
                self.optimization_parameters,
                accum_steps=self.grad_accum_steps,
                accum_points=gradient_noise_accum_points,
                categories=gradient_noise_parameter_groups,
                output=gradient_noise_output,
                interval=gradient_noise_interval,
                loss_scale=self.loss_scale,
                adamw_optimizer=self.optimizer if adamw_update_monitor else None,
                adamw_output=adamw_update_output if adamw_update_monitor else None,
                grad_clip=self.grad_clip,
            ) if (gradient_noise_monitor or adamw_update_monitor) else None
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
        if self.streaming_cache is not None:
            print("  Remote training streams (validation remains local):")
            for spec in self.streaming_cache.specs.values():
                print(f"    {spec.name}: {spec.dataset}@{spec.revision} ({spec.weight:.1%})")
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
        
    def _refresh_parameter_partitions(self):
        """Refresh active/frozen parameter views after progressive growth."""
        self.optimization_parameters = list(self.optimizer.parameters)
        optimized_ids = {id(p) for p in self.optimization_parameters}
        self.frozen_parameters = [
            p for p in self.model.parameters() if id(p) not in optimized_ids
        ]

        # Only a subset of frozen tensors participates in router-only backward.
        # Progressive depth is deliberately not enabled with terminal-landmark
        # memory in 0064B, but preserve the established router-only behavior.
        self.frozen_backward_parameters = self.frozen_parameters
        if (
            self.memory_context is not None
            and self.memory_context.integration_mode == "terminal_landmark"
            and self.memory_context.memory_training == "router_only"
        ):
            layer = int(self.model.memory_attention_layer)
            relevant = []
            for block in self.model.blocks[layer:]:
                relevant.extend(block.parameters())
            relevant.extend(self.model.final_norm.parameters())
            self.frozen_backward_parameters = [
                p for p in relevant if id(p) not in optimized_ids
            ]

    def activate_next_progressive_layer(self):
        """Activate one random zero-gated block and extend Adam in place."""
        if not getattr(self.model, "progressive_depth_enabled", False):
            raise RuntimeError("progressive depth is disabled for this model")
        before_ids = {id(p) for p in self.optimizer.parameters}
        layer_idx = self.model.activate_next_layer()
        if layer_idx is None:
            return None

        new_params = [
            p for p in self.model.optimization_parameters()
            if id(p) not in before_ids
        ]
        added = self.optimizer.add_parameters(new_params)
        if added != len(new_params):
            raise RuntimeError("optimizer did not add every newly active parameter")
        self._refresh_parameter_partitions()
        if self.gradient_noise is not None:
            self.gradient_noise.refresh(self.optimization_parameters)
        block = self.model.blocks[layer_idx]
        print(
            "Progressive depth: activated layer "
            f"{layer_idx + 1}/{self.model.max_layers} with "
            f"{added} fresh optimizer tensors; "
            f"alpha_attn={float(block.alpha_attn.data):.6g}, "
            f"alpha_mlp={float(block.alpha_mlp.data):.6g}"
        )
        return layer_idx

    def _apply_scheduled_progressive_growth(self):
        """Make the explicit growth schedule idempotent across resume."""
        if not getattr(self.model, "progressive_depth_enabled", False):
            return
        initial = int(self.model.config.progressive_initial_layers)
        due = sum(step <= int(self.step) for step in self.progressive_growth_steps)
        target = min(int(self.model.max_layers), initial + due)
        while int(self.model.active_layers) < target:
            if self.activate_next_progressive_layer() is None:
                break

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

    def _choose_train_source(self, rng) -> str:
        if self.streaming_cache is None:
            return self._choose_source(rng)
        index = int(rng.choice(len(self._train_source_names), p=self._train_source_probabilities))
        return self._train_source_names[index]

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

    def _source_metrics_log_path(self) -> Optional[Path]:
        if self.log_file is None:
            return None
        return self.log_file.with_name(
            f"{self.log_file.stem}.sources{self.log_file.suffix or '.csv'}"
        )

    def restore_source_metrics(self, state):
        """Restore observational counters without backfilling older checkpoints."""
        if not state:
            return
        if state.get("version") != 1:
            raise ValueError("unsupported source metrics checkpoint version")
        self.source_metrics_since_step = state.get("since_step")
        self.source_metrics = {}
        for name, saved in state.get("sources", {}).items():
            record = empty_record()
            record.update(saved)
            self.source_metrics[str(name)] = record
        self._source_metrics_interval = {}

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
    
    def _gpu_memory_boundary(self, phase: str, *, trim_idle_pool=False):
        """0078H: diagnose live allocations versus retained CuPy pool blocks.

        Collect at safe training/validation boundaries only; the optional
        trim is *not* a remedy for live-reference leaks.  No GPU behavior is
        changed with the default (both environment flags disabled).
        """
        if not (self.gpu_memory_audit or trim_idle_pool):
            return None
        get_pool = getattr(xp, "get_default_memory_pool", None)
        if get_pool is None:
            return None
        pool = get_pool()
        # Synchronize only at infrequent checkpoints; without this, temporary
        # allocations still used by queued kernels can confuse the diagnosis.
        synchronize()
        if trim_idle_pool:
            pool.free_all_blocks()
        live = int(pool.used_bytes())
        reserved = int(pool.total_bytes())
        # Use the CUDA driver's own accounting as a complementary measure.
        free, total = xp.cuda.runtime.memGetInfo()
        mib = 1024.0 ** 2
        record = {
            "step": int(self.step),
            "phase": str(phase),
            "device_free_mib": round(int(free) / mib, 2),
            "device_total_mib": round(int(total) / mib, 2),
            "cupy_live_mib": round(live / mib, 2),
            "cupy_reserved_mib": round(reserved / mib, 2),
            "cupy_idle_mib": round((reserved - live) / mib, 2),
            "trimmed": int(bool(trim_idle_pool)),
        }
        if self.gpu_memory_audit:
            print(
                f"  GPU memory [{phase}] step={record['step']}: "
                f"CuPy live={record['cupy_live_mib']:.1f} MiB, "
                f"reserved={record['cupy_reserved_mib']:.1f} MiB, "
                f"idle={record['cupy_idle_mib']:.1f} MiB, "
                f"CUDA free={record['device_free_mib']:.1f} MiB"
            )
            if self.log_file is not None:
                path = self.log_file.with_name(
                    self.log_file.stem + ".gpu_memory.csv"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                new_file = not path.exists() or path.stat().st_size == 0
                with path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(record))
                    if new_file:
                        writer.writeheader()
                    writer.writerow(record)
            if getattr(self, "gpu_memory_cache_audit", False):
                self._gpu_cache_snapshot(phase)
        return record

    def _gpu_cache_snapshot(self, phase: str):
        """Log sizes of known persistent CUDA caches, without allocating GPU data."""
        from mini_llm.diagnostics.gpu_cache_audit import persistent_gpu_cache_inventory

        inventory = persistent_gpu_cache_inventory(getattr(self, "model", None))
        mib = 1024.0 ** 2
        parts = []
        rows = []
        for name, stat in sorted(inventory.items()):
            entries = int(stat["entries"])
            cuda_mib = float(stat["cuda_bytes"]) / mib
            parts.append(f"{name}={entries} entries/{cuda_mib:.2f} MiB")
            rows.append({
                "step": int(self.step), "phase": str(phase), "cache": name,
                "entries": entries, "cuda_mib": round(cuda_mib, 3),
            })
        print(f"  GPU cache inventory [{phase}] step={self.step}: " + "; ".join(parts))
        if self.log_file is not None:
            path = self.log_file.with_name(self.log_file.stem + ".gpu_caches.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "step", "phase", "cache", "entries", "cuda_mib"
                ))
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)
        return inventory

    def _gpu_memory_post_validation_gc_probe(self):
        """Compare live/reserved before GC, after GC, and after idle-pool trim.

        Requires the 0078H memory audit. Validation has already completed;
        the last per-stage trim, if enabled, may have run. The GC probe is
        diagnostic and never changes model parameters or training batches.
        """
        if not (self.gpu_memory_audit and getattr(self, "gpu_memory_gc_probe", False)):
            return
        # The caller already recorded 'after_all_validation' immediately
        # before this method, providing the pre-GC snapshot.
        synchronize()
        started = time.perf_counter()
        collected = gc.collect()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(f"  GPU memory Python GC step={self.step}: "
              f"collected={collected}, elapsed_ms={elapsed_ms:.1f}")
        self._gpu_memory_boundary("after_validation_python_gc")
        self._gpu_memory_boundary("after_validation_python_gc_trimmed", trim_idle_pool=True)

    def _gpu_memory_after_validation_stage(self, phase: str):
        """Record validation-stage usage and optionally release idle pool blocks."""
        self._gpu_memory_boundary(phase)
        if self.gpu_memory_trim_after_validation:
            self._gpu_memory_boundary(
                phase + "_trimmed", trim_idle_pool=True,
            )

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
            source = self._choose_train_source(self.train_rng)
            if self.streaming_cache is not None and source in self.streaming_cache.sources:
                shard_path, generation, retired = self.streaming_cache.next_shard(source)
                for old_path in retired:
                    old_key = (source, generation - 1)
                    old_shard = self.shards.pop(old_key, None)
                    if old_shard is not None:
                        self._close_mapped_shard(old_shard)
                    self.document_indices.pop(old_key, None)
                    old_path.unlink(missing_ok=True)
                    Path(str(old_path) + ".docidx.npy").unlink(missing_ok=True)
                cache_key = (source, generation)
                shard_data = self._get_cached_shard(
                    self.shards, cache_key, shard_path, self.load_train_shard,
                )
                self.train_source_batch_counts[source] = (
                    self.train_source_batch_counts.get(source, 0) + 1
                )
            else:
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
            sampling_policy = "legacy"
            if self.memory_context.integration_mode == "routed_prefix":
                # 0078E: document length is an *outcome* of sampling, never
                # a prerequisite. Keep router stochastic training probability
                # independent of which document is selected.
                sampling_policy = "document_first"
                routed_prefix_retrieval = bool(
                    self.train_rng.random()
                    < float(self.memory_context.retrieval_batch_probability)
                )
                min_history_tokens = (
                    self.memory_context.top_k_blocks
                    * self.memory_context.block_size
                    + self.memory_context.block_size - 1
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
                sampling_policy=sampling_policy,
            )
            if routed_prefix_retrieval is not None:
                available = (
                    int(self.memory_context.distant_memory_length)
                    - np.asarray(metadata["history_valid_starts"], dtype=np.int64)
                )
                valid = np.asarray(metadata["target_valid_lengths"], dtype=np.int64)
                full_target = bool(np.all(valid == self.memory_context.target_length))
                # Without per-row attention masks, use the common genuine
                # history suffix for medium-length mixed microbatches. It may
                # be shorter than an individual document's full prefix.
                direct_history = bool(
                    full_target and np.all(available > 0)
                    and int(np.max(available)) < min_history_tokens
                )
                # For long samples preserve the existing router-training
                # probability. For short genuine partial prefixes, direct
                # history needs no learned selection and is always enabled.
                metadata["retrieval_requested"] = routed_prefix_retrieval
                metadata["use_retrieval"] = bool(
                    full_target and np.all(available > 0)
                    and (routed_prefix_retrieval or direct_history)
                )
                metadata["history_mode_requested"] = (
                    "direct_history" if direct_history else
                    "routed_candidate" if metadata["use_retrieval"] else
                    "dense"
                )

        source_name = source if self.train_source_shards is not None else "local"
        self._last_train_batch_source = source_name
        supervised = (
            int((np.asarray(metadata["target_valid_lengths"]) > 0).sum())
            if metadata is not None and "target_valid_lengths" in metadata
                and self.memory_context is not None
                and self.memory_context.memory_training == "router_only"
            else (
                int(np.asarray(metadata["target_valid_lengths"]).sum())
                if metadata is not None and "target_valid_lengths" in metadata
                else self.batch_size * self.seq_length
            )
        )
        self.train_source_supervised_tokens[source_name] = (
            self.train_source_supervised_tokens.get(source_name, 0) + supervised
        )
        if self.streaming_cache is not None or self.training_objective is not None:
            audit = hashlib.sha256(bytes.fromhex(self.sample_trace_sha256))
            audit.update(source_name.encode("utf-8"))
            audit.update(str(cache_key).encode("utf-8"))
            audit.update(memoryview(np.ascontiguousarray(inputs)).cast("B"))
            audit.update(memoryview(np.ascontiguousarray(targets)).cast("B"))
            if self.training_objective is not None and metadata is not None:
                # 0080 traces also pin valid-target and true-history masks.
                # Empty/padded rows must not collide with fully valid rows.
                for name in ("target_loss_mask", "target_valid_lengths",
                             "history_valid_starts", "document_starts", "document_ends"):
                    value = metadata.get(name)
                    if value is not None:
                        array = np.ascontiguousarray(value)
                        audit.update(name.encode("ascii") + b"\0")
                        audit.update(str((array.shape, array.dtype.str)).encode("ascii"))
                        audit.update(memoryview(array).cast("B"))
            self.sample_trace_sha256 = audit.hexdigest()
            self.sample_trace_microbatches += 1
        self.tokens_processed += self.batch_size * self.seq_length
        if return_metadata:
            return inputs, targets, metadata
        return inputs, targets
    
    @staticmethod
    def _new_routing_mode_stats():
        return {
            "microbatches": 0, "samples": 0, "supervised_token_weight": 0.0,
            "available_history_tokens": 0, "used_history_tokens": 0,
            "requested_microbatches": 0,
            "dense_microbatches": 0, "direct_microbatches": 0,
            "routed_microbatches": 0,
            "dense_supervised_tokens": 0.0, "direct_supervised_tokens": 0.0,
            "routed_supervised_tokens": 0.0,
        }

    @staticmethod
    def _merge_routing_mode_stats(total, other):
        for key in total:
            total[key] += other[key]

    def restore_routing_mode_stats(self, state):
        """Start a new measurement epoch for old checkpoints, or resume it."""
        self.routing_modes_total = self._new_routing_mode_stats()
        self._routing_modes_interval = self._new_routing_mode_stats()
        self._last_step_routing_modes = self._new_routing_mode_stats()
        self.routing_modes_since_step = None
        if not isinstance(state, dict) or state.get("version") != 1:
            return
        saved = state.get("total", {})
        for key in self.routing_modes_total:
            if key in saved:
                self.routing_modes_total[key] = type(self.routing_modes_total[key])(saved[key])
        self.routing_modes_since_step = state.get("since_step")

    def _routing_mode_observation(self, cache, batch_metadata, token_weight, batch_size):
        """Account for *used* history, not just available source history.

        Runs before backward frees the context cache. Reads Python cache
        metadata and CPU source masks only; no GPU synchronization or backward
        computation. Exactly one execution mode is recorded per microbatch.
        """
        stats = self._new_routing_mode_stats()
        stats["microbatches"] = 1
        stats["samples"] = int(batch_size)
        stats["supervised_token_weight"] = float(token_weight)
        metadata = batch_metadata or {}
        stats["requested_microbatches"] = int(bool(metadata.get("use_retrieval", False)))
        if self.memory_context is None:
            mode, prefix_length = "dense", 0
        else:
            memory_cache = cache.get("memory_context_cache") or {}
            raw_mode = memory_cache.get("prefix_mode", "none")
            mode_map = {"none": "dense", "direct_history": "direct",
                        "routed_blocks": "routed"}
            if raw_mode not in mode_map:
                raise ValueError(f"unknown routed-prefix execution mode: {raw_mode!r}")
            mode = mode_map[raw_mode]
            prefix_length = int(memory_cache.get("prefix_length", 0))
            if mode != "dense" and prefix_length <= 0:
                raise ValueError("non-dense history mode requires a nonzero prefix")
            starts = metadata.get("history_valid_starts")
            if starts is not None:
                starts_cpu = np.asarray(starts, dtype=np.int64)
                history_capacity = int(self.memory_context.distant_memory_length)
                stats["available_history_tokens"] = int(np.clip(
                    history_capacity - starts_cpu, 0, history_capacity
                ).sum())
        stats["used_history_tokens"] = prefix_length * int(batch_size)
        stats[mode + "_microbatches"] = 1
        stats[mode + "_supervised_tokens"] = float(token_weight)
        return stats

    def _log_routing_mode_stats(self, interval_stats):
        """Write interval and lifetime totals to an independent CSV."""
        count = max(1, interval_stats["microbatches"])
        weight = max(1., interval_stats["supervised_token_weight"])
        print(
            "  Actual history execution (all accumulation microbatches): "
            f"dense={interval_stats['dense_microbatches']}/{count}, "
            f"direct={interval_stats['direct_microbatches']}/{count}, "
            f"routed={interval_stats['routed_microbatches']}/{count}; "
            f"routed_target_share={100 * interval_stats['routed_supervised_tokens'] / weight:.1f}%, "
            f"history_used={interval_stats['used_history_tokens']:,}, "
            f"history_available={interval_stats['available_history_tokens']:,}"
        )
        if self.log_file is None:
            return
        path = self.log_file.with_name(self.log_file.stem + ".routing_modes.csv")
        columns = (
            "step", "scope", "since_step", *self._new_routing_mode_stats().keys(),
            "routed_microbatch_fraction", "routed_supervised_token_fraction",
            "actual_to_available_history_fraction",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if new_file:
                writer.writeheader()
            for scope, record in (("interval", interval_stats),
                                  ("cumulative_since_patch", self.routing_modes_total)):
                row = dict(record)
                row.update({
                    "step": int(self.step), "scope": scope,
                    "since_step": self.routing_modes_since_step,
                    "routed_microbatch_fraction": record["routed_microbatches"] / max(1, record["microbatches"]),
                    "routed_supervised_token_fraction": record["routed_supervised_tokens"] / max(1., record["supervised_token_weight"]),
                    "actual_to_available_history_fraction": record["used_history_tokens"] / max(1, record["available_history_tokens"]),
                })
                writer.writerow(row)

    def get_val_batch(
        self, return_metadata: bool = False, *, source_override=None,
        rng_override=None, shard_index_override=None,
        min_history_override=None, sampling_policy_override=None,
    ):
        """Sample a validation batch; explicit source/rng/index leave validation state intact.

        Explicit-source mode is used by fixed per-source and fixed mixed
        evaluation. The ordinary mixed metric retains its mutable RNG and
        equal-microbatch aggregation but uses document-first sampling from 0078E.
        """
        if source_override is None and (rng_override is not None or shard_index_override is not None or min_history_override is not None or sampling_policy_override is not None):
            raise ValueError("validation overrides require an explicit source")
        if min_history_override is not None and int(min_history_override) < 0:
            raise ValueError("min_history_override must be nonnegative")
        if sampling_policy_override not in (None, "legacy", "document_first"):
            raise ValueError("invalid validation sampling_policy_override")
        rng = self.val_rng if rng_override is None else rng_override
        if self.val_source_shards is not None:
            source = self._choose_source(self.val_rng) if source_override is None else str(source_override)
            if source not in self.val_source_shards:
                raise ValueError(f"unknown validation source: {source!r}")
            paths = self.val_source_shards[source]
            shard_idx = (
                self.current_val_source_shard_idx[source]
                if source_override is None else int(shard_index_override or 0) % len(paths)
            )
            cache_key = (source, shard_idx)
            shard_path = paths[shard_idx]
            shard_data = self._get_cached_shard(
                self.val_shards, cache_key, shard_path, self.load_val_shard,
            )
            if source_override is None:
                self.current_val_source_shard_idx[source] = (
                    shard_idx + 1
                ) % len(paths)
                self.val_source_batch_counts[source] += 1
        else:
            if source_override is not None:
                raise ValueError("per-source validation requires local validation source shards")
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
                shard_data, self.batch_size, self.seq_length, rng=rng
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
                rng=rng,
                eos_token_id=self.eos_token_id,
                document_index=document_index,
                document_aware=(shard_data.ndim == 1),
                return_metadata=True,
                min_history_tokens=(
                    int(min_history_override)
                    if min_history_override is not None
                    else (
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
                    )
                ),
                sampling_policy=(
                    sampling_policy_override
                    if sampling_policy_override is not None else
                    "document_first"
                    if self.memory_context.integration_mode == "routed_prefix"
                    and (source_override is None or min_history_override is not None)
                    else "legacy"
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
                metadata["document_policy_validation"] = metadata.get(
                    "document_sampling_policy", "legacy"
                )

        if return_metadata:
            return inputs, targets, metadata
        return inputs, targets
    
    def _memory_profile_due(self):
        return (
            self._memory_profiled_steps < self.memory_profile_steps
            and (self.step + 1) % self.memory_profile_interval == 0
        )

    def _record_gpu_history_shape(self, cache):
        """CPU-only histogram of the *actual* active-prefix shapes.

        Used only when GPU audit is enabled; does not inspect device tensor
        values or synchronize CUDA.  Histogram is reported at train intervals.
        """
        if not self.gpu_memory_audit or self.gpu_memory_audit_train_interval <= 0:
            return
        mem = cache.get("memory_context_cache") or {}
        mode = str(mem.get("prefix_mode", "none"))
        length = int(mem.get("prefix_length", 0))
        histogram = self._gpu_memory_history_shapes.setdefault(mode, {})
        histogram[length] = histogram.get(length, 0) + 1

    def _print_gpu_history_shapes(self):
        if not self._gpu_memory_history_shapes:
            return
        details = []
        for mode in ("none", "direct_history", "routed_blocks"):
            counts = self._gpu_memory_history_shapes.get(mode, {})
            if not counts:
                continue
            details.append(
                f"{mode}: n={sum(counts.values())}, "
                f"distinct_lengths={len(counts)}, "
                f"min={min(counts)}, max={max(counts)}"
            )
        if details:
            print("  GPU audited history shapes: " + "; ".join(details))
        self._gpu_memory_history_shapes.clear()

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
        return snapshot

    @staticmethod
    def _format_memory_gib(value):
        return float(value) / float(1024 ** 3)

    @staticmethod
    def _update_mode_memory_peak(peaks, mode, prefix_length, phase, snapshot):
        if snapshot is None:
            return
        item = peaks.setdefault(mode, {
            "microbatches": 0, "lengths": set(), "snapshots": {}
        })
        item["lengths"].add(int(prefix_length))
        if phase == "forward":
            item["microbatches"] += 1
        previous = item["snapshots"].get(phase)
        if previous is None or snapshot["device_used"] > previous["device_used"]:
            item["snapshots"][phase] = snapshot

    def _print_memory_report(self, records, mode_peaks=None):
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
        if mode_peaks:
            print("  Per-execution-mode peaks (observed on profiled microbatches):")
            for mode, item in sorted(mode_peaks.items()):
                def _mib(phase, field):
                    snap = item["snapshots"].get(phase, {})
                    return float(snap.get(field, 0)) / (1024 ** 2)
                print(
                    f"    {mode}: n={item['microbatches']}, "
                    f"prefix_lengths={sorted(item['lengths'])}, "
                    f"forward live peak={_mib('forward', 'pool_used'):.0f} MiB, "
                    f"backward live peak={_mib('backward', 'pool_used'):.0f} MiB, "
                    f"released live peak={_mib('released', 'pool_used'):.0f} MiB, "
                    f"released reserved peak={_mib('released', 'pool_reserved'):.0f} MiB"
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

    def _validation_batch_loss(self, inputs, targets, batch_metadata):
        """Apply the exact normal validation objective without backward/updates."""
        batch_metadata = self._configure_routed_prefix_metadata(
            batch_metadata, training=False
        )
        loss_mask = (
            None if batch_metadata is None
            else batch_metadata.get("target_loss_mask")
        )
        loss_mask = self._effective_loss_mask(batch_metadata, loss_mask)
        targets, loss_mask = self._loss_targets_and_mask(targets, loss_mask)
        if self.training_objective is not None:
            metrics = self.training_objective.evaluate_batch(
                inputs, targets, loss_mask, batch_metadata,
            )
            # Retain only compact per-horizon sums; full decoder caches are
            # already released by the objective after this call.
            self._last_block_validation_metrics = metrics
            return float(metrics["final_ce"]), targets, loss_mask
        # Retain the production validation head path. Full logits can OOM for
        # the 65k-vocabulary model, so honor chunking for the new diagnostics.
        if self.lm_head_chunk_tokens > 0:
            head_input, target_slice = self.model.forward_body(
                inputs, return_cache=False, memory_metadata=batch_metadata,
                return_target_slice=True,
            )
            loss, loss_cache = self.model.chunked_lm_head_loss_forward(
                head_input, targets,
                chunk_tokens=self.lm_head_chunk_tokens,
                loss_mask=loss_mask, target_slice=target_slice,
            )
            del loss_cache, head_input
        else:
            logits = self.model.forward(
                inputs, return_cache=False, memory_metadata=batch_metadata,
            )
            loss, _ = self.model.compute_loss(
                logits, targets, loss_mask=loss_mask,
            )
            del logits
        return float(loss), targets, loss_mask

    def compute_val_loss(self) -> float:
        """Historical mixed-source validation: mean of sampled batch losses.

        This is deliberately unchanged in aggregation and RNG behavior for
        backwards compatibility. See compute_val_loss_by_source for stable,
        supervised-token-weighted held-out comparisons.
        """
        losses = []
        for _ in range(self.val_steps):
            inputs, targets, metadata = self.get_val_batch(return_metadata=True)
            loss, _, _ = self._validation_batch_loss(inputs, targets, metadata)
            losses.append(loss)
        return float(xp.mean(xp.asarray(losses, dtype="float32")))

    def compute_val_loss_fixed_mixed(self):
        """0078F: independent fixed held-out source mixture, token-weighted CE.

        Source draws follow configured local weights, shard selection cycles
        independently per source, and the whole sample trace is seeded once per
        evaluation. The training, legacy validation, and shard cursor RNG
        states are untouched. Repeated checkpoints use identical windows.
        """
        if self.val_fixed_mixed_steps <= 0:
            return None
        if self.val_source_shards is None:
            raise ValueError("fixed mixed validation requires local validation sources")
        rng = np.random.default_rng(self.val_fixed_mixed_seed)
        counts = {name: 0 for name in self.val_source_shards}
        digest = hashlib.sha256()
        nll_sum, weight_sum, capacity = 0., 0., 0
        from mini_llm.block_proposal.objective import BlockValidationAccumulator
        block_metrics = (BlockValidationAccumulator(self.training_objective.net.proposal_head.block_size)
                         if self.training_objective is not None else None)
        for _ in range(self.val_fixed_mixed_steps):
            source = self._choose_source(rng)
            shard_idx = counts[source] % len(self.val_source_shards[source])
            counts[source] += 1
            inputs, targets, metadata = self.get_val_batch(
                return_metadata=True, source_override=source,
                rng_override=rng, shard_index_override=shard_idx,
                sampling_policy_override="document_first",
            )
            ce, scored_targets, mask = self._validation_batch_loss(inputs, targets, metadata)
            if block_metrics is not None:
                block_metrics.add(self._last_block_validation_metrics)
            weight = self._supervised_token_weight(scored_targets, mask)
            if not math.isfinite(ce):
                raise ValueError("nonfinite fixed mixed validation loss")
            nll_sum += ce * weight
            weight_sum += weight
            capacity += int(np.size(scored_targets))
            digest.update(source.encode("utf8") + b"\0")
            for part in (inputs, scored_targets, mask):
                if part is None:
                    digest.update(b"none")
                else:
                    arr = np.ascontiguousarray(part)
                    digest.update(str((arr.shape, arr.dtype.str)).encode("ascii"))
                    digest.update(memoryview(arr).cast("B"))
        output = {
            "step": int(self.step), "ce_nats_per_target": nll_sum / weight_sum,
            "perplexity": math.exp(nll_sum / weight_sum),
            "supervised_token_weight": weight_sum, "target_capacity_tokens": capacity,
            "target_fill": weight_sum / capacity,
            "validation_batches": int(self.val_fixed_mixed_steps),
            "fixed_seed": int(self.val_fixed_mixed_seed),
            "sample_sha256": digest.hexdigest(),
            "mixture_manifest_sha256": self.local_mixture_manifest_sha256 or "",
            "tokenizer_sha256": getattr(self, "tokenizer_sha256", "") or "",
        }
        if block_metrics is not None:
            output.update(block_metrics.summary())
        return output

    def _log_fixed_mixed_loss(self, result):
        if result is None:
            return
        print("  Fixed mixed validation CE (document-first, token-weighted): "
              f"{result['ce_nats_per_target']:.4f}, "
              f"supervised={result['supervised_token_weight']:,.0f}, "
              f"fill={100 * result['target_fill']:.1f}%, "
              f"sample_sha256={result['sample_sha256'][:16]}...")
        if self.log_file is None:
            return
        path = self.log_file.with_name(self.log_file.stem + ".val_fixed_mixed.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(result))
            if new:
                writer.writeheader()
            writer.writerow(result)

    def compute_val_loss_by_source(self):
        """Preserve 0078B's exact fixed long-history validation protocol."""
        return self._compute_val_loss_by_source_impl(representative=False)

    def compute_val_loss_by_source_representative(self):
        """0078C: fixed source-token sampled windows without history-length filtering.

        Preserves EOS and loss masks. For the routed-prefix configuration,
        short contexts naturally disable retrieval when ineligible. Samples are
        length-biased by the existing sampler; not uniform over documents.
        """
        return self._compute_val_loss_by_source_impl(representative=True)

    def _compute_val_loss_by_source_impl(self, *, representative):
        """Evaluate identical fixed held-out windows for each local source.

        No optimizer updates, training RNG changes, or changes to the historical
        validation RNG/shard cursors. Each source has its own fixed seed and
        cycles over its validation shards. Means are weighted by effective CE
        mask mass (not by microbatch count). Streaming sources are excluded:
        their validation counterpart is intentionally local-only.
        """
        if self.val_source_shards is None:
            raise ValueError("per-source validation requires local validation source shards")
        batch_count = (self.val_representative_steps if representative
                       else self.val_per_source_steps)
        source_seed = (self.val_representative_seed if representative
                       else self.val_source_seed)
        if batch_count < 1:
            return {}
        output = {}
        for source in sorted(self.val_source_shards):
            seed_data = (f"0078C|{source_seed}|{source}" if representative
                         else f"0078B|{source_seed}|{source}").encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            sampled = hashlib.sha256()
            nll_sum, token_weight, token_count, capacity = 0., 0., 0, 0
            from mini_llm.block_proposal.objective import BlockValidationAccumulator
            block_metrics = (BlockValidationAccumulator(self.training_objective.net.proposal_head.block_size)
                             if self.training_objective is not None else None)
            doc_lengths = []
            history_lengths = []
            unique_docs = set()
            fallback_count = 0
            for i in range(batch_count):
                inputs, targets, metadata = self.get_val_batch(
                    return_metadata=True, source_override=source,
                    rng_override=rng, shard_index_override=i,
                    sampling_policy_override="document_first" if representative else "legacy",
                )
                if representative and metadata is not None and "document_starts" in metadata:
                    starts = np.asarray(metadata["document_starts"], dtype=np.int64)
                    ends = np.asarray(metadata["document_ends"], dtype=np.int64)
                    lengths = ends - starts
                    doc_lengths.extend(int(x) for x in lengths)
                    history_starts = np.asarray(metadata["history_valid_starts"], dtype=np.int64)
                    history_lengths.extend(int(x) for x in (
                        self.memory_context.distant_memory_length - history_starts))
                    shard_no = i % len(self.val_source_shards[source])
                    unique_docs.update((shard_no, int(a), int(b))
                                       for a, b in zip(starts, ends))
                    fallback_count += int(np.count_nonzero(
                        metadata.get("document_selection_fallback", np.zeros(len(starts), dtype=bool))))
                ce, scored_targets, loss_mask = self._validation_batch_loss(
                    inputs, targets, metadata
                )
                if block_metrics is not None:
                    block_metrics.add(self._last_block_validation_metrics)
                weight = self._supervised_token_weight(scored_targets, loss_mask)
                if not math.isfinite(ce):
                    raise ValueError(f"nonfinite validation CE for {source!r}")
                nll_sum += ce * weight
                token_weight += weight
                token_count += (
                    int(np.count_nonzero(loss_mask)) if loss_mask is not None
                    else int(np.size(scored_targets))
                )
                capacity += int(np.size(scored_targets))
                # A fingerprint of the *actual* batches and effective masks.
                # If two checkpoints have different fingerprints, their losses
                # must not be treated as paired measurements.
                for part in (inputs, scored_targets, loss_mask):
                    if part is None:
                        sampled.update(b"none")
                        continue
                    array = np.ascontiguousarray(part)
                    sampled.update(str((array.shape, array.dtype.str)).encode("ascii"))
                    sampled.update(memoryview(array).cast("B"))
            mean_ce = nll_sum / token_weight
            output[source] = {
                "step": int(self.step),
                "source": source,
                "schema_version": 1,
                "mixture_manifest_sha256": self.local_mixture_manifest_sha256 or "",
                "tokenizer_sha256": getattr(self, "tokenizer_sha256", "") or "",
                "memory_mode": (
                    self.memory_context.memory_training if self.memory_context
                    else "plain"
                ),
                "ce_nats_per_target": mean_ce,
                "perplexity": math.exp(mean_ce) if mean_ce < 700 else float("inf"),
                "loss_nll_sum": nll_sum,
                "supervised_token_weight": token_weight,
                "supervised_tokens": token_count,
                "target_capacity_tokens": capacity,
                "target_fill": token_weight / capacity,
                "validation_batches": batch_count,
                "fixed_seed": source_seed,
                "sample_sha256": sampled.hexdigest(),
            }
            if block_metrics is not None:
                output[source].update(block_metrics.summary())
            if representative:
                # Lengths describe sampled fragments, not an unbiased census
                # of documents; individual fragments can be sampled repeatedly.
                arr = np.asarray(doc_lengths, dtype=np.int64)
                hist = np.asarray(history_lengths, dtype=np.int64)
                output[source].update({
                    "sampling_mode": "representative_token_sampled",
                    "sampled_document_count": int(arr.size),
                    "unique_sampled_documents": len(unique_docs),
                    "sampled_document_length_mean": float(np.mean(arr)) if arr.size else None,
                    "sampled_document_length_median": float(np.median(arr)) if arr.size else None,
                    "sampled_document_length_p10": float(np.percentile(arr, 10)) if arr.size else None,
                    "sampled_document_length_p90": float(np.percentile(arr, 90)) if arr.size else None,
                    "short_document_fraction": (
                        float(np.mean(arr - 1 < self.target_seq_length)) if arr.size else None
                    ),
                    "valid_history_mean": float(np.mean(hist)) if hist.size else None,
                    "document_selection_fallback_count": int(fallback_count),
                })
        return output

    def _log_val_source_losses(self, results):
        if not results:
            return
        print("  Per-source held-out validation CE (fixed samples; target-weighted):")
        for name in sorted(results):
            rec = results[name]
            print(
                f"    {name}: CE={rec['ce_nats_per_target']:.4f}, "
                f"supervised={rec['supervised_tokens']:,}, "
                f"fill={100.0 * rec['target_fill']:.1f}%"
            )
        if self.log_file is None:
            return
        path = self.log_file.with_name(self.log_file.stem + ".val_sources.csv")
        columns = (
            "step", "source", "schema_version", "mixture_manifest_sha256",
            "tokenizer_sha256", "memory_mode", "ce_nats_per_target",
            "perplexity", "loss_nll_sum",
            "supervised_token_weight", "supervised_tokens", "target_capacity_tokens",
            "target_fill", "validation_batches", "fixed_seed", "sample_sha256",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if new_file:
                writer.writeheader()
            for name in sorted(results):
                writer.writerow(results[name])

    def _log_representative_source_losses(self, results):
        """Separate 0078C output; never modify historical .val_sources.csv."""
        if not results:
            return
        print("  Per-source representative validation CE (fixed token-sampled documents):")
        for name in sorted(results):
            r = results[name]
            print(
                f"    {name}: CE={r['ce_nats_per_target']:.4f}, "
                f"supervised={r['supervised_tokens']:,}, "
                f"fill={100 * r['target_fill']:.1f}%, "
                f"unique_docs={r['unique_sampled_documents']}/"
                f"{r['sampled_document_count']}, "
                f"sampled_doc_median={r['sampled_document_length_median']}"
            )
        if self.log_file is None:
            return
        path = self.log_file.with_name(self.log_file.stem + ".val_sources_representative.csv")
        columns = (
            "step", "source", "schema_version", "sampling_mode",
            "mixture_manifest_sha256", "tokenizer_sha256", "memory_mode",
            "ce_nats_per_target", "perplexity", "loss_nll_sum",
            "supervised_token_weight", "supervised_tokens", "target_capacity_tokens",
            "target_fill", "validation_batches", "fixed_seed", "sample_sha256",
            "sampled_document_count", "unique_sampled_documents",
            "sampled_document_length_mean", "sampled_document_length_median",
            "sampled_document_length_p10", "sampled_document_length_p90",
            "short_document_fraction", "valid_history_mean",
            "document_selection_fallback_count",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if new_file:
                writer.writeheader()
            for name in sorted(results):
                writer.writerow(results[name])

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

    @staticmethod
    def _supervised_token_weight(targets, loss_mask):
        """CE's denominator before per-microbatch averaging.

        The minibatch sampler produces CPU masks.  Counting CPU-side avoids
        an extra GPU synchronization per accumulation microstep.  Sum mask
        values rather than count nonzeros to agree with masked CE when masks
        are fractional.  Router-only training uses the already restricted mask.
        """
        if loss_mask is None:
            return float(np.prod(targets.shape))
        if isinstance(loss_mask, np.ndarray):
            count = float(np.sum(loss_mask, dtype=np.float64))
        else:
            count = float(_array_to_float(xp.sum(loss_mask)))
        if not np.isfinite(count) or count <= 0.0:
            raise ValueError("a training microbatch must have positive finite supervised-token weight")
        return count

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
        memory_active = self._memory_profile_due()
        mode_memory_peaks = {}
        step_memory_token_stats = self._empty_memory_token_stats()
        noise = self.gradient_noise
        noise_active = noise is not None and noise.begin(self.step)
        if memory_active:
            self._record_memory_snapshot(memory_records, "step_start")

        # Accumulate gradients over multiple steps
        # Use Python floats for loss accumulation (minimal overhead)
        # The main optimization is avoiding host-device sync during gradient computation
        # Keep microbatch losses on-device; converting each one to a
        # Python float forces a CUDA stream synchronization.
        loss_sum_backend = xp.asarray(0.0, dtype="float32")
        accumulation_weight_sum = 0.0
        supervised_tokens_in_step = 0.0
        source_observations = []   # (source, CPU coverage, on-device CE scalar)
        block_step_metrics = None
        step_routing_modes = self._new_routing_mode_stats()
        # Rescale by nominal token capacity before backward to avoid large
        # intermediate gradients.  The common scale cancels at normalization.
        nominal_microbatch_tokens = float(self.batch_size * self.target_seq_length)

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
            valid_tokens = self._supervised_token_weight(targets, loss_mask)
            supervised_tokens_in_step += valid_tokens
            source = getattr(self, "_last_train_batch_source", "local")
            coverage = batch_coverage(
                batch_size=self.batch_size, source_length=self.seq_length,
                target_shape=targets.shape, loss_mask=loss_mask,
                memory_length=(self.memory_context.distant_memory_length
                               if self.memory_context is not None else 0),
                target_length=self.target_seq_length, metadata=batch_metadata,
            )
            if self.gradient_accumulation_weighting == "supervised_tokens":
                microbatch_weight = valid_tokens / nominal_microbatch_tokens
            else:
                microbatch_weight = 1.0
            accumulation_weight_sum += microbatch_weight
            if noise_active:
                noise.count_tokens(targets, loss_mask)
            self._accumulate_memory_batch_token_stats(
                step_memory_token_stats, batch_metadata, loss_mask
            )
            
            if self.training_objective is not None:
                # Objective-specific math delegates to the existing 0080
                # single-pass trainer; optimizer/accumulation remain here.
                # This branch never runs for the default production AR path.
                with performance_scope("train.block_proposal.forward_backward"):
                    metrics = self.training_objective.forward_backward(
                        inputs, targets, loss_mask, batch_metadata,
                        grad_scale=self.loss_scale * microbatch_weight,
                    )
                if int(metrics["unique_supervised_tokens"]) != int(valid_tokens):
                    raise ValueError("0080/production supervised-token denominators disagree")
                if (self.memory_context is not None and
                        self.memory_context.integration_mode == "routed_prefix"):
                    self._merge_routing_mode_stats(
                        step_routing_modes,
                        self._routing_mode_observation(
                            {"memory_context_cache": metrics["routing_cache"]},
                            batch_metadata, valid_tokens, int(inputs.shape[0]),
                        ),
                    )
                loss_sum_backend += xp.asarray(metrics["total_loss"], dtype=xp.float32) * microbatch_weight
                source_observations.append((source, coverage, xp.asarray(metrics["final_ce"], dtype=xp.float32)))
                if noise_active:
                    noise.capture(accum_step + 1)
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_model_backward")
                if block_step_metrics is None:
                    block_step_metrics = {
                        "unique_supervised_tokens": 0,
                        "position_counts": xp.zeros_like(metrics["position_counts"]),
                        "proposal_position_ce_sums": xp.zeros_like(metrics["proposal_position_ce_sums"]),
                        "final_position_ce_sums": xp.zeros_like(metrics["final_position_ce_sums"]),
                        # Compact device-resident scalars: no .item() or
                        # float() on every 0080 microbatch. These are read at
                        # the original bounded training logging cadence.
                        "proposal_nll_sum": xp.zeros((), dtype=xp.float64),
                        "final_nll_sum": xp.zeros((), dtype=xp.float64),
                        "joint_loss_sum": xp.zeros((), dtype=xp.float64),
                    }
                block_step_metrics["unique_supervised_tokens"] += int(valid_tokens)
                for name in ("position_counts", "proposal_position_ce_sums", "final_position_ce_sums"):
                    block_step_metrics[name] += metrics[name]
                for dst, key in (("proposal_nll_sum", "proposal_ce"),
                                 ("final_nll_sum", "final_ce"),
                                 ("joint_loss_sum", "total_loss")):
                    block_step_metrics[dst] += xp.asarray(metrics[key], dtype=xp.float64) * valid_tokens
                del inputs, targets, batch_metadata, loss_mask
                continue

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
                memory_context_cache = cache.get("memory_context_cache") or {}
                profiled_mode = memory_context_cache.get("prefix_mode", "none")
                profiled_prefix_length = int(memory_context_cache.get("prefix_length", 0))
                snap = self._record_memory_snapshot(memory_records, "after_model_forward")
                self._update_mode_memory_peak(
                    mode_memory_peaks, profiled_mode, profiled_prefix_length,
                    "forward", snap,
                )

            self._record_gpu_history_shape(cache)

            if (self.memory_context is not None and
                    self.memory_context.integration_mode == "routed_prefix"):
                self._merge_routing_mode_stats(
                    step_routing_modes,
                    self._routing_mode_observation(
                        cache, batch_metadata, valid_tokens, int(inputs.shape[0])
                    ),
                )

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
                        "Memory router (first microbatch snapshot only): "
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
                loss_sum_backend += loss_backend * microbatch_weight
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_forward")
                del logits
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_logits_release")

                with performance_scope("train.loss_backward"):
                    d_logits = self.model.backward_loss(loss_cache)
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_backward")
                if self.loss_scale * microbatch_weight != 1.0:
                    d_logits = d_logits * (self.loss_scale * microbatch_weight)
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
                loss_sum_backend += loss_backend * microbatch_weight
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
                        loss_cache, grad_scale=self.loss_scale * microbatch_weight
                    )
                if memory_active:
                    self._record_memory_snapshot(memory_records, "after_loss_backward")
                with performance_scope("train.model_backward"):
                    self.model.backward_body(dx_head, cache)
                del dx_head
                if target_slice is None:
                    del head_input

            # Retain only a scalar on device, without per-microbatch GPU sync.
            # Raw CE is not scaled by loss_scale / gradient accumulation policy.
            source_observations.append((source, coverage, loss_backend))
            if noise_active:
                noise.capture(accum_step + 1)
            if memory_active:
                snap = self._record_memory_snapshot(memory_records, "after_model_backward")
                self._update_mode_memory_peak(
                    mode_memory_peaks, profiled_mode, profiled_prefix_length,
                    "backward", snap,
                )

            # Critical 0055A lifetime cleanup: Python evaluates the RHS of the
            # next ``logits, cache = model.forward(...)`` before replacing the
            # old locals.  Without these deletes, the previous microbatch's
            # entire backward cache and CE gradient can stay alive during the
            # next microbatch forward, artificially doubling activation peak.
            del cache, loss_cache, inputs, targets, batch_metadata, loss_mask
            if memory_active:
                snap = self._record_memory_snapshot(memory_records, "after_microbatch_release")
                self._update_mode_memory_peak(
                    mode_memory_peaks, profiled_mode, profiled_prefix_length,
                    "released", snap,
                )

        if memory_active:
            self._record_memory_snapshot(memory_records, "after_grad_accum")

        # Preserve exact CPU-side document coverage for train() diagnostics.
        self._last_step_memory_token_stats = step_memory_token_stats

        # Average loss - only sync once at the end
        if accumulation_weight_sum <= 0.0:
            raise ValueError("no supervised tokens in accumulated optimizer step")
        avg_loss = float(_array_to_float(loss_sum_backend / accumulation_weight_sum))
        self._last_step_supervised_tokens = supervised_tokens_in_step
        if block_step_metrics is not None:
            self._last_step_block_proposal_metrics = block_step_metrics
            self._block_proposal_supervised_tokens += block_step_metrics["unique_supervised_tokens"]
        self._last_step_supervised_utilization = (
            supervised_tokens_in_step / (
                nominal_microbatch_tokens * self.grad_accum_steps
            )
        )
        if not np.isfinite(avg_loss):
            raise ValueError(f"NaN/Inf loss detected at step {self.step}!")
        # Read original cumulative gradients before normalization or clipping.
        # Do not write the CSV until the optimizer update has succeeded.
        noise_rows = noise.finish() if noise_active else []

        # Scale down gradients by loss_scale to cancel out the scaling
        if self.loss_scale != 1.0:
            for p in self.optimization_parameters:
                if p.grad is not None:
                    p.grad[...] = p.grad / self.loss_scale
        
        # Microbatch mode: each CE-mean gradient has equal coefficient.
        # Supervised-token mode: each valid target has equal coefficient across
        # all accumulation microsteps. This is normalized BEFORE gradient clip.
        # In both modes the optimizer sees an average, not a raw gradient sum.
        if accumulation_weight_sum != 1.0:
            for p in self.optimization_parameters:
                if p.grad is not None:
                    p.grad[...] = p.grad / accumulation_weight_sum
        
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
        
        # One device-to-host transfer per optimizer step, after the update.
        # No new loss computation or modification of gradients is performed.
        if source_observations:
            losses_backend = xp.stack([item[2] for item in source_observations])
            losses_cpu = (losses_backend.get() if hasattr(losses_backend, "get")
                          else np.asarray(losses_backend))
            step_source = {}
            for (name, coverage, _), ce in zip(source_observations, losses_cpu):
                observation = dict(coverage)
                observation["loss_nll_sum"] = (
                    float(ce) * float(coverage["supervised_token_weight"])
                )
                merge_record(step_source.setdefault(name, empty_record()), observation)
            self._last_step_source_metrics = step_source
            if self.source_metrics_since_step is None:
                self.source_metrics_since_step = int(self.step)
            for name, observation in step_source.items():
                merge_record(self.source_metrics.setdefault(name, empty_record()), observation)
                merge_record(self._source_metrics_interval.setdefault(name, empty_record()), observation)

        if step_routing_modes["microbatches"]:
            if self.routing_modes_since_step is None:
                self.routing_modes_since_step = int(self.step)
            self._last_step_routing_modes = step_routing_modes
            self._merge_routing_mode_stats(self.routing_modes_total, step_routing_modes)
            self._merge_routing_mode_stats(self._routing_modes_interval, step_routing_modes)
        self.step += 1
        if noise_active:
            noise.commit(noise_rows)

        if memory_active:
            self._print_memory_report(memory_records, mode_memory_peaks)
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
    
    def block_proposal_resume_contract(self):
        """Fields which must match to call an 0080 optimizer resume exact.

        Intentionally excludes run directories/logging intervals and corpus
        transitions: data fork provenance has its own production protocol.
        """
        if self.training_objective is None:
            return None
        schedule = self.scheduler
        return {
            "version": 1,
            "batch_size": int(self.batch_size),
            "grad_accum_steps": int(self.grad_accum_steps),
            "seq_length": int(self.seq_length),
            "target_seq_length": int(self.target_seq_length),
            "eos_token_id": self.eos_token_id,
            "gradient_accumulation_weighting": self.gradient_accumulation_weighting,
            "grad_clip": float(self.grad_clip),
            "loss_scale": float(self.loss_scale),
            "weight_decay": float(self.optimizer.weight_decay),
            "adam_beta1": float(self.optimizer.beta1),
            "adam_beta2": float(self.optimizer.beta2),
            "adam_eps": float(self.optimizer.eps),
            "schedule": {
                "peak_lr": float(schedule.peak_lr),
                "warmup_steps": int(schedule.warmup_steps),
                "total_steps": int(schedule.total_steps),
                "min_lr": float(schedule.min_lr),
            },
            "source_choice_granularity": "one_source_per_microbatch",
        }

    def validate_block_proposal_resume_contract(self, saved_state):
        if self.training_objective is None:
            return
        expected = saved_state.get("block_proposal_resume_contract")
        if expected is None:
            # Legacy 0080M and 0080I checkpoints have no such fingerprint.
            # Loading is permitted but exact scheduling cannot be certified.
            print("WARNING: legacy 0080 checkpoint lacks production resume contract; "
                  "exact schedule/accumulation continuity is unverified")
            return
        actual = self.block_proposal_resume_contract()
        if expected != actual:
            mismatched = [k for k in sorted(set(expected) | set(actual))
                          if expected.get(k) != actual.get(k)]
            raise ValueError("0080 resume training contract mismatch: " +
                             ", ".join(mismatched))

    def _log_block_proposal_step(self):
        """Log K-horizon statistics at existing bounded logging cadence."""
        metrics = self._last_step_block_proposal_metrics
        if self.training_objective is None or metrics is None:
            return
        from mini_llm.backend import asnumpy
        import numpy as np
        counts = np.asarray(asnumpy(metrics["position_counts"]), dtype=np.float64)
        proposal = np.asarray(asnumpy(metrics["proposal_position_ce_sums"]), dtype=np.float64)
        final = np.asarray(asnumpy(metrics["final_position_ce_sums"]), dtype=np.float64)
        total = int(metrics["unique_supervised_tokens"])
        # Logging is a bounded synchronization point. Convert the two joint
        # CE sums once here, not in every decoder group / microbatch.
        proposal_sum = float(asnumpy(metrics["proposal_nll_sum"]))
        final_sum = float(asnumpy(metrics["final_nll_sum"]))
        print("  0080: final CE={:.4f}, proposal CE={:.4f}, gain={:+.4f}, "
              "unique_targets={:,}, K={}".format(
                  final_sum / total, proposal_sum / total,
                  (proposal_sum-final_sum) / total, total, len(counts)))
        if self.log_file is None:
            return
        path = self.log_file.with_name(self.log_file.stem + ".block_proposal.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as handle:
            columns = ("step", "horizon", "count", "proposal_ce", "final_ce", "decoder_gain")
            writer = csv.DictWriter(handle, fieldnames=columns)
            if handle.tell() == 0:
                writer.writeheader()
            for idx, count in enumerate(counts):
                count = int(count)
                writer.writerow({
                    "step": self.step, "horizon": idx+1, "count": count,
                    "proposal_ce": proposal[idx] / count if count else "",
                    "final_ce": final[idx] / count if count else "",
                    "decoder_gain": (proposal[idx]-final[idx]) / count if count else "",
                })

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
        
        params = {p.name: p.data for p in (
            self.model.parameters() if self.training_objective is None
            else self.training_objective.parameters()
        )}
        
        # Issue #14: Include optimizer state
        # Note: AdamW stores m and v internally, not on Parameter objects
        # We store the master weights (FP32 copies) and moments
        if isinstance(self.optimizer, HybridMuonAdamW):
            optimizer_state = self.optimizer.checkpoint_state()
        else:
            optimizer_state = {
                "step": self.optimizer.step_index,
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

        # Persist exact active remote segment bytes before saving the checkpoint.
        saved_stream = (self.streaming_cache.checkpoint_state(self.checkpoint_dir)
                        if self.streaming_cache else None)

        # Issue #16: Track tokens_processed
        training_state = {
            "step": self.step,
            "tokens_processed": self.tokens_processed,  # New: track cumulative tokens
            "gradient_accumulation_weighting": self.gradient_accumulation_weighting,
            "gradient_weighting_history": getattr(self, "gradient_weighting_history", []),
            "total_steps": self.total_steps,
            "current_train_shard_idx": self.current_train_shard_idx,
            "current_val_shard_idx": self.current_val_shard_idx,
            "current_train_source_shard_idx": self.current_train_source_shard_idx,
            "current_val_source_shard_idx": self.current_val_source_shard_idx,
            "train_source_batch_counts": self.train_source_batch_counts,
            "train_source_supervised_tokens": self.train_source_supervised_tokens,
            "routing_mode_metrics": {
                "version": 1,
                "since_step": self.routing_modes_since_step,
                "total": self.routing_modes_total,
            },
            "source_metrics": {
                "version": 1,
                "since_step": self.source_metrics_since_step,
                "sources": self.source_metrics,
            },
            "data_provenance": {
                "version": 1,
                "fork": getattr(self, "data_fork", None),
                "mixture_history": getattr(self, "mixture_history", []),
                "mixture_phase_start_source_counts": getattr(
                    self, "mixture_phase_start_source_counts", {}
                ),
                "mixture_phase_start_batch_counts": getattr(
                    self, "mixture_phase_start_batch_counts", {}
                ),
                "local_mixture_manifest_sha256": self.local_mixture_manifest_sha256,
                "tokenizer_sha256": getattr(self, "tokenizer_sha256", None),
                "sample_trace_sha256": self.sample_trace_sha256 if (
                    self.streaming_cache or self.training_objective is not None) else None,
                "sample_trace_microbatches": self.sample_trace_microbatches,
                "effective_weights": {
                    name: float(weight) for name, weight in zip(
                        self._train_source_names,
                        self._train_source_probabilities
                        if self._train_source_probabilities is not None else [],
                    )
                },
                "streaming": saved_stream,
                "short_document_packing": {"enabled": False},
            },
            "val_source_batch_counts": self.val_source_batch_counts,
            "progressive_active_layers": int(
                getattr(self.model, "active_layers", len(self.model.blocks))
            ),
            "progressive_growth_steps": list(self.progressive_growth_steps),
            "optimizer_parameter_birth_steps": (
                self.optimizer.parameter_birth_step_state()
            ),
            "optimizer_kind": "hybrid-muon" if isinstance(
                self.optimizer, HybridMuonAdamW) else "adamw",
            "muon_hyperparameters": ({
                "momentum": self.optimizer.momentum,
                "ns_steps": self.optimizer.ns_steps,
                "matmul_precision": self.optimizer.matmul_precision,
                "group_names": sorted(self.optimizer.muon_names),
            } if isinstance(self.optimizer, HybridMuonAdamW) else None),
            # 0074C: preserve the optimizer configuration across resume even
            # when the CLI flags are omitted on the resumed run.
            "adamw_hyperparameters": {
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "eps": self.optimizer.eps,
            },
        }
        
        if self.training_objective is not None:
            # Discriminates joint 0080 checkpoints from AR checkpoints. Loading
            # and strict provenance checks remain in the 0080N milestone.
            training_state["architecture"] = self.training_objective.checkpoint_metadata()
            training_state["prediction_objective"] = self.training_objective.name
            # The existing 0080 legacy reader checks these exact fields.
            # The full production resume/provenance bridge is deferred to 0080N.
            training_state["optimizer_step"] = int(self.optimizer.step_index)
            training_state["optimizer_options"] = {
                key: float(getattr(self.optimizer, key))
                for key in ("lr", "beta1", "beta2", "eps", "weight_decay")
            }
            training_state["tokenizer_hash"] = getattr(self, "tokenizer_sha256", None)
            training_state["supervised_tokens"] = int(self._block_proposal_supervised_tokens)
            training_state["block_proposal_resume_contract"] = self.block_proposal_resume_contract()

        # Issue #12: Include RNG states
        training_state["train_rng_state"] = self.train_rng.bit_generator.state
        training_state["val_rng_state"] = self.val_rng.bit_generator.state
        
        save_checkpoint(
            path=self.checkpoint_dir,
            model_params=params,
            optimizer_state=optimizer_state,
            training_state=training_state,
        )
        if self.streaming_cache is not None:
            self.streaming_cache.prune_checkpoint_snapshots(
                self.checkpoint_dir, saved_stream,
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
        if getattr(self.model, "progressive_depth_enabled", False):
            print(
                "Progressive depth: "
                f"{self.model.active_layers}/{self.model.max_layers} active layers"
            )
            if self.progressive_growth_steps:
                print(
                    "Progressive growth steps: "
                    + ", ".join(str(x) for x in self.progressive_growth_steps)
                )
        
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
        
        if self.streaming_cache is not None:
            self.streaming_cache.prefetch()
        for step in range(num_steps):
            # Growth step N means: complete N optimizer updates at the previous
            # depth, then activate the next exact-identity block before update
            # N+1.  Calling this before every step makes resume idempotent.
            self._apply_scheduled_progressive_growth()

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
            if (
                self.gpu_memory_audit
                and self.gpu_memory_audit_train_interval > 0
                and self.step % self.gpu_memory_audit_train_interval == 0
            ):
                self._gpu_memory_boundary("after_training_step")
                self._print_gpu_history_shapes()
            if self.step % self.val_interval == 0:
                self._gpu_memory_boundary("before_validation")
                val_loss = self.compute_val_loss()
                self._gpu_memory_after_validation_stage("after_standard_validation")
                print(f"  Validation loss: {val_loss:.4f}")
                if self.val_per_source_steps:
                    self._log_val_source_losses(self.compute_val_loss_by_source())
                    self._gpu_memory_after_validation_stage("after_fixed_source_validation")
                if self.val_representative_steps:
                    self._log_representative_source_losses(
                        self.compute_val_loss_by_source_representative()
                    )
                    self._gpu_memory_after_validation_stage("after_representative_validation")
                if self.val_fixed_mixed_steps:
                    self._log_fixed_mixed_loss(self.compute_val_loss_fixed_mixed())
                    self._gpu_memory_after_validation_stage("after_fixed_mixed_validation")
                self._gpu_memory_boundary("after_all_validation")
                self._gpu_memory_post_validation_gc_probe()
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

                depth_text = ""
                if getattr(self.model, "progressive_depth_enabled", False):
                    depth_text = (
                        f"depth={self.model.active_layers}/{self.model.max_layers}, "
                    )
                print(
                    f"Step {self.step}/{num_steps + start_step}: "
                    f"loss={avg_loss:.4f}, "
                    f"{depth_text}"
                    f"lr={lr:.6f}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"grad_weighting={self.gradient_accumulation_weighting}, "
                    f"target_fill={100.0 * self._last_step_supervised_utilization:.1f}%, "
                    f"{steps_per_sec:.2f} steps/sec, "
                    f"step_time={precise_step_seconds * 1000.0:.1f} ms, "
                    f"{throughput_text}"
                )

                if self.training_objective is not None:
                    self._log_block_proposal_step()
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

                # 0078A: source-level CE and token/document accounting.
                if self._source_metrics_interval:
                    print("  Per-source train CE (valid-target weighted; this log interval):")
                    for name in sorted(self._source_metrics_interval):
                        stats = self._source_metrics_interval[name]
                        mean_ce = stats["loss_nll_sum"] / stats["supervised_token_weight"]
                        fill = stats["supervised_token_weight"] / max(stats["target_capacity_tokens"], 1)
                        doc_n = stats["sampled_document_spans"]
                        doc_text = (
                            f", sampled_doc_mean={stats['sampled_document_span_tokens_sum'] / doc_n:,.0f}"
                            if doc_n else ""
                        )
                        print(f"    {name}: CE={mean_ce:.4f}, "
                              f"supervised={stats['supervised_tokens']:,}, "
                              f"target_fill={100.0 * fill:.1f}%{doc_text}")
                    source_path = self._source_metrics_log_path()
                    if source_path is not None:
                        rows = []
                        for name in sorted(self._source_metrics_interval):
                            rows.append(source_metrics_row(
                                name, self._source_metrics_interval[name], self.step,
                                "interval", self.source_metrics_since_step,
                            ))
                            rows.append(source_metrics_row(
                                name, self.source_metrics[name], self.step,
                                "cumulative_since_patch", self.source_metrics_since_step,
                            ))
                        write_source_rows(source_path, rows)
                    self._source_metrics_interval = {}

                if self._routing_modes_interval["microbatches"]:
                    self._log_routing_mode_stats(self._routing_modes_interval)
                    self._routing_modes_interval = self._new_routing_mode_stats()

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
                self._gpu_memory_boundary("before_checkpoint_save")
                self.save()
                self._gpu_memory_boundary("after_checkpoint_save")
                if self.gpu_memory_trim_after_checkpoint:
                    self._gpu_memory_boundary(
                        "after_checkpoint_save_trimmed", trim_idle_pool=True,
                    )
        
        print(f"\nTraining complete!")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {np.mean(losses):.4f}")
        
        return losses
