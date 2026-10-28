"""Decoder-only language model built from Transformer blocks."""

import os
from types import SimpleNamespace

from mini_llm.backend import xp, resolve_dtype, is_low_precision_dtype, is_bfloat16_dtype
from mini_llm.config import ModelConfig
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.linear import Linear, TiedOutputProjection
from mini_llm.ops.loss import (
    cross_entropy_forward,
    cross_entropy_backward,
    chunked_bf16_cross_entropy_loss,
    chunked_bf16_cross_entropy_grad_inplace,
)
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.rope import prepare_rope_positions
from mini_llm.ops.hierarchical_memory import (
    ActiveContext,
    TerminalMemoryContext,
    HierarchicalMemoryRouter,
    ExternalMemoryReader,
)
from mini_llm.performance_profiler import performance_scope


class DecoderLanguageModel:
    """
    Decoder-only language model.
    
    Architecture:
        Token IDs → Embedding → Transformer blocks → RMSNorm → Output Proj → Logits → Loss
        
    Uses separate output projection layer (untied embeddings by default).
    The embedding matrix can be optionally tied to the output projection via weight sharing.
    """
    
    def __init__(self, config: ModelConfig, rng_seed: int = 0, dtype: str = None):
        """
        Initialize the decoder language model.
        
        Args:
            config: ModelConfig with all hyperparameters
            rng_seed: Seed for random number generation
            dtype: Data type (overrides config if provided)
        """
        self.config = config
        self.dtype = resolve_dtype(dtype if dtype is not None else config.dtype)
        # 0078I: bound the number of direct-history active lengths if explicitly
        # requested. The 0078E document-first sampler is left untouched.
        # Zero (default) retains byte-for-byte 0078E context selection.
        self.direct_history_bucket_tokens = int(
            os.environ.get("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS", "0")
        )
        if self.direct_history_bucket_tokens < 0:
            raise ValueError("MINI_LLM_DIRECT_HISTORY_BUCKET_TOKENS must be >= 0")
        
        # Random stream for initialization
        from mini_llm.backend import RandomStream
        rng = RandomStream(rng_seed)
        
        # Embedding layer
        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            std=config.init_std,
            rng=rng,
            name="embedding",
            dtype=self.dtype
        )
        
        # 0064B progressive depth keeps the final architecture allocated but
        # executes/optimizes only a prefix of blocks.  Inactive blocks are
        # independently random-initialized once and become exact identities
        # when activated because their learned residual gates start at zero.
        self.progressive_depth_enabled = bool(config.progressive_depth)
        self.active_layers = (
            int(config.progressive_initial_layers)
            if self.progressive_depth_enabled
            else int(config.n_layers)
        )

        # Create transformer blocks
        self.blocks = []
        for layer_idx in range(config.n_layers):
            attention_config = (
                None
                if config.attention_layers is None
                else config.attention_layers[layer_idx]
            )
            block = TransformerBlock(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                n_experts=config.n_experts,
                top_k=config.top_k,
                input_std=config.init_std,
                output_std=config.residual_init_std,
                rope_base=config.rope_base,
                rng=rng,
                eps=config.rms_eps,
                name=f"blocks.{layer_idx}",
                dtype=self.dtype,
                attention_config=attention_config,
                residual_gates=self.progressive_depth_enabled,
                ffn_mode=getattr(config, "ffn_mode", "moe"),
                dual_resolution_config=config.dual_resolution,
                residual_gate_init=(
                    1.0 if layer_idx < self.active_layers else 0.0
                ),
            )
            self.blocks.append(block)
        
        # Final normalization
        self.final_norm = RMSNorm(
            config.d_model,
            eps=config.rms_eps,
            name="final_norm",
            dtype=self.dtype
        )
        
        # Output projection layer (maps d_model -> vocab_size)
        if config.tie_embeddings:
            self.output_proj = TiedOutputProjection(self.embedding.W)
        else:
            self.output_proj = Linear(
                config.d_model,
                config.vocab_size,
                config.init_std,
                rng,
                name="output_proj",
                dtype=self.dtype
            )

        # 0058: instantiate the top-level memory selector *after* all existing
        # model parameters so the RNG initialization sequence of the 0057A base
        # Transformer remains unchanged.  Existing wide-500M checkpoints can
        # therefore initialize only these new, small router parameters.
        self.memory_router = None
        self.memory_reader = None
        self.memory_attention_layer = None
        self.historical_memory = None
        self.direct_full_history = bool(
            config.memory_context.enabled
            and config.memory_context.integration_mode == "direct_full_history"
        )
        if (config.memory_context.enabled and
                config.memory_context.integration_mode not in {
                    "historical_read", "direct_full_history"
                }):
            self.memory_router = HierarchicalMemoryRouter(
                d_model=config.d_model,
                config=config.memory_context,
                rng=rng,
                input_std=config.init_std,
                name="memory_router",
                dtype=self.dtype,
            )
            if config.memory_context.integration_mode == "pretransformer_read":
                self.memory_reader = ExternalMemoryReader(
                    d_model=config.d_model,
                    d_head=config.d_head,
                    config=config.memory_context,
                    rng=rng,
                    input_std=config.init_std,
                    output_std=config.residual_init_std,
                    rope_base=config.rope_base,
                    name="memory_reader",
                    dtype=self.dtype,
                )
            elif config.memory_context.integration_mode == "terminal_landmark":
                layer = int(config.memory_context.memory_attention_layer)
                if layer < 0:
                    layer += len(self.blocks)
                if not (0 <= layer < len(self.blocks)):
                    raise ValueError("memory_attention_layer is outside model depth")
                self.memory_attention_layer = layer
                # Attach only after all base-model parameters were initialized so
                # existing 4k checkpoint arrays retain their exact RNG sequence.
                self.blocks[layer].attention.configure_terminal_memory(
                    config.memory_context, rng
                )
        if (config.memory_context.enabled and
                config.memory_context.integration_mode == "historical_read"):
            from mini_llm.ops.historical_read import HistoricalReadMemory
            self.historical_memory = HistoricalReadMemory(
                config, self.embedding.W, rng, self.dtype,
            )
            for i, block in enumerate(self.blocks):
                block.attention.historical_reader = self.historical_memory
                block.attention.historical_layer = i

        # The new heads are created AFTER all original parameters to preserve
        # original RNG initialization and checkpoint array names/shapes.
        if config.full_history_layers:
            from mini_llm.ops.full_history_head import FullHistoryHead
            for layer_idx in config.full_history_layers:
                self.blocks[layer_idx].full_history_head = FullHistoryHead(
                    d_model=config.d_model, d_head=config.d_head,
                    embedding=self.embedding.W, rng=rng,
                    init_std=config.init_std, rope_base=config.rope_base,
                    name=f"blocks.{layer_idx}.full_history", dtype=self.dtype,
                    zero_output_init=not self.direct_full_history,
                    output_init_std=config.residual_init_std,
                )
                if self.direct_full_history:
                    self.blocks[layer_idx].attention.replace_full_head_with_history = True
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_proj.parameters())
        if self.memory_router is not None:
            params.extend(self.memory_router.parameters())
        if self.memory_reader is not None:
            params.extend(self.memory_reader.parameters())
        if self.historical_memory is not None:
            params.extend(self.historical_memory.parameters())
        return params
    
    @property
    def max_layers(self):
        return len(self.blocks)

    def set_active_layers(self, count: int):
        """Set the executed Transformer prefix without modifying gate values.

        This is used when restoring a checkpoint/inference state.  Growth during
        training should use :meth:`activate_next_layer`, which additionally
        resets the newly exposed residual gates to exact zero.
        """
        count = int(count)
        if not 1 <= count <= len(self.blocks):
            raise ValueError("active layer count must lie in [1, max_layers]")
        if not self.progressive_depth_enabled and count != len(self.blocks):
            raise ValueError("partial depth requires config.progressive_depth=True")
        self.active_layers = count

    def activate_next_layer(self):
        """Expose one new random block as an exact zero-gated identity.

        Returns the zero-based layer index, or ``None`` if already at maximum
        depth.  Optimizer state is intentionally managed by the trainer because
        it must preserve existing Adam history while adding fresh state only for
        the newly active parameters.
        """
        if not self.progressive_depth_enabled:
            raise RuntimeError("progressive depth is disabled for this model")
        if self.active_layers >= len(self.blocks):
            return None
        layer_idx = int(self.active_layers)
        block = self.blocks[layer_idx]
        if not getattr(block, "residual_gates", False):
            raise RuntimeError("progressive block is missing residual gates")
        block.alpha_attn.data[...] = 0.0
        block.alpha_mlp.data[...] = 0.0
        block.alpha_attn.zero_grad()
        block.alpha_mlp.zero_grad()
        block.zero_grad()
        self.active_layers += 1
        if hasattr(block, "refresh_compute_buffers"):
            block.refresh_compute_buffers()
        return layer_idx

    def active_block_parameters(self):
        params = []
        for block in self.blocks[: self.active_layers]:
            params.extend(block.parameters())
        return params

    def memory_parameters(self):
        """Return only the 0058C trainable retrieval subsystem parameters."""
        if not self.hierarchical_memory_enabled:
            return []
        if self.historical_memory is not None:
            return list(self.historical_memory.parameters())
        params = list(self.memory_router.parameters())
        if self.config.memory_context.integration_mode == "terminal_landmark":
            layer = self.blocks[self.memory_attention_layer].attention
            if layer.terminal_memory is not None:
                params.extend(layer.terminal_memory.parameters())
        elif self.memory_reader is not None:
            params.extend(self.memory_reader.parameters())
        return params

    def optimization_parameters(self):
        """Parameter set used by the optimizer for the configured training phase."""
        if (
            self.hierarchical_memory_enabled
            and self.config.memory_context.memory_training == "router_only"
        ):
            return self.memory_parameters()
        if not self.progressive_depth_enabled:
            return self.parameters()

        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.active_block_parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_proj.parameters())
        if self.memory_router is not None:
            params.extend(self.memory_router.parameters())
        if self.memory_reader is not None:
            params.extend(self.memory_reader.parameters())
        if self.historical_memory is not None:
            params.extend(self.historical_memory.parameters())
        return params

    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.zero_grad()

    def refresh_compute_buffers(self):
        """Refresh derived compute buffers after trainable weights change."""
        blocks = (
            self.blocks[: self.active_layers]
            if self.progressive_depth_enabled
            else self.blocks
        )
        for block in blocks:
            if hasattr(block, "refresh_compute_buffers"):
                block.refresh_compute_buffers()
    
    @property
    def hierarchical_memory_enabled(self):
        return (self.memory_router is not None or self.historical_memory is not None
                or self.direct_full_history)

    def _pretransformer_context_forward(self, token_ids, memory_metadata=None):
        """Build the dense window after one causal per-position memory read.

        Source layout is ``[historical store][dense training window]``.  For
        training row ``j`` the router query is built from the causal history
        ending at that row, while candidate historical blocks must end before
        the configured recent-history horizon.  Selected old tokens are read
        by indexed cross-attention and reduced to one residual vector per row;
        they are never concatenated into the Transformer sequence.
        """
        cfg = self.config.memory_context
        source_ids = xp.asarray(token_ids)
        if source_ids.ndim != 2:
            raise ValueError(
                "hierarchical memory expects token_ids with shape [B,T_source]"
            )
        expected = int(cfg.source_input_length)
        if int(source_ids.shape[1]) != expected:
            raise ValueError(
                f"hierarchical memory requires {expected} input tokens "
                f"(memory_length + target_length), got {source_ids.shape[1]}"
            )

        batch = int(source_ids.shape[0])
        memory_length = int(cfg.memory_length)
        target_length = int(cfg.target_length)
        recent_length = int(cfg.recent_length)
        history_ids = source_ids[:, :memory_length]
        target_ids = source_ids[:, memory_length:memory_length + target_length]
        if memory_metadata is None:
            history_valid_starts = xp.zeros((batch,), dtype=xp.int64)
        else:
            raw_history_valid_starts = memory_metadata.get("history_valid_starts")
            if raw_history_valid_starts is None:
                history_valid_starts = xp.zeros((batch,), dtype=xp.int64)
            else:
                history_valid_starts = xp.asarray(
                    raw_history_valid_starts, dtype=xp.int64
                )
                if history_valid_starts.shape != (batch,):
                    raise ValueError(
                        "memory_metadata.history_valid_starts must have shape (batch,)"
                    )
                if bool(xp.any(history_valid_starts < 0)) or bool(
                    xp.any(history_valid_starts > memory_length)
                ):
                    raise ValueError("history_valid_starts lie outside memory store")

        # Router queries for the dense target rows need at most the tail recent
        # history plus the target rows themselves.  Avoid retaining/repooling
        # the full 64k tensor for this causal sliding query.
        query_prefix_start = max(0, memory_length - recent_length)
        query_source_ids = source_ids[
            :, query_prefix_start:memory_length + target_length
        ]
        query_prefix_length = memory_length - query_prefix_start
        query_valid_starts = xp.maximum(
            history_valid_starts - query_prefix_start, 0
        )
        # Row j predicts the token after target_ids[j], so its causal routing
        # boundary includes target_ids[j] itself.
        route_starts = (
            query_prefix_length
            + xp.arange(target_length, dtype=xp.int64)
            + 1
        )
        # Absolute sample-local boundary before which blocks are old enough to
        # retrieve.  This is the attention-like causal mask discussed in 0058A.
        current_boundaries = (
            memory_length + xp.arange(target_length, dtype=xp.int64) + 1
        )
        candidate_cutoffs = xp.maximum(
            current_boundaries - recent_length, 0
        )

        with performance_scope("memory_context.history_embedding.forward"):
            history_x = self.embedding.W.data[history_ids]
        with performance_scope("memory_context.query_embedding.forward"):
            query_source_x = self.embedding.W.data[query_source_ids]
        with performance_scope("memory_context.router.forward"):
            (
                route_weights,
                selected_blocks,
                selected_valid,
                route_valid,
                query_pooled,
                router_cache,
            ) = self.memory_router.forward(
                history_x,
                query_source_x,
                route_starts,
                candidate_cutoffs,
                history_valid_starts=history_valid_starts,
                query_valid_starts=query_valid_starts,
            )

        with performance_scope("memory_context.active_build.forward"):
            target_x = self.embedding.W.data[target_ids]
            target_positions = xp.broadcast_to(
                xp.arange(
                    memory_length,
                    memory_length + target_length,
                    dtype=xp.int64,
                )[None, :],
                (batch, target_length),
            )

        with performance_scope("memory_context.reader.forward"):
            memory_out, reader_cache = self.memory_reader.forward(
                history_x,
                target_x,
                selected_blocks,
                route_weights,
                selected_valid,
                history_position_ids=xp.arange(memory_length, dtype=xp.int64),
                query_position_ids=target_positions,
                return_cache=True,
            )

        with performance_scope("memory_context.active_build.forward"):
            active_x = target_x + memory_out * float(cfg.reader_residual_scale)

        # No full-resolution 64k embedding tensor is kept through the deep
        # Transformer.  Reader K/V and compact probability/router caches remain;
        # history embeddings are replayed cheaply for projection weight grads.
        del history_x, query_source_x, target_x, memory_out, query_pooled

        active = ActiveContext(
            embeddings=active_x,
            token_ids=target_ids,
            position_ids=target_positions,
            source_indices=target_positions,
            target_start=0,
            target_end=target_length,
            selected_blocks=selected_blocks,
            route_weights=route_weights,
            selected_valid=selected_valid,
            route_valid=route_valid,
        )
        cache = {
            "mode": "pretransformer_read",
            "source_token_ids": source_ids,
            "history_ids": history_ids,
            "query_source_ids": query_source_ids,
            "target_ids": target_ids,
            "selected_blocks": selected_blocks,
            "route_weights": route_weights,
            "selected_valid": selected_valid,
            "route_valid": route_valid,
            "history_valid_starts": history_valid_starts,
            "router_cache": router_cache,
            "reader_cache": reader_cache,
            "target_slice": (0, target_length),
        }
        return active, cache

    def _pretransformer_context_backward(self, dactive, cache):
        """Backpropagate dense-window, reader, router, and embedding paths."""
        cfg = self.config.memory_context
        history_ids = cache.pop("history_ids")
        query_source_ids = cache.pop("query_source_ids")
        target_ids = cache.pop("target_ids")
        cache.pop("source_token_ids", None)
        cache.pop("selected_blocks", None)
        cache.pop("route_weights", None)
        cache.pop("selected_valid", None)
        cache.pop("route_valid", None)
        cache.pop("history_valid_starts", None)
        reader_cache = cache.pop("reader_cache")
        router_cache = cache.pop("router_cache")

        # The active residual contains the target embedding directly plus the
        # external-memory read.  Replay history embedding only now, after the
        # deep Transformer has released its layer caches.
        d_target_direct = dactive
        d_reader_out = dactive * float(cfg.reader_residual_scale)
        with performance_scope("memory_context.history_embedding.replay"):
            history_x_replay = self.embedding.W.data[history_ids]

        with performance_scope("memory_context.reader.backward"):
            dhistory_reader, dtarget_reader, dweights = self.memory_reader.backward(
                d_reader_out, reader_cache, history_x_replay
            )
        with performance_scope("memory_context.router.backward"):
            dhistory_router, dquery_source = self.memory_router.backward(
                dweights,
                router_cache,
            )

        with performance_scope("memory_context.embedding.backward"):
            # Avoid a third full-history FP32 temporary at the 64k horizon.
            dhistory_reader += dhistory_router.astype(
                dhistory_reader.dtype, copy=False
            )
            self.embedding.backward(
                dhistory_reader,
                {"token_ids": history_ids},
            )
            self.embedding.backward(
                dquery_source, {"token_ids": query_source_ids}
            )
            self.embedding.backward(
                d_target_direct + dtarget_reader, {"token_ids": target_ids}
            )

        del (
            history_x_replay,
            dhistory_reader,
            dhistory_router,
            dquery_source,
            dtarget_reader,
            dweights,
        )

    def _direct_history_length(self, available, direct_limit):
        """0078I: floor to a bounded shape set, NEVER pad unknown history.

        The exact direct_limit (cfg.retrieved_length) is the pre-existing
        maximum prefix length. If the optional bucket is larger than available
        history, use the dense 4k window (zero prefix); no padded tokens enter
        attention. Router-selected full blocks are not modified.
        """
        length = min(int(available), int(direct_limit))
        bucket = self.direct_history_bucket_tokens
        if bucket > 1:
            length = (length // bucket) * bucket
        return length

    def _routed_prefix_context_forward(self, token_ids, memory_metadata=None):
        """0060A/B2: construct a real historical prefix plus the dense 4k window.

        Source layout is ``[searchable same-document history][current window]``.
        When no selection decision is needed, every available historical token
        is reopened directly (up to the configured retrieved-token budget).
        Once history is large enough to require compression, the shallow router
        selects K complete old blocks exactly as in 0060A.  Router scores,
        probabilities, and selection rank are never exposed to the Transformer.
        """
        cfg = self.config.memory_context
        source_ids = xp.asarray(token_ids)
        if source_ids.ndim != 2 or int(source_ids.shape[1]) != int(cfg.source_input_length):
            raise ValueError(
                f"routed-prefix memory requires [B,{cfg.source_input_length}] source IDs"
            )
        batch = int(source_ids.shape[0])
        history_length = int(cfg.distant_memory_length)
        working_length = int(cfg.target_length)
        if memory_metadata is not None and 'compact_working_length' in memory_metadata:
            # Both block refinement and 0080BN AR compaction retain the full
            # source buffer, router coordinates, and original document sample.
            block_compact = (memory_metadata.get('causal_prefix_query') is True and
                             memory_metadata.get('causal_masked_target_refinement') is True)
            ar_compact = memory_metadata.get('compact_ar_target') is True
            if not (block_compact or ar_compact):
                raise ValueError('compact working window requires causal block refinement or compact AR')
            if ar_compact and (block_compact or memory_metadata.get('causal_prefix_query', False)):
                raise ValueError('compact AR cannot enable block proposal refinement')
            working_length = int(memory_metadata['compact_working_length'])
            if working_length <= 0 or working_length > int(cfg.target_length):
                raise ValueError('compact working length outside configured target window')
        history_ids = source_ids[:, :history_length]
        working_ids = source_ids[:, history_length:history_length + working_length]

        # 0079Q1: a block-refinement training window must not choose its
        # historical prefix using *any* real token within that window. This
        # opt-in mode asks the unchanged production router using the last
        # router_query_length tokens of the PRE-WINDOW history. The normal
        # autoregressive training/inference behavior stays unchanged.
        causal_prefix_query = bool(
            cfg.routed_prefix_causal_policy == "prewindow"
            or (memory_metadata is not None and
                memory_metadata.get("causal_prefix_query", False))
        )
        query_ids = None
        if causal_prefix_query:
            query_length = int(cfg.router_query_length)
            if query_length > history_length:
                raise ValueError(
                    "causal_prefix_query needs router_query_length <= history length"
                )
            query_ids = history_ids[:, history_length - query_length:]

        if memory_metadata is None:
            history_valid_starts = xp.zeros((batch,), dtype=xp.int64)
            target_valid_lengths = xp.full((batch,), working_length, dtype=xp.int64)
            use_retrieval = True
            stochastic = False
            gumbel_seed = None
            temperature = float(cfg.router_temperature)
        else:
            history_valid_starts = xp.asarray(
                memory_metadata.get(
                    "history_valid_starts", xp.zeros((batch,), dtype=xp.int64)
                ),
                dtype=xp.int64,
            )
            target_valid_lengths = xp.asarray(
                memory_metadata.get(
                    "target_valid_lengths", xp.full((batch,), working_length, dtype=xp.int64)
                ),
                dtype=xp.int64,
            )
            use_retrieval = bool(memory_metadata.get("use_retrieval", True))
            stochastic = bool(memory_metadata.get("router_stochastic", False))
            gumbel_seed = memory_metadata.get("router_gumbel_seed")
            temperature = float(
                memory_metadata.get("router_temperature", cfg.router_temperature)
            )
        if cfg.memory_training == "disabled":
            use_retrieval = False
        if history_valid_starts.shape != (batch,) or target_valid_lengths.shape != (batch,):
            raise ValueError("hierarchical memory metadata batch shape mismatch")
        masked_refinement = bool(
            memory_metadata is not None
            and memory_metadata.get("causal_masked_target_refinement", False)
        )
        if masked_refinement and not causal_prefix_query:
            raise ValueError("masked refinement requires causal_prefix_query")
        if bool(xp.any((target_valid_lengths < 0) | (target_valid_lengths > working_length))):
            raise ValueError("invalid target_valid_lengths")
        if use_retrieval and bool(xp.any(target_valid_lengths != working_length)) and not masked_refinement:
            # The ordinary autoregressive model has no per-row padding mask.
            # Q4 instead applies explicit valid-block masks to every provisional
            # read, and the causal trunk cannot expose tail padding to earlier
            # valid positions. Keep the production behavior unchanged otherwise.
            use_retrieval = False
        insufficient_pre_window_query = bool(xp.any(
            history_valid_starts > history_length - int(cfg.router_query_length)
        ))
        if causal_prefix_query and use_retrieval and insufficient_pre_window_query and not masked_refinement:
            if cfg.routed_prefix_causal_policy == "prewindow":
                # Document-first sampling often yields <4K genuine preceding
                # tokens. Never route those samples using target-window text.
                use_retrieval = False
            else:
                # Preserve strict Q1 opt-in validation on legacy checkpoints.
                raise ValueError(
                "causal_prefix_query requires a fully valid pre-window router query"
            )

        with performance_scope("memory_context.working_embedding.forward"):
            working_x = self.embedding.W.data[working_ids]
        working_positions = xp.broadcast_to(
            xp.arange(
                history_length,
                history_length + working_length,
                dtype=xp.int64,
            )[None, :],
            (batch, working_length),
        )

        selected_blocks = xp.zeros((batch, int(cfg.top_k_blocks)), dtype=xp.int64)
        selected_valid = xp.zeros(selected_blocks.shape, dtype=bool)
        route_weights = xp.zeros(selected_blocks.shape, dtype=working_x.dtype)
        route_valid = xp.zeros((batch,), dtype=bool)
        router_cache = None
        selected_token_ids = None
        prefix_length = 0
        prefix_mode = "none"

        # The document-aware sampler right-aligns genuine history in the fixed
        # external store.  If the entire available history fits in the 2k
        # retrieved budget, there is no useful routing decision: reopen every
        # real token directly.  Production partial-history batches are
        # homogeneous; direct callers with ragged per-example history fall back
        # to the dense window rather than introducing unmasked padding rows.
        available_history = history_length - history_valid_starts
        direct_limit = int(cfg.retrieved_length)
        route_threshold = direct_limit + int(cfg.block_size) - 1
        max_available = int(xp.max(available_history).item())
        min_available = int(xp.min(available_history).item())
        if use_retrieval and min_available <= 0:
            use_retrieval = False
        if use_retrieval and min_available > 0:
            if max_available < route_threshold or (
                masked_refinement and insufficient_pre_window_query
            ):
                # With ragged genuine histories the router query cannot be
                # constructed from an invalid/padded pre-window prefix.
                # Instead reopen the largest common genuine history suffix.
                # 0078E: different documents in one microbatch may have
                # different amounts of genuine history. Without a per-row
                # attention-padding mask, use the longest suffix valid for
                # *every* sample rather than dropping all direct history.
                prefix_length = self._direct_history_length(min_available, direct_limit)
                if prefix_length == 0:
                    use_retrieval = False
                else:
                    prefix_start = history_length - prefix_length
                    selected_token_ids = history_ids[:, prefix_start:history_length]
                    with performance_scope("memory_context.history_gather.forward"):
                        selected_x = self.embedding.W.data[selected_token_ids]
                    prefix_positions = xp.broadcast_to(
                        xp.arange(prefix_start, history_length, dtype=xp.int64)[None, :],
                        (batch, prefix_length),
                    )
                    active_x = xp.concatenate((selected_x, working_x), axis=1)
                    active_ids = xp.concatenate((selected_token_ids, working_ids), axis=1)
                    active_positions = xp.concatenate(
                        (prefix_positions, working_positions), axis=1
                    )
                    prefix_mode = "direct_history"
                    del selected_x

        if use_retrieval and prefix_mode == "none":
            with performance_scope("memory_context.history_embedding.forward"):
                history_x = self.embedding.W.data[history_ids]
            with performance_scope("memory_context.router.forward"):
                (
                    route_weights,
                    selected_blocks,
                    selected_valid,
                    route_valid,
                    router_cache,
                ) = self.memory_router.forward_routed_prefix(
                    history_x,
                    (self.embedding.W.data[query_ids]
                     if causal_prefix_query else working_x),
                    history_valid_starts=history_valid_starts,
                    stochastic=(stochastic and bool(cfg.router_gumbel_noise)),
                    gumbel_seed=gumbel_seed,
                    temperature=temperature,
                )

            # If there are not yet K complete block-aligned candidates but the
            # raw history already exceeds the exact-prefix budget by less than a
            # block, keep the newest 2k exact tokens.  This avoids dropping all
            # memory during the narrow transition into true routed operation.
            if not bool(xp.all(selected_valid)):
                router_cache = None
                selected_blocks[...] = 0
                selected_valid[...] = False
                route_weights[...] = 0
                route_valid[...] = False
                # Same safe common-prefix policy as the direct path: only
                # real same-document history can reach the Transformer.
                prefix_length = self._direct_history_length(min_available, direct_limit)
                if prefix_length > 0:
                    prefix_start = history_length - prefix_length
                    selected_token_ids = history_ids[:, prefix_start:history_length]
                    with performance_scope("memory_context.history_gather.forward"):
                        selected_x = self.embedding.W.data[selected_token_ids]
                    prefix_positions = xp.broadcast_to(
                        xp.arange(prefix_start, history_length, dtype=xp.int64)[None, :],
                        (batch, prefix_length),
                    )
                    active_x = xp.concatenate((selected_x, working_x), axis=1)
                    active_ids = xp.concatenate((selected_token_ids, working_ids), axis=1)
                    active_positions = xp.concatenate(
                        (prefix_positions, working_positions), axis=1
                    )
                    prefix_mode = "direct_history"
                    del selected_x
                else:
                    use_retrieval = False
            else:
                block_size = int(cfg.block_size)
                offsets = xp.arange(block_size, dtype=xp.int64)
                selected_positions = (
                    selected_blocks[..., None] * block_size + offsets
                ).reshape(batch, -1)
                batch_ids = xp.arange(batch, dtype=xp.int64)[:, None]
                selected_token_ids = history_ids[batch_ids, selected_positions]
                with performance_scope("memory_context.history_gather.forward"):
                    selected_x = self.embedding.W.data[selected_token_ids]
                prefix_length = int(cfg.retrieved_length)
                active_x = xp.concatenate((selected_x, working_x), axis=1)
                active_ids = xp.concatenate((selected_token_ids, working_ids), axis=1)
                active_positions = xp.concatenate(
                    (selected_positions, working_positions), axis=1
                )
                prefix_mode = "routed_blocks"
                del selected_x
            del history_x

        if not use_retrieval:
            active_x = working_x
            active_ids = working_ids
            active_positions = working_positions
            prefix_length = 0
            prefix_mode = "none"

        active = ActiveContext(
            embeddings=active_x,
            token_ids=active_ids,
            position_ids=active_positions,
            source_indices=active_positions,
            target_start=prefix_length,
            target_end=prefix_length + working_length,
            selected_blocks=selected_blocks,
            route_weights=route_weights,
            selected_valid=selected_valid,
            route_valid=route_valid,
            terminal_memory=None,
        )
        cache = {
            "mode": "routed_prefix",
            "history_ids": history_ids,
            "working_ids": working_ids,
            "causal_prefix_query_ids": query_ids,
            "selected_token_ids": selected_token_ids,
            "history_valid_starts": history_valid_starts,
            "router_cache": router_cache,
            "selected_blocks": selected_blocks,
            "route_weights": route_weights,
            "selected_valid": selected_valid,
            "route_valid": route_valid,
            "retrieval_active": bool(prefix_length > 0),
            "prefix_mode": prefix_mode,
            "allow_unaligned_refinement_blocks": bool(masked_refinement),
            "prefix_length": int(prefix_length),
            "target_slice": (prefix_length, prefix_length + working_length),
        }
        return active, cache

    def _routed_prefix_context_backward(self, dactive, cache):
        """Backward for exact direct-history and ST-routed historical prefixes."""
        working_ids = cache.pop("working_ids")
        history_ids = cache.pop("history_ids")
        query_ids = cache.pop("causal_prefix_query_ids", None)
        selected_token_ids = cache.pop("selected_token_ids")
        router_cache = cache.pop("router_cache")
        cache.pop("selected_blocks", None)
        cache.pop("route_weights", None)
        cache.pop("selected_valid", None)
        cache.pop("route_valid", None)
        retrieval_active = bool(cache.pop("retrieval_active"))
        prefix_mode = cache.pop("prefix_mode", "routed_blocks")
        prefix_length = int(cache.pop("prefix_length"))
        cache.pop("history_valid_starts", None)
        cache.pop("allow_unaligned_refinement_blocks", None)

        if not retrieval_active:
            with performance_scope("memory_context.embedding.backward"):
                self.embedding.backward(dactive, {"token_ids": working_ids})
            return

        dselected = dactive[:, :prefix_length, :]
        dworking_direct = dactive[:, prefix_length:, :]
        if prefix_mode == "direct_history":
            # No selection choice exists while all useful history fits inside
            # the prefix budget, so there is deliberately no router gradient.
            with performance_scope("memory_context.embedding.backward"):
                self.embedding.backward(dselected, {"token_ids": selected_token_ids})
                self.embedding.backward(dworking_direct, {"token_ids": working_ids})
            return

        with performance_scope("memory_context.history_gather.replay"):
            selected_x = self.embedding.W.data[selected_token_ids]
        with performance_scope("memory_context.router.backward"):
            dhistory_router, dworking_router = self.memory_router.backward_routed_prefix(
                dselected,
                selected_x,
                router_cache,
            )
        with performance_scope("memory_context.embedding.backward"):
            # Direct deep-prefix gradient goes only to exact reopened tokens.
            # The straight-through router also backpropagates through *all*
            # candidate history embeddings. In 0079Q1, its query is historical
            # too: routing gradients must NEVER be written into future targets.
            self.embedding.backward(dhistory_router, {"token_ids": history_ids})
            self.embedding.backward(dselected, {"token_ids": selected_token_ids})
            if query_ids is not None:
                self.embedding.backward(dworking_router, {"token_ids": query_ids})
                self.embedding.backward(dworking_direct, {"token_ids": working_ids})
            else:
                dworking_total = dworking_direct + dworking_router.astype(
                    dworking_direct.dtype, copy=False
                )
                self.embedding.backward(dworking_total, {"token_ids": working_ids})
        del selected_x, dhistory_router, dworking_router

    def _terminal_landmark_context_forward(self, token_ids, memory_metadata=None):
        """0058C: route once from the full working window, keep exact old K/V.

        Source layout is ``[old external store][working 4k]``. ``memory_length``
        denotes the total causal horizon, so the external store has
        ``memory_length - target_length`` rows.  No historical row is inserted
        into the Transformer sequence; selected exact embeddings are retained
        only for the terminal attention override in the configured layer.
        """
        cfg = self.config.memory_context
        source_ids = xp.asarray(token_ids)
        if source_ids.ndim != 2 or int(source_ids.shape[1]) != int(cfg.source_input_length):
            raise ValueError(
                f"terminal Landmark memory requires [B,{cfg.source_input_length}] source IDs"
            )
        batch = int(source_ids.shape[0])
        history_length = int(cfg.distant_memory_length)
        working_length = int(cfg.target_length)
        if memory_metadata is not None and 'compact_working_length' in memory_metadata:
            # Both block refinement and 0080BN AR compaction retain the full
            # source buffer, router coordinates, and original document sample.
            block_compact = (memory_metadata.get('causal_prefix_query') is True and
                             memory_metadata.get('causal_masked_target_refinement') is True)
            ar_compact = memory_metadata.get('compact_ar_target') is True
            if not (block_compact or ar_compact):
                raise ValueError('compact working window requires causal block refinement or compact AR')
            if ar_compact and (block_compact or memory_metadata.get('causal_prefix_query', False)):
                raise ValueError('compact AR cannot enable block proposal refinement')
            working_length = int(memory_metadata['compact_working_length'])
            if working_length <= 0 or working_length > int(cfg.target_length):
                raise ValueError('compact working length outside configured target window')
        history_ids = source_ids[:, :history_length]
        working_ids = source_ids[:, history_length:history_length + working_length]

        if memory_metadata is None:
            history_valid_starts = xp.zeros((batch,), dtype=xp.int64)
            target_valid_lengths = xp.full((batch,), working_length, dtype=xp.int64)
        else:
            history_valid_starts = xp.asarray(
                memory_metadata.get(
                    "history_valid_starts", xp.zeros((batch,), dtype=xp.int64)
                ), dtype=xp.int64,
            )
            target_valid_lengths = xp.asarray(
                memory_metadata.get(
                    "target_valid_lengths", xp.full((batch,), working_length, dtype=xp.int64)
                ), dtype=xp.int64,
            )
        if history_valid_starts.shape != (batch,) or target_valid_lengths.shape != (batch,):
            raise ValueError("hierarchical memory metadata batch shape mismatch")
        terminal_rows = xp.maximum(target_valid_lengths - 1, 0)

        with performance_scope("memory_context.working_embedding.forward"):
            working_x = self.embedding.W.data[working_ids]
        selected_blocks = xp.zeros((batch, int(cfg.top_k_blocks)), dtype=xp.int64)
        selected_valid = xp.zeros(selected_blocks.shape, dtype=bool)
        route_weights = xp.zeros(selected_blocks.shape, dtype=working_x.dtype)
        gate_scores = xp.zeros(selected_blocks.shape, dtype="float32")
        route_valid = xp.zeros((batch,), dtype=bool)
        router_cache = None
        terminal_memory = None

        if cfg.memory_training != "disabled":
            with performance_scope("memory_context.history_embedding.forward"):
                history_x = self.embedding.W.data[history_ids]
            with performance_scope("memory_context.router.forward"):
                (
                    route_weights, selected_blocks, gate_scores,
                    selected_valid, route_valid, router_cache,
                ) = self.memory_router.forward_terminal(
                    history_x, working_x,
                    history_valid_starts=history_valid_starts,
                    terminal_rows=terminal_rows,
                )
            if bool(xp.any(route_valid & (target_valid_lengths != working_length))):
                raise ValueError(
                    "a terminal Landmark route may only be active for a complete "
                    "working window; otherwise padded future rows could enter the router query"
                )

            # Reopen only K*block_size exact historical tokens. This is the only
            # full-resolution old-memory tensor retained through the deep trunk.
            block_size = int(cfg.block_size)
            offsets = xp.arange(block_size, dtype=xp.int64)
            selected_positions = (
                selected_blocks[..., None] * block_size + offsets
            ).reshape(batch, -1)
            safe_positions = xp.where(
                xp.repeat(selected_valid, block_size, axis=-1),
                selected_positions,
                0,
            )
            batch_ids = xp.arange(batch, dtype=xp.int64)[:, None]
            selected_token_ids = history_ids[batch_ids, safe_positions]
            with performance_scope("memory_context.history_gather.forward"):
                selected_embeddings = self.embedding.W.data[selected_token_ids]
            terminal_memory = TerminalMemoryContext(
                selected_embeddings=selected_embeddings,
                selected_token_ids=selected_token_ids,
                selected_position_ids=safe_positions,
                selected_blocks=selected_blocks,
                gate_scores=gate_scores,
                route_weights=route_weights,
                selected_valid=selected_valid,
                route_valid=route_valid,
                terminal_rows=terminal_rows,
            )
            del history_x

        working_positions = xp.broadcast_to(
            xp.arange(history_length, history_length + working_length, dtype=xp.int64)[None, :],
            (batch, working_length),
        )
        active = ActiveContext(
            embeddings=working_x,
            token_ids=working_ids,
            position_ids=working_positions,
            source_indices=working_positions,
            target_start=0,
            target_end=working_length,
            selected_blocks=selected_blocks,
            route_weights=route_weights,
            selected_valid=selected_valid,
            route_valid=route_valid,
            terminal_memory=terminal_memory,
        )
        cache = {
            "mode": "terminal_landmark",
            "history_ids": history_ids,
            "working_ids": working_ids,
            "history_valid_starts": history_valid_starts,
            "router_cache": router_cache,
            "terminal_memory": terminal_memory,
            "target_slice": (0, working_length),
        }
        return active, cache

    def _terminal_landmark_context_backward(self, dworking, cache):
        """Finish embedding/router backward after the 4k Transformer trunk."""
        working_ids = cache.pop("working_ids")
        history_ids = cache.pop("history_ids")
        terminal_memory = cache.pop("terminal_memory")
        router_cache = cache.pop("router_cache")
        cache.pop("history_valid_starts", None)

        dworking_total = dworking
        if terminal_memory is not None and router_cache is not None:
            dgate = terminal_memory.d_gate_scores
            if dgate is None:
                dgate = xp.zeros(terminal_memory.gate_scores.shape, dtype="float32")
            with performance_scope("memory_context.router.backward"):
                dhistory_router, dworking_router = self.memory_router.backward_terminal(
                    dgate, router_cache
                )
            dworking_total = dworking_total + dworking_router.astype(
                dworking_total.dtype, copy=False
            )
            if self.config.memory_context.memory_training != "router_only":
                with performance_scope("memory_context.embedding.backward"):
                    self.embedding.backward(dhistory_router, {"token_ids": history_ids})
                    if terminal_memory.d_selected_embeddings is not None:
                        self.embedding.backward(
                            terminal_memory.d_selected_embeddings,
                            {"token_ids": terminal_memory.selected_token_ids},
                        )
        if self.config.memory_context.memory_training != "router_only":
            with performance_scope("memory_context.embedding.backward"):
                self.embedding.backward(dworking_total, {"token_ids": working_ids})

    def _historical_context_forward(self, token_ids, memory_metadata=None, *, inference=False):
        """Keep all 4096 current rows; block-encode preceding same-document history."""
        cfg = self.config.memory_context
        source = xp.asarray(token_ids)
        if source.ndim != 2 or source.shape[1] != cfg.source_input_length:
            raise ValueError("historical_read requires the complete source horizon")
        batch, _ = source.shape
        h = int(cfg.distant_memory_length)
        w = int(cfg.target_length)
        if memory_metadata is not None and memory_metadata.get("compact_ar_target"):
            w = int(memory_metadata["compact_working_length"])
            if not 0 < w <= int(cfg.target_length):
                raise ValueError("invalid compact AR length")
        starts = (xp.zeros((batch,), dtype=xp.int64) if memory_metadata is None
                  else xp.asarray(memory_metadata.get("history_valid_starts",
                       xp.zeros((batch,), dtype=xp.int64)), dtype=xp.int64))
        working_ids = source[:, h:h+w]
        valid_mask = (None if memory_metadata is None else
                      memory_metadata.get("history_valid_mask"))
        valid_lengths = (xp.full((batch,), w, dtype=xp.int64)
                         if memory_metadata is None else xp.asarray(
                             memory_metadata.get("target_valid_lengths",
                                                 xp.full((batch,), w, dtype=xp.int64)),
                             dtype=xp.int64))
        if valid_lengths.shape != (batch,) or bool(xp.any(
            (valid_lengths < 0) | (valid_lengths > w)
        )):
            raise ValueError("invalid historical_read target_valid_lengths")
        current_valid = xp.arange(w)[None, :] < valid_lengths[:, None]
        state = self.historical_memory.open(
            source[:, :h], starts, valid_mask, inference=inference
        )
        pos = xp.broadcast_to(xp.arange(h, h+w, dtype=xp.int64)[None, :],
                              (batch, w))
        active = ActiveContext(
            embeddings=self.embedding.W.data[working_ids], token_ids=working_ids,
            position_ids=pos, source_indices=pos, target_start=0,
            target_end=w, selected_blocks=None, route_weights=None,
            selected_valid=None, route_valid=None, terminal_memory=None,
        )
        return active, {"mode": "historical_read", "working_ids": working_ids,
                        "state": state, "target_slice": (0, w),
                        "history_valid_starts": starts,
                        "current_valid_mask": current_valid}

    def _historical_context_backward(self, dactive, cache):
        self.embedding.backward(dactive, {"token_ids": cache["working_ids"]})
        self.historical_memory.close(cache["state"])

    def _direct_full_history_context_forward(self, token_ids, memory_metadata=None):
        """Expose raw pre-window IDs only to the rectangular replacement heads."""
        cfg = self.config.memory_context
        source = xp.asarray(token_ids)
        if source.ndim != 2 or source.shape[1] != cfg.source_input_length:
            raise ValueError("direct_full_history requires the complete source horizon")
        batch = int(source.shape[0])
        history_length = int(cfg.distant_memory_length)
        working_length = int(cfg.target_length)
        if memory_metadata is not None and memory_metadata.get("compact_ar_target"):
            working_length = int(memory_metadata["compact_working_length"])
        starts = (xp.zeros((batch,), dtype=xp.int64) if memory_metadata is None
                  else xp.asarray(memory_metadata.get(
                      "history_valid_starts", xp.zeros((batch,), dtype=xp.int64)
                  ), dtype=xp.int64))
        valid_lengths = (xp.full((batch,), working_length, dtype=xp.int64)
                         if memory_metadata is None else xp.asarray(
                             memory_metadata.get(
                                 "target_valid_lengths",
                                 xp.full((batch,), working_length, dtype=xp.int64),
                             ), dtype=xp.int64))
        if starts.shape != (batch,) or valid_lengths.shape != (batch,):
            raise ValueError("direct_full_history validity metadata must be per-sample")
        history_valid = xp.arange(history_length)[None, :] >= starts[:, None]
        if memory_metadata is not None and memory_metadata.get("history_valid_mask") is not None:
            history_valid &= xp.asarray(
                memory_metadata["history_valid_mask"], dtype=xp.bool_
            )
        current_valid = (
            xp.arange(working_length)[None, :] < valid_lengths[:, None]
        )
        history_ids = source[:, :history_length]
        working_ids = source[:, history_length:history_length + working_length]
        current_positions = xp.broadcast_to(
            xp.arange(
                history_length, history_length + working_length, dtype=xp.int64
            )[None, :],
            (batch, working_length),
        )
        state = SimpleNamespace(
            ids=history_ids,
            valid=history_valid,
            positions=xp.arange(history_length, dtype=xp.int64),
        )
        active = ActiveContext(
            embeddings=self.embedding.W.data[working_ids], token_ids=working_ids,
            position_ids=current_positions, source_indices=current_positions,
            target_start=0, target_end=working_length, selected_blocks=None,
            route_weights=None, selected_valid=None, route_valid=None,
            terminal_memory=None,
        )
        return active, {
            "mode": "direct_full_history", "working_ids": working_ids,
            "state": state, "target_slice": (0, working_length),
            "history_valid_starts": starts,
            "current_valid_mask": current_valid,
        }

    def _direct_full_history_context_backward(self, dactive, cache):
        self.embedding.backward(dactive, {"token_ids": cache["working_ids"]})

    def _hierarchical_context_forward(self, token_ids, memory_metadata=None, *, inference=False):
        mode = self.config.memory_context.integration_mode
        if mode == "historical_read":
            return self._historical_context_forward(
                token_ids, memory_metadata=memory_metadata, inference=inference
            )
        if mode == "direct_full_history":
            return self._direct_full_history_context_forward(
                token_ids, memory_metadata=memory_metadata
            )
        if mode == "terminal_landmark":
            return self._terminal_landmark_context_forward(
                token_ids, memory_metadata=memory_metadata
            )
        if mode == "routed_prefix":
            return self._routed_prefix_context_forward(
                token_ids, memory_metadata=memory_metadata
            )
        return self._pretransformer_context_forward(
            token_ids, memory_metadata=memory_metadata
        )

    def _hierarchical_context_backward(self, dactive, cache):
        mode = cache.pop("mode", None)
        if mode == "historical_read":
            return self._historical_context_backward(dactive, cache)
        if mode == "direct_full_history":
            return self._direct_full_history_context_backward(dactive, cache)
        if mode == "terminal_landmark":
            return self._terminal_landmark_context_backward(dactive, cache)
        if mode == "routed_prefix":
            return self._routed_prefix_context_backward(dactive, cache)
        return self._pretransformer_context_backward(dactive, cache)

    def memory_routing_diagnostics(self, cache):
        """Return compact memory-router diagnostics for 0058A/B or 0058C."""
        memory_cache = cache.get("memory_context_cache")
        if memory_cache is None:
            return None
        selected = memory_cache.get("selected_blocks")
        if selected is None:
            terminal_memory = memory_cache.get("terminal_memory")
            if terminal_memory is None:
                return None
            selected = terminal_memory.selected_blocks
            weights = terminal_memory.route_weights.astype("float32", copy=False)
            selected_valid = terminal_memory.selected_valid
            route_valid = terminal_memory.route_valid
            terminal_mode = True
        else:
            weights = memory_cache["route_weights"].astype("float32", copy=False)
            selected_valid = memory_cache["selected_valid"]
            route_valid = memory_cache["route_valid"]
            terminal_mode = selected.ndim == 2
        cfg = self.config.memory_context

        probs = xp.maximum(weights, xp.asarray(1e-12, dtype=weights.dtype))
        entropy = -xp.sum(
            xp.where(selected_valid, weights * xp.log(probs), 0.0), axis=-1
        )
        valid_count = xp.maximum(xp.sum(route_valid), 1)
        entropy_mean = xp.sum(entropy * route_valid.astype("float32")) / valid_count

        block_centers = (selected.astype("float32") + 0.5) * int(cfg.block_size)
        if terminal_mode:
            current = float(cfg.distant_memory_length + cfg.target_length)
            distance = current - block_centers
        else:
            current_boundaries = (
                int(cfg.memory_length)
                + xp.arange(int(selected.shape[1]), dtype="float32")
                + 1.0
            )[None, :, None]
            distance = current_boundaries - block_centers
        valid_selected_count = xp.maximum(xp.sum(selected_valid), 1)
        source_distance_mean = xp.sum(
            xp.where(selected_valid, distance, 0.0)
        ) / valid_selected_count

        if bool(xp.any(selected_valid)):
            flattened = selected[selected_valid]
            histogram = xp.bincount(flattened, minlength=int(cfg.searchable_blocks))
            unique_blocks = xp.unique(flattened).size
            recent_cut = max(0, int(cfg.searchable_blocks) - max(1, int(cfg.searchable_blocks) // 4))
            recent_fraction = xp.mean((flattened >= recent_cut).astype("float32"))
        else:
            histogram = xp.zeros((int(cfg.searchable_blocks),), dtype=xp.int64)
            unique_blocks = 0
            recent_fraction = xp.asarray(0.0, dtype="float32")

        sort_order = xp.argsort(selected, axis=-1)
        sorted_selected = xp.take_along_axis(selected, sort_order, axis=-1)
        sorted_valid = xp.take_along_axis(selected_valid, sort_order, axis=-1)
        duplicate_mask = (
            (xp.diff(sorted_selected, axis=-1) == 0)
            & sorted_valid[..., 1:] & sorted_valid[..., :-1]
        )
        result = {
            "selected_blocks": selected,
            "weights": weights,
            "route_valid": route_valid,
            "selected_valid": selected_valid,
            "valid_route_fraction": xp.mean(route_valid.astype("float32")),
            "selected_slot_fraction": xp.mean(selected_valid.astype("float32")),
            "selected_block_histogram": histogram,
            "entropy_mean": entropy_mean,
            "source_distance_mean": source_distance_mean,
            "recent_quartile_fraction": recent_fraction,
            "duplicate_count": xp.sum(duplicate_mask),
            "unique_blocks": unique_blocks,
        }
        router_cache = memory_cache.get("router_cache")
        if router_cache is not None and router_cache.get("scores") is not None:
            scores = router_cache["scores"].astype("float32", copy=False)
            candidate_mask = router_cache.get("candidate_mask")
            if candidate_mask is not None:
                score_values = scores[xp.broadcast_to(candidate_mask, scores.shape)]
            else:
                score_values = scores.reshape(-1)
            if score_values.size:
                result.update({
                    "score_mean": xp.mean(score_values),
                    "score_std": xp.std(score_values),
                    "score_min": xp.min(score_values),
                    "score_max": xp.max(score_values),
                })
        terminal_memory = memory_cache.get("terminal_memory")
        if terminal_memory is not None:
            valid = terminal_memory.route_valid.astype("float32")
            denom = xp.maximum(xp.sum(valid), 1.0)
            if terminal_memory.attention_history_mass is not None:
                per_batch = xp.mean(
                    terminal_memory.attention_history_mass.astype("float32"), axis=1
                )
                result["history_attention_mass_mean"] = xp.sum(per_batch * valid) / denom
            if terminal_memory.attention_block_probs is not None:
                per_batch = xp.max(
                    terminal_memory.attention_block_probs.astype("float32"), axis=(1, 2)
                )
                result["max_block_attention_mean"] = xp.sum(per_batch * valid) / denom
            if terminal_memory.attention_max_token_prob is not None:
                per_batch = xp.max(
                    terminal_memory.attention_max_token_prob.astype("float32"), axis=(1, 2)
                )
                result["max_within_block_token_prob_mean"] = xp.sum(per_batch * valid) / denom
            if terminal_memory.d_gate_scores is not None:
                result["gate_grad_norm"] = xp.sqrt(
                    xp.sum(terminal_memory.d_gate_scores.astype("float32") ** 2)
                )
        return result

    def forward_body(
        self, token_ids, finite_trace=None, return_cache=True,
        activation_checkpoint=False, position_ids=None, memory_metadata=None,
        return_target_slice=False, input_embeddings=None, attention_allow_mask=None,
        block_context_length=None, block_valid_mask=None,
        dual_resolution_metadata=None,
    ):
        """Run embedding/Transformer/final-norm without materializing logits.

        0058A optionally performs causal per-position external-memory reads
        before the deep trunk.  The Transformer receives only the dense current
        training window; explicit source positions preserve its location within
        the long sample for RoPE.
        """
        if input_embeddings is not None and self.hierarchical_memory_enabled:
            raise NotImplementedError(
                "external active-block embeddings require direct dense context; "
                "routed-memory support needs separate causal cache integration"
            )
        if block_context_length is not None:
            if self.hierarchical_memory_enabled:
                raise NotImplementedError("0079F does not support routed hierarchical memory")
            if attention_allow_mask is not None:
                raise ValueError("use either dense attention_allow_mask or compact block mode")
            if input_embeddings is None:
                raise ValueError("compact refinement requires explicit input embeddings")
        if attention_allow_mask is not None and self.hierarchical_memory_enabled:
            raise NotImplementedError("block masks do not support routed-memory mode")
        memory_context_cache = None
        target_slice = None
        terminal_memory = None
        historical_state = None
        full_history_current_valid_mask = None
        if self.hierarchical_memory_enabled:
            if position_ids is not None:
                raise ValueError(
                    "explicit position_ids are constructed internally for "
                    "hierarchical-memory models"
                )
            with performance_scope("memory_context.forward"):
                active, memory_context_cache = self._hierarchical_context_forward(
                    token_ids, memory_metadata=memory_metadata,
                    inference=not return_cache,
                )
            x = active.embeddings
            effective_position_ids = active.position_ids
            target_slice = (active.target_start, active.target_end)
            if (
                self.config.memory_context.integration_mode == "terminal_landmark"
                and self.config.memory_context.memory_training == "router_only"
            ):
                # Post-training router optimization materializes only the one
                # terminal vocabulary row whose loss is allowed to train memory.
                target_slice = (active.target_end - 1, active.target_end)
            terminal_memory = active.terminal_memory
            if self.historical_memory is not None or self.direct_full_history:
                historical_state = memory_context_cache["state"]
                full_history_current_valid_mask = memory_context_cache.get(
                    "current_valid_mask"
                )
            embed_cache = None
            if finite_trace is not None:
                finite_trace.append(
                    ("memory_active_embedding", xp.all(xp.isfinite(x)))
                )
        else:
            if memory_metadata is not None:
                raise ValueError(
                    "memory_metadata is only valid for hierarchical-memory models"
                )
            if input_embeddings is not None:
                if input_embeddings.ndim != 3 or input_embeddings.shape[-1] != self.config.d_model:
                    raise ValueError("input_embeddings must have shape [B,T,d_model]")
                x = input_embeddings
                embed_cache = None
            else:
                with performance_scope("model.embedding.forward"):
                    x, embed_cache = self.embedding.forward(token_ids)
            effective_position_ids = position_ids
            if finite_trace is not None:
                finite_trace.append(("embedding", xp.all(xp.isfinite(x))))

        # The independent reader requires original token coordinates, even when
        # the regular attention path replaces positions with prepared RoPE.
        full_history_positions = (effective_position_ids if
            self.config.full_history_layers else None)

        # 0080BU: validate/index explicit source positions ONCE per active
        # context, then reuse the same Q/K RoPE factors across all deep layers.
        # This leaves compact routed-history coordinates exactly unchanged.
        # Checkpoint replay retains the prepared table via position_ids below.
        if (effective_position_ids is not None and
                os.environ.get("MINI_LLM_ROPE_PREPARED_POSITIONS", "0").strip().lower()
                not in {"0", "false", "off", "no", ""}):
            with performance_scope("model.rope_positions.prepare"):
                effective_position_ids = prepare_rope_positions(
                    effective_position_ids, d_head=self.config.d_head,
                    base=self.config.rope_base, dtype=self.dtype,
                )

        # 0057 block checkpointing now retains only bounded active-sequence
        # residual inputs in hierarchical mode.  Top-level routing is performed
        # exactly once and is never rerun independently for each block replay.
        checkpoint_blocks = bool(return_cache and activation_checkpoint)
        router_only = bool(
            return_cache
            and self.hierarchical_memory_enabled
            and self.config.memory_context.integration_mode == "terminal_landmark"
            and self.config.memory_context.memory_training == "router_only"
        )
        # In router-only post-training, layers below the memory-aware layer are
        # frozen and cannot receive gradient from any optimized parameter. Run
        # them inference-style and retain caches only from the injection layer
        # onward. This is the same recompute/cache-lifetime principle already
        # used elsewhere in the repository, applied at a coarser layer boundary.
        cache_start_layer = (
            int(self.memory_attention_layer) if router_only else 0
        )
        block_caches = [] if (return_cache and not checkpoint_blocks) else None
        block_checkpoints = [] if checkpoint_blocks else None
        active_layers = int(self.active_layers)
        # 0080AY: optional observational memory probe, disabled by default.
        # It deliberately exits before backward when the chosen layer limit
        # is reached, so this can diagnose an otherwise OOM-ing full-cache run.
        layer_memory_audit = None
        if (return_cache and not checkpoint_blocks and
                os.environ.get('MINI_LLM_MEMORY_AUDIT', '0') == '1'):
            from mini_llm.activation_memory_audit import LayerMemoryAudit
            layer_memory_audit = LayerMemoryAudit(active_layers, x)
        for i, block in enumerate(self.blocks[:active_layers]):
            retain_block = bool(return_cache and i >= cache_start_layer)
            if checkpoint_blocks and retain_block:
                block_checkpoints.append(x)
                with performance_scope(f"model.layer{i}.forward"):
                    x = block.forward(
                        x,
                        finite_trace=finite_trace,
                        layer_idx=i,
                        return_cache=False,
                        position_ids=effective_position_ids,
                        terminal_memory=(
                            terminal_memory if i == self.memory_attention_layer else None
                        ),
                        attention_allow_mask=attention_allow_mask,
                        block_context_length=block_context_length,
                        block_valid_mask=block_valid_mask,
                        historical_state=historical_state,
                        full_history_positions=full_history_positions,
                        full_history_current_valid_mask=full_history_current_valid_mask,
                        dual_resolution_metadata=dual_resolution_metadata,
                    )
            elif retain_block and not checkpoint_blocks:
                if layer_memory_audit is not None:
                    layer_memory_audit.before_layer(i)
                try:
                    with performance_scope(f"model.layer{i}.forward"):
                        x, block_cache = block.forward(
                            x,
                            finite_trace=finite_trace,
                            layer_idx=i,
                            position_ids=effective_position_ids,
                            terminal_memory=(
                                terminal_memory if i == self.memory_attention_layer else None
                            ),
                            attention_allow_mask=attention_allow_mask,
                            block_context_length=block_context_length,
                            block_valid_mask=block_valid_mask,
                            historical_state=historical_state,
                            full_history_positions=full_history_positions,
                            full_history_current_valid_mask=full_history_current_valid_mask,
                            dual_resolution_metadata=dual_resolution_metadata,
                        )
                except Exception as exc:
                    if layer_memory_audit is not None and (
                        isinstance(exc, MemoryError) or
                        ('out of memory' in str(exc).lower() and
                         type(exc).__module__.startswith('cupy'))
                    ):
                        layer_memory_audit.forward_failure(i, exc)
                        raise SystemExit(42) from exc
                    raise
                block_caches.append(block_cache)
                if layer_memory_audit is not None:
                    layer_memory_audit.after_layer(i, block_cache)
            else:
                with performance_scope(f"model.layer{i}.forward"):
                    x = block.forward(
                        x,
                        finite_trace=finite_trace,
                        layer_idx=i,
                        return_cache=False,
                        position_ids=effective_position_ids,
                        terminal_memory=(
                            terminal_memory if i == self.memory_attention_layer else None
                        ),
                        attention_allow_mask=attention_allow_mask,
                        block_context_length=block_context_length,
                        block_valid_mask=block_valid_mask,
                        historical_state=historical_state,
                        full_history_positions=full_history_positions,
                        full_history_current_valid_mask=full_history_current_valid_mask,
                        dual_resolution_metadata=dual_resolution_metadata,
                    )

        with performance_scope("model.final_norm.forward"):
            if is_low_precision_dtype(self.dtype):
                x, final_norm_cache = self.final_norm.forward_compute(x, self.dtype)
            else:
                x, final_norm_cache = self.final_norm.forward(x)
        if finite_trace is not None:
            finite_trace.append(("final_norm", xp.all(xp.isfinite(x))))

        head_input = (
            x.astype(self.dtype, copy=False)
            if is_low_precision_dtype(self.dtype) else x
        )
        if not return_cache:
            if return_target_slice:
                return head_input, target_slice
            return head_input

        cache = {
            "final_norm_cache": final_norm_cache,
            "activation_checkpoint": "block" if checkpoint_blocks else "none",
            "block_cache_start_layer": cache_start_layer,
            "active_layers": active_layers,
        }
        if checkpoint_blocks:
            cache["block_checkpoints"] = block_checkpoints
            # Replay must use exactly the same explicit source positions.
            cache["position_ids"] = effective_position_ids
            cache["terminal_memory"] = terminal_memory
            cache["historical_state"] = historical_state
            cache["full_history_positions"] = full_history_positions
            cache["full_history_current_valid_mask"] = full_history_current_valid_mask
            cache["attention_allow_mask"] = attention_allow_mask
            cache["block_context_length"] = block_context_length
            cache["block_valid_mask"] = block_valid_mask
            cache["dual_resolution_metadata"] = dual_resolution_metadata
        else:
            cache["block_caches"] = block_caches

        if self.hierarchical_memory_enabled:
            cache["memory_context_cache"] = memory_context_cache
            cache["target_slice"] = target_slice
        else:
            cache["embed_cache"] = embed_cache
            cache["external_input_embeddings"] = input_embeddings is not None
        return head_input, cache

    def forward(
        self, token_ids, finite_trace=None, return_cache=True,
        activation_checkpoint=False, position_ids=None, memory_metadata=None,
        input_embeddings=None, attention_allow_mask=None,
        block_context_length=None, block_valid_mask=None,
        dual_resolution_metadata=None,
    ):
        """Full decoder forward.

        Direct-context models preserve the historical full-logits API.  For a
        hierarchical-memory model, only target-region logits are materialized;
        conditioning rows never enter the vocabulary projection.
        """
        if return_cache:
            head_input, cache = self.forward_body(
                token_ids, finite_trace=finite_trace, return_cache=True,
                activation_checkpoint=activation_checkpoint,
                position_ids=position_ids,
                memory_metadata=memory_metadata,
                input_embeddings=input_embeddings,
                attention_allow_mask=attention_allow_mask,
                block_context_length=block_context_length,
                block_valid_mask=block_valid_mask,
                dual_resolution_metadata=dual_resolution_metadata,
            )
            target_slice = cache.get("target_slice")
        else:
            head_input, target_slice = self.forward_body(
                token_ids, finite_trace=finite_trace, return_cache=False,
                position_ids=position_ids, memory_metadata=memory_metadata,
                return_target_slice=True,
                input_embeddings=input_embeddings,
                attention_allow_mask=attention_allow_mask,
                block_context_length=block_context_length,
                block_valid_mask=block_valid_mask,
                dual_resolution_metadata=dual_resolution_metadata,
            )

        head_for_logits = head_input
        if target_slice is not None:
            start, stop = target_slice
            # Copy the small target slice so the output-projection cache cannot
            # pin the complete active final-normalized tensor via a view.
            head_for_logits = xp.array(
                head_input[:, start:stop, :], copy=True, order="C"
            )

        with performance_scope("model.output_projection.forward"):
            logits, output_proj_cache = self.output_proj.forward(head_for_logits)
        if finite_trace is not None:
            finite_trace.append(("logits", xp.all(xp.isfinite(logits))))

        if not return_cache:
            return logits
        cache["output_proj_cache"] = output_proj_cache
        if target_slice is not None:
            cache["output_proj_target_slice"] = target_slice
            cache["output_proj_full_shape"] = tuple(head_input.shape)
        return logits, cache

    def chunked_lm_head_loss_forward(
        self,
        head_input,
        targets,
        *,
        chunk_tokens=512,
        loss_mask=None,
        return_device_loss=False,
        finite_trace=None,
        target_slice=None,
    ):
        """Chunked projection+CE, optionally on target hidden rows only.

        0058's hierarchical-memory path copies only ``target_slice`` out of the
        full active hidden tensor before vocabulary projection.  Backward later
        scatters the target gradient into an all-zero active-sequence gradient.
        """
        full_head_shape = None
        selected_head = head_input
        normalized_slice = None
        if target_slice is not None:
            start, stop = map(int, target_slice)
            if not (0 <= start < stop <= int(head_input.shape[1])):
                raise ValueError("target_slice is outside the active sequence")
            normalized_slice = (start, stop)
            full_head_shape = tuple(head_input.shape)
            # A compact copy prevents a target view from retaining the complete
            # final-normalized active tensor through LM-head backward.
            selected_head = xp.array(
                head_input[:, start:stop, :], copy=True, order="C"
            )

        flat_x = selected_head.reshape(-1, selected_head.shape[-1])
        flat_targets = xp.asarray(targets, dtype=xp.int32).reshape(-1)
        n = int(flat_x.shape[0])
        chunk_tokens = max(1, min(n, int(chunk_tokens)))
        if flat_targets.shape[0] != n:
            raise ValueError("targets must match the number of LM-head token rows")

        if loss_mask is None:
            mask_f32 = None
            normalizer_count = float(n)
        else:
            mask_f32 = xp.asarray(loss_mask, dtype=xp.float32).reshape(-1)
            if mask_f32.shape[0] != n:
                raise ValueError("loss_mask must match the number of token rows")
            normalizer_count = float(xp.sum(mask_f32).item())
            if normalizer_count <= 0.0:
                raise ValueError("loss_mask must select at least one target token")

        loss_sum = xp.asarray(0.0, dtype=xp.float32)
        finite_ok = None
        for start in range(0, n, chunk_tokens):
            end = min(start + chunk_tokens, n)
            with performance_scope("model.output_projection.chunk_forward"):
                logits = flat_x[start:end] @ self.output_proj.W.data
            if finite_trace is not None:
                ok = xp.all(xp.isfinite(logits))
                finite_ok = ok if finite_ok is None else (finite_ok & ok)
            with performance_scope("model.loss.chunk_forward"):
                loss_sum += chunked_bf16_cross_entropy_loss(
                    logits,
                    flat_targets[start:end],
                    loss_mask=None if mask_f32 is None else mask_f32[start:end],
                    normalizer_count=normalizer_count,
                )
            del logits

        if finite_trace is not None:
            finite_trace.append(("logits", finite_ok))
        cache = {
            "head_input": selected_head,
            "targets": flat_targets,
            "loss_mask_f32": mask_f32,
            "normalizer_count": normalizer_count,
            "chunk_tokens": chunk_tokens,
            "target_slice": normalized_slice,
            "full_head_shape": full_head_shape,
        }
        return (loss_sum if return_device_loss else float(loss_sum.item())), cache

    def chunked_lm_head_backward(self, loss_cache, *, grad_scale=1.0,
                                 frozen_output_projection=False):
        """Recompute LM-head tiles and scatter target-only gradients.

        When `frozen_output_projection` is True, gradients flow into the
        hidden representation but never into the shared frozen vocabulary head.
        The default maintains the original autoregressive training behavior.
        """
        head_input = loss_cache.pop("head_input")
        flat_x = head_input.reshape(-1, head_input.shape[-1])
        targets = loss_cache["targets"]
        mask_f32 = loss_cache.get("loss_mask_f32")
        normalizer_count = float(loss_cache["normalizer_count"])
        chunk_tokens = int(loss_cache["chunk_tokens"])
        n, _ = map(int, flat_x.shape)

        dx = xp.empty_like(flat_x)
        W = self.output_proj.W.data
        router_only = (
            self.hierarchical_memory_enabled
            and self.config.memory_context.memory_training == "router_only"
        )
        W_grad = (None if router_only or frozen_output_projection
                  else self.output_proj.W.grad)
        for start in range(0, n, chunk_tokens):
            end = min(start + chunk_tokens, n)
            x_chunk = flat_x[start:end]
            with performance_scope("model.output_projection.chunk_recompute"):
                logits = x_chunk @ W
            with performance_scope("model.loss.chunk_backward"):
                d_logits = chunked_bf16_cross_entropy_grad_inplace(
                    logits,
                    targets[start:end],
                    loss_mask=None if mask_f32 is None else mask_f32[start:end],
                    normalizer_count=normalizer_count,
                    grad_scale=grad_scale,
                )
            with performance_scope("model.output_projection.chunk_backward"):
                if W_grad is not None:
                    W_grad += x_chunk.T @ d_logits
                dx[start:end] = d_logits @ W.T
            del logits, d_logits

        dx_selected = dx.reshape(head_input.shape)
        target_slice = loss_cache.get("target_slice")
        full_head_shape = loss_cache.get("full_head_shape")
        if target_slice is None:
            return dx_selected

        start, stop = target_slice
        dx_full = xp.zeros(full_head_shape, dtype=dx_selected.dtype)
        dx_full[:, start:stop, :] = dx_selected
        return dx_full

    def compute_loss(self, logits, targets, loss_mask=None, return_device_loss=False):
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Logits from forward pass, shape (B, T, vocab_size)
            targets: Target token IDs, shape (B, T)
            loss_mask: Optional mask, shape (B, T). Non-zero entries select
                target positions that contribute to the loss. This is used for
                assistant-only supervised fine-tuning.

        Returns:
            loss: Scalar loss value
            cache: Dictionary for backward pass
        """
        return cross_entropy_forward(
            logits, targets, loss_mask=loss_mask,
            return_device_loss=return_device_loss,
        )
    
    def backward_loss(self, loss_cache):
        """
        Backward pass through the loss layer.
        
        Args:
            loss_cache: Cache (second element) from compute_loss
            
        Returns:
            d_logits: Gradient w.r.t. logits
        """
        # loss_cache is the cache dict returned by compute_loss (second element)
        return cross_entropy_backward(loss_cache)
    
    def forward_shared_context_block(self, active_embeddings, context_kv, *,
                                     context_length, valid_block_mask=None,
                                     return_cache=True, block_position_starts=None,
                                     layer_group_plans=None, layer_retrieval_tasks=None):
        """Only K provisional rows traverse the complete shared Transformer.

        This is a bounded research path; it never alters committed-context
        hidden states or the standard autoregressive forward implementation.
        """
        if self.config.full_history_layers:
            raise NotImplementedError(
                "shared-context provisional decoding must implement the new "
                "full-history heads before it is supported"
            )
        if self.hierarchical_memory_enabled:
            if (self.config.memory_context.integration_mode != "routed_prefix"
                    or any(b.attention.terminal_memory is not None
                           for b in self.blocks[:self.active_layers])):
                raise NotImplementedError("provisional terminal memory not supported")
            if any(b.attention._retrieval_groups for b in self.blocks[:self.active_layers]) \
                    and layer_retrieval_tasks is None:
                raise ValueError("learned retrieval requires causal Q3 route tasks")
        if len(context_kv) != self.active_layers:
            raise ValueError("context KV count must equal active Transformer depth")
        x = active_embeddings
        caches = []
        if layer_group_plans is not None and len(layer_group_plans) != self.active_layers:
            raise ValueError("layer_group_plans must match the active depth")
        if layer_retrieval_tasks is not None and len(layer_retrieval_tasks) != self.active_layers:
            raise ValueError("layer_retrieval_tasks must match the active depth")
        for layer_idx, (layer, kv) in enumerate(zip(self.blocks[:self.active_layers], context_kv)):
            if return_cache:
                x, cache = layer.forward(
                    x, shared_context_kv=kv,
                    shared_context_length=context_length,
                    block_valid_mask=valid_block_mask, return_cache=True,
                    shared_context_block_starts=block_position_starts,
                    shared_context_plans=(
                        None if layer_group_plans is None else layer_group_plans[layer_idx]
                    ),
                    shared_context_retrieval_tasks=(
                        None if layer_retrieval_tasks is None else layer_retrieval_tasks[layer_idx]
                    ),
                )
                caches.append(cache)
            else:
                x = layer.forward(
                    x, shared_context_kv=kv,
                    shared_context_length=context_length,
                    block_valid_mask=valid_block_mask, return_cache=False,
                    shared_context_block_starts=block_position_starts,
                    shared_context_plans=(
                        None if layer_group_plans is None else layer_group_plans[layer_idx]
                    ),
                    shared_context_retrieval_tasks=(
                        None if layer_retrieval_tasks is None else layer_retrieval_tasks[layer_idx]
                    ),
                )
        if is_low_precision_dtype(self.dtype):
            head, norm_cache = self.final_norm.forward_compute(x, self.dtype)
        else:
            head, norm_cache = self.final_norm.forward(x)
        logits, output_cache = self.output_proj.forward(head)
        if not return_cache:
            return logits
        return logits, (caches, norm_cache, output_cache)

    def backward_shared_context_block(self, dlogits, cache):
        """Return K input gradients and per-layer causal-context K/V grads."""
        blocks, norm_cache, output_cache = cache
        # Match the original decoder backward's BF16 projection contract.
        dlogits = (
            dlogits.astype(self.dtype, copy=False)
            if is_bfloat16_dtype(self.dtype) else dlogits
        )
        dx = self.output_proj.backward(dlogits, output_cache)
        dx = self.final_norm.backward(dx, norm_cache)
        kv_grads = [None] * len(blocks)
        for index in range(len(blocks) - 1, -1, -1):
            dx, kv_grads[index] = self.blocks[index].backward(
                dx, blocks[index], return_shared_context_kv_grad=True,
            )
        return dx, kv_grads

    def backward_body(
        self, dx, cache, *, shared_context_kv_grads=None,
        backbone_parameter_grads=True,
    ):
        """Backpropagate from the final-normalized hidden representation.

        In 0057 ``activation_checkpoint="block"`` stores only each block's
        residual-stream input.  The block forward is replayed once during
        backward to recreate its exact short-lived cache, then that cache is
        consumed immediately.  Parameters are not updated until the complete
        accumulated backward finishes, so recomputation sees identical weights.
        """
        checkpoint_mode = cache.get("activation_checkpoint", "none")

        final_norm_cache = cache.pop("final_norm_cache")
        with performance_scope("model.final_norm.backward"):
            dx = self.final_norm.backward(dx, final_norm_cache)
        del final_norm_cache

        active_layers = int(cache.pop("active_layers", self.active_layers))
        if not 1 <= active_layers <= len(self.blocks):
            raise ValueError("cached active depth lies outside model depth")

        if checkpoint_mode == "block":
            block_checkpoints = cache.pop("block_checkpoints")
            position_ids = cache.pop("position_ids", None)
            terminal_memory = cache.pop("terminal_memory", None)
            historical_state = cache.pop("historical_state", None)
            full_history_positions = cache.pop("full_history_positions", None)
            full_history_current_valid_mask = cache.pop(
                "full_history_current_valid_mask", None
            )
            attention_allow_mask = cache.pop("attention_allow_mask", None)
            block_context_length = cache.pop("block_context_length", None)
            block_valid_mask = cache.pop("block_valid_mask", None)
            dual_resolution_metadata = cache.pop("dual_resolution_metadata", None)
            cache_start_layer = int(cache.pop("block_cache_start_layer", 0))
            expected = active_layers - cache_start_layer
            if len(block_checkpoints) != expected:
                raise ValueError(
                    "block activation checkpoint count does not match retained depth"
                )
            for original_idx in range(active_layers - 1, cache_start_layer - 1, -1):
                block_input = block_checkpoints.pop()
                # 0080AQ: backward checkpoint replay is not a new observed
                # training token; exclude it from expert-utilization counts.
                replay_router = getattr(self.blocks[original_idx].moe, "router", None)
                original_collector = (
                    None if replay_router is None else replay_router.usage_collector
                )
                if replay_router is not None:
                    replay_router.usage_collector = None
                try:
                    with performance_scope(f"model.layer{original_idx}.recompute"):
                        recomputed_output, block_cache = self.blocks[original_idx].forward(
                            block_input, finite_trace=None, layer_idx=original_idx,
                            return_cache=True, position_ids=position_ids,
                        terminal_memory=(
                            terminal_memory if original_idx == self.memory_attention_layer else None
                        ),
                        attention_allow_mask=attention_allow_mask,
                        block_context_length=block_context_length,
                            block_valid_mask=block_valid_mask,
                            historical_state=historical_state,
                            full_history_positions=full_history_positions,
                            full_history_current_valid_mask=full_history_current_valid_mask,
                            dual_resolution_metadata=dual_resolution_metadata,
                        )
                finally:
                    if replay_router is not None:
                        replay_router.usage_collector = original_collector
                # The backward only needs the reconstructed cache; dropping the
                # replay output before launching backward minimizes transient VRAM.
                del recomputed_output, block_input
                with performance_scope(f"model.layer{original_idx}.backward"):
                    dx = self.blocks[original_idx].backward(
                        dx, block_cache,
                        shared_context_kv_grad=(
                            None if shared_context_kv_grads is None
                            else shared_context_kv_grads[original_idx]
                        ),
                        backbone_parameter_grads=backbone_parameter_grads,
                    )
                del block_cache
                if (
                    self.hierarchical_memory_enabled
                    and self.config.memory_context.memory_training == "router_only"
                    and original_idx == self.memory_attention_layer
                ):
                    break
        elif checkpoint_mode == "none":
            block_caches = cache.pop("block_caches")
            cache_start_layer = int(cache.pop("block_cache_start_layer", 0))
            expected = active_layers - cache_start_layer
            if len(block_caches) != expected:
                raise ValueError("block cache count does not match retained depth")
            for original_idx in range(active_layers - 1, cache_start_layer - 1, -1):
                block_cache = block_caches.pop()
                with performance_scope(f"model.layer{original_idx}.backward"):
                    dx = self.blocks[original_idx].backward(
                        dx, block_cache,
                        shared_context_kv_grad=(
                            None if shared_context_kv_grads is None
                            else shared_context_kv_grads[original_idx]
                        ),
                        backbone_parameter_grads=backbone_parameter_grads,
                    )
                del block_cache
                if (
                    self.hierarchical_memory_enabled
                    and self.config.memory_context.memory_training == "router_only"
                    and original_idx == self.memory_attention_layer
                ):
                    # Earlier frozen blocks cannot influence any optimized 0058C
                    # parameter.  Stop-gradient here avoids 11 unnecessary block
                    # backward passes when the memory-aware layer is the final one.
                    break
        else:
            raise ValueError(f"unknown activation checkpoint mode: {checkpoint_mode!r}")

        memory_context_cache = cache.pop("memory_context_cache", None)
        if memory_context_cache is not None:
            # target_slice is metadata for the LM-head path only.
            cache.pop("target_slice", None)
            if backbone_parameter_grads:
                with performance_scope("memory_context.backward"):
                    self._hierarchical_context_backward(dx, memory_context_cache)
        else:
            embed_cache = cache.pop("embed_cache")
            if cache.pop("external_input_embeddings", False):
                # The caller owns the gradient: soft positions backpropagate
                # through QE; actual-token prefix positions use scatter-add.
                return dx
            if backbone_parameter_grads:
                with performance_scope("model.embedding.backward"):
                    self.embedding.backward(dx, embed_cache)
            del embed_cache
        return None

    def backward(self, d_logits, cache):
        """Full historical backward from a materialized logits gradient."""
        d_logits_compute = (
            d_logits.astype(self.dtype, copy=False)
            if is_bfloat16_dtype(self.dtype) else d_logits
        )
        output_proj_cache = cache.pop("output_proj_cache")
        with performance_scope("model.output_projection.backward"):
            if (
                self.hierarchical_memory_enabled
                and self.config.memory_context.memory_training == "router_only"
            ):
                dx = self.output_proj.backward_input(
                    d_logits_compute, output_proj_cache
                )
            else:
                dx = self.output_proj.backward(d_logits_compute, output_proj_cache)
        del output_proj_cache, d_logits_compute

        target_slice = cache.pop("output_proj_target_slice", None)
        full_shape = cache.pop("output_proj_full_shape", None)
        if target_slice is not None:
            start, stop = target_slice
            dx_target = dx
            dx = xp.zeros(full_shape, dtype=dx_target.dtype)
            dx[:, start:stop, :] = dx_target
            del dx_target
        return self.backward_body(dx, cache)

