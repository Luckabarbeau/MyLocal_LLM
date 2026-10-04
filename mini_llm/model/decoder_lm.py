"""Decoder-only language model built from Transformer blocks."""

from mini_llm.backend import xp, resolve_dtype, is_low_precision_dtype, is_bfloat16_dtype
from mini_llm.config import ModelConfig
from mini_llm.blocks.transformer_block import TransformerBlock
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.linear import Linear
from mini_llm.ops.loss import (
    cross_entropy_forward,
    cross_entropy_backward,
    chunked_bf16_cross_entropy_loss,
    chunked_bf16_cross_entropy_grad_inplace,
)
from mini_llm.ops.rmsnorm import RMSNorm
from mini_llm.ops.context_blocks import block_token_indices
from mini_llm.ops.hierarchical_memory import (
    ActiveContext,
    HierarchicalMemoryRouter,
    router_weight_gate,
    router_weight_gate_backward,
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
                attention_config=attention_config
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
        if config.memory_context.enabled:
            self.memory_router = HierarchicalMemoryRouter(
                d_model=config.d_model,
                config=config.memory_context,
                rng=rng,
                input_std=config.init_std,
                name="memory_router",
                dtype=self.dtype,
            )
    
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
        return params
    
    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.zero_grad()

    def refresh_compute_buffers(self):
        """Refresh derived compute buffers after trainable weights change."""
        for block in self.blocks:
            if hasattr(block, "refresh_compute_buffers"):
                block.refresh_compute_buffers()
    
    @property
    def hierarchical_memory_enabled(self):
        return self.memory_router is not None

    def _hierarchical_context_forward(self, token_ids):
        """Search long history once and build the bounded active sequence.

        ``memory_length`` is the complete pre-target history horizon.  The
        router sees only distant history + recent *pre-target* context; target
        tokens are structurally absent from the routing API, preventing future
        leakage in memory selection.
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
        distant_end = int(cfg.distant_memory_length)
        recent_end = int(cfg.memory_length)
        target_end = recent_end + int(cfg.target_length)

        history_ids = source_ids[:, :distant_end]
        recent_ids = source_ids[:, distant_end:recent_end]
        target_ids = source_ids[:, recent_end:target_end]

        # The long history representation is intentionally temporary.  Only
        # cheap pooled/router state survives into the deep Transformer.
        with performance_scope("memory_context.history_embedding.forward"):
            history_x = self.embedding.W.data[history_ids]
        with performance_scope("memory_context.recent_embedding.forward"):
            recent_x = self.embedding.W.data[recent_ids]
        with performance_scope("memory_context.router.forward"):
            route_weights, selected_blocks, router_cache = self.memory_router.forward(
                history_x, recent_x
            )

        with performance_scope("memory_context.active_build.forward"):
            selected_positions = block_token_indices(
                selected_blocks, int(cfg.block_size)
            )
            flat_positions = selected_positions.reshape(batch, -1)
            batch_ids = xp.arange(batch, dtype=xp.int64)[:, None]
            retrieved_ids = source_ids[batch_ids, flat_positions]

            # Reopen exact full-resolution embeddings only after block selection.
            # A neutral-at-uniform differentiable gate carries the LM signal to
            # selected router scores without replacing token content by summaries.
            retrieved_exact = self.embedding.W.data[retrieved_ids].reshape(
                batch, int(cfg.top_k_blocks), int(cfg.block_size), self.config.d_model
            )
            gate = router_weight_gate(
                route_weights, int(cfg.top_k_blocks), cfg.router_weight_scale
            )
            retrieved_x = (
                retrieved_exact * gate[:, :, None, None]
            ).reshape(batch, int(cfg.retrieved_length), self.config.d_model)

            target_x = self.embedding.W.data[target_ids]
            active_x = xp.concatenate((retrieved_x, recent_x, target_x), axis=1)
            active_ids = xp.concatenate((retrieved_ids, recent_ids, target_ids), axis=1)

            recent_positions = xp.broadcast_to(
                xp.arange(distant_end, recent_end, dtype=xp.int64)[None, :],
                (batch, int(cfg.recent_length)),
            )
            target_positions = xp.broadcast_to(
                xp.arange(recent_end, target_end, dtype=xp.int64)[None, :],
                (batch, int(cfg.target_length)),
            )
            active_positions = xp.concatenate(
                (flat_positions, recent_positions, target_positions), axis=1
            )

        # For the default mean history pool, no full history embedding tensor is
        # retained by router_cache.  Dropping this local now lets the temporary
        # 64k embedding allocation die before Transformer caches accumulate.
        del history_x, retrieved_exact, target_x

        target_start = int(cfg.retrieved_length) + int(cfg.recent_length)
        target_stop = target_start + int(cfg.target_length)
        active = ActiveContext(
            embeddings=active_x,
            token_ids=active_ids,
            position_ids=active_positions,
            source_indices=active_positions,
            target_start=target_start,
            target_end=target_stop,
            selected_blocks=selected_blocks,
            route_weights=route_weights,
        )
        cache = {
            # Token IDs/source indices are cheap to retain and are sufficient to
            # replay embedding lookup/scatter during handwritten backward.
            "source_token_ids": source_ids,
            "selected_positions": flat_positions,
            "route_weights": route_weights,
            "selected_blocks": selected_blocks,
            "router_cache": router_cache,
            "target_slice": (target_start, target_stop),
        }
        return active, cache

    def _hierarchical_context_backward(self, dactive, cache):
        """Scatter active-token and router gradients into the shared embedding."""
        cfg = self.config.memory_context
        source_ids = cache.pop("source_token_ids")
        selected_positions = cache.pop("selected_positions")
        route_weights = cache.pop("route_weights")
        router_cache = cache.pop("router_cache")

        batch = int(source_ids.shape[0])
        retrieved_length = int(cfg.retrieved_length)
        recent_length = int(cfg.recent_length)
        target_length = int(cfg.target_length)
        distant_end = int(cfg.distant_memory_length)
        recent_end = int(cfg.memory_length)

        d_retrieved = dactive[:, :retrieved_length, :]
        d_recent_active = dactive[
            :, retrieved_length:retrieved_length + recent_length, :
        ]
        d_target = dactive[:, -target_length:, :]

        batch_ids = xp.arange(batch, dtype=xp.int64)[:, None]
        retrieved_ids = source_ids[batch_ids, selected_positions]
        retrieved_exact = self.embedding.W.data[retrieved_ids].reshape(
            batch, int(cfg.top_k_blocks), int(cfg.block_size), self.config.d_model
        )
        d_retrieved_blocks = d_retrieved.reshape(retrieved_exact.shape)
        gate = router_weight_gate(
            route_weights, int(cfg.top_k_blocks), cfg.router_weight_scale
        )

        # y = gate(w) * embedding.  The direct value path updates selected token
        # embeddings; the scalar gate path trains the selected router logits.
        exact_work = retrieved_exact.astype(d_retrieved_blocks.dtype, copy=False)
        dgate = xp.sum(d_retrieved_blocks * exact_work, axis=(2, 3))
        dweights_sorted = router_weight_gate_backward(
            dgate, int(cfg.top_k_blocks), cfg.router_weight_scale
        )
        d_retrieved_exact = d_retrieved_blocks * gate[:, :, None, None].astype(
            d_retrieved_blocks.dtype, copy=False
        )

        with performance_scope("memory_context.router.backward"):
            d_history_router, d_recent_router = self.memory_router.backward(
                dweights_sorted, router_cache
            )

        history_ids = source_ids[:, :distant_end]
        recent_ids = source_ids[:, distant_end:recent_end]
        target_ids = source_ids[:, recent_end:recent_end + target_length]

        with performance_scope("memory_context.embedding.backward"):
            self.embedding.backward(
                d_retrieved_exact.reshape(batch, retrieved_length, self.config.d_model),
                {"token_ids": retrieved_ids},
            )
            self.embedding.backward(
                d_history_router, {"token_ids": history_ids}
            )
            self.embedding.backward(
                d_recent_active + d_recent_router, {"token_ids": recent_ids}
            )
            self.embedding.backward(d_target, {"token_ids": target_ids})

        del retrieved_exact, d_retrieved_exact, d_history_router, d_recent_router

    def memory_routing_diagnostics(self, cache):
        """Return optional compact diagnostics before a hierarchical cache is consumed."""
        memory_cache = cache.get("memory_context_cache")
        if memory_cache is None:
            return None
        selected = memory_cache["selected_blocks"]
        weights = memory_cache["route_weights"].astype("float32", copy=False)
        probs = xp.maximum(weights, xp.asarray(1e-12, dtype=weights.dtype))
        entropy = -xp.sum(probs * xp.log(probs), axis=-1)
        cfg = self.config.memory_context
        block_centers = (selected.astype("float32") + 0.5) * int(cfg.block_size)
        distance = float(cfg.memory_length) - block_centers
        histogram = xp.bincount(
            selected.reshape(-1), minlength=int(cfg.searchable_blocks)
        )
        sorted_selected = xp.sort(selected, axis=-1)
        duplicate_count = xp.sum(xp.diff(sorted_selected, axis=-1) == 0)
        recent_cut = max(0, int(cfg.searchable_blocks) - max(1, int(cfg.searchable_blocks) // 4))
        recent_fraction = xp.mean((selected >= recent_cut).astype("float32"))
        scores = memory_cache["router_cache"].get("scores")
        result = {
            "selected_blocks": selected,
            "weights": weights,
            "selected_block_histogram": histogram,
            "entropy_mean": xp.mean(entropy),
            "source_distance_mean": xp.mean(distance),
            "recent_quartile_fraction": recent_fraction,
            "duplicate_count": duplicate_count,
            "unique_blocks": xp.unique(selected).size,
        }
        if scores is not None:
            score_work = scores.astype("float32", copy=False)
            result.update({
                "score_mean": xp.mean(score_work),
                "score_std": xp.std(score_work),
                "score_min": xp.min(score_work),
                "score_max": xp.max(score_work),
            })
        return result

    def forward_body(
        self, token_ids, finite_trace=None, return_cache=True,
        activation_checkpoint=False, position_ids=None,
    ):
        """Run embedding/Transformer/final-norm without materializing logits.

        0058 optionally performs one top-level long-memory search before this
        deep trunk.  In that mode the Transformer receives only the bounded
        active context, while ``position_ids`` preserve original source-time
        distances for RoPE.
        """
        memory_context_cache = None
        target_slice = None
        if self.hierarchical_memory_enabled:
            if position_ids is not None:
                raise ValueError(
                    "explicit position_ids are constructed internally for "
                    "hierarchical-memory models"
                )
            with performance_scope("memory_context.forward"):
                active, memory_context_cache = self._hierarchical_context_forward(
                    token_ids
                )
            x = active.embeddings
            effective_position_ids = active.position_ids
            target_slice = (active.target_start, active.target_end)
            embed_cache = None
            if finite_trace is not None:
                finite_trace.append(
                    ("memory_active_embedding", xp.all(xp.isfinite(x)))
                )
        else:
            with performance_scope("model.embedding.forward"):
                x, embed_cache = self.embedding.forward(token_ids)
            effective_position_ids = position_ids
            if finite_trace is not None:
                finite_trace.append(("embedding", xp.all(xp.isfinite(x))))

        # 0057 block checkpointing now retains only bounded active-sequence
        # residual inputs in hierarchical mode.  Top-level routing is performed
        # exactly once and is never rerun independently for each block replay.
        checkpoint_blocks = bool(return_cache and activation_checkpoint)
        block_caches = [] if (return_cache and not checkpoint_blocks) else None
        block_checkpoints = [] if checkpoint_blocks else None
        for i, block in enumerate(self.blocks):
            if checkpoint_blocks:
                block_checkpoints.append(x)
                with performance_scope(f"model.layer{i}.forward"):
                    x = block.forward(
                        x,
                        finite_trace=finite_trace,
                        layer_idx=i,
                        return_cache=False,
                        position_ids=effective_position_ids,
                    )
            elif return_cache:
                with performance_scope(f"model.layer{i}.forward"):
                    x, block_cache = block.forward(
                        x,
                        finite_trace=finite_trace,
                        layer_idx=i,
                        position_ids=effective_position_ids,
                    )
                block_caches.append(block_cache)
            else:
                with performance_scope(f"model.layer{i}.forward"):
                    x = block.forward(
                        x,
                        finite_trace=finite_trace,
                        layer_idx=i,
                        return_cache=False,
                        position_ids=effective_position_ids,
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
            return head_input

        cache = {
            "final_norm_cache": final_norm_cache,
            "activation_checkpoint": "block" if checkpoint_blocks else "none",
        }
        if checkpoint_blocks:
            cache["block_checkpoints"] = block_checkpoints
            # Replay must use exactly the same explicit source positions.
            cache["position_ids"] = effective_position_ids
        else:
            cache["block_caches"] = block_caches

        if self.hierarchical_memory_enabled:
            cache["memory_context_cache"] = memory_context_cache
            cache["target_slice"] = target_slice
        else:
            cache["embed_cache"] = embed_cache
        return head_input, cache

    def forward(
        self, token_ids, finite_trace=None, return_cache=True,
        activation_checkpoint=False, position_ids=None,
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
            )
            target_slice = cache.get("target_slice")
        else:
            head_input = self.forward_body(
                token_ids, finite_trace=finite_trace, return_cache=False,
                position_ids=position_ids,
            )
            target_slice = None
            if self.hierarchical_memory_enabled:
                cfg = self.config.memory_context
                start = int(cfg.retrieved_length) + int(cfg.recent_length)
                target_slice = (start, start + int(cfg.target_length))

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

    def chunked_lm_head_backward(self, loss_cache, *, grad_scale=1.0):
        """Recompute each LM-head tile and scatter target-only gradients."""
        head_input = loss_cache.pop("head_input")
        flat_x = head_input.reshape(-1, head_input.shape[-1])
        targets = loss_cache["targets"]
        mask_f32 = loss_cache.get("loss_mask_f32")
        normalizer_count = float(loss_cache["normalizer_count"])
        chunk_tokens = int(loss_cache["chunk_tokens"])
        n, _ = map(int, flat_x.shape)

        dx = xp.empty_like(flat_x)
        W = self.output_proj.W.data
        W_grad = self.output_proj.W.grad
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
    
    def backward_body(self, dx, cache):
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

        if checkpoint_mode == "block":
            block_checkpoints = cache.pop("block_checkpoints")
            position_ids = cache.pop("position_ids", None)
            if len(block_checkpoints) != len(self.blocks):
                raise ValueError(
                    "block activation checkpoint count does not match model depth"
                )
            for original_idx in range(len(self.blocks) - 1, -1, -1):
                block_input = block_checkpoints.pop()
                with performance_scope(f"model.layer{original_idx}.recompute"):
                    recomputed_output, block_cache = self.blocks[original_idx].forward(
                        block_input, finite_trace=None, layer_idx=original_idx,
                        return_cache=True, position_ids=position_ids,
                    )
                # The backward only needs the reconstructed cache; dropping the
                # replay output before launching backward minimizes transient VRAM.
                del recomputed_output, block_input
                with performance_scope(f"model.layer{original_idx}.backward"):
                    dx = self.blocks[original_idx].backward(dx, block_cache)
                del block_cache
        elif checkpoint_mode == "none":
            block_caches = cache.pop("block_caches")
            for original_idx in range(len(self.blocks) - 1, -1, -1):
                block_cache = block_caches.pop()
                with performance_scope(f"model.layer{original_idx}.backward"):
                    dx = self.blocks[original_idx].backward(dx, block_cache)
                del block_cache
        else:
            raise ValueError(f"unknown activation checkpoint mode: {checkpoint_mode!r}")

        memory_context_cache = cache.pop("memory_context_cache", None)
        if memory_context_cache is not None:
            # target_slice is metadata for the LM-head path only.
            cache.pop("target_slice", None)
            with performance_scope("memory_context.backward"):
                self._hierarchical_context_backward(dx, memory_context_cache)
        else:
            embed_cache = cache.pop("embed_cache")
            with performance_scope("model.embedding.backward"):
                self.embedding.backward(dx, embed_cache)
            del embed_cache

    def backward(self, d_logits, cache):
        """Full historical backward from a materialized logits gradient."""
        d_logits_compute = (
            d_logits.astype(self.dtype, copy=False)
            if is_bfloat16_dtype(self.dtype) else d_logits
        )
        output_proj_cache = cache.pop("output_proj_cache")
        with performance_scope("model.output_projection.backward"):
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
        self.backward_body(dx, cache)

