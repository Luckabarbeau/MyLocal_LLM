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
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_proj.parameters())
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
    
    def forward_body(
        self, token_ids, finite_trace=None, return_cache=True,
        activation_checkpoint=False,
    ):
        """Run embedding/transformer/final-norm without materializing logits.

        0056 uses this training-only split so the vocabulary projection can be
        evaluated in bounded token tiles.  Inference and ordinary callers keep
        using :meth:`forward`, which preserves the established full-logits API.
        """
        # Embed tokens - unpack output and cache.
        with performance_scope("model.embedding.forward"):
            x, embed_cache = self.embedding.forward(token_ids)
        if finite_trace is not None:
            finite_trace.append(("embedding", xp.all(xp.isfinite(x))))

        # 0057 block-level activation checkpointing keeps only one compact
        # residual-stream input per Transformer block.  Attention/MoE/RMSNorm
        # internals are deliberately discarded in the original forward and are
        # reconstructed one block at a time immediately before its backward.
        # This preserves the exact block math while changing activation memory
        # from "all internal caches in all blocks" to O(L * B * T * d_model).
        checkpoint_blocks = bool(return_cache and activation_checkpoint)
        block_caches = [] if (return_cache and not checkpoint_blocks) else None
        block_checkpoints = [] if checkpoint_blocks else None
        for i, block in enumerate(self.blocks):
            if checkpoint_blocks:
                # ``block.forward`` never mutates its residual-stream input.
                # Keeping this reference therefore snapshots exactly the input
                # that must be replayed during backward without an extra copy.
                block_checkpoints.append(x)
                with performance_scope(f"model.layer{i}.forward"):
                    x = block.forward(
                        x, finite_trace=finite_trace, layer_idx=i, return_cache=False
                    )
            elif return_cache:
                with performance_scope(f"model.layer{i}.forward"):
                    x, block_cache = block.forward(
                        x, finite_trace=finite_trace, layer_idx=i
                    )
                block_caches.append(block_cache)
            else:
                with performance_scope(f"model.layer{i}.forward"):
                    x = block.forward(
                        x, finite_trace=finite_trace, layer_idx=i, return_cache=False
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
            "token_ids": token_ids,
            "embed_cache": embed_cache,
            "final_norm_cache": final_norm_cache,
            "activation_checkpoint": "block" if checkpoint_blocks else "none",
        }
        if checkpoint_blocks:
            cache["block_checkpoints"] = block_checkpoints
        else:
            cache["block_caches"] = block_caches
        return head_input, cache

    def forward(
        self, token_ids, finite_trace=None, return_cache=True,
        activation_checkpoint=False,
    ):
        """Full decoder forward preserving the historical logits API."""
        if return_cache:
            head_input, cache = self.forward_body(
                token_ids, finite_trace=finite_trace, return_cache=True,
                activation_checkpoint=activation_checkpoint,
            )
        else:
            head_input = self.forward_body(
                token_ids, finite_trace=finite_trace, return_cache=False
            )

        with performance_scope("model.output_projection.forward"):
            logits, output_proj_cache = self.output_proj.forward(head_input)
        if finite_trace is not None:
            finite_trace.append(("logits", xp.all(xp.isfinite(logits))))

        if not return_cache:
            return logits
        cache["output_proj_cache"] = output_proj_cache
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
    ):
        """0056 projection+CE forward without a full ``[B,T,V]`` tensor.

        Only the final normalized hidden states, targets and scalar normalizer
        survive forward.  Each logits tile is discarded immediately after its
        loss contribution is accumulated and will be recomputed in backward.
        """
        flat_x = head_input.reshape(-1, head_input.shape[-1])
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
            "head_input": head_input,
            "targets": flat_targets,
            "loss_mask_f32": mask_f32,
            "normalizer_count": normalizer_count,
            "chunk_tokens": chunk_tokens,
        }
        return (loss_sum if return_device_loss else float(loss_sum.item())), cache

    def chunked_lm_head_backward(self, loss_cache, *, grad_scale=1.0):
        """0056 recompute each LM-head tile and backpropagate it immediately."""
        head_input = loss_cache.pop("head_input")
        flat_x = head_input.reshape(-1, head_input.shape[-1])
        targets = loss_cache["targets"]
        mask_f32 = loss_cache.get("loss_mask_f32")
        normalizer_count = float(loss_cache["normalizer_count"])
        chunk_tokens = int(loss_cache["chunk_tokens"])
        n, d_model = map(int, flat_x.shape)

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
            # Match the established Linear.backward ordering: accumulate dW,
            # then form dX, but only for this bounded vocabulary tile.
            with performance_scope("model.output_projection.chunk_backward"):
                W_grad += x_chunk.T @ d_logits
                dx[start:end] = d_logits @ W.T
            del logits, d_logits
        return dx.reshape(head_input.shape)

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
            if len(block_checkpoints) != len(self.blocks):
                raise ValueError(
                    "block activation checkpoint count does not match model depth"
                )
            for original_idx in range(len(self.blocks) - 1, -1, -1):
                block_input = block_checkpoints.pop()
                with performance_scope(f"model.layer{original_idx}.recompute"):
                    recomputed_output, block_cache = self.blocks[original_idx].forward(
                        block_input, finite_trace=None, layer_idx=original_idx,
                        return_cache=True,
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
        self.backward_body(dx, cache)

