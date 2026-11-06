"""Configuration-driven PyTorch decoder for native MyLocal_LLM architectures.

The implementation intentionally owns a PyTorch autograd graph instead of
trying to emulate the repository's NumPy/CuPy ``xp`` API. Canonical parameter
names and matrix orientations match native checkpoints.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from .attention import (
    apply_rope,
    flex_attention_with_mask,
    global_history_mask_mod,
    local_mask_mod,
    standard_causal_attention,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The PyTorch backend requires torch. Install the optional pytorch dependency."
        ) from exc
    return torch


def resolve_torch_dtype(dtype):
    torch = _torch()
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "bf16-mixed": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return aliases[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported PyTorch model dtype: {dtype!r}") from exc


def _parameter(shape, std: float, dtype):
    torch = _torch()
    value = torch.empty(tuple(int(v) for v in shape), dtype=dtype)
    torch.nn.init.normal_(value, mean=0.0, std=float(std))
    return torch.nn.Parameter(value)


class RMSNorm(_torch().nn.Module):
    def __init__(self, width: int, eps: float, dtype):
        super().__init__()
        torch = _torch()
        self.gamma = torch.nn.Parameter(torch.ones(width, dtype=dtype))
        self.eps = float(eps)

    def forward(self, x):
        torch = _torch()
        normalized = x.float() * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.gamma.float()).to(self.gamma.dtype)


class ExpertFFN(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.W_gate = _parameter((config.d_model, config.d_ff), config.init_std, dtype)
        self.W_up = _parameter((config.d_model, config.d_ff), config.init_std, dtype)
        self.W_down = _parameter(
            (config.d_ff, config.d_model), config.residual_init_std, dtype
        )

    def forward(self, x):
        torch = _torch()
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        hidden = torch.nn.functional.silu(flat @ self.W_gate) * (flat @ self.W_up)
        return (hidden @ self.W_down).reshape(shape)


class DenseFFN(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.expert = ExpertFFN(config, dtype)

    def forward(self, x):
        return self.expert(x)


class ExpertCollection(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.count = int(config.n_experts)
        for index in range(self.count):
            self.add_module(f"expert_{index}", ExpertFFN(config, dtype))

    def __iter__(self):
        for index in range(self.count):
            yield getattr(self, f"expert_{index}")


class Router(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        torch = _torch()
        self.W_router = _parameter(
            (config.d_model, config.n_experts), config.init_std, dtype
        )
        self.b_router = torch.nn.Parameter(torch.zeros(config.n_experts, dtype=dtype))


class MoE(_torch().nn.Module):
    """Native top-k rule: full softmax, selected probabilities not renormalized."""
    def __init__(self, config, dtype):
        super().__init__()
        self.router = Router(config, dtype)
        self.experts = ExpertCollection(config, dtype)
        self.top_k = int(config.top_k)

    def forward(self, x):
        torch = _torch()
        logits = x @ self.router.W_router + self.router.b_router
        probabilities = torch.softmax(logits.float(), dim=-1)
        _, indices = torch.topk(logits.float(), self.top_k, dim=-1, largest=True, sorted=True)
        selected = torch.gather(probabilities, -1, indices).to(dtype=x.dtype)
        output = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.experts):
            expert_output = expert(x)
            route_weight = torch.sum(
                selected * (indices == expert_index).to(selected.dtype), dim=-1,
                keepdim=True,
            )
            output = output + route_weight * expert_output
        return output


class StandardAttention(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        q_width = config.n_q_heads * config.d_head
        kv_width = config.n_kv_heads * config.d_head
        self.Wq = _parameter((config.d_model, q_width), config.init_std, dtype)
        self.Wk = _parameter((config.d_model, kv_width), config.init_std, dtype)
        self.Wv = _parameter((config.d_model, kv_width), config.init_std, dtype)
        self.Wo = _parameter((q_width, config.d_model), config.residual_init_std, dtype)
        self.n_q_heads = int(config.n_q_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.d_head = int(config.d_head)
        self.rope_base = float(config.rope_base)

    def forward(self, x, *, position_ids, document_ids=None, **ignored):
        del ignored
        batch, length, _ = x.shape
        q = (x @ self.Wq).reshape(batch, length, self.n_q_heads, self.d_head)
        k = (x @ self.Wk).reshape(batch, length, self.n_kv_heads, self.d_head)
        v = (x @ self.Wv).reshape(batch, length, self.n_kv_heads, self.d_head)
        q = apply_rope(q, position_ids, self.rope_base)
        k = apply_rope(k, position_ids, self.rope_base)
        context = standard_causal_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3), document_ids=document_ids,
        ).permute(0, 2, 1, 3)
        return context.reshape(batch, length, -1) @ self.Wo


class MatrixPool(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        torch = _torch()
        dual = config.dual_resolution
        self.score = _parameter(
            (config.d_model, dual.summaries_per_block), config.init_std, dtype
        )
        self.position_bias = (
            torch.nn.Parameter(torch.zeros(
                dual.block_size, dual.summaries_per_block, dtype=dtype
            ))
            if dual.pool_position_bias else None
        )


class GlobalAttention(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        width = config.dual_resolution.global_heads * config.dual_resolution.global_head_dim
        self.Wq = _parameter((config.d_model, width), config.init_std, dtype)
        self.Wk = _parameter((config.d_model, width), config.init_std, dtype)
        self.Wv = _parameter((config.d_model, width), config.init_std, dtype)
        self.Wo = _parameter((width, config.d_model), config.residual_init_std, dtype)


class MatrixPoolDualResolutionAttention(_torch().nn.Module):
    """Exact Model B local + learned summaries + compressed-history graph."""
    def __init__(self, config, dtype, *, layer_index: int):
        super().__init__()
        dual = config.dual_resolution
        q_width = config.n_q_heads * config.d_head
        kv_width = config.n_kv_heads * config.d_head
        self.Wq = _parameter((config.d_model, q_width), config.init_std, dtype)
        self.Wk = _parameter((config.d_model, kv_width), config.init_std, dtype)
        self.Wv = _parameter((config.d_model, kv_width), config.init_std, dtype)
        self.Wo = _parameter((q_width, config.d_model), config.residual_init_std, dtype)
        self.matrix_pool = MatrixPool(config, dtype)
        self.global_attention = GlobalAttention(config, dtype) if dual.global_enabled else None
        # Expose the native canonical prefix ``attention.global.*``.
        if self.global_attention is not None:
            self.add_module("global", self.global_attention)
            del self._modules["global_attention"]
            object.__setattr__(self, "global_attention", self._modules["global"])

        self.d_model = int(config.d_model)
        self.n_q_heads = int(config.n_q_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.d_head = int(config.d_head)
        self.group_size = self.n_q_heads // self.n_kv_heads
        self.rope_base = float(config.rope_base)
        self.scale = self.d_head ** -0.5
        self.block_size = int(dual.block_size)
        self.local_window = int(dual.local_window)
        self.long_local_window = int(dual.long_local_window)
        self.long_local_heads = int(dual.long_local_heads)
        self.summaries_per_block = int(dual.summaries_per_block)
        self.global_heads = int(dual.global_heads)
        self.global_head_dim = int(dual.global_head_dim)
        self.sparse_stride = int(dual.sparse_retrieval_stride)
        self.sparse_head = int(dual.sparse_retrieval_head)
        self.sparse_phase = -1
        if self.sparse_stride and layer_index in dual.sparse_retrieval_layers:
            phases = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
            self.sparse_phase = phases[list(dual.sparse_retrieval_layers).index(layer_index)]

    @property
    def global_module(self):
        return self._modules.get("global")

    def _metadata(self, x, position_ids, metadata):
        torch = _torch()
        batch, length = x.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device, dtype=torch.long)
        position_ids = torch.as_tensor(position_ids, device=x.device, dtype=torch.long)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if metadata is None:
            block_count = length // self.block_size
            block_indices = torch.arange(
                block_count * self.block_size, device=x.device, dtype=torch.long
            ).reshape(1, block_count, self.block_size).expand(batch, -1, -1)
            return {
                "document_ids": torch.zeros((batch, length), device=x.device, dtype=torch.long),
                "position_ids": position_ids,
                "block_indices": block_indices,
                "block_valid": torch.ones((batch, block_count), device=x.device, dtype=torch.bool),
                "block_document_ids": torch.zeros((batch, block_count), device=x.device, dtype=torch.long),
                "block_end_positions": (
                    position_ids[:, block_indices[0, :, -1]] if block_count
                    else torch.empty((batch, 0), device=x.device, dtype=torch.long)
                ),
            }
        required = (
            "document_ids", "position_ids", "block_indices", "block_valid",
            "block_document_ids", "block_end_positions",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError(f"dual_resolution_metadata missing fields: {missing}")
        result = {
            name: torch.as_tensor(metadata[name], device=x.device) for name in required
        }
        result["document_ids"] = result["document_ids"].long()
        result["position_ids"] = position_ids if position_ids is not None else result["position_ids"].long()
        result["block_indices"] = result["block_indices"].long()
        result["block_valid"] = result["block_valid"].bool()
        result["block_document_ids"] = result["block_document_ids"].long()
        result["block_end_positions"] = result["block_end_positions"].long()
        if result["document_ids"].shape != (batch, length):
            raise ValueError("document_ids must have shape [B,T]")
        if result["block_indices"].ndim != 3 or result["block_indices"].shape[2] != self.block_size:
            raise ValueError("block_indices must have shape [B,M,block_size]")
        return result

    def _pool(self, hidden, metadata):
        torch = _torch()
        indices = metadata["block_indices"]
        valid = metadata["block_valid"]
        batch, block_count, _ = indices.shape
        if block_count == 0:
            return hidden.new_zeros((batch, 0, self.summaries_per_block, self.d_model))
        safe_indices = torch.where(valid[:, :, None], indices, torch.zeros_like(indices))
        rows = torch.arange(batch, device=hidden.device)[:, None, None]
        blocks = hidden[rows, safe_indices]
        scores = blocks.to(self.matrix_pool.score.dtype) @ self.matrix_pool.score
        scores = scores.float()
        if self.matrix_pool.position_bias is not None:
            scores = scores + self.matrix_pool.position_bias.float()[None, None, :, :]
        probabilities = torch.softmax(scores, dim=2)
        summaries = torch.matmul(probabilities.transpose(-2, -1), blocks.float())
        return summaries * valid[:, :, None, None]

    def forward(self, x, *, residual_input, position_ids=None,
                document_ids=None, dual_resolution_metadata=None):
        torch = _torch()
        batch, length, _ = x.shape
        metadata = self._metadata(x, position_ids, dual_resolution_metadata)
        if document_ids is not None:
            metadata["document_ids"] = torch.as_tensor(
                document_ids, device=x.device, dtype=torch.long
            )
        positions = metadata["position_ids"]
        q = (x @ self.Wq).reshape(batch, length, self.n_q_heads, self.d_head)
        k = (x @ self.Wk).reshape(batch, length, self.n_kv_heads, self.d_head)
        v = (x @ self.Wv).reshape(batch, length, self.n_kv_heads, self.d_head)
        q = apply_rope(q, positions, self.rope_base)
        k = apply_rope(k, positions, self.rope_base)
        mask = local_mask_mod(
            metadata["document_ids"], local_window=self.local_window,
            long_local_window=self.long_local_window,
            long_local_heads=self.long_local_heads,
            sparse_head=self.sparse_head if self.sparse_phase >= 0 else None,
            sparse_stride=self.sparse_stride, sparse_phase=self.sparse_phase,
        )
        local_context = flex_attention_with_mask(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3), mask, scale=self.scale, enable_gqa=True,
        ).permute(0, 2, 1, 3)
        local_delta = local_context.reshape(batch, length, self.d_model) @ self.Wo
        local_state = residual_input.float() + local_delta.float()

        global_module = self.global_module
        if global_module is None or metadata["block_indices"].shape[1] == 0:
            return local_delta
        summaries = self._pool(local_state, metadata)
        summary_flat = summaries.reshape(batch, -1, self.d_model).to(x.dtype)
        summary_valid = metadata["block_valid"].repeat_interleave(
            self.summaries_per_block, dim=1
        )
        summary_documents = metadata["block_document_ids"].repeat_interleave(
            self.summaries_per_block, dim=1
        )
        summary_ends = metadata["block_end_positions"].repeat_interleave(
            self.summaries_per_block, dim=1
        )
        summary_rope = torch.where(summary_valid, summary_ends, torch.zeros_like(summary_ends))
        qg = (local_state.to(x.dtype) @ global_module.Wq).reshape(
            batch, length, self.global_heads, self.global_head_dim
        )
        kg = (summary_flat @ global_module.Wk).reshape(
            batch, -1, self.global_heads, self.global_head_dim
        )
        vg = (summary_flat @ global_module.Wv).reshape(
            batch, -1, self.global_heads, self.global_head_dim
        )
        qg = apply_rope(qg, positions, self.rope_base)
        kg = apply_rope(kg, summary_rope, self.rope_base)
        global_mask = global_history_mask_mod(
            metadata["document_ids"], positions, summary_documents,
            summary_ends, summary_valid,
        )
        global_context = flex_attention_with_mask(
            qg.permute(0, 2, 1, 3), kg.permute(0, 2, 1, 3),
            vg.permute(0, 2, 1, 3), global_mask, scale=self.scale,
        ).permute(0, 2, 1, 3)
        global_delta = global_context.reshape(batch, length, -1) @ global_module.Wo
        return local_delta + global_delta.to(local_delta.dtype)


class TransformerBlock(_torch().nn.Module):
    def __init__(self, config, dtype, layer_index: int):
        super().__init__()
        torch = _torch()
        self.norm1 = RMSNorm(config.d_model, config.rms_eps, dtype)
        if config.dual_resolution.enabled:
            if config.dual_resolution.compression_mode != "matrix_pool":
                raise NotImplementedError(
                    "PyTorch transformer-compression dual resolution is not implemented"
                )
            self.attention = MatrixPoolDualResolutionAttention(
                config, dtype, layer_index=layer_index
            )
        else:
            if config.attention_layers is not None:
                raise NotImplementedError(
                    "PyTorch mixed retrieval/dilated attention presets are not yet implemented"
                )
            self.attention = StandardAttention(config, dtype)
        self.norm2 = RMSNorm(config.d_model, config.rms_eps, dtype)
        if config.ffn_mode == "dense":
            self.dense_ffn = DenseFFN(config, dtype)
        else:
            self.moe = MoE(config, dtype)
        self.alpha_attn = None
        self.alpha_mlp = None
        if config.progressive_depth:
            self.alpha_attn = torch.nn.Parameter(torch.tensor(
                1.0 if layer_index < config.progressive_initial_layers else 0.0,
                dtype=torch.float32 if dtype != torch.float32 else dtype,
            ))
            self.alpha_mlp = torch.nn.Parameter(torch.tensor(
                1.0 if layer_index < config.progressive_initial_layers else 0.0,
                dtype=torch.float32 if dtype != torch.float32 else dtype,
            ))

    def forward(self, x, *, position_ids, document_ids=None,
                dual_resolution_metadata=None):
        normalized = self.norm1(x)
        kwargs = dict(
            position_ids=position_ids, document_ids=document_ids,
            dual_resolution_metadata=dual_resolution_metadata,
        )
        if isinstance(self.attention, MatrixPoolDualResolutionAttention):
            kwargs["residual_input"] = x
        attention = self.attention(normalized, **kwargs)
        if self.alpha_attn is not None:
            attention = attention * self.alpha_attn
        residual = x.float() + attention.float() if x.dtype != _torch().float32 else x + attention
        normalized_ffn = self.norm2(residual)
        ffn = self.dense_ffn(normalized_ffn) if hasattr(self, "dense_ffn") else self.moe(normalized_ffn)
        if self.alpha_mlp is not None:
            ffn = ffn * self.alpha_mlp
        return residual + ffn.float() if residual.dtype != ffn.dtype else residual + ffn


class Embedding(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.W = _parameter((config.vocab_size, config.d_model), config.init_std, dtype)


class OutputProjection(_torch().nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.W = _parameter((config.d_model, config.vocab_size), config.init_std, dtype)


class TorchDecoderLanguageModel(_torch().nn.Module):
    """PyTorch autograd implementation preserving native architecture identity."""
    def __init__(self, config, *, dtype="bfloat16", device=None,
                 activation_checkpointing: bool = True):
        super().__init__()
        torch = _torch()
        if config.memory_context.enabled:
            raise NotImplementedError(
                "PyTorch hierarchical-memory/direct-full-history presets are not yet implemented"
            )
        if config.full_history_layers:
            raise NotImplementedError("PyTorch full-history heads are not yet implemented")
        self.config = config
        self.compute_dtype = resolve_torch_dtype(dtype)
        self.embedding = Embedding(config, self.compute_dtype)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(config, self.compute_dtype, index)
            for index in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, config.rms_eps, self.compute_dtype)
        self.output_proj = None if config.tie_embeddings else OutputProjection(
            config, self.compute_dtype
        )
        self.active_layers = (
            int(config.progressive_initial_layers)
            if config.progressive_depth else int(config.n_layers)
        )
        self.activation_checkpointing = bool(activation_checkpointing)
        if device is not None:
            self.to(device=device)

    @property
    def dtype(self):
        return self.compute_dtype

    def set_active_layers(self, count: int):
        count = int(count)
        if not 1 <= count <= len(self.blocks):
            raise ValueError("active layer count must lie in [1, max_layers]")
        if not self.config.progressive_depth and count != len(self.blocks):
            raise ValueError("partial depth requires progressive_depth")
        self.active_layers = count

    def canonical_named_parameters(self):
        # Module structure deliberately matches native names. Tied LM-head weights
        # are not re-enumerated because the native checkpoint stores embedding.W once.
        yield from self.named_parameters(remove_duplicate=True)

    def decay_policy(self):
        policy = {}
        for name, _ in self.canonical_named_parameters():
            policy[name] = not (
                name == "embedding.W"
                or name.endswith(".gamma")
                or name.endswith("matrix_pool.position_bias")
                or name.endswith("alpha_attn")
                or name.endswith("alpha_mlp")
            )
        return policy

    def forward_body(self, token_ids, *, position_ids=None, document_ids=None,
                     dual_resolution_metadata=None):
        torch = _torch()
        token_ids = torch.as_tensor(token_ids, device=self.embedding.W.device, dtype=torch.long)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B,T]")
        batch, length = token_ids.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=token_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        else:
            position_ids = torch.as_tensor(position_ids, device=token_ids.device, dtype=torch.long)
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if document_ids is None and dual_resolution_metadata is not None:
            document_ids = dual_resolution_metadata.get("document_ids")
        if document_ids is not None:
            document_ids = torch.as_tensor(document_ids, device=token_ids.device, dtype=torch.long)
        x = self.embedding.W[token_ids]
        if self.compute_dtype != torch.float32:
            x = x.float()
        for block in self.blocks[:self.active_layers]:
            def run(hidden, block=block):
                return block(
                    hidden, position_ids=position_ids, document_ids=document_ids,
                    dual_resolution_metadata=dual_resolution_metadata,
                )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False)
            else:
                x = run(x)
        return self.final_norm(x)

    def forward(self, token_ids, *, position_ids=None, document_ids=None,
                dual_resolution_metadata=None, return_hidden=False):
        hidden = self.forward_body(
            token_ids, position_ids=position_ids, document_ids=document_ids,
            dual_resolution_metadata=dual_resolution_metadata,
        )
        if return_hidden:
            return hidden
        weight = self.embedding.W if self.config.tie_embeddings else self.output_proj.W.T
        return hidden @ weight.T if self.config.tie_embeddings else hidden @ self.output_proj.W

    def forward_loss(self, token_ids, targets, *, loss_mask=None,
                     position_ids=None, document_ids=None,
                     dual_resolution_metadata=None):
        torch = _torch()
        logits = self(
            token_ids, position_ids=position_ids, document_ids=document_ids,
            dual_resolution_metadata=dual_resolution_metadata,
        )
        targets = torch.as_tensor(targets, device=logits.device, dtype=torch.long)
        losses = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        if loss_mask is None:
            return losses.mean()
        weights = torch.as_tensor(loss_mask, device=losses.device, dtype=torch.float32)
        denominator = weights.sum()
        if not bool(denominator > 0):
            raise ValueError("loss_mask must supervise at least one token")
        return torch.sum(losses * weights) / denominator

    def compile(self, *, mode="default", dynamic=True):
        torch = _torch()
        return torch.compile(self, mode=mode, dynamic=bool(dynamic))


# Public short name used by adapters.
DecoderLanguageModel = TorchDecoderLanguageModel

__all__ = [
    "DecoderLanguageModel",
    "MatrixPoolDualResolutionAttention",
    "TorchDecoderLanguageModel",
    "resolve_torch_dtype",
]
