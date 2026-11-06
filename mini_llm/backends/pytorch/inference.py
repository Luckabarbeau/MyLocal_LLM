"""PyTorch-native autoregressive inference and incremental KV caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attention import apply_rope
from .model import MatrixPoolDualResolutionAttention, StandardAttention


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch inference requires torch") from exc
    return torch


@dataclass
class LayerCache:
    k: Any
    v: Any
    summary_k: Any | None = None
    summary_v: Any | None = None
    summary_count: int = 0
    incomplete_hidden: Any | None = None
    incomplete_count: int = 0


@dataclass
class GenerationState:
    layers: list[LayerCache]
    max_length: int
    position: int = 0
    execution_report: Any = None


class TorchGenerationEngine:
    """Incremental decoder for standard and matrix-pooled native decoders.

    Model B retains full projected K/V because its stride-16 branch may address
    the entire horizon. Local heads select bounded suffix views, while completed
    matrix-pooled summaries are stored in a separate compact cache.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = model.embedding.W.device
        self.dtype = model.compute_dtype

    def new_state(self, *, batch_size: int, max_length: int):
        torch = _torch()
        batch_size, max_length = int(batch_size), int(max_length)
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        layers = []
        for block in self.model.blocks[:self.model.active_layers]:
            attention = block.attention
            k = torch.empty(
                batch_size, max_length, attention.n_kv_heads, attention.d_head,
                device=self.device, dtype=self.dtype,
            )
            v = torch.empty_like(k)
            if isinstance(attention, MatrixPoolDualResolutionAttention):
                max_blocks = max_length // attention.block_size
                max_summaries = max_blocks * attention.summaries_per_block
                summary_k = torch.empty(
                    batch_size, max_summaries, attention.global_heads,
                    attention.global_head_dim, device=self.device, dtype=self.dtype,
                )
                summary_v = torch.empty_like(summary_k)
                incomplete = torch.empty(
                    batch_size, attention.block_size, attention.d_model,
                    device=self.device, dtype=torch.float32,
                )
                layers.append(LayerCache(k, v, summary_k, summary_v, 0, incomplete, 0))
            else:
                layers.append(LayerCache(k, v))
        from mini_llm.model_contracts import ExecutionReport
        report = ExecutionReport(
            backend="pytorch", device=str(self.device), dtype=str(self.dtype),
            acceleration_verified=False,
            active_optimizations=("incremental-kv-cache", "native-pytorch-ops"),
            notes=("Performance has not been established by an end-to-end benchmark",),
        )
        return GenerationState(layers, max_length, execution_report=report)

    @staticmethod
    def _single_query_attention(q, k, v, scale):
        torch = _torch()
        if k.shape[1] == 0:
            return torch.zeros_like(q)
        # [B,1,D] x [B,S,D], FP32 softmax/reduction for parity and stability.
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v.float()).to(q.dtype)

    def _project_current(self, attention, normalized, position):
        batch = normalized.shape[0]
        q = (normalized[:, 0] @ attention.Wq).reshape(
            batch, 1, attention.n_q_heads, attention.d_head
        )
        k = (normalized[:, 0] @ attention.Wk).reshape(
            batch, 1, attention.n_kv_heads, attention.d_head
        )
        v = (normalized[:, 0] @ attention.Wv).reshape(
            batch, 1, attention.n_kv_heads, attention.d_head
        )
        torch = _torch()
        positions = torch.full(
            (batch, 1), int(position), device=self.device, dtype=torch.long
        )
        return (
            apply_rope(q, positions, attention.rope_base),
            apply_rope(k, positions, attention.rope_base),
            v,
        )

    def _standard_attention(self, attention, normalized, cache, position):
        torch = _torch()
        q, k, v = self._project_current(attention, normalized, position)
        cache.k[:, position:position + 1].copy_(k)
        cache.v[:, position:position + 1].copy_(v)
        contexts = []
        for head in range(attention.n_q_heads):
            kv_head = head // (attention.n_q_heads // attention.n_kv_heads)
            contexts.append(self._single_query_attention(
                q[:, :, head], cache.k[:, :position + 1, kv_head],
                cache.v[:, :position + 1, kv_head], attention.d_head ** -0.5,
            ))
        context = torch.stack(contexts, dim=2)
        return context.reshape(normalized.shape[0], 1, -1) @ attention.Wo

    def _matrix_attention(self, attention, normalized, residual, cache, position):
        torch = _torch()
        q, k, v = self._project_current(attention, normalized, position)
        cache.k[:, position:position + 1].copy_(k)
        cache.v[:, position:position + 1].copy_(v)
        contexts = []
        for head in range(attention.n_q_heads):
            kv_head = head // attention.group_size
            if head == attention.sparse_head and attention.sparse_phase >= 0:
                indices = (
                    torch.arange(
                        attention.sparse_phase, position + 1, attention.sparse_stride,
                        device=self.device,
                    )
                    if position >= attention.sparse_phase
                    else torch.empty(0, device=self.device, dtype=torch.long)
                )
            else:
                window = (
                    attention.long_local_window
                    if head < attention.long_local_heads else attention.local_window
                )
                start = max(0, position - window + 1)
                indices = torch.arange(start, position + 1, device=self.device)
            contexts.append(self._single_query_attention(
                q[:, :, head], cache.k[:, indices, kv_head],
                cache.v[:, indices, kv_head], attention.scale,
            ))
        local_context = torch.stack(contexts, dim=2)
        local_delta = local_context.reshape(normalized.shape[0], 1, -1) @ attention.Wo
        local_state = residual.float() + local_delta.float()
        global_delta = torch.zeros_like(local_delta)
        global_module = attention.global_module
        if global_module is not None and cache.summary_count:
            qg = (local_state.to(self.dtype)[:, 0] @ global_module.Wq).reshape(
                normalized.shape[0], 1, attention.global_heads,
                attention.global_head_dim,
            )
            positions = torch.full(
                (normalized.shape[0], 1), position, device=self.device,
                dtype=torch.long,
            )
            qg = apply_rope(qg, positions, attention.rope_base)
            global_heads = []
            for head in range(attention.global_heads):
                global_heads.append(self._single_query_attention(
                    qg[:, :, head], cache.summary_k[:, :cache.summary_count, head],
                    cache.summary_v[:, :cache.summary_count, head], attention.scale,
                ))
            global_context = torch.stack(global_heads, dim=2)
            global_delta = global_context.reshape(normalized.shape[0], 1, -1) @ global_module.Wo

        if global_module is not None:
            cache.incomplete_hidden[:, cache.incomplete_count].copy_(local_state[:, 0])
            cache.incomplete_count += 1
            if cache.incomplete_count == attention.block_size:
                block = cache.incomplete_hidden[:, :attention.block_size]
                scores = block.to(self.dtype) @ attention.matrix_pool.score
                scores = scores.float()
                if attention.matrix_pool.position_bias is not None:
                    scores = scores + attention.matrix_pool.position_bias.float()[None]
                probabilities = torch.softmax(scores, dim=1)
                summaries = torch.matmul(probabilities.transpose(1, 2), block)
                summary_rows = summaries.to(self.dtype)
                keys = (summary_rows @ global_module.Wk).reshape(
                    normalized.shape[0], attention.summaries_per_block,
                    attention.global_heads, attention.global_head_dim,
                )
                values = (summary_rows @ global_module.Wv).reshape_as(keys)
                summary_positions = torch.full(
                    (normalized.shape[0], attention.summaries_per_block),
                    position, device=self.device, dtype=torch.long,
                )
                keys = apply_rope(keys, summary_positions, attention.rope_base)
                start = cache.summary_count
                stop = start + attention.summaries_per_block
                if stop > cache.summary_k.shape[1]:
                    raise ValueError("compressed-history KV capacity exceeded")
                cache.summary_k[:, start:stop].copy_(keys)
                cache.summary_v[:, start:stop].copy_(values)
                cache.summary_count = stop
                cache.incomplete_count = 0
        return local_delta + global_delta.to(local_delta.dtype)

    def _decode(self, token_ids, state: GenerationState):
        torch = _torch()
        if state.position >= state.max_length:
            raise ValueError("generation state capacity exceeded")
        ids = torch.as_tensor(token_ids, device=self.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[:, None]
        if ids.ndim != 2 or ids.shape[1] != 1:
            raise ValueError("decode_one expects [B,1] token IDs")
        position = int(state.position)
        residual = self.model.embedding.W[ids]
        if self.dtype != torch.float32:
            residual = residual.float()
        for block, cache in zip(
            self.model.blocks[:self.model.active_layers], state.layers
        ):
            normalized = block.norm1(residual)
            if isinstance(block.attention, MatrixPoolDualResolutionAttention):
                delta = self._matrix_attention(
                    block.attention, normalized, residual, cache, position
                )
            elif isinstance(block.attention, StandardAttention):
                delta = self._standard_attention(
                    block.attention, normalized, cache, position
                )
            else:
                raise NotImplementedError("unsupported PyTorch incremental attention")
            if block.alpha_attn is not None:
                delta = delta * block.alpha_attn
            residual = residual.float() + delta.float()
            normalized_ffn = block.norm2(residual)
            ffn = (
                block.dense_ffn(normalized_ffn)
                if hasattr(block, "dense_ffn") else block.moe(normalized_ffn)
            )
            if block.alpha_mlp is not None:
                ffn = ffn * block.alpha_mlp
            residual = residual + ffn.float()
        hidden = self.model.final_norm(residual)
        if self.model.config.tie_embeddings:
            logits = hidden @ self.model.embedding.W.T
        else:
            logits = hidden @ self.model.output_proj.W
        state.position += 1
        return logits

    def prefill(self, token_ids, state: GenerationState):
        torch = _torch()
        ids = torch.as_tensor(token_ids, device=self.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError("prefill expects [B,T] token IDs")
        logits = None
        with torch.no_grad():
            for index in range(ids.shape[1]):
                logits = self._decode(ids[:, index:index + 1], state)
        return logits

    def decode_one(self, token_ids, state: GenerationState):
        torch = _torch()
        with torch.no_grad():
            return self._decode(token_ids, state)


class TorchGeneratorAdapter:
    def __init__(self, model, tokenizer, context_length):
        self.engine = TorchGenerationEngine(model)
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        token = getattr(tokenizer, "eos_token", None)
        mapping = getattr(tokenizer, "token_to_id", None)
        eos = mapping.get(token) if isinstance(mapping, dict) and token is not None else None
        if eos is None and getattr(tokenizer, "_tokenizer", None) is not None and token is not None:
            eos = tokenizer._tokenizer.token_to_id(token)
        self.stop_ids = set() if eos is None else {int(eos)}

    def encode(self, prompt):
        return list(self.tokenizer.encode(prompt))

    def decode(self, ids):
        return self.tokenizer.decode(list(ids))

    def create_state(self, total_tokens):
        return self.engine.new_state(batch_size=1, max_length=int(total_tokens))

    def prefill(self, ids, state):
        return self.engine.prefill([ids], state)

    def decode_one(self, token_id, state):
        return self.engine.decode_one([[int(token_id)]], state)

    def execution_report(self, state):
        return state.execution_report


def sample_token(logits, temperature=0.0, top_p=1.0, top_k=None, *,
                 repetition_penalty=1.0, frequency_penalty=0.0,
                 presence_penalty=0.0, recent_counts=None):
    torch = _torch()
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid temperature or top_p")
    row = logits.reshape(-1).float()
    if recent_counts and (repetition_penalty != 1.0 or frequency_penalty or presence_penalty):
        row = row.clone()
        ids = torch.tensor(list(recent_counts), device=row.device, dtype=torch.long)
        counts = torch.tensor(list(recent_counts.values()), device=row.device, dtype=torch.float32)
        values = row[ids]
        if repetition_penalty != 1.0:
            values = torch.where(values > 0, values / repetition_penalty,
                                 values * repetition_penalty)
        values = values - float(frequency_penalty) * counts
        if presence_penalty:
            values = values - float(presence_penalty)
        row[ids] = values
    if temperature == 0:
        return int(torch.argmax(row).item())
    count = row.numel() if top_k is None else min(row.numel(), int(top_k))
    if count < 1:
        raise ValueError("top_k must be >= 1")
    values, indices = torch.topk(row / float(temperature), count)
    probabilities = torch.softmax(values, dim=0)
    if top_p < 1:
        cumulative = torch.cumsum(probabilities, dim=0)
        keep = cumulative - probabilities < float(top_p)
        probabilities = probabilities * keep
        probabilities = probabilities / probabilities.sum()
    slot = torch.multinomial(probabilities, 1)
    return int(indices[slot].item())


__all__ = [
    "GenerationState", "TorchGenerationEngine", "TorchGeneratorAdapter", "sample_token",
]
