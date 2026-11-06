"""PyTorch-native attention operators for MyLocal_LLM.

The CUDA path uses FlexAttention and block masks.  A dense PyTorch reference is
kept for CPU correctness tests and unsupported accelerator configurations; it
never dispatches to the NumPy/CuPy backend.
"""

from __future__ import annotations

import math
from typing import Any, Callable


_COMPILED_FLEX_ATTENTION = None


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The PyTorch backend requires torch. Install the optional pytorch dependency."
        ) from exc
    return torch


def apply_rope(x, position_ids, base: float):
    """Apply interleaved RoPE to ``[B,T,H,D]`` with FP32 angle construction."""
    torch = _torch()
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("RoPE expects [B,T,H,even_head_dim]")
    positions = position_ids.to(device=x.device, dtype=torch.float32)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
    if positions.shape != x.shape[:2]:
        raise ValueError("position_ids must have shape [T] or [B,T]")
    frequencies = torch.pow(
        torch.tensor(float(base), device=x.device, dtype=torch.float32),
        -torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
        / x.shape[-1],
    )
    angles = positions[..., None] * frequencies
    cos = torch.cos(angles).to(dtype=x.dtype)[:, :, None, :]
    sin = torch.sin(angles).to(dtype=x.dtype)[:, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _dense_attention(q, k, v, allowed, scale: float, *, enable_gqa: bool):
    torch = _torch()
    # Inputs use [B,H,T,D]. Expand shared KV heads exactly as GQA specifies.
    if enable_gqa and q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1]:
            raise ValueError("query heads must be divisible by KV heads")
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
    if allowed.shape != scores.shape:
        allowed = allowed.expand_as(scores)
    active = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = torch.where(active, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = probabilities * allowed
    return torch.matmul(probabilities, v.float()).to(dtype=q.dtype)


def _materialize_mask(mask_mod: Callable, batch: int, heads: int, q_len: int,
                      kv_len: int, device):
    torch = _torch()
    b = torch.arange(batch, device=device)[:, None, None, None]
    h = torch.arange(heads, device=device)[None, :, None, None]
    q = torch.arange(q_len, device=device)[None, None, :, None]
    k = torch.arange(kv_len, device=device)[None, None, None, :]
    return mask_mod(b, h, q, k).expand(batch, heads, q_len, kv_len)


def flex_attention_with_mask(q, k, v, mask_mod: Callable, *, scale: float,
                             enable_gqa: bool = False,
                             block_mask_cache: dict | None = None,
                             cache_key: Any = None):
    """Execute FlexAttention on CUDA and an exact dense PyTorch reference on CPU."""
    torch = _torch()
    if q.device.type == "cuda":
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            global _COMPILED_FLEX_ATTENTION
            if _COMPILED_FLEX_ATTENTION is None:
                _COMPILED_FLEX_ATTENTION = torch.compile(
                    flex_attention, mode="max-autotune", dynamic=True
                )
            block_mask = (
                block_mask_cache.get(cache_key)
                if block_mask_cache is not None and cache_key is not None
                else None
            )
            if block_mask is None:
                block_mask = create_block_mask(
                    mask_mod,
                    B=int(q.shape[0]),
                    H=int(q.shape[1]),
                    Q_LEN=int(q.shape[2]),
                    KV_LEN=int(k.shape[2]),
                    device=q.device,
                    BLOCK_SIZE=128,
                    _compile=True,
                )
                if block_mask_cache is not None and cache_key is not None:
                    block_mask_cache[cache_key] = block_mask
            return _COMPILED_FLEX_ATTENTION(
                q, k, v, block_mask=block_mask, scale=float(scale),
                enable_gqa=bool(enable_gqa),
            )
        except (ImportError, NotImplementedError):
            # This remains native PyTorch. It is reported as a fallback by the engine.
            pass
    allowed = _materialize_mask(
        mask_mod, int(q.shape[0]), int(q.shape[1]), int(q.shape[2]),
        int(k.shape[2]), q.device,
    )
    return _dense_attention(q, k, v, allowed, scale, enable_gqa=enable_gqa)


def local_mask_mod(document_ids, *, local_window: int, long_local_window: int,
                   long_local_heads: int, sparse_head: int | None = None,
                   sparse_stride: int = 0, sparse_phase: int = -1):
    """Model B local/sparse visibility, including packed-document boundaries."""
    torch = _torch()

    def mask(batch, head, query, key):
        causal = key <= query
        same_document = document_ids[batch, query] == document_ids[batch, key]
        window = torch.where(
            head < int(long_local_heads),
            torch.full_like(head, int(long_local_window)),
            torch.full_like(head, int(local_window)),
        )
        local = causal & (key >= query - window + 1)
        if sparse_head is not None and sparse_stride > 0 and sparse_phase >= 0:
            selected_head = head == int(sparse_head)
            sparse = causal & (torch.remainder(key, int(sparse_stride)) == int(sparse_phase))
            local = torch.where(selected_head, sparse, local)
        return same_document & local

    return mask


def global_history_mask_mod(document_ids, position_ids, summary_documents,
                            summary_ends, summary_valid):
    """Compressed-history visibility: only completed same-document blocks."""
    def mask(batch, head, query, key):
        del head
        return (
            summary_valid[batch, key]
            & (document_ids[batch, query] == summary_documents[batch, key])
            & (summary_ends[batch, key] < position_ids[batch, query])
        )

    return mask


def standard_causal_attention(q, k, v, document_ids=None):
    """Full causal GQA, optionally bounded by packed-document IDs."""
    torch = _torch()
    batch, _, length, d_head = q.shape
    if document_ids is None:
        document_ids = torch.zeros((batch, length), device=q.device, dtype=torch.long)

    def mask(batch_index, head, query, key):
        del head
        return ((key <= query)
                & (document_ids[batch_index, query] == document_ids[batch_index, key]))

    return flex_attention_with_mask(
        q, k, v, mask, scale=d_head ** -0.5, enable_gqa=True,
    )


__all__ = [
    "apply_rope",
    "flex_attention_with_mask",
    "global_history_mask_mod",
    "local_mask_mod",
    "standard_causal_attention",
]
