"""Shared 0087B attention contract and conditional PyTorch implementations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

N_Q_HEADS = 16
N_KV_HEADS = 4
D_HEAD = 64
LOCAL_WINDOW = 512
ATTENTION_SCALE = D_HEAD ** -0.5

_HEAD_PATTERNS = {
    "15": tuple(range(1, 16)),
    "13": tuple(range(1, 14)),
    "9": tuple(range(1, 10)),
    "scattered": (1, 2, 4, 5, 7, 8, 10, 12, 15),
}


def resolve_active_heads(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        if value in _HEAD_PATTERNS:
            return _HEAD_PATTERNS[value]
        heads = tuple(int(part) for part in value.split(",") if part.strip())
    else:
        heads = tuple(int(head) for head in value)
    if not heads or tuple(sorted(set(heads))) != heads:
        raise ValueError("active local heads must be unique and increasing")
    if heads[0] < 1 or heads[-1] >= N_Q_HEADS:
        raise ValueError("local heads must retain original identities in [1, 15]")
    return heads


def original_q_to_kv(head: int) -> int:
    head = int(head)
    if not 0 <= head < N_Q_HEADS:
        raise ValueError("query head outside [0, 15]")
    return head // (N_Q_HEADS // N_KV_HEADS)


def mask_allows(head: int, query: int, key: int, active_local_heads: Iterable[int]) -> bool:
    head, query, key = int(head), int(query), int(key)
    if key > query:
        return False
    if head == 0:
        return True
    if head not in set(int(value) for value in active_local_heads):
        return False
    return key >= max(0, query - LOCAL_WINDOW + 1)


def boundary_contract(active_local_heads: Iterable[int]) -> dict:
    active = tuple(active_local_heads)
    return {
        "query": 519,
        "local_key_7_allowed": mask_allows(1, 519, 7, active),
        "local_key_8_allowed": mask_allows(1, 519, 8, active),
        "local_future_key_520_allowed": mask_allows(1, 519, 520, active),
        "full_key_7_allowed": mask_allows(0, 519, 7, active),
        "full_future_key_520_allowed": mask_allows(0, 519, 520, active),
    }


def bf16_bits_from_float32(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 and return portable uint16 bit patterns."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + rounding_bias
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    upper = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return upper.view(np.float32)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def load_fixture(path: Path | str) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        arrays = {name: np.array(archive[name], copy=True) for name in ("q_bits", "k_bits", "v_bits", "d_out")}
    for name, array in arrays.items():
        expected = manifest["sha256"][name]
        actual = array_sha256(array)
        if actual != expected:
            raise ValueError(f"fixture hash mismatch for {name}: {actual} != {expected}")
    expected_shapes = {
        "q_bits": (manifest["batch"], manifest["seq"], N_Q_HEADS, D_HEAD),
        "k_bits": (manifest["batch"], manifest["seq"], N_KV_HEADS, D_HEAD),
        "v_bits": (manifest["batch"], manifest["seq"], N_KV_HEADS, D_HEAD),
        "d_out": (manifest["batch"], manifest["seq"], N_Q_HEADS, D_HEAD),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != tuple(expected):
            raise ValueError(f"fixture {name} shape {arrays[name].shape} != {expected}")
    return manifest, arrays


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual64 = np.asarray(actual, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    if actual64.shape != reference64.shape:
        raise ValueError(f"shape mismatch: {actual64.shape} != {reference64.shape}")
    difference = actual64 - reference64
    absolute = np.abs(difference)
    worst_flat = int(np.argmax(absolute)) if absolute.size else 0
    worst_index = tuple(int(value) for value in np.unravel_index(worst_flat, absolute.shape))
    ref_norm = float(np.linalg.norm(reference64.reshape(-1)))
    actual_norm = float(np.linalg.norm(actual64.reshape(-1)))
    dot = float(np.dot(actual64.reshape(-1), reference64.reshape(-1)))
    cosine = 1.0 if actual_norm == 0.0 and ref_norm == 0.0 else dot / max(actual_norm * ref_norm, 1e-300)
    return {
        "max_abs": float(absolute.reshape(-1)[worst_flat]) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / max(ref_norm, 1e-300)),
        "cosine": cosine,
        "bias": float(np.mean(difference)) if difference.size else 0.0,
        "worst_index": list(worst_index),
        "actual_at_worst": float(actual64[worst_index]) if absolute.size else 0.0,
        "reference_at_worst": float(reference64[worst_index]) if absolute.size else 0.0,
    }


def build_mask_mod(torch, active_local_heads: Iterable[int], *, include_full: bool = True):
    active = tuple(int(head) for head in active_local_heads)

    def mask_mod(batch, head, query, key):
        del batch
        enabled = head == 0 if include_full else torch.zeros_like(head, dtype=torch.bool)
        for active_head in active:
            enabled = enabled | (head == active_head)
        causal = key <= query
        full = (head == 0) & causal if include_full else torch.zeros_like(causal)
        local = (head != 0) & causal & (key >= query - (LOCAL_WINDOW - 1))
        return enabled & (full | local)

    return mask_mod


def build_block_mask(torch, active_local_heads: Iterable[int], seq: int, *, include_full: bool = True):
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        build_mask_mod(torch, active_local_heads, include_full=include_full),
        B=None,
        H=N_Q_HEADS,
        Q_LEN=int(seq),
        KV_LEN=int(seq),
        device="cuda",
        BLOCK_SIZE=128,
        _compile=True,
    )


def flex_forward_bthd(torch, q, k, v, block_mask):
    from torch.nn.attention.flex_attention import flex_attention

    q_bhtd = q.permute(0, 2, 1, 3)
    k_bhtd = k.permute(0, 2, 1, 3)
    v_bhtd = v.permute(0, 2, 1, 3)
    output = flex_attention(
        q_bhtd,
        k_bhtd,
        v_bhtd,
        block_mask=block_mask,
        scale=ATTENTION_SCALE,
        enable_gqa=True,
    )
    return output.permute(0, 2, 1, 3).float()


def vendor_hybrid_forward_bthd(torch, q, k, v, local_block_mask):
    """Flash SDPA for original head 0, FlexAttention for local original heads."""
    import torch.nn.functional as functional
    from torch.nn.attention.flex_attention import flex_attention

    q_bhtd = q.permute(0, 2, 1, 3)
    k_bhtd = k.permute(0, 2, 1, 3)
    v_bhtd = v.permute(0, 2, 1, 3)
    local_output = flex_attention(
        q_bhtd,
        k_bhtd,
        v_bhtd,
        block_mask=local_block_mask,
        scale=ATTENTION_SCALE,
        enable_gqa=True,
    )
    full_output = functional.scaled_dot_product_attention(
        q_bhtd[:, 0:1],
        k_bhtd[:, 0:1],
        v_bhtd[:, 0:1],
        is_causal=True,
        scale=ATTENTION_SCALE,
    )
    output = torch.cat((full_output, local_output[:, 1:]), dim=1)
    return output.permute(0, 2, 1, 3).float()
