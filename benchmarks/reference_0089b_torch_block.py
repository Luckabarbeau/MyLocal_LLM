"""Shared contracts and compiled-PyTorch reference for PATCH 0089B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from benchmarks.reference_0087b_torch_attention import (
    D_HEAD,
    LOCAL_WINDOW,
    N_KV_HEADS,
    N_Q_HEADS,
    boundary_contract as attention_boundary_contract,
    build_block_mask as build_attention_block_mask,
    error_metrics,
)

D_MODEL = 1024
D_FF = 4096
RMS_EPS = 1e-6
ROPE_BASE = 1_000_000.0
POSITION_START = 61_440

PARAMETER_NAMES = (
    "block.norm1.gamma",
    "block.attention.Wq",
    "block.attention.Wk",
    "block.attention.Wv",
    "block.attention.Wo",
    "block.norm2.gamma",
    "block.dense_ffn.expert.W_gate",
    "block.dense_ffn.expert.W_up",
    "block.dense_ffn.expert.W_down",
)
PARAMETER_KEYS = {
    "block.norm1.gamma": "norm1_gamma_bits",
    "block.attention.Wq": "wq_bits",
    "block.attention.Wk": "wk_bits",
    "block.attention.Wv": "wv_bits",
    "block.attention.Wo": "wo_bits",
    "block.norm2.gamma": "norm2_gamma_bits",
    "block.dense_ffn.expert.W_gate": "w_gate_bits",
    "block.dense_ffn.expert.W_up": "w_up_bits",
    "block.dense_ffn.expert.W_down": "w_down_bits",
}
PARAMETER_SHAPES = {
    "block.norm1.gamma": (D_MODEL,),
    "block.attention.Wq": (D_MODEL, D_MODEL),
    "block.attention.Wk": (D_MODEL, N_KV_HEADS * D_HEAD),
    "block.attention.Wv": (D_MODEL, N_KV_HEADS * D_HEAD),
    "block.attention.Wo": (D_MODEL, D_MODEL),
    "block.norm2.gamma": (D_MODEL,),
    "block.dense_ffn.expert.W_gate": (D_MODEL, D_FF),
    "block.dense_ffn.expert.W_up": (D_MODEL, D_FF),
    "block.dense_ffn.expert.W_down": (D_FF, D_MODEL),
}
RESULT_TENSOR_NAMES = (
    "output",
    "d_input",
    "grad_norm1_gamma",
    "grad_wq",
    "grad_wk",
    "grad_wv",
    "grad_wo",
    "grad_norm2_gamma",
    "grad_w_gate",
    "grad_w_up",
    "grad_w_down",
)


def bf16_bits_from_float32(values: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16, ties-to-even, and return portable uint16 bits."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    upper = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return upper.view(np.float32)


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and bytes so reshapes/reinterpretations cannot collide."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def expected_array_shapes(batch: int, seq: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "x": (int(batch), int(seq), D_MODEL),
        "d_out": (int(batch), int(seq), D_MODEL),
        "position_ids": (int(seq),),
    }
    shapes.update(
        {PARAMETER_KEYS[name]: PARAMETER_SHAPES[name] for name in PARAMETER_NAMES}
    )
    return shapes


def validate_manifest(manifest: Mapping) -> None:
    if manifest.get("schema_version") != "0089B-block-fixture-v1":
        raise ValueError("unsupported 0089B fixture schema")
    config = manifest.get("config", {})
    required = {
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "q_heads": N_Q_HEADS,
        "kv_heads": N_KV_HEADS,
        "d_head": D_HEAD,
        "local_window": LOCAL_WINDOW,
        "rope_base": ROPE_BASE,
        "rms_eps": RMS_EPS,
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"fixture config {name}={config.get(name)!r}, expected {expected!r}")
    if tuple(manifest.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("fixture parameter order does not match 0089B contract")


def load_fixture(path: Path | str) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        expected_names = tuple(expected_array_shapes(manifest["batch"], manifest["seq"]))
        arrays = {name: np.array(archive[name], copy=True) for name in expected_names}
    validate_manifest(manifest)
    expected_shapes = expected_array_shapes(manifest["batch"], manifest["seq"])
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if value.shape != expected_shape:
            raise ValueError(f"fixture {name} shape {value.shape} != {expected_shape}")
        expected_dtype = np.float32 if name in {"x", "d_out"} else (
            np.int64 if name == "position_ids" else np.uint16
        )
        if value.dtype != expected_dtype:
            raise ValueError(f"fixture {name} dtype {value.dtype} != {expected_dtype}")
        actual_hash = array_sha256(value)
        expected_hash = manifest["sha256"][name]
        if actual_hash != expected_hash:
            raise ValueError(f"fixture hash mismatch for {name}: {actual_hash} != {expected_hash}")
    expected_positions = np.arange(
        int(manifest["position_start"]),
        int(manifest["position_start"]) + int(manifest["seq"]),
        dtype=np.int64,
    )
    if not np.array_equal(arrays["position_ids"], expected_positions):
        raise ValueError("fixture position_ids are not the declared contiguous absolute positions")
    return manifest, arrays


def boundary_contract(sequence_length: int | None = None) -> dict:
    contract = attention_boundary_contract(tuple(range(1, N_Q_HEADS)))
    contract["sequence_length"] = None if sequence_length is None else int(sequence_length)
    return contract


def build_block_mask(torch, sequence_length: int):
    return build_attention_block_mask(
        torch, tuple(range(1, N_Q_HEADS)), int(sequence_length), include_full=True
    )


def torch_fixture_tensors(torch, arrays: Mapping[str, np.ndarray]) -> dict:
    """Return fixture arrays by semantic name; the runner owns device transfer."""
    parameters = {
        name: arrays[PARAMETER_KEYS[name]] for name in PARAMETER_NAMES
    }
    return {
        "x": arrays["x"],
        "d_out": arrays["d_out"],
        "position_ids": arrays["position_ids"],
        "parameters": parameters,
        **parameters,
    }


def _torch_rope_tables(torch, position_ids):
    positions = position_ids.to(device="cuda", dtype=torch.float32)
    frequencies = torch.pow(
        torch.tensor(ROPE_BASE, device="cuda", dtype=torch.float32),
        -torch.arange(0, D_HEAD, 2, device="cuda", dtype=torch.float32) / D_HEAD,
    )
    angles = positions[:, None] * frequencies[None, :]
    return (
        torch.cos(angles).to(torch.bfloat16).reshape(1, -1, 1, D_HEAD // 2),
        torch.sin(angles).to(torch.bfloat16).reshape(1, -1, 1, D_HEAD // 2),
    )


def build_torch_forward(torch, block_mask, position_ids):
    """Build the exact 0089B FP32-residual/BF16-branch block forward."""
    flex_attention = torch.nn.attention.flex_attention.flex_attention

    cos, sin = _torch_rope_tables(torch, position_ids)

    def rmsnorm_bf16(x, gamma):
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_EPS)
        return (x * inv_rms * gamma.float()).to(torch.bfloat16)

    def rope(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    def forward(
        x,
        norm1_gamma,
        packed_wqkv,
        wo,
        norm2_gamma,
        w_gate,
        w_up,
        w_down,
    ):
        batch, seq, _ = x.shape
        norm1 = rmsnorm_bf16(x, norm1_gamma)
        qkv = norm1.reshape(-1, D_MODEL) @ packed_wqkv
        q_end = D_MODEL
        k_end = q_end + N_KV_HEADS * D_HEAD
        q = rope(qkv[:, :q_end].reshape(batch, seq, N_Q_HEADS, D_HEAD))
        k = rope(qkv[:, q_end:k_end].reshape(batch, seq, N_KV_HEADS, D_HEAD))
        v = qkv[:, k_end:].reshape(batch, seq, N_KV_HEADS, D_HEAD)
        context = flex_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            block_mask=block_mask,
            scale=D_HEAD ** -0.5,
            enable_gqa=True,
        ).permute(0, 2, 1, 3).float()
        attention = (
            context.reshape(-1, D_MODEL).to(torch.bfloat16) @ wo
        ).reshape(batch, seq, D_MODEL)
        residual2 = x + attention.float()
        norm2 = rmsnorm_bf16(residual2, norm2_gamma).reshape(-1, D_MODEL)
        gate = norm2 @ w_gate
        up = norm2 @ w_up
        hidden = torch.nn.functional.silu(gate) * up
        ffn = (hidden @ w_down).reshape(batch, seq, D_MODEL)
        return residual2 + ffn.float()

    return forward


__all__ = [
    "PARAMETER_NAMES",
    "PARAMETER_KEYS",
    "PARAMETER_SHAPES",
    "RESULT_TENSOR_NAMES",
    "array_sha256",
    "bf16_bits_from_float32",
    "bf16_bits_to_float32",
    "boundary_contract",
    "build_block_mask",
    "build_torch_forward",
    "error_metrics",
    "expected_array_shapes",
    "load_fixture",
    "torch_fixture_tensors",
    "validate_manifest",
]
