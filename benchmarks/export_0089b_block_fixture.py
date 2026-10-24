#!/usr/bin/env python3
"""Export a checkpoint-derived deterministic fixture for PATCH 0089B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.reference_0089b_torch_block import (
    D_FF,
    D_HEAD,
    D_MODEL,
    LOCAL_WINDOW,
    N_KV_HEADS,
    N_Q_HEADS,
    PARAMETER_KEYS,
    PARAMETER_NAMES,
    PARAMETER_SHAPES,
    POSITION_START,
    RMS_EPS,
    ROPE_BASE,
    array_sha256,
    bf16_bits_from_float32,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=129)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--position-start", type=int, default=POSITION_START)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.layer < 0 or args.batch <= 0 or args.seq <= 0 or args.position_start < 0:
        parser.error("layer/position-start must be non-negative and batch/seq positive")
    return args


def _source_name(block_name: str, layer: int) -> str:
    if not block_name.startswith("block."):
        raise ValueError(f"unexpected block parameter name: {block_name}")
    return f"blocks.{int(layer)}.{block_name[len('block.'):]}"


def _parameter_bits(value) -> np.ndarray:
    """Preserve exact BF16 values via an FP32 conversion and exact re-encoding."""
    return bf16_bits_from_float32(np.asarray(value, dtype=np.float32))


def _validate_config(config, layer: int) -> None:
    expected = {
        "d_model": D_MODEL,
        "d_ff": D_FF,
        "n_q_heads": N_Q_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "d_head": D_HEAD,
        "rope_base": ROPE_BASE,
        "rms_eps": RMS_EPS,
        "ffn_mode": "dense",
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"checkpoint config {name}={getattr(config, name)!r}, expected {value!r}"
            )
    if not 0 <= layer < int(config.n_layers):
        raise ValueError(f"layer {layer} outside checkpoint depth {config.n_layers}")
    if config.attention_layers is None:
        raise ValueError("checkpoint must define the hybrid attention topology")
    heads = config.attention_layers[layer].heads
    windows = tuple(int(head.window) for head in heads)
    if windows != (4096,) + (LOCAL_WINDOW,) * (N_Q_HEADS - 1):
        raise ValueError(f"unexpected layer attention windows: {windows}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("MINI_LLM_BACKEND", "numpy")

    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig

    config_path = args.checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = ModelConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    _validate_config(config, args.layer)

    source_names = tuple(_source_name(name, args.layer) for name in PARAMETER_NAMES)
    loaded, _, _ = load_checkpoint(
        args.checkpoint,
        param_names=list(source_names),
        skip_optimizer=True,
        skip_training=True,
    )
    missing = sorted(set(source_names) - set(loaded))
    extra = sorted(set(loaded) - set(source_names))
    if missing or extra:
        raise ValueError(f"checkpoint parameter mismatch: missing={missing}, extra={extra}")

    arrays: dict[str, np.ndarray] = {}
    for block_name, source_name in zip(PARAMETER_NAMES, source_names):
        bits = _parameter_bits(loaded[source_name])
        expected_shape = PARAMETER_SHAPES[block_name]
        if bits.shape != expected_shape:
            raise ValueError(f"{source_name} shape {bits.shape} != {expected_shape}")
        arrays[PARAMETER_KEYS[block_name]] = bits

    rng = np.random.default_rng(args.seed)
    arrays["x"] = rng.normal(
        0.0, 0.2, (args.batch, args.seq, D_MODEL)
    ).astype(np.float32)
    arrays["d_out"] = rng.normal(
        0.0, 0.2, (args.batch, args.seq, D_MODEL)
    ).astype(np.float32)
    arrays["position_ids"] = np.arange(
        args.position_start, args.position_start + args.seq, dtype=np.int64
    )

    manifest = {
        "schema_version": "0089B-block-fixture-v1",
        "description": (
            "checkpoint-derived layer parameters plus deterministic FP32 residual "
            "input/upstream gradient; no checkpoint I/O in timed regions"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "layer": int(args.layer),
        "batch": int(args.batch),
        "seq": int(args.seq),
        "seed": int(args.seed),
        "position_start": int(args.position_start),
        "parameter_names": list(PARAMETER_NAMES),
        "checkpoint_parameter_names": list(source_names),
        "parameter_keys": dict(PARAMETER_KEYS),
        "parameter_storage": "uint16 byte-exact BF16 bit patterns",
        "x_storage": "float32 residual stream",
        "d_out_storage": "float32 upstream residual gradient",
        "config": {
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "q_heads": N_Q_HEADS,
            "kv_heads": N_KV_HEADS,
            "d_head": D_HEAD,
            "local_window": LOCAL_WINDOW,
            "full_window": 4096,
            "rope_base": ROPE_BASE,
            "rms_eps": RMS_EPS,
            "ffn_mode": "dense",
            "packed_qkv": True,
            "fused_expert_gemm": False,
            "fp32_residual": True,
        },
        "sha256": {name: array_sha256(value) for name, value in arrays.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest_json": np.asarray(json.dumps(manifest))}
    payload.update(arrays)
    with args.output.open("wb") as handle:
        np.savez(handle, **payload)  # type: ignore[arg-type]
    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
