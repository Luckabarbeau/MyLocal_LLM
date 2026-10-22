#!/usr/bin/env python3
"""Export deterministic post-RoPE BF16 attention tensors for isolated 0087B runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.reference_0087b_torch_attention import (
    ATTENTION_SCALE,
    D_HEAD,
    LOCAL_WINDOW,
    N_KV_HEADS,
    N_Q_HEADS,
    array_sha256,
    bf16_bits_from_float32,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seq", type=int, default=2304)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch <= 0 or args.seq <= 0:
        parser.error("batch and sequence length must be positive")
    return args


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    q_bits = bf16_bits_from_float32(
        rng.normal(0.0, 0.2, (args.batch, args.seq, N_Q_HEADS, D_HEAD)).astype(np.float32)
    )
    k_bits = bf16_bits_from_float32(
        rng.normal(0.0, 0.2, (args.batch, args.seq, N_KV_HEADS, D_HEAD)).astype(np.float32)
    )
    v_bits = bf16_bits_from_float32(
        rng.normal(0.0, 0.2, (args.batch, args.seq, N_KV_HEADS, D_HEAD)).astype(np.float32)
    )
    d_out = rng.normal(
        0.0, 0.2, (args.batch, args.seq, N_Q_HEADS, D_HEAD)
    ).astype(np.float32)
    arrays = {"q_bits": q_bits, "k_bits": k_bits, "v_bits": v_bits, "d_out": d_out}
    manifest = {
        "schema_version": "0087B-attention-fixture-v1",
        "description": "deterministic post-RoPE tensors; no checkpoint I/O in timed regions",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": args.seed,
        "batch": args.batch,
        "seq": args.seq,
        "q_heads": N_Q_HEADS,
        "kv_heads": N_KV_HEADS,
        "d_head": D_HEAD,
        "local_window": LOCAL_WINDOW,
        "scale": ATTENTION_SCALE,
        "qkv_storage": "uint16 byte-exact BF16 bit patterns",
        "d_out_storage": "float32 production upstream-gradient contract",
        "q_to_kv": [head // 4 for head in range(N_Q_HEADS)],
        "sha256": {name: array_sha256(array) for name, array in arrays.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, manifest_json=np.asarray(json.dumps(manifest)), **arrays)
    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
