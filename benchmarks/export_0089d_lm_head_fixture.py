#!/usr/bin/env python3
"""Export the deterministic synthetic exact-geometry fixture for PATCH 0089D."""

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

from benchmarks.reference_0089d_torch_lm_head import (
    BATCH,
    CHUNK_TOKENS,
    D_MODEL,
    FIXTURE_ARRAY_NAMES,
    GRAD_SCALE,
    NORMALIZER_COUNT,
    ROWS,
    SEQ,
    TARGET_SLICE,
    VALID_LENGTHS,
    VOCAB,
    array_sha256,
    bf16_bits_from_float32,
    graph_contract,
)

DEFAULT_SEED = 20261024


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _synthetic_loss_mask() -> np.ndarray:
    return (
        np.arange(SEQ, dtype=np.int64)[None, :]
        < np.asarray(VALID_LENGTHS, dtype=np.int64)[:, None]
    ).astype(np.float32)


def _generate_arrays(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    head_f32 = rng.normal(0.0, 0.2, size=(BATCH, SEQ, D_MODEL)).astype(
        np.float32
    )
    head_bits = bf16_bits_from_float32(head_f32)
    del head_f32
    weight_f32 = rng.normal(0.0, 0.02, size=(D_MODEL, VOCAB)).astype(
        np.float32
    )
    weight_bits = bf16_bits_from_float32(weight_f32)
    del weight_f32
    targets = rng.integers(
        0, VOCAB, size=(BATCH, SEQ), dtype=np.int32
    )
    return {
        "head_bits": head_bits,
        "weight_bits": weight_bits,
        "targets": targets,
        "loss_mask": _synthetic_loss_mask(),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    arrays = _generate_arrays(args.seed)
    if tuple(arrays) != FIXTURE_ARRAY_NAMES:
        raise RuntimeError("generated fixture array order changed")
    normalizer = float(np.sum(arrays["loss_mask"], dtype=np.float64))
    if normalizer != NORMALIZER_COUNT:
        raise RuntimeError(
            f"synthetic mask normalizer {normalizer} != {NORMALIZER_COUNT}"
        )
    manifest = {
        "schema_version": "0089D-lm-head-fixture-v1",
        "description": (
            "deterministic synthetic BF16 final-normalized head input and tied "
            "output weight at exact production B3/T2304/D1024/V32768 geometry"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": int(args.seed),
        "batch": BATCH,
        "seq": SEQ,
        "rows": ROWS,
        "d_model": D_MODEL,
        "vocab": VOCAB,
        "chunk_tokens": CHUNK_TOKENS,
        "target_slice": list(TARGET_SLICE),
        "valid_lengths": list(VALID_LENGTHS),
        "normalizer_count": NORMALIZER_COUNT,
        "grad_scale": GRAD_SCALE,
        "array_names": list(FIXTURE_ARRAY_NAMES),
        "bf16_storage": "uint16 byte-exact BF16 bit patterns",
        "target_storage": "int32",
        "loss_mask_storage": "float32 prefix-valid binary mask",
        "fixture_source": "synthetic because checkpoint model_params are absent",
        "graph": graph_contract(),
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
