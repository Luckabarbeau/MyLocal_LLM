"""Shared fixture contract and compiled-Torch graph for PATCH 0089D."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

BATCH = 3
SEQ = 2304
D_MODEL = 1024
VOCAB = 32768
ROWS = BATCH * SEQ
CHUNK_TOKENS = 8192
TARGET_SLICE = (0, SEQ)
VALID_LENGTHS = (SEQ, 1792, 1024)
NORMALIZER_COUNT = float(sum(VALID_LENGTHS))
GRAD_SCALE = 1.0

FIXTURE_ARRAY_NAMES = (
    "head_bits",
    "weight_bits",
    "targets",
    "loss_mask",
)
RESULT_TENSOR_NAMES = ("loss", "dx", "dweight")


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
    """Hash dtype, shape, and bytes so reinterpretations cannot collide."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def expected_array_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "head_bits": (BATCH, SEQ, D_MODEL),
        "weight_bits": (D_MODEL, VOCAB),
        "targets": (BATCH, SEQ),
        "loss_mask": (BATCH, SEQ),
    }


def expected_array_dtypes() -> dict[str, np.dtype]:
    return {
        "head_bits": np.dtype(np.uint16),
        "weight_bits": np.dtype(np.uint16),
        "targets": np.dtype(np.int32),
        "loss_mask": np.dtype(np.float32),
    }


def graph_contract() -> dict:
    """Exact production graph boundaries represented by the fixture."""
    return {
        "head_shape": [BATCH, SEQ, D_MODEL],
        "weight_shape": [D_MODEL, VOCAB],
        "rows": ROWS,
        "chunk_capacity": CHUNK_TOKENS,
        "effective_tiles": [[0, ROWS]],
        "target_slice": list(TARGET_SLICE),
        "normalizer_count": NORMALIZER_COUNT,
        "grad_scale": GRAD_SCALE,
        "head_dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "logits_dtype": "bfloat16",
        "loss_dtype": "float32",
        "softmax_compute_dtype": "float32",
        "probability_rounding": "float32_to_bfloat16_before_target_subtraction",
        "dlogits_dtype": "bfloat16",
        "dweight_tile_dtype": "bfloat16",
        "dweight_accumulator_dtype": "float32",
        "dx_dtype": "bfloat16",
        "forward_retains_logits": False,
        "backward_recomputes_logits": True,
        "target_slice_copy": True,
        "target_gradient_zero_scatter": True,
        "tied_weight_orientation": "d_model_by_vocab_transpose_view",
    }


def validate_manifest(manifest: Mapping) -> None:
    if manifest.get("schema_version") != "0089D-lm-head-fixture-v1":
        raise ValueError("unsupported 0089D fixture schema")
    exact = {
        "batch": BATCH,
        "seq": SEQ,
        "rows": ROWS,
        "d_model": D_MODEL,
        "vocab": VOCAB,
        "chunk_tokens": CHUNK_TOKENS,
        "normalizer_count": NORMALIZER_COUNT,
        "grad_scale": GRAD_SCALE,
    }
    for name, expected in exact.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"fixture {name}={manifest.get(name)!r}, expected {expected!r}"
            )
    if tuple(manifest.get("target_slice", ())) != TARGET_SLICE:
        raise ValueError("fixture target_slice does not match production contract")
    if tuple(manifest.get("valid_lengths", ())) != VALID_LENGTHS:
        raise ValueError("fixture valid_lengths do not match synthetic mask contract")
    if tuple(manifest.get("array_names", ())) != FIXTURE_ARRAY_NAMES:
        raise ValueError("fixture array order does not match 0089D contract")
    if manifest.get("graph") != graph_contract():
        raise ValueError("fixture graph contract does not match 0089D")


def load_fixture(path: Path | str) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        arrays = {
            name: np.array(archive[name], copy=True) for name in FIXTURE_ARRAY_NAMES
        }
    validate_manifest(manifest)
    shapes = expected_array_shapes()
    dtypes = expected_array_dtypes()
    hashes = manifest.get("sha256", {})
    for name in FIXTURE_ARRAY_NAMES:
        value = arrays[name]
        if value.shape != shapes[name]:
            raise ValueError(f"fixture {name} shape {value.shape} != {shapes[name]}")
        if value.dtype != dtypes[name]:
            raise ValueError(f"fixture {name} dtype {value.dtype} != {dtypes[name]}")
        actual_hash = array_sha256(value)
        if hashes.get(name) != actual_hash:
            raise ValueError(
                f"fixture hash mismatch for {name}: {actual_hash} != {hashes.get(name)}"
            )
    targets = arrays["targets"]
    if np.any(targets < 0) or np.any(targets >= VOCAB):
        raise ValueError("fixture targets lie outside the vocabulary")
    mask = arrays["loss_mask"]
    expected_mask = (
        np.arange(SEQ, dtype=np.int64)[None, :]
        < np.asarray(VALID_LENGTHS, dtype=np.int64)[:, None]
    ).astype(np.float32)
    if not np.array_equal(mask, expected_mask):
        raise ValueError("fixture loss_mask is not the declared prefix-valid mask")
    if float(np.sum(mask, dtype=np.float64)) != NORMALIZER_COUNT:
        raise ValueError("fixture masked normalizer does not match manifest")
    return manifest, arrays


def torch_fixture_tensors(torch, arrays: Mapping[str, np.ndarray]) -> dict:
    """Decode portable fixture storage; the runner owns device transfer."""
    return {
        "head_bits": arrays["head_bits"],
        "weight_bits": arrays["weight_bits"],
        "targets": arrays["targets"],
        "loss_mask": arrays["loss_mask"],
    }


def build_torch_functions(
    torch,
    *,
    batch: int = BATCH,
    seq: int = SEQ,
    d_model: int = D_MODEL,
    vocab: int = VOCAB,
    chunk_tokens: int = CHUNK_TOKENS,
    target_slice: tuple[int, int] = TARGET_SLICE,
    normalizer_count: float = NORMALIZER_COUNT,
    grad_scale: float = GRAD_SCALE,
):
    """Build explicit forward and recompute-backward functions.

    The exact 0089D geometry has one effective tile. Forward and backward are
    intentionally separate callables so ``torch.compile`` cannot common-subexpr
    eliminate the production logits recomputation in combined measurements.
    """
    batch = int(batch)
    seq = int(seq)
    d_model = int(d_model)
    vocab = int(vocab)
    chunk_tokens = int(chunk_tokens)
    start, stop = map(int, target_slice)
    rows = batch * (stop - start)
    normalizer_count = float(normalizer_count)
    grad_scale = float(grad_scale)
    if min(batch, seq, d_model, vocab, chunk_tokens) <= 0:
        raise ValueError("Torch LM-head dimensions and chunk capacity must be positive")
    if not (0 <= start < stop <= seq):
        raise ValueError("target_slice lies outside the full head sequence")
    if rows > chunk_tokens:
        raise ValueError("0089D reference supports the declared one-tile contract only")
    if normalizer_count <= 0.0:
        raise ValueError("normalizer_count must be positive")

    def forward_loss(head, weight, targets, loss_mask):
        selected = head[:, start:stop, :].clone(
            memory_format=torch.contiguous_format
        )
        flat_x = selected.reshape(rows, d_model)
        flat_targets = targets.reshape(rows)
        flat_mask = loss_mask.reshape(rows)
        logits = flat_x @ weight
        work = logits.float()
        row_max = torch.amax(work, dim=-1, keepdim=True)
        shifted = work - row_max
        target_shifted = torch.gather(
            shifted, 1, flat_targets[:, None]
        ).squeeze(1)
        denominator = torch.sum(torch.exp(shifted), dim=-1)
        row_losses = (torch.log(denominator) - target_shifted) * flat_mask
        loss = torch.sum(row_losses) / normalizer_count
        return loss, selected

    def backward_recompute(selected, weight, targets, loss_mask, dweight_accum):
        flat_x = selected.reshape(rows, d_model)
        flat_targets = targets.reshape(rows)
        flat_mask = loss_mask.reshape(rows)
        logits = flat_x @ weight
        work = logits.float()
        row_max = torch.amax(work, dim=-1, keepdim=True)
        probabilities = torch.exp(work - row_max)
        probabilities = probabilities / torch.sum(
            probabilities, dim=-1, keepdim=True
        )
        rounded_probabilities = probabilities.to(torch.bfloat16).float()
        minus_one = -torch.ones(
            (rows, 1), device=rounded_probabilities.device, dtype=torch.float32
        )
        rounded_probabilities.scatter_add_(
            1, flat_targets[:, None], minus_one
        )
        row_scale = flat_mask[:, None] * (grad_scale / normalizer_count)
        dlogits = (rounded_probabilities * row_scale).to(torch.bfloat16)
        dweight_tile = flat_x.transpose(0, 1) @ dlogits
        dweight_accum.add_(dweight_tile.float())
        dx_selected = (dlogits @ weight.transpose(0, 1)).reshape(
            batch, stop - start, d_model
        )
        dx_full = torch.zeros(
            (batch, seq, d_model),
            device=selected.device,
            dtype=torch.bfloat16,
        )
        dx_full[:, start:stop, :].copy_(dx_selected)
        return dx_full, dweight_accum

    return forward_loss, backward_recompute


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual_f = np.asarray(actual, dtype=np.float32)
    reference_f = np.asarray(reference, dtype=np.float32)
    if actual_f.shape != reference_f.shape:
        raise ValueError(
            f"parity shape mismatch: actual {actual_f.shape}, reference {reference_f.shape}"
        )
    difference = actual_f - reference_f
    flat_actual = actual_f.reshape(-1)
    flat_reference = reference_f.reshape(-1)
    flat_difference = difference.reshape(-1)
    worst = int(np.argmax(np.abs(flat_difference))) if flat_difference.size else 0
    actual_norm = float(np.linalg.norm(flat_actual))
    reference_norm = float(np.linalg.norm(flat_reference))
    difference_norm = float(np.linalg.norm(flat_difference))
    denominator = actual_norm * reference_norm
    cosine = (
        float(np.dot(flat_actual, flat_reference) / denominator)
        if denominator
        else (1.0 if difference_norm == 0.0 else 0.0)
    )
    return {
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference)))
        if difference.size
        else 0.0,
        "relative_l2": difference_norm / reference_norm
        if reference_norm
        else (0.0 if difference_norm == 0.0 else float("inf")),
        "cosine": cosine,
        "bias": float(np.mean(difference)) if difference.size else 0.0,
        "worst_pair": {
            "flat_index": worst,
            "index": list(np.unravel_index(worst, actual_f.shape))
            if actual_f.size
            else [],
            "actual": float(flat_actual[worst]) if flat_actual.size else 0.0,
            "reference": float(flat_reference[worst]) if flat_reference.size else 0.0,
        },
    }


__all__ = [
    "BATCH",
    "SEQ",
    "D_MODEL",
    "VOCAB",
    "ROWS",
    "CHUNK_TOKENS",
    "TARGET_SLICE",
    "VALID_LENGTHS",
    "NORMALIZER_COUNT",
    "GRAD_SCALE",
    "FIXTURE_ARRAY_NAMES",
    "RESULT_TENSOR_NAMES",
    "array_sha256",
    "bf16_bits_from_float32",
    "bf16_bits_to_float32",
    "build_torch_functions",
    "error_metrics",
    "expected_array_dtypes",
    "expected_array_shapes",
    "graph_contract",
    "load_fixture",
    "torch_fixture_tensors",
    "validate_manifest",
]
