"""CPU-only contracts for native heterogeneous supervised-token accumulation."""

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest

import train_heterogeneous as heterogeneous


def _config(tmp_path):
    return {
        "smoke": True,
        "dtype": "float32",
        "seed": 42,
        "timeout_s": 120,
        "log_dir": str(tmp_path / "logs"),
        "anchor": "anchor",
        "workers": [
            {"name": "anchor", "python": sys.executable, "backend": "numpy",
             "batch_size": 1, "context_length": 9, "target_microbatches": 2,
             "lm_head_chunk_tokens": 4},
            {"name": "other", "python": sys.executable, "backend": "numpy",
             "batch_size": 2, "context_length": 5, "lm_head_chunk_tokens": 3},
        ],
    }


def test_two_native_workers_train_two_steps_and_save_optimizer(tmp_path):
    config = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    config["checkpoint_dir"] = str(checkpoint)
    try:
        results = heterogeneous.run(config, steps=2, work_dir=str(tmp_path))
    except Exception as exc:
        logs = "\n".join(
            f"{path.name}:\n{path.read_text()}"
            for path in sorted((tmp_path / "logs").glob("*.log"))
        )
        pytest.fail(f"native subprocess training failed: {exc}\n{logs}")

    assert [result["step"] for result in results] == [1, 2]
    for result in results:
        counts = result["microbatches"]
        assert counts["anchor"] == 2
        assert counts["other"] >= 1
        assert result["supervised_tokens"] == 9 * counts["anchor"] + 10 * counts["other"]
        assert math.isfinite(result["loss"]) and result["loss"] > 0
        assert math.isfinite(result["grad_norm"]) and result["grad_norm"] > 0
    assert len(list((tmp_path / "logs").glob("heterogeneous-worker-*.log"))) == 2
    assert not list(tmp_path.glob("native-heterogeneous-*"))

    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["dtype"] == "float32"
    assert saved_config["n_layers"] == 2
    assert saved_config["d_model"] == 1024
    assert saved_config["n_q_heads"] == 16
    assert saved_config["n_kv_heads"] == 4
    assert saved_config["d_head"] == 64
    assert saved_config["dual_resolution"]["local_window"] == 512
    state = json.loads((checkpoint / "training_state.json").read_text())
    assert state["step"] == 2
    assert state["heterogeneous_results"] == results
    assert state["gradient_accumulation_weighting"] == "supervised_tokens"
    assert state["exact_resume_supported"] is False
    optimizer = checkpoint / "optimizer_state"
    assert json.loads((optimizer / "scalars.json").read_text())["step"] == 2
    for group in ("master_weights", "m", "v", "muon_m"):
        arrays = list((optimizer / group).glob("*.npy"))
        assert arrays, group
        assert all(np.isfinite(np.load(path, allow_pickle=False)).all() for path in arrays)
    assert any(np.any(np.load(path, allow_pickle=False))
               for path in (optimizer / "muon_m").glob("*.npy"))


@pytest.mark.parametrize("chunk_elements", [1, 4, 64])
def test_reduce_gradients_unequal_token_counts_and_chunk_tails(chunk_elements):
    means = [np.linspace(-2, 3, 11, dtype=np.float32),
             np.linspace(4, -1, 11, dtype=np.float32)]
    counts = [18, 30]
    sources = [mean * count for mean, count in zip(means, counts)]
    originals = [source.copy() for source in sources]
    destination = np.full(11, np.nan, dtype=np.float32)
    heterogeneous.reduce_gradients(sources, destination, sum(counts), chunk_elements)
    expected = (means[0].astype(np.float64) * 18 + means[1] * 30) / 48
    np.testing.assert_allclose(destination, expected, rtol=2e-6, atol=2e-7)
    assert not np.allclose(destination, (means[0] + means[1]) / 2)
    for source, original in zip(sources, originals):
        np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_worker_gradient_rejected_before_update(bad):
    source = np.ones(9, dtype=np.float32)
    source[-1] = bad  # Reject even when earlier reduction chunks were finite.
    update = Mock()
    with pytest.raises(ValueError, match="nonfinite worker gradient; update aborted"):
        heterogeneous.reduce_gradients([source], np.empty_like(source), 9, chunk_elements=4)
        update()
    update.assert_not_called()


def test_nonfinite_aggregate_rejected():
    source = np.full(5, np.finfo(np.float32).max, dtype=np.float32)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="nonfinite aggregate gradient; update aborted"):
            heterogeneous.reduce_gradients([source, source], np.empty_like(source), 2, 3)


@pytest.mark.parametrize("count", [0, -1, np.nan, np.inf])
def test_reduce_gradients_rejects_invalid_global_count(count):
    destination = np.full(5, 123, dtype=np.float32)
    with pytest.raises(ValueError, match="supervised-token count"):
        heterogeneous.reduce_gradients([np.ones(5, dtype=np.float32)], destination, count)
    np.testing.assert_array_equal(destination, np.full(5, 123, dtype=np.float32))


@pytest.mark.parametrize("sources", [[], [np.zeros(4, dtype=np.float32)]])
def test_reduce_gradients_rejects_buffer_mismatch(sources):
    with pytest.raises(ValueError, match="gradient buffer shape mismatch"):
        heterogeneous.reduce_gradients(sources, np.empty(5, dtype=np.float32), 1)


def test_validate_config_rejects_duplicate_names(tmp_path):
    config = _config(tmp_path)
    heterogeneous.validate_config(config)
    config["workers"][1]["name"] = "anchor"
    with pytest.raises(ValueError, match="worker names must be unique"):
        heterogeneous.validate_config(config)


def test_validate_config_requires_nonexact_initialization_acknowledgement(tmp_path):
    config = _config(tmp_path)
    config["init_from"] = str(tmp_path / "weights-only")
    with pytest.raises(ValueError, match="acknowledge_nonexact_init must be true"):
        heterogeneous.validate_config(config)
    config["acknowledge_nonexact_init"] = True
    heterogeneous.validate_config(config)


def test_native_worker_token_sums_match_serial_full_logit_microbatches(monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "numpy")
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.model_registry import registry
    from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata

    assert BACKEND_NAME == "numpy", "this regression must run on the native CPU backend"
    base = registry.config("model_b_348m_sparse16")
    config = replace(
        base, dtype="float32", tokenizer_vocab_size=64, context_length=32,
        n_layers=2, d_model=256, n_q_heads=4, n_kv_heads=2, d_head=64, d_ff=128,
        dual_resolution=replace(
            base.dual_resolution, block_size=4, local_window=4, long_local_window=8,
            long_local_heads=1, max_blocks=8, summaries_per_block=2,
            global_heads=2, global_head_dim=64, sparse_retrieval_layers=(0, 1)),
    )
    worker = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    serial = DecoderLanguageModel(config, rng_seed=42, dtype="float32")
    rng = np.random.default_rng(91)
    batches = []
    for shape in [(1, 9), (1, 9), (2, 5)]:
        raw = rng.integers(0, 63, size=(shape[0], shape[1] + 1), dtype=np.int64)
        mask = np.ones(shape, dtype=np.float32)
        if shape[0] == 2:
            mask[0, -3:] = 0
        batches.append((raw[:, :-1].copy(), raw[:, 1:].copy(), mask))
    total = sum(float(mask.sum()) for _, _, mask in batches)
    serial.zero_grad()
    serial_loss = 0.0
    for inputs, targets, mask in batches:
        metadata = build_dual_resolution_metadata(inputs, 63, 4)
        logits, cache = serial.forward(inputs, dual_resolution_metadata=metadata)
        loss, loss_cache = serial.compute_loss(logits, targets, loss_mask=mask)
        weight = float(mask.sum()) / total
        serial_loss += float(loss) * weight
        serial.backward(serial.backward_loss(loss_cache) * weight, cache)

    sources = []
    worker_loss = 0.0
    # Two local microbatches on worker zero, one on worker one, at identical weights.
    for local_batches in (batches[:2], batches[2:]):
        worker.zero_grad()
        for inputs, targets, mask in local_batches:
            count = float(mask.sum())
            metadata = build_dual_resolution_metadata(inputs, 63, 4)
            head, cache = worker.forward_body(
                inputs, activation_checkpoint=True, dual_resolution_metadata=metadata)
            loss, loss_cache = worker.chunked_lm_head_loss_forward(
                head, targets, loss_mask=mask, chunk_tokens=4, return_device_loss=True)
            worker.backward_body(worker.chunked_lm_head_backward(loss_cache, grad_scale=count), cache)
            worker_loss += float(loss) * count / total
        sources.append(np.concatenate([p.grad.ravel().copy() for p in worker.parameters()]))
    reduced = np.empty_like(sources[0])
    heterogeneous.reduce_gradients(sources, reduced, total, chunk_elements=127)
    reference = np.concatenate([p.grad.ravel() for p in serial.parameters()])
    assert np.linalg.norm(reference) > 0
    assert worker_loss == pytest.approx(serial_loss, rel=2e-6, abs=2e-6)
    np.testing.assert_allclose(reduced, reference, rtol=3e-4, atol=3e-6)
