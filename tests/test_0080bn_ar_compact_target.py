"""0080BN: AR trailing-padding compaction, preserving causal predictions."""

import numpy as np
import pytest

from mini_llm.data.compact_target import ar_compact_metadata
from mini_llm.train_extended import ExtendedTrainer
from test_document_first_sampling_0078e import _make_trainer


def _metadata(lengths, capacity=16):
    return {
        "target_valid_lengths": np.asarray(lengths, dtype=np.int64),
        "target_loss_mask": (
            np.arange(capacity)[None, :] < np.asarray(lengths)[:, None]
        ).astype(np.float32),
    }


@pytest.mark.parametrize("lengths,bucket,expected", [
    ((1, 1), 128, 16), ((4, 9), 4, 12), ((7, 16), 4, 16),
    ((1000, 1200), 128, 1280), ((3000,), 128, 3072),
    ((4096,), 128, 4096),
])
def test_compact_length_only_trims_padding(lengths, bucket, expected):
    capacity = 4096 if max(lengths) > 16 else 16
    md = _metadata(lengths, capacity)
    out = ar_compact_metadata(md, target_length=capacity, batch_size=len(lengths),
                              bucket_tokens=bucket)
    assert out.get("compact_working_length", capacity) == expected
    assert out.get("compact_ar_target", False) == (expected < capacity)
    assert "compact_ar_target" not in md
    np.testing.assert_array_equal(out["target_loss_mask"], md["target_loss_mask"])


@pytest.mark.parametrize("problem", ["hole", "fractional", "mismatched", "zero"])
def test_reject_unsafe_or_inconsistent_target_masks(problem):
    md = _metadata((5, 3))
    if problem == "hole":
        md["target_loss_mask"][0, 2] = 0
    elif problem == "fractional":
        md["target_loss_mask"][1, 0] = 0.5
    elif problem == "mismatched":
        md["target_valid_lengths"] = np.array([5, 12])
    else:
        md["target_valid_lengths"] = np.array([5, 0])
    with pytest.raises(ValueError, match="compact AR"):
        ar_compact_metadata(md, target_length=16, batch_size=2)


def _forward_backward(trainer, source, targets, metadata, *, enabled, chunked):
    from mini_llm.backend import asnumpy
    trainer.ar_compact_target = enabled
    md = trainer._configure_routed_prefix_metadata(metadata, training=False)
    tgt, mask = trainer._loss_targets_and_mask(targets, metadata["target_loss_mask"], md)
    trainer.model.zero_grad()
    if chunked:
        head, cache = trainer.model.forward_body(source, memory_metadata=md)
        loss, loss_cache = trainer.model.chunked_lm_head_loss_forward(
            head, tgt, chunk_tokens=2, loss_mask=mask, target_slice=cache["target_slice"],
        )
        dx = trainer.model.chunked_lm_head_backward(loss_cache)
        trainer.model.backward_body(dx, cache)
    else:
        logits, cache = trainer.model.forward(source, memory_metadata=md)
        loss, lc = trainer.model.compute_loss(logits, tgt, loss_mask=mask)
        trainer.model.backward(trainer.model.backward_loss(lc), cache)
    grads = {p.name: np.asarray(asnumpy(p.grad)).copy() for p in trainer.model.parameters()}
    return float(loss), grads, tgt, mask, md


@pytest.mark.parametrize("chunked", [True, False])
def test_exact_loss_and_all_gradients_match_padding_baseline(monkeypatch, tmp_path, chunked):
    monkeypatch.delenv("MINI_LLM_AR_COMPACT_TARGET", raising=False)
    trainer = _make_trainer(tmp_path, documents=[[1, 2, 3], [4, 5]], batch_size=2)
    source = np.full((2, 16), 31, dtype=np.uint16)
    source[0, 12:15] = [1, 2, 3]
    source[1, 12:14] = [4, 5]
    labels = np.full((2, 4), 31, dtype=np.uint16)
    labels[0, :2] = [2, 3]
    labels[1, 0] = 5
    md = _metadata((2, 1), 4)
    md.update(history_valid_starts=np.array([12, 12], dtype=np.int64),
              use_retrieval=False, router_stochastic=False)
    original = source.copy()
    reference = _forward_backward(trainer, source, labels, md, enabled=False, chunked=chunked)
    compact = _forward_backward(trainer, source, labels, md, enabled=True, chunked=chunked)
    np.testing.assert_array_equal(source, original)
    assert reference[2].shape == (2, 4) and compact[2].shape == (2, 2)
    assert compact[4]["compact_working_length"] == 2
    np.testing.assert_allclose(reference[0], compact[0], rtol=1e-6, atol=1e-6)
    assert reference[1].keys() == compact[1].keys()
    for name in reference[1]:
        np.testing.assert_allclose(reference[1][name], compact[1][name],
                                   rtol=2e-5, atol=2e-6, err_msg=name)


def test_full_training_step_and_validation_with_ar_compaction(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_LLM_AR_COMPACT_TARGET", "1")
    trainer = _make_trainer(tmp_path, documents=[[1, 2, 3], [4, 5]], batch_size=1)
    loss, norm = trainer.train_step()
    assert np.isfinite(loss) and np.isfinite(norm)
    assert trainer.step == 1
    assert trainer._last_step_ar_compact["executed"] < trainer._last_step_ar_compact["capacity"]
    assert np.isfinite(trainer.compute_val_loss())


def test_block_mode_rejects_ar_compact_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("MINI_LLM_AR_COMPACT_TARGET", raising=False)
    trainer = _make_trainer(tmp_path, documents=[[1, 2]])
    source, _, md = trainer.get_train_batch(return_metadata=True)
    md["compact_working_length"] = 2
    md["compact_ar_target"] = True
    md["causal_prefix_query"] = True
    with pytest.raises(ValueError, match="block proposal"):
        trainer.model.forward_body(source, memory_metadata=md, return_cache=False)
