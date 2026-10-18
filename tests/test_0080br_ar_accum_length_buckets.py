"""0080BR: loss-preserving re-batching within gradient accumulation."""
import collections

import numpy as np
import pytest

from mini_llm.data.accumulation_length_buckets import regroup_ar_accumulation_batches
from mini_llm.backend import BACKEND_NAME
from test_document_first_sampling_0078e import _make_trainer


def _make_sample_batches(lengths, *, batch_size=3, target=16, source="fineweb_edu",
                         retrieval=False):
    batches = []
    for start in range(0, len(lengths), batch_size):
        part = lengths[start:start + batch_size]
        inputs = np.zeros((batch_size, 24), dtype=np.uint16)
        targets = np.zeros((batch_size, target), dtype=np.uint16)
        md = {
            "target_valid_lengths": np.asarray(part, dtype=np.int64),
            "target_loss_mask": np.arange(target)[None, :] < np.asarray(part)[:, None],
            "history_valid_starts": np.full(batch_size, 7, dtype=np.int64),
            "document_starts": np.asarray(list(range(start, start+batch_size)), dtype=np.int64),
            "document_ends": np.asarray(list(range(start, start+batch_size)), dtype=np.int64) + 20,
            "use_retrieval": retrieval,
            "retrieval_requested": retrieval,
            "document_sampling_policy": "document_first",
            "document_aware": True,
        }
        for row in range(batch_size):
            unique = start + row + 1
            inputs[row, :] = unique
            targets[row, :] = unique + 1
        batches.append((source, inputs, targets, md))
    return batches


def _rows(batches):
    result = []
    for source, inputs, targets, md in batches:
        for i in range(len(inputs)):
            result.append((source, int(inputs[i, 0]), int(targets[i, 0]),
                           int(md["target_valid_lengths"][i]),
                           int(md["document_starts"][i]),
                           bool(md["use_retrieval"]),
                           tuple(np.asarray(md["target_loss_mask"])[i])))
    return collections.Counter(result)


def test_presampling_reduces_work_and_preserves_all_rows():
    lengths = [2, 14, 3, 4, 16, 5, 6, 15, 7, 8, 13, 9]
    batches = _make_sample_batches(lengths)
    reordered, stats = regroup_ar_accumulation_batches(
        batches, batch_size=3, target_length=16, bucket_tokens=4,
    )
    assert _rows(reordered) == _rows(batches)
    assert stats["bucketed_rows"] < stats["original_rows"]
    assert len(reordered) == len(batches)
    assert stats["target_capacity_rows"] == len(lengths)*16
    assert all(x[0] == "fineweb_edu" for x in reordered)


def test_retrieval_and_source_modes_do_not_mix():
    batches = (
        _make_sample_batches([3, 12, 4], source="fineweb_edu")
        + _make_sample_batches([9, 7, 10], source="github")
        + _make_sample_batches([16, 16, 16], source="fineweb_edu", retrieval=True)
    )
    reordered, stats = regroup_ar_accumulation_batches(
        batches, batch_size=3, target_length=16, bucket_tokens=4,
    )
    assert _rows(reordered) == _rows(batches)
    assert stats["strata"] == 3


def test_reject_non_prefix_masks():
    batches = _make_sample_batches([4, 5, 6])
    batches[0][3]["target_loss_mask"][0, 1] = False
    with pytest.raises(ValueError, match="contiguous binary"):
        regroup_ar_accumulation_batches(
            batches, batch_size=3, target_length=16, bucket_tokens=4,
        )


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU optimizer parity test")
def test_train_step_gradient_matches_without_rebatching(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_LLM_AR_COMPACT_TARGET", "1")
    # All samples have valid 4-position window or fewer, and are sampled in
    # exactly the same original order regardless of subsequent regrouping.
    docs = [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _make_trainer(tmp_path / "a", documents=docs, batch_size=2)
    b = _make_trainer(tmp_path / "b", documents=docs, batch_size=2)
    for trainer in (a, b):
        trainer.grad_accum_steps = 4
        trainer.gradient_accumulation_weighting = "supervised_tokens"
        trainer.ar_accum_length_buckets = trainer is b
    loss_a, norm_a = a.train_step()
    loss_b, norm_b = b.train_step()
    assert np.isfinite(loss_a) and np.isfinite(loss_b)
    assert abs(loss_a-loss_b) < 1e-5
    assert abs(float(norm_a)-float(norm_b)) < 1e-4
    assert b._last_step_ar_bucket["bucketed_rows"] <= b._last_step_ar_bucket["original_rows"]
    assert a.train_source_supervised_tokens == b.train_source_supervised_tokens
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        np.testing.assert_allclose(np.asarray(pa.data), np.asarray(pb.data), atol=1e-5, rtol=1e-5)
