"""0102X: document-first length grouping, mask safety, optimizer parity."""
import collections

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.data.dual_resolution_length_buckets import (
    regroup_dual_resolution_document_batches,
)
from mini_llm.ops.dual_resolution_attention import (
    build_dual_resolution_metadata, dual_resolution_loss_mask,
)

EOS = 99


def _batches(lengths, *, batch=2, cap=24, source="fineweb", shape_bucket="none"):
    results = []
    for j in range(0, len(lengths), batch):
        chosen = lengths[j:j + batch]
        width = max(chosen)
        if shape_bucket == "geometric":
            from mini_llm.data.token_shards import _dual_resolution_geometric_length
            width = _dual_resolution_geometric_length(width, cap)
        x = np.full((batch, width), EOS, dtype=np.uint16)
        y = np.full_like(x, EOS)
        mask = np.zeros(x.shape, dtype=np.float32)
        for i, n in enumerate(chosen):
            token = j + i + 1
            x[i, :n] = token
            y[i, :n] = token + 20
            mask[i, :n] = 1
        meta = {
            "document_sampling_policy": "document_first_full_sequence",
            "document_aware": True,
            "document_starts": np.arange(j, j + batch, dtype=np.int64) * 1000,
            "document_ends": np.arange(j, j + batch, dtype=np.int64) * 1000 + 100,
            "target_source_starts": np.arange(j, j + batch, dtype=np.int64) * 1000,
            "target_valid_lengths": np.asarray(chosen, dtype=np.int64),
            "target_loss_mask": mask,
            "executed_sequence_length": width,
            "context_capacity": cap,
            "shape_bucket": shape_bucket,
            "dual_resolution_metadata": {"stale": True},
        }
        results.append((source, x, y, meta))
    return results


def _row_signatures(batches):
    rows = []
    for source, x, y, md in batches:
        for row, length in enumerate(md["target_valid_lengths"]):
            n = int(length)
            rows.append((source, n, tuple(x[row, :n]), tuple(y[row, :n]),
                         tuple(md["target_loss_mask"][row, :n]),
                         int(md["document_starts"][row]),
                         int(md["document_ends"][row]),
                         int(md["target_source_starts"][row])))
    return collections.Counter(rows)


def test_repacking_preserves_all_documents_and_reduces_padding():
    batches = _batches([24, 2, 23, 3, 22, 4, 21, 5], cap=24)
    repacked, stats = regroup_dual_resolution_document_batches(
        batches, batch_size=2, context_length=24, eos_token_id=EOS)
    assert _row_signatures(batches) == _row_signatures(repacked)
    assert len(repacked) == len(batches)
    assert stats["original_rows"] == 2 * (24 + 23 + 22 + 21)
    assert stats["bucketed_rows"] == 2 * (24 + 22 + 5 + 3)
    assert stats["padding_saved_rows"] == stats["original_rows"] - stats["bucketed_rows"]
    assert stats["genuine_rows"] == sum([24, 2, 23, 3, 22, 4, 21, 5])
    for _, x, _, md in repacked:
        assert x.shape[1] == max(md["target_valid_lengths"])
        np.testing.assert_array_equal(
            dual_resolution_loss_mask(x, EOS, md["target_loss_mask"]),
            md["target_loss_mask"])
        for row, length in enumerate(md["target_valid_lengths"]):
            assert np.all(x[row, length:] == EOS)


def test_source_strata_no_cross_source_mixing():
    batches = (_batches([24, 2, 23, 3], source="github")
               + _batches([20, 4, 19, 5], source="finepdfs"))
    result, stats = regroup_dual_resolution_document_batches(
        batches, batch_size=2, context_length=24, eos_token_id=EOS)
    assert _row_signatures(batches) == _row_signatures(result)
    assert stats["strata"] == 2
    assert [item[0] for item in result] == ["github", "github", "finepdfs", "finepdfs"]


def test_batch_three_preserves_all_sampled_targets():
    batches = _batches([24, 1, 13, 23, 2, 14, 22, 3, 15], batch=3, cap=24)
    result, stats = regroup_dual_resolution_document_batches(
        batches, batch_size=3, context_length=24, eos_token_id=EOS)
    assert len(result) == 3
    assert _row_signatures(result) == _row_signatures(batches)
    assert stats["bucketed_rows"] < stats["original_rows"]


def test_geometric_padding_and_unscored_internal_eos():
    batches = _batches([13, 2, 11, 3], cap=24, shape_bucket="geometric")
    batches[0][1][0, 6] = EOS
    batches[0][3]["target_loss_mask"][0, 6] = 0
    result, _ = regroup_dual_resolution_document_batches(
        batches, batch_size=2, context_length=24, eos_token_id=EOS)
    assert _row_signatures(batches) == _row_signatures(result)
    for _, x, _, md in result:
        loss = dual_resolution_loss_mask(x, EOS, md["target_loss_mask"])
        np.testing.assert_array_equal(loss, md["target_loss_mask"])
        dual = build_dual_resolution_metadata(x, EOS, block_size=4)
        ends = dual["block_indices"][:, :, -1]
        if dual["block_valid"].size:
            dual["block_valid"] &= ends < md["target_valid_lengths"][:, None]
            for row, length in enumerate(md["target_valid_lengths"]):
                assert np.all(ends[row, dual["block_valid"][row]] < length)


def test_rejects_mask_that_scores_padding():
    batches = _batches([3, 7])
    batches[0][3]["target_loss_mask"][0, 5] = 1
    with pytest.raises(ValueError, match="padding"):
        regroup_dual_resolution_document_batches(
            batches, batch_size=2, context_length=24, eos_token_id=EOS)


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU optimizer parity only")
def test_real_trainer_optimizer_parity_and_executed_token_accounting(monkeypatch, tmp_path):
    from test_dual_resolution_document_sampling_0102e import _trainer
    # The two trainers see identical RNG/shard draws. Grouping changes the
    # microbatch order but not the set of training targets or their weights.
    docs = [list(range(1, 23)), [30, 31], list(range(40, 58)), [60, 61, 62],
            list(range(65, 75))]
    (tmp_path / "original").mkdir()
    (tmp_path / "bucketed").mkdir()
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_ACCUM_LENGTH_BUCKETS", "0")
    a = _trainer(tmp_path / "original", documents=docs, cap=24, batch=2)
    b = _trainer(tmp_path / "bucketed", documents=docs, cap=24, batch=2)
    b.dual_accum_length_buckets = True
    for trainer in (a, b):
        trainer.grad_accum_steps = 4
        trainer.gradient_accumulation_weighting = "supervised_tokens"
    loss_a, norm_a = a.train_step()
    loss_b, norm_b = b.train_step()
    assert np.isfinite(loss_a) and np.isfinite(loss_b)
    assert abs(loss_a - loss_b) < 2e-4
    assert abs(float(norm_a) - float(norm_b)) < 2e-3
    assert a._last_step_supervised_tokens == b._last_step_supervised_tokens
    assert b._last_step_executed_tokens <= a._last_step_executed_tokens
    assert a.tokens_processed == a._last_step_executed_tokens
    assert b.tokens_processed == b._last_step_executed_tokens
    assert b._last_step_dual_bucket["padding_saved_rows"] == (
        a._last_step_executed_tokens - b._last_step_executed_tokens)
    assert a.train_source_supervised_tokens == b.train_source_supervised_tokens
    assert a._last_step_source_metrics["local"]["supervised_tokens"] == (
        b._last_step_source_metrics["local"]["supervised_tokens"]
    )
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        np.testing.assert_allclose(np.asarray(pa.data), np.asarray(pb.data), atol=2e-4, rtol=2e-4)
