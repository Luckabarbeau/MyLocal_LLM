"""0084E convergence monitoring: mathematical null and real trainer parity."""
from __future__ import annotations

import csv
import math
import sys

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.diagnostics.gradient_noise import select_parameters
from mini_llm.diagnostics.gradient_convergence import (
    GradientConvergenceMonitor, convergence_statistics,
    parse_convergence_points, FIELDS,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.parameter import Parameter
from mini_llm.train_extended import ExtendedTrainer
from train_model import parse_args


def _param(name="blocks.0.attention.Wq", cols=4):
    return Parameter(np.zeros((1, cols), dtype=np.float32), name=name)


def _run(values, weights, tmp_path, *, loss_scale=1.0, points="1,2,3,4"):
    p = _param(cols=len(values[0]))
    mon = GradientConvergenceMonitor(
        [p], accum_steps=len(values), accum_points=points,
        categories="attention", interval=1, loss_scale=loss_scale,
        output=str(tmp_path / "norm.csv"),
    )
    assert mon.begin(0)
    for index, (v, weight) in enumerate(zip(values, weights), start=1):
        p.grad += np.asarray(v, dtype=np.float32)[None, :] * weight * loss_scale
        mon.capture(index, microbatch_weight=weight,
                    valid_tokens=round(weight * 100))
    original_grad = p.grad.copy()
    rows = mon.finish()
    np.testing.assert_array_equal(p.grad, original_grad)
    mon.commit(rows)
    with (tmp_path / "norm.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(rows) == len(csv_rows)
    assert tuple(csv_rows[0]) == FIELDS
    return rows


def test_constant_signal_plateaus_and_null_decays(tmp_path):
    g = np.array([0.3, 0.4, 0., 0.])
    rows = _run([g] * 4, [1, 2, 1, 4], tmp_path, loss_scale=128)
    for k, row in enumerate(rows, start=1):
        assert row["gradient_mean_norm"] == pytest.approx(0.5, abs=2e-7)
        assert row["cosine_to_final_gradient"] == pytest.approx(1., abs=1e-7)
        if k > 1:
            assert row["estimated_signal_norm_squared"] == pytest.approx(0.25, abs=1e-7)
            assert row["estimated_noise_variance_of_mean"] == pytest.approx(0, abs=1e-7)
    assert rows[-1]["random_reference_norm"] == pytest.approx(
        0.5 * math.sqrt(1 + 4 + 1 + 16) / 8, abs=1e-7)
    assert rows[-1]["valid_supervised_tokens"] == 800


def test_orthogonal_null_exact_random_reference(tmp_path):
    vals = list(np.eye(4, dtype=np.float32))
    rows = _run(vals, [1.] * 4, tmp_path)
    for k, row in enumerate(rows, start=1):
        assert row["gradient_mean_norm"] == pytest.approx(1 / math.sqrt(k))
        assert row["random_reference_norm"] == pytest.approx(1 / math.sqrt(k))
        assert row["actual_to_random_norm_ratio"] == pytest.approx(1.)
        assert row["cosine_to_final_gradient"] == pytest.approx(math.sqrt(k / 4))
        assert row["random_cosine_to_final_reference"] == pytest.approx(math.sqrt(k / 4))
        if k >= 2:
            assert row["estimated_signal_norm_squared"] == pytest.approx(0.)
            assert row["signal_positive"] is False


def test_negative_signal_estimate_not_clipped(tmp_path):
    rows = _run([[1, 0], [-1, 0]], [1, 1], tmp_path, points="1,2")
    assert rows[-1]["estimated_signal_norm_squared"] == pytest.approx(-1.)
    assert rows[-1]["estimated_signal_norm"] is None
    assert rows[-1]["noise_to_signal_squared"] is None


def test_weighted_variance_and_effective_microbatches():
    # One-dimensional weighted microbatch means 1,3 with weights 1,2.
    # R=7, Q=37, T=19, W=3, sum(w²)=5.
    x = convergence_statistics(49., 37., 19., 3., 5., 1.)
    assert x["gradient_mean_norm"] == pytest.approx(7/3)
    assert x["random_reference_norm"] == pytest.approx(math.sqrt(37)/3)
    assert x["estimated_signal_norm_squared"] == pytest.approx(3.)
    assert x["effective_microbatches"] == pytest.approx(9/5)
    assert x["estimated_noise_variance_of_mean"] == pytest.approx(10/9)


def test_points_validation_and_auto():
    assert parse_convergence_points("auto", 22) == (2, 4, 8, 16, 22)
    assert parse_convergence_points("auto", 1) == (1,)
    assert parse_convergence_points("2,4,8,12,16,20", 22)[-1] == 22
    for text in ("2,2", "5,3", "2,24", "wat"):
        with pytest.raises(ValueError):
            parse_convergence_points(text, 22)


def test_zero_token_microbatch_skipped_without_division(tmp_path):
    vals = [[0., 0.], [1., 0.], [1., 0.]]
    rows = _run(vals, [0., 1., 1.], tmp_path, points="1,2,3")
    assert rows[-1]["gradient_mean_norm"] == pytest.approx(1.)
    assert rows[-1]["estimated_signal_norm_squared"] == pytest.approx(1.)


def test_selection_includes_dense_ffn_across_layers():
    ps = [_param(f"blocks.{i}.dense_ffn.expert.W_gate") for i in (0, 1, 2)]
    ps += [_param(f"blocks.{i}.attention.Wq") for i in (0, 1, 2)]
    selected = select_parameters(ps, categories="attention,ffn")
    assert [name for name, _, _ in selected] == [
        "attention_early", "attention_middle", "attention_final",
        "ffn_early", "ffn_middle", "ffn_final",
    ]


def _trainer(root, enabled):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(),
                                 rng_seed=42, dtype="float32")
    path = root / "train_shard_00000.bin"
    if not path.exists():
        np.random.default_rng(999).integers(0, 256, size=2048,
                                           dtype=np.uint16).tofile(path)
    return ExtendedTrainer(
        model=model, train_shard_paths=[str(path)], val_shard_paths=[str(path)],
        batch_size=1, seq_length=16, grad_accum_steps=4, warmup_steps=2,
        total_steps=20, peak_lr=1e-3, rng_seed=123,
        checkpoint_dir=str(root / ('diagnostic' if enabled else 'baseline')),
        gradient_convergence_monitor=enabled,
        gradient_convergence_interval=1,
        gradient_convergence_points="1,2,3,4",
        gradient_convergence_output=str(root / "convergence.csv"),
        gradient_convergence_groups="attention",
    )


def test_full_trainer_is_bitwise_identical(tmp_path):
    plain, monitored = _trainer(tmp_path, False), _trainer(tmp_path, True)
    for _ in range(2):
        a_loss, a_grad = plain.train_step()
        b_loss, b_grad = monitored.train_step()
        assert a_loss == pytest.approx(b_loss, rel=0, abs=1e-6)
        assert a_grad == pytest.approx(b_grad, rel=0, abs=1e-6)
    assert plain.tokens_processed == monitored.tokens_processed
    assert plain.step == monitored.step == 2
    assert plain.train_rng.bit_generator.state == monitored.train_rng.bit_generator.state
    for a, b in zip(plain.model.parameters(), monitored.model.parameters()):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
        np.testing.assert_array_equal(np.asarray(a.grad), np.asarray(b.grad))
    for a, b in zip(plain.optimizer.m, monitored.optimizer.m):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with (tmp_path / "convergence.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert {int(r["step"]) for r in rows} == {1, 2}
    assert rows[-1]["accumulation_length"] == '4'
    assert int(rows[-1]["valid_supervised_tokens"]) == 64


def test_cli_parse(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_model.py", "--gradient-convergence-monitor",
        "--gradient-convergence-points", "2,4,8,12,16,20,22",
    ])
    args = parse_args()
    assert args.gradient_convergence_monitor
    assert args.gradient_convergence_interval == 100
    assert args.gradient_convergence_points.endswith(",22")


def test_dense_hybrid_muon_training_exact_parity(tmp_path):
    """Muon's matrix-gradient updates must be unaffected by the monitor."""
    from dataclasses import replace
    config = replace(ModelConfig.tiny_inspection(), n_layers=3,
                     ffn_mode="dense", n_experts=1, top_k=1)
    path = tmp_path / "dense_shard_00000.bin"
    np.random.default_rng(12).integers(0, 256, size=2048,
                                       dtype=np.uint16).tofile(path)

    def build(enabled):
        return ExtendedTrainer(
            model=DecoderLanguageModel(config, rng_seed=17, dtype="float32"),
            train_shard_paths=[str(path)], val_shard_paths=[str(path)],
            batch_size=1, seq_length=16, grad_accum_steps=4,
            total_steps=8, warmup_steps=2, peak_lr=1e-4,
            optimizer_kind="hybrid-muon", rng_seed=7,
            gradient_convergence_monitor=enabled,
            gradient_convergence_interval=1,
            gradient_convergence_points="1,2,3,4",
            gradient_convergence_groups="attention,ffn",
            gradient_convergence_output=str(tmp_path / "dense_curve.csv"),
        )

    plain = build(False)
    observed = build(True)
    for _ in range(2):
        loss_plain, norm_plain = plain.train_step()
        loss_observed, norm_observed = observed.train_step()
        assert loss_plain == pytest.approx(loss_observed, rel=0, abs=1e-6)
        assert norm_plain == pytest.approx(norm_observed, rel=0, abs=1e-6)
    for left, right in zip(plain.model.parameters(), observed.model.parameters()):
        np.testing.assert_array_equal(np.asarray(left.data), np.asarray(right.data))
        np.testing.assert_array_equal(np.asarray(left.grad), np.asarray(right.grad))
    with (tmp_path / "dense_curve.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 4 * 6
    assert {r["parameter_group"] for r in rows} == {
        "attention_early", "attention_middle", "attention_final",
        "ffn_early", "ffn_middle", "ffn_final",
    }
