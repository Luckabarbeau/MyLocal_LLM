"""0074A tests: estimator correctness, observational training and resumption."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.diagnostics.gradient_noise import (
    GradientNoiseMonitor, half_statistics, parse_accum_points, select_parameters,
)
from mini_llm.parameter import Parameter
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.checkpoint import load_checkpoint


def test_identical_and_scaled():
    zero = np.zeros(3, np.float32)
    half = np.array([8., 16., -8.], np.float32)  # sum of 4 identical g
    full = 2 * half
    direct = half_statistics(zero, half, full, 8)
    scaled = half_statistics(zero, half * 128, full * 128, 8, loss_scale=128)
    assert direct["gradient_cosine"] == pytest.approx(1)
    assert direct["estimated_gradient_noise"] == pytest.approx(0)
    assert direct["relative_gradient_uncertainty"] == pytest.approx(0)
    assert scaled["estimated_signal_norm_squared"] == pytest.approx(direct["estimated_signal_norm_squared"])
    assert scaled["relative_gradient_uncertainty"] == pytest.approx(direct["relative_gradient_uncertainty"])


def test_opposing_and_orthogonal():
    zero = np.zeros(2, np.float32)
    a = np.array([4., 0.], np.float32)
    opposite = half_statistics(zero, a, zero, 8)
    assert opposite["gradient_cosine"] == pytest.approx(-1)
    assert opposite["estimated_signal_norm_squared"] < 0
    assert opposite["relative_gradient_uncertainty"] is None
    orthogonal = half_statistics(zero, a, a + np.array([0., 4.], np.float32), 8)
    assert orthogonal["gradient_cosine"] == pytest.approx(0)
    assert orthogonal["estimated_gradient_noise"] == pytest.approx(0.5)


def test_gaussian_estimator_unbiased_in_average():
    rng = np.random.default_rng(74)
    mu = np.array([0.5, 1.0, -0.2], dtype=np.float32)
    std = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    values = []
    for _ in range(1000):
        grads = (mu + rng.normal(size=(8, 3)).astype(np.float32) * std)
        first = grads[:4].sum(axis=0)
        whole = grads.sum(axis=0)
        values.append(half_statistics(np.zeros(3, np.float32), first, whole, 8)["estimated_gradient_noise"])
    target = float(np.sum(std ** 2) / 8)
    assert np.mean(values) == pytest.approx(target, rel=0.12)


def test_missing_inactive_and_invalid_points(tmp_path):
    with pytest.raises(ValueError):
        parse_accum_points("8,16,32", 16)
    with pytest.raises(ValueError):
        parse_accum_points("4,4", 8)
    p = Parameter(np.zeros((2, 2), np.float32), name="blocks.0.moe.router.W_router")
    monitor = GradientNoiseMonitor([p], accum_steps=8, accum_points="4,8",
                                   categories="moe_router", interval=1,
                                   output=str(tmp_path / "noise.csv"))
    assert monitor.begin(0)
    for i in range(8):
        if i >= 4:
            p.grad += 1.0
        monitor.count_tokens(np.zeros((2, 2), np.int64), np.array([[1, 0], [0, 1]]))
        monitor.capture(i + 1)
    rows = monitor.finish()
    assert len(rows) == 2
    assert rows[0]["nonzero_gradient_fraction_a"] == 0
    assert rows[0]["nonzero_gradient_fraction_b"] == 0
    assert rows[1]["nonzero_gradient_fraction_a"] == 0
    assert rows[1]["nonzero_gradient_fraction_b"] == 1
    assert rows[1]["valid_supervised_tokens"] == 16
    assert not (tmp_path / "noise.csv").exists()  # no premature write
    monitor.commit(rows)
    with (tmp_path / "noise.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    p.grad[...] = 0
    assert monitor.begin(1)
    monitor.capture(4)  # aborted/invalid cycle: never commit
    assert len(list(csv.DictReader((tmp_path / "noise.csv").open()))) == 2


def _make_trainer(root, monitored, *, dtype="float32", accum_steps=8, interval=1):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=42, dtype=dtype)
    data = root / "train_shard_00000.bin"
    if not data.exists():
        np.random.default_rng(999).integers(0, 256, size=2048, dtype=np.uint16).tofile(data)
    return ExtendedTrainer(
        model=model, train_shard_paths=[str(data)], val_shard_paths=[str(data)],
        batch_size=1, seq_length=16, grad_accum_steps=accum_steps,
        total_steps=20, warmup_steps=2, peak_lr=1e-3, rng_seed=123,
        checkpoint_dir=str(root / ('yes' if monitored else 'no')),
        gradient_noise_monitor=monitored, gradient_noise_interval=interval,
        gradient_noise_accum_points="4,8" if accum_steps == 8 else "2,4",
        gradient_noise_output=str(root / "noise.csv"),
    )


def test_trainer_gradient_noise_exact_equivalence(tmp_path):
    plain = _make_trainer(tmp_path, False)
    monitor = _make_trainer(tmp_path, True)
    for _ in range(2):
        loss_plain, grad_plain = plain.train_step()
        loss_monitor, grad_monitor = monitor.train_step()
        assert loss_plain == pytest.approx(loss_monitor, rel=0, abs=1e-6)
        assert grad_plain == pytest.approx(grad_monitor, rel=0, abs=1e-6)
    assert plain.step == monitor.step == 2
    assert plain.tokens_processed == monitor.tokens_processed
    assert plain.train_rng.bit_generator.state == monitor.train_rng.bit_generator.state
    assert plain.optimizer.step_index == monitor.optimizer.step_index
    for pa, pb in zip(plain.model.parameters(), monitor.model.parameters()):
        np.testing.assert_array_equal(np.asarray(pa.data), np.asarray(pb.data))
        np.testing.assert_array_equal(np.asarray(pa.grad), np.asarray(pb.grad))
    for ma, mb in zip(plain.optimizer.m, monitor.optimizer.m):
        np.testing.assert_array_equal(np.asarray(ma), np.asarray(mb))
    for va, vb in zip(plain.optimizer.v, monitor.optimizer.v):
        np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))
    with (tmp_path / "noise.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and {int(r["step"]) for r in rows} == {1, 2}
    assert {int(r["accumulation_length"]) for r in rows} == {4, 8}
    assert all(int(r["valid_supervised_tokens"]) == 16 * int(r["accumulation_length"]) for r in rows)
    plain.save()
    monitor.save()
    p_names = [p.name for p in plain.model.parameters()]
    loaded_a, optim_a, state_a = load_checkpoint(tmp_path / 'no', param_names=p_names)
    loaded_b, optim_b, state_b = load_checkpoint(tmp_path / 'yes', param_names=p_names)
    assert state_a.keys() == state_b.keys()
    for name in p_names:
        np.testing.assert_array_equal(np.asarray(loaded_a[name]), np.asarray(loaded_b[name]))
    assert optim_a['step'] == optim_b['step']


def test_interval_and_disabled_no_allocations(tmp_path):
    p = Parameter(np.zeros((4, 4), np.float32), name="blocks.0.attention.Wq")
    monitor = GradientNoiseMonitor([p], accum_steps=8, accum_points="4,8",
                                   categories="attention", interval=100,
                                   output=str(tmp_path / "noise.csv"))
    assert not monitor.begin(0)
    assert not monitor.snapshots
    assert monitor.begin(99)
    assert not monitor.snapshots


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy GPU backend")
def test_gpu_device_snapshot_smoke(tmp_path):
    p = Parameter(xp.zeros((8, 8), dtype=xp.float32), name="blocks.0.attention.Wq")
    monitor = GradientNoiseMonitor([p], accum_steps=8, accum_points="4,8",
                                   categories="attention", interval=1,
                                   output=str(tmp_path / "device.csv"))
    assert monitor.begin(0)
    for i in range(8):
        p.grad += 1
        monitor.capture(i + 1)
    rows = monitor.finish()
    assert rows[1]['estimated_gradient_noise'] == pytest.approx(0)


def test_bf16_gradient_buffer_is_fp32():
    ml = pytest.importorskip("ml_dtypes")
    p = Parameter(xp.zeros((4, 4), dtype=ml.bfloat16),
                  name="blocks.0.attention.Wq")
    assert p.grad.dtype == xp.dtype("float32")
    selected = select_parameters([p], categories="attention")
    assert len(selected) == 1


def test_requested_endpoint_below_actual_accumulation(tmp_path):
    p = Parameter(np.zeros((2, 3), np.float32), name="blocks.0.attention.Wq")
    noise = GradientNoiseMonitor([p], accum_steps=12, accum_points="4,8",
                                 categories="attention", interval=1,
                                 output=str(tmp_path / "below.csv"))
    assert noise.begin(0)
    for i in range(12):
        p.grad += (i + 1)
        noise.capture(i + 1)
    rows = noise.finish()
    assert len(rows) == 2
    # g_i=(i+1), so G4=10, G8=36, G12=78, and n=8 must use G8.
    assert rows[1]["gradient_norm_full"] == pytest.approx(np.sqrt(6) * (36/8))
