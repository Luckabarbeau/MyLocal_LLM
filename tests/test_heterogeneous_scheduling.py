"""CPU contracts for autonomous accumulation, warmup, and scheduling telemetry."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import train_heterogeneous as heterogeneous


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    return {
        "smoke": True,
        "dtype": "float32",
        "seed": 42,
        "timeout_s": 120,
        "log_dir": str(tmp_path / "logs"),
        "anchor": "anchor",
        # run() overlays this on the inherited environment for each worker.
        "env": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "workers": [
            {"name": "anchor", "python": sys.executable, "backend": "numpy",
             "batch_size": 1, "context_length": 9, "target_microbatches": 1,
             "max_microbatches": 1, "lm_head_chunk_tokens": 4},
            {"name": "other", "python": sys.executable, "backend": "numpy",
             "batch_size": 2, "context_length": 5, "max_microbatches": 1,
             "lm_head_chunk_tokens": 3},
        ],
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "base,maximum,microbatch_s,barrier_s,fraction,expected",
    [
        (2, 20, 1.0, 4.0, None, 2),
        (2, 20, 0.0, 4.0, 0.5, 2),
        (2, 20, -1.0, 4.0, 0.5, 2),
        (2, 20, 1.0, 0.0, 0.5, 2),
        (2, 20, 1.0, -4.0, 0.5, 2),
        (2, 20, 0.5, 0.5, 0.5, 2),  # Raw target 1 clamps to base.
        (2, 20, 1.0, 4.0, 0.5, 4),
        (2, 20, 2.0, 3.0, 0.75, 5),  # ceil(4.5), not round/floor.
        (2, 20, 1.0, 40.0, 0.5, 20),
        (2, 2, 1.0, 40.0, 0.5, 2),
    ],
)
def test_next_anchor_target(base, maximum, microbatch_s, barrier_s, fraction, expected):
    target = heterogeneous.next_anchor_target(
        base, maximum, microbatch_s, barrier_s, fraction,
    )
    assert target == expected
    assert isinstance(target, int)


@pytest.mark.unit
@pytest.mark.parametrize("warmup", [None, 0, 1, 2])
def test_validate_config_accepts_default_and_explicit_warmup(tmp_path, warmup):
    config = _config(tmp_path)
    if warmup is not None:
        config["warmup_microbatches"] = warmup
    heterogeneous.validate_config(config)


@pytest.mark.unit
@pytest.mark.parametrize("warmup", [True, False, -1, 1.5, "2"])
def test_validate_config_rejects_invalid_warmup(tmp_path, warmup):
    config = _config(tmp_path)
    config["warmup_microbatches"] = warmup
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
@pytest.mark.parametrize("worker_index", [0, 1])
@pytest.mark.parametrize("cap", [True, False, 0, -1, 1.5, "1"])
def test_validate_config_rejects_invalid_worker_caps(tmp_path, worker_index, cap):
    config = _config(tmp_path)
    config["workers"][worker_index]["max_microbatches"] = cap
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
def test_validate_config_accepts_uncapped_workers(tmp_path):
    config = _config(tmp_path)
    for worker in config["workers"]:
        del worker["max_microbatches"]
    heterogeneous.validate_config(config)


@pytest.mark.unit
def test_validate_config_rejects_anchor_cap_below_target(tmp_path):
    config = _config(tmp_path)
    config["workers"][0]["target_microbatches"] = 2
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_validate_config_rejects_invalid_compute_fraction(tmp_path, fraction):
    config = _config(tmp_path)
    config.update(target_compute_fraction=fraction, max_anchor_microbatches=20)
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
def test_validate_config_requires_maximum_for_compute_fraction(tmp_path):
    config = _config(tmp_path)
    config["target_compute_fraction"] = 0.5
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
@pytest.mark.parametrize("maximum", [True, False, 0, -1, 1.5, "2"])
def test_validate_config_rejects_invalid_anchor_maximum(tmp_path, maximum):
    config = _config(tmp_path)
    config.update(target_compute_fraction=0.5, max_anchor_microbatches=maximum)
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
def test_validate_config_rejects_maximum_below_base_target(tmp_path):
    config = _config(tmp_path)
    config["workers"][0].update(target_microbatches=2, max_microbatches=2)
    config.update(target_compute_fraction=0.5, max_anchor_microbatches=1)
    with pytest.raises(ValueError):
        heterogeneous.validate_config(config)


@pytest.mark.unit
@pytest.mark.parametrize("maximum", [1, 20])
def test_validate_config_accepts_adaptive_target(tmp_path, maximum):
    config = _config(tmp_path)
    # Leave the anchor uncapped so adaptation can raise its target.
    del config["workers"][0]["max_microbatches"]
    config.update(target_compute_fraction=0.5, max_anchor_microbatches=maximum)
    heterogeneous.validate_config(config)


def _run_one_update(directory, warmup):
    directory.mkdir()
    config = _config(directory)
    checkpoint = directory / "checkpoint"
    config.update(warmup_microbatches=warmup, checkpoint_dir=str(checkpoint))
    config_path = directory / "run.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env.update(config["env"])
    completed = subprocess.run(
        [sys.executable, str(ROOT / "train_heterogeneous.py"),
         "--config", str(config_path), "--steps", "1", "--work-dir", str(directory)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    logs = "\n".join(
        f"{path.name}:\n{path.read_text()}"
        for path in sorted((directory / "logs").glob("*.log"))
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr + logs
    results = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert [result["step"] for result in results] == [1], results
    state = json.loads((checkpoint / "training_state.json").read_text())
    assert state["step"] == 1
    assert state["heterogeneous_results"] == results
    assert len(list((directory / "logs").glob("heterogeneous-worker-*.log"))) == 2
    assert not list(directory.glob("native-heterogeneous-*"))
    return results[0], checkpoint


def _assert_nonnegative_finite(value):
    assert math.isfinite(value)
    assert value >= 0


def _assert_training_telemetry(result):
    assert math.isfinite(result["loss"]) and result["loss"] > 0
    assert math.isfinite(result["grad_norm"]) and result["grad_norm"] > 0
    assert result["anchor_target"] == result["next_anchor_target"] == 1
    assert result["microbatches"] == {"anchor": 1, "other": 1}
    assert result["supervised_tokens"] == 19
    assert set(result["workers"]) == {"anchor", "other"}
    for name, tokens in (("anchor", 9), ("other", 10)):
        metrics = result["workers"][name]
        assert metrics["microbatches"] == result["microbatches"][name]
        assert metrics["supervised_tokens"] == tokens
        for key in ("compute_s", "export_s", "wait_for_stop_s"):
            _assert_nonnegative_finite(metrics[key])
    assert sum(worker["supervised_tokens"] for worker in result["workers"].values()) == 19
    for key in ("accumulation", "reduce", "update", "sync"):
        _assert_nonnegative_finite(result["timing_s"][key])
    _assert_nonnegative_finite(result["time_s"])


@pytest.mark.integration
def test_warmup_preserves_rng_gradients_weights_and_optimizer_exactly(tmp_path):
    cold, cold_checkpoint = _run_one_update(tmp_path / "cold", warmup=0)
    warm, warm_checkpoint = _run_one_update(tmp_path / "warm", warmup=2)
    for result in (cold, warm):
        _assert_training_telemetry(result)
    assert set(warm["warmup"]) == {"anchor", "other"}
    for metrics in warm["warmup"].values():
        assert metrics["microbatches"] == 2
        _assert_nonnegative_finite(metrics["compute_s"])

    # Equal draws and clean gradients must produce exactly the same first update.
    assert cold["loss"] == warm["loss"]
    assert cold["grad_norm"] == warm["grad_norm"]
    for group in ("model_params", "optimizer_state"):
        cold_arrays = {path.relative_to(cold_checkpoint / group): path
                       for path in (cold_checkpoint / group).rglob("*.npy")}
        warm_arrays = {path.relative_to(warm_checkpoint / group): path
                       for path in (warm_checkpoint / group).rglob("*.npy")}
        assert cold_arrays, group
        assert cold_arrays.keys() == warm_arrays.keys()
        for name, cold_path in cold_arrays.items():
            cold_array = np.load(cold_path, allow_pickle=False)
            warm_array = np.load(warm_arrays[name], allow_pickle=False)
            assert np.isfinite(cold_array).all(), (group, name, "cold")
            assert np.isfinite(warm_array).all(), (group, name, "warm")
            assert cold_array.dtype == warm_array.dtype, (group, name)
            np.testing.assert_array_equal(cold_array, warm_array, err_msg=f"{group}/{name}")

    for checkpoint in (cold_checkpoint, warm_checkpoint):
        optimizer = checkpoint / "optimizer_state"
        assert json.loads((optimizer / "scalars.json").read_text())["step"] == 1
        for group in ("master_weights", "m", "v", "muon_m"):
            assert list((optimizer / group).glob("*.npy")), group
    cold_scalars = {path.relative_to(cold_checkpoint / "optimizer_state"): json.loads(path.read_text())
                    for path in (cold_checkpoint / "optimizer_state").rglob("*.json")}
    warm_scalars = {path.relative_to(warm_checkpoint / "optimizer_state"): json.loads(path.read_text())
                    for path in (warm_checkpoint / "optimizer_state").rglob("*.json")}
    assert cold_scalars == warm_scalars
