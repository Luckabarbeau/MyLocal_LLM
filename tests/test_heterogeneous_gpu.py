"""Opt-in test of both physical GPUs; never load production weights or data."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MINI_LLM_TEST_HETEROGENEOUS_GPU") != "1",
    reason="set MINI_LLM_TEST_HETEROGENEOUS_GPU=1 only when both GPUs are available",
)
def test_native_cuda_hip_two_shared_updates(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/heterogeneous_smoke.json").read_text())
    assert config["smoke"] is True
    assert config["dtype"] == "bfloat16"
    assert "init_from" not in config and "checkpoint_dir" not in config
    assert {worker["expected_runtime"] for worker in config["workers"]} == {"cuda", "hip"}
    config["log_dir"] = str(tmp_path / "worker-logs")
    path = tmp_path / "gpu-smoke.json"
    path.write_text(json.dumps(config))
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, str(root / "train_heterogeneous.py"), "--config", str(path),
         "--steps", "2", "--work-dir", str(tmp_path)],
        cwd=root, env=env, capture_output=True, text=True, timeout=360,
    )
    logs = "\n".join(
        f"{log.name}:\n{log.read_text()}" for log in sorted((tmp_path / "worker-logs").glob("*.log"))
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr + logs
    results = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert [result["step"] for result in results] == [1, 2]
    anchor = next(worker for worker in config["workers"] if worker["name"] == config["anchor"])
    for result in results:
        assert math.isfinite(result["loss"]) and result["loss"] > 0
        assert math.isfinite(result["grad_norm"]) and result["grad_norm"] > 0
        counts = result["microbatches"]
        assert counts[anchor["name"]] == result["anchor_target"]
        assert anchor["target_microbatches"] <= result["anchor_target"] <= config.get("max_anchor_microbatches", anchor["target_microbatches"])
        assert result["next_anchor_target"] <= config.get("max_anchor_microbatches", anchor["target_microbatches"])
        assert all(counts[worker["name"]] >= 1 for worker in config["workers"])
        expected_tokens = sum(
            worker["batch_size"] * worker["context_length"] * counts[worker["name"]]
            for worker in config["workers"]
        )
        assert result["supervised_tokens"] == expected_tokens
    assert not list(tmp_path.glob("native-heterogeneous-*"))
