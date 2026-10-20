"""Contract tests for the single configurable training launcher."""

from pathlib import Path
import os
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_training.sh"


def _run(tmp_path, mode="smoke", *, extra_env=None, extra_args=()):
    mixture = tmp_path / "shards"
    mixture.mkdir(exist_ok=True)
    (mixture / "tokenizer.json").write_text("{}")
    (mixture / "mixture_manifest.json").write_text("{}")
    env = dict(
        os.environ,
        CHECKPOINT_DIR=str(tmp_path / "checkpoint"),
        MIXED_SHARD_ROOT=str(mixture),
        DRY_RUN="1",
        MINI_LLM_BACKEND="numpy",
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT), mode, "--", *extra_args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_smoke_builds_default_command_without_execution(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_model.py" in result.stdout
    assert "--model ensemble-deep-512-dense-memory-64k" in result.stdout
    assert "--objective ar" in result.stdout
    assert "--max-run-steps 2" in result.stdout
    assert "--resume-from" not in result.stdout
    assert not (tmp_path / "checkpoint").exists()


def test_continue_requires_checkpoint_state(tmp_path):
    result = _run(tmp_path, "continue")
    assert result.returncode != 0
    assert "training_state.json is missing" in result.stderr


def test_continue_and_extra_arguments(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_state.json").write_text("{}")
    result = _run(
        tmp_path,
        "continue",
        extra_env={"MODEL": "wide24-500m-memory-64k", "OBJECTIVE": "block-proposal"},
        extra_args=("--block-proposal-k", "4"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--model wide24-500m-memory-64k" in result.stdout
    assert "--objective block-proposal" in result.stdout
    assert "--resume-from" in result.stdout
    assert "--block-proposal-k 4" in result.stdout


@pytest.mark.parametrize("mode", ["", "unknown"])
def test_rejects_invalid_mode(tmp_path, mode):
    result = _run(tmp_path, mode)
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_requires_explicit_checkpoint_directory(tmp_path):
    mixture = tmp_path / "shards"
    mixture.mkdir()
    (mixture / "tokenizer.json").write_text("{}")
    (mixture / "mixture_manifest.json").write_text("{}")
    env = dict(os.environ, MIXED_SHARD_ROOT=str(mixture), DRY_RUN="1")
    env.pop("CHECKPOINT_DIR", None)
    result = subprocess.run(
        ["bash", str(SCRIPT), "smoke"], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "CHECKPOINT_DIR is required" in result.stderr
