"""Native owner/peer logging integration on temporary shards, without GPUs.

Run only once the production logging integration is ready, with
MINI_LLM_BACKEND=numpy. The imported fixture launches the real native peer.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from uuid import UUID

import numpy as np
import pytest

from test_heterogeneous_production import (
    native,  # noqa: F401 -- imported pytest fixture
    _assert_tree_equal,
    _session,
    _trainer,
    _weights,
)


@pytest.fixture(autouse=True)
def log_environment(monkeypatch):
    monkeypatch.setenv("MINI_LLM_BACKEND", "numpy")
    monkeypatch.setenv("NATIVE_GPU_LOG_LAYERS", "1")
    monkeypatch.setenv("NATIVE_GPU_LOG_MAX_MB", "64")
    monkeypatch.setenv("NATIVE_GPU_LOG_BACKUPS", "0")


def _records(path):
    lines = path.read_text().splitlines()
    assert lines, f"empty diagnostic log: {path}"
    return [json.loads(line) for line in lines]


def _logs(directory):
    paths = sorted(directory.glob("*.jsonl"))
    assert len(paths) == 2, paths
    result = {}
    run_ids = set()
    pids = set()
    for path in paths:
        role, run_id, suffix = path.name.split(".")
        assert suffix == "jsonl"
        UUID(run_id)
        assert role in {"owner-numpy", "peer-numpy"}
        assert role not in result
        run_ids.add(run_id)
        manifest = json.loads(
            path.with_name(f"{role}.{run_id}.manifest.json").read_text()
        )
        assert manifest["role"] == role
        assert manifest["run_id"] == run_id
        assert manifest["runtime"]["backend"] == "numpy"
        assert "device_properties" not in manifest["runtime"]
        assert "is_hip" not in manifest["runtime"]
        pids.add(manifest["runtime"]["pid"])
        records = _records(path)
        for record in records:
            assert record["role"] == role
            assert record["run_id"] == run_id
            assert record["pid"] == manifest["runtime"]["pid"]
            assert record["phase"] in {"train", "warmup"}
            assert "memory_pool" not in record
        result[role] = records
    assert len(run_ids) == 1
    assert len(pids) == 2, "owner and peer must be distinct native processes"
    return result


def _events(records, event, phase="train"):
    return [r for r in records if r["event"] == event and r["phase"] == phase]


@pytest.mark.parametrize("enabled", ["1", None], ids=["explicit", "default"])
def test_native_shared_update_logs_batches_stages_and_checkpoint(
    native, tmp_path, monkeypatch, enabled,
):
    from mini_llm.checkpoint import load_checkpoint

    directory = tmp_path / "diagnostics"
    monkeypatch.setenv("NATIVE_GPU_LOG_DIR", str(directory))
    if enabled is None:
        monkeypatch.delenv("NATIVE_GPU_LOG", raising=False)
    else:
        monkeypatch.setenv("NATIVE_GPU_LOG", enabled)
    original_config = json.loads(native[2].read_text())
    owner = _trainer(native, tmp_path)
    sampled = []
    get_batch = owner.get_train_batch

    def capture_batch(*args, **kwargs):
        batch = get_batch(*args, **kwargs)
        if owner.gpu_diagnostic_log.phase == "train":
            inputs, targets, metadata = batch
            sampled.append((inputs.copy(), targets.copy(), deepcopy(metadata)))
        return batch

    monkeypatch.setattr(owner, "get_train_batch", capture_batch)
    with _session(owner, native) as session:
        loss, norm = owner.train_step()
        assert session.last_report["microbatches"] == 1
        assert session.last_report["supervised_tokens"] == 10
        owner.save()
        names = [p.name for p in owner.model.parameters()]
        _, optimizer, saved = load_checkpoint(owner.checkpoint_dir, param_names=names)
        assert saved["heterogeneous_state"]["contract"] == session.contract
        assert session.contract["peer"] == original_config
        assert optimizer["step"] == owner.step == 1
    assert session.process.poll() == 0
    assert json.loads(native[2].read_text()) == original_config
    assert len(sampled) == 2

    logs = _logs(directory)
    for role, count, shape in (
        ("owner-numpy", 2, [1, 9]), ("peer-numpy", 1, [2, 5]),
    ):
        records = logs[role]
        begins = _events(records, "microbatch_begin")
        batches = _events(records, "batch_ready")
        completed = _events(records, "microbatch_complete")
        assert len(begins) == len(batches) == len(completed) == count
        assert len(_events(records, "microbatch_begin", "warmup")) == 1
        assert len(_events(records, "batch_ready", "warmup")) == 1
        assert len(_events(records, "microbatch_complete", "warmup")) == 1
        assert not _events(records, "error")
        for begin, batch, complete in zip(begins, batches, completed):
            assert isinstance(begin["train_rng_state"], dict)
            assert "bit_generator" in begin["train_rng_state"]
            assert records.index(begin) < records.index(batch) < records.index(complete)
            for record in (begin, batch, complete):
                assert record["weight_version"] == 0
                assert record["update_step"] == 1
                assert record["microbatch"] == begin["microbatch"]
            for key in ("inputs", "targets"):
                assert batch[key]["shape"] == shape
                assert batch[key]["type"].startswith("numpy.")
                digest = batch[key]["sha256"]
                assert len(digest) == 64
                assert len(bytes.fromhex(digest)) == 32
            assert batch["source"]
            assert Path(batch["shard_path"]) == Path(native[1][0])
            assert batch["loss_mask"]["shape"] == shape
            assert batch["loss_mask"]["min"] == batch["loss_mask"]["max"] == 1
            assert batch["valid_tokens"] == int(np.prod(shape))
            for key in ("document_starts", "document_ends"):
                summary = batch["metadata"][key]
                assert summary["type"].startswith("numpy.")
                assert summary["count"] == shape[0]
                assert len(summary["values"]) == shape[0]
                assert summary["truncated"] is False

        for event in ("stage_begin", "stage_end"):
            stages = _events(records, event)
            names = {r["name"] for r in stages}
            assert {"sample", "forward", "loss_forward", "loss_backward", "backward"} <= names
            for name in ("layer.forward", "layer.backward", "attention.forward", "attention.backward"):
                selected = [r for r in stages if r["name"] == name]
                assert {r["layerindex"] for r in selected} == {0, 1}
            for begin in begins:
                selected = [r for r in stages if r["microbatch"] == begin["microbatch"]]
                assert {"forward", "loss_forward", "loss_backward", "backward"} <= {
                    r["name"] for r in selected
                }

    for batch, (inputs, targets, metadata) in zip(
        _events(logs["owner-numpy"], "batch_ready"), sampled,
    ):
        assert batch["inputs"]["sha256"] == hashlib.sha256(inputs.tobytes(order="C")).hexdigest()
        assert batch["targets"]["sha256"] == hashlib.sha256(targets.tobytes(order="C")).hexdigest()
        for key in ("document_starts", "document_ends"):
            assert batch["metadata"][key]["values"] == metadata[key].reshape(-1).tolist()
    updates = _events(logs["owner-numpy"], "update_complete")
    assert len(updates) == 1
    update = updates[0]
    assert update["loss"] == loss
    assert update["grad_norm"] == norm
    assert np.isfinite(loss) and np.isfinite(norm)
    assert update["saved_weight_version"] == update["optimizer_step"] == 1
    assert update["local_microbatches"] == 2
    assert update["supervised_tokens"] == 28


def test_native_logging_preserves_exact_update_optimizer_and_sampler(
    native, tmp_path, monkeypatch,
):
    from mini_llm.backend import BACKEND_NAME, xp
    from mini_llm.heterogeneous_production import sampler_state

    assert BACKEND_NAME == "numpy" and xp is np
    gpu_modules_before = {name for name in sys.modules if name.split(".")[0] == "cupy"}
    original_config = json.loads(native[2].read_text())
    runs = []
    for enabled in ("0", "1"):
        directory = tmp_path / f"logs-{enabled}"
        monkeypatch.setenv("NATIVE_GPU_LOG", enabled)
        monkeypatch.setenv("NATIVE_GPU_LOG_DIR", str(directory))
        owner = _trainer(native, tmp_path / f"owner-{enabled}")
        with _session(owner, native) as session:
            result = owner.train_step()
            state = session.state_dict(owner)
            assert state["contract"]["peer"] == original_config
            assert session.last_report["microbatches"] == 1
            runs.append({
                "result": result,
                "weights": _weights(owner),
                "optimizer": deepcopy(owner.optimizer.checkpoint_state()),
                "owner_sampler": sampler_state(owner),
                "peer_sampler": deepcopy(state["peer_sampler"]),
                "contract": deepcopy(state["contract"]),
                "source_metrics": deepcopy(owner.source_metrics),
            })
        assert session.process.poll() == 0
        if enabled == "0":
            assert not directory.exists() or not list(directory.iterdir())
        else:
            _logs(directory)
    # Deliberately exclude timing-dependent reports and global diagnostic RNGs.
    _assert_tree_equal(runs[1], runs[0])
    assert json.loads(native[2].read_text()) == original_config
    gpu_modules_after = {name for name in sys.modules if name.split(".")[0] == "cupy"}
    assert gpu_modules_after == gpu_modules_before


def test_native_owner_forward_failure_logs_layer_and_traceback(
    native, tmp_path, monkeypatch,
):
    directory = tmp_path / "failure-logs"
    monkeypatch.setenv("NATIVE_GPU_LOG", "1")
    monkeypatch.setenv("NATIVE_GPU_LOG_DIR", str(directory))
    owner = _trainer(native, tmp_path)
    with _session(owner, native) as session:
        block = owner.model.blocks[0]
        assert hasattr(block.forward, "__wrapped__"), "production layer hook must be attached"

        def testfail(*args, **kwargs):
            raise RuntimeError("testfail: injected owner block forward failure")

        # Fail inside the already-wrapped block, rather than replacing its hook.
        # This is installed after warmup/attachment and cannot affect the peer.
        monkeypatch.setattr(block.attention, "forward", testfail)
        with pytest.raises(RuntimeError, match="testfail"):
            owner.train_step()
        assert owner.step == owner.optimizer.step_index == 0
    # An in-flight peer may reject stop; production close bounds its shutdown.
    assert session.process.poll() is not None
    paths = list(directory.glob("owner-numpy.*.jsonl"))
    assert len(paths) == 1
    records = _records(paths[0])
    errors = _events(records, "error")
    layer_errors = [r for r in errors if r.get("name") == "layer.forward"]
    assert layer_errors
    error = layer_errors[0]
    assert error["layerindex"] == 0
    assert error["exception_type"] == "RuntimeError"
    assert "testfail" in error["message"]
    assert "testfail" in error["traceback"]
    assert "Traceback (most recent call last)" in error["traceback"]
    assert error["weight_version"] == 0 and error["update_step"] == 1
    assert not _events(records, "update_complete")
