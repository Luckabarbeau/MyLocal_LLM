"""CPU-only diagnostics contracts; no CuPy import or GPU runtime required."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.diagnostics.gpu_events import GpuEventLog


class FakeGPU:
    shape = (2, 3)
    dtype = "float32"
    strides = (12, 4)

    def __array__(self, *args, **kwargs):
        raise AssertionError("device conversion")

    def get(self, *args, **kwargs):
        raise AssertionError("device read")

    def item(self, *args, **kwargs):
        raise AssertionError("device scalar read")

    def __bool__(self):
        raise AssertionError("device truth test")

    def min(self):
        raise AssertionError("device reduction")

    max = min


@dataclass
class Config:
    n_layers: int = 2
    precision: str = "float32"


class Attention:
    def forward(self, x, **kwargs):
        return x

    def backward(self, dx, **kwargs):
        return dx


class Block:
    def __init__(self):
        self.attention = Attention()
        self.grad = np.array([1., 2.])

    def forward(self, x, **kwargs):
        return self.attention.forward(x, **kwargs)

    def backward(self, dx, **kwargs):
        return self.attention.backward(dx, **kwargs)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ("NATIVE_GPU_LOG_MAX_MB", "NATIVE_GPU_LOG_BACKUPS", "NATIVE_GPU_LOG_LAYERS"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def trainer():
    return SimpleNamespace(
        backend=np, model=SimpleNamespace(config=Config(), dtype=np.dtype("float32"),
                                          blocks=[Block(), Block()]),
        step=7, train_rng=np.random.Generator(np.random.MT19937(123)),
        current_train_shard_idx=0, current_train_source_shard_idx={"code": 0},
        train_source_shards={"code": [Path("a.bin"), Path("b.bin")]},
        train_shard_paths=[Path("local.bin")], val_shard_paths=[],
        source_weights={"code": 1.0}, _last_train_batch_source="code",
        tokens_processed=1234, train_source_batch_counts={"code": 3},
        train_source_supervised_tokens={"code": 100}, grad_accum_steps=4,
        gradient_accumulation_weighting="supervised_tokens", eos_token_id=2,
    )


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_log(trainer, tmp_path, **kwargs):
    return GpuEventLog(trainer, role="owner-cuda", directory=tmp_path, run_id="unique-123", **kwargs)


def test_cpu_batch_hash_rng_context_and_manifest(trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_LLM_DIAGNOSTIC_TEST", "visible")
    monkeypatch.setenv("MINI_LLM_API_TOKEN", "hidden")
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "4096")
    monkeypatch.setenv("UNRELATED_SECRET", "hidden")
    log = make_log(trainer, tmp_path)
    assert isinstance(log.path, Path)
    assert log.path.name == "owner-cuda.unique-123.jsonl"
    assert log.phase == "train"
    rng_state = deepcopy(trainer.train_rng.bit_generator.state)
    inputs = np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2]
    targets = inputs + 1
    original_inputs = inputs.copy()
    original_targets = targets.copy()
    mask = np.array([[0., 1.], [1., .5], [1., 1.]], dtype=np.float32)
    gradients = [block.grad.copy() for block in trainer.model.blocks]
    log.microbatch_begin(trainer, 2)
    log.batch_ready(trainer, inputs, targets,
                    {"document_starts": np.arange(40), "router_gumbel_seed": 99,
                     "executed_sequence_length": 2, "shape_bucket": "small",
                     "blocks": np.ones((3, 2)), "unknown": inputs},
                    mask, 5, .25, True)
    log.close()
    log.close()
    begin, batch = records(log.path)
    for record in (begin, batch):
        assert record["pid"] == os.getpid()
        assert isinstance(record["time_ns"], int) and record["time_ns"] > 0
        assert isinstance(record["monotonic_ns"], int) and record["monotonic_ns"] > 0
    assert batch["monotonic_ns"] >= begin["monotonic_ns"]
    assert begin["weight_version"] == 7 and begin["update_step"] == 8
    assert begin["microbatch"] == 2 and begin["phase"] == "train"
    assert begin["current_train_source_shard_idx"] == {"code": 0}
    assert begin["tokens_processed"] == 1234
    assert begin["train_rng_state"]["state"]["key"] == rng_state["state"]["key"].tolist()
    np.testing.assert_array_equal(trainer.train_rng.bit_generator.state["state"]["key"], rng_state["state"]["key"])
    assert trainer.train_rng.bit_generator.state["state"]["pos"] == rng_state["state"]["pos"]
    assert batch["source"] == "code" and batch["shard_path"] == "b.bin"
    assert batch["inputs"]["sha256"] == hashlib.sha256(inputs.tobytes()).hexdigest()
    assert batch["targets"]["sha256"] == hashlib.sha256(targets.tobytes()).hexdigest()
    assert batch["inputs"]["strides"] == list(inputs.strides)
    assert batch["inputs"]["shape"] == [3, 2]
    assert batch["inputs"]["min"] == 0 and batch["inputs"]["max"] == 10
    assert "values" not in batch["inputs"]
    assert batch["loss_mask"]["min"] == 0 and batch["loss_mask"]["max"] == 1
    assert batch["metadata"]["document_starts"]["values"] == list(range(32))
    assert batch["metadata"]["document_starts"]["truncated"]
    assert "unknown" not in batch["metadata"]
    assert batch["block_metadata"]["blocks"]["count"] == 6
    np.testing.assert_array_equal(inputs, original_inputs)
    np.testing.assert_array_equal(targets, original_targets)
    for block, grad in zip(trainer.model.blocks, gradients):
        np.testing.assert_array_equal(block.grad, grad)
    assert trainer.tokens_processed == 1234 and trainer.current_train_source_shard_idx == {"code": 0}
    manifest = json.loads(log.manifest_path.read_text())
    assert manifest["model_config"] == {"n_layers": 2, "precision": "float32"}
    assert manifest["training"]["grad_accum_steps"] == 4
    assert manifest["training"]["eos_token_id"] == 2
    assert manifest["training"]["train_source_shards"] == {"code": ["a.bin", "b.bin"]}
    assert manifest["runtime"]["backend"] == "numpy"
    assert manifest["runtime"]["pid"] == os.getpid()
    assert manifest["versions"]["numpy"] == np.__version__
    assert manifest["environment"]["MINI_LLM_DIAGNOSTIC_TEST"] == "visible"
    assert manifest["environment"]["MINI_LLM_LM_HEAD_CHUNK_TOKENS"] == "4096"
    assert "MINI_LLM_API_TOKEN" not in manifest["environment"]
    assert "hidden" not in json.dumps(manifest)


def test_gpu_descriptors_only_and_hooks_restored(trainer, tmp_path):
    tensor = FakeGPU()
    log = make_log(trainer, tmp_path)
    block = trainer.model.blocks[0]
    original = block.forward
    log.install_model_hooks(trainer.model)
    log.install_model_hooks(trainer.model)
    assert block.forward.__wrapped__ == original
    log.microbatch_begin(trainer, 0)
    log.batch_ready(trainer, tensor, tensor,
                    {"document_starts": tensor, "block_data": tensor,
                     "blocks": [tensor], "shape_bucket": tensor}, tensor, tensor, tensor, False)
    assert block.forward(tensor, metadata=tensor) is tensor
    assert block.backward(tensor) is tensor
    log.emit("arbitrary", tensor=tensor, cpu=np.arange(10), scalar=np.int64(3))
    log.close()
    assert block.forward == original
    assert "forward" not in vars(block) and "backward" not in vars(block.attention)
    data = records(log.path)
    batch = next(record for record in data if record["event"] == "batch_ready")
    assert batch["inputs"]["shape"] == [2, 3]
    assert "sha256" not in batch["inputs"] and "min" not in batch["loss_mask"]
    stages = [record for record in data if record["event"] == "stage_begin"]
    assert [record["name"] for record in stages] == [
        "layer.forward", "attention.forward", "layer.backward", "attention.backward"]
    assert all(record["layerindex"] == 0 for record in stages)
    assert "values" not in data[-1]["cpu"]


def test_stage_error_context_and_reraise(trainer, tmp_path):
    log = make_log(trainer, tmp_path)
    log.phase = "warmup"
    log.microbatch_begin(trainer, 3)
    error = RuntimeError("kernel launch failed")
    with pytest.raises(RuntimeError) as caught:
        with log.stage("outer", label="test"):
            with log.stage("inner", layerindex=1):
                raise error
    assert caught.value is error
    log.emit("after")
    log.close()
    data = records(log.path)
    errors = [record for record in data if record["event"] == "error"]
    assert len(errors) == 2
    assert "RuntimeError: kernel launch failed" in errors[0]["traceback"]
    assert errors[0]["stages"] == ["outer", "inner"]
    assert errors[0]["phase"] == "warmup" and errors[0]["microbatch"] == 3
    ends = [record for record in data if record["event"] == "stage_end"]
    assert len(ends) == 2
    assert all(record["host_elapsed_s"] >= 0 for record in ends)
    assert all(record["semantics"] == "host_return_not_gpu_completion" for record in ends)
    assert data[-1]["stages"] == []


def test_hook_exception_and_instance_method_restoration(trainer, tmp_path):
    block = trainer.model.blocks[1]
    def fail(x):
        raise ValueError("backward failure")
    block.backward = fail
    log = make_log(trainer, tmp_path)
    log.install_model_hooks(trainer.model)
    with pytest.raises(ValueError, match="backward failure"):
        block.backward(FakeGPU())
    log.close()
    assert block.backward is fail
    error = next(record for record in records(log.path) if record["event"] == "error")
    assert error["name"] == "layer.backward" and error["layerindex"] == 1


@pytest.mark.parametrize("backups", [0, 2])
def test_rotation_record_atomic_and_context(trainer, tmp_path, monkeypatch, backups):
    monkeypatch.setenv("NATIVE_GPU_LOG_MAX_MB", "1")
    monkeypatch.setenv("NATIVE_GPU_LOG_BACKUPS", str(backups))
    log = make_log(trainer, tmp_path)
    manifest = log.manifest_path.read_bytes()
    log.microbatch_begin(trainer, 6)
    for index in range(6):
        log.emit("large", index=index, payload="x" * (1024 * 1024 + 1))
    with log.stage("last"):
        log.emit("latest")
    log.close()
    paths = list(tmp_path.glob("*.jsonl*"))
    assert len(paths) == backups + 1
    assert log.manifest_path.read_bytes() == manifest
    assert records(log.path)[-1]["name"] == "last"
    assert records(log.path)[-1]["microbatch"] == 6
    for path in paths:
        data = records(path)
        assert data and all(record["weight_version"] == 7 for record in data)
        assert path.stat().st_size < 1024 * 1024 + 4096


@pytest.mark.parametrize("key,value", [
    ("NATIVE_GPU_LOG_MAX_MB", "0"), ("NATIVE_GPU_LOG_MAX_MB", "1.5"),
    ("NATIVE_GPU_LOG_MAX_MB", "-1"), ("NATIVE_GPU_LOG_MAX_MB", " 1"),
    ("NATIVE_GPU_LOG_BACKUPS", "-1"), ("NATIVE_GPU_LOG_BACKUPS", "many"),
    ("NATIVE_GPU_LOG_LAYERS", "2"),
])
def test_strict_environment(trainer, tmp_path, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        make_log(trainer, tmp_path)
    assert not list(tmp_path.iterdir())


def test_disabled_hooks_still_log_stages(trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVE_GPU_LOG_LAYERS", "0")
    log = make_log(trainer, tmp_path)
    log.install_model_hooks(trainer.model)
    assert "forward" not in vars(trainer.model.blocks[0])
    with log.stage("batch"):
        pass
    log.close()
    assert [record["event"] for record in records(log.path)] == ["stage_begin", "stage_end"]


def test_global_shard_and_cpu_empty_values(trainer, tmp_path):
    trainer.train_source_shards = None
    trainer.train_shard_paths = [Path("first"), Path("second")]
    trainer.current_train_shard_idx = 1
    log = make_log(trainer, tmp_path)
    empty = np.empty((0,), dtype=np.int32)
    log.batch_ready(trainer, empty, empty, None, None, np.int64(0), np.float32(1), False)
    log.close()
    data = records(log.path)[0]
    assert data["shard_path"] == "first"
    assert data["inputs"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert "min" not in data["inputs"]


def test_exclusive_files_and_fail_loud(trainer, tmp_path):
    log = make_log(trainer, tmp_path)
    with pytest.raises(FileExistsError):
        make_log(trainer, tmp_path)
    class BrokenWriter:
        def write(self, payload):
            raise OSError("disk full")
        def close(self):
            pass
    log._file.close()
    log._file = BrokenWriter()
    with pytest.raises(OSError, match="disk full"):
        log.emit("must_not_drop")
    log.close()
    log.close()
    with pytest.raises(ValueError, match="closed"):
        log.emit("late")


def test_runtime_queries_only_at_init_and_pool_only_at_batch_start(trainer, tmp_path):
    calls = []
    pool = SimpleNamespace(used_bytes=lambda: calls.append("used") or 4,
                           total_bytes=lambda: calls.append("total") or 8)
    runtime = SimpleNamespace(is_hip=True,
                              getDevice=lambda: calls.append("device") or 0,
                              getDeviceProperties=lambda device: calls.append("properties") or {"name": b"fake"})
    trainer.backend = SimpleNamespace(__name__="cupy", cuda=SimpleNamespace(runtime=runtime),
                                      get_default_memory_pool=lambda: pool)
    log = make_log(trainer, tmp_path)
    assert calls == ["device", "properties"]
    with log.stage("no_gpu_queries"):
        log.emit("device", tensor=FakeGPU())
    assert calls == ["device", "properties"]
    log.microbatch_begin(trainer, 0)
    assert calls == ["device", "properties", "used", "total"]
    log.close()
    assert records(log.path)[-1]["memory_pool"] == {"used_bytes": 4, "total_bytes": 8}
