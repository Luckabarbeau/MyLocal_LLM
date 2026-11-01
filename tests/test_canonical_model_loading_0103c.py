"""0103C — one native loader for facade, benchmark, CLI and HTTP serving."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.model_loading import load_native_checkpoint, resolve_native_context
from mini_llm.model_sources import ModelCatalog, ModelSpec, load_specs
from mini_llm.serving_models import load_model


def _tiny_checkpoint(path, *, missing=False):
    from mini_llm.checkpoint import save_checkpoint
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer
    cfg = ModelConfig(tokenizer_vocab_size=128, context_length=128,
                      n_layers=1, d_model=32, d_head=8,
                      n_q_heads=4, n_kv_heads=2, n_experts=1, top_k=1,
                      d_ff=64, ffn_mode="dense", dtype="float32")
    path.mkdir()
    (path / "config.json").write_text(json.dumps(asdict(cfg)))
    tokenizer = SimpleBPETokenizer(vocab_size=128)
    tokenizer.train(["hello world!", "model loading tests"], vocab_size=128)
    tokenizer.save(str(path / "tokenizer.json"))
    model = DecoderLanguageModel(cfg, rng_seed=172, dtype=cfg.dtype)
    params = model.parameters()
    arrays = {p.name: p.data for p in params[1 if missing else 0:]}
    save_checkpoint(path, arrays, training_state={"step": 7})
    return model


def test_canonical_native_across_public_benchmark_and_service(tmp_path):
    from mini_llm import Model
    from benchmarks.evaluate_generation import load_generator
    checkpoint = tmp_path / "native"
    original = _tiny_checkpoint(checkpoint)
    canonical = load_native_checkpoint(checkpoint, requested_context=40)
    facade = Model.load(checkpoint, backend="numpy", device="cpu")
    generator, info = load_generator(checkpoint, requested_context=40, runtime_profile="reference")
    service = load_model(ModelSpec(id="tiny", path=checkpoint, format="native",
                                   dtype="float32", context_length=40))
    assert (canonical.context_length, facade._context_length, generator.max_context,
            service.context_length) == (40, 128, 40, 40)
    assert info["step"] == 7 and canonical.training_state["step"] == 7
    for expected, observed in zip(original.parameters(), canonical.model.parameters()):
        np.testing.assert_array_equal(observed.data, expected.data)
    for expected, observed in zip(canonical.model.parameters(), generator.model.parameters()):
        np.testing.assert_array_equal(observed.data, expected.data)
    assert [p.name for p in facade._native_model.parameters()] == [p.name for p in original.parameters()]
    text = "hello world"
    assert canonical.tokenizer.encode(text) == generator.tokenizer.encode(text)
    assert canonical.tokenizer.encode(text) == service.encode(text)
    assert service.native and service.engine is not None


def test_partial_checkpoint_is_rejected_by_every_loader(tmp_path):
    from mini_llm import Model
    from benchmarks.evaluate_generation import load_generator
    path = tmp_path / "incomplete"
    _tiny_checkpoint(path, missing=True)
    for action in (
        lambda: load_native_checkpoint(path),
        lambda: Model.load(path),
        lambda: load_generator(path, None, "reference"),
        lambda: load_model(ModelSpec(id="bad", path=path, format="native")),
    ):
        with pytest.raises(ValueError, match="Incomplete checkpoint"):
            action()


def test_legacy_text_generation_import_is_same_class():
    from inference import TextGenerator as CLITextGenerator
    from mini_llm.legacy_generation import TextGenerator
    assert CLITextGenerator is TextGenerator
    from mini_llm.block_proposal import benchmark_adapter
    assert "from inference import" not in Path(benchmark_adapter.__file__).read_text()


def test_remote_model_registry_is_framework_neutral_and_declared_only(tmp_path):
    manifest = tmp_path / "catalog.json"
    manifest.write_text(json.dumps({"models": [
        {"id": "model-b", "format": "native", "path": "relative/model-b"},
        {"id": "smollm2", "format": "gguf", "path": "models/smollm.gguf"},
        {"id": "lfm-server", "format": "remote", "api_url": "http://127.0.0.1:18080/v1",
         "remote_model": "lfm", "api_key_env": "LOCAL_LLM_API_KEY"},
    ]}))
    catalog = ModelCatalog.from_manifest(manifest)
    assert catalog.names() == ("model-b", "smollm2", "lfm-server")
    assert catalog.get("model-b").path == tmp_path / "relative/model-b"
    assert catalog.get("smollm2").resolved_format() == "gguf"
    remote = catalog.get("lfm-server")
    assert remote.path is None and remote.resolved_format() == "remote"
    engine = load_model(remote)
    assert engine.is_remote and engine.context_length is None
    assert engine.spec.remote_model == "lfm"
    with pytest.raises(KeyError):
        catalog.get("unknown")


@pytest.mark.parametrize("value", [
    {"format": "remote", "api_url": "http://user:secret@localhost/v1", "remote_model": "x"},
    {"format": "remote", "api_url": "file:///tmp/model", "remote_model": "x"},
    {"format": "remote", "api_url": "http://127.0.0.1/v1"},
    {"format": "remote", "api_url": "https://localhost/v1", "remote_model": "x", "path": "x"},
    {"format": "native", "path": "x", "api_url": "http://localhost"},
])
def test_source_rejects_invalid_remote_configuration(value, tmp_path):
    with pytest.raises(ValueError):
        ModelSpec.from_dict({"id": "x", **value}, tmp_path)


def test_native_context_policy_matches_memory_horizon():
    cfg = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    assert resolve_native_context(cfg) == cfg.context_length
    assert resolve_native_context(cfg, 1000) == 1000
    with pytest.raises(ValueError, match="positive"):
        resolve_native_context(cfg, 0)


def test_cli_uses_shared_native_loader_end_to_end(tmp_path):
    """Run the real CLI so loader-extraction regressions are not missed."""
    import os
    import subprocess
    import sys
    checkpoint = tmp_path / "tiny-cli"
    _tiny_checkpoint(checkpoint)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MINI_LLM_BACKEND="numpy")
    result = subprocess.run(
        [sys.executable, str(root / "inference.py"), "--checkpoint", str(checkpoint),
         "--prompt", "hello", "--max-new-tokens", "1", "--runtime-profile", "reference"],
        cwd=root, env=env, capture_output=True, text=True, timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Generated:" in result.stdout
    assert "Model loading:" in result.stdout
    assert "Inference context/cache horizon: 128" in result.stdout


def test_serving_pretrained_loader_delegates_to_shared_library(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mini_llm import model_loading
    seen = []
    class ExternalTokenizer:
        def token_to_id(self, text):
            return 2 if text == "<|eos|>" else None
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1, 2])
        def decode(self, ids, skip_special_tokens=False):
            return "ok"
    engine = object()
    def fake_load(path, *, format, dtype):
        seen.append((path, format, dtype))
        return model_loading.LoadedPretrained(engine, ExternalTokenizer(), 128, format)
    monkeypatch.setattr(model_loading, "load_pretrained_weights", fake_load)
    spec = ModelSpec(id="smollm", path=tmp_path / "smollm.gguf", format="gguf",
                     context_length=64, dtype="float16")
    service = load_model(spec)
    assert service.engine is engine
    assert service.context_length == 64
    assert not service.native
    assert service.stop_ids == {2}
    assert seen == [(spec.path, "gguf", "float16")]
    with pytest.raises(ValueError, match="exceeds model capacity"):
        load_model(ModelSpec(id="large", path=spec.path, format="gguf", context_length=256))
