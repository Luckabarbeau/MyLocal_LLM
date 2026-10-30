"""0103-0: CPU interface contracts; no changes to active training kernels."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from mini_llm import Model
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.model_contracts import (
    ArchitectureSpec, UnsupportedExecutionBackend, UnsupportedModelOperation,
    UnverifiedExecutionPath,
)
from mini_llm.model_registry import ModelRegistry, registry


def tiny_config():
    return ModelConfig(tokenizer_vocab_size=128, context_length=128,
                       n_layers=1, d_model=64, d_head=64,
                       n_q_heads=1, n_kv_heads=1,
                       n_experts=1, top_k=1, d_ff=128, ffn_mode="dense")


def test_model_b_is_the_sparse16_core_preset_without_allocating_348m():
    c = registry.config("model_b_348m_sparse16")
    spec = ArchitectureSpec.from_config(c)
    assert spec.family == "dual_resolution_matrix_pool_sparse"
    assert (spec.layers, spec.hidden_size, spec.ffn_size) == (20, 1024, 4096)
    assert (spec.block_size, spec.summaries_per_block) == (128, 4)
    assert (spec.local_window, spec.long_local_window) == (512, 4096)
    assert spec.sparse_stride == 16
    assert spec.sparse_layers == tuple(range(2, 18))
    assert spec.sparse_head == 1
    assert "sparse_stride_history" in [o.name for o in spec.operators]


def test_native_wrap_inspection_does_not_copy_or_allocate_model_weights():
    original = DecoderLanguageModel(tiny_config(), rng_seed=12)
    before = [id(p.data) for p in original.parameters()]
    facade = Model.wrap_native(original, model_id="tiny")
    info = facade.inspect()
    assert [id(p.data) for p in original.parameters()] == before
    assert info.parameter_count == sum(np.prod(p.shape) for p in original.parameters())
    assert dict(info.parameter_shapes) == {
        p.name: tuple(p.shape) for p in original.parameters()
    }
    assert info.model_id == "tiny"
    assert facade.capabilities.inspect and facade.capabilities.accessible_weights
    assert not facade.capabilities.generate  # No tokenizer was provided.
    assert not facade.capabilities.train  # Not wired yet, despite native capability.
    with pytest.raises(UnsupportedModelOperation, match="tokenizer"):
        facade.generate("Hello")


def test_unverified_acceleration_cannot_be_claimed_fast():
    facade = Model.wrap_native(DecoderLanguageModel(tiny_config()))
    report = facade.execution_info()
    assert report.backend == "numpy"
    assert report.device == "cpu"
    assert not report.acceleration_verified
    with pytest.raises(UnverifiedExecutionPath):
        report.require_fast()


def test_backend_is_explicit_and_no_silent_pytorch_fallback(tmp_path):
    from mini_llm.model_api import _validate_backend
    assert _validate_backend("auto", "auto") == "numpy"
    assert _validate_backend("numpy", "cpu") == "numpy"
    with pytest.raises(UnsupportedExecutionBackend, match="PyTorch"):
        _validate_backend("torch", "cpu")
    with pytest.raises(UnsupportedExecutionBackend, match="MINI_LLM_BACKEND"):
        _validate_backend("cupy", "auto")
    with pytest.raises(UnsupportedExecutionBackend, match="device"):
        _validate_backend("numpy", "cuda")
    with pytest.raises(ValueError, match="backend"):
        _validate_backend("bogus", "auto")


def test_registry_rejects_duplicate_names_and_resolves_lazily():
    reg = ModelRegistry()
    called = []
    reg.register("test", lambda: called.append(True) or tiny_config())
    assert not called
    assert reg.names() == ("test",)
    assert reg.config("test").d_model == 64
    assert called == [True]
    with pytest.raises(ValueError, match="already registered"):
        reg.register("test", tiny_config)
    with pytest.raises(KeyError, match="Unknown"):
        reg.config("missing")


def test_native_checkpoint_load_and_schema_validation(tmp_path, monkeypatch):
    from mini_llm.checkpoint import save_checkpoint
    from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer

    class DummyTokenizer:
        def encode(self, text):
            return [1]
        def decode(self, ids):
            return ""

    monkeypatch.setattr(SimpleBPETokenizer, "load", lambda p: DummyTokenizer())
    config = tiny_config()
    original = DecoderLanguageModel(config, rng_seed=3)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text(
        '{"vocab_size": 128, "token_to_id": {}}', encoding="utf-8"
    )
    save_checkpoint(checkpoint, {p.name: p.data for p in original.parameters()})
    loaded = Model.load(checkpoint, backend="numpy", device="cpu")
    assert loaded.checkpoint == str(checkpoint.resolve())
    assert loaded.capabilities.generate
    for before, after in zip(original.parameters(), loaded._native_model.parameters()):
        assert before.name == after.name
        np.testing.assert_array_equal(before.data, after.data)
    assert loaded.inspect().parameter_count == sum(p.size for p in original.parameters())
    # Missing weights must never produce a partly randomized 'loaded' model.
    target = next(iter(original.parameters()))
    (checkpoint / "model_params" / f"{target.name}.npy").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        Model.load(checkpoint)


def test_generate_uses_existing_engine_and_does_not_claim_fast(monkeypatch):
    from inference import TextGenerator

    class DummyTokenizer:
        pass

    seen = []

    def fake_generate(self, prompt, **kwargs):
        seen.append((self.model, self.tokenizer, prompt, kwargs))
        return "output", {"generated_token_ids": [1, 2], "seconds": 0.3}

    monkeypatch.setattr(TextGenerator, "generate", fake_generate)
    raw = DecoderLanguageModel(tiny_config())
    model = Model.wrap_native(raw, tokenizer=DummyTokenizer())
    result = model.generate("hey", max_new_tokens=2)
    assert result.text == "output" and result.token_ids == (1, 2)
    assert result.timing == {"seconds": 0.3}
    assert seen[0][0] is raw and seen[0][1] is model.tokenizer
    assert seen[0][3]["use_kv_cache"] is True
    with pytest.raises(UnverifiedExecutionPath):
        model.generate("hey", require_fast=True)
    assert len(seen) == 1  # Fail before doing expensive work.


def test_shared_operator_registry_is_backend_independent_and_not_implicitly_fast():
    from mini_llm.model_contracts import OperatorSpec
    from mini_llm.model_execution import KernelImplementation, KernelRegistry
    op = OperatorSpec("rmsnorm")
    reg = KernelRegistry()
    impl = KernelImplementation("reference", op, backend="numpy", mode="forward")
    reg.register(impl)
    assert reg.eligible(op, backend="numpy", mode="forward") == ()
    assert reg.eligible(op, backend="numpy", mode="forward", require_verified=False) == (impl,)
    assert reg.eligible(op, backend="cupy", mode="forward", require_verified=False) == ()
    reg.register(KernelImplementation("fused", op, backend="cupy", mode="forward", verified=True))
    assert [i.name for i in reg.eligible(op, backend="cupy", mode="forward")] == ["fused"]
    with pytest.raises(ValueError, match="Duplicate"):
        reg.register(impl)
