"""CPU-only contract tests for frozen residual-specialist foundation (Phase 1)."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from mini_llm.backend import xp
from mini_llm.checkpoint import save_checkpoint
from mini_llm.config import ModelConfig, MemoryContextConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.parameter import Parameter, no_parameter_grad_allocation
from mini_llm.specialists import FrozenFoundation


def tiny_config():
    return ModelConfig(
        tokenizer_vocab_size=64, context_length=8, n_layers=2,
        d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, d_ff=32, ffn_mode="dense",
        dtype="float32",
    )


def build_checkpoint(tmp_path, cfg=None):
    cfg = cfg or tiny_config()
    regular = DecoderLanguageModel(cfg, rng_seed=42)
    save_checkpoint(tmp_path, {p.name: p.data for p in regular.parameters()})
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    frozen = FrozenFoundation.from_checkpoint(tmp_path)
    return regular, frozen


def test_no_grad_allocation_is_scoped():
    with no_parameter_grad_allocation():
        param = Parameter(xp.ones((3, 4)), "frozen.W")
        assert param.grad is None
        param.zero_grad()
    normal = Parameter(xp.ones((3, 4)), "normal.W")
    assert normal.grad is not None
    normal.grad[...] = 1
    normal.zero_grad()
    assert np.array_equal(normal.grad, np.zeros((3, 4)))


def test_checkpoint_load_forward_and_zero_residual_parity(tmp_path):
    regular, frozen = build_checkpoint(tmp_path)
    inputs = np.arange(16, dtype=np.int64).reshape(2, 8)
    old_logits = regular.forward(inputs, return_cache=False)
    hidden, target_slice = frozen.hidden(inputs)
    assert target_slice is None
    new_logits = frozen.project(hidden, target_slice)
    np.testing.assert_array_equal(new_logits, old_logits)
    np.testing.assert_array_equal(frozen.predict(inputs), old_logits)
    assert not any(p.grad is not None for p in frozen.model.parameters())

    # Simulate an untrained specialist with an exactly zero correction.
    residual = np.zeros_like(hidden)
    np.testing.assert_array_equal(frozen.project(hidden + residual), old_logits)


def test_frozen_chunked_ce_gradient_matches_trainable_head(tmp_path):
    regular, frozen = build_checkpoint(tmp_path)
    ids = np.arange(16, dtype=np.int64).reshape(2, 8)
    hidden, target_slice = frozen.hidden(ids)
    targets = (ids + 1) % 64
    mask = np.ones_like(targets, dtype=np.float32)
    mask[0, 0] = 0.0
    loss_a, cache_a = regular.chunked_lm_head_loss_forward(
        hidden, targets, loss_mask=mask, chunk_tokens=5
    )
    dx_a = regular.chunked_lm_head_backward(cache_a)
    loss_b, cache_b = frozen.head_loss(
        hidden, targets, loss_mask=mask, chunk_tokens=5
    )
    dx_b = frozen.head_backward(cache_b)
    np.testing.assert_allclose(loss_b, loss_a, atol=1e-6)
    np.testing.assert_allclose(dx_b, dx_a, rtol=1e-5, atol=1e-6)
    assert any(np.any(p.grad) for p in regular.output_proj.parameters())
    assert all(p.grad is None for p in frozen.model.parameters())


def test_frozen_target_slice_chunked_gradient(tmp_path):
    _, frozen = build_checkpoint(tmp_path)
    ids = np.arange(16, dtype=np.int64).reshape(2, 8)
    hidden, _ = frozen.hidden(ids)
    target_slice = (3, 7)
    targets = np.full((2, 4), 3, dtype=np.int64)
    loss, cache = frozen.head_loss(hidden, targets, target_slice=target_slice, chunk_tokens=3)
    gradient = frozen.head_backward(cache)
    assert np.isfinite(loss)
    assert gradient.shape == hidden.shape
    assert not np.any(gradient[:, :3, :])
    assert not np.any(gradient[:, 7:, :])
    assert np.any(gradient[:, 3:7, :])


def test_checkpoint_missing_tensor_refused(tmp_path):
    _, frozen = build_checkpoint(tmp_path)
    missing_name = frozen.model.parameters()[0].name
    (tmp_path / "model_params" / f"{missing_name}.npy").unlink()
    with pytest.raises(ValueError, match="missing"):
        FrozenFoundation.from_checkpoint(tmp_path)


def test_historical_forward_only_requests_inference_without_changing_training(tmp_path):
    # The real historical_read heads require one of the production schedules.
    # Here a controlled stand-in proves the inference flag travels through the
    # decoder routing path; a 20x1024 production GPU parity test is still needed.
    cfg = tiny_config()
    model = DecoderLanguageModel(cfg, rng_seed=42)
    memory_cfg = replace(
        MemoryContextConfig(), enabled=True, integration_mode="historical_read",
        memory_length=64, recent_length=32, target_length=32, block_size=32,
        router_query_length=16, top_k_blocks=1,
    )
    model.config = replace(cfg, memory_context=memory_cfg)
    class StubMemory:
        def __init__(self):
            self.calls = []
        def open(self, ids, starts, mask, *, inference=False):
            self.calls.append(inference)
            return object()
    stub = StubMemory()
    model.historical_memory = stub
    input_ids = np.arange(64, dtype=np.int64).reshape(1, 64) % 64
    model._hierarchical_context_forward(input_ids, inference=True)
    model._hierarchical_context_forward(input_ids, inference=False)
    assert stub.calls == [True, False]
