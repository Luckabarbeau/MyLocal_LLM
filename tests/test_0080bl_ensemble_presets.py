"""0080BL: new dense ensemble presets and actual tied-head gradients."""

from dataclasses import replace

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from train_model import model_config_from_name


@pytest.mark.parametrize("name,layers,width,params,head_counts", [
    ("ensemble-deep-512-dense-memory-64k", 32, 512, 107069952, (3, 1, 2, 2)),
    ("ensemble-wide-768-dense-memory-64k", 13, 768, 106764288, (6, 2, 2, 2)),
])
def test_preset_geometry_and_parameter_estimate(name, layers, width, params, head_counts):
    c = model_config_from_name(name)
    assert c.n_layers == layers and c.d_model == width
    assert c.vocab_size == 32768 and c.ffn_mode == "dense" and c.tie_embeddings
    assert c.n_experts == c.top_k == 1
    assert c.estimated_parameter_count() == params
    mem = c.memory_context
    assert mem.integration_mode == "routed_prefix"
    assert (mem.memory_length, mem.distant_memory_length, mem.target_length,
            mem.retrieved_length, mem.active_length) == (65536, 61440, 4096, 2048, 6144)
    from mini_llm.config import (LocalAttentionConfig, DilatedAttentionConfig,
                                GlobalSparseAttentionConfig, RetrievalAttentionConfig)
    heads = c.attention_layers[0].heads
    assert tuple(sum(isinstance(h, t) for h in heads) for t in
                 (LocalAttentionConfig, DilatedAttentionConfig,
                  GlobalSparseAttentionConfig, RetrievalAttentionConfig)) == head_counts


def _small(tie=True):
    return ModelConfig(tokenizer_vocab_size=29, context_length=8, n_layers=1,
                       d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
                       n_experts=1, top_k=1, ffn_mode="dense", d_ff=32,
                       tie_embeddings=tie)


def test_tied_head_single_parameter_and_gradients():
    config = _small()
    model = DecoderLanguageModel(config, rng_seed=11, dtype="float64")
    assert model.output_proj.parameters() == []
    assert len(set(id(p) for p in model.parameters())) == len(model.parameters())
    assert sum(p.size for p in model.parameters()) == config.estimated_parameter_count()
    assert np.shares_memory(model.embedding.W.data, model.output_proj.W.data)
    assert np.shares_memory(model.embedding.W.grad, model.output_proj.W.grad)
    ids = np.array([[1, 2, 4]], dtype=np.int64)
    targets = np.array([[2, 4, 3]], dtype=np.int64)

    def loss():
        logits = model.forward(ids, return_cache=False)
        value, _ = model.compute_loss(logits, targets)
        return float(value)

    logits, cache = model.forward(ids)
    _, loss_cache = model.compute_loss(logits, targets)
    grad = model.backward_loss(loss_cache)
    model.zero_grad()
    model.backward(grad, cache)
    w = model.embedding.W.data
    numeric_indices = [(1, 0), (2, 3), (5, 7)]
    for index in numeric_indices:
        old = float(w[index]); epsilon = 1e-5
        w[index] = old + epsilon; plus = loss()
        w[index] = old - epsilon; minus = loss()
        w[index] = old
        numeric = (plus - minus) / (2 * epsilon)
        assert np.isclose(model.embedding.W.grad[index], numeric,
                          rtol=5e-4, atol=3e-5), (index, model.embedding.W.grad[index], numeric)


def test_chunked_tied_head_matches_full_projection():
    model = DecoderLanguageModel(_small(), rng_seed=12, dtype="float64")
    ids = np.array([[1, 2, 4, 3]], dtype=np.int64)
    targets = np.array([[2, 4, 3, 5]], dtype=np.int64)
    full, cache = model.forward(ids)
    full_loss, loss_cache = model.compute_loss(full, targets)
    grads = model.backward_loss(loss_cache)
    model.zero_grad(); model.backward(grads, cache)
    expected = model.embedding.W.grad.copy()
    model.zero_grad()
    head, body_cache = model.forward_body(ids)
    chunk_loss, chunk_cache = model.chunked_lm_head_loss_forward(
        head, targets, chunk_tokens=2)
    dx = model.chunked_lm_head_backward(chunk_cache)
    model.backward_body(dx, body_cache)
    assert abs(float(full_loss) - float(chunk_loss)) < 1e-5
    np.testing.assert_allclose(model.embedding.W.grad, expected, rtol=3e-5, atol=3e-5)


def test_tied_checkpoint_contains_only_one_vocab_matrix(tmp_path):
    from mini_llm.checkpoint import save_checkpoint, load_checkpoint
    model = DecoderLanguageModel(_small(), rng_seed=17, dtype="float32")
    values = {p.name: p.data for p in model.parameters()}
    assert "embedding.W" in values and "output_proj.W" not in values
    save_checkpoint(tmp_path / "tied", values)
    loaded, _, _ = load_checkpoint(tmp_path / "tied", skip_optimizer=True)
    restored = DecoderLanguageModel(_small(), rng_seed=39, dtype="float32")
    for p in restored.parameters():
        np.copyto(p.data, loaded[p.name])
    np.testing.assert_allclose(
        restored.output_proj.W.data, restored.embedding.W.data.T, rtol=0, atol=0
    )
    ids = np.array([[1, 2, 4]], dtype=np.int64)
    np.testing.assert_allclose(
        model.forward(ids, return_cache=False),
        restored.forward(ids, return_cache=False), rtol=0, atol=0,
    )
