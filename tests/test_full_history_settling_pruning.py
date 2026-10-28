from types import SimpleNamespace

import numpy as np

from mini_llm.backend import RandomStream, xp
from mini_llm.ops.attention import GQAAttention
from mini_llm.ops.embedding import Embedding
from mini_llm.ops.experts import ExpertFFN
from mini_llm.ops.full_history_head import FullHistoryHead


def _assert_close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_dense_ffn_pruning_preserves_input_gradient_and_skips_weight_grads(monkeypatch):
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", "0")
    reference = ExpertFFN(8, 16, 0.02, 0.02, RandomStream(101), dtype="float32")
    pruned = ExpertFFN(8, 16, 0.02, 0.02, RandomStream(101), dtype="float32")
    rng = np.random.default_rng(102)
    x = xp.asarray(rng.normal(size=(6, 8)).astype(np.float32))
    dy = xp.asarray(rng.normal(size=(6, 8)).astype(np.float32))

    _, reference_cache = reference.forward(x, return_cache=True, cache_input=True)
    _, pruned_cache = pruned.forward(x, return_cache=True, cache_input=True)
    reference_dx = reference.backward(dy, reference_cache)
    pruned_dx = pruned.backward(
        dy, pruned_cache, accumulate_parameter_grads=False
    )

    _assert_close(pruned_dx, reference_dx)
    assert all(np.any(np.asarray(parameter.grad) != 0) for parameter in reference.parameters())
    assert all(np.all(np.asarray(parameter.grad) == 0) for parameter in pruned.parameters())


def test_attention_pruning_preserves_input_gradient_and_skips_weight_grads(monkeypatch):
    monkeypatch.setenv("MINI_LLM_TRAIN_ATTN_BACKEND", "grouped")
    reference = GQAAttention(
        d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        input_std=0.02, output_std=0.02, rng=RandomStream(103),
        dtype="float32",
    )
    pruned = GQAAttention(
        d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        input_std=0.02, output_std=0.02, rng=RandomStream(103),
        dtype="float32",
    )
    rng = np.random.default_rng(104)
    x = xp.asarray(rng.normal(size=(1, 5, 8)).astype(np.float32))
    dy = xp.asarray(rng.normal(size=(1, 5, 8)).astype(np.float32))

    _, reference_cache = reference.forward(x, return_cache=True)
    _, pruned_cache = pruned.forward(x, return_cache=True)
    reference_dx = reference.backward(dy, reference_cache)
    pruned_dx = pruned.backward(
        dy, pruned_cache, accumulate_parameter_grads=False
    )

    _assert_close(pruned_dx, reference_dx)
    assert all(np.any(np.asarray(parameter.grad) != 0) for parameter in reference.parameters())
    assert all(np.all(np.asarray(parameter.grad) == 0) for parameter in pruned.parameters())


def _full_history_fixture(seed):
    embedding = Embedding(32, 8, 0.02, RandomStream(seed), dtype="float32")
    head = FullHistoryHead(
        d_model=8, d_head=4, embedding=embedding.W,
        rng=RandomStream(seed + 1), init_std=0.02, rope_base=10_000.0,
        name="blocks.0.full_history", dtype="float32",
        zero_output_init=False,
    )
    state = SimpleNamespace(
        ids=xp.asarray([[3, 4, 5]], dtype=xp.int32),
        valid=xp.asarray([[True, True, True]], dtype=xp.bool_),
        positions=xp.asarray([0, 1, 2], dtype=xp.int64),
    )
    return embedding, head, state


def test_full_history_pruning_keeps_head_grads_but_skips_frozen_embedding_grad():
    reference_embedding, reference, reference_state = _full_history_fixture(105)
    pruned_embedding, pruned, pruned_state = _full_history_fixture(105)
    rng = np.random.default_rng(107)
    current = xp.asarray(rng.normal(size=(1, 4, 8)).astype(np.float32))
    positions = xp.asarray([[3, 4, 5, 6]], dtype=xp.int64)
    dy = xp.asarray(rng.normal(size=(1, 4, 8)).astype(np.float32))

    _, reference_cache = reference.forward(
        current, reference_state, positions, return_cache=True
    )
    _, pruned_cache = pruned.forward(
        current, pruned_state, positions, return_cache=True
    )
    reference_dx = reference.backward(dy, reference_cache)
    pruned_dx = pruned.backward(
        dy, pruned_cache, accumulate_embedding_grad=False
    )

    _assert_close(pruned_dx, reference_dx)
    for reference_parameter, pruned_parameter in zip(
        reference.parameters(), pruned.parameters()
    ):
        _assert_close(pruned_parameter.grad, reference_parameter.grad)
    assert np.any(np.asarray(reference_embedding.W.grad) != 0)
    assert np.all(np.asarray(pruned_embedding.W.grad) == 0)
