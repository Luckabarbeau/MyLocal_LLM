"""0078E: sample first, decide usable historical context second."""
import numpy as np
import pytest

from mini_llm.data.token_shards import (
    build_packed_document_index,
    create_hierarchical_memory_minibatch,
)


def _draw(docs, *, batch_size=1, seed=10, history=3, target=4):
    eos = 99
    packed = np.asarray([tok for doc in docs for tok in [*doc, eos]], dtype=np.uint16)
    inputs, targets, meta = create_hierarchical_memory_minibatch(
        packed, batch_size=batch_size, memory_length=16,
        target_length=target, rng=np.random.default_rng(seed),
        eos_token_id=eos, document_index=build_packed_document_index(packed, eos),
        document_aware=True, return_metadata=True,
        min_history_tokens=history, sampling_policy="document_first",
    )
    return packed, inputs, targets, meta


@pytest.mark.parametrize("pairs,expected_prefix,expected_valid", [
    (2, 0, 2),  # under 4k: padded dense
    (3, 0, 3),
    (4, 0, 4),  # exactly four pairs
    (5, 1, 4),  # between 4k and ~6k: exact one-token prefix
    (6, 2, 4),
    (7, 3, 4),  # threshold: full-router-eligible
    (8, 3, 4),
])
def test_document_first_context_length_transitions(pairs, expected_prefix, expected_valid):
    packed, inputs, labels, meta = _draw([list(range(1, pairs + 1))], seed=19)
    assert meta["document_sampling_policy"] == "document_first"
    assert int(meta["target_valid_lengths"][0]) == expected_valid
    assert not bool(meta["document_selection_fallback"][0])
    assert int(meta["target_source_starts"][0]) >= expected_prefix
    if pairs <= 7:
        assert int(meta["target_source_starts"][0]) == expected_prefix
    start = int(meta["target_source_starts"][0])
    actual_hist = min(start, 16)
    assert actual_hist >= expected_prefix
    np.testing.assert_array_equal(inputs[0, 16-actual_hist:16], packed[start-actual_hist:start])
    np.testing.assert_array_equal(inputs[0, 16:16+expected_valid], packed[start:start+expected_valid])
    np.testing.assert_array_equal(labels[0, :expected_valid], packed[start+1:start+expected_valid+1])
    np.testing.assert_array_equal(meta["target_loss_mask"][0, :expected_valid], np.ones(expected_valid))
    np.testing.assert_array_equal(meta["target_loss_mask"][0, expected_valid:], np.zeros(4-expected_valid))
    assert np.all(inputs[0, :16-actual_hist] == 99)


def test_very_short_documents_remain_sampled_despite_impossible_history():
    # Both short and long documents must remain in the source population.
    short = [11, 12, 13, 14, 15]
    long = list(range(20, 40))
    packed, inputs, labels, meta = _draw([short, long], seed=21, batch_size=800, history=3)
    assert not np.any(meta["document_selection_fallback"])
    selected = np.asarray(meta["document_starts"])
    assert 0 < np.count_nonzero(selected == 0) < len(selected)
    assert 0 < np.count_nonzero(selected == 6) < len(selected)
    short_idx = np.flatnonzero(selected == 0)
    assert np.all(meta["target_valid_lengths"][short_idx] == 4)
    assert np.all(meta["history_valid_starts"][short_idx] == 15)
    np.testing.assert_array_equal(
        inputs[short_idx[0], 15:16], np.asarray([11], dtype=np.uint16)
    )


def test_all_short_cosmopedia_like_docs_have_zero_forced_fallbacks():
    # All docs are too short for router + 4k target, but no doc is reselected.
    docs = [list(range(i+1, i+1+length)) for i, length in zip(range(1, 20), [2, 3, 4, 5, 6, 7, 8] * 3)]
    packed, inputs, labels, meta = _draw(docs, seed=2026, batch_size=300, history=3)
    assert not np.any(meta["document_selection_fallback"])
    assert len(set(map(int, meta["document_starts"]))) >= 10
    assert np.all(meta["target_valid_lengths"] > 0)


def test_policy_rejects_unknown_value():
    eos = 99
    with pytest.raises(ValueError, match="sampling_policy"):
        create_hierarchical_memory_minibatch(
            np.asarray([1, 2, eos]), 1, 16, 4, eos_token_id=eos,
            document_aware=True, sampling_policy="unknown"
        )


def _make_trainer(tmp_path, *, documents, batch_size=1, retrieval_probability=0.0):
    from mini_llm.config import MemoryContextConfig, ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer
    eos = 31
    data = np.asarray([v for doc in documents for v in [*doc, eos]], dtype=np.uint16)
    path = tmp_path / "packed.bin"
    data.tofile(path)
    memory = MemoryContextConfig(
        enabled=True, memory_length=16, recent_length=4, target_length=4,
        block_size=2, top_k_blocks=1, router_query_length=4,
        router_dim=4, read_heads=2, read_kv_heads=1, read_query_chunk=2,
        integration_mode="routed_prefix", memory_training="joint",
        retrieval_batch_probability=retrieval_probability,
    )
    cfg = ModelConfig(
        tokenizer_vocab_size=32, context_length=memory.active_length,
        n_layers=1, d_model=8, n_q_heads=2, n_kv_heads=1,
        d_head=4, n_experts=2, top_k=1, d_ff=16,
        dtype="float32", memory_context=memory,
    )
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(cfg, rng_seed=12),
        train_shard_paths=[str(path)], val_shard_paths=[str(path)],
        batch_size=batch_size, seq_length=memory.source_input_length,
        grad_accum_steps=1, total_steps=2, warmup_steps=1,
        val_steps=1, val_interval=100, save_interval=100,
        eos_token_id=eos,
    )
    return trainer


def test_trainer_uses_direct_prefix_without_retrieval_probability(tmp_path):
    from mini_llm.backend import xp
    trainer = _make_trainer(tmp_path, documents=[[1, 2, 3, 4, 5]], retrieval_probability=0.0)
    x, y, meta = trainer.get_train_batch(return_metadata=True)
    assert meta["document_sampling_policy"] == "document_first"
    assert meta["retrieval_requested"] is False
    assert meta["use_retrieval"] is True
    assert meta["history_mode_requested"] == "direct_history"
    assert np.all(meta["target_valid_lengths"] == 4)
    active, cache = trainer.model._hierarchical_context_forward(xp.asarray(x), memory_metadata=meta)
    assert cache["prefix_mode"] == "direct_history"
    assert cache["prefix_length"] == 1
    assert active.target_start == 1
    assert active.target_end == 5


def test_trainer_does_not_force_longest_doc_for_short_cosmo_source(tmp_path):
    # Exact bug: requested router in a source dominated by short docs must
    # not collapse into one longest document per shard.
    docs = [list(range(1, 1 + (i % 4 + 2))) for i in range(50)]
    trainer = _make_trainer(tmp_path, documents=docs, batch_size=2, retrieval_probability=1.0)
    observed = set()
    for _ in range(80):
        _, _, meta = trainer.get_train_batch(return_metadata=True)
        observed.update(int(x) for x in meta["document_starts"])
        assert not np.any(meta["document_selection_fallback"])
    assert len(observed) > 25


def test_ragged_medium_documents_use_common_real_suffix(tmp_path):
    from mini_llm.backend import xp
    # Test real model behavior, not just sampler flags: no padding token can
    # enter the prefix, and target positions still index only the final 4k.
    trainer = _make_trainer(tmp_path, documents=[[1,2,3,4,5], [11,12,13,14,15,16]],
                            batch_size=2, retrieval_probability=0.0)
    packed = np.asarray([1,2,3,4,5,31,11,12,13,14,15,16,31], dtype=np.uint16)
    from mini_llm.data.token_shards import create_hierarchical_memory_minibatch, build_packed_document_index
    # Deterministic choice of both lengths (one row with one history token,
    # the other row with two) using a small sequence of fixed draws.
    for seed in range(120):
        x, _, meta = create_hierarchical_memory_minibatch(
            packed, batch_size=2, memory_length=12, target_length=4,
            rng=np.random.default_rng(seed), eos_token_id=31,
            document_index=build_packed_document_index(packed,31),
            document_aware=True, return_metadata=True,
            min_history_tokens=3, sampling_policy="document_first",
        )
        available = 12 - meta["history_valid_starts"]
        if set(map(int, available)) == {1, 2}:
            break
    else:
        pytest.fail("failed to sample both medium-length document classes")
    meta["use_retrieval"] = True
    active, cache = trainer.model._hierarchical_context_forward(xp.asarray(x), memory_metadata=meta)
    assert cache["prefix_mode"] == "direct_history"
    assert cache["prefix_length"] == 1
    np.testing.assert_array_equal(
        np.asarray(active.token_ids)[:,0], x[:,11]
    )
    assert active.target_start == 1
    assert active.target_end == 5
