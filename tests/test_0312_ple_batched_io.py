"""Exact PLE row/sequence parity with bounded disk I/O, no real checkpoint."""
import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_ple import Qwen4ExpPLE
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0206_qwen4exp_ple import store, weights, TOKENS, hash_oracle, row_oracle


def module(store, monkeypatch, enabled):
    monkeypatch.setenv('MINI_LLM_PLE_BATCHED_ROWS', str(int(enabled)))
    return Qwen4ExpPLE.from_store(store, **weights()[0])


def test_exact_rows_order_bounded_decode_and_cache(store, monkeypatch):
    ple = module(store, monkeypatch, True)
    original = PackedGGUFWeight.dequantize_reference
    shapes = []
    def decode(weight):
        shapes.append(weight.shape)
        assert weight.shape == (16, 160)
        return original(weight)
    monkeypatch.setattr(PackedGGUFWeight, 'dequantize_reference', decode)
    # The optimized path must never dereference the full mmap or call its row view.
    def forbidden(*args, **kwargs):
        pytest.fail('unexpected mapped payload accessor')
    monkeypatch.setattr(store, 'rows', forbidden)
    monkeypatch.setattr(store, 'tensor_bytes', forbidden)
    try:
        for _ in range(2):
            result = ple.embedding(7, (-1, -1))
            indices = hash_oracle(7, (-1, -1), ple.config)
            np.testing.assert_array_equal(asnumpy(result), np.concatenate([row_oracle(i) for i in indices]))
        assert shapes == [(16, 160), (16, 160)]
        stats = ple._row_reader.stats()
        assert stats['cache_hits'] == 16
        assert stats['disk_logical_read_bytes'] == 16 * 90
        assert stats['cached_bytes'] == 16 * 90
    finally:
        ple.close()
    assert ple._row_reader.closed
    ple.close()
    with pytest.raises(RuntimeError, match='closed'):
        ple.embedding(7, (-1, -1))


def test_complete_sequence_and_request_reset_are_bit_exact(store, monkeypatch):
    reference = module(store, monkeypatch, False)
    candidate = module(store, monkeypatch, True)
    rng = np.random.default_rng(312)
    streams = xp.asarray(rng.normal(0, .4, (len(TOKENS), 4, 2560)).astype(np.float32))
    try:
        for sequence in (TOKENS, [9, 8, 7], TOKENS):
            a, b = reference.fresh_state(), candidate.fresh_state()
            for token, x in zip(sequence, streams):
                expected, a = reference.step(token, x, a)
                actual, b = candidate.step(token, x, b)
                np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
                np.testing.assert_array_equal(asnumpy(b.normalized_history), asnumpy(a.normalized_history))
                assert b.predecessors == a.predecessors
        assert candidate._row_reader.stats()['cached_bytes'] <= 65536 * 90
    finally:
        reference.close()
        candidate.close()


def test_closed_store_is_rejected_even_when_rows_cached(store, monkeypatch):
    candidate = module(store, monkeypatch, True)
    try:
        candidate.embedding(7, (-1, -1))
        store.close()
        with pytest.raises(RuntimeError, match='store is closed'):
            candidate.embedding(7, (-1, -1))
    finally:
        candidate.close()


def test_failed_or_invalid_lookup_does_not_publish_reader(store, monkeypatch):
    candidate = module(store, monkeypatch, True)
    try:
        with pytest.raises(ValueError, match='two oldest'):
            candidate.embedding(7, (-1,))
        assert candidate._row_reader is None
    finally:
        candidate.close()
