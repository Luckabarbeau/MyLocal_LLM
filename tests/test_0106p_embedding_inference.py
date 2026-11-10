"""CPU contracts for generic provider-backed row lookup."""

import os
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops import embedding_inference
from mini_llm.ops.embedding_inference import RowEmbeddingInference


pytestmark = pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU NumPy tests only")


def _lookup(*, dtype="float32", cache_bytes=0):
    table = (np.arange(35).reshape(7, 5) / 7).astype(np.float64)
    calls = []

    def provider(row):
        assert type(row) is int
        calls.append(row)
        return table[row]

    return RowEmbeddingInference(7, 5, provider, dtype=dtype,
                                 cache_bytes=cache_bytes), table, calls


@pytest.mark.parametrize("dtype", ["float16", np.float32, np.dtype("float64")])
@pytest.mark.parametrize("shape", [(), (1,), (6,), (2, 3), (1, 2, 3)])
@pytest.mark.parametrize("id_dtype", ["int8", "int16", "int32", "int64",
                                      "uint8", "uint16", "uint32", "uint64"])
def test_values_dtype_and_arbitrary_shapes(dtype, shape, id_dtype):
    layer, table, calls = _lookup(dtype=dtype)
    ids = xp.asarray(np.resize([6, 2, 6, 0, 2, 1], shape), dtype=id_dtype)
    actual = layer.forward(ids)
    assert isinstance(actual, xp.ndarray)
    assert actual.shape == shape + (5,)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(asnumpy(actual), table[asnumpy(ids)].astype(dtype))
    assert calls == list(dict.fromkeys(asnumpy(ids).reshape(-1).tolist()))
    assert layer.statistics()["resident_rows"] == 0


def test_noncontiguous_readonly_ids_are_not_mutated():
    layer, table, _ = _lookup()
    owner = np.asarray([[0, 6, 1, 5], [2, 4, 3, 0]], dtype=np.int64)
    ids = owner.T[::-1, ::-1]
    ids.flags.writeable = False
    before, flags = owner.copy(), str(ids.flags)
    actual = layer.forward(ids)
    np.testing.assert_array_equal(asnumpy(actual), table[ids].astype(np.float32))
    np.testing.assert_array_equal(owner, before)
    assert str(ids.flags) == flags
    assert owner.flags.writeable


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3), (0, 0)])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_empty_has_correct_shape_no_reads_or_cache_changes(monkeypatch, shape, dtype):
    layer, _, calls = _lookup(dtype=dtype, cache_bytes=100)
    layer.forward(xp.asarray([3], dtype="int32"))
    before = layer.statistics()

    def unexpected_readback(value):
        pytest.fail("empty IDs must not be read back")

    monkeypatch.setattr(embedding_inference, "asnumpy", unexpected_readback)
    actual = layer.forward(xp.empty(shape, dtype="uint64"))
    assert actual.shape == shape + (5,)
    assert actual.dtype == np.dtype(dtype)
    assert calls == [3]
    assert layer.statistics() == before


@pytest.mark.parametrize("bad", [
    [0, 1], 0, np.int64(1), np.array([1.0]), np.array([True]),
    np.array([1j]), np.array(["1"]), np.array([1], dtype=object),
    np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool),
    np.array([0, -1]), np.array([0, 7]), np.array([0, 2**63 - 1]),
    np.array([0, 2**64 - 1], dtype=np.uint64), np.array(-1), np.array(7),
])
def test_all_id_validation_precedes_provider_and_cache_mutation(bad):
    layer, _, calls = _lookup(cache_bytes=40)
    layer.forward(xp.asarray([0, 2], dtype="int64"))
    before = layer.statistics()
    with pytest.raises((TypeError, ValueError)):
        layer.forward(bad)
    assert calls == [0, 2]
    assert layer.statistics() == before
    # Invalid IDs must not even promote an earlier valid cache hit.
    layer.forward(xp.asarray([3, 0], dtype="int64"))
    assert calls == [0, 2, 3, 0]


@pytest.mark.parametrize("cache_bytes", [0, 1, 19, 20, 40, 41])
def test_per_forward_dedup_even_when_rows_are_evicted(cache_bytes):
    layer, table, calls = _lookup(cache_bytes=cache_bytes)
    ids = xp.asarray([6, 2, 6, 0, 2, 6, 0], dtype="int32")
    np.testing.assert_array_equal(asnumpy(layer.forward(ids)), table[ids].astype("float32"))
    assert calls == [6, 2, 0]
    assert layer.statistics() == {
        "resident_rows": min(3, cache_bytes // 20),
        "used_bytes": min(3, cache_bytes // 20) * 20,
        "provider_loads": 3,
        "cache_hits": 0,
    }
    layer.forward(ids)
    if cache_bytes < 20:
        assert calls == [6, 2, 0, 6, 2, 0]
    assert layer.statistics()["used_bytes"] <= cache_bytes


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_bounded_lru_and_statistics_snapshot(dtype):
    row_bytes = 5 * np.dtype(dtype).itemsize
    layer, _, calls = _lookup(dtype=dtype, cache_bytes=2 * row_bytes + 1)
    layer.forward(xp.asarray([2, 0, 2], dtype="int64"))
    snapshot = layer.statistics()
    layer.forward(xp.asarray([2, 2], dtype="int64"))  # Promote 2, not 0.
    layer.forward(xp.asarray([1], dtype="int64"))  # Evict 0.
    layer.forward(xp.asarray([2], dtype="int64"))  # Still resident.
    layer.forward(xp.asarray([0], dtype="int64"))  # Reload 0, evict 1.
    assert calls == [2, 0, 1, 0]
    assert layer.statistics() == {
        "resident_rows": 2, "used_bytes": 2 * row_bytes,
        "provider_loads": 4, "cache_hits": 2,
    }
    assert snapshot == {
        "resident_rows": 2, "used_bytes": 2 * row_bytes,
        "provider_loads": 2, "cache_hits": 0,
    }
    snapshot["resident_rows"] = 999
    assert layer.statistics()["resident_rows"] == 2


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_cache_owns_rows_without_owner_data_or_flags_mutation(readonly, dtype):
    owner = np.arange(12, dtype=dtype).reshape(2, 6).copy()
    row = owner[0, ::2]
    row.flags.writeable = not readonly
    before = owner.copy()
    owner_flags, row_flags = str(owner.flags), str(row.flags)
    layer = RowEmbeddingInference(2, 3, lambda index: row, cache_bytes=100, dtype=dtype)
    actual = layer.forward(xp.asarray([0, 0], dtype="int32"))
    np.testing.assert_array_equal(owner, before)
    assert str(owner.flags) == owner_flags
    assert str(row.flags) == row_flags
    owner[0, ::2] = -50
    actual[...] = 77
    expected = before[0, ::2]
    np.testing.assert_array_equal(asnumpy(layer.forward(xp.asarray(0))), expected)
    cached = layer._cache[0]
    assert isinstance(cached, np.ndarray)
    assert cached.dtype == np.dtype(dtype)
    assert cached.flags.owndata
    assert not np.shares_memory(cached, owner)
    assert layer.statistics()["provider_loads"] == 1


def test_reused_provider_scratch_and_array_like_rows():
    scratch = np.zeros(3)
    calls = []

    def provider(index):
        calls.append(index)
        scratch[:] = [index, index + 1, index + 2]
        return scratch if index != 2 else scratch.tolist()

    layer = RowEmbeddingInference(4, 3, provider, cache_bytes=36)
    ids = xp.asarray([3, 1, 3, 2], dtype="int32")
    expected = np.asarray([[3, 4, 5], [1, 2, 3], [3, 4, 5], [2, 3, 4]], dtype="float32")
    np.testing.assert_array_equal(asnumpy(layer.forward(ids)), expected)
    scratch[:] = -1
    np.testing.assert_array_equal(asnumpy(layer.forward(ids)), expected)
    assert calls == [3, 1, 2]


@pytest.mark.parametrize("bad_row", [
    1.0, None, [1, 2], [1, 2, 3, 4], [[1, 2, 3]],
    np.zeros((3, 1)), ["bad", "row", "values"], [[1], [2, 3]],
])
def test_malformed_provider_row_is_not_cached(bad_row):
    layer = RowEmbeddingInference(2, 3, lambda index: bad_row, cache_bytes=100)
    with pytest.raises(ValueError, match=r"row_provider\(1\)"):
        layer.forward(xp.asarray([1, 1], dtype="int64"))
    assert layer.statistics() == {
        "resident_rows": 0, "used_bytes": 0, "provider_loads": 1, "cache_hits": 0,
    }


def test_provider_exception_propagates_without_caching_failed_row():
    def provider(index):
        raise RuntimeError("provider unavailable")

    layer = RowEmbeddingInference(2, 3, provider, cache_bytes=100)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        layer.forward(xp.asarray([0], dtype="int32"))
    assert layer.statistics()["resident_rows"] == 0
    assert layer.statistics()["provider_loads"] == 1


@pytest.mark.parametrize("kwargs", [
    {"n_rows": 0}, {"n_rows": -1}, {"n_rows": True}, {"n_rows": 1.5},
    {"embedding_dim": 0}, {"embedding_dim": -1}, {"embedding_dim": True},
    {"embedding_dim": 1.5}, {"cache_bytes": -1}, {"cache_bytes": True},
    {"cache_bytes": 1.5}, {"row_provider": None}, {"dtype": "int32"},
    {"dtype": "complex64"}, {"dtype": "bool"}, {"dtype": "invalid"},
    {"dtype": object},
])
def test_constructor_validation_does_not_call_provider(kwargs):
    def unexpected_provider(index):
        pytest.fail("constructor must not invoke provider")

    arguments = dict(n_rows=2, embedding_dim=3, row_provider=unexpected_provider)
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        RowEmbeddingInference(**arguments)


def test_numpy_integer_metadata_and_large_logical_table():
    calls = []

    def provider(index):
        calls.append(index)
        return [index, 1]

    layer = RowEmbeddingInference(np.int64(10**12), np.int32(2), provider,
                                 cache_bytes=np.int64(8))
    actual = layer.forward(xp.asarray([10**12 - 1], dtype="int64"))
    np.testing.assert_array_equal(asnumpy(actual), np.asarray([[10**12 - 1, 1]], dtype="float32"))
    assert calls == [10**12 - 1]
    assert layer.statistics()["used_bytes"] == 8


def test_one_explicit_readback_per_nonempty_forward(monkeypatch):
    layer, _, _ = _lookup(cache_bytes=100)
    reads = []
    original = embedding_inference.asnumpy

    def readback(value):
        reads.append(value)
        return original(value)

    monkeypatch.setattr(embedding_inference, "asnumpy", readback)
    ids = xp.asarray([2, 2, 1], dtype="int64")
    layer.forward(ids)
    layer.forward(ids)
    assert len(reads) == 2
    assert reads[0] is ids and reads[1] is ids


def test_numpy_lookup_does_not_import_or_initialize_cupy():
    code = """
import sys
class RejectCupy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise AssertionError('NumPy lookup attempted a CuPy import')
sys.meta_path.insert(0, RejectCupy())
import numpy as np
from mini_llm.ops.embedding_inference import RowEmbeddingInference
layer = RowEmbeddingInference(4, 2, lambda i: [i, i + 1], cache_bytes=8)
np.testing.assert_array_equal(layer.forward(np.array([2, 2])), [[2, 3], [2, 3]])
assert not any(name == 'cupy' or name.startswith('cupy.') for name in sys.modules)
"""
    environment = dict(os.environ, MINI_LLM_BACKEND="numpy", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
