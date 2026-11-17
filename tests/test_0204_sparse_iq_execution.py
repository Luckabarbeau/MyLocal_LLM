"""0204 source ownership, bounded execution and sparse capability contracts."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.gguf_store import MultiShardTensorStore
from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
from mini_llm.ops import expert_residency as residency
from mini_llm.ops import quantized_linear as linear
from mini_llm.ops.expert_residency import AdaptiveExpertCache, ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.expert_source import ExpertBundle
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from test_0203_native_iq_experts import packed, expert_oracle


IQ_FORMATS = ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0")


def test_gpu_capability_preflight_before_allocation(monkeypatch):
    assert linear.SUPPORTED_GPU_PACKED_FORMATS == frozenset(linear._FORMAT_ID)
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported format must fail before allocation")
    monkeypatch.setattr(linear, "BACKEND_NAME", "cupy")
    monkeypatch.setattr(linear, "xp", SimpleNamespace(asarray=forbidden))
    for fmt in IQ_FORMATS:
        w = packed(fmt, 2, 256, 204)
        assert not linear.supports_packed_gpu_type(fmt)
        with pytest.raises(NotImplementedError, match="GPU kernel"):
            linear.packed_to_device(w)
        with pytest.raises(NotImplementedError, match="GPU kernel"):
            linear.DevicePackedGGUFWeight(None, w.shape, fmt)


def test_bundle_readonly_complete_owner_and_views():
    key = ExpertKey("q", 0, 1)
    original = np.ones((4, 4), np.float32)
    with pytest.raises(ValueError, match="read-only"):
        ExpertBundle(key, (original,) * 3, original, "host")
    view = original.view()
    view.flags.writeable = False
    bundle = ExpertBundle(key, (view,) * 3, original, "host")
    assert original.flags.writeable and not bundle.weights[0].flags.writeable
    assert bundle.owner is original and bundle.nbytes == 3 * original.nbytes
    with pytest.raises(FrozenInstanceError):
        bundle.source_tier = "mmap"
    with pytest.raises(ValueError, match="gate, up and down"):
        ExpertBundle(key, (view,), original, "host")
    with pytest.raises(ValueError, match="source_tier"):
        ExpertBundle(key, (view,) * 3, original, "pinned")


def sparse_store(tmp_path):
    # Real GGUF headers; holes, not 48*512 materialized expert banks.
    from test_0201_gguf_store import write_gguf, STRING, U32
    path = tmp_path / "sparse.gguf"
    metadata = [("general.architecture", STRING, "qwen4exp")]
    metadata += [("qwen4exp." + name, U32, value) for name, value in (
        ("block_count", 48), ("embedding_length", 32), ("expert_count", 512),
        ("expert_used_count", 10), ("expert_feed_forward_length", 64))]
    tensors, offset = [], 0
    for layer in range(48):
        for suffix, dims, kind, size in (
                ("gate_inp", (32, 512), 0, 32 * 512 * 4),
                ("gate_exps", (32, 64, 512), 2, 512 * 64 * 18),
                ("up_exps", (32, 64, 512), 2, 512 * 64 * 18),
                ("down_exps", (64, 32, 512), 2, 512 * 32 * 36)):
            tensors.append((f"blk.{layer}.ffn_{suffix}.weight", dims, kind, offset, None))
            offset = (offset + size + 31) // 32 * 32
    begin = write_gguf(path, tensors, metadata)
    with path.open("r+b") as stream:
        stream.truncate(begin + offset)
    return MultiShardTensorStore.open(path)


def test_index_headers_sparse_selection_owner_memory_plan(tmp_path, monkeypatch):
    store = sparse_store(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("No full tensor bank or decode")
    monkeypatch.setattr(store, "packed_weight", forbidden)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    index = Qwen4ExpGGUFIndex(store, model_id="q")
    assert index.n_layers == 48 and index.n_experts == 512 and index.top_k == 10
    calls = []
    rows = store.rows
    def track(name, first, last, **kwargs):
        calls.append((name, kwargs["outer"]))
        return rows(name, first, last, **kwargs)
    monkeypatch.setattr(store, "rows", track)
    key = ExpertKey("q", 47, 511)
    bundle = index.source(key)
    assert bundle.owner is store and bundle.source_tier == "mmap"
    assert [w.shape for w in bundle.weights] == [(64, 32), (64, 32), (32, 64)]
    assert len(calls) == 3 and all(outer == (511,) for _, outer in calls)
    assert all(not w.data.flags.writeable for w in index.provider(key))
    plan = index.memory_plan(static=1, state=2, activation=3, cache=4,
                             staging=5, headroom=6, device_total=30)
    assert plan["reserved"] == 21 and plan["remaining"] == 9
    assert plan["canonical_mmap_bytes"] > 0
    with pytest.raises(ValueError, match="exceed"):
        index.memory_plan(static=1, state=2, activation=3, cache=4,
                          staging=5, headroom=6, device_total=20)
    with pytest.raises(IndexError):
        index.source(ExpertKey("q", 48, 0))
    with pytest.raises(ValueError, match="identity"):
        index.source(ExpertKey("other", 0, 0))
    store.close()
    assert bundle.nbytes > 0 and int(bundle.weights[0].data[0]) == 0


@pytest.mark.parametrize("defect", ["architecture", "dimension", "missing", "shape"])
def test_index_rejects_bad_headers(tmp_path, defect):
    store = sparse_store(tmp_path)
    if defect == "architecture":
        store.metadata["general.architecture"] = "llama"
    elif defect == "dimension":
        store.metadata["qwen4exp.embedding_length"] = 0
    elif defect == "missing":
        del store.tensors["blk.47.ffn_up_exps.weight"]
    else:
        name = "blk.47.ffn_gate_inp.weight"
        store.tensors[name] = replace(store.tensors[name], shape=(32, 512))
    try:
        with pytest.raises(ValueError):
            Qwen4ExpGGUFIndex(store)
    finally:
        store.close()


def test_sparse_512_top10_native_matches_bounded_oracle(monkeypatch):
    calls = []
    def provider(key):
        calls.append(key.expert_id)
        return tuple(packed(IQ_FORMATS[(key.expert_id + i) % 6], 256, 256,
                            key.expert_id * 3 + i) for i in range(3))
    ids = np.array([[[0, 9, 51, 80, 127, 255, 301, 400, 500, 511]]])
    router = np.arange(1, 11, dtype=np.float32).reshape(1, 1, 10) / 55
    x = np.random.default_rng(204).normal(0, .01, (1, 1, 256)).astype(np.float32)
    expected = sum(router[0, 0, i] * expert_oracle(x.reshape(1, 256), provider(
        ExpertKey("q", 0, int(e)))) for i, e in enumerate(ids[0, 0]))
    calls.clear()
    def forbidden(*args, **kwargs):
        pytest.fail("Native executor must not decode reference matrices")
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    executor = HeterogeneousExpertExecutor(provider, model_id="q", d_model=256,
        d_ff=256, compute_dtype="float32", cpu_packed_backend="native", execution_mode="cpu")
    managed = ExpertsInference(256, 256, 512, expert_executor=executor)
    got = managed.forward(xp.asarray(x), xp.asarray(router), xp.asarray(ids))
    np.testing.assert_allclose(asnumpy(got).reshape(1, 256), expected, atol=2e-5, rtol=5e-5)
    assert sorted(calls) == sorted(ids.reshape(-1)) and len(calls) == 10
    assert not managed.experts and not executor.cache.resident


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_host_budget_bounded_reference_precision(dtype, monkeypatch):
    weights = tuple(packed("IQ4_NL", 64, 64, 204 + i) for i in range(3))
    x = np.random.default_rng(204).normal(0, .01, (2, 64)).astype(dtype)
    matrices = [w.dequantize_reference() for w in weights]
    g, u = x @ matrices[0].T, x @ matrices[1].T
    expected = ((g / (1 + np.exp(-g))) * u) @ matrices[2].T
    decode = PackedGGUFWeight.dequantize_reference
    def bounded(self):
        assert len(self.shape) == 1 or self.shape[0] == 1
        return decode(self)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", bounded)
    executor = HeterogeneousExpertExecutor(lambda key: weights, model_id="q",
        d_model=64, d_ff=64, execution_mode="cpu")
    got = executor._cpu_forward_numpy(ExpertKey("q", 0, 0), weights, x)
    assert got.dtype == dtype
    np.testing.assert_allclose(got, expected, atol=1e-10 if dtype == np.float64 else 2e-7,
                               rtol=1e-12 if dtype == np.float64 else 5e-5)
    assert executor.host_cache.used_bytes == 0


def test_unsupported_cache_miss_fallback_and_strict_gpu(monkeypatch):
    weights = tuple(packed("IQ3_S", 256, 256, 204 + i) for i in range(3))
    monkeypatch.setattr(residency, "BACKEND_NAME", "cupy")
    executor = HeterogeneousExpertExecutor(lambda key: weights, model_id="q",
        d_model=256, d_ff=256, device_budget_bytes=10**6, promote_after=1)
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported format must not materialize")
    monkeypatch.setattr(executor, "_device_expert", forbidden)
    key, expert, selected = executor._select(0, 0, np.ones((1, 256), np.float32))
    assert expert is None and selected == weights and not executor.cache.resident
    executor.execution_mode = "gpu"
    with pytest.raises(NotImplementedError, match="GPU kernel"):
        executor._select(0, 0, np.ones((1, 256), np.float32))
    executor.host_provider = lambda key: (weights[0], weights[1], packed("IQ3_S", 1, 256, 1))
    with pytest.raises(ValueError, match="weight"):
        executor._select(0, 0, np.ones((1, 256), np.float32))
    monkeypatch.setattr(residency, "BACKEND_NAME", "numpy")
    with pytest.raises(RuntimeError, match="cupy"):
        HeterogeneousExpertExecutor(lambda key: weights, model_id="q", d_model=256,
                                    d_ff=256, execution_mode="gpu")


def test_complete_publication_failure_and_inflight():
    cache = AdaptiveExpertCache(30, promote_after=1, replacement_margin=1)
    a, b = ExpertKey("q", 0, 0), ExpertKey("q", 0, 1)
    def partial():
        assert a not in cache.resident and cache.used_bytes == 0
        raise RuntimeError("up projection kernel bug")
    with pytest.raises(RuntimeError, match="kernel bug"):
        cache.request(a, 30, partial)
    assert not cache.resident and cache.used_bytes == 0
    complete = object()
    assert cache.request(a, 30, lambda: complete) is complete
    event = SimpleNamespace(done=False)
    cache.record_inflight((a,), event)
    for _ in range(5):
        assert cache.request(b, 30, lambda: object()) is None
    assert cache.resident[a][0] is complete
    event.done = True
    assert cache.request(b, 30, lambda: "complete") == "complete"
    assert a not in cache.resident and cache.used_bytes == 30


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires actual CuPy backend/device")
@pytest.mark.parametrize("fmt", sorted(linear.SUPPORTED_GPU_PACKED_FORMATS))
def test_actual_gpu_old_formats_and_iq_mixed_fallback(fmt):
    # Use small, aligned experts; kernel/runtime errors are not skipped.
    from test_0106e_cpu_packed_and_gpu_swaps import packed as finite_packed
    weights = tuple(finite_packed(fmt, 256, 256, 204 + i) for i in range(3))
    iq = tuple(packed("IQ3_S", 256, 256, 210 + i) for i in range(3))
    x = np.random.default_rng(204).normal(0, .01, (1, 256)).astype(np.float32)
    executor = HeterogeneousExpertExecutor(lambda key: weights if key.expert_id == 0 else iq,
        model_id="q", d_model=256, d_ff=256, device_budget_bytes=10**6,
        compute_dtype="float32", promote_after=1, cpu_packed_backend="native")
    for e, ws in enumerate((weights, iq)):
        got = executor.forward(0, e, xp.asarray(x))
        xp.cuda.get_current_stream().synchronize()
        expected = expert_oracle(x, ws)
        result = asnumpy(got)
        assert np.isfinite(expected).all(), f"{fmt}, expert {e}: non-finite oracle"
        assert np.isfinite(result).all(), f"{fmt}, expert {e}: non-finite result"
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=5e-4)
    assert ExpertKey("q", 0, 0) in executor.cache.resident
    assert ExpertKey("q", 0, 1) not in executor.cache.resident
    executor.execution_mode = "gpu"
    with pytest.raises(NotImplementedError):
        executor.forward(0, 1, xp.asarray(x))
