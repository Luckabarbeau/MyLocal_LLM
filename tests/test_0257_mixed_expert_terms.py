"""Patch0257 building blocks only: exact terms, two-launch mixed combine."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import iq_packed_gpu as iq
from mini_llm.ops import mixed_expert_combine_gpu as mixed
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0231_grouped_iq_gpu import handles
from test_0228_iq_gpu_integration import forbidden


@pytest.fixture(scope="module")
def cp():
    if BACKEND_NAME != "cupy":
        pytest.skip("requires MINI_LLM_BACKEND=cupy")
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def test_heterogeneous_mixed_group_integration(cp, monkeypatch):
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor, ExpertKey
    hosts, experts = handles(('IQ3_S', 'IQ3_S', 'IQ4_NL'), count=3, d=768, f=256)
    executor = HeterogeneousExpertExecutor(lambda key: hosts[key.expert_id], model_id='0257',
        d_model=768, d_ff=256, compute_dtype='float32', cpu_workers=2,
        cpu_packed_backend='native', group_mixed_decode=True)
    monkeypatch.setattr(executor, '_prepare', lambda layer, e, x:
        (ExpertKey('0257', layer, e), experts[e] if e != 1 else None,
         hosts[e] if e == 1 else None, None))
    x = cp.asarray(np.random.default_rng(257).normal(0, .1, (1,768)).astype(np.float32))
    routes = cp.asarray([.2, .3, .5], dtype=cp.float32)
    try:
        with executor.dispatch_scope(0, (0,1,2)):
            actual = executor.try_forward_decode_batch(0, (0,1,2), x, routes, (2,0,1))
        assert executor.counters['mixed_grouped_gpu_dispatches'] == 1
        executor.group_mixed_decode = False
        with executor.dispatch_scope(0, (0,1,2)):
            expected = executor.try_forward_decode_batch(0, (0,1,2), x, routes, (2,0,1))
        a, b = cp.asnumpy(actual), cp.asnumpy(expected)
        assert np.max(np.abs(a-b)) <= 1e-7 + 2e-5*np.max(np.abs(b))
        assert executor.counters['native_cpu_experts'] == 2
        assert not executor.cache._protected
    finally:
        executor.close()


def bits_equal(cp, actual, expected):
    np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),
                                  cp.asnumpy(expected).view(np.uint32))


def ordered(cp, gpu, cpu, routes, positions, slots):
    out = cp.zeros((1, gpu.shape[1]), cp.float32)
    for p, s in zip(positions, slots):
        term = gpu[p:p + 1] if p >= 0 else cpu[-p - 1:-p] * routes[s]
        out += term
    return out


@pytest.mark.parametrize("kind", ["contiguous", "sliced", "negative", "zero"])
def test_mixed_mapping_strides_stream(cp, kind):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        gpu = cp.arange(2 * 769, dtype=cp.float32).reshape(2, 769) / cp.float32(7)
        cpu = -gpu.copy()
        base = cp.asarray(np.array([.5, 99, -2, 99, .25, 99, 3, 99], np.float32))
        routes = base[::2]
        if kind == "contiguous":
            routes = routes.copy()
        elif kind == "negative":
            routes = routes[::-1]
        elif kind == "zero":
            routes = cp.broadcast_to(base[:1], (4,))
        positions, slots = (-2, 1, -1, 0), (2, 0, 3, 1)
        expected = ordered(cp, gpu, cpu, routes, positions, slots)
        got = mixed.combine(gpu, cpu, routes, positions, slots)
        consumed = got * cp.float32(1.5)
        reference = expected * cp.float32(1.5)
        assert got.shape == (1, 769) and got.dtype == cp.float32
        assert got.data.ptr not in (gpu.data.ptr, cpu.data.ptr)
    stream.synchronize()
    bits_equal(cp, consumed, reference)


def test_cancellation_order_and_no_double_weight(cp):
    gpu = cp.full((1, 257), 2**25, cp.float32)
    cpu = cp.stack((cp.full(257, -(2**24), cp.float32), cp.ones(257, cp.float32)))
    routes = cp.asarray(np.array([7, 2, 1], np.float32))
    got = mixed.combine(gpu, cpu, routes, (0, -1, -2), (0, 1, 2))
    bits_equal(cp, got, ordered(cp, gpu, cpu, routes, (0, -1, -2), (0, 1, 2)))
    bits_equal(cp, got, cp.ones((1, 257), cp.float32))
    alternative = mixed.combine(gpu, cpu, routes, (0, -2, -1), (0, 2, 1))
    bits_equal(cp, alternative, cp.zeros((1, 257), cp.float32))


def track_module(module, calls):
    class Tracked:
        def get_function(self, name):
            kernel = module.get_function(name)
            def launch(grid, block, args):
                calls.append(name)
                return kernel(grid, block, args)
            return launch
    return Tracked()


def track_memory(cp, monkeypatch):
    allocations, uploads = [], []
    empty, asarray = cp.empty, cp.asarray
    def allocate(shape, dtype):
        allocations.append((shape, np.dtype(dtype)))
        return empty(shape, dtype)
    def upload(value, *args, **kwargs):
        assert isinstance(value, np.ndarray)
        uploads.append((value.shape, value.dtype))
        return asarray(value, *args, **kwargs)
    monkeypatch.setattr(cp, "empty", allocate)
    monkeypatch.setattr(cp, "asarray", upload)
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
    return allocations, uploads


def test_mixed_two_launches_bounded_upload(cp, monkeypatch):
    gpu, cpu = cp.ones((32, 257), cp.float32), cp.ones((32, 257), cp.float32)
    routes = cp.ones(64, cp.float32)
    positions = tuple(range(32)) + tuple(-i - 1 for i in range(32))
    module, calls = mixed._module(gpu.device.id), []
    with monkeypatch.context() as m:
        m.setattr(mixed, "_module", lambda device: track_module(module, calls))
        allocations, uploads = track_memory(cp, m)
        got = mixed.combine(gpu, cpu, routes, positions, tuple(range(64)))
    assert calls == ["mixed_materialize", "mixed_ordered_reduce"]
    assert uploads == [((64, 2), np.dtype("int32"))]
    assert allocations == [((64, 257), np.dtype("float32")), ((1, 257), np.dtype("float32"))]
    bits_equal(cp, got, cp.full((1, 257), 64, cp.float32))


@pytest.mark.parametrize("case", ["all_cpu", "all_gpu", "dtype", "width", "layout", "rank",
                                  "positions_list", "empty", "limit", "position", "bool",
                                  "slots", "slots_list", "routes_shape", "routes_dtype"])
def test_mixed_invalid_metadata_only(cp, monkeypatch, case):
    gpu, cpu = cp.ones((1, 8), cp.float32), cp.ones((1, 8), cp.float32)
    routes, positions, slots = cp.ones(2, cp.float32), (0, -1), (1, 0)
    if case == "all_cpu":
        gpu = cp.empty((0, 8), cp.float32)
    elif case == "all_gpu":
        cpu = cp.empty((0, 8), cp.float32)
    elif case == "dtype":
        gpu = gpu.astype(cp.float16)
    elif case == "width":
        cpu = cp.ones((1, 7), cp.float32)
    elif case == "layout":
        gpu = gpu[:, ::2]
    elif case == "rank":
        gpu = gpu[0]
    elif case == "positions_list":
        positions = list(positions)
    elif case == "empty":
        positions = ()
    elif case == "limit":
        positions = (0,) * 65
    elif case == "position":
        positions = (0, -2)
    elif case == "bool":
        positions = (False, -1)
    elif case == "slots":
        slots = (0, 2)
    elif case == "slots_list":
        slots = list(slots)
    elif case == "routes_shape":
        routes = routes[:1]
    elif case == "routes_dtype":
        routes = routes.astype(cp.float16)
    monkeypatch.setattr(cp, "empty", forbidden)
    monkeypatch.setattr(cp, "asarray", forbidden)
    monkeypatch.setattr(mixed, "_module", forbidden)
    with pytest.raises(ValueError):
        mixed.combine(gpu, cpu, routes, positions, slots)


@pytest.mark.parametrize("scheduling", ["block", "warp", "warp_specialized"])
@pytest.mark.parametrize("slots", [None, (2, 0, 1), (1, 1, 0)])
def test_grouped_terms_independent_and_prior_grouped(cp, monkeypatch, scheduling, slots):
    monkeypatch.setenv("MINI_LLM_IQ_GPU_GROUPED_KERNEL", scheduling)
    _, experts = handles(("IQ3_S", "IQ3_S", "IQ4_NL"), count=3, d=768, f=256)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        x = cp.full((1, 768), .03125, cp.float32)
        routes = cp.asarray(np.array([.625, -.25, .125], np.float32))[::-1]
        # Block projections share the single-expert reduction tree. Warp modes
        # have a different established tree, so isolate each term with the same
        # scheduling mode rather than imposing block-mode rounding on them.
        expected = cp.concatenate([
            (iq.linear(iq.swiglu(x, h.gate, h.up), h.down)
             * routes[e if slots is None else slots[e]]) if scheduling == "block"
            else iq.grouped(x, [h], routes[e if slots is None else slots[e]:][:1])
            for e, h in enumerate(experts)], axis=0)
        prior = iq.grouped(x, experts, routes, slots)
        reference = cp.zeros_like(x)
        for e in range(3):
            reference += expected[e:e + 1]
        monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference", forbidden)
        got = iq.grouped_terms(x, experts, routes, slots)
    stream.synchronize()
    assert got.shape == (3, 768) and got.dtype == cp.float32
    bits_equal(cp, got, expected)
    bits_equal(cp, prior, reference)
    reduced = cp.zeros_like(x)
    for e in range(3):
        reduced += got[e:e + 1]
    bits_equal(cp, reduced, prior)


def test_grouped_terms_two_launches_no_output(cp, monkeypatch):
    monkeypatch.setenv("MINI_LLM_IQ_GPU_GROUPED_KERNEL", "block")
    _, experts = handles(("IQ3_S", "IQ3_S", "IQ4_NL"), count=3, d=768, f=256)
    x, routes = cp.ones((1, 768), cp.float32), cp.ones(3, cp.float32)
    module, calls = iq._grouped_module(x.device.id), []
    with monkeypatch.context() as m:
        m.setattr(iq, "_grouped_module", lambda device: track_module(module, calls))
        allocations, uploads = track_memory(cp, m)
        got = iq.grouped_terms(x, experts, routes)
    assert got.shape == (3, 768)
    assert calls == ["iq_grouped_gate_up", "iq_grouped_down_weighted"]
    assert allocations == [((3, 256), np.dtype("float32")), ((3, 768), np.dtype("float32"))]
    assert uploads == [((3, 3), np.dtype("uint64"))]


@pytest.mark.parametrize("case", ["cardinality", "slots", "tile", "precision", "legacy"])
def test_grouped_terms_validation(cp, monkeypatch, case):
    formats = ("Q8_0",) * 3 if case == "legacy" else ("IQ3_S", "IQ3_S", "IQ4_NL")
    _, experts = handles(formats, count=3, d=768, f=256)
    x, routes = cp.ones((1, 768), cp.float32), cp.ones(3, cp.float32)
    kwargs = {}
    if case == "cardinality":
        routes = cp.ones(4, cp.float32)
    elif case == "slots":
        kwargs["slots"] = (0, 1, 3)
    elif case == "tile":
        kwargs["block_k"] = 128
    elif case == "precision":
        x = x.astype(cp.float16)
    monkeypatch.setattr(cp, "empty", forbidden)
    monkeypatch.setattr(cp, "asarray", forbidden)
    with pytest.raises(ValueError):
        iq.grouped_terms(x, experts, routes, **kwargs)
