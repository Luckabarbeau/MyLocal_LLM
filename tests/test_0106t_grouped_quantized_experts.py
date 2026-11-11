"""Grouped packed FFN: CPU launch-contract oracle, CUDA parity and bounded timing.

Run GPU tests with MINI_LLM_BACKEND=cupy. The opt-in benchmark uses
MINI_LLM_GROUPED_BENCHMARK=1 and only one warmup plus ten measured forwards.
"""

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mini_llm.ops import grouped_quantized_experts as grouped
from mini_llm.ops import quantized_linear as ql
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_0106c_fused_packed_expert import make_quant


FORMATS = [(fmt, fmt, fmt) for fmt in ("Q8_0", "Q4_K", "Q5_K", "Q6_K")]
FORMATS += [("Q4_K", "Q5_K", "Q6_K"), ("Q3_K", "Q4_0", "Q8_0")]


def _weight(fmt, n, k, seed):
    # Reuse existing packed-byte fixtures; small exact binary scales make all
    # formats finite and keep the complete FP32 FFN well conditioned.
    w = make_quant(fmt, n, k, seed)
    raw = w.data.reshape(-1, ql.BLOCK_FORMATS[fmt][1])
    offset = {"Q3_K": 108, "Q6_K": 208}.get(fmt, 0)
    raw[:, offset:offset + 2] = np.frombuffer(np.float16(1 / 1024).tobytes(), np.uint8)
    if fmt in ("Q4_K", "Q5_K"):
        raw[:, 2:4] = np.frombuffer(np.float16(1 / 2048).tobytes(), np.uint8)
    return w


def _hosts(formats, d=256, f=256, count=3):
    return [ql.PackedExpertFFNInference(*[
        _weight(fmt, n, k, 10600 + expert * 3 + projection)
        for projection, (fmt, n, k) in enumerate(zip(formats, (f, f, d), (d, d, f)))
    ]) for expert in range(count)]


def _dense_fp64(x, handles, routes):
    """Independent dense FP64 projections; honor the FP32 storage boundaries."""
    out = np.zeros(x.shape, np.float32)
    for handle, route in zip(handles, routes):
        g, u, down = [w.dequantize_reference().astype(np.float64)
                      for w in (handle.gate, handle.up, handle.down)]
        gate, up = x.astype(np.float64) @ g.T, x.astype(np.float64) @ u.T
        hidden = ((gate / (1 + np.exp(-gate))) * up).astype(np.float32)
        result = (hidden.astype(np.float64) @ down.T).astype(np.float32)
        term = np.multiply(result, route, dtype=np.float32)
        np.add(out, term, out=out)
    return out


def _sequential(cp, x, handles, routes):
    out = cp.zeros_like(x)
    for i, handle in enumerate(handles):
        # Separate ufunc calls enforce the same multiply/store/add graph.
        term = handle.forward(x) * routes[i]
        out += term
    return out


class _CPUDeviceArray(np.ndarray):
    """CPU-only tensor protocol stub; never passed to an actual CUDA kernel."""

    @property
    def data(self) -> Any:
        return SimpleNamespace(ptr=self.__array_interface__["data"][0])

    @property
    def device(self) -> Any:
        return SimpleNamespace(id=0)


@pytest.fixture
def simulated_gpu(monkeypatch):
    """Exercise real validation, pointer-table construction and launch wiring.

    Only kernel execution is replaced with existing CPU packed references. The
    CUDA tests below exercise the unmodified JIT code, not this simulation.
    """
    transfers, calls, lookup = [], [], {}

    def array(value, dtype=None):
        value = np.asarray(value, dtype=dtype)
        transfers.append((value.shape, value.dtype))
        return value.copy().view(_CPUDeviceArray)

    cp = SimpleNamespace(
        ndarray=_CPUDeviceArray, float32=np.float32, uint8=np.uint8,
        asarray=array,
        empty=lambda shape, dtype: np.empty(shape, dtype).view(_CPUDeviceArray),
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0),
                             get_current_stream=lambda: SimpleNamespace(ptr=0x106)),
    )
    monkeypatch.setattr(ql, "BACKEND_NAME", "cupy")
    monkeypatch.setattr(ql, "xp", cp)
    monkeypatch.setattr(ql, "triton", object())
    monkeypatch.setattr(ql, "tl", SimpleNamespace(float16="f16", float32="f32",
                                                bfloat16="bf16", uint8="u8"))
    monkeypatch.setattr(grouped, "triton", SimpleNamespace(cdiv=lambda n, k: (n + k - 1) // k),
                        raising=False)
    for name in ("_grouped_gate_up", "_grouped_down_weighted", "_ordered_reduce"):
        monkeypatch.setattr(grouped, name, name, raising=False)

    def device(handles):
        result = []
        for h in handles:
            weights = []
            for w in (h.gate, h.up, h.down):
                data = array(w.data)
                dw = ql.DevicePackedGGUFWeight(data, w.shape, w.quant_type)
                lookup[data.data.ptr] = w
                weights.append(dw)
            result.append(ql.PackedExpertFFNInference(*weights))
        return result

    def launch(kernel, grid, *args, **options):
        calls.append((kernel, grid, args, options))
        pointer_count = {"_grouped_gate_up": 3, "_grouped_down_weighted": 4,
                         "_ordered_reduce": 2}[kernel]
        assert all(isinstance(p, ql._TritonCuPyPointer) for p in args[:pointer_count])
        arrays = [np.asarray(p.array) for p in args[:pointer_count]]
        if kernel == "_grouped_gate_up":
            x, table, hidden = arrays
            pointers = table.view(np.uint64).reshape(-1, options.get('TABLE_WIDTH', 3))
            for e, entry in enumerate(pointers):
                g, u = entry[:2]
                hidden[e] = ql.quantized_swiglu(x, lookup[int(g)], lookup[int(u)])[0]
        elif kernel == "_grouped_down_weighted":
            hidden, table, routes_base, terms = arrays
            pointers = table.view(np.uint64).reshape(-1, options.get('TABLE_WIDTH', 3))
            stride = args[6] * np.dtype(np.float32).itemsize
            routes = np.lib.stride_tricks.as_strided(routes_base,
                shape=(len(pointers),), strides=(stride,))
            for e, entry in enumerate(pointers):
                down = entry[2]
                slot = int(entry[3]) if len(entry) == 4 else e
                result = ql.quantized_linear(hidden[e:e + 1], lookup[int(down)])
                terms[e] = np.multiply(result[0], routes[slot], dtype=np.float32)
        else:
            terms, out = arrays
            out.fill(0)
            for term in terms:
                np.add(out, term, out=out)

    monkeypatch.setattr(grouped, "_launch", launch)
    return SimpleNamespace(cp=cp, device=device, calls=calls, transfers=transfers)


@pytest.mark.parametrize("formats", FORMATS)
def test_cpu_launch_graph_matches_dense_fp64(simulated_gpu, formats):
    sim = simulated_gpu
    hosts = _hosts(formats)
    handles = sim.device(hosts)
    # IDs 2, 5, 9 can legally reference the same underlying expert allocation.
    handles, hosts = [handles[0], handles[1], handles[0]], [hosts[0], hosts[1], hosts[0]]
    x_host = np.random.default_rng(106).normal(0, .05, (1, 256)).astype(np.float32)
    routes_host = np.array([.7, -.2, .3], np.float32)
    expected = _dense_fp64(x_host, hosts, routes_host)
    x = sim.cp.asarray(x_host)
    routes = sim.cp.asarray(np.array([.7, 99, -.2, 99, .3, 99], np.float32))[::2]
    before = [w.data.copy() for h in handles for w in (h.gate, h.up, h.down)]
    sim.transfers.clear()
    assert grouped.supports_grouped_quantized_experts(x, handles, routes)
    assert not sim.transfers and not sim.calls
    actual = grouped.grouped_quantized_experts_forward(x, handles, routes)
    # FP64 GEMV oracle vs FP32 GEMV/activation, following established packed FFN tolerances.
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=8e-6)
    assert actual.shape == (1, 256) and actual.dtype == np.float32
    assert not np.shares_memory(actual, x)
    np.testing.assert_array_equal(x, x_host)
    np.testing.assert_array_equal(routes, routes_host)
    for old, w in zip(before, (w for h in handles for w in (h.gate, h.up, h.down))):
        np.testing.assert_array_equal(old, w.data)
    assert sim.transfers == [((3, 3), np.dtype("uint64"))]
    assert [call[1] for call in sim.calls] == [(1, 256, 3), (256, 1, 3), (1, 1, 1)]
    assert sim.calls[-1][3] == {"enable_fp_fusion": False}


@pytest.mark.parametrize("case,reason", [
    ("batch", "contiguous"), ("noncontiguous", "contiguous"), ("dtype", "FP32"),
    ("routes_shape", "routing_weights"), ("routes_dtype", "routing_weights"),
    ("empty", "nonempty"), ("handle", "handles"), ("host", "DevicePacked"),
    ("mixed", "per-projection"), ("shapes", "shapes"), ("block", "block_k"),
])
def test_cpu_rejects_unsupported_without_launch(simulated_gpu, case, reason):
    sim = simulated_gpu
    x = sim.cp.asarray(np.zeros((1, 256), np.float32))
    hosts = _hosts(FORMATS[0], count=1)
    handles = sim.device(hosts)
    routes = sim.cp.asarray(np.ones(1, np.float32))
    kwargs = {}
    if case == "batch":
        x = sim.cp.asarray(np.zeros((2, 256), np.float32))
    elif case == "noncontiguous":
        x = sim.cp.asarray(np.zeros((1, 512), np.float32))[:, ::2]
    elif case == "dtype":
        x = x.astype(np.float16)
    elif case == "routes_shape":
        routes = routes.reshape(1, 1)
    elif case == "routes_dtype":
        routes = routes.astype(np.float64)
    elif case == "empty":
        handles = []
    elif case == "handle":
        handles = [object()]
    elif case == "host":
        handles = hosts
    elif case == "mixed":
        handles += sim.device(_hosts(FORMATS[1], count=1))
        routes = sim.cp.asarray(np.ones(2, np.float32))
    elif case == "shapes":
        x = sim.cp.asarray(np.zeros((1, 512), np.float32))
    elif case == "block":
        kwargs["block_k"] = 128
    sim.transfers.clear()
    assert not grouped.supports_grouped_quantized_experts(x, handles, routes, **kwargs)
    with pytest.raises(ValueError, match=reason):
        grouped.grouped_quantized_experts_forward(x, handles, routes, **kwargs)
    assert not sim.calls and not sim.transfers


def test_cpu_backend_has_no_gpu_fallback():
    if ql.BACKEND_NAME != "numpy":
        pytest.skip("NumPy backend contract")
    x, routes = np.zeros((1, 256), np.float32), np.ones(1, np.float32)
    assert not grouped.supports_grouped_quantized_experts(x, _hosts(FORMATS[0], count=1), routes)
    with pytest.raises(ValueError, match="CuPy backend and Triton"):
        grouped.grouped_quantized_experts_forward(x, [], routes)


def test_explicit_launch_uses_current_cupy_stream(monkeypatch):
    calls = []

    class Compiled:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append((grid, args, kwargs))
            return run

    class Kernel:
        def warmup(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Compiled()

    monkeypatch.setattr(ql, "xp", SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(ptr=0x106106))))
    grouped._launch(Kernel(), (2, 3, 4), "operand", enable_fp_fusion=False)
    assert calls == [(("operand",), {"grid": (2, 3, 4), "num_warps": 4,
                                   "enable_fp_fusion": False}),
                     ((2, 3, 4), ("operand",), {"stream": 0x106106})]


@pytest.fixture
def cuda():
    if ql.BACKEND_NAME != "cupy":
        pytest.skip("Requires MINI_LLM_BACKEND=cupy")
    pytest.importorskip("triton")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA GPU")
    return cp


def _upload(hosts) -> list[Any]:
    return [ql.PackedExpertFFNInference(*[ql.packed_to_device(w)
            for w in (h.gate, h.up, h.down)]) for h in hosts]


def _report(label, actual, expected):
    error = float(np.max(np.abs(actual - expected)))
    scale = max(1., float(np.max(np.abs(expected))))
    print(f"{label}: max_abs={error:.9g}, max_abs/scale={error / scale:.9g}, scale={scale:.9g}")
    return error, scale


@pytest.mark.parametrize("formats", FORMATS)
@pytest.mark.parametrize("block_k", [256, 512, 1024])
def test_cuda_grouped_matches_sequential_and_fp64(cuda, formats, block_k, monkeypatch):
    cp = cuda
    hosts = _hosts(formats, d=768, f=256)
    x_host = np.random.default_rng(107).normal(0, .05, (1, 768)).astype(np.float32)
    routes_host = np.array([.7, -.2, .3, 0], np.float32)
    selected = [hosts[0], hosts[1], hosts[0], hosts[2]]
    oracle = _dense_fp64(x_host, selected, routes_host)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        handles = _upload(hosts)
        handles = [handles[0], handles[1], handles[0], handles[2]]
        x = cp.asarray(x_host)
        # Actual sliced routing weights, created on the nondefault stream.
        routes = cp.asarray(np.repeat(routes_host, 2))[::2]
        before = [w.data.copy() for h in handles for w in (h.gate, h.up, h.down)]
        expected = _sequential(cp, x, handles, routes)
        monkeypatch.setattr(PackedGGUFWeight, "dequantize_reference",
                            lambda _: pytest.fail("Grouped GPU path must not expand weights"))
        # No host reference or per-expert FFN invocation is allowed here.
        monkeypatch.setattr(ql.PackedExpertFFNInference, "forward",
                            lambda *_: pytest.fail("Must group, not dispatch per expert"))
        assert grouped.supports_grouped_quantized_experts(x, handles, routes, block_k=block_k)
        actual = grouped.grouped_quantized_experts_forward(x, handles, routes, block_k=block_k)
        got = actual.get(stream=stream)
        reference = expected.get(stream=stream)
        for old, w in zip(before, (w for h in handles for w in (h.gate, h.up, h.down))):
            np.testing.assert_array_equal(old.get(stream=stream), w.data.get(stream=stream))
        np.testing.assert_array_equal(x.get(stream=stream), x_host)
        np.testing.assert_array_equal(routes.get(stream=stream), routes_host)
    assert actual.shape == (1, 768) and actual.dtype == cp.float32
    assert actual.data.ptr != x.data.ptr
    error, scale = _report(f"{formats}, BK={block_k}, vs sequential", got, reference)
    # Same GPU decoder/FP32 graph; allow tiling-dependent GEMV accumulation.
    assert error <= 1e-5 * scale
    _report(f"{formats}, BK={block_k}, vs FP64", got, oracle)
    np.testing.assert_allclose(got, oracle, rtol=4e-4, atol=8e-6)


def test_cuda_ordered_reduce_is_not_tree_or_fma(cuda):
    cp = cuda
    # Cancellation distinguishes sequential FP32 summation from a tree.
    terms_host = np.array([[1e8], [1], [-1e8], [3]], np.float32)
    terms = cp.asarray(np.repeat(terms_host, 257, axis=1))
    out = cp.empty((1, 257), cp.float32)
    grouped._launch(grouped._ordered_reduce, (2, 1, 1),
        grouped.as_triton_tensor(terms), grouped.as_triton_tensor(out), 257, 4, 256,
        enable_fp_fusion=False)
    np.testing.assert_array_equal(out.get(), np.full((1, 257), 3, np.float32))
    # Storage-rounded product + add differs from a fused double-precision product/add.
    a, b = np.float32(1 + 2 ** -23), np.float32(1 - 2 ** -23)
    stored = np.float32(a * b)
    assert np.float32(float(a) * float(b) - 1) != np.float32(stored - np.float32(1))
    terms = cp.asarray(np.array([[-1], [stored]], np.float32))
    out = cp.empty((1, 1), cp.float32)
    grouped._launch(grouped._ordered_reduce, (1, 1, 1),
        grouped.as_triton_tensor(terms), grouped.as_triton_tensor(out), 1, 2, 256,
        enable_fp_fusion=False)
    np.testing.assert_array_equal(out.get(), [[0]])


@pytest.mark.parametrize("slice_kind", ["reverse", "broadcast", "singleton"])
def test_cuda_route_views_and_repeated_handles(cuda, slice_kind):
    cp = cuda
    handles = _upload(_hosts(FORMATS[0], count=1)) * 3
    x = cp.asarray(np.random.default_rng(108).normal(0, .05, (1, 256)).astype(np.float32))
    if slice_kind == "reverse":
        routes = cp.asarray([.2, -.3, .7], dtype=cp.float32)[::-1]
    elif slice_kind == "broadcast":
        routes = cp.broadcast_to(cp.asarray([.25], dtype=cp.float32), (3,))
    else:
        handles = handles[:1]
        routes = cp.asarray([1.], dtype=cp.float32)
    expected = _sequential(cp, x, handles, routes)
    got = grouped.grouped_quantized_experts_forward(x, handles, routes)
    error, scale = _report(slice_kind, got.get(), expected.get())
    assert error <= 1e-5 * scale


@pytest.mark.skipif(os.environ.get("MINI_LLM_GROUPED_BENCHMARK") != "1",
                    reason="Opt-in bounded CUDA event benchmark")
def test_cuda_bounded_latency(cuda):
    cp = cuda
    d, f, e, repeats = 2048, 512, 10, 10
    handles = _upload(_hosts(("Q4_K", "Q5_K", "Q6_K"), d=d, f=f, count=e))
    x = cp.asarray(np.random.default_rng(109).normal(0, .01, (1, d)).astype(np.float32))
    routes = cp.asarray(np.linspace(.01, .1, e, dtype=np.float32))
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()  # uploads precede isolated stream

    def measure(fn):
        with stream:
            warm = fn()
        stream.synchronize()
        result = warm
        starts, ends = [cp.cuda.Event() for _ in range(repeats)], [cp.cuda.Event() for _ in range(repeats)]
        with stream:
            for start, end in zip(starts, ends):
                start.record(stream)
                result = fn()
                end.record(stream)
        stream.synchronize()
        times = [cp.cuda.get_elapsed_time(a, b) for a, b in zip(starts, ends)]
        return result.get(), float(np.median(times)), float(np.min(times)), warm

    grouped_result, grouped_ms, grouped_min, _ = measure(
        lambda: grouped.grouped_quantized_experts_forward(x, handles, routes))
    sequential_result, sequential_ms, sequential_min, _ = measure(
        lambda: _sequential(cp, x, handles, routes))
    error, scale = _report("D=2048 F=512 E=10", grouped_result, sequential_result)
    assert error <= 1e-5 * scale
    print(f"CUDA events, one warmup + {repeats} repeats, D={d} F={f} E={e}: "
          f"grouped median/min={grouped_ms:.6f}/{grouped_min:.6f} ms; "
          f"sequential median/min={sequential_ms:.6f}/{sequential_min:.6f} ms; "
          f"speedup={sequential_ms / grouped_ms:.3f}x. "
          "Forward-inclusive: allocations, pointer upload, and host launch gaps included; no speed assertion.")
