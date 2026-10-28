"""0102H: packed projection gradients, weight refresh, and EOS scatter parity."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream, xp
from mini_llm.config import DualResolutionConfig
from mini_llm.ops.dual_resolution_attention import (
    DualResolutionAttention,
    build_dual_resolution_metadata,
)


def _attention(seed=123, dtype="float64"):
    config = DualResolutionConfig(
        enabled=True, block_size=4, local_window=8,
        compression_dim=16, compression_heads=2, compression_depth=2,
        max_blocks=12, global_query_chunk=6,
    )
    return DualResolutionAttention(
        d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        input_std=0.07, output_std=0.07,
        rng=RandomStream(seed), dtype=dtype, config=config,
    )


def _case():
    # Two rows, unequal complete-block counts, EOS boundaries, partial block.
    # Direct index assignment must not write into invalid slot placeholders.
    tokens = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 17, 18],
        [1, 2, 3, 4, 5, 6, 0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ])
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=4)
    return metadata


def _run(model, metadata, x, dy):
    output, cache = model.forward(x, dual_resolution_metadata=metadata)
    model.zero_grad()
    dx = model.backward(dy, cache)
    grads = [p.grad.copy() for p in model.parameters()]
    return output, dx, grads


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_packed_reference_full_attention_and_gradients(monkeypatch, dtype):
    if BACKEND_NAME != "numpy":
        pytest.skip("Deterministic FP32/FP64 reference parity is a NumPy test")
    metadata = _case()
    rng = np.random.default_rng(21)
    x = rng.normal(size=(2, 18, 16)).astype(dtype)
    dy = rng.normal(size=(2, 18, 16)).astype(dtype)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    reference = _attention(dtype=dtype)
    old_y, old_dx, old_grads = _run(reference, metadata, x, dy)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    optimized = _attention(dtype=dtype)
    new_y, new_dx, new_grads = _run(optimized, metadata, x, dy)
    atol = 1e-5 if dtype == "float32" else 1e-10
    rtol = 1e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(new_y, old_y, rtol=rtol, atol=atol)
    np.testing.assert_allclose(new_dx, old_dx, rtol=rtol, atol=atol)
    assert len(new_grads) == len(old_grads)
    for p, new, old in zip(optimized.parameters(), new_grads, old_grads):
        np.testing.assert_allclose(new, old, rtol=rtol, atol=atol, err_msg=p.name)


def test_packed_gradients_accumulate_and_refresh_after_weight_update(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("NumPy reference test")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    metadata = _case()
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 18, 16)).astype("float32")
    dy = rng.normal(size=(2, 18, 16)).astype("float32")
    model = _attention(dtype="float32")
    model.refresh_compute_buffers()
    _, cache = model.forward(x, dual_resolution_metadata=metadata)
    model.zero_grad()
    model.backward(dy, cache)
    grads_once = [p.grad.copy() for p in model.parameters()]
    # A second backward adds contributions to the same canonical FP32 grads.
    model.backward(dy, cache)
    for p, old in zip(model.parameters(), grads_once):
        np.testing.assert_allclose(p.grad, old * 2, rtol=1e-5, atol=1e-5)
    model.Wq.data[...] += 0.01
    model.compression_layers[0]["Wk"].data[...] -= 0.01
    model.refresh_compute_buffers()
    new_output, _ = model.forward(x, dual_resolution_metadata=metadata)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    ref_output, _ = model.forward(x, dual_resolution_metadata=metadata)
    np.testing.assert_allclose(new_output, ref_output, rtol=1e-5, atol=1e-5)


def test_compressor_scatter_no_invalid_slot_ghost_gradient(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("NumPy reference test")
    metadata = _case()
    rng = np.random.default_rng(27)
    x = rng.normal(size=(2, 18, 16))
    upstream = rng.normal(size=(2, metadata["block_valid"].shape[1], 16))
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    ref = _attention(dtype="float64")
    _, cache = ref._compress_forward(x, ref._prepare_metadata(metadata, 2, 18, None))
    ref_dx = ref._compress_backward(upstream, cache, True)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    fast = _attention(dtype="float64")
    _, cache = fast._compress_forward(x, fast._prepare_metadata(metadata, 2, 18, None))
    fast_dx = fast._compress_backward(upstream, cache, True)
    np.testing.assert_allclose(fast_dx, ref_dx, rtol=1e-10, atol=1e-10)
    for p, other in zip(fast.parameters(), ref.parameters()):
        np.testing.assert_allclose(p.grad, other.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy/CUDA")
def test_bf16_fast_projections_tiny_gpu(monkeypatch):
    """Hardware validation: native BF16 packed GEMM, Triton, EOS masking."""
    import cupy as cp
    pytest.importorskip("triton")
    # This deliberately tiny d_head=8 model is a BF16 projection parity test.
    # All three specialized Triton kernels require head width 64; those are
    # covered separately by the production-width integration test below.
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL", "cupy")
    metadata = _case()
    rng = np.random.default_rng(31)
    dtype = cp.dtype("bfloat16")
    x = cp.asarray(rng.normal(size=(2, 18, 16)).astype("float32")).astype(dtype)
    dy = cp.asarray(rng.normal(size=(2, 18, 16)).astype("float32"))
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    base = _attention(dtype="bfloat16")
    out_ref, dx_ref, grads_ref = _run(base, metadata, x, dy)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    fast = _attention(dtype="bfloat16")
    out, dx, grads = _run(fast, metadata, x, dy)
    # Numerical differences are expected from intentionally BF16 projection
    # gradients; guard for gross errors / nonfinite values, not bitwise parity.
    cp.testing.assert_allclose(out.astype(cp.float32), out_ref.astype(cp.float32), rtol=0.08, atol=0.08)
    cp.testing.assert_allclose(dx.astype(cp.float32), dx_ref.astype(cp.float32), rtol=0.13, atol=0.13)
    for p, grad, grad_ref in zip(fast.parameters(), grads, grads_ref):
        assert bool(cp.all(cp.isfinite(grad))), p.name
        cp.testing.assert_allclose(grad, grad_ref, rtol=0.15, atol=0.15)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy/CUDA")
def test_production_width_all_three_triton_fast_projection_gpu(monkeypatch):
    """All 3 Triton attention paths + real 1024d/512d packed projection parity."""
    pytest.importorskip("triton")
    import cupy as cp
    config = DualResolutionConfig(
        enabled=True, block_size=128, local_window=512,
        compression_dim=512, compression_heads=8,
        compression_depth=1, max_blocks=8,
        global_query_chunk=128, recompute_attention_probs=True,
    )
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL", "triton")
    def model():
        return DualResolutionAttention(
            d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
            input_std=0.01, output_std=0.01, rng=RandomStream(101),
            dtype="bfloat16", config=config,
        )
    tokens = np.full((2, 257), 3, dtype=np.int32)
    tokens[0, 255] = 0  # documents with different summary counts
    tokens[1, 127] = 0
    tokens[1, 192] = 0
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=128)
    rng = np.random.default_rng(32)
    dtype = cp.dtype("bfloat16")
    x = cp.asarray(rng.normal(scale=0.05, size=(2, 257, 1024)).astype("float32")).astype(dtype)
    dy = cp.asarray(rng.normal(scale=0.04, size=(2, 257, 1024)).astype("float32")).astype(dtype)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    reference = model()
    old_y, old_dx, old_grads = _run(reference, metadata, x, dy)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    fast = model()
    new_y, new_dx, new_grads = _run(fast, metadata, x, dy)
    cp.testing.assert_allclose(new_y.astype(cp.float32), old_y.astype(cp.float32), atol=0.1, rtol=0.1)
    cp.testing.assert_allclose(new_dx.astype(cp.float32), old_dx.astype(cp.float32), atol=0.15, rtol=0.2)
    for parameter, got, expected in zip(fast.parameters(), new_grads, old_grads):
        assert bool(cp.all(cp.isfinite(got))), parameter.name
        cp.testing.assert_allclose(got, expected, atol=0.2, rtol=0.25)


def test_fast_packed_frozen_inference_does_not_require_grads(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("NumPy reference test")
    from mini_llm.parameter import no_parameter_grad_allocation
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    with no_parameter_grad_allocation():
        model = _attention(dtype="float32")
    assert all(p.grad is None for p in model.parameters())
    model.refresh_compute_buffers()
    output = model.forward(np.zeros((1, 9, 16), dtype="float32"), return_cache=False)
    assert output.shape == (1, 9, 16)


def test_0102i_compressor_bf16_toggle(monkeypatch):
    """The compressor dtype can be isolated from packed GEMM selection."""
    from mini_llm.ops.dual_resolution_attention import _bf16_compressor_qkv_enabled
    monkeypatch.delenv("MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV", raising=False)
    assert _bf16_compressor_qkv_enabled()
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV", "0")
    assert not _bf16_compressor_qkv_enabled()
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV", "1")
    assert _bf16_compressor_qkv_enabled()


def test_0102i_warm_benchmark_cli():
    from benchmarks.benchmark_dual_resolution_full_model_0102d import parse_args
    args = parse_args(['--warmup-iterations', '2', '--repeat-iterations', '3'])
    assert args.warmup_iterations == 2
    assert args.repeat_iterations == 3
    import pytest
    with pytest.raises(SystemExit):
        parse_args(['--repeat-iterations', '0'])
