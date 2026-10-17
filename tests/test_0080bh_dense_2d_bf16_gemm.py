"""0080BH: dense FFN must avoid CuPy's unsupported BF16 batched @ path."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.blocks.dense_ffn_block import DenseFFN
from mini_llm.blocks.dense_ffn_block_inference import DenseFFNInference


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU reference fixture")
def test_dense_training_and_inference_use_rank2_expert_gemms(monkeypatch, recompute):
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", str(int(recompute)))
    ffn = DenseFFN(8, 16, 0.1, 0.1, RandomStream(123), dtype="float64")
    inference = DenseFFNInference(8, 16, dtype="float64")
    inference.set_weights(ffn)
    x = np.random.default_rng(12).normal(size=(2, 3, 8))
    upstream = np.random.default_rng(13).normal(size=x.shape)

    ranks = []
    orig_forward = ffn.expert.forward
    orig_backward = ffn.expert.backward
    orig_infer = inference.expert.forward

    def wrapped_forward(data, *args, **kwargs):
        ranks.append(("forward", data.ndim))
        return orig_forward(data, *args, **kwargs)

    def wrapped_backward(data, cache, *args, **kwargs):
        ranks.append(("backward", data.ndim, cache["x"].ndim))
        return orig_backward(data, cache, *args, **kwargs)

    def wrapped_infer(data, *args, **kwargs):
        ranks.append(("inference", data.ndim))
        return orig_infer(data, *args, **kwargs)

    monkeypatch.setattr(ffn.expert, "forward", wrapped_forward)
    monkeypatch.setattr(ffn.expert, "backward", wrapped_backward)
    monkeypatch.setattr(inference.expert, "forward", wrapped_infer)

    y, cache = ffn.forward(x)
    dx = ffn.backward(upstream, cache)
    assert y.shape == x.shape
    assert dx.shape == x.shape
    assert ffn.forward(x, return_cache=False).shape == x.shape
    assert inference.prefill(x).shape == x.shape
    assert inference.decode_one(x[:, :1]).shape == (2, 1, 8)
    assert np.allclose(inference.prefill(x), y, rtol=1e-12, atol=1e-12)
    assert ranks == [
        ("forward", 2), ("backward", 2, 2), ("forward", 2),
        ("inference", 2), ("inference", 2), ("inference", 2),
    ]


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU reference fixture")
def test_dense_forward_backward_preserves_2d_shape():
    ffn = DenseFFN(8, 16, 0.1, 0.1, RandomStream(456), dtype="float64")
    x = np.random.default_rng(6).normal(size=(7, 8))
    dy = np.random.default_rng(7).normal(size=x.shape)
    y, cache = ffn.forward(x)
    assert y.shape == x.shape
    assert ffn.backward(dy, cache).shape == x.shape


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy/CUDA")
def test_cuda_bf16_dense_ffn_training_and_inference(monkeypatch, recompute):
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", str(int(recompute)))
    from mini_llm.backend import xp
    from mini_llm.backend import validate_bfloat16_backend
    validate_bfloat16_backend()  # 2D BF16 GEMMs must work on the target GPU
    ffn = DenseFFN(64, 128, 0.02, 0.02, RandomStream(456), dtype="bfloat16")
    inference = DenseFFNInference(64, 128, dtype="bfloat16")
    inference.set_weights(ffn)
    x = xp.ones((2, 8, 64), dtype=ffn.expert.W_gate.data.dtype)
    dy = xp.ones_like(x)
    y, cache = ffn.forward(x)
    dx = ffn.backward(dy, cache)
    assert y.shape == x.shape
    assert dx.shape == x.shape
    assert bool(xp.all(xp.isfinite(y)).item())
    assert bool(xp.all(xp.isfinite(dx)).item())
    assert inference.prefill(x).shape == x.shape
    assert inference.decode_one(x[:, :1]).shape == (2, 1, 64)
    assert any(bool(xp.any(p.grad != 0).item()) for p in ffn.parameters())
