"""0080BG: parameter-matched dense vs MoE AR reference."""
import json
from dataclasses import replace
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.config import ModelConfig, MemoryContextConfig
from mini_llm.blocks.dense_ffn_block import DenseFFN
from mini_llm.blocks.dense_ffn_block_inference import DenseFFNInference
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from train_model import model_config_from_name, parse_args


def test_two_reference_parameter_budgets():
    moe = ModelConfig.wide24_500m_memory_64k()
    dense = ModelConfig.wide24_500m_dense_memory_64k()
    assert moe.estimated_parameter_count() == 501_212_208
    assert dense.estimated_parameter_count() == 501_163_008
    assert (dense.ffn_mode, dense.n_experts, dense.top_k, dense.d_ff) == ("dense", 1, 1, 4096)
    assert (moe.ffn_mode, moe.n_experts, moe.top_k, moe.d_ff) == ("moe", 2, 1, 2048)
    for field in ('n_layers', 'd_model', 'n_q_heads', 'n_kv_heads', 'd_head', 'memory_context', 'attention_layers'):
        assert getattr(dense, field) == getattr(moe, field)
    assert model_config_from_name('wide24-500m-dense-memory-64k') == dense
    assert model_config_from_name('wide24-500m-memory-64k') == moe


def test_dense_model_config_roundtrip():
    from dataclasses import asdict
    dense = ModelConfig.wide24_500m_dense_memory_64k()
    roundtrip = ModelConfig(**json.loads(json.dumps(asdict(dense))))
    assert roundtrip == dense
    with pytest.raises(ValueError, match='ffn_mode'):
        replace(dense, ffn_mode='unknown')
    with pytest.raises(ValueError, match='n_experts=1'):
        replace(dense, n_experts=2)


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy reference finite differences')
def test_dense_ffn_gradient_and_inference():
    ffn = DenseFFN(6, 9, 0.15, 0.14, RandomStream(17), dtype='float64')
    x = np.random.default_rng(4).normal(size=(2, 3, 6))
    dy = np.random.default_rng(7).normal(size=(2, 3, 6))
    out, cache = ffn.forward(x)
    dx = ffn.backward(dy, cache)
    assert out.shape == x.shape
    assert dx.shape == x.shape
    eps = 1e-5
    x2 = x.copy(); x2[0, 0, 0] += eps
    x3 = x.copy(); x3[0, 0, 0] -= eps
    fd = np.sum((ffn.forward(x2, False) - ffn.forward(x3, False)) * dy) / (2*eps)
    assert np.allclose(dx[0,0,0], fd, rtol=1e-4, atol=1e-7)
    p = ffn.parameters()[0]
    old = float(p.data[0,0]); p.data[0,0] = old + eps
    plus = np.sum(ffn.forward(x, False)*dy)
    p.data[0,0] = old - eps
    minus = np.sum(ffn.forward(x, False)*dy)
    p.data[0,0] = old
    assert np.allclose(p.grad[0,0], (plus-minus)/(2*eps), rtol=1e-4, atol=1e-7)
    inf = DenseFFNInference(6, 9, dtype='float64'); inf.set_weights(ffn)
    assert np.allclose(inf.prefill(x), out)
    assert np.allclose(inf.decode_one(x[:, :1]), ffn.forward(x[:, :1], False))


@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU reference test')
def test_tiny_dense_24_layer_forward_and_backward():
    base = ModelConfig.wide24_500m_dense_memory_64k()
    cfg = replace(base, tokenizer_vocab_size=41, d_model=32, n_q_heads=16,
                  n_kv_heads=4, d_head=2, d_ff=48, attention_layers=None,
                  context_length=12, memory_context=MemoryContextConfig(enabled=False))
    model = DecoderLanguageModel(cfg, rng_seed=11, dtype='float64')
    assert len(model.blocks) == 24
    assert all(not hasattr(b.moe, 'router') for b in model.blocks)
    toks = np.array([[1, 2, 3, 4]], dtype=np.int64)
    model.zero_grad()
    h, cache = model.forward_body(toks)
    model.backward_body(np.full_like(h, 0.01), cache)
    for index in (0, 11, 23):
        assert any(np.any(p.grad != 0) for p in model.blocks[index].moe.parameters())
    assert len(model.parameters()) == sum(len(b.parameters()) for b in model.blocks) + 3
    assert cfg.estimated_parameter_count() == sum(int(p.data.size) for p in model.parameters())
    inf = InferenceModel(cfg, dtype="float64")
    inf.set_weights(model)
    assert all(not hasattr(b.moe, "router") for b in inf.blocks)
    state = inf.create_generation_state(1, 12)
    pred = inf.prefill(toks, state)
    assert np.asarray(pred).shape[-1] == cfg.vocab_size

