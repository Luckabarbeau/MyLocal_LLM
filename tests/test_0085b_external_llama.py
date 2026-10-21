"""Independent NumPy reference checks for external Llama inference wiring."""

import numpy as np
import pytest

from mini_llm.external import ExternalLlamaConfig, ExternalLlamaModel


def make_config(**overrides):
    cfg = dict(
        model_type="llama", architectures=["LlamaForCausalLM"],
        hidden_size=24, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1, head_dim=8,
        vocab_size=49, max_position_embeddings=128,
        rms_norm_eps=1e-5, rope_theta=100000.0,
        hidden_act="silu", attention_bias=False,
        tie_word_embeddings=True,
    )
    cfg.update(overrides)
    return ExternalLlamaConfig.from_dict(cfg)


def make_weights(model, *, seed=47):
    rng = np.random.default_rng(seed)
    result = {}
    for name, shape in model.expected_shapes().items():
        if name.endswith("norm.weight") or "layernorm.weight" in name:
            result[name] = (np.ones(shape, dtype="float32") +
                            0.025 * rng.normal(size=shape)).astype("float32")
        else:
            result[name] = (rng.normal(size=shape) * 0.07).astype("float32")
    return result


def ref_norm(x, gamma, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def ref_rope(q, base, positions):
    """Independent half-split HF rotate_half math, using NumPy FP32."""
    n = q.shape[-1] // 2
    rates = 1.0 / (base ** (np.arange(n, dtype=np.float32) * 2.0 / (n * 2)))
    phase = np.asarray(positions, dtype=np.float32)[:, None] * rates[None, :]
    cos, sin = np.cos(phase)[:, None, :], np.sin(phase)[:, None, :]
    return np.concatenate([q[..., :n] * cos - q[..., n:] * sin,
                           q[..., :n] * sin + q[..., n:] * cos], axis=-1)


def ref_full_forward(ids, weights, cfg):
    """No calls into MyLocal_LLM attention, normalization or MLP functions."""
    assert ids.shape[0] == 1
    x = weights["model.embed_tokens.weight"][ids[0]].copy()
    t = x.shape[0]
    d = cfg.hidden_size
    hq, hk, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    for i in range(cfg.num_hidden_layers):
        root = f"model.layers.{i}."
        def w(name):
            return weights[root + name]
        xn = ref_norm(x, w("input_layernorm.weight"), cfg.rms_norm_eps)
        q = (xn @ w("self_attn.q_proj.weight").T).reshape(t, hq, hd)
        k = (xn @ w("self_attn.k_proj.weight").T).reshape(t, hk, hd)
        v = (xn @ w("self_attn.v_proj.weight").T).reshape(t, hk, hd)
        q = ref_rope(q, cfg.rope_theta, np.arange(t))
        k = ref_rope(k, cfg.rope_theta, np.arange(t))
        k = np.repeat(k, hq // hk, axis=1)
        v = np.repeat(v, hq // hk, axis=1)
        score = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
        score = np.where(np.tril(np.ones((t, t), dtype=bool))[None], score, -np.inf)
        prob = np.exp(score - score.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        ctx = np.einsum("hts,shd->thd", prob, v)
        x = x + ctx.reshape(t, d) @ w("self_attn.o_proj.weight").T
        xn = ref_norm(x, w("post_attention_layernorm.weight"), cfg.rms_norm_eps)
        gate = xn @ w("mlp.gate_proj.weight").T
        up = xn @ w("mlp.up_proj.weight").T
        silu = gate / (1.0 + np.exp(-gate))
        x = x + (silu * up) @ w("mlp.down_proj.weight").T
    x = ref_norm(x, weights["model.norm.weight"], cfg.rms_norm_eps)
    head = (weights["model.embed_tokens.weight"] if cfg.tie_word_embeddings
            else weights["lm_head.weight"])
    return (x @ head.T)[None]


def test_odd_head_counts_and_unsupported_features():
    for qh, kvh in [(9, 3), (15, 5)]:
        cfg = make_config(hidden_size=qh * 64, num_attention_heads=qh,
                          num_key_value_heads=kvh, head_dim=64)
        assert cfg.num_attention_heads // cfg.num_key_value_heads == 3
    for bad in ({"rope_scaling": {"rope_type": "linear", "factor": 2}},
                {"attention_bias": True}, {"hidden_act": "gelu"},
                {"rope_interleaved": True}, {"pretraining_tp": 2},
                {"num_key_value_heads": 2}, {"head_dim": 12}):
        with pytest.raises(ValueError):
            make_config(**bad)


def test_exact_causal_logits_and_kv_cache_float32():
    cfg = make_config()
    model = ExternalLlamaModel(cfg)
    weights = make_weights(model)
    model.set_weights(weights)
    assert model.engine.blocks[0].norm1.eps == cfg.rms_norm_eps
    assert model.engine.blocks[1].attention.rope_layout == "half"
    assert model.engine.lm_head.base is weights["model.embed_tokens.weight"] or np.array_equal(
        model.engine.lm_head, weights["model.embed_tokens.weight"].T
    )
    ids = np.array([[1, 7, 22, 3, 11, 14, 0]], dtype=np.int32)
    expected = ref_full_forward(ids, weights, cfg)
    actual = model.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
    for prompt in (1, 3, 6):
        state = model.create_generation_state(1, 12)
        logits = model.prefill(ids[:, :prompt], state)
        np.testing.assert_allclose(logits, expected[:, prompt-1], atol=2e-5, rtol=1e-4)
        for row in range(prompt, ids.shape[1]):
            logits = model.decode_one(ids[:, row], state)
            np.testing.assert_allclose(logits, expected[:, row], atol=2e-5, rtol=1e-4)


def test_bad_shape_missing_tensor_and_untied_output():
    model = ExternalLlamaModel(make_config(tie_word_embeddings=False))
    weights = make_weights(model)
    weights["lm_head.weight"] = weights["lm_head.weight"] + 0.4
    model.set_weights(weights)
    logits = model.forward(np.array([[1, 2, 3]], dtype=np.int32))
    assert logits.shape == (1, 3, 49)
    with pytest.raises(ValueError, match="expected shape"):
        bad = dict(weights)
        bad["model.layers.0.self_attn.k_proj.weight"] = np.empty((3, 4))
        model.set_weights(bad)
    with pytest.raises(ValueError, match="missing"):
        bad = dict(weights)
        del bad["model.norm.weight"]
        model.set_weights(bad)
    with pytest.raises(ValueError, match="extra"):
        model.set_weights({**weights, "something.unexpected": np.zeros((3,))})
