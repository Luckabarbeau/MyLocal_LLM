"""Read-only numerical contracts for the Model B training-correctness audit."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, resolve_dtype, xp
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.hybrid_muon import HybridMuonAdamW, muon_eligible
from mini_llm.parameter import Parameter


def _numpy(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _independent_ns(matrix, steps=5):
    """Independent FP32 transcription; does not call the production NS helper."""
    x = np.asarray(matrix, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, dtype=np.float32), dtype=np.float32)
    x = (x / max(float(norm), 1.0e-7)).astype(np.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        polynomial = (-4.7750 * gram + 2.0315 * (gram @ gram)).astype(np.float32)
        x = (3.4445 * x + polynomial @ x).astype(np.float32)
    return (x.T if transposed else x).astype(np.float32)


@pytest.mark.parametrize("shape", [(6, 6), (4, 7), (7, 4)])
def test_hybrid_muon_matches_independent_fp32_reference(shape):
    rng = np.random.default_rng(108)
    initial = rng.normal(size=shape).astype(np.float32)
    parameter = Parameter(
        xp.asarray(initial), name="blocks.0.attention.Wq", decay=True,
    )
    optimizer = HybridMuonAdamW(
        [parameter], lr=3.0e-4, beta1=0.9, beta2=0.95, eps=1.0e-8,
        weight_decay=0.1, muon_momentum=0.95, muon_ns_steps=5,
        muon_matmul_precision="fp32",
    )

    reference_weight = initial.copy()
    reference_momentum = np.zeros(shape, dtype=np.float32)
    assert parameter.grad is not None
    for _ in range(3):
        gradient = rng.normal(size=shape).astype(np.float32)
        parameter.grad[...] = xp.asarray(gradient)

        reference_momentum *= np.float32(0.95)
        reference_momentum += np.float32(0.05) * gradient
        nesterov = np.float32(0.95) * reference_momentum + np.float32(0.05) * gradient
        polar = _independent_ns(nesterov)
        direction = polar * np.float32(0.2 * np.sqrt(max(shape)))
        reference_weight *= np.float32(1.0 - 3.0e-4 * 0.1)
        reference_weight -= np.float32(3.0e-4) * direction

        optimizer.step()
        np.testing.assert_allclose(
            _numpy(optimizer.muon.m[0]), reference_momentum, rtol=2e-6, atol=2e-7,
        )
        np.testing.assert_allclose(
            _numpy(parameter.data), reference_weight, rtol=2e-6, atol=2e-7,
        )

        production_delta = (initial - _numpy(parameter.data)).astype(np.float64)
        reference_delta = (initial - reference_weight).astype(np.float64)
        np.testing.assert_allclose(
            np.linalg.svd(production_delta, compute_uv=False),
            np.linalg.svd(reference_delta, compute_uv=False),
            rtol=3e-6, atol=3e-7,
        )


def _model_b_inventory():
    config = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    d = int(config.d_model)
    kv = int(config.n_kv_heads * config.d_head)
    dual = config.dual_resolution
    global_width = int(dual.global_heads * dual.global_head_dim)
    rows: list[tuple[str, tuple[int, ...], bool]] = [
        ("embedding.W", (config.vocab_size, d), False)
    ]
    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}"
        rows.extend([
            (f"{prefix}.norm1.gamma", (d,), False),
            (f"{prefix}.attention.Wq", (d, d), True),
            (f"{prefix}.attention.Wk", (d, kv), True),
            (f"{prefix}.attention.Wv", (d, kv), True),
            (f"{prefix}.attention.Wo", (d, d), True),
            (f"{prefix}.attention.matrix_pool.score", (d, dual.summaries_per_block), True),
            (f"{prefix}.attention.matrix_pool.position_bias", (dual.block_size, dual.summaries_per_block), False),
            (f"{prefix}.attention.global.Wq", (d, global_width), True),
            (f"{prefix}.attention.global.Wk", (d, global_width), True),
            (f"{prefix}.attention.global.Wv", (d, global_width), True),
            (f"{prefix}.attention.global.Wo", (global_width, d), True),
            (f"{prefix}.norm2.gamma", (d,), False),
            (f"{prefix}.dense_ffn.expert.W_gate", (d, config.d_ff), True),
            (f"{prefix}.dense_ffn.expert.W_up", (d, config.d_ff), True),
            (f"{prefix}.dense_ffn.expert.W_down", (config.d_ff, d), True),
        ])
    rows.append(("final_norm.gamma", (d,), False))
    return config, rows


def test_complete_model_b_parameter_group_inventory():
    config, rows = _model_b_inventory()
    inventory = []
    for name, shape, decay in rows:
        proxy = SimpleNamespace(name=name, data=SimpleNamespace(shape=shape))
        inventory.append((name, shape, int(np.prod(shape)), "muon" if muon_eligible(proxy) else "adamw", decay))

    assert len(inventory) == 302
    assert len({row[0] for row in inventory}) == len(inventory)
    assert sum(row[2] for row in inventory) == config.estimated_parameter_count() == 348_261_376
    assert sum(row[3] == "muon" for row in inventory) == 220
    assert sum(row[3] == "adamw" for row in inventory) == 82
    assert sum(row[2] for row in inventory if row[3] == "muon") == 314_572_800
    assert sum(row[2] for row in inventory if row[3] == "adamw" and row[4]) == 81_920
    assert sum(row[2] for row in inventory if row[3] == "adamw" and not row[4]) == 33_606_656
    assert not any(name == "output_proj.W" for name, *_ in inventory)


def test_tied_matrix_ratio_scales_update_without_changing_raw_gradient():
    config = replace(ModelConfig.tiny_inspection(), tie_embeddings=True)
    model = DecoderLanguageModel(config, rng_seed=41, dtype="float32")
    tokens = xp.asarray([[1, 2, 3, 4]], dtype=xp.int64)
    targets = xp.asarray([[2, 3, 4, 5]], dtype=xp.int64)
    logits, cache = model.forward(tokens)
    _, loss_cache = model.compute_loss(logits, targets)
    model.zero_grad()
    model.backward(model.backward_loss(loss_cache), cache)
    assert model.embedding.W.grad is not None
    raw_gradient = _numpy(model.embedding.W.grad).copy()
    assert np.linalg.norm(raw_gradient) > 0.0
    assert model.output_proj.parameters() == []

    initial = _numpy(model.embedding.W.data).copy()
    parameters = []
    for ratio in (1.0, 0.1):
        parameter = Parameter(xp.asarray(initial.copy()), name="embedding.W", decay=False)
        assert parameter.grad is not None
        parameter.grad[...] = xp.asarray(raw_gradient)
        optimizer = HybridMuonAdamW(
            [parameter], lr=2.0e-3, beta1=0.9, beta2=0.95, eps=1.0e-8,
            weight_decay=0.1, muon_matmul_precision="fp32",
            unembedding_lr_ratio=ratio, unembedding_parameter_name="embedding.W",
        )
        optimizer.step()
        np.testing.assert_array_equal(_numpy(parameter.grad), raw_gradient)
        parameters.append(parameter)

    full_delta = initial - _numpy(parameters[0].data)
    reduced_delta = initial - _numpy(parameters[1].data)
    np.testing.assert_allclose(reduced_delta, 0.1 * full_delta, rtol=2e-4, atol=2e-7)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy and CUDA")
def test_cupy_bf16_full_offload_matches_device_optimizer(monkeypatch):
    import cupy as cp

    rng = np.random.default_rng(109)
    specs = [
        ("blocks.0.attention.Wq", (16, 16), True),
        ("blocks.0.attention.Wk", (16, 8), True),
        ("blocks.0.dense_ffn.expert.W_down", (24, 16), True),
        ("embedding.W", (32, 16), False),
    ]

    def build(offload):
        monkeypatch.setenv("MINI_LLM_OPTIMIZER_OFFLOAD", offload)
        params = []
        for name, shape, decay in specs:
            values = xp.asarray(rng.normal(size=shape).astype(np.float32), dtype=resolve_dtype("bfloat16"))
            params.append(Parameter(values, name=name, decay=decay))
        return params, HybridMuonAdamW(
            params, lr=1.5e-4, beta1=0.9, beta2=0.95, eps=1.0e-8,
            weight_decay=0.1, muon_momentum=0.95, muon_ns_steps=5,
            muon_matmul_precision="bf16",
        )

    rng_state = rng.bit_generator.state
    device_params, device = build("none")
    rng.bit_generator.state = rng_state
    offload_params, offload = build("full")
    assert offload.moments_offloaded and offload.master_weights_offloaded

    for step in range(2):
        for index, (left, right) in enumerate(zip(device_params, offload_params)):
            gradient = rng.normal(size=left.shape).astype(np.float32) * (step + 1) / (index + 1)
            left.grad[...] = xp.asarray(gradient)
            right.grad[...] = xp.asarray(gradient)
        device.step()
        offload.step()

    for left, right in zip(device_params, offload_params):
        np.testing.assert_array_equal(cp.asnumpy(left.data), cp.asnumpy(right.data))
    for left, right in zip(device.muon.m, offload.muon.m):
        np.testing.assert_allclose(cp.asnumpy(left), np.asarray(right), rtol=0, atol=0)
    for left, right in zip(device.adam.m, offload.adam.m):
        np.testing.assert_allclose(cp.asnumpy(left), np.asarray(right), rtol=0, atol=0)
    for left, right in zip(device.adam.v, offload.adam.v):
        np.testing.assert_allclose(cp.asnumpy(left), np.asarray(right), rtol=0, atol=0)
