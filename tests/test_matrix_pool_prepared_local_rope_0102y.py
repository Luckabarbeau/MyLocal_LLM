"""0102Y: Model B reuses per-token RoPE factors across layers and checkpoint replay."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
from mini_llm.ops.rope import PreparedRoPEPositions
from mini_llm.ops import matrix_pool_dual_resolution_attention as pool_module
from tests.test_matrix_pool_dual_resolution_0102o import _attention, _tiny_model


def _packed_tokens():
    # Documents restart RoPE coordinates at EOS. A second row ensures no
    # implicit assumption that every row has the same source positions.
    return np.array([
        [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11],
        [7, 8, 9, 10, 11, 0, 3, 4, 5, 6, 7, 8],
    ], dtype=np.int64)


def test_local_rope_reuses_prepared_factors_across_global_and_layers(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU deterministic reference")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_GLOBAL_ROPE", "1")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE", "1")
    tokens = _packed_tokens()
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=4)
    operator1 = _attention(dtype="float64")
    operator2 = _attention(seed=1025, dtype="float64")
    device_md = operator1._prepare_metadata(metadata, 2, 12, None)
    assert operator2._prepare_metadata(metadata, 2, 12, None) is device_md
    counts = []
    original = pool_module.prepare_rope_positions

    def counted(*args, **kwargs):
        counts.append(tuple(args[0].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(pool_module, "prepare_rope_positions", counted)
    token_a = operator1._local_rope_positions(device_md, np.dtype("float64"))
    token_b = operator2._local_rope_positions(device_md, np.dtype("float64"))
    assert isinstance(token_a, PreparedRoPEPositions)
    assert token_b is token_a
    summary_md = operator1._summary_metadata(device_md)
    global_q, global_k = operator1._global_rope_positions(
        device_md, summary_md, np.dtype("float64")
    )
    assert global_q is token_a
    assert global_k is not token_a
    again_q, again_k = operator2._global_rope_positions(
        device_md, summary_md, np.dtype("float64")
    )
    assert again_q is global_q and again_k is global_k
    assert counts == [(2, 12), (2, 12)]
    np.testing.assert_array_equal(asnumpy(token_a.cos[:, 0, 0]),
                                  asnumpy(global_q.cos[:, 0, 0]))


def test_prepared_local_rope_packed_document_and_checkpoint_parity(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU float64 numerical parity")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_GLOBAL_ROPE", "1")
    tokens = _packed_tokens()
    rng = np.random.default_rng(10211)
    grad = rng.normal(size=(2, 12, 16))

    def execute(checkpoint, prepared):
        monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE",
                           "1" if prepared else "0")
        config, model = _tiny_model(seed=1021, dtype="float64")
        metadata = build_dual_resolution_metadata(tokens, eos_token_id=0,
                                                   block_size=config.dual_resolution.block_size)
        output, cache = model.forward_body(
            xp.asarray(tokens), activation_checkpoint=checkpoint,
            dual_resolution_metadata=metadata,
        )
        # Preparing RoPE on the first layer MUST survive all subsequent layers
        # and, in the checkpointed path, the entire reverse replay.
        device_md = metadata["_dual_resolution_device_metadata_0102f"][1]
        prepared_rope = device_md.get("_matrix_pool_local_rope_0102y", {})
        assert bool(prepared_rope) is bool(prepared)
        first = next(iter(prepared_rope.values())) if prepared else None
        model.zero_grad()
        dx = model.backward_body(xp.asarray(grad), cache)
        if prepared:
            assert next(iter(prepared_rope.values())) is first
            assert len(prepared_rope) == 1
        return (asnumpy(output),
                tuple(asnumpy(p.grad).copy() for p in model.parameters()))

    for checkpoint in (False, True):
        reference = execute(checkpoint, prepared=False)
        optimized = execute(checkpoint, prepared=True)
        np.testing.assert_allclose(optimized[0], reference[0], rtol=1e-12, atol=1e-12)
        for i, (got, expected) in enumerate(zip(optimized[1], reference[1])):
            np.testing.assert_allclose(got, expected, rtol=2e-11, atol=2e-11,
                                       err_msg=f"parameter {i}, checkpoint={checkpoint}")


def test_two_layer_checkpoint_replay_prepares_only_one_token_and_summary_table(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("CPU deterministic instrumentation")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE", "1")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_GLOBAL_ROPE", "1")
    config, model = _tiny_model(dtype="float64")
    tokens = _packed_tokens()
    metadata = build_dual_resolution_metadata(
        tokens, eos_token_id=0, block_size=config.dual_resolution.block_size
    )
    calls = []
    original = pool_module.prepare_rope_positions

    def counted(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(pool_module, "prepare_rope_positions", counted)
    output, cache = model.forward_body(
        xp.asarray(tokens), activation_checkpoint=True,
        dual_resolution_metadata=metadata,
    )
    model.zero_grad()
    model.backward_body(xp.ones_like(output), cache)
    # 2 Transformer layers, with reverse checkpoint replays: the token
    # table is not rebuilt on either the second layer or either replay.
    assert calls == [(2, 12), (2, 12)]


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CUDA BF16")
def test_bf16_prepared_local_rope_forward_backward_parity(monkeypatch):
    # Small operator-level CUDA check; full production-width benchmark is
    # performed separately on the RTX 5080.
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "cupy")
    rng = np.random.default_rng(10212)
    from mini_llm.backend import resolve_dtype

    bf16 = resolve_dtype("bfloat16")
    x = xp.asarray(rng.normal(scale=0.1, size=(1, 8, 16)).astype(np.float32), dtype=bf16)
    residual = xp.asarray(rng.normal(size=(1, 8, 16)).astype(np.float32))
    grad = xp.asarray(rng.normal(size=(1, 8, 16)).astype(np.float32), dtype=bf16)
    tokens = xp.asarray([[1, 2, 3, 0, 4, 5, 6, 7]], dtype=xp.int64)
    metadata = build_dual_resolution_metadata(asnumpy(tokens), eos_token_id=0,
                                               block_size=4)

    def run(enabled):
        monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE",
                           "1" if enabled else "0")
        operator = _attention(seed=1042, dtype="bfloat16")
        operator.zero_grad()
        y, cache = operator.forward(x, residual_input=residual,
                                    dual_resolution_metadata=metadata)
        dx, dresid = operator.backward(grad, cache)
        return (asnumpy(y.astype(xp.float32)),
                asnumpy(dx.astype(xp.float32)),
                asnumpy(dresid.astype(xp.float32)),
                [asnumpy(p.grad.astype(xp.float32)).copy() for p in operator.parameters()])

    ref, opt = run(False), run(True)
    for a, b in zip(ref[:3], opt[:3]):
        np.testing.assert_allclose(a, b, rtol=0.02, atol=0.02)
    for a, b in zip(ref[3], opt[3]):
        np.testing.assert_allclose(a, b, rtol=0.02, atol=0.02)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CUDA BF16")
def test_bf16_model_b_fused_packed_qkv_rope_parity(monkeypatch):
    """Model B reuses the historical fused QKV/RoPE graph without gradient drift."""
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "cupy")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "1")
    monkeypatch.setenv("MINI_LLM_MATRIX_POOL_PREPARED_LOCAL_ROPE", "1")
    monkeypatch.setenv("MINI_LLM_COMPACT_PACKED_V", "1")
    rng = np.random.default_rng(10213)
    from mini_llm.backend import resolve_dtype

    bf16 = resolve_dtype("bfloat16")
    x = xp.asarray(rng.normal(scale=0.1, size=(1, 8, 16)).astype(np.float32), dtype=bf16)
    residual = xp.asarray(rng.normal(size=(1, 8, 16)).astype(np.float32))
    grad = xp.asarray(rng.normal(size=(1, 8, 16)).astype(np.float32), dtype=bf16)
    tokens = np.array([[1, 2, 3, 0, 4, 5, 6, 7]], dtype=np.int64)
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=4)
    fused_results = []
    original = pool_module._fused_packed_qkv_rope_forward_bf16

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        fused_results.append(result is not None)
        return result

    monkeypatch.setattr(pool_module, "_fused_packed_qkv_rope_forward_bf16", observed)

    def run(enabled):
        monkeypatch.setenv("MINI_LLM_FUSED_QKV_ROPE_FORWARD", "1" if enabled else "0")
        operator = _attention(seed=1043, dtype="bfloat16")
        operator.zero_grad()
        y, cache = operator.forward(
            x, residual_input=residual, dual_resolution_metadata=metadata
        )
        dx, dresid = operator.backward(grad, cache)
        return (
            asnumpy(y.astype(xp.float32)),
            asnumpy(dx.astype(xp.float32)),
            asnumpy(dresid.astype(xp.float32)),
            [asnumpy(p.grad.astype(xp.float32)).copy() for p in operator.parameters()],
        )

    reference, optimized = run(False), run(True)
    assert fused_results == [False, True]
    for actual, expected in zip(optimized[:3], reference[:3]):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(optimized[3], reference[3]):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
