"""Focused GPU parity for fused packed-QKV forward RoPE."""

from __future__ import annotations

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.attention import _fused_packed_qkv_rope_forward_bf16
from mini_llm.ops.rope import prepare_rope_positions, rope_forward


pytestmark = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy backend")


@pytest.mark.parametrize("batch_specific_positions", [False, True])
def test_fused_packed_qkv_rope_is_bitwise_equal_to_native_graph(
    monkeypatch, batch_specific_positions
):
    rng = np.random.default_rng(89_014)
    batch, seq_len, q_heads, kv_heads, d_head = 3, 11, 4, 2, 8
    q_width = q_heads * d_head
    kv_width = kv_heads * d_head
    packed = xp.asarray(
        rng.normal(size=(batch * seq_len, q_width + 2 * kv_width)),
        dtype=xp.dtype("bfloat16"),
    )
    if batch_specific_positions:
        raw_positions = xp.asarray(
            np.arange(batch * seq_len, dtype=np.int64).reshape(batch, seq_len) + 31
        )
    else:
        raw_positions = xp.arange(31, 31 + seq_len, dtype=xp.int64)
    positions = prepare_rope_positions(
        raw_positions, d_head=d_head, base=1_000_000.0, dtype="bfloat16"
    )

    q_pre = packed[:, :q_width].reshape(batch, seq_len, q_heads, d_head)
    k_pre = packed[:, q_width : q_width + kv_width].reshape(
        batch, seq_len, kv_heads, d_head
    )
    v_ref = xp.array(
        packed[:, q_width + kv_width :].reshape(
            batch, seq_len, kv_heads, d_head
        ),
        copy=True,
        order="C",
    )
    q_ref, q_cache = rope_forward(
        q_pre, base=1_000_000.0, position_ids=positions
    )
    k_ref, k_cache = rope_forward(
        k_pre, base=1_000_000.0, position_ids=positions
    )

    monkeypatch.setenv("MINI_LLM_FUSED_QKV_ROPE_FORWARD", "1")
    result = _fused_packed_qkv_rope_forward_bf16(
        packed,
        positions,
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        d_head=d_head,
    )
    assert result is not None
    q, k, v, fused_q_cache, fused_k_cache = result

    for actual, expected in ((q, q_ref), (k, k_ref), (v, v_ref)):
        np.testing.assert_array_equal(
            asnumpy(actual.view(xp.uint16)), asnumpy(expected.view(xp.uint16))
        )
    assert fused_q_cache["cos"] is q_cache["cos"]
    assert fused_q_cache["sin"] is q_cache["sin"]
    assert fused_k_cache["cos"] is k_cache["cos"]
    assert fused_k_cache["sin"] is k_cache["sin"]
