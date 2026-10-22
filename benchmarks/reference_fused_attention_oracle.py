#!/usr/bin/env python3
"""NumPy oracle for causal/sliding-window grouped-query attention.

This module is deliberately slow and explicit. Q has shape [B,T,Hq,D], K/V
have shape [B,T,Hkv,D], and ``kv_map`` maps each supplied Q head to a native
KV head. Inputs are post-RoPE. Historical heads omitted through
``active_heads`` are left at zero and must be handled by the existing reader.
"""

from __future__ import annotations

import numpy as np


def _validate(q, k, v, kv_map, head_windows, active_heads):
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected Q/K/V [B,T,H,D]")
    if k.shape != v.shape:
        raise ValueError("K and V must have identical shapes")
    batch, length, n_q_heads, d_head = q.shape
    if k.shape[:2] != (batch, length) or k.shape[-1] != d_head:
        raise ValueError("incompatible Q/K/V shapes")
    n_kv_heads = int(k.shape[2])
    mapping = tuple(int(head) for head in kv_map)
    windows = tuple(int(window) for window in head_windows)
    if len(mapping) != n_q_heads or len(windows) != n_q_heads:
        raise ValueError("mapping and windows must cover every supplied Q head")
    if any(head < 0 or head >= n_kv_heads for head in mapping):
        raise ValueError("kv_map contains an invalid native KV head")
    if any(window < 1 for window in windows):
        raise ValueError("attention windows must be positive")
    active = tuple(range(n_q_heads) if active_heads is None else active_heads)
    if len(set(active)) != len(active) or any(h < 0 or h >= n_q_heads for h in active):
        raise ValueError("active_heads must contain unique supplied-Q indices")
    return q, k, v, mapping, windows, active


def forward_reference(q, k, v, kv_map, head_windows, *, scale=None, active_heads=None):
    """Return FP64 output and FP64 LSE in unambiguous [B,T,Hq] layout."""
    q, k, v, mapping, windows, active = _validate(
        q, k, v, kv_map, head_windows, active_heads
    )
    batch, length, _, d_head = q.shape
    scale = float(d_head ** -0.5 if scale is None else scale)
    out = np.zeros_like(q)
    lse = np.full(q.shape[:3], -np.inf, dtype=np.float64)

    for b in range(batch):
        for h in active:
            kv_head = mapping[h]
            window = windows[h]
            for i in range(length):
                lo = max(0, i - window + 1)
                keys = k[b, lo : i + 1, kv_head]
                values = v[b, lo : i + 1, kv_head]
                scores = (keys @ q[b, i, h]) * scale
                row_max = float(np.max(scores))
                exponentials = np.exp(scores - row_max)
                denominator = float(np.sum(exponentials))
                probabilities = exponentials / denominator
                out[b, i, h] = probabilities @ values
                lse[b, i, h] = row_max + np.log(denominator)
    return out, lse


def backward_reference(
    q,
    k,
    v,
    d_out,
    kv_map,
    head_windows,
    *,
    scale=None,
    active_heads=None,
):
    """Return exact-reference dQ/dK/dV with shared-KV contributions summed."""
    q, k, v, mapping, windows, active = _validate(
        q, k, v, kv_map, head_windows, active_heads
    )
    d_out = np.asarray(d_out, dtype=np.float64)
    if d_out.shape != q.shape:
        raise ValueError("d_out must match Q/output")
    _, length, _, d_head = q.shape
    scale = float(d_head ** -0.5 if scale is None else scale)
    d_q = np.zeros_like(q)
    d_k = np.zeros_like(k)
    d_v = np.zeros_like(v)

    for b in range(q.shape[0]):
        for h in active:
            kv_head = mapping[h]
            window = windows[h]
            for i in range(length):
                lo = max(0, i - window + 1)
                keys = k[b, lo : i + 1, kv_head]
                values = v[b, lo : i + 1, kv_head]
                query = q[b, i, h]
                upstream = d_out[b, i, h]
                scores = (keys @ query) * scale
                exponentials = np.exp(scores - np.max(scores))
                probabilities = exponentials / np.sum(exponentials)
                output = probabilities @ values
                d_probabilities = values @ upstream
                delta = float(output @ upstream)
                d_scores = probabilities * (d_probabilities - delta)
                d_q[b, i, h] += (d_scores @ keys) * scale
                d_k[b, lo : i + 1, kv_head] += (
                    d_scores[:, None] * query[None, :] * scale
                )
                d_v[b, lo : i + 1, kv_head] += (
                    probabilities[:, None] * upstream[None, :]
                )
    return d_q, d_k, d_v


def _self_test():
    rng = np.random.default_rng(20261021)
    batch, length, n_q_heads, n_kv_heads, d_head = 2, 9, 8, 2, 8
    q = rng.normal(0, 0.2, (batch, length, n_q_heads, d_head))
    k = rng.normal(0, 0.2, (batch, length, n_kv_heads, d_head))
    v = rng.normal(0, 0.2, (batch, length, n_kv_heads, d_head))
    d_out = rng.normal(0, 0.2, q.shape)
    mapping = tuple(h // 4 for h in range(n_q_heads))
    windows = (length,) + (3,) * (n_q_heads - 1)
    output, lse = forward_reference(q, k, v, mapping, windows)
    gradients = backward_reference(q, k, v, d_out, mapping, windows)
    assert np.isfinite(output).all() and np.isfinite(lse).all()
    assert all(np.isfinite(gradient).all() for gradient in gradients)

    epsilon = 1e-5
    for label, tensor, gradient, index in (
        ("Q", q, gradients[0], (0, 6, 3, 2)),
        ("K", k, gradients[1], (1, 4, 0, 1)),
        ("V", v, gradients[2], (1, 4, 1, 3)),
    ):
        original = tensor[index]
        tensor[index] = original + epsilon
        plus = float(np.sum(forward_reference(q, k, v, mapping, windows)[0] * d_out))
        tensor[index] = original - epsilon
        minus = float(np.sum(forward_reference(q, k, v, mapping, windows)[0] * d_out))
        tensor[index] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient[index], numerical, rtol=5e-7, atol=1e-8)

    perturbed = v.copy()
    perturbed[:, 0] += 100.0
    changed = forward_reference(q, k, perturbed, mapping, windows)[0]
    np.testing.assert_array_equal(changed[:, -1, 1:], output[:, -1, 1:])
    assert np.max(np.abs(changed[:, -1, 0] - output[:, -1, 0])) > 0.0
    omitted = forward_reference(
        q, k, v, mapping, windows, active_heads=set(range(n_q_heads)) - {5}
    )[0]
    assert np.all(omitted[:, :, 5] == 0.0)
    print(
        "PASS: FP64 forward, backward, finite differences, window, "
        "uneven shared KV, skipped head"
    )


if __name__ == "__main__":
    _self_test()
