"""Numeric checks for mini_llm/ops/triton_prefill_experts.py kernels.

Part 1 is exact: device dequant must equal the host FP32 dequantize_reference
value rounded to FP16 (the policy bakes per-block scales into FP16 weights).
Part 2 is tolerance-based against FP32 matmul, as befits an FP16 tensor-core
policy; it is not a bitwise-equivalence claim.
"""
import os
import sys

os.environ.setdefault('MINI_LLM_BACKEND', 'cupy')
import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.quantized_weight import PackedGGUFWeight, BLOCK_FORMATS
import mini_llm.ops.triton_prefill_experts as tpe

rng = np.random.default_rng(20261122)


def _valid_bytes(n_rows, k, qb, qk):
    # Random payload bytes but a finite small FP16 scale at b[0:2] of every
    # super-block, so dequantize_reference cannot produce NaN/Inf bit patterns.
    data = rng.integers(0, 256, (n_rows, k // qk, qb), dtype=np.uint8)
    data[:, :, 0:2] = np.frombuffer(np.float16(0.02).tobytes(), np.uint8)
    return data.reshape(n_rows, -1)


def _check_dequant(qtype, n_out, k):
    qk, qb = BLOCK_FORMATS[qtype]
    data = _valid_bytes(n_out, k, qb, qk)
    weight = PackedGGUFWeight(np.ascontiguousarray(data.reshape(-1)),
                              (n_out, k), qtype)
    ref = weight.dequantize_reference().astype(np.float32)
    dev = xp.asarray(np.ascontiguousarray(data).reshape(-1))
    out = xp.zeros((n_out * k,), xp.float16)
    fmt = tpe._FORMAT_ID[qtype]
    G2S, G3S, G3X, KSG, KV = tpe._codebooks()
    tpe._dequant_kernel[(n_out, k // 32, 1)](
        tpe._ptr(dev), tpe._ptr(out), G2S, G3S, G3X, KSG, KV,
        int(dev.nbytes), N=n_out, K=k, QK=qk, QB=qb, FMT=fmt)
    got = xp.asnumpy(out).reshape(n_out, k)
    want = ref.astype(np.float16)
    ok = np.array_equal(got, want)
    err = float(np.max(np.abs(got.astype(np.float32) - ref)))
    print(f'dequant {qtype:8s} ({n_out},{k}): exact_vs_fp16(round(ref))={ok} '
          f'max_abs_vs_ref={err:.3e}')
    return ok


def _check_ffn():
    T, K, Nff, D = 96, 2560, 640, 2560
    topk = 4
    # Small activations keep the FP32 host reference out of exp overflow with
    # random synthetic weights (real checkpoint weights are bounded by ~0.1).
    x = (rng.standard_normal((T, K)) * 0.002).astype(np.float32)
    # Four experts; rows bucketed by expert exactly like the policy; unique
    # expert IDs per token as the production path requires.
    ids = (np.arange(T)[:, None] + np.arange(topk)[None, :]) % 4
    routes = rng.random((T, topk), dtype=np.float32)
    weights = {}
    for e in range(4):
        gate = PackedGGUFWeight(_valid_bytes(Nff, K, 110, 256).reshape(-1),
                                (Nff, K), 'IQ3_S')
        up = PackedGGUFWeight(_valid_bytes(Nff, K, 110, 256).reshape(-1),
                              (Nff, K), 'IQ3_S')
        down = PackedGGUFWeight(_valid_bytes(D, Nff, 18, 32).reshape(-1),
                                (D, Nff), 'IQ4_NL')
        weights[e] = (gate, up, down)
    # Host reference from exact dequantized weights in FP32.
    wref = {e: tuple(w.dequantize_reference().astype(np.float32) for w in trio)
            for e, trio in weights.items()}
    href = np.zeros((T, D), np.float32)
    for t in range(T):
        for j in range(topk):
            e = int(ids[t, j])
            g, u, d = wref[e]
            h = (x[t] @ g.T)
            hs = (x[t] @ u.T)
            act = h / (1.0 + np.exp(-h)) * hs
            href[t] += routes[t, j] * (act @ d.T)

    # Build the policy's device structures for both experts in one wave.
    flat_ids, flat_routes = ids.ravel(), routes.ravel()
    rows = flat_ids.size
    counts = np.bincount(flat_ids, minlength=4)
    starts = np.zeros(5, np.int64); np.cumsum(counts, out=starts[1:])
    tok_seq = np.repeat(np.arange(T, dtype=np.int64), topk)
    order = np.lexsort((tok_seq, flat_ids))
    rowid = np.empty(rows, np.int32)
    rowid[order] = (starts[flat_ids[order]] + np.arange(rows)
                    - starts[flat_ids[order]]).astype(np.int32)
    token_of_row = tok_seq[order].astype(np.int32)
    order2 = np.lexsort((flat_ids, tok_seq))
    tok_row = rowid[order2].reshape(T, topk)
    tok_w = np.ascontiguousarray(flat_routes[order2].reshape(T, topk), np.float32)
    tok_id = flat_ids[order2].reshape(T, topk)

    seg = [int(weights[0][m].nbytes) for m in range(3)]
    ring = tpe._StageRing(seg, 4, slots=1)
    for s in range(4):
        for m in range(3):
            ring.memmove(0, int(weights[s][m].data.ctypes.data),
                         weights[s][m].nbytes,
                         [0, ring.offu, ring.offd][m] + s * seg[m])
    ring.upload(0)
    xp.cuda.get_current_stream().wait_event(ring.h2d[0])
    G2S, G3S, G3X, KSG, KV = tpe._codebooks()
    g16 = xp.zeros((4, Nff * K), xp.float16); u16 = xp.zeros((4, Nff * K), xp.float16)
    d16 = xp.zeros((4, D * Nff), xp.float16)
    for fp_out, off, qtype, N, Kk, m in (
            (g16, 0, 'IQ3_S', Nff, K, 0), (u16, ring.offu, 'IQ3_S', Nff, K, 1),
            (d16, ring.offd, 'IQ4_NL', D, Nff, 2)):
        qk, qb = BLOCK_FORMATS[qtype]
        tpe._dequant_kernel[(N, Kk // 32, 4)](
            tpe._ptr(ring.device, off), tpe._ptr(fp_out), G2S, G3S, G3X, KSG, KV,
            seg[m], N=N, K=Kk, QK=qk, QB=qb, FMT=tpe._FORMAT_ID[qtype])
    a16 = xp.asarray(x.astype(np.float16))
    hbuf = xp.zeros((rows, Nff), xp.float32)
    values = xp.zeros((rows, D), xp.float32)
    out = xp.zeros((T, D), xp.float32)
    # Tiles of BM=64 rows per expert, exactly like the production grouping.
    tl_s, tl_c, tl_slot = [], [], []
    for e in range(4):
        for off in range(0, int(counts[e]), 64):
            tl_s.append(int(starts[e]) + off)
            tl_c.append(min(64, int(counts[e]) - off))
            tl_slot.append(e)
    ntiles = len(tl_s)
    ts = xp.asarray(np.array(tl_s, np.int32))
    tc = xp.asarray(np.array(tl_c, np.int32))
    tslot = xp.asarray(np.array(tl_slot, np.int32))
    dev_tokrow = xp.asarray(token_of_row)
    dev_tok_row = xp.asarray(tok_row)
    dev_tok_w = xp.asarray(tok_w)
    mask = xp.asarray(np.ones((T, topk), np.uint8))
    tpe._gateup_kernel[(ntiles, Nff // 64)](
        tpe._ptr(a16), tpe._ptr(g16), tpe._ptr(u16), tpe._ptr(hbuf),
        tpe._ptr(dev_tokrow), tpe._ptr(ts), tpe._ptr(tc),
        tpe._ptr(tslot), N=Nff, K=K, BM=64, BN=64, BK=64)
    hwave = hbuf.astype(np.float16)
    tpe._down_kernel[(ntiles, D // 64)](
        tpe._ptr(hwave), tpe._ptr(d16), tpe._ptr(values),
        tpe._ptr(ts), tpe._ptr(tc), tpe._ptr(tslot), int(starts[0]),
        N=D, K=Nff, BM=64, BN=64, BK=64)
    tpe._combine_kernel[(T, D // 256)](
        tpe._ptr(values), tpe._ptr(out), tpe._ptr(dev_tok_row),
        tpe._ptr(dev_tok_w), tpe._ptr(mask), D=D, K=topk, DC=256)
    got = xp.asnumpy(out)
    denom = np.maximum(np.abs(href), 1.0)
    rel = float(np.max(np.abs(got - href) / denom))
    print(f'ffn policy vs fp32 reference: max rel(denominator-floored)={rel:.3e}')
    return rel < 5e-2


def main():
    ok = True
    for qtype, n_out, k in (('IQ2_S', 32, 256), ('IQ3_S', 32, 256),
                            ('IQ3_XXS', 32, 256), ('IQ4_XS', 32, 256),
                            ('IQ4_NL', 32, 32), ('Q2_0', 32, 64)):
        ok &= _check_dequant(qtype, n_out, k)
    ok &= _check_ffn()
    print('TRITON KERNEL CHECKS', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
