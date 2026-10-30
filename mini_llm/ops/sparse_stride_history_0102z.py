"""0102Z: deterministic stride-16 full-resolution historical attention.

One existing GQA head uses original-token K/V at absolute indices
``phase + 16*j``. All queries are retained, with same-document and causal
masks. There is no selection network, extra projection, detached history,
or dense T*T attention mask. The original K/V arrays are accessed as
strided views. Empty history gives exactly zero output and gradient.
"""

from __future__ import annotations

from mini_llm.backend import BACKEND_NAME, xp


def sparse_forward(q, k, v, docs, *, stride, phase, scale, kernel,
                   out=None, lse=None):
    """Return [B,T,1,64] FP32 and a backward cache.

    q is one existing query head, k/v the corresponding shared GQA KV
    head, already RoPE-rotated. Absolute tensor indices determine phases;
    matching document IDs prevent leakage across packed documents.
    """
    if not 0 <= phase < stride:
        raise ValueError("invalid sparse phase")
    if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[2:] != (1, 64) or k.shape[2:] != (1, 64):
        raise ValueError("sparse retrieval expects single-head [B,T,1,64] Q/K/V")
    if q.shape[1] != k.shape[1] or docs.shape != q.shape[:2]:
        raise ValueError("sparse retrieval needs full-resolution K/V and document IDs")
    if kernel == "triton":
        if BACKEND_NAME != "cupy":
            raise RuntimeError("Triton sparse history needs CuPy")
        from .triton_dual_resolution_dense import forward
        return forward(
            q, k[:, phase::stride], v[:, phase::stride], scale=scale,
            documents=docs, bounded_causal=True,
            sparse_stride=stride, sparse_phase=phase, out=out, lse=lse,
        )
    # Reference supports NumPy as well as CuPy and serves as the parity oracle.
    from .attention import softmax_forward
    # Store only the identities of eligible original tokens. Recompute logits
    # in backward to avoid retaining a full attention probability matrix.
    import numpy as np
    batch, length = q.shape[:2]
    out = xp.zeros(q.shape, dtype=xp.float32)
    indices = []
    for row in range(batch):
        row_indices = []
        for qi in range(length):
            candidates = [i for i in range(phase, qi + 1, stride)
                          if int(docs[row, i]) == int(docs[row, qi])]
            ix = np.asarray(candidates, dtype=np.int64)
            row_indices.append(ix)
            if not len(ix):
                continue
            qq = q[row, qi, 0].astype(xp.float32)
            kk = k[row, ix, 0].astype(xp.float32)
            vv = v[row, ix, 0].astype(xp.float32)
            probs = softmax_forward((kk @ qq) * scale, axis=-1)
            out[row, qi, 0] = probs @ vv
        indices.append(row_indices)
    return out, (q, k, v, scale, indices, stride, phase)


def sparse_backward(dout, cache, *, kernel, dq=None, dk=None, dv=None,
                    add_dkdv=False):
    if kernel == "triton":
        from .triton_dual_resolution_dense import backward
        return backward(
            dout, cache, dq=dq, dk=dk, dv=dv, add_dkdv=add_dkdv,
        )
    from .attention import softmax_forward
    q, k, v, scale, indices, stride, phase = cache
    dq = xp.zeros(q.shape, dtype=xp.float32)
    dk = xp.zeros(k[:, phase::stride].shape, dtype=xp.float32)
    dv = xp.zeros(v[:, phase::stride].shape, dtype=xp.float32)
    for row, row_indices in enumerate(indices):
        for qi, ix in enumerate(row_indices):
            if not len(ix):
                continue
            qq = q[row, qi, 0].astype(xp.float32)
            kk = k[row, ix, 0].astype(xp.float32)
            vv = v[row, ix, 0].astype(xp.float32)
            grad = dout[row, qi, 0].astype(xp.float32)
            probs = softmax_forward((kk @ qq) * scale, axis=-1)
            dp = vv @ grad
            ds = probs * (dp - xp.sum(probs * dp))
            dq[row, qi, 0] += (ds @ kk) * scale
            dk[row, (ix - phase) // stride, 0] += ds[:, None] * qq[None, :] * scale
            dv[row, (ix - phase) // stride, 0] += probs[:, None] * grad[None, :]
    return dq, dk, dv
