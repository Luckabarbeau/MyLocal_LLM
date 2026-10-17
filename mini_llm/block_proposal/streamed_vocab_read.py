"""0080AH--AJ: exact, memory-bounded full-vocabulary read and manual VJP.

0080AJ collapses the all-source axis before the expensive Q@E GEMM:
    M[j,m,v] = sum_r exp(log P[r,v] + B[m,j,r])
    score[j,m,v] = log M[j,m,v] + beta * Q[j,m] @ E[v,m] / sqrt(d).
This is EXACTLY the K*V joint attention marginal over v.  No source or
vocabulary candidates are discarded.  Backward routes d(score) back to each
source through its conditional mixture responsibility, then differentiates
per-source logsoftmax.  The preceding source_loop variant remains selectable
for differential testing and benchmarking, not for production checkpoints.

Only [block_tile*K, vocab_tile] score/gradient tiles, [block_tile,K,V_tile]
source logits, and compact statistics are retained in temporary memory.
The NumPy implementation shares the same tiling for parity tests.  The
CuPy implementation is still a correctness-first library-kernel pipeline,
not a fused CUDA kernel.
"""

import math
from mini_llm.backend import xp


def _tiles(length, tile):
    for lo in range(0, length, tile):
        yield lo, min(length, lo + tile)


def _source_log_norm(read, logits):
    """FP32 logsumexp(L/T) for each source, streaming over vocabulary."""
    g, k, v = logits.shape
    mx = xp.full((g, k), -xp.inf, dtype=read.acc_dtype)
    mass = xp.zeros((g, k), dtype=read.acc_dtype)
    for b0, b1 in _tiles(g, read.block_tile):
        for v0, v1 in _tiles(v, read.vocab_tile):
            scores = logits[b0:b1, :, v0:v1].astype(read.acc_dtype) / read.temperature
            tile_max = xp.max(scores, axis=-1)
            new_max = xp.maximum(mx[b0:b1], tile_max)
            old = xp.exp(mx[b0:b1] - new_max)
            mass[b0:b1] = mass[b0:b1] * old + xp.sum(
                xp.exp(scores - new_max[:, :, None]), axis=-1)
            mx[b0:b1] = new_max
    return mx + xp.log(mass)


def _scores(read, logits, queries, embed, head, src, b0, b1, v0, v1,
            source_norm):
    """Score tile [block_tile*K, vocab_tile]; no hidden V x d axis."""
    gb, k, d = queries.shape
    raw = logits[b0:b1, :, v0:v1] if read.source_scope == 'own' else (
        logits[b0:b1, src:src + 1, v0:v1])
    raw = raw.astype(read.acc_dtype) / read.temperature
    if read.source_scope == 'all':
        raw = raw - source_norm[b0:b1, src:src + 1, None]
        raw = raw + read.source_bias.data[head, :, src].astype(read.acc_dtype)[None, :, None]
    # Broadcasting creates no V x embedding tensor (only a score tile).
    raw = xp.broadcast_to(raw, (gb, k, v1 - v0)).reshape(gb*k, v1-v0)
    return (raw + (read.read_scale / math.sqrt(read.head_width)) *
            (queries.reshape(gb*k, d) @ embed.T))


def _source_mixture_scores(read, logits, queries, embed, head,
                           b0, b1, v0, v1, source_norm):
    """Exact logsumexp over proposal sources *before* the query GEMM.

    Return scores [gb*K,V_tile], plus source log-probability [gb,K,V_tile]
    and log mixture [gb,K,V_tile] for the backward responsibilities.
    Neither temporaries nor cached outputs scale with the full vocabulary.
    """
    gb, k, dh = queries.shape
    log_prior = (logits[b0:b1, :, v0:v1].astype(read.acc_dtype) /
                 read.temperature - source_norm[b0:b1, :, None])
    bias = read.source_bias.data[head].astype(read.acc_dtype)
    # All sources are guaranteed to have finite log probabilities because
    # log_prior comes from FP32 logsumexp, not log(BF16 softmax) values.
    log_mix = xp.full((gb, k, v1-v0), -xp.inf, dtype=read.acc_dtype)
    for src in range(k):
        log_mix = xp.logaddexp(
            log_mix, log_prior[:, src:src+1, :] + bias[None, :, src:src+1])
    score = (log_mix.reshape(gb*k, v1-v0) +
             (read.read_scale / math.sqrt(read.head_width)) *
             (queries.reshape(gb*k, dh) @ embed.T))
    return score, log_prior, log_mix


def forward_tiled(read, decoder_states, proposal_logits):
    """Recompute-friendly online softmax; returns residual and normalizers."""
    g, k, v = proposal_logits.shape
    # FP32 projection avoids mixed BF16/FP32 GEMMs in the existing Linear.
    states = decoder_states.astype(read.acc_dtype)
    q = (states.reshape(-1, read.decoder_width) @
         read.query_projection.W.data.astype(read.acc_dtype)).reshape(
             g, k, read.n_heads, read.head_width)
    source_norm = (_source_log_norm(read, proposal_logits)
                   if read.source_scope == 'all' else None)
    output = xp.empty((g, k, read.model_width), dtype=read.acc_dtype)
    attention_norm = xp.empty((g, k, read.n_heads), dtype=read.acc_dtype)
    for m in range(read.n_heads):
        s = slice(m*read.head_width, (m+1)*read.head_width)
        for b0, b1 in _tiles(g, read.block_tile):
            gb = b1-b0
            qh = q[b0:b1, :, m, :]
            maximum = xp.full((gb*k,), -xp.inf, dtype=read.acc_dtype)
            mass = xp.zeros((gb*k,), dtype=read.acc_dtype)
            value = xp.zeros((gb*k, read.head_width), dtype=read.acc_dtype)
            mixture = (read.source_scope == 'all' and
                       read.all_source_algorithm == 'source_mixture')
            sources = (0,) if (mixture or read.source_scope == 'own') else range(k)
            for src in sources:
                for v0, v1 in _tiles(v, read.vocab_tile):
                    e = read.embedding.data[v0:v1, s].astype(read.acc_dtype)
                    if mixture:
                        score, _, _ = _source_mixture_scores(
                            read, proposal_logits, qh, e, m,
                            b0, b1, v0, v1, source_norm)
                    else:
                        score = _scores(read, proposal_logits, qh, e, m, src,
                                        b0, b1, v0, v1, source_norm)
                    tile_max = xp.max(score, axis=-1)
                    new_max = xp.maximum(maximum, tile_max)
                    old = xp.exp(maximum - new_max)
                    weight = xp.exp(score-new_max[:, None])
                    mass = mass*old + xp.sum(weight, axis=-1)
                    value = value*old[:, None] + weight @ e
                    maximum = new_max
            output[b0:b1, :, s] = (value/mass[:, None]).reshape(gb, k, -1)
            attention_norm[b0:b1, :, m] = (maximum+xp.log(mass)).reshape(gb, k)
    correction = (output.reshape(-1, read.model_width) @
                  read.output_projection.W.data.astype(read.acc_dtype)).reshape(
                      g, k, read.decoder_width)
    return correction, q, output, attention_norm, source_norm, states


def backward_tiled(read, d_correction, cache):
    """Exact streamed VJP, including logsoftmax Jacobian and shared E key/value."""
    logits = cache.proposal_logits
    g, k, v = logits.shape
    dy = d_correction.astype(read.acc_dtype)
    read_flat = cache.read.reshape(-1, read.model_width)
    dy_flat = dy.reshape(-1, read.decoder_width)
    read.output_projection.W.grad += read_flat.T @ dy_flat
    d_read = (dy_flat @ read.output_projection.W.data.astype(read.acc_dtype).T).reshape(
        g, k, read.n_heads, read.head_width)
    d_q = xp.zeros_like(cache.queries, dtype=read.acc_dtype)
    d_logits = xp.zeros((g, k, v), dtype=read.acc_dtype)
    source_total = (xp.zeros((g, k), dtype=read.acc_dtype)
                    if read.source_scope == 'all' else None)
    scale = read.read_scale/math.sqrt(read.head_width)
    for m in range(read.n_heads):
        s = slice(m*read.head_width, (m+1)*read.head_width)
        for b0, b1 in _tiles(g, read.block_tile):
            gb = b1-b0
            queries = cache.queries[b0:b1, :, m, :]
            qflat = queries.reshape(gb*k, read.head_width)
            upstream = d_read[b0:b1, :, m, :].reshape(gb*k, read.head_width)
            rval = cache.read[b0:b1, :, s].reshape(gb*k, read.head_width)
            row_dot = xp.sum(upstream*rval, axis=-1, keepdims=True)
            norm = cache.attention_norm[b0:b1, :, m].reshape(gb*k, 1)
            mixture = (read.source_scope == 'all' and
                       read.all_source_algorithm == 'source_mixture')
            sources = (0,) if (mixture or read.source_scope == 'own') else range(k)
            for src in sources:
                for v0, v1 in _tiles(v, read.vocab_tile):
                    e = read.embedding.data[v0:v1, s].astype(read.acc_dtype)
                    if mixture:
                        score, log_prior, log_mix = _source_mixture_scores(
                            read, logits, queries, e, m,
                            b0, b1, v0, v1, cache.source_norm)
                    else:
                        score = _scores(read, logits, queries, e, m, src,
                                        b0, b1, v0, v1, cache.source_norm)
                    weight = xp.exp(score - norm)
                    d_score = weight * (upstream @ e.T - row_dot)
                    d_q[b0:b1, :, m, :] += (scale * (d_score @ e)).reshape(gb, k, -1)
                    read.embedding.grad[v0:v1, s] += (
                        weight.T @ upstream + scale*(d_score.T @ qflat))
                    d_by_query = d_score.reshape(gb, k, v1-v0)
                    if read.source_scope == 'own':
                        d_logits[b0:b1, :, v0:v1] += d_by_query/read.temperature
                    elif mixture:
                        # d(log sum_r P_r(v)*exp B_jr) / d(log P_r(v))
                        # = P_r(v)*exp B_jr / sum_s P_s(v)*exp B_js.
                        # This is a per-vocabulary responsibility; it is NOT
                        # a source-independent attention mixing coefficient.
                        bias = read.source_bias.data[m].astype(read.acc_dtype)
                        for r in range(k):
                            responsibility = xp.exp(
                                log_prior[:, r:r+1, :] +
                                bias[None, :, r:r+1] - log_mix)
                            d_source = d_by_query * responsibility
                            d_logits[b0:b1, r, v0:v1] += xp.sum(d_source, axis=1)
                            source_total[b0:b1, r] += xp.sum(d_source, axis=(1, 2))
                            read.source_bias.grad[m, :, r] += xp.sum(
                                d_source, axis=(0, 2))
                    else:
                        d_logits[b0:b1, src, v0:v1] += xp.sum(d_by_query, axis=1)
                        source_total[b0:b1, src] += xp.sum(d_by_query, axis=(1, 2))
                        read.source_bias.grad[m, :, src] += xp.sum(d_by_query, axis=(0, 2))
    if read.source_scope == 'all':
        # dL = (dS - P * sum_v dS) / T, separately per proposal source.
        for b0, b1 in _tiles(g, read.block_tile):
            for v0, v1 in _tiles(v, read.vocab_tile):
                prior = xp.exp(logits[b0:b1, :, v0:v1].astype(read.acc_dtype) /
                               read.temperature - cache.source_norm[b0:b1, :, None])
                d_logits[b0:b1, :, v0:v1] -= (
                    prior * source_total[b0:b1, :, None])
        d_logits /= read.temperature
    qflat = d_q.reshape(-1, read.model_width)
    read.query_projection.W.grad += cache.projection_input.reshape(-1, read.decoder_width).T @ qflat
    d_input = (qflat @ read.query_projection.W.data.astype(read.acc_dtype).T).reshape(
        g, k, read.decoder_width)
    return d_input, d_logits


def _batched_mixture_tile(read, logits, query_heads, b0, b1, v0, v1,
                          source_norm, embedding_heads):
    """0080AJ2: batch H query-head GEMMs for one bounded vocabulary tile.

    query_heads: [H, (b1-b0)*K, Dh], embedding_heads: [H, Vtile, Dh].
    The temporary score/mix arrays are at most [H, block_tile*K, Vtile].
    In particular neither a V*Dh nor a K_source*H*G*K*V tensor is formed.
    """
    gb, k = b1-b0, logits.shape[1]
    h, n, dh = query_heads.shape
    width = v1-v0
    log_prior = (logits[b0:b1, :, v0:v1].astype(read.acc_dtype) /
                 read.temperature - source_norm[b0:b1, :, None])
    bias = read.source_bias.data.astype(read.acc_dtype)  # [H,target,source]
    mix = xp.full((h, gb, k, width), -xp.inf, dtype=read.acc_dtype)
    for r in range(k):
        mix = xp.logaddexp(mix,
                           log_prior[None, :, r:r+1, :] +
                           bias[:, None, :, r:r+1])
    scale = read.read_scale / math.sqrt(read.head_width)
    score = (mix.reshape(h, n, width) +
             scale * xp.matmul(query_heads, embedding_heads.transpose(0, 2, 1)))
    return score, log_prior, mix


def forward_batched_mixture(read, decoder_states, proposal_logits):
    """Fully batched heads/queries, streamed vocabulary, exact all-source read.

    Uses strided-batched GEMM (CuPy's batched matmul on device) for both QK^T
    and AV, instead of the per-head GEMM/softmax launch chain. On NumPy this
    implements the exact same algebra as a portable testing oracle.
    """
    g, k, v = proposal_logits.shape
    h, dh = read.n_heads, read.head_width
    states = decoder_states.astype(read.acc_dtype)
    q = (states.reshape(-1, read.decoder_width) @
         read.query_projection.W.data.astype(read.acc_dtype)).reshape(g, k, h, dh)
    source_norm = _source_log_norm(read, proposal_logits)
    output = xp.empty((g, k, read.model_width), dtype=read.acc_dtype)
    attention_norm = xp.empty((g, k, h), dtype=read.acc_dtype)
    for b0, b1 in _tiles(g, read.block_tile):
        gb = b1-b0
        # One GEMM batch per head. Explicitly compact: no full V dimension.
        qh = xp.ascontiguousarray(q[b0:b1].transpose(2, 0, 1, 3).reshape(h, gb*k, dh))
        maximum = xp.full((h, gb*k), -xp.inf, dtype=read.acc_dtype)
        mass = xp.zeros((h, gb*k), dtype=read.acc_dtype)
        value = xp.zeros((h, gb*k, dh), dtype=read.acc_dtype)
        for v0, v1 in _tiles(v, read.vocab_tile):
            vt = v1-v0
            # Same shared E as source_mixture; packed head slices for GEMM.
            e = xp.ascontiguousarray(
                read.embedding.data[v0:v1].astype(read.acc_dtype).reshape(vt, h, dh)
                .transpose(1, 0, 2))
            score, _, _ = _batched_mixture_tile(
                read, proposal_logits, qh, b0, b1, v0, v1, source_norm, e)
            tile_max = xp.max(score, axis=-1)
            new_max = xp.maximum(maximum, tile_max)
            old = xp.exp(maximum-new_max)
            weight = xp.exp(score-new_max[:, :, None])
            mass = mass*old + xp.sum(weight, axis=-1)
            value = value*old[:, :, None] + xp.matmul(weight, e)
            maximum = new_max
        output[b0:b1] = (value/mass[:, :, None]).reshape(h, gb, k, dh).transpose(
            1, 2, 0, 3).reshape(gb, k, read.model_width)
        attention_norm[b0:b1] = (maximum+xp.log(mass)).reshape(h, gb, k).transpose(1, 2, 0)
    correction = (output.reshape(-1, read.model_width) @
                  read.output_projection.W.data.astype(read.acc_dtype)).reshape(
                      g, k, read.decoder_width)
    return correction, q, output, attention_norm, source_norm, states


def backward_batched_mixture(read, d_correction, cache):
    """Exact batched VJP: full cross-source logit, bias, query and shared-E grads.

    Recomputes the attention scores for each tile; caches only Q, read output,
    normalizers and logits. Never stores a dense attention matrix in the tape.
    """
    logits = cache.proposal_logits
    g, k, v = logits.shape
    h, dh = read.n_heads, read.head_width
    dy = d_correction.astype(read.acc_dtype)
    flat = dy.reshape(-1, read.decoder_width)
    read.output_projection.W.grad += cache.read.reshape(-1, read.model_width).T @ flat
    d_read = (flat @ read.output_projection.W.data.astype(read.acc_dtype).T).reshape(
        g, k, h, dh)
    d_q = xp.zeros_like(cache.queries, dtype=read.acc_dtype)
    d_logits = xp.zeros((g, k, v), dtype=read.acc_dtype)
    source_total = xp.zeros((g, k), dtype=read.acc_dtype)
    scale = read.read_scale / math.sqrt(dh)
    bias = read.source_bias.data.astype(read.acc_dtype)
    for b0, b1 in _tiles(g, read.block_tile):
        gb = b1-b0
        qflat = xp.ascontiguousarray(cache.queries[b0:b1].transpose(
            2, 0, 1, 3).reshape(h, gb*k, dh))
        upstream = xp.ascontiguousarray(d_read[b0:b1].transpose(
            2, 0, 1, 3).reshape(h, gb*k, dh))
        rval = xp.ascontiguousarray(cache.read[b0:b1].reshape(gb, k, h, dh).transpose(
            2, 0, 1, 3).reshape(h, gb*k, dh))
        row_dot = xp.sum(upstream*rval, axis=-1, keepdims=True)
        norm = cache.attention_norm[b0:b1].transpose(2, 0, 1).reshape(h, gb*k, 1)
        dq_flat = xp.zeros_like(qflat)
        for v0, v1 in _tiles(v, read.vocab_tile):
            vt = v1-v0
            e = xp.ascontiguousarray(
                read.embedding.data[v0:v1].astype(read.acc_dtype).reshape(vt, h, dh)
                .transpose(1, 0, 2))
            score, log_prior, mix = _batched_mixture_tile(
                read, logits, qflat, b0, b1, v0, v1, cache.source_norm, e)
            weight = xp.exp(score-norm)
            d_score = weight * (xp.matmul(upstream, e.transpose(0, 2, 1))-row_dot)
            dq_flat += scale * xp.matmul(d_score, e)
            d_embed = (xp.matmul(weight.transpose(0, 2, 1), upstream) +
                       scale * xp.matmul(d_score.transpose(0, 2, 1), qflat))
            read.embedding.grad[v0:v1] += d_embed.transpose(1, 0, 2).reshape(
                vt, read.model_width)
            d_per_query = d_score.reshape(h, gb, k, vt)
            for r in range(k):
                responsibility = xp.exp(log_prior[None, :, r:r+1, :] +
                                        bias[:, None, :, r:r+1] - mix)
                ds = d_per_query*responsibility
                d_logits[b0:b1, r, v0:v1] += xp.sum(ds, axis=(0, 2))
                source_total[b0:b1, r] += xp.sum(ds, axis=(0, 2, 3))
                read.source_bias.grad[:, :, r] += xp.sum(ds, axis=(1, 3))
        d_q[b0:b1] += dq_flat.reshape(h, gb, k, dh).transpose(1, 2, 0, 3)
    # Backprop through log P, computed per source before source mixing.
    for b0, b1 in _tiles(g, read.block_tile):
        for v0, v1 in _tiles(v, read.vocab_tile):
            prior = xp.exp(logits[b0:b1, :, v0:v1].astype(read.acc_dtype) /
                           read.temperature - cache.source_norm[b0:b1, :, None])
            d_logits[b0:b1, :, v0:v1] -= prior*source_total[b0:b1, :, None]
    d_logits /= read.temperature
    dq_flat = d_q.reshape(-1, read.model_width)
    read.query_projection.W.grad += (
        cache.projection_input.reshape(-1, read.decoder_width).T @ dq_flat)
    d_input = (dq_flat @ read.query_projection.W.data.astype(read.acc_dtype).T).reshape(
        g, k, read.decoder_width)
    return d_input, d_logits
