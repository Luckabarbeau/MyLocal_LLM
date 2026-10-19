"""0082A: model-identity-aware, variable-cardinality probability fusion.

Inputs are *frozen* LM logits, arranged [rows, members, vocabulary].  The reference is a trainable, model-ID-aware probability mixture, initially
identical to the arithmetic mean. An unbounded residual is added in
log-probability space, and its final head is exactly zero at initialization.

The member encoder shares weights across members and uses learned model-ID
embeddings.  Learned vocabulary embeddings are used twice: (1) model/token
identity interactions, and (2) probability-weighted moments for inexpensive
cross-token information.  Every member's entire distribution is retained in
the tokenwise features; the moments supplement rather than replace it.

Explicit float32 backward gradients match the codebase's NumPy/CuPy approach.
This initial implementation intentionally trains the combiner separately from
its (frozen) parent LLMs and returns no gradients to member logits.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import math

import numpy as np

from ..backend import xp, asnumpy, scalar, BACKEND_NAME
from ..parameter import Parameter


def _log_softmax(x, axis=-1):
    shift = x - xp.max(x, axis=axis, keepdims=True)
    return shift - xp.log(xp.sum(xp.exp(shift), axis=axis, keepdims=True))


@dataclass(frozen=True)
class FusionConfig:
    vocab_size: int
    max_models: int = 10
    embed_dim: int = 16
    member_width: int = 8
    decoder_width: int = 24
    decoder_layers: int = 1
    seed: int = 82
    context_heads: int = 0  # 0 preserves 0082I behavior; 2 enables model-query context attention
    context_head_dim: int = 16
    context_hidden: int = 32

    def __post_init__(self):
        if self.context_heads not in (0, 2):
            raise ValueError("context_heads must be 0 (disabled) or 2")
        for key in ("vocab_size", "max_models", "embed_dim", "member_width", "decoder_width", "decoder_layers", "context_head_dim", "context_hidden"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")


class ProbabilityResidualCombiner:
    """No model selector, no bounded residual, and no special case for N=1.

    Identical (model_id, logits) pairs in any order produce the same output.
    Swapping distributions *between* different identities may change output.
    A member ID is a stable integer in [0, max_models); IDs in a call must be
    distinct.  At training time, only registered IDs should be considered
    calibrated; unused embedding slots have not learned specialization.
    """

    def __init__(self, config: FusionConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        d, h, k = config.embed_dim, config.member_width, config.decoder_width
        self.parameters_by_name = {}
        # 0082Q: reused CuPy GEMV reduction vector. Not a model parameter or
        # checkpoint field; refreshed only when the batch geometry changes.
        self._decoder_reduction_ones = None

        def add(name, shape, std=0.05, zero=False, decay=True):
            data = (np.zeros(shape, dtype=np.float32) if zero else
                    rng.normal(0, std, size=shape).astype(np.float32))
            param = Parameter(xp.asarray(data), name=f"fusion.{name}", decay=decay)
            self.parameters_by_name[name] = param
            return param

        self.token_embedding = add("token_embedding", (config.vocab_size, d), std=0.06)
        self.model_embedding = add("model_embedding", (config.max_models, d), std=0.06)
        self.w_log_ratio = add("w_log_ratio", (h,), std=0.08)
        self.w_identity = add("w_identity", (h,), std=0.08)
        self.w_context = add("w_context", (h,), std=0.08)
        self.w_entropy = add("w_entropy", (h,), std=0.08)
        self.member_bias = add("member_bias", (h,), zero=True, decay=False)
        # 2*h = mean and variance of member encodings.
        # Four mathematical features: log P_A, D, log-ratio variance, count.
        self.fusion_weight = add("fusion_weight", (2 * h + 4, k), std=0.06)
        self.fusion_bias = add("fusion_bias", (k,), zero=True, decay=False)
        # Each additional decoder layer uses token-wise k -> k projections.
        # decoder_layers=1 preserves the original 0082A architecture, names,
        # numerical outputs and old model checkpoints.
        self.decoder_hidden_weights = []
        self.decoder_hidden_biases = []
        for layer in range(1, config.decoder_layers):
            self.decoder_hidden_weights.append(
                add(f"decoder_hidden_weight_{layer}", (k, k), std=0.06))
            self.decoder_hidden_biases.append(
                add(f"decoder_hidden_bias_{layer}", (k,), zero=True, decay=False))
        # Exact arithmetic ensemble initialization, independently of input.
        self.output_weight = add("output_weight", (k,), zero=True)
        self.output_bias = add("output_bias", (config.vocab_size,), zero=True, decay=False)
        # Keep last to preserve existing optimizer parameter order. Softmax is
        # applied only over the model IDs participating in a given call.
        self.reference_logits = add("reference_logits", (config.max_models,),
                                    zero=True, decay=False)
        # 0082J: two model-identity queries attend to causally available tokens.
        # Reuse vocabulary/model embeddings. The attention is linear in available
        # context length, NOT vocabulary-sized or sequence-to-sequence attention.
        # Append new params so older optimizer parameter ordering stays stable.
        if config.context_heads:
            n, c = config.context_heads, config.context_head_dim
            self.ctx_q = add("ctx_q", (d, n * c), std=0.05)
            self.ctx_k = add("ctx_k", (d, n * c), std=0.05)
            self.ctx_v = add("ctx_v", (d, n * c), std=0.05)
            # Effective initial recency scales: ~10 and ~200 tokens. Both
            # are *learnable* and independently parameterized per model ID.
            half_lives = np.array([0.1, 0.005], dtype=np.float32)
            init = np.log(np.expm1(half_lives))
            self.ctx_recency = add("ctx_recency", (config.max_models, n), zero=True,
                                   decay=False)
            self.ctx_recency.data[...] = xp.asarray(np.broadcast_to(init, (config.max_models, n)))
            self.ctx_hidden_weight = add("ctx_hidden_weight", (n*c, config.context_hidden), std=0.05)
            self.ctx_hidden_bias = add("ctx_hidden_bias", (config.context_hidden,), zero=True, decay=False)
            self.ctx_score = add("ctx_score", (config.context_hidden,), zero=True)

    def _decoder_bias_gradient(self, dpre):
        """CUDA: GEMV over token positions, without a slow generic axis sum.

        A contiguous [positions,channels] matrix is reduced with a shared
        all-ones vector; only positions*4 bytes are cached (16 MiB for 128x32K).
        NumPy keeps the original exact-reference reduction ordering.
        """
        if BACKEND_NAME != "cupy":
            return xp.sum(dpre, axis=(0, 1))
        positions = int(dpre.shape[0] * dpre.shape[1])
        channels = int(dpre.shape[2])
        ones = self._decoder_reduction_ones
        if ones is None or int(ones.size) != positions:
            ones = xp.ones((positions,), dtype=xp.float32)
            self._decoder_reduction_ones = ones
        return dpre.reshape(positions, channels).T @ ones

    def parameters(self):
        return list(self.parameters_by_name.values())

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def _validate(self, logits, model_ids):
        if getattr(logits, "ndim", None) != 3:
            raise ValueError("logits must have shape [rows, models, vocabulary]")
        rows, members, vocab = (int(x) for x in logits.shape)
        if rows < 1 or members < 1 or vocab != self.config.vocab_size:
            raise ValueError("nonempty rows/models and configured vocabulary are required")
        ids = np.asarray(model_ids, dtype=np.int64)
        if ids.shape != (members,):
            raise ValueError("model_ids must have one entry per input model")
        if len(set(ids.tolist())) != members:
            raise ValueError("model IDs must be unique within a sample")
        if np.any(ids < 0) or np.any(ids >= self.config.max_models):
            raise ValueError("model ID outside configured embedding range")
        return ids

    def member_weights(self, model_ids, *, context_tokens=None, context_lengths=None):
        """Learned nonnegative mixture weights for present model identities.

        Softmax normalization is over the supplied IDs only. Reordering
        (model_id, logits) pairs reorders the weights without changing output.
        """
        ids = np.asarray(model_ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or len(set(ids.tolist())) != len(ids):
            raise ValueError("member_weights requires distinct model IDs")
        if np.any(ids < 0) or np.any(ids >= self.config.max_models):
            raise ValueError("model ID outside configured embedding range")
        if self.config.context_heads:
            weights, _ = self._context_mixture(ids, context_tokens, context_lengths)
            return weights
        scores = self.reference_logits.data[ids]
        shifted = scores - xp.max(scores)
        numerator = xp.exp(shifted)
        return numerator / xp.sum(numerator)

    def _context_mixture(self, ids, context_tokens, context_lengths, *, cache=False):
        """Causal model-query attention over preceding *tokens*, not LM states.

        context_tokens is [rows, padded_tokens], context_lengths [rows].
        Row r can attend only to tokens at indices < context_lengths[r].
        No fixed maximum sequence length is part of the fusion model.
        """
        if context_tokens is None or context_lengths is None:
            raise ValueError("context attention requires context_tokens and context_lengths")
        ids = np.asarray(ids, dtype=np.int64)
        tok = np.asarray(asnumpy(context_tokens), dtype=np.int64)
        lengths = np.asarray(asnumpy(context_lengths), dtype=np.int64)
        if tok.ndim != 2 or lengths.shape != (tok.shape[0],) or tok.shape[1] < 1:
            raise ValueError("context_tokens [rows,time] and context_lengths [rows] required")
        if np.any(tok < 0) or np.any(tok >= self.config.vocab_size):
            raise ValueError("context token outside vocabulary")
        if np.any(lengths < 1) or np.any(lengths > tok.shape[1]):
            raise ValueError("invalid causal context lengths")
        xp_tokens = xp.asarray(tok)
        e = self.token_embedding.data[xp_tokens]              # [R,T,d]
        identities = self.model_embedding.data[ids]           # [N,d]
        rows, time = tok.shape
        n, dhead = self.config.context_heads, self.config.context_head_dim
        q = (identities @ self.ctx_q.data).reshape(len(ids), n, dhead)
        keys = (e @ self.ctx_k.data).reshape(rows, time, n, dhead)
        values = (e @ self.ctx_v.data).reshape(rows, time, n, dhead)
        scale = 1.0 / math.sqrt(dhead)
        scores = xp.einsum('rthd,mhd->rmht', keys, q) * scale
        rho = self.ctx_recency.data[ids]
        recency = xp.logaddexp(xp.float32(0.0), rho)
        age = xp.asarray(lengths[:, None] - 1 - np.arange(time)[None, :], dtype=xp.float32)
        scores = scores - recency[None, :, :, None] * age[:, None, None, :]
        mask = xp.asarray(np.arange(time)[None, :] < lengths[:, None])
        scores = xp.where(mask[:, None, None, :], scores, xp.float32(-1e20))
        attn = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
        attn /= xp.sum(attn, axis=-1, keepdims=True)
        context = xp.einsum('rmht,rthd->rmhd', attn, values).reshape(rows, len(ids), n*dhead)
        # Model identity influences the query, not an independent shortcut
        # through the reliability MLP; the latter must interpret attended text.
        summary = context
        hidden = xp.tanh(summary @ self.ctx_hidden_weight.data + self.ctx_hidden_bias.data)
        adjustments = hidden @ self.ctx_score.data
        logits = self.reference_logits.data[ids][None, :] + adjustments
        logw = _log_softmax(logits, axis=1)
        weights = xp.exp(logw)
        if not cache:
            return weights, None
        return weights, {'ids': ids, 'tokens': xp_tokens, 'embedding': e,
                         'identities': identities, 'q': q, 'keys': keys, 'values': values,
                         'rho': rho, 'recency': recency, 'age': age, 'attn': attn,
                         'summary': summary, 'hidden': hidden}

    def _context_backward(self, dscore, cache):
        """Exact gradient for the contextual mixture, including embedding rows."""
        hidden = cache['hidden']
        d = self.config.embed_dim
        n, dh = self.config.context_heads, self.config.context_head_dim
        rows, members = dscore.shape
        self.ctx_score.grad += xp.sum(hidden * dscore[..., None], axis=(0, 1))
        dhidden = dscore[..., None] * self.ctx_score.data * (1.0 - hidden ** 2)
        self.ctx_hidden_weight.grad += cache['summary'].reshape(-1, n*dh).T @ dhidden.reshape(-1, self.config.context_hidden)
        self.ctx_hidden_bias.grad += xp.sum(dhidden, axis=(0, 1))
        dsummary = dhidden @ self.ctx_hidden_weight.data.T
        dcontext = dsummary.reshape(rows, members, n, dh)
        attn, values, keys, q = cache['attn'], cache['values'], cache['keys'], cache['q']
        dvalues = xp.einsum('rmht,rmhd->rthd', attn, dcontext)
        dattention = xp.einsum('rmhd,rthd->rmht', dcontext, values)
        dscores = attn * (dattention - xp.sum(attn*dattention, axis=-1, keepdims=True))
        dr = -xp.sum(dscores * cache['age'][:, None, None, :], axis=(0, 3))
        sigmoid = 1.0 / (1.0 + xp.exp(-cache['rho']))
        for j, mid in enumerate(cache['ids']):
            self.ctx_recency.grad[int(mid)] += dr[j] * sigmoid[j]
        scale = 1.0 / math.sqrt(dh)
        dq = xp.einsum('rmht,rthd->mhd', dscores, keys) * scale
        dkeys = xp.einsum('rmht,mhd->rthd', dscores, q) * scale
        e = cache['embedding']
        ef = e.reshape(-1, d)
        dk = dkeys.reshape(-1, n*dh)
        dv = dvalues.reshape(-1, n*dh)
        self.ctx_k.grad += ef.T @ dk
        self.ctx_v.grad += ef.T @ dv
        de = (dk @ self.ctx_k.data.T + dv @ self.ctx_v.data.T)
        xp.add.at(self.token_embedding.grad, cache['tokens'].reshape(-1), de)
        dqf = dq.reshape(members, n*dh)
        self.ctx_q.grad += cache['identities'].T @ dqf
        didentities = dqf @ self.ctx_q.data.T
        for j, mid in enumerate(cache['ids']):
            self.model_embedding.grad[int(mid)] += didentities[j]

    def forward(self, logits, model_ids, *, return_cache=False, context_tokens=None,
                context_lengths=None, check_finite=True, profile=None):
        """Return [rows, vocab] probabilities; optionally retain backward cache.

        The full-vocabulary inputs should be presented in small row tiles for
        VRAM use.  Float32 is deliberate: 10 full distributions and their
        member embeddings would be costly if cached for 4K positions at once.
        """
        ids = self._validate(logits, model_ids)
        z = xp.asarray(logits, dtype=xp.float32)
        # Streaming fast path checks the host-side exported batch BEFORE upload,
        # avoiding an otherwise mandatory device synchronization per optimizer step.
        # Public forward defaults to the original strict GPU validation.
        if check_finite and not bool(scalar(xp.all(xp.isfinite(z)))):
            raise ValueError("LM logits must be finite")
        rows, members, vocab = z.shape
        log_p = _log_softmax(z)
        p = xp.exp(log_p)
        if profile is not None:
            profile.mark('forward.input_softmax')
        pa = xp.mean(p, axis=1)
        # Arithmetic/geometric averages remain independent auxiliary features
        # and reportable baselines. Only the skip connection is weighted.
        log_pa = xp.log(xp.maximum(pa, xp.float32(1e-37)))
        if self.config.context_heads:
            if context_tokens is None or len(context_tokens) != rows:
                raise ValueError('context_tokens must match logit rows')
            weights, context_cache = self._context_mixture(
                ids, context_tokens, context_lengths, cache=return_cache)
            if profile is not None:
                profile.mark('forward.model_context_attention')
            weighted = xp.sum(p * weights[:, :, None], axis=1)
        else:
            if context_tokens is not None or context_lengths is not None:
                raise ValueError('context inputs provided to a context-disabled combiner')
            weights, context_cache = self.member_weights(ids), None
            weighted = xp.sum(p * weights[None, :, None], axis=1)
        log_weighted = xp.log(xp.maximum(weighted, xp.float32(1e-37)))
        log_pg = _log_softmax(xp.mean(log_p, axis=1))
        difference = log_pg - log_pa
        log_ratio = log_p - log_pa[:, None, :]
        log_ratio_variance = xp.var(log_ratio, axis=1)
        # Center entropy across members so magnitude is comparable when N varies.
        entropy = -xp.sum(p * log_p, axis=2)
        entropy_centered = entropy - xp.mean(entropy, axis=1, keepdims=True)
        if profile is not None:
            profile.mark('forward.reference_features')

        e = self.token_embedding.data
        identities = self.model_embedding.data[ids]
        scale = 1.0 / math.sqrt(self.config.embed_dim)
        model_token = (identities @ e.T) * scale               # [N,V]
        moments = p @ e                                         # [R,N,d]
        token_context = xp.matmul(moments, e.T) * scale        # [R,N,V]
        if profile is not None:
            profile.mark('forward.vocabulary_moments')
        if BACKEND_NAME == "cupy":
            # 0082N: one fused GPU launch replaces many huge broadcast
            # intermediates, tanh, mean, and variance kernels. No opt-in:
            # FP32 math and learned features are unchanged.
            from .fusion_cuda_kernels import member_encode_aggregate
            member, member_mean, member_var = member_encode_aggregate(
                log_ratio, model_token, token_context, entropy_centered,
                self.w_log_ratio.data, self.w_identity.data,
                self.w_context.data, self.w_entropy.data, self.member_bias.data)
        else:
            pre = (log_ratio[..., None] * self.w_log_ratio.data +
                   model_token[None, ..., None] * self.w_identity.data +
                   token_context[..., None] * self.w_context.data +
                   entropy_centered[:, :, None, None] * self.w_entropy.data +
                   self.member_bias.data)
            member = xp.tanh(pre)
            member_mean = xp.mean(member, axis=1)
            member_var = xp.mean((member - member_mean[:, None, ...]) ** 2, axis=1)
        if profile is not None:
            profile.mark('forward.member_encoder_aggregate')
        count = xp.full_like(log_pa, math.log1p(members) / math.log(11.0))
        features = xp.concatenate((
            member_mean, member_var,
            log_pa[..., None] / 10.0,
            difference[..., None] / 10.0,
            log_ratio_variance[..., None] / 100.0,
            count[..., None],
        ), axis=-1)
        if profile is not None:
            profile.mark('forward.assemble_features')
        decoder = xp.tanh(features @ self.fusion_weight.data + self.fusion_bias.data)
        decoder_states = [decoder]
        for weight, bias in zip(self.decoder_hidden_weights, self.decoder_hidden_biases):
            decoder = xp.tanh(decoder @ weight.data + bias.data)
            decoder_states.append(decoder)
        residual = decoder @ self.output_weight.data + self.output_bias.data
        if profile is not None:
            profile.mark('forward.residual_decoder')
        log_pc = _log_softmax(log_weighted + residual)
        result = xp.exp(log_pc)
        if profile is not None:
            profile.mark('forward.output_softmax')
        if not return_cache:
            return result
        return result, {
            "ids": ids, "p": p, "log_pa": log_pa, "log_pg": log_pg,
            "log_weighted": log_weighted, "weighted": weighted,
            "weights": weights, "context_cache": context_cache,
            "difference": difference, "log_ratio": log_ratio,
            "entropy_centered": entropy_centered, "model_token": model_token,
            "moments": moments, "token_context": token_context,
            "member": member, "member_mean": member_mean,
            "features": features, "decoder": decoder,
            "decoder_states": decoder_states, "final": result,
        }

    def loss_and_backward(self, logits, model_ids, targets, *, return_weighted=False,
                          context_tokens=None, context_lengths=None,
                          defer_metrics=False, check_finite=True, profile=None, gradient_scale=1.0):
        """Mean next-token CE; accumulate exact FP32 parameter gradients.

        Returns (fusion_ce, arithmetic_ce), plus weighted_ce when requested.
        defer_metrics=True returns backend-resident scalar arrays rather than
        synchronizing to Python floats every step. The caller must read them
        *before reusing/overwriting the corresponding GPU memory*.
        check_finite=False is only valid after an independent host-side finite
        validation of the *same* logits. Teacher models remain frozen.
        """
        pc, cache = self.forward(logits, model_ids, return_cache=True,
                                 context_tokens=context_tokens, context_lengths=context_lengths,
                                 check_finite=check_finite, profile=profile)
        rows, vocab = pc.shape
        y = np.asarray(targets, dtype=np.int64)
        if y.shape != (rows,) or np.any(y < 0) or np.any(y >= vocab):
            raise ValueError("targets must be valid [rows] vocabulary indices")
        indices = xp.arange(rows)
        targets_xp = xp.asarray(y)
        # Logits-space loss is stable even when softmax underflows in FP32.
        final_logits = cache["log_weighted"] + (cache["decoder"] @ self.output_weight.data) + self.output_bias.data
        log_final = _log_softmax(final_logits)
        loss = -xp.mean(log_final[indices, targets_xp])
        baseline = -xp.mean(cache["log_pa"][indices, targets_xp])
        weighted_ce = -xp.mean(cache["log_weighted"][indices, targets_xp]) if return_weighted else None
        if not defer_metrics:
            loss = scalar(loss)
            baseline = scalar(baseline)
            if return_weighted:
                weighted_ce = scalar(weighted_ce)
        dz = pc.copy()
        dz[indices, targets_xp] -= 1.0
        # One microbatch contributes its fraction of the *total supervised rows*
        # to the accumulated mean gradient.  The loss values returned below
        # remain unscaled for correct row-weighted metric aggregation.
        if not math.isfinite(float(gradient_scale)) or gradient_scale <= 0:
            raise ValueError('gradient_scale must be finite and positive')
        dz *= float(gradient_scale) / float(rows)
        if profile is not None:
            profile.mark('backward.loss_and_seed')

        # Derivative of log(sum_i alpha_i * p_i(v)) w.r.t. per-ID mixture
        # logits, including softmax coupling of the participating weights.
        # Accumulate one member at a time to avoid a [rows,N,vocab] temporary.
        weighted = cache["weighted"]
        valid = weighted > xp.float32(1e-37)
        dbase = xp.where(valid, dz / xp.maximum(weighted, xp.float32(1e-37)), 0.0)
        weights = cache["weights"]
        if self.config.context_heads:
            # dbase is dCE/dP_W; score gradient must retain the row axis.
            dw_rows = xp.stack([xp.sum(dbase * cache['p'][:, j, :], axis=1)
                                for j in range(len(cache['ids']))], axis=1)
            dscore = weights * (dw_rows - xp.sum(weights * dw_rows, axis=1, keepdims=True))
            for j, model_id in enumerate(cache['ids']):
                self.reference_logits.grad[int(model_id)] += xp.sum(dscore[:, j])
            if profile is not None:
                profile.mark('backward.mixture_weights')
            self._context_backward(dscore, cache['context_cache'])
            if profile is not None:
                profile.mark('backward.model_context_attention')
        else:
            dweights = xp.stack([xp.sum(dbase * cache["p"][:, j, :])
                                 for j in range(len(cache["ids"]))])
            dscore = weights * (dweights - xp.sum(weights * dweights))
            for j, model_id in enumerate(cache["ids"]):
                self.reference_logits.grad[int(model_id)] += dscore[j]
            if profile is not None:
                profile.mark('backward.mixture_weights')

        k = self.config.decoder_width
        positions = rows * vocab
        if BACKEND_NAME == "cupy":
            # 0082Q: cuBLAS GEMV instead of materializing a [R,V,K] broadcast
            # product and reducing it to K scalar weight gradients.
            self.output_weight.grad += (cache["decoder"].reshape(positions, k).T @
                                        dz.reshape(positions))
        else:
            self.output_weight.grad += xp.sum(cache["decoder"] * dz[..., None], axis=(0, 1))
        if profile is not None and BACKEND_NAME == "cupy":
            profile.mark('backward.output_head.weight_grad')
        self.output_bias.grad += xp.sum(dz, axis=0)
        if profile is not None and BACKEND_NAME == "cupy":
            profile.mark('backward.output_head.bias_grad')
        # Reverse through the optional extra token-wise decoder layers.
        states = cache["decoder_states"]
        if BACKEND_NAME == "cupy" and not self.decoder_hidden_weights:
            from .fusion_cuda_kernels import decoder_first_layer_backward
            # Fuse (dz * w_out) * (1 - tanh^2) without a full-size ddecoder.
            dpre_decoder = decoder_first_layer_backward(states[0], dz,
                                                        self.output_weight.data)
        else:
            ddecoder = dz[..., None] * self.output_weight.data
            for j in range(len(self.decoder_hidden_weights) - 1, -1, -1):
                layer_grad = ddecoder * (1.0 - states[j + 1] ** 2)
                self.decoder_hidden_weights[j].grad += (
                    states[j].reshape(-1, k).T @ layer_grad.reshape(-1, k))
                self.decoder_hidden_biases[j].grad += self._decoder_bias_gradient(layer_grad)
                ddecoder = layer_grad @ self.decoder_hidden_weights[j].data.T
            dpre_decoder = ddecoder * (1.0 - states[0] ** 2)
        if profile is not None:
            if BACKEND_NAME == "cupy":
                profile.mark('backward.output_head.propagate_and_decoder_activation')
            else:
                profile.mark('backward.output_head')
        n_features = 2 * self.config.member_width + 4
        self.fusion_weight.grad += (cache["features"].reshape(-1, n_features).T @
                                    dpre_decoder.reshape(-1, k))
        if profile is not None and BACKEND_NAME == "cupy":
            profile.mark('backward.decoder.weight_grad')
        self.fusion_bias.grad += self._decoder_bias_gradient(dpre_decoder)
        if profile is not None and BACKEND_NAME == "cupy":
            profile.mark('backward.decoder.bias_grad')
        dfeatures = dpre_decoder @ self.fusion_weight.data.T
        if profile is not None:
            if BACKEND_NAME == "cupy":
                profile.mark('backward.decoder.features_grad')
            else:
                profile.mark('backward.decoder')
        h = self.config.member_width
        member = cache["member"]
        members = member.shape[1]
        if BACKEND_NAME == "cupy":
            from .fusion_cuda_kernels import member_backward
            dpre = member_backward(member, cache["member_mean"], dfeatures)
        else:
            dmean = dfeatures[..., :h]
            dvar = dfeatures[..., h:2*h]
            dmember = (dmean[:, None, :, :] +
                       2.0 * dvar[:, None, :, :] *
                       (member - cache["member_mean"][:, None, :, :])) / members
            dpre = dmember * (1.0 - member ** 2)
        if profile is not None:
            profile.mark('backward.member_encoder')
        ratio = cache["log_ratio"]
        if BACKEND_NAME == "cupy":
            # 0082P: one fused gradient pass replaces five vocabulary-wide
            # broadcast/reduce operations plus two feature-gradient reductions.
            # This optimization has no runtime flag or model-size trade-off.
            from .fusion_cuda_kernels import member_parameter_grads
            g_ratio, g_identity, g_context, g_entropy, g_bias, did, dcontext = (
                member_parameter_grads(
                    dpre, ratio, cache["model_token"], cache["token_context"],
                    cache["entropy_centered"],
                    self.w_identity.data, self.w_context.data))
            self.w_log_ratio.grad += g_ratio
            self.w_identity.grad += g_identity
            self.w_context.grad += g_context
            self.w_entropy.grad += g_entropy
            self.member_bias.grad += g_bias
        else:
            self.w_log_ratio.grad += xp.sum(dpre * ratio[..., None], axis=(0, 1, 2))
            self.w_identity.grad += xp.sum(dpre * cache["model_token"][None, ..., None], axis=(0, 1, 2))
            self.w_context.grad += xp.sum(dpre * cache["token_context"][..., None], axis=(0, 1, 2))
            self.w_entropy.grad += xp.sum(dpre * cache["entropy_centered"][:, :, None, None], axis=(0, 1, 2))
            self.member_bias.grad += xp.sum(dpre, axis=(0, 1, 2))
            did = xp.sum(dpre * self.w_identity.data, axis=-1)       # [R,N,V]
            dcontext = xp.sum(dpre * self.w_context.data, axis=-1)  # [R,N,V]
        e = self.token_embedding.data
        identities = self.model_embedding.data[cache["ids"]]
        scale = 1.0 / math.sqrt(self.config.embed_dim)
        did_summed = xp.sum(did, axis=0)                        # [N,V]
        id_grads = (did_summed @ e) * scale                     # [N,d]
        for j, model_id in enumerate(cache["ids"]):
            self.model_embedding.grad[int(model_id)] += id_grads[j]
        if profile is not None:
            profile.mark('backward.member_parameter_grads')
        # The token embedding participates in model-token identities AND in
        # cross-token moments. Include the direct and moment-transpose terms.
        self.token_embedding.grad += (did_summed.T @ identities) * scale
        self.token_embedding.grad += (dcontext.reshape(-1, vocab).T @ cache["moments"].reshape(-1, self.config.embed_dim)) * scale
        dmoments = xp.matmul(dcontext, e) * scale               # [R,N,d]
        self.token_embedding.grad += (cache["p"].reshape(-1, vocab).T @ dmoments.reshape(-1, self.config.embed_dim))
        if profile is not None:
            profile.mark('backward.vocabulary_embedding_grads')
        if return_weighted:
            return loss, baseline, weighted_ce
        return loss, baseline

    def state_dict(self):
        return {name: asnumpy(param.data).copy() for name, param in self.parameters_by_name.items()}

    def load_state_dict(self, state):
        expected = set(self.parameters_by_name)
        supplied = set(state)
        # Older 0082A-H best-model checkpoints have no mixture weights.
        # Loading them should produce the exact previous arithmetic reference.
        optional_missing = {"reference_logits", "ctx_q", "ctx_k", "ctx_v",
                            "ctx_recency", "ctx_hidden_weight", "ctx_hidden_bias", "ctx_score"}
        if supplied - expected or expected - supplied - optional_missing:
            raise ValueError(f"parameter keys mismatch: missing={expected-supplied}, extra={supplied-expected}")
        for name, param in self.parameters_by_name.items():
            if name not in state:
                # Keep initialized contextual attention projections/recency
                # when upgrading a trained 0082I checkpoint to 0082J.
                if name == 'reference_logits':
                    param.data[...] = 0.0
                continue
            source = np.asarray(state[name], dtype=np.float32)
            if source.shape != param.shape:
                raise ValueError(f"wrong shape for {name}: {source.shape} != {param.shape}")
            param.data[...] = xp.asarray(source)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        payload["__config_json__"] = np.asarray(json.dumps(self.config.__dict__))
        np.savez(path, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            config = FusionConfig(**json.loads(str(archive["__config_json__"])))
            model = cls(config)
            model.load_state_dict({k: archive[k] for k in archive.files if k != "__config_json__"})
        return model
