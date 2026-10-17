"""0080AF–AI: full-soft vocabulary reads with explicit VJP.

Default NumPy execution is the tiny-vocabulary mathematical reference.
0080AH+AI adds a shared NumPy/CuPy tiled implementation for both own- and
all-source modes, including stable per-source logsoftmax for the latter.
The tiled implementation recomputes bounded score tiles during backward and
keeps neither full attention weights nor vocabulary-by-embedding tensors.
Training integration remains experimental until cached generation and v2
checkpoint migration pass their separate parity audits.
"""

from dataclasses import dataclass
import math

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, RandomStream
from mini_llm.ops.linear import Linear
from mini_llm.parameter import Parameter


@dataclass(frozen=True)
class OwnVocabReadCache:
    """Per-call forward tape: no cached probabilities, mutable global tape, or E copy."""

    proposal_logits: object
    queries: object
    read: object
    query_linear_cache: dict
    output_linear_cache: dict
    attention_norm: object = None
    source_norm: object = None
    projection_input: object = None
    all_source_algorithm: str = None


class SoftVocabularyRead:
    """Full-vocabulary own/all-source read with two learned projections.

    ``shared_embeddings`` is the *same* Parameter as the main token embedding;
    it is deliberately not part of ``parameters()`` to avoid duplicate optimizer
    registration. Its FP32/FP64 gradients are nonetheless accumulated in
    ``shared_embeddings.grad`` through both the attention keys and values.

    The output projection is exactly zero on construction. Consequently a
    decoder residual addition initially has zero effect, while its output
    projection receives a gradient from the first nonzero upstream signal.
    As with any zero-output residual, the query projection receives its first
    nonzero gradient after the output weights become nonzero.

    The dense reference supports NumPy float32/64 only and tiny vocabulary.
    The streamed path supports the same equations with FP32 accumulation on
    CuPy including BF16 weights. Neither path enables production CLI yet.
    """

    def __init__(self, shared_embeddings: Parameter, decoder_width: int, *,
                 n_heads: int = 4, temperature: float = 1.0,
                 read_scale: float = 1.0, rng_seed: int = 8091,
                 name: str = 'block_decoder.soft_vocab_read',
                 source_scope: str = 'own',
                 max_reference_vocab: int = 512,
                 max_reference_scores: int = 500_000, block_size: int = None,
                 vocab_tile: int = 2048, block_tile: int = 8, execution: str = "auto",
                 all_source_algorithm: str = "source_mixture"):
        if source_scope not in ('own', 'all'):
            raise ValueError('source_scope must be own or all')
        if source_scope == 'all' and (not isinstance(block_size, int) or block_size < 1):
            raise ValueError('all-source requires a positive block_size')
        if execution not in ('auto', 'reference', 'tiled'):
            raise ValueError('execution must be auto, reference, or tiled')
        if execution == 'reference' and BACKEND_NAME != 'numpy':
            raise NotImplementedError('dense reference requires NumPy')
        if all_source_algorithm not in ('source_mixture', 'source_loop', 'batched_mixture', 'fused_mixture'):
            raise ValueError('all_source_algorithm must be source_mixture, source_loop, batched_mixture or fused_mixture')
        # Source_loop is the verified 0080AH+AI kernel path.  The exact
        # source_mixture factorization collapses source before the Q@E GEMM.
        # Keeping all three permits numerical and timing comparisons on one GPU.
        if source_scope != 'all' and all_source_algorithm in ('batched_mixture', 'fused_mixture'):
            raise ValueError('batched/fused_mixture requires all-source scope')
        self.all_source_algorithm = all_source_algorithm
        self.execution = ('tiled' if BACKEND_NAME == 'cupy' else 'reference') if execution == 'auto' else execution
        if all_source_algorithm in ('batched_mixture', 'fused_mixture') and self.execution != 'tiled':
            raise ValueError('batched/fused_mixture requires tiled execution')
        if int(vocab_tile) < 1 or int(block_tile) < 1:
            raise ValueError('positive vocabulary and block tile sizes required')
        self.vocab_tile = int(vocab_tile)
        self.block_tile = int(block_tile)
        self.block_size = block_size
        if not isinstance(shared_embeddings, Parameter) or shared_embeddings.data.ndim != 2:
            raise TypeError('shared_embeddings must be a [V,model_width] Parameter')
        v, model_width = map(int, shared_embeddings.data.shape)
        if v < 1 or (self.execution == 'reference' and v > int(max_reference_vocab)):
            raise ValueError('dense CPU reference requires a tiny vocabulary')
        if not isinstance(n_heads, int) or n_heads < 1 or model_width % n_heads:
            raise ValueError('model embedding width must be divisible by positive read heads')
        if not isinstance(decoder_width, int) or decoder_width < 1:
            raise ValueError('decoder width must be a positive integer')
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not math.isfinite(float(read_scale)) or read_scale < 0:
            raise ValueError('read_scale must be finite and nonnegative')
        if (not isinstance(max_reference_scores, int) or max_reference_scores < 1):
            raise ValueError('max_reference_scores must be positive')
        if (self.execution == 'reference' and
                shared_embeddings.data.dtype not in (np.dtype('float32'), np.dtype('float64'))):
            raise TypeError('dense reference requires float32 or float64 embeddings')
        self.acc_dtype = (xp.float64 if BACKEND_NAME == "numpy" and
                          shared_embeddings.data.dtype == np.dtype("float64")
                          else xp.float32)
        self.embedding = shared_embeddings
        self.model_width = model_width
        self.decoder_width = decoder_width
        self.n_heads = n_heads
        self.head_width = model_width // n_heads
        self.temperature = float(temperature)
        self.read_scale = float(read_scale)
        self.max_reference_scores = max_reference_scores
        self.source_scope = source_scope
        self.source_bias = (Parameter(xp.zeros((n_heads, block_size, block_size),
                                               dtype=shared_embeddings.data.dtype),
                                      name=f"{name}.source_bias", decay=False)
                            if source_scope == "all" else None)
        rng = RandomStream(rng_seed)  # never consume the legacy decoder RNG
        dtype = shared_embeddings.data.dtype
        self.query_projection = Linear(decoder_width, model_width, 0.02, rng,
                                       name=f'{name}.query', dtype=dtype)
        self.output_projection = Linear(model_width, decoder_width, 0.0, rng,
                                        name=f'{name}.output', dtype=dtype)

    def parameters(self):
        """Module-owned parameters only; embedding is already main-owned."""
        params = self.query_projection.parameters() + self.output_projection.parameters()
        if self.source_bias is not None:
            params.append(self.source_bias)
        return params

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def _validate_forward(self, decoder_states, proposal_logits):
        if (decoder_states.ndim != 3 or proposal_logits.ndim != 3 or
                decoder_states.shape[:2] != proposal_logits.shape[:2] or
                decoder_states.shape[-1] != self.decoder_width or
                proposal_logits.shape[-1] != self.embedding.data.shape[0]):
            raise ValueError('expected DEC [G,K,decoder_width] and logits [G,K,V]')
        g, k, v = map(int, proposal_logits.shape)
        if not g or not k:
            raise ValueError('empty decoder block batch or horizon')
        if self.source_scope == "all" and k != self.block_size:
            raise ValueError("proposal K differs from source-bias block size")
        if (self.execution == 'reference' and
                g * k * self.n_heads * v * (k if self.source_scope == 'all' else 1)
                > self.max_reference_scores):
            raise ValueError('dense reference score budget exceeded; use tiled execution')
        if (self.execution == 'reference' and
                (decoder_states.dtype != self.embedding.data.dtype or
                 proposal_logits.dtype != self.embedding.data.dtype)):
            raise TypeError('reference states, logits and embeddings must share float32/float64 dtype')
        if self.execution == 'reference' and (not np.all(np.isfinite(decoder_states)) or
                                               not np.all(np.isfinite(proposal_logits))):
            raise ValueError('nonfinite states or proposal logits')
        if self.execution == "reference" and not np.all(np.isfinite(self.embedding.data)):
            raise ValueError('nonfinite embedding table')

    def _one_head(self, queries, logits, embedding_slice):
        # [G,K,d_head] @ [d_head,V] -> [G,K,V]; never [G,K,V,d_head].
        scale = self.read_scale / math.sqrt(self.head_width)
        scores = (logits / self.temperature + scale *
                  np.matmul(queries, embedding_slice.T))
        # Per-row FP32/FP64 stable softmax. Avoid exp(log(P)) on BF16.
        scores -= np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        read = np.matmul(weights, embedding_slice)
        return weights, read

    def _all_head(self, queries, logits, embedding_slice, head):
        # Small reference only: normalizes each source's P separately, then
        # normalizes attention over the *joint* [source, vocabulary] axes.
        g, k, v = logits.shape
        scaled = logits / self.temperature
        high = np.max(scaled, axis=-1, keepdims=True)
        log_norm = high + np.log(np.sum(np.exp(scaled-high), axis=-1, keepdims=True))
        log_prior = scaled-log_norm
        bias = self.source_bias.data[head]  # [target, source]
        score = (log_prior[:, None, :, :] + bias[None, :, :, None] +
                 (self.read_scale / math.sqrt(self.head_width)) *
                 (queries @ embedding_slice.T)[:, :, None, :])
        flat = score.reshape(g, k, k*v)
        flat = flat - np.max(flat, axis=-1, keepdims=True)
        weight = np.exp(flat)
        weight /= np.sum(weight, axis=-1, keepdims=True)
        weight = weight.reshape(g,k,k,v)
        read = np.matmul(np.sum(weight, axis=2), embedding_slice)
        return weight, read, np.exp(log_prior)

    def forward(self, decoder_states, proposal_logits):
        """Return [G,K,decoder_width] residual and independently owned tape."""
        self._validate_forward(decoder_states, proposal_logits)
        if self.execution == 'tiled':
            if self.source_scope == 'all' and self.all_source_algorithm == 'fused_mixture' and BACKEND_NAME == 'cupy':
                from .fused_vocab_cuda import forward_fused_mixture
                impl = forward_fused_mixture
            elif self.source_scope == 'all' and self.all_source_algorithm in ('batched_mixture', 'fused_mixture'):
                from .streamed_vocab_read import forward_batched_mixture
                impl = forward_batched_mixture
            else:
                from .streamed_vocab_read import forward_tiled
                impl = forward_tiled
            residual, q, output, a_norm, s_norm, states = impl(
                self, decoder_states, proposal_logits)
            return residual, OwnVocabReadCache(
                proposal_logits=proposal_logits.copy(), queries=q, read=output,
                query_linear_cache={}, output_linear_cache={},
                attention_norm=a_norm, source_norm=s_norm, projection_input=states,
                all_source_algorithm=self.all_source_algorithm)
        queries_flat, q_cache = self.query_projection.forward(decoder_states)
        g, k, _ = queries_flat.shape
        queries = queries_flat.reshape(g, k, self.n_heads, self.head_width)
        read = np.empty((g, k, self.model_width), dtype=self.embedding.data.dtype)
        for m in range(self.n_heads):
            s = slice(m * self.head_width, (m + 1) * self.head_width)
            if self.source_scope == 'own':
                _, read[:, :, s] = self._one_head(
                    queries[:, :, m, :], proposal_logits, self.embedding.data[:, s])
            else:
                _, read[:, :, s], _ = self._all_head(
                    queries[:, :, m, :], proposal_logits, self.embedding.data[:, s], m)
        correction, out_cache = self.output_projection.forward(read)
        return correction, OwnVocabReadCache(
            proposal_logits=np.array(proposal_logits, copy=True),
            queries=queries, read=read, query_linear_cache=q_cache,
            output_linear_cache=out_cache)

    def backward(self, d_correction, cache: OwnVocabReadCache):
        """Manually differentiate output, query, proposal logits and shared E.

        Uses dS_v=A_v*(g @ (E_v-R)); dL_v=dS_v/temperature; dQ=beta/sqrt(d)
        * sum_v dS_v*E_v; and dE_v=A_v*g + beta/sqrt(d)*dS_v*Q.
        Returns ``(d_decoder_states, d_proposal_logits)``. These gradients
        must be added to the residual identity and existing proposal pathways
        by the later decoder integration. Parameter gradients accumulate.
        """
        if not isinstance(cache, OwnVocabReadCache):
            raise TypeError('backward expects the forward read cache')
        if self.execution == 'tiled':
            if cache.all_source_algorithm != self.all_source_algorithm:
                raise ValueError('all_source_algorithm changed between forward and backward')
            if self.source_scope == 'all' and self.all_source_algorithm == 'fused_mixture' and BACKEND_NAME == 'cupy':
                from .fused_vocab_cuda import backward_fused_mixture
                return backward_fused_mixture(self, d_correction, cache)
            if self.source_scope == 'all' and self.all_source_algorithm in ('batched_mixture', 'fused_mixture'):
                from .streamed_vocab_read import backward_batched_mixture
                return backward_batched_mixture(self, d_correction, cache)
            from .streamed_vocab_read import backward_tiled
            return backward_tiled(self, d_correction, cache)
        expected_shape = (*cache.read.shape[:2], self.decoder_width)
        if d_correction.shape != expected_shape:
            raise ValueError('upstream read gradient has wrong shape')
        if d_correction.dtype != self.embedding.data.dtype:
            raise TypeError('upstream reference gradient dtype mismatch')
        g, k, v = cache.proposal_logits.shape
        d_read = self.output_projection.backward(d_correction, cache.output_linear_cache)
        d_queries = np.zeros_like(cache.queries)
        d_logits = np.zeros((g, k, v), dtype=self.embedding.data.dtype)
        scale = self.read_scale / math.sqrt(self.head_width)
        for m in range(self.n_heads):
            s = slice(m * self.head_width, (m + 1) * self.head_width)
            keys_values = self.embedding.data[:, s]  # shared key AND value
            q = cache.queries[:, :, m, :]
            upstream = d_read[:, :, s]
            if self.source_scope == 'own':
                weights, read = self._one_head(q, cache.proposal_logits, keys_values)
            else:
                weights, read, source_p = self._all_head(
                    q, cache.proposal_logits, keys_values, m)
            # (g @ E_v) - (g @ R), no [G,K,V,d_head] intermediates.
            if self.source_scope == 'own':
                d_score = weights * (np.matmul(upstream, keys_values.T) -
                                     np.sum(upstream * read, axis=-1, keepdims=True))
                d_logits += d_score / self.temperature
                effective_weight, effective_dscore = weights, d_score
            else:
                dot = np.matmul(upstream, keys_values.T)
                d_score = weights * (dot[:, :, None, :] -
                                     np.sum(upstream*read, axis=-1, keepdims=True)[:, :, :, None])
                d_per_source = np.sum(d_score, axis=1)  # [G,source,V]
                d_logits += (d_per_source - source_p*np.sum(d_per_source,axis=-1,keepdims=True)) / self.temperature
                self.source_bias.grad[m] += np.sum(d_score, axis=(0,3))
                effective_weight = np.sum(weights, axis=2)
                effective_dscore = np.sum(d_score, axis=2)
            d_queries[:, :, m, :] = scale * np.matmul(effective_dscore, keys_values)
            # Both value and key terms; accumulate in the shared parameter.
            d_embedding = (np.einsum('gkv,gkd->vd', effective_weight, upstream) +
                           scale * np.einsum('gkv,gkd->vd', effective_dscore, q))
            self.embedding.grad[:, s] += d_embedding.astype(self.embedding.grad.dtype, copy=False)
        d_input = self.query_projection.backward(
            d_queries.reshape(g, k, self.model_width), cache.query_linear_cache)
        return d_input, d_logits
