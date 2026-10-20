# 0080AF — Tiny-vocabulary CPU full-soft read and manual VJP

**Parent:** 0080AE (configuration only). This patch adds a **new isolated
reference module**, `mini_llm/block_proposal/soft_vocab_read.py`; it does
not modify the original `BlockPrefixDecoder`, `train_model.py`, the
`SoftInterfaceConfig.require_implemented` gate, optimizer, generation,
checkpoint metadata or production kernels. It must be applied **after 0080AE**.

## Objective and concrete API

```
read = SoftVocabularyRead(shared_embedding_parameter, decoder_width,
                          n_heads=4, temperature=1., read_scale=1.,
                          rng_seed=8091)
residual, tape = read.forward(decoder_states, proposal_logits)
#   decoder_states [G,K,width]             (causal DEC states supplied by caller)
#   proposal_logits [G,K,V]                (all V alternatives, own source)
#   residual [G,K,width], tape.read [G,K,d_model]
d_dec_states, d_proposal_logits = read.backward(d_residual, tape)
# Parameter gradients are accumulated in module projections AND shared E.
```

- Two module-owned Parameters: `query.W [decoder_width,d_model]` initialized
  using an **independent RNG seed**, and `output.W [d_model,decoder_width]`
  initialized **exactly zero**. The caller-owned shared E is **not** returned
  in `read.parameters()`, preventing duplicate optimizer registration; its
  `.grad` receives the key **and** value contributions.
- One read for each DEC row, from its **own** proposal row. Independent
  per-head E slices have shape `[V,d_h]`, where `d_h=d_model/H`. Scores are
  `L_j(v)/T + beta*dot(Q_jm,E_vm)/sqrt(d_h)`, normalized over **all V**.
  Output is the concatenation of all head reads times the output projection.
  The normalizations presently are `1/sqrt(d_h)` and stable rowwise softmax;
  there is **no undocumented query/key RMS normalization**, extra temperature,
  dropout, top-k restriction, or learned bias in this mathematical reference.
- When `beta=0`, `tape.read == softmax(L/T)@E` (headwise-concatenated).
  Different causal DEC queries can retrieve different modes from the same
  multimodal proposal distribution even when the mean proposal embedding
  agrees. This is the essential 0080AF expressivity test.
- The output residual is initially zero and its output projection receives
  gradients immediately. The query projection / logits / E pathways receive
  gradients after the output projection first becomes nonzero. **Do not**
  interpret an all-zero initial upstream VJP as detached gradients.

## Manual backward, exact and additive

For an individual head with read gradient `g` and read `R`:

```
dS_v = A_v * dot(g, E_v-R)
dL_v = dS_v / T
dQ = beta/sqrt(d_h) * sum_v dS_v E_v
dE_v = A_v*g + beta/sqrt(d_h) * dS_v*Q
```

The query/output projections use the repository's manual `Linear.backward`.
`backward()` returns `(d_dec_states, d_proposal_logits)`, **not including**
the identity gradient of the future residual connection, the existing PLAN
expectation VJP, or the proposal loss gradient. Those sums belong to 0080AG.
Each `forward()` has a separate `OwnVocabReadCache`; backward recomputes
one-head weights and does not use module-global mutable tapes. The reference
stores a small copy of proposal logits to keep the tape safe from caller
mutation. The canonical embedding and projection weights must be unchanged
between forward and backward (normal per-step training semantics).

## Guardrails / limitations

- **CPU NumPy only**, 32/64-bit float; CuPy is explicitly rejected.
- Tiny vocabulary only (`V<=512` by default, plus score-count cap).
- Intermediate attention is `[G,K,V]` **per head**. No
  `[G,K,H,V,d_h]` materialization; no persistent probability tensor in tape.
  This is not a production memory/performance claim: **RTX 5080 16 GiB**
  full-V operation needs new bounded kernels, scheduled for 0080AH.
- `source_scope='own'` only. The crucial `all` mode's log-softmax-normalized
  cross-horizon reads, source bias and cross-horizon gradients are deliberately
  **not** implemented until 0080AI. Unsupported `all` fails explicitly.
- Query states must already be block-causal, constructed by a future decoder
  integration from preceding committed tokens; this reference has no access
  to labels. No teacher-forced/cached parity or v2 checkpoint claim is made.
- Baseline `SoftInterfaceConfig(mode='full_vocab')` remains gated: this patch
  cannot accidentally activate a half-integrated production mode.

## Tests / numerical evidence

```
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_block_proposal_0080af_soft_vocab_read.py
```

The test suite independently finite-differences the query projection, output
projection, DEC states, proposal logits and E; tests `beta=0` identity, per-
source logit-shift invariance, multimodal query response, horizon isolation,
zero-output initialization, separate forward tapes, numerical extremes,
duplicate embeddings and forbidden production-scale input. New tests are
isolated from and do not replace prior regression tests. With the five
previously identified missing diagnostic-file test modules excluded, the
available CPU block-proposal suite passes **219 passed, 12 skipped**.

**Next:** 0080AG must insert one read after decoder layer 0, add its residual
and both gradient branches, preserve the v1 no-feature path, then implement
the same operation in cached generation before attempting any production
training experiment. A safe complete v2 checkpoint is required before long
runs. The next CuPy patch must not reuse this NumPy implementation for V=65K.
