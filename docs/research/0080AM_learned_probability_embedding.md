# 0080AM — Learned full-distribution probability PLAN embedding

## Research question

Can a small decoder learn conditional relationships between K=4 proposals using a
**separately trainable embedding of the full predicted vocabulary probability
vector**, rather than query-dependent access to every vocabulary item at every
decoder layer?

This is **not** latent-only and **not** top-N. We retain all 65,280 probabilities
in the one PLAN embedding calculation; the PLAN vectors are fixed once produced
for a block, but the DEC attention can query all four PLAN positions conditioned
on previously committed tokens. The compressed representation is **not
mathematically guaranteed to preserve all distributions** at `dim<768` (or in
an unconstrained learned table), so compare learning quality empirically.

## Forward

For proposal logits `L:[G,K,V]`, learn one shared table `E_prob:[V,d_prob]`
for all K horizons and a small adapter `A:[d_prob,d_model]`:

```
P = softmax_FP32(L)           # exactly full V; no top-k
Z = P @ E_prob               # [G,K,d_prob]
PLAN = plan_projection(Z @ A) + plan_positions
DEC  = token_projection([BOS,y1,...,y_(K-1)]) + dec_positions
X    = prefix_LM_3_layers([PLAN,DEC])
Delta = output_projection(correction(final_norm(X_DEC)))
L_final = L + Delta
```

The **objective has NOT changed**:

```
loss = CE(L_final, target, valid) + lambda_proposal * CE(L, target, valid)
```

Both CE terms are weighted by **unique valid supervised tokens**; training
retains the existing `--gradient-accumulation-weighting supervised_tokens`.
Manual backward accumulates `dL` from the additive logit prior, from the
learned probability expectation and from the proposal CE; gradients also reach
`E_prob`, `A`, the existing decoder and the original proposal head. The main
Transformer still receives one aggregated boundary gradient per microbatch.

## Versioning and initialization

- New `--block-proposal-soft-interface probability_embedding` and
  `--block-proposal-prob-embedding-dim <N>` (default 256).
- Independent trainable table `block_decoder.probability_embedding`, width N.
- Independent `block_decoder.probability_adapter.W` maps N -> d_model.
- The original v1/v2 parameters, initialization order, checkpoint metadata and
  full-vocab path are unchanged. The v3 checkpoint stores the dimension and
  embedding-sharing convention and rejects mismatched resumes.
- Fresh scratch training uses a **new v3 checkpoint directory**. V1-only
  decoder-inclusive weight migration is supported through the existing
  `--block-proposal-init-from-v1` option with **fresh optimizer state**.
- V2 full-soft optimizer-state resume into v3 is **intentionally refused**.
- The new table adds V*N parameters; at V=65,280 and N=256 that is
  16,711,680 parameters, ~31.9 MiB BF16 weights and ~63.8 MiB FP32 grads,
  excluding optimizer/offload state.

## Validation and limitations

- CPU finite differences on proposal logits, `E_prob`, adapter and weighted
  end-to-end objective including horizon proposal parameters.
- K=1/K=4 cached vs teacher-forced parity; no true future label influence.
- v3 optimizer step + checkpoint round trip; v1 and v2 metadata unchanged.
- CUDA BF16 gate exists but **not executed in this environment**.
- v3 `neutral`, `shuffled`, and `external` PLAN ablations are explicitly
  rejected until their distinct gradient paths are implemented; normal
  training and held-out evaluation are fully supported.
- Full 500M VRAM, speed and learning quality must be measured on RTX 5080.
- Per-block expectation is still a full softmax/GEMM over V. This removes the
  56-second query-dependent vocabulary-read path observed in the first-step
  profile, but that number is not a measured end-to-end speedup.

## Run

Use the unified launcher from the repository root and pass this experiment's
block-proposal options after `--`. Run `smoke`, inspect full-backward memory and
gradients, then use the same environment and `CHECKPOINT_DIR` with `continue`:

```bash
CHECKPOINT_DIR=./checkpoints/probability_embedding \
MODEL=deep-500m-memory-64k OBJECTIVE=block-proposal \
bash ./run_training.sh smoke -- \
  --block-proposal-soft-interface learned_probability_embedding
```
