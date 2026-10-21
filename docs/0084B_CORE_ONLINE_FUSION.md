# 0084B — Core online frozen-teacher probability fusion

## Scope

This adds a production-facing fusion objective to **`train_model.py`**; it is
implemented inside `mini_llm/collective`, **not** as an `examples/` script.
The existing autoregressive and block-proposal objectives are unchanged.

```text
EOS-bounded mixed-shard tokens (teacher validation files split by shard)
   -> same input windows -> frozen Deep forward -> full vocabulary logits
                         -> frozen Wide forward -> full vocabulary logits
   -> identity-aware weighted mixture + model-to-context cross-attention
   -> unrestricted residual probability combiner -> CE of the next token
   -> backward/AdamW ONLY on the combiner parameters
```

The teachers are kept resident but are run one after the other. Temporary
forward activations and prediction tiles are discarded after each bounded
batch; teacher gradients/backward graphs are never computed. This uses NO
persisted token-window files and NO cached teacher-logit files. Scores are
computed for the configured positions at full vocab (32,768 tokens), never
top-k-truncated. Accumulated combiner gradients use **exact row weighting**.

**Current deliberate limit:** teacher inference is *plain, local 512-token
context* (the checkpoint's external memory is disabled for this evaluation).
This does **not** train on the 61K-token historical-reader mode. The
contextual combiner attends over the preceding text tokens in each 512-token
window, not teacher hidden states. This is the intended initial 0082J
model-to-context design.

## Production launch (two teachers, from scratch)

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_ROPE_PREPARED_POSITIONS=1 \
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=256 \
MINI_LLM_FUSED_EXPERT_GEMM=0 \
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM=1 \
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT=1 \
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1 \
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1 \
python train_model.py --objective fusion \
  --member 0:./checkpoints/0080BX_deep512_fullmix_100k \
  --member 1:./checkpoints/0080BX_wide768_fullmix_100k_4400 \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --context-length 512 --score-last-k 512 --score-stride 1 \
  --batch-windows 1 --lm-head-rows 64 \
  --batch-rows 128 --grad-accum-steps 128 \
  --max-models 10 --min-members 2 \
  --context-heads 2 --context-head-dim 16 --context-hidden 32 \
  --steps 4000 --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --val-windows 16 --eval-interval 200 --save-interval 200 \
  --early-stop-patience 12 --log-interval 10 \
  --partition-seed 82 --seed 82 --runtime-profile gpu-fast \
  --output ./checkpoints/0084b_fusion_deepwide_4400_online.npz \
  --log-file ./logs/0084b_fusion_deepwide_4400_online.csv
```

**Two-update memory smoke:** use exactly this command with `--steps 2`,
`--batch-rows 8 --grad-accum-steps 2 --val-windows 2`, and a DIFFERENT
`--output`/`--log-file` path. Smoke with `MINI_LLM_BACKEND=cupy` on the RTX
5080; CUDA success is not established by CPU validation.

**Resume (including AdamW moments):** repeat the same command with `--resume`
and increase `--steps`. The sibling `...online.last.npz` saves optimizer
moments, step index, sampler RNG state, teacher fingerprints and split settings.
The primary `--output` is the best portable combiner checkpoint in the same
`ProbabilityResidualCombiner` format consumed by existing inference tools.

## Interpret training metrics

- `fusion_ce`: learned residual combiner validation CE.
- `arithmetic_ce`: uniform teacher probability average CE.
- `weighted_ce`: context-aware identity-weighted probability mixture CE,
  **without** the unrestricted residual.
- `member_ce`: each frozen teacher CE on the *same* exact targets and contexts.
- `row/s`: scored supervised token rows per second (not input-source tokens/s).

The split is deterministic by whole teacher-validation-shard files, with
separate fusion train, validation, and test partitions. It prevents overlapping
fusion train/val files and avoids training the combiner directly on the
teachers' pretraining train shards. **It is not guaranteed to be independent of
previous teacher hyperparameter selection**, which may have examined the
parent validation data. For a final reported benchmark, use a truly untouched
held-out corpus (e.g. WikiText-2) and ensure the evaluator handles context
attention.

## Limitations and next steps

1. The current WikiText-2 and fused generation evaluation scripts may reject
   `context_heads=2`; those inference integrations remain a separate patch.
2. Frozen teachers keep weights and short-context inference state on GPU; they
   have no backward tape. Small combiner parameters do **not** imply zero
   inference memory for teacher outputs or temporary activations.
3. Contextual full-vocabulary combiner training is repeated online, so it can
   be slower than reusing cached teacher logits over multiple optimizer steps.
4. If >2 teachers are supplied, `--min-members` controls random subset size
   each update; validation always uses all configured teachers.
5. Fine-grained CPU tests, end-to-end real tiny checkpoint tests and historical
   0082/0083 CPU regressions passed. The exact large RTX 5080 GPU training
   run has NOT yet been performed in the assistant environment.
