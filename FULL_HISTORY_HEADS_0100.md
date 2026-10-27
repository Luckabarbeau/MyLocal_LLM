# Independent full-history attention upgrade (0100B)

## Architecture contract

The feature is opt-in through zero-based `full_history_layers`. The production
layout is `[0, 1, 2, 3, 5, 7, 9, 11]`, corresponding to one-based layers
**1, 2, 3, 4, 6, 8, 10, 12**. Every selected layer owns independent `Wq`,
`Wk`, `Wv`, and `Wo`; no trainable matrix is shared. For width 1024 and head
width 64, each layer adds 262,144 parameters and all eight add 2,097,152.

The branch performs one causal softmax over compacted valid raw-token history
and valid current-layer states. Historical K/V projection is chunked and packs
that layer's independent K/V weights into one GEMM. Empty history skips history
projection while retaining ordinary full-causal attention over the current
window. `Wo` is initialized to exact zero, so the branch is an exact additive
identity before the first update.

## 0100A audit and 0100B corrections

The 0100A prototype correctly provided independent projections, zero-output
initialization, per-row compaction of valid historical tokens, exact absolute
RoPE positions, chunked historical projection/recomputation, parameter
enumeration, activation-checkpoint replay, and prefill/decode caches. Its
PyTorch SDPA/DLPack execution boundary was a prototype limitation and has been
replaced by native Triton over CuPy allocations.

0100B corrects the following production blockers:

- The current-state gradient returned by `FullHistoryHead.backward()` was
  discarded by the Transformer block. It is now accumulated before RMSNorm
  backward, restoring gradients to norm1, the residual input, prior layers, and
  current embeddings.
- Progressive residual gates now scale full-history gradients and receive the
  full-history contribution to `alpha_attn.grad`.
- Current validity is derived from historical-read `target_valid_lengths` and
  compacted before Q/K/V attention. Invalid keys are not projected into the
  effective attention problem, inactive queries return exact zero, and fully
  masked rows cannot produce NaNs.
- Dense current allow masks and optional history-coordinate allow masks are
  combined with causal validity in the CPU oracle. The CUDA production path
  accepts compact causal validity and explicitly rejects arbitrary dense masks
  until document boundaries have a structured in-kernel representation.
- NumPy uses an explicit mathematical reference implementation and backward.
  CUDA uses `triton_full_history_attention.py` directly on CuPy allocations;
  PyTorch and DLPack are no longer used by this branch.
- The Triton kernel implements rectangular lower-right causality for
  `Q<=4096`, `K<=65536`, streams softmax without score/probability matrices,
  caches only FP32 output/LSE, and recomputes probabilities for FP32 gradients.
- Full-history checkpoints now use their dedicated historical/current state for
  short-horizon inference too; the branch is no longer omitted or allowed to
  fail after ordinary cache mutation.
- Unsupported provisional/shared-context routes reject the architecture before
  mutating caches.
- Migration now uses the authoritative saved optimizer kind, derives each new
  group with the real `muon_eligible()` function, updates saved Muon group names,
  initializes AdamW birth steps at the preserved global step, checks exact
  optimizer tensor sets, hashes the source before/after, stages in a temporary
  sibling, and atomically publishes only after audit success.

## Mask contract

`FullHistoryHead.forward()` accepts:

- `current_valid_mask[B,T]` for current query/key validity;
- `attention_allow_mask[T,T]` or `[B,T,T]` in current coordinates;
- optional `historical_allow_mask[B,T,H_capacity]` in the original history-bank
  coordinates.

The head first filters `historical_state.valid`, then compacts valid current
rows. Thus a row with 512 valid history tokens and 1024 current tokens computes
`1024 × 1536`, not `1024 × 65536`. Batch rows may have different effective
history lengths. Masked historical IDs receive no embedding gradient through
this branch.

The production document-aware historical sampler supplies same-document history
through `historical_state.valid` and prefix current validity through
`target_valid_lengths`. Explicit cross-document tests use both current and
historical allow masks. The broader model still rejects combinations that its
original 16-head sparse/routed attention does not support; 0100B does not relax
those original-attention safety checks.

## Safe checkpoint conversion

From the repository root:

```bash
.venv/bin/python -m examples.migrate_full_history_heads \
  --source ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --destination ./checkpoints/0083_intermediate_20L_1024_full_history_8
```

The source and destination must be distinct, nonnested paths, and the destination
must not exist. Conversion requires complete model, FP32 master-weight, optimizer,
configuration, and training state. It never hard-links arrays. A failure removes
the temporary staging directory and leaves no destination.

Audit an existing conversion without modifying it:

```bash
.venv/bin/python -m examples.migrate_full_history_heads \
  --source ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --destination ./checkpoints/0083_intermediate_20L_1024_full_history_8 \
  --audit-only
```

The audit checks source preservation, original tensor equality, new shapes,
zero output projections, exact optimizer tensor sets, Muon groups, Adam birth
steps, optimizer kind, and preserved training metadata. A real production
resume/save/reload rehearsal remains required on the RTX 5080.

## CPU regression

```bash
MINI_LLM_BACKEND=numpy \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
.venv/bin/python -m unittest \
  tests.test_full_history_upgrade_0100 -v
```

The suite covers exact initialization parity; empty, one-token, partial, mixed,
and 61,440-token history; current padding; inactive queries; document masks;
future/masked-token perturbations; masked-history gradients; finite-difference
current-input gradients; activation recomputation; short and long cached
inference; Hybrid-Muon restoration/update; AdamW migration age; and
non-destructive atomic migration.

## RTX 5080 GPU gates

The isolated empty-history gate must be run first:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python \
  -m examples.benchmark_full_history_head_0100 \
  --queries 256 --history 0 --batch-size 1 --repeats 5
```

Then run the requested production geometry and history sweep. The integrated
matched full-model benchmark command, using the repository's established probe,
is:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_TRAIN_ATTN_STRICT=1 \
MINI_LLM_HISTORY_RECOMPUTE=1 \
.venv/bin/python benchmarks/profile_0087a_training.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_full_history_8 \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --batch-size 3 --microbatches 1 --working-length 4096 \
  --seed 42 --warmup 2 --repeats 20 --attention-backend triton \
  --no-optimizer \
  --out benchmarks/results/0100b_full_history_b3_t4096.json
```

Run the same command against the original checkpoint for the matched baseline.
Record process/device and CuPy-pool memory. Verify BF16 forward/backward, finite
outputs/gradients, and absence of dense score, probability, or boolean
`4096 × 65536` allocations.

Native kernel regression on a CUDA development GPU:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest \
  tests/test_full_history_triton_0100.py -q
```

On the available 4 GiB Quadro P1000, all eight FP32/BF16 native GPU oracle,
ragged-mask, and end-to-end tests passed. Isolated measurements were:

| Q | History | Forward | Backward |
|---:|---:|---:|---:|
| 64 | 0 | 2.94 ms | 2.63 ms |
| 4,096 | 0 | 87.57 ms | 214.84 ms |
| 64 | 512 | 14.27 ms | 19.31 ms |
| 256 | 8,192 | 38.84 ms | 43.01 ms |
| 1 | 61,440 | 197.82 ms | 213.77 ms |
| 4,096 | 61,440 | 3,088.88 ms | 3,562.41 ms |

The full `4096 × 65536` attention geometry completed with an approximately
172 MiB run-device delta, establishing execution without a quadratic score,
probability, or mask allocation. The first full-geometry result includes
shape-specific JIT/allocator effects and all P1000 timings are functional
measurements, not RTX 5080 performance estimates.

### Matched 4K square baseline

The rectangular traversal was also measured at `Q=K=4096`, B1, D64 against
both the public 0088A mixed-attention wrapper and a direct launch of its private
full-head kernels. The direct path uses one-head output, LSE, delta, and gradient
storage, eliminating the wrapper's 16Q/4KV storage and inactive-head backward
grid overhead:

```bash
MINI_LLM_BACKEND=cupy \
.venv/bin/python benchmarks/benchmark_0100b_rectangular_vs_square.py \
  --sequence-length 4096 --warmup 2 --repeats 5
```

| Phase | 0100B rectangular | 0088A direct one-head | Ratio |
|---|---:|---:|---:|
| Forward | 20.19 ms | 306.38 ms | 0.066 |
| Backward | 216.79 ms | 340.51 ms | 0.637 |
| Combined | 295.96 ms | 1,392.15 ms | 0.213 |

Relative L2 differences against the direct square baseline were `0.00108` for
output, `0.00252` for dQ, `0.00247` for dK, and `0.00219` for dV. Thus no
rectangular-kernel slowdown is visible at square scale on the P1000. These
numbers should not be interpreted as an intrinsic 4.7–15× algorithmic speedup:
the kernels differ in online versus two-pass softmax, accumulator conversion,
and backward tiling, and combined timings on this small legacy GPU showed
substantial allocator/thermal variability. The same direct comparison must be
rerun on the RTX 5080 before selecting production tile sizes.

## Resume rehearsal

This command repeats the documented production optimizer, precision, schedule,
batch, accumulation, validation, and numerical-path settings. It intentionally
writes resumed checkpoints to a new directory rather than the converted input:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=auto \
MINI_LLM_HISTORY_QUERY_CHUNK=1024 \
MINI_LLM_HISTORY_RECOMPUTE=1 \
MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1 \
MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1 \
MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1 \
MINI_LLM_ROPE_PREPARED_POSITIONS=1 \
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1 \
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1 \
MINI_LLM_AR_ACCUM_LENGTH_BUCKETS=1 \
.venv/bin/python -u train_model.py \
  --resume-from ./checkpoints/0083_intermediate_20L_1024_full_history_8 \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --precision bf16-mixed --runtime-profile consumer-gpu \
  --batch-size 3 --grad-accum-steps 128 \
  --gradient-accumulation-weighting supervised_tokens \
  --optimizer hybrid-muon --muon-momentum 0.95 \
  --muon-ns-steps 5 --muon-matmul-precision bf16 \
  --total-steps 100000 --warmup-steps 750 --peak-lr 2.5e-4 \
  --grad-clip 1.0 --weight-decay 0.1 \
  --val-interval 500 --val-steps 10 --val-per-source-steps 2 \
  --save-interval 1 --log-interval 1 --max-run-steps 2 \
  --checkpoint-dir ./checkpoints/0083_intermediate_20L_1024_full_history_8_rehearsal \
  --log-file ./logs/0083_intermediate_20L_1024_full_history_8_rehearsal.csv
```

The shard root and real source checkpoint are not present in every checkout;
verify those documented paths locally before running. After two updates, confirm
finite loss/gradients, changed `Wo`, nonzero Q/K/V gradients after `Wo` moves,
then save, reload, and compare predictions.

## Inference and remaining limitations

Standard full forward, history-aware prefill, short-horizon prefill, and
single-token cached decode are integrated and covered on CPU with nonzero output
projections. Independent current K/V caches are per selected layer.

The existing amortized historical refresh policy remains approximate between
refreshes: an evicted working token becomes visible to the independent history
bank at the next refresh. Set `MINI_LLM_HISTORICAL_REFRESH_TOKENS=1` for strict
parity. Provisional/shared-context refinement remains explicitly unsupported.
The full-history branch routes around the native short-state CUDA-graph path;
CUDA-graph capture for the Triton training/prefill execution is not validated.
Cross-layer grouped historical projection is not implemented because retaining
eight full K/V banks can increase VRAM; this optimization requires measurement
on the target GPU.

This environment has a Quadro P1000 with about 1.1 GiB free. Triton 3.8.0 was
installed and native FP32/BF16 rectangular forward/backward tests passed within
that limit. The P1000 is not representative of the RTX 5080 target, so 4K-by-65K
throughput, 16 GiB peak memory, microbatch 3, real-checkpoint migration, and the
two-step production resume were not executed here.

## Rollback

The original checkpoint is unchanged. Stop the rehearsal, point inference or
training back to:

```text
./checkpoints/0083_intermediate_20L_1024_fullmix_100k
```

Remove only the independently created conversion/rehearsal directories after
confirming their paths:

```bash
rm -rf ./checkpoints/0083_intermediate_20L_1024_full_history_8
rm -rf ./checkpoints/0083_intermediate_20L_1024_full_history_8_rehearsal
```

To remove code changes applied by the delivery patch:

```bash
git apply -R PATCH_0100B_FULL_HISTORY_PRODUCTION_READY.patch
```
