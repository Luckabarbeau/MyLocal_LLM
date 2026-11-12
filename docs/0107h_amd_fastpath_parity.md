# R9700 Model B: executed fast-path coverage and optimized backward

## Final measured state

Full 348,261,376-parameter sparse Model B, BF16, batch 1, fixed 8,192-token
synthetic document, all 8,192 targets scored, model capacity 65,536,
consumer-gpu runtime, no block activation checkpointing, query chunk 1,536.
Fresh processes, **two warmups and five measured iterations**, no optimizer
update/data sampling/checkpoint I/O in throughput timings.

| Stage | R9700 | RTX 5080 |
|---|---:|---:|
| Forward median | 108.208 ms | 87.059 ms |
| Loss forward median | 7.321 ms | 7.288 ms |
| Backward median | 317.415 ms | 226.710 ms |
| Throughput, median iteration total | **18,918.55 tok/s** | **25,181.07 tok/s** |

AMD iteration totals were 432.838–433.218 ms; NVIDIA totals 313.457–328.037 ms.
Throughput uses median iteration total, not the sum of stage medians.
AMD is **75.13% of NVIDIA throughput**, approximately **24.87% lower**. This is
not performance parity. It is about **70% faster** than the 11,136.70 tok/s
verified state at the beginning of this task, and **8.06x** the original
2,347.90 tok/s warmed baseline. It is within 5.41% of 20,000 tok/s.

These are random-weight compute benchmarks, not actual resumed-training rates.
Variable document lengths, supervision masks, optimizer offload, accumulation,
monitoring, validation and checkpoint saves affect real training throughput.
Cross-vendor RNG/rounding differences mean synthetic losses are not identical.
Do not infer multi-step convergence or bitwise continuation from these runs.

## Restored CUDA-family optimizations on HIP

Strict compilation established missing `<cuda_bf16.h>` failures in training
RMSNorm, standalone inverse-RoPE backward, and packed QKV/RoPE forward.
`mini_llm/ops/hip_bf16_source.py` adapts the header, BF16 types and conversion
spellings for HIP; CUDA source is returned unchanged. The existing equations,
residual precision, reduction precision and explicit BF16 rounding points remain.

HIP packed QKV/RoPE additionally requires `-ffp-contract=off`: a bitwise reference
test demonstrated that contraction otherwise bypassed an intended BF16 rounding
boundary. One local compiler-option correction restored parity; CUDA options
are unchanged. The unused packed inverse-RoPE helper was not rewritten.

Together with the earlier SwiGLU repair, these changes activate the relevant
existing fused model kernels rather than silently executing vectorized fallbacks.

## HIP-specific backward schedules

### Local attention

- Keep dQ32 geometry but use one **inner-loop** stage on HIP: 143 registers
  instead of 233; long/short isolated dQ ~0.311/0.586 ms instead of ~0.596/1.326.
  This is distinct from launch staging, which earlier experiments had changed
  without affecting the explicit inner pipeline.
- `triton_hip_local_kv.py` uses separate compile-time dK/dV passes, 32x32 tiles,
  one query-head owner per key tile, private FP32 partials and deterministic
  reduce-four. Both dot passes report zero spills. It reuses delta from dQ.
- Matched isolated delta/KV/reduction: 7.715 -> 2.890 ms, ~2.67x faster.
- GQA, per-head windows, document masks and skipped-head semantics are retained.
  Temporary dK/dV partials consume 8 KiB per batch-token (64 MiB at batch-1 8K).

### Global and sparse attention

- HIP BF16 global/sparse dQ uses 32x32 backward geometry. Forward and CUDA
  geometry are unchanged.
- `triton_hip_dense_kv.py` separates dK/dV and partitions the query reduction
  into at most 16 aligned segments. Each key tile/query head/segment has one
  FP32 accumulator. Fixed-order segment/GQA reductions avoid atomics.
- Reuses the original `_visible` predicate, full global metadata and sparse
  stride/phase semantics; preserves strided destinations and `add_dkdv`.
- Isolated global KV: 2.216 -> 0.331 ms; sparse KV: 1.810 -> 0.352 ms. Candidate
  totals include delta, both passes and reduction; allocations/host overhead
  are excluded. Representative scratch: 8 MiB global / 4 MiB sparse.

Comparisons against independently executed original launches passed atol/rtol
1e-5, including partial tiles, packed/EOS batches, GQA, sparse additive buffers,
canaries and deterministic repeats. For the sparse8K case, max absolute dV
error was 1.45e-5; all values satisfied the combined absolute/relative tolerance.
These scheduling tests complement the independent FP64 reference tests; they
are not themselves mathematical oracles.

Rejected experiments remain nondefaults: eight warps, smaller 16-key tiles for
separate local passes, combined per-query-head dK/dV, and oversized asymmetric
query tiles were slower or gave marginal benefit. No throughput claim uses them.

### Generic FP32 attention portability

The existing FP32 compressor branch hardcoded `tf32x3`, rejected by HIP.
It now uses IEEE FP32 on HIP while preserving CUDA's `tf32x3`. This branch is
not the BF16 Model B path. A Triton global must be instantiated with
`tl.constexpr(value)`, not merely annotated; one local declaration correction
was needed before the final GPU test runs passed.

## FP32-output projection weight gradients

The old attention output-wgrad code promotes BF16 activations and gradients
to FP32. Matched isolated wgrad timings were 6–7x slower on AMD than NVIDIA,
while BF16 timings were close. BF16 input gradients were already selected on
both GPUs; they were not the missing branch.

`hip_projection_wgrad.py` uses HIP GEMMEx with **FP32 output and accumulation**:

- BF16 activation + BF16 gradient: one BF16-input GEMM, without rounding its
  result to BF16.
- BF16 activation + FP32 gradient: high/middle/low BF16 residual decomposition
  of the gradient, then three FP32-output GEMMs. A fused one-pass decomposition
  avoids repeated full-array casts/subtractions. It does not simply discard
  the FP32 gradient's low bits.

The borrowed handle is bound to the current stream and its pointer mode is
restored. No device-value host reads or synchronization are introduced.

Validation: **38 helper tests**, NumPy FP64/CuPy FP32 oracles, odd/strided/reversed
inputs, residual-sensitive cases, pointer modes/errors, current/nondefault
streams. Random and multiscale reconstruction was exact; full benchmark
reconstruction had zero mismatches over 16,777,216 tested FP32 values. Components
matched the original cast/subtract sequence bitwise over 50,010 strided bit
patterns after a local NaN-payload correction. Full-size output comparisons
passed atol/rtol 1e-5, max absolute difference vs CuPy FP32 2.50e-5, relative L2
at most 2.09e-6. FP32-gradient square helper: 3.260 -> 0.664 ms including casts
and GEMMs; BF16-gradient global helper: 0.953 -> 0.062 ms.

**Precision boundary:** this is FP32-precision residual decomposition in the
validated ordinary training range, not bitwise FP32 SGEMM. Extreme FP32 values
can overflow BF16 parts or underflow residuals. Disable this policy for literal
SGEMM or unsupported dynamic-range requirements. No cross-vendor exact training
trajectory is claimed.

## Default policies and checkpoint contracts

The successful HIP policies are defaults, with explicit overrides retained:

| Variable | Default | Reference/disable option |
|---|---:|---|
| `MINI_LLM_HIP_SPLIT_KV_BACKWARD` | `2` | `0`: original combined; `1`: diagnostic combined per-head |
| `MINI_LLM_HIP_DQ_LOOP_STAGES` | `1` | `2`: original two-stage inner loop |
| `MINI_LLM_HIP_DENSE_BACKWARD_TILES` | `1` | `0`: original dense/sparse tile policy |
| `MINI_LLM_HIP_DENSE_KV_BACKWARD` | `1` | `0`: original combined dense/sparse KV kernel |
| `MINI_LLM_HIP_FP32_PROJECTION_WGRAD` | `1` | `0`: original promoted FP32 matmul |

All dispatch guards are HIP-only, with dtype/layout contracts retained.
`runtime_defaults.py` materializes these choices for consumer-gpu Model B on
HIP so saved runtime environments record them. New policy keys can require
`--allow-nonexact-resume` when migrating an older checkpoint; that is intentional
and does not mean resetting optimizer/sampler state. Preserve original checkpoints
and direct experiments to distinct output locations. No checkpoints were used
or modified in this task.

## Actual full-model activation audit

```sh
sh scripts/run_r9700_preflight.sh -u scripts/probe_0107j_amd_fastpaths.py
```

This audit fails on fallback and checks actual executions for one full-8K pass:

| Path | Verified executions |
|---|---:|
| Packed QKV/RoPE forward | 20 |
| Fused RMSNorm backward, cache and eligibility | 41 |
| Inverse RoPE backward | 80 |
| Fused SwiGLU forward/recompute and backward | 40 / 20 |
| BF16 global / sparse forward | 20 / 16 |
| Fused matrix-pool backward | 20 |
| FP32-output HIP projection wgrad | 40 |
| Separate local KV backward | 20 |
| Segmented global/sparse KV backward | 36 |
| Actual fused CE loss / gradient kernels | 1 / 1 |

This covers the audited Model B training path, **not every CUDA optimization
in the repository**. Inference-only kernels and grouped-cuBLAS paths not used
by Model B are outside this claim. Compilation/flags alone are not activation
proof. Report: `logs/r9700_fastpaths_final.json`.

## Final validation and benchmark reproduction

Final GPU regression selection:

- AMD: **77 passed, 1 skipped, 6 deselected**.
- CUDA: **33 passed, 1 skipped, 6 deselected**.
- CPU: **49 passed, 4 skipped** including runtime defaults and matrix-pool
  mathematical/reference tests.
- `git --no-pager diff --check`: passed.

GPU selection included projection helper (AMD only), local attention/FP64
oracle, split-head comparisons, remaining Triton attention, sparse history,
QKV/RoPE, residual RMSNorm, prepared RoPE and SwiGLU. Four deep-memory/checkpoint
cases were not run. Two pre-existing NumPy-input-only sparse-reference tests
were run under NumPy, not CuPy: their GPU runs initially fail on unsupported
NumPy/CuPy mixed operands. Their production paths were not modified to hide
that test-selection issue. The generic HIP FP32 compiler failure was fixed;
most recent backend-appropriate selections all pass.

Bounded optimizer check with consumer-gpu defaults applied, full348M model,
256 synthetic tokens/eight scored rows: FP32 gradients, full offload, 220 Muon
and 82 AdamW tensors, clipping1.0, five BF16 Newton–Schulz steps. Finite state,
master/model updates in both groups, counters at1; loss **10.313887 -> 5.767154**.
Update1.53 s, total22.86 s. This is not an 8K optimizer-throughput or stability run.

Final warmed benchmark (AMD):

```sh
sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --architecture matrix-pool-sparse --context 8192 --model-context 65536 \
  --score-last-k 8192 --head-chunk-tokens 8192 --activation-checkpoint none \
  --runtime-profile consumer-gpu --query-chunk 1536 \
  --warmup-iterations 2 --repeat-iterations 5 \
  --output logs/r9700_8k_optimized_final.json
```

NVIDIA: replace the launcher with `MINI_LLM_BACKEND=cupy .venv/bin/python -u`
and change the output filename. Reference:
`logs/rtx5080_8k_optimized_reference_final.json`.

Other artifact families: `r9700_8k_fusion_portability`, `r9700_8k_separate_kv_*`,
`r9700_8k_all_hip_candidates`, `r9700_hip_projection_{candidate,fused}`,
`r9700_dense_tile32*`, `r9700_projection_gemm`, `rtx5080_projection_gemm` under
`logs/`. Component benchmark scripts retain explicit original-reference policy
where production defaults have changed.

## Remaining gap

The measured residual is about **25% in throughput**, not equality. This task
established missing fast paths, FP32 GEMM costs, spilling, and inadequate
parallelism; it does not establish that the residual is entirely hardware-limited
or that another unmeasured schedule can close it. Preserve this verified state.
The next useful experiment is a fresh post-optimization device kernel trace to
attribute the remaining ~108 ms median iteration-time difference, rather than
another ungrounded sequence of tile/warp choices.
