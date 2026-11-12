# HIP local-attention backward: smaller live-state tiles

## Evidence and implementation

The prior 8K isolated backward test reported 214–215 dQ spills and 866 combined
dK/dV spills with 64x64 tiles. Reducing launch staging did not reduce spills;
eight warps reduced them but slowed execution. See `0107f_amd_nvidia_ffn_comparison.md`.

A 32x32 tile experiment using the **existing kernels** passed comparison with
the original 64x64 launch, then was selected for HIP in
`mini_llm/ops/triton_dual_resolution_local.py:local_attention_backward`.

- HIP dQ and dK/dV: 32x32, four warps, three launch stages.
- CUDA backward: unchanged 64x64 configuration.
- Forward and delta-kernel geometry: unchanged.
- No new atomics, gradient casts, tensor formats, or checkpoint changes.

The computational graph remains O = softmax(Q K^T * scale + mask) V.
Backward computes row delta, then dQ; combined dK/dV sums all enabled query
heads sharing each KV head. Smaller output tiles preserve one owner per output
row. Masks remain causal, document/EOS-bounded, and per-head-window-specific.
Changing tile boundaries can change floating-point reduction order; bitwise
cross-hardware continuation is not promised.

## Isolated candidate result (8K, one 4096-window head, fifteen 512-window heads)

| Kernel | Original 64 tile, ms | Candidate 32 tile, ms | Reported spills, 64 -> 32 |
|---|---:|---:|---:|
| Long-window dQ | 1.247 | 0.596 | 215 -> 0 |
| Short-window dQ | 2.597 | 1.326 | 214 -> 0 |
| Combined dK/dV | 17.410 | 7.897 | 866 -> 199 |

Candidate maximum absolute differences versus original: dQ 6.61e-8,
dK 0, dV 0; passed atol/rtol 1e-5. Register/resource figures are for these
specializations, not universal values. Stage 1 did not materially improve
32-tile timings, so launch staging stayed unchanged.

`benchmarks/benchmark_0107f_local_backward.py` now accepts `--tile 32|64`.
It computes the candidate grid from context length (not rounded reference grid)
to cover partial tiles correctly. It temporarily wraps launches, then restores
the module globals. Its reference is the installed production launch policy;
after this change that policy is 32 on HIP. It is not an independent oracle.

## Correctness validation

Both commands passed **8 tests**:

```sh
OPENBLAS_NUM_THREADS=1 sh scripts/run_r9700_preflight.sh -m pytest -q tests/test_triton_dual_resolution_local_0102b.py
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_triton_dual_resolution_local_0102b.py
```

Existing tests cover CuPy parity and integrated attention projections/backward.
Added independent NumPy FP64 graph/reverse-graph checks cover:

- Batch 2 / length 65 and batch 1 / length 641, including partial tiles.
- Packed-strided Q/K/V, BF16 inputs and FP32 gradients.
- Non-tile-aligned document boundaries.
- Mixed 512/4096 query-head windows.
- Both all-head and skip-head-0 operation.
- GQA reduction of four query heads into every KV head.

The test compares to FP64 with atol 0.003, rtol 0.06, allowing the established
BF16 probability/upstream quantization. It does not finite-difference the
quantized graph. The first run exposed an incorrect test assumption that
skipped output/dQ storage was zero; source inspection established this storage
is intentionally unwritten and owned/replaced by the sparse caller. One local
test correction restricts output/dQ comparisons to owned heads while still
checking all dK/dV, including exclusion of the skipped head. No kernel correction
was needed. Both exact test commands passed on the subsequent run.

Additional checks:

- Partial-context isolated benchmark at length 129, two repeats: passed,
  zero reported gradient differences for its 64-tile comparison.
- CPU `tests/test_dual_resolution_full_model_0102d.py tests/test_swiglu.py`:
  **11 passed**.
- `git --no-pager diff --check`: passed.

## Full Model B benchmark

Same fixed-8K, fully scored, batch-1 synthetic BF16 workload as the preceding
comparison: full sparse Model B, no block checkpointing, consumer-gpu runtime,
one warmup and three measured iterations. No checkpoint I/O or optimizer update.

| Measurement | R9700 after SwiGLU fix | R9700 + tile change | RTX 5080 prior reference |
|---|---:|---:|---:|
| Forward median, ms | 130.51 | 130.17 | 87.12 |
| Loss-forward median, ms | 7.19 | 7.40 | 7.22 |
| Backward median, ms | 770.05 | 598.04 | 220.81 |
| Tokens/s (median iteration total) | 9023.03 | **11136.70** | 25119.71 |
| Synthetic loss | 10.610338 | 10.610338 | 10.602576 |

End-to-end throughput improves approximately **23%** over the previous AMD
verified state, and 4.74x over the original 2347.90 tok/s baseline. The CUDA
reference was not rebenchmarked in this task; its launch geometry is unchanged
and its attention test file passed.

Separate intrusive operation profile across 20 layers:

- Local backward: **389.59 -> 210.84 ms** (~1.85x faster).
- All attention backward: **566.22 -> 385.20 ms**.
- Forward: effectively unchanged.

Profiler scopes are nested; do not add local backward to all-attention backward.
Device timing, random-weight initialization and synthetic supervision do not
constitute optimizer/training convergence or actual shard throughput validation.

Reports:

- `logs/r9700_local_backward_tile32.json`: original-policy candidate experiment.
- `logs/r9700_8k_attention_tile32.json`: unprofiled warmed full model.
- `logs/r9700_8k_attention_tile32_profile.json`: operation profile.
- `logs/r9700_local_backward_partial129.json`: partial-grid check.

Reproduce the warmed full-model run:

```sh
sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --architecture matrix-pool-sparse --context 8192 --model-context 65536 \
  --score-last-k 8192 --head-chunk-tokens 8192 --activation-checkpoint none \
  --runtime-profile consumer-gpu --query-chunk 1536 \
  --warmup-iterations 1 --repeat-iterations 3 \
  --output logs/r9700_8k_attention_tile32.json
```

## Remaining limit

This is a verified improvement, **not attention performance parity**. Combined
dK/dV still spills 199 registers in the measured specialization, and the full
attention path includes other projection/RoPE work outside the local kernel.
The next isolated experiment would compare a smaller-live-state dK/dV schedule
(e.g. separate accumulator passes) against this verified baseline, including
extra-load costs and independent gradient tests. Do not replace the verified
implementation merely on the basis of another speculative launch setting.
