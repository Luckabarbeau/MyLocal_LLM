# Probability fusion GPU optimization report

Date: 2026-10-25  
Baseline revision: `96ed2c39f18334ccef4f24457bbd0e2077ea52c5`

## Scope

The optimized path preserves the full-vocabulary fusion objective, parameter shapes,
checkpoint format, FP32 parameter-gradient accumulation, decoder depth, member width,
contextual attention, and online gradient-accumulation semantics.

The frozen teacher implementation and Transformer architecture were not modified.

## Accepted changes

1. **Generalized fused multilayer decoder backward**
   - Reuses the fused output-head/tanh derivative at the terminal layer for every
     decoder depth.
   - Replaces each separate square/subtract/multiply activation derivative with one
     CUDA pass.

2. **Fused decoder bias and activation**
   - Keeps decoder GEMMs on cuBLAS.
   - Applies bias and `tanh` in-place in one CUDA pass.

3. **Stable log-probability reuse**
   - Caches the already-computed forward `log_pc` for the training loss.
   - Removes a duplicate output projection and full-vocabulary log-softmax.

4. **Wide-feature contraction without full feature concatenation**
   - For `member_width >= 64`, computes the first decoder layer from member mean,
     member variance, and four scalar features without materializing `[R,V,2H+4]`.
   - Uses two direct FP32 cuBLAS SGEMMs with `beta=1` accumulation.
   - Fuses the four scalar projections, decoder bias, and `tanh`.
   - Computes matching fusion-weight gradients by blocks and propagates only the
     first `2H` feature gradients consumed by the member encoder.
   - Narrow widths retain the faster original single-GEMM implementation.

5. **Coalesced member-parameter reduction**
   - Changes the reduction tile so each warp spans 32 hidden channels at one
     vocabulary position.
   - Makes `dpre` loads contiguous and broadcasts scalar features within a warp.
   - Reduces the width-64 partial-reduction grid from 262,144 blocks to 8,192 blocks.
   - Retains deterministic FP32 partial buffers and the existing final reduction.

## Matched standalone benchmark

RTX 5080, CuPy 14.2.0, CUDA runtime 13.2, two members, vocabulary 32,768,
64 supervised rows, embedding width 16, three warmups, seven measured updates.
All variants use the same seeded fixture.

| Configuration | Parameters | Original ms | Optimized ms | Latency change | Original tok/s | Optimized tok/s | Original MiB | Optimized MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small: H8/K24/L1/C0 | 557,794 | 13.80 | 11.52 | -16.5% | 4,639 | 5,556 | 1,221 | 1,213 |
| Context: H8/K24/L1/C2 | 560,438 | 15.67 | 12.74 | -18.7% | 4,084 | 5,022 | 1,232 | 1,224 |
| Wide: H64/K24/L1/C0 | 560,762 | 53.16 | 24.54 | -53.8% | 1,204 | 2,609 | 5,253 | 4,677 |
| Deep: H8/K24/L4/C0 | 559,594 | 32.12 | 21.82 | -32.1% | 1,993 | 2,934 | 2,181 | 1,981 |
| Target: H64/K24/L4/C2 | 565,206 | 73.14 | 36.38 | **-50.3%** | 875 | **1,759** | 6,224 | **5,456** |
| Larger decoder: H64/K64/L4/C2 | 581,246 | not measured | 60.37 | n/a | n/a | 1,060 | n/a | 7,377 |

`MiB` is peak sampled live CuPy allocation from the benchmark profiler. Pool
reservation was not instrumented by the existing benchmark and is not inferred from
live allocation.

## Target stage comparison

| Stage | Original ms | Optimized ms | Change |
|---|---:|---:|---:|
| `backward.member_parameter_grads` | 21.825 | 3.569 | -83.6% |
| `backward.output_head.propagate_and_decoder_activation` | 13.113 | 7.679 | -41.4% |
| `forward.assemble_features` | 9.338 | 2.204 | -76.4% |
| `forward.residual_decoder` | 8.902 | 4.081 | -54.2% |
| `backward.member_encoder` | 7.088 | 6.539 | -7.7% |

The remaining largest isolated stages are decoder backward (7.68 ms), member
encoder backward (6.54 ms), decoder forward (4.08 ms), and member parameter
gradients (3.57 ms). The original 21.8 ms member-gradient stage is no longer dominant.

## Production-equivalent benchmark

Actual frozen teachers:

- `0083_intermediate_20L_1024_fullmix_100k_14k`
- `0083_intermediate_20L_1024_fullmix_100k`

Training geometry retained `context_length=512`, `score_last_k=511`,
`batch_windows=4`, `batch_rows=64`, and `grad_accum_steps=512`.
The run used separate files `checkpoints/fusion_opt_benchmark_20261025*.npz`.

| Metric | Result |
|---|---:|
| Supervised rows | 32,768 |
| Combiner microbatches | 513 |
| Reported fusion throughput | **1,785 tok/s** |
| Reported end-to-end throughput | 1,668 tok/s |
| Training interval | 18.358 s |
| Frozen-teacher GPU time | 1.151 s |
| Previous production report | ~937 tok/s |
| Production throughput improvement | ~90.5% |

The microbatch count and successful run establish that gradient accumulation remains
sequential. Teacher logits are deleted after each bounded teacher-window group; no
hundreds-microbatch graph or teacher-output retention was introduced.

Production peak VRAM was not separately sampled during this run. The matched
standalone target allocation fell by 768 MiB, and the production run completed with
both teachers resident on the 16 GB device.

## Correctness validation

CuPy-focused suites:

```text
12 passed, 2 skipped
```

Coverage includes:

- Member widths 8, 16, and 64.
- Decoder depths 1, 2, 3, and 4.
- Context heads 0 and 2.
- Member counts 1, 2, 3, 4, and 5 in direct reduction tests.
- Incomplete and multi-tile vocabulary reductions.
- Direct FP32 cuBLAS `beta=1` accumulation parity.
- Direct fused forward/backward CUDA-kernel parity.
- Complete loss and every parameter-gradient tensor against the unfused CuPy/NumPy
  reference path.
- Checkpoint and optimizer-state round trips.
- Full-batch versus multiple gradient-accumulation partitions.
- Online training with bounded teacher windows and mutable accumulation settings.

The broad fusion suite produced:

```text
127 passed, 9 skipped, 1 unrelated failure
```

The unrelated failure is
`test_complete_csv_stream_and_resumption_track_exact_batch_sources`: its legacy CLI
fixture enables context attention but does not provide the now-required validation
context files. It exits in argument validation before fusion computation.

## Rejected experiments

### Recompute `dpre` in both member-gradient consumers

This removed approximately 1 GiB of peak allocation but duplicated expensive
strided activation-derivative work:

| Configuration | Before | Experiment | Memory change |
|---|---:|---:|---:|
| Wide | 53.16 ms | 93.43 ms | 5,253 -> 4,229 MiB |
| Full | 73.14 ms | 113.33 ms | 6,224 -> 5,200 MiB |

The experiment was reverted.

### `float4` decoder elementwise kernels

Direct parity passed, but depth-4 median changed from 24.11 to 24.33 ms. The
specialization was removed because the difference was noise and did not justify the
extra implementation.

### One SGEMM for four scalar feature gradients

A `[4,P] @ [P,K]` contraction selected a large-K SGEMM taking about 8.4 ms. It was
replaced with three GEMVs plus reuse of the decoder bias reduction, reducing the
complete decoder weight-gradient stage from 10.24 to 3.04 ms.

## Remaining recommendations

1. Profile/fuse decoder hidden-layer propagation GEMMs with activation derivatives;
   this is now the largest decoder-specific stage.
2. Consider selective recomputation of `member_var` only if production memory—not
   latency—becomes limiting for larger member widths.
3. Add peak CuPy pool reservation and production peak-device-memory sampling to the
   benchmark harness.
4. Do not revisit the original member-gradient reduction unless a new profile shows
   it dominant; the decoder/member activation backward stages are now larger.
