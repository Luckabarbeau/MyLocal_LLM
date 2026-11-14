# Development log

This is the single durable human/agent change log for MyLocal_LLM. It consolidates the numbered and unnumbered patch reports that were historically spread across the repository root, `docs/`, `docs/research/`, and `benchmarks/`.

The standalone reports were removed after consolidation. Use the migration ledger and `git show <commit>:<path>` to recover their full historical text.

## Current status

- **Protected architecture contract:** Model B is registry ID `model_b_348m_sparse16` / CLI preset `dual-resolution-matrix-pool-sparse-1024-20l`, with **348,261,376 parameters**, 20 layers, width 1024, FFN 4096, four matrix-pooled summaries per completed 128-token block, 512-token local attention, one 4096-token long-local head, and the stride-16 historical head across the established 16-layer phase schedule. This identity is encoded in current code/tests and must not be changed incidentally by refactoring. Its bounded CPU contracts passed during this cleanup; the full-size external run and GPU throughput were not validated here.
- **Checkpoint/run status:** an external **`step165`** artifact or run is **user-reported/unverified**. It is not present as verified repository evidence in this checkout; no checkpoint manifest, hashes, logs, loss, benchmark, or resume test was inspected here. Do not present it as validated until those artifacts are recorded.
- **Specialist status:** production-oriented specialist training code exists through 0107–0108, but the historical 0105 report explicitly left full-size GPU parity, 12-layer memory/performance, long-range quality, and composed generation open. Commit `eb61df0` closes several code and CPU-contract gaps; it does not by itself prove a successful external training run.
- **Historical reports:** consolidated and removed from the working tree; their filenames and Git provenance remain in the migration ledger. New durable changes go in this file, not in a new patch-report Markdown file.

## Mandatory logging policy

1. **Humans and coding agents must log every durable behavior, architecture, compatibility, data-policy, checkpoint, runtime-default, or operational-safety change here.** Pure formatting changes may be omitted.
2. Add entries **newest first**.
3. Every entry must include: **date, author, commit, status, subsystem, behavior, compatibility, validation, and remaining work**.
4. Use explicit status words: `proposed`, `experimental`, `accepted`, `rejected`, `superseded`, or `user-reported/unverified`. A feature can be accepted but remain opt-in; say so.
5. Separate repository-observed evidence from user reports. Record commands and results only when actually run; otherwise write `not run` or cite the historical report as a reported result.
6. Link durable evidence to code, tests, benchmark/result schemas, and commits. Do not make a deleted patch file the only evidence.
7. **Do not create a new patch-report Markdown file.** Update this log, current guides, code comments where necessary, and the relevant tests instead.
8. Never edit, move, overwrite, or benchmark against a live checkpoint/runtime directory without explicit approval and a verified path.

## 2026-11-13 — R9700 local-attention dK/dV key tile 64 (measured backward gain)

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted; enabled by default for the HIP consumer-gpu Model-B training profile, opt-out provided
- **Subsystem:** AMD/HIP Triton local-attention backward schedule
- **Behavior:** `mini_llm/ops/triton_hip_local_kv.py` reads `MINI_LLM_HIP_LOCAL_KV_KEY_TILE` (`32`|`64`, default `32`, invalid values raise `ValueError`) and uses it only for the split-pass dK/dV launch grid (`cdiv(length, key_tile)`) and `block_n`; `block_m=32`, `num_warps=4`, `num_stages=1`, internal `tl.range(num_stages=1)`, dQ path, delta, merge and accumulation dtype are untouched. `mini_llm/runtime_defaults.py` materializes `'64'` inside the existing HIP-only Model-B consumer-training block, so it is recorded in the resume contract and is not applied on CUDA, NumPy, reference, inference or gpu-fast paths.
- **Root cause evidence:** `benchmarks/profile_r9700_local_backward.py` attributes the local backward stage-by-stage. At tile32 (earlier run, `logs/r9700_local_backward_stages_20261113.json`): delta 0.178 ms, dQ long/short 0.307/0.565 ms, **dK 1.487 ms, dV 0.996 ms**, merge 0.142 ms, whole backward 3.764 ms versus forward 0.844 ms. dK held 239 registers with 24% more work but the same 32-row ownership as dV (166 registers); the 64-key-row tile raises dK to 254 registers and dV to 184 with **zero spills**, 8 KiB shared, halving launches from grid 256→128.
- **Validation (kernel level):** `benchmarks/tune_r9700_local_kv_tiles.py` (`logs/r9700_local_kv_tiles_20261113.json`) independently swept dK and dV over (16,32),(32,16),(32,64),(64,32),(64,64) at 4 warps/stage1, warmup2/repeat7, with full dQ/dK/dV parity (atol=rtol=1e-5 and normwise 1e-3). Best for both: **(32,64)** — dK **1.482→0.923 ms (1.605×)**, dV **0.990→0.641 ms (1.544×)**. `32×16`, `64×32` and `64×64` regressed badly (dK 2.73/9.33/5.26 ms), consistent with register/LDS pressure rather than occupancy.
- **Validation (stage level, tile64):** `logs/r9700_local_backward_stages_tile64_20261113.json` reproduces dK 0.929 ms and dV 0.646 ms inside the production call, whole backward event **3.764→2.839 ms (-24.6%)**, forward unchanged at 0.844 ms, parity against the uninstrumented production backward exact (max absolute errors 0.0/0.0/0.0). dK remains the largest single stage (33.8% of summed stage medians).
- **Validation (full model, matched A/B):** `benchmarks/benchmark_r9700_local_kv_ab.py` alternates tile 32/64 per iteration inside one initialized model (`logs/r9700_kv_ab_{8k,16k,1k}_20261113.json`), 6 alternating warmups and 20 measured iterations, full supervision, no optimizer/checkpoint, FFN flag 0, all samples reported untrimmed: 8K **434.40→416.99 ms (+4.01%)**, 16K **904.02→874.51 ms (+3.26%)**, 1K **111.19→110.42 ms (+0.70%)**; forward medians matched within 0.1 ms, so the gain is backward-only. The earlier single-process 64-tile jitter (forward 108→270 ms, total to 765 ms) did not recur in matched sampling, so it was an artifact of comparing separate processes, not the tile change.
- **Validation (production default path):** `logs/r9700_kv_default_full_8k_20261113.json` runs with only `--runtime-profile consumer-gpu` and records `MINI_LLM_HIP_LOCAL_KV_KEY_TILE: 64` in applied profile; forward 108.71 ms, LM loss 7.21 ms, backward **317.5→300.5 ms**, total 432.7→**416.3 ms**, throughput **18,914→19,678 supervised tok/s (+4.04%)** with identical loss 10.610414505004883 and finite/nonzero compressor/global/FFN/embedding gradients.
- **Compatibility:** The reduction order of the FP32 dK/dV accumulators changes with tile ownership, so results are numerically equivalent within tolerance but **not bitwise identical** to the tile-32 path; set `MINI_LLM_HIP_LOCAL_KV_KEY_TILE=32` to restore the prior schedule for ablation or reproducibility against old checkpoints. Older checkpoints will report one additional non-exact-resume delta (`MINI_LLM_HIP_LOCAL_KV_KEY_TILE: None -> 64`) under `--allow-nonexact-resume`; exact resume from pre-change checkpoints requires pinning 32 or accepting that delta. Kernel default remains 32 outside the Model-B HIP training profile, so inference, CUDA and CPU behavior are unchanged. Model B identity, precision boundaries and optimizer state untouched.
- **Validation (tests):** `sh scripts/run_r9700_preflight.sh -m pytest -q tests/test_triton_dual_resolution_local_0102b.py tests/test_triton_dual_resolution_local_hip_split_kv.py tests/test_hip_local_kv_tiles.py`: **25 passed** on the HIP device, including the FP64-oracle packed mixed-window backward and split-KV baseline parity, now running under the tile-64 default (`logs/r9700_local_tile64_tests_20261113.txt`). New `tests/test_hip_local_kv_tiles.py` (11 cases) compares tile 64 against 32 through the real local attention API for packed noncontiguous token/head strides, document boundaries inside key tiles, B2/T65 and B1/T641, skip heads -1/0/5, three long heads/window 4096, a contiguous T8192 case, determinism, and invalid-tile `ValueError`. `MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_hip_local_kv_policy.py tests/test_runtime_defaults.py tests/test_amd_blas_policy.py tests/test_ffn_wgrad_layout.py`: **106 passed**, covering profile materialization, explicit-32 override, and non-materialization for CUDA/NumPy/reference/inference/gpu-fast/non-BF16.
- **GPU safety:** AMD idle confirmed before each GPU command (`amd-smi process`). NVIDIA training, live checkpoint directories and the `model_b_batch1_8k_norecompute_long` source checkpoint were not touched.
- **Remaining work:** dQ long/short (0.857 ms total) plus delta/merge (0.320 ms) are now a larger share of the 2.839 ms backward; the next measured target is dQ (143 registers, no spills) and the FP32 dQ/dK/dV allocation/reduction traffic, not further KV tiling (sweep exhausted the cheap tile space). The strict all-parameter bitwise-equality gate for the separate FFN transpose change remains unresolved and must be re-run for the KV tile too if cross-tile bitwise parity is ever required; numeric parity within 1e-5 is what is established. Long-run optimizer/validation-quality impact of a 4% faster backward is not measured.

## 2026-11-13 — Production AMD bottleneck profile, opt-in FFN layout and BLAS preference

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted opt-in startup configuration; experimental FFN optimization, default disabled pending strict parity resolution
- **Subsystem:** production AMD backend / FFN backward / diagnostics
- **Behavior:** Profiled current root production Model B rather than the older isolated source copy. Added `MINI_LLM_AMD_BLAS=auto|tensile|hipblaslt` via environment-only `mini_llm/amd_blas_policy.py`, invoked before CuPy import in `mini_llm/backend.py`. Auto (default) preserves the raw ROCBLAS setting; explicit policies map to0/1 and log at INFO. No dynamic backend switching or automatic Tensile default. Added guarded opt-in `MINI_LLM_HIP_FFN_TRANSPOSE_WGRAD=1` in `mini_llm/ops/experts.py`: BF16 HIP4096->1024 down-weight gradients at T>=8192 use `(dy.T @ h).T`, retaining FP32 gradient accumulation. Other shapes, dtypes, NumPy and CUDA retain original multiplication.
- **Compatibility:** No architecture, optimizer precision, checkpoint tensor format, or NVIDIA kernel changes. MINI_LLM policies participate in the existing runtime-environment resume contract. Raw externally supplied ROCBLAS settings in auto mode are not captured by that MINI_LLM-only contract; explicit policy is preferable for reproducible experiments. Live checkpoint directories were not accessed. New RVS21.168TFLOP/s report is user-reported and not a directly comparable local CuPy result; earlier measured BF16/FP16 Tensile regressions remain established, so do not force Tensile globally for mixed-precision training.
- **Bottleneck evidence:** `logs/r9700_production_profile_20261113.json`, 8K fully supervised no-checkpoint full model, warmups2/repeats3, synchronized instrumentation: attention backward143.84ms/iteration, FFN backward111.72ms, LM-head backward42.79ms; local attention backward74.66ms is nested within attention. Forward attention59.86ms, FFN50.86ms. Do not sum nested scopes or treat instrumented timings as throughput. Dense/projection GEMMs use CuPy BLAS (including HIP gemmEx compatibility binding); custom Triton/HIP attention, activations and quantized GEMV bypass rocBLAS dispatch. No PyTorch policy was added because this production path uses CuPy.
- **Validation:** `MINI_LLM_BACKEND=numpy python -m pytest -q tests/test_amd_blas_policy.py tests/test_ffn_wgrad_layout.py`:75 passed. `sh scripts/run_r9700_preflight.sh -u benchmarks/check_ffn_wgrad_layout.py --gpu` passed dispatch, odd/strided input, repeated accumulation and full ExpertFFN checks (`logs/r9700_ffn_production_validation_20261113.txt`). Full-model timing commands used current root benchmark, matrix-pool-sparse, context/score8192 or16384, head tile8192, query chunk128, no checkpoint, consumer-gpu, warmups3/repeats9. 8K flag0->1:18,912.65->19,660.69 supervised tok/s (+3.96%), backward317.49->300.68ms. 16K:18,284.30->19,024.84tok/s (+4.05%), backward656.40->619.94ms. Both modes passed finite/nonzero representative gradient checks and identical loss at each length. Artifacts `logs/r9700_ffn_{off,on}_{8k,16k}_20261113.{json,txt}`. These are synthetic fully supervised F+B timings, not data-loader/optimizer production throughput or long-run quality validation.
- **Current verification failure:** New `benchmarks/check_model_b_ffn_wgrad_parity.py` compares all302 gradients on the same model with block recomputation and full8K supervision. Initial execution failed JSON serialization of runtime metadata; one localized correction used `vars(runtime_profile)`. Retest completed both passes with identical loss10.61061954498291 and all gradients finite, but exited unsuccessfully:42 tensors fail the harness's bitwise equality requirement for non-down gradients. Their relative L2 differences range7.83e-10..1.07e-7; worst normwise error across all302 tensors is8.45e-5 for an intended down-weight gradient (below0.001 tolerance). Example `blocks.8.norm2.gamma`: reference norm0.004500676445855849, actual0.004500676442670244, absolute L2 error4.813668390270587e-10, relative1.0695433115844033e-7, FP32 gradient dtype. No derivative finite differences/step size used. Exact non-down tolerance is zero. See `logs/r9700_full_gradient_parity_retest_20261113.txt`. Do not claim the strict parity test passed or production rollout complete. No second corrective edit was made.
- **Remaining work:** Single next experiment: repeat flag0 against flag0 with identical model/input to measure baseline atomic-reduction nondeterminism before changing any parity criterion. The observed tiny differences alone do not justify a production numerical correction. Keep FFN flag default0 until that gate is resolved; do not loosen tolerance merely to obtain a pass. Largest remaining measured kernel target is local-attention backward, not blanket rocBLAS replacement. Selective FP32 solution tuning for actual training shapes and sustained optimizer/validation-quality trials remain open. Earlier isolated-copy nonfinite-gradient failure is not reproduced in these current-root timing runs; no source-history diagnosis of that old copy is claimed.

## 2026-11-12 — R9700 per-call FP32 solution and isolated FFN layout validation

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** experimental; targeted correctness accepted, full-model optimization not promoted
- **Subsystem:** AMD BLAS dispatch / FFN backward / numerical policy
- **Behavior:** Added isolated prototypes under `.r9700_backward_tuning/`: opt-in `MINI_LLM_HIP_FFN_TRANSPOSE_WGRAD=1` changes only HIP BF16 FFN down-weight-gradient orientation for width4096->1024 with at least8192 rows, from `h.T @ dy` to `(dy.T @ h).T`; FP32 gradient accumulation remains unchanged. Separate ctypes rocBLAS probe owns its handle/output and enumerates actual installed solution IDs without changing backend environment. No root production implementation/defaults or live checkpoints changed.
- **Compatibility:** Layout identity preserves real arithmetic but not bitwise reduction order. BF16 output and FP32 gradient buffers remain. Solution IDs are installed-library/architecture-specific and must not be hardcoded as portable defaults. Literal BF16-only training is not implemented: large accumulation counts, small optimizer updates, normalization, attention and CE reductions need separate numerical validation; retain FP32 accumulation/masters for now.
- **Validation:** `sh scripts/run_r9700_preflight.sh -u .r9700_backward_tuning/benchmarks/check_ffn_wgrad_layout.py --gpu` passed 14 dispatch cases, CPU fallback, HIP T1024/8192/16384/8193 and strided cases, determinism, repeated accumulation into nonzero FP32 gradients, full ExpertFFN backward and loss probes. Candidate-vs-original BF16 relative L2 error <=7.87e-5; versus actual BF16 operands multiplied in FP32 ~0.00166 (tolerance0.003). See `logs/r9700_ffn_layout_validation_20261112.txt`. Prior isolated orientation timings including gradient addition: T8192 1.750->0.766ms; T16384 3.379->1.380ms; T1024 slower, excluded.
- **Validation (BLAS):** Initial probe returned positive IDs only and filtered all out. One localized correction removed the sign restriction. Rerun `sh scripts/run_r9700_preflight.sh -u .r9700_backward_tuning/benchmarks/tune_rocblas_solution.py --output logs/r9700_rocblas_solution_tuning_positive_ids.json --max-candidates 5` completed, all five candidates passed pre/post-timing sampled float64 checks. FP32 8192 square default889.717ms; returned solution91221 75.834ms /14.499 dense TFLOP/s, **11.73x** faster, without global Tensile override. Other IDs91210/91211/91208/91209:891.526/304.323/483.851/261.955ms. This demonstrates per-call selection opportunity, not a Model B speedup.
- **Current failure:** Full-model baseline with layout flag0, context16384, all16384 rows supervised, LM-head tile8192, query-chunk128, no checkpoint, consumer-gpu, warmups3/repeats7 failed its finite-gradient assertion before publishing timing: compression/global/embedding nonfinite, FFN finite (`logs/r9700_ffn_layout_off_16k.txt`). No production edit is justified by this failure; the layout optimization was disabled. Previous passing full-model measurements remain historical verified states, not invalidated or replaced by this run.
- **Remaining work:** First reproduce the earlier passing 16K configuration exactly and isolate the baseline nonfinite-gradient discrepancy with parameter-group diagnostics. No specific defective equation has been established, so no speculative correction was made. End-to-end layout speedup, full-model gradient parity, real training-shape solution selection, optimizer/validation trajectory and long-run training quality remain unverified. Do not promote either prototype until these gates pass.

## 2026-11-12 — R9700 rocBLAS backend A/B and telemetry

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted experimental measurements; global Tensile override rejected
- **Subsystem:** benchmark / runtime diagnostics
- **Behavior:** Tested the user-supplied research suggestion using separate processes with `ROCBLAS_USE_HIPBLASLT` unset/0/1, existing plain GEMM benchmark, sizes4096/8192, three warmups, five batches of three GEMMs. No implementation/defaults or GPU power/clock settings changed; NVIDIA training untouched.
- **Compatibility:** FP32 accumulation/output in all precisions, sampled float64-reference checks passed for all 18 cases. Backend setting is process-wide; do not toggle it around calls or enable it globally based only on the FP32 win. No full-model or exact trajectory parity claim.
- **Validation:** At 8192, unset/0/1 dense TFLOP/s: FP32=1.24/13.77/1.24; FP16=115.96/12.79/115.58; BF16=119.78/10.97/118.12. At4096: FP32=15.53/13.44/15.04; FP16=132.10/12.91/127.72; BF16=134.15/11.09/132.24. Forced Tensile removes the 8K FP32 collapse (~11.1x speedup) but is ~9-11x slower for low precision. Reports `logs/r9700_gemm_backend_{default,0,1}_20261112.json`, matching stdout logs. Separate logging-enabled diagnostic run generated rocBLAS gemm_ex replay commands but did not expose selected kernel names. Replay confirms f32_r computation/output for f32_r/f16_r/bf16_r inputs; see `logs/r9700_backend_diagnostic_*_20261112.txt`. AMD SMI telemetry during that short diagnostic reported 100% GFX utilization, 278-302W versus300W cap, 64-72C GPU temperature and GFX_CLK3183-3195MHz in three busy samples (`logs/r9700_backend_telemetry_20261112.txt`). These are tool-reported values, not independently validated clocks; workload phases are not correlated per sample. Do not assume sustained2350MHz to claim85% of peak.
- **Remaining work:** Exact hipBLASLt kernel identity and solution override remain unverified. rocblas-bench/hipblaslt-bench were not found in the standard ROCm bin location; no exhaustive install search performed. Measure Model B's actual GEMM shapes before any selective backend workaround. The external report's linked issue/article claims were not independently verified here; local backend evidence supports a dispatch-dependent performance problem.

## 2026-11-12 — Standalone Triton versus CuPy R9700 GEMM

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted benchmark/evidence; no production dispatch change
- **Subsystem:** benchmark
- **Behavior:** Added `benchmarks/benchmark_r9700_triton_gemm.py`: independent grouped tiled Triton GEMM versus a fresh CuPy gemmEx baseline on identical preallocated inputs. FP32 uses explicit `input_precision='ieee'`, never TF32/BF16 truncation; FP16/BF16 use native low-precision inputs with FP32 accumulation/output. Counts dense `2*N^3` FLOPs. Checks that Triton and CuPy current streams match before GPU-event timing. AMD was idle; no NVIDIA GPU work or live training-code changes.
- **Compatibility:** Benchmark only, no model/checkpoint/runtime-default changes. Tiles and reduction association differ, so comparisons are numerical rather than bitwise. A sampled correctness check is not a full large-matrix parity guarantee. These ordinary tiled kernels are not an instruction-level peak-throughput test.
- **Validation:** `sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_r9700_triton_gemm.py --output logs/r9700_triton_vs_cupy_gemm_20261112.json` passed full 129x129 tail-tile checks and sampled 8x8 large-output checks against float64 CPU calculations of quantized inputs (rtol=2e-4, atol=2e-5). Ten warmups, seven batches of five GEMMs. Best initial Triton FP32/FP16/BF16 dense TFLOP/s: N2048=10.36/87.97/88.64; N4096=11.03/90.79/92.16; N8192=10.69/76.24/77.07. Fresh CuPy values: N2048=14.19/110.51/119.14; N4096=14.98/130.77/133.41; N8192=1.23/115.58/117.71. Larger initial Triton tiles spilled registers (FP32 up to 407 reported spill slots, low-precision up to 80); these were slower, not promoted.
- **Validation (follow-up):** One bounded 8K tile/stage sweep, three warmups and five batches of three GEMMs, passed all sampled checks. Best genuine FP32: tile32x64x32, four warps, stage1, 189 registers, zero spills, 98.875 ms / 11.12 TFLOP/s (~9x faster than the pathological 8K CuPy baseline). Best FP16/BF16: tile64x128x64, four warps, stage1, 200 registers, zero spills, 88.11/90.26 TFLOP/s; still slower than CuPy. Results/configurations/errors in `logs/r9700_triton_gemm_tile_sweep_20261112.json`, matching text log includes each result. No test failures or corrective source edits.
- **Remaining work:** The 8K FP32 collapse is not a hardware FLOP ceiling; replacing the BLAS kernel removes it. Exact rocBLAS kernel/algorithm cause remains unprofiled. Triton is not universally faster: retain BLAS for the measured low-precision cases. Training-shape dispatch or GEMM replacement requires separate full-gradient/end-to-end measurements.

## 2026-11-12 — Plain CuPy R9700 dense GEMM throughput baseline

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted benchmark and measurements; not a hardware-peak claim
- **Subsystem:** benchmark
- **Behavior:** Added standalone `benchmarks/benchmark_r9700_gemm_peak.py`: direct HIP CuPy BLAS GEMMEx, row-major `C=A@B`, FP32/FP16/BF16 inputs, FP32 accumulation/output, preallocated buffers, alpha=1/beta=0. Dense work counted as `2*N^3`; no sparse multiplier. Allocation, casts, host transfers, and correctness checks excluded from timed intervals. No training code/defaults or clock/power settings changed; NVIDIA run untouched.
- **Compatibility:** HIP-only, uses the installed BF16 datatype binding repair. Each reported time is the median of seven GPU-event batches of ten GEMMs after ten warmups. Small output-grid checks use float64 CPU dot products of the actual quantized inputs (rtol=2e-4, atol=2e-5); this is sampled correctness, not exhaustive validation. Runtime SM count reports 32 while ROCm's hardware table identifies R9700 as 64 CUs; do not derive peak FLOPs directly from that runtime count.
- **Validation:** `sh scripts/run_r9700_preflight.sh -u benchmarks/benchmark_r9700_gemm_peak.py --output logs/r9700_plain_gemm_peak_20261112.json` completed successfully; all nine sampled checks passed. FP32/FP16/BF16 dense TFLOP/s respectively: N2048=14.31/118.82/120.93; N4096=15.00/123.15/126.23; N8192=1.23/115.62/117.97. Independent ordinary CuPy FP32 `matmul` cross-check reproduced ~14.81 TFLOP/s at 4096 and ~1.24 at 8192 (`logs/r9700_plain_fp32_matmul_crosscheck_20261112.json`). Sapphire's linked compute brochure explicitly lists R9700 FP32 vector 47.8 TFLOPS, FP16 matrix dense 191 and sparse 383: `https://cdn.sapphiretech.global/assets/documents/commercial/ai-pro/AI-PRO_TFLOPS_TOPS_Brochure_17Aug26.pdf`, downloaded as `logs/r9700_vendor_compute_specs.pdf`. That table does not separately list BF16. ROCm SMI's combined clock query aborted; idle-use/memory queries worked. No compute test failed.
- **Remaining work:** The very slow 8192 FP32 default BLAS path is an observed size-dependent performance anomaly, not the card's theoretical limit. Exact cause and alternate algorithms/Triton kernels are untested. BF16 theoretical peak needs a separately verified issue-rate reference; do not substitute sparse AI TOPS or FP16 vector figures for dense matrix throughput.

## 2026-11-11 — R9700 full-supervision 65K forward/backward test

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted smoke/performance measurement; no-checkpoint configuration rejected for memory
- **Subsystem:** training / benchmark
- **Behavior:** Tested unchanged 348,261,376-parameter matrix-pool-sparse Model B at batch 1 and 65,536 context with all 65,536 next-token rows supervised, BF16, fine shapes, LM-head tile 8192 and local query chunk 1536. AMD GPU verified idle before execution. No production defaults, implementation or user checkpoints changed.
- **Compatibility:** Synthetic randomly initialized model, not a production checkpoint or training-quality evaluation. No optimizer state/update included. Full supervision, not a tail-only surrogate. Finite/nonzero group checks do not establish full gradient parity.
- **Validation:** No activation checkpointing: failed with CuPy OOM during body forward, requesting another 256 MiB with 33,634,123,776 bytes already allocated. Retried once using existing block activation checkpointing: one warmup and three measured complete forward/loss/backward iterations passed. Median forward 979.986 ms, loss 58.026 ms, backward 3996.434 ms; throughput 13,017.6 supervised tok/s, approximately 5.034 s per iteration. Compressor/global/FFN/embedding monitored gradients finite and nonzero. Device-memory samples approximately 13.90 GiB, reserved pool 13.20 GiB; samples are not exact peak measurements. Reports: `logs/r9700_65k_fullsupervision_blockcheckpoint_20261111.json` and matching text log; failed run: `logs/r9700_65k_fullsupervision_nocheckpoint_20261111.txt`.
- **Remaining work:** Production-resume optimizer-inclusive 65K fit/throughput and long-context quality/parity remain untested. This does not establish that 65K fits without activation checkpointing or with every optimizer-state placement.

## 2026-11-11 — R9700 production phase profile and shape-dependent tuning

- **Author:** GPT-6.1-Sol with Lucka
- **Commit:** uncommitted (working tree)
- **Status:** accepted measurements; rejected default changes; fused pool forward remains experimental
- **Subsystem:** training / benchmark / documentation
- **Behavior:** Ran two bounded real Model B updates from `checkpoints/model_b_batch1_8k_norecompute_long`, batch 1, accumulation 128, context 8192, fine buckets, source-preserving length grouping, Hybrid Muon and no optimizer offload. Used fresh `checkpoints/r9700_tuning_phase_20261111` and `logs/r9700_tuning_phase_20261111.*`; preserved source and existing user outputs. No production code or runtime defaults changed.
- **Compatibility:** Existing uncommitted optimizer-placement/resume changes were left untouched. Profiling synchronizes nested scopes and perturbs throughput; nested timings must not be summed. The profiler's 'effective throughput' uses nominal capacity, not supervised/executed rows. Synthetic benchmark losses/finite gradients are not proof of full gradient parity.
- **Validation:** Production run completed and saved a separate step-15152 checkpoint. Second profiled step: wall 32.568 s; model forward 8.824 s, model backward 18.200 s, loss forward 0.398 s, loss backward 4.162 s, optimizer 0.595 s, clipping 0.252 s, data preparation 0.097 s, buffer refresh 0.003 s and zero-grad 0.004 s. Nested dominant regions include FFN backward 6.154 s, local-attention backward 4.342 s and output-projection backward 3.988 s. All synthetic cases used `benchmarks/benchmark_dual_resolution_full_model_0102d.py`, matrix-pool-sparse, model-context 8192, batch 1, full supervision, no activation checkpoint, query chunk 1536, consumer-gpu profile, three warmups and nine measured repeats. Baseline throughput by active length: 256 = 3,126 tok/s; 1024 = 9,396; 4096 = 17,333; 8192 = 18,794. Packed FFN: 8,903 at 1K and 16,960 at 8K, slower than separate projections; keep off. Generic local probability recompute=0: 9,392/18,747 at 1K/8K, no useful improvement. Fused pool forward: 9,519/18,809 at 1K/8K; scalar losses differ from reference by approximately 7.82e-5/3.81e-5, so no promotion without parity checks. At 8K, LM-head chunks 1024/4096 produce 13,467/17,725 tok/s versus 18,794 with chunk 8192; retain 8192. Every synthetic report records finite, nonzero gradients in compressor/global/FFN/embedding groups. Reports: `logs/r9700_tune_{ffn,norecompute,pool,length,head1024,head4096}_*_20261111.json` and matching text logs. No unit tests were run because no implementation changed.
- **Remaining work:** Actual short-document shape efficiency is a demonstrated factor; its exact contribution to the mixed production workload is not yet quantified. No large no-offload optimizer-transfer tax was observed. Further implementation work should target measured short-shape FFN/local-attention/LM-head backward costs, with reference gradient checks and warmed matched A/B before changing defaults. Do not infer a safe speedup from the small fused-forward result or treat aggregate timings as proof of host launch starvation.

## Mandatory future-entry template

Copy this block directly below this section and fill every field:

```markdown
## YYYY-MM-DD — Short durable-change title

- **Author:** Human name or agent/model plus operator
- **Commit:** `<hash>` or `uncommitted (working tree)`
- **Status:** proposed | experimental | accepted | rejected | superseded | user-reported/unverified
- **Subsystem:** architecture / training / inference / serving / data / checkpoint / benchmark / documentation
- **Behavior:** What changed, including defaults and opt-in flags.
- **Compatibility:** Parameter/checkpoint/API/data/runtime impact; migration or rollback requirements.
- **Validation:** Exact commands and observed results, or `not run` with the reason. Label user-reported evidence.
- **Remaining work:** Open correctness, performance, quality, migration, or cleanup gates.
- **Evidence:** `path/to/code.py`, `path/to/test.py`, benchmark result paths/schemas, and relevant commits.
```

---

## 2026-11-04 — Second readability and duplication cleanup

- **Author:** GPT-5.6-Sol coding agent, at user request
- **Commit:** `uncommitted (working tree)`
- **Status:** accepted documentation/test-maintenance cleanup; high-risk runtime refactors deferred
- **Subsystem:** documentation, tests, benchmark maintenance, repository hygiene
- **Behavior:** Established explicit canonical ownership for the landing page, exact baseline, operational guide, developer APIs, checkpoints, data, tests, benchmarks, roadmap, and development history. Reduced `README.md` to a compact landing page and moved the full native/SmolLM2 GGUF/Safetensors/remote serving workflow into `docs/USER_GUIDE.md`; standardized local examples on `serving_models.local.json` and port 8000; defined foundation and specialist flag arrays once; expanded `docs/FUTURE_DEVELOPMENT.md` into a risk-ranked cleanup plan with affected symbols, benefits, gates, and prerequisites. Added all 0107–0108 specialist contracts to the permanent CPU runner, made the optional Safetensors test skip cleanly when its extra is absent, corrected stale benchmark/data/history instructions, and ignored local downloaded-model/registry state. Removed the unreferenced `tests/fixtures/0080l_cpu_reference.json` and two obsolete shell wrappers, `benchmarks/run_checkpoint_evaluations.sh` and `benchmarks/run_fusion_evaluations.sh`, which hard-coded absent historical checkpoints and duplicated maintained runners.
- **Compatibility:** No model mathematics, parameter names/order, training hot path, optimizer, sampler, checkpoint schema/loading, resume semantics, inference kernel, or serving implementation changed. The removed fixture had no consumers; the removed wrappers targeted retired experiments and had no repository references. Optional Safetensors coverage still runs when the dependency is installed and is an explicit skip otherwise. Existing checkpoints, including a future immutable copy of the protected Model B run, retain the same code compatibility boundary.
- **Validation:** CPU-only collection succeeded with **2,203 tests collected** and no optional-Safetensors import error. The expanded permanent CPU gate passed **282 tests with 4 expected skips** across ten isolated suites, including **30** 0107–0108 specialist tests. Focused cleanup/regression and historical benchmark contracts passed **49 tests with 5 expected skips**. All local Markdown links resolved, both serving example JSON files parsed, removed-file references were absent, and `git diff --check` passed. No GPU test, model download, checkpoint load, or training step was run.
- **Remaining work:** Follow the prioritized work in `docs/FUTURE_DEVELOPMENT.md`: extract benchmark/test helpers, add explicit item-level GPU opt-in, decide the orphan SFT workflow and old launcher policy, then consider medium-risk constant/preset/tokenizer/inference consolidation one change at a time. Trainer, checkpoint/resume, config/inference-model, generation-adapter, and process-global runtime refactors remain gated on copied-checkpoint and idle-GPU parity.
- **Evidence:** `README.md`, `docs/README.md`, `docs/CURRENT_BASELINE.md`, `docs/USER_GUIDE.md`, `docs/MODEL_CENTERED_DEVELOPER_GUIDE.md`, `docs/CHECKPOINT_COMPATIBILITY.md`, `docs/FUTURE_DEVELOPMENT.md`, `tests/README.md`, `scripts/run_0103a_regressions.py`, `tests/test_regression_runner_0103a.py`, `tests/test_0085c_external_safetensors.py`, `/tmp/model_b_cpu_cleanup.json`.

## 2026-11-04 — Restore pretrained-model loading and local serving guide

- **Author:** GPT-5.6-Sol coding agent, at user request
- **Commit:** `uncommitted (working tree)`
- **Status:** accepted documentation and example configuration
- **Subsystem:** model loading, serving, external models, documentation
- **Behavior:** Restored the useful end-to-end workflow from the former README using current interfaces and model names. The root README now documents compatible native, GGUF, Safetensors, and remote sources; an official SmolLM2 Safetensors download; GGUF inspection and direct generation; mixed registry validation; lazy local serving; completion/chat/streaming requests; IDE base-URL use; remote-provider proxying; and security/compatibility limits. Updated `serving_models.example.json` from the retired native-338M example to Model B plus SmolLM2 GGUF and Safetensors entries.
- **Compatibility:** Documentation and example-registry change only. No loader, tokenizer, model, checkpoint, inference kernel, server implementation, or training hot path changed. Registry paths remain placeholders and models are loaded lazily.
- **Validation:** The example registry parsed through `ModelCatalog` and resolved `native`, `gguf`, and `safetensors` formats. Serving/native-loading/GGUF/tokenizer contracts passed **48 tests**. `serve.py --help`, `benchmarks.try_external_gguf --help`, all local Markdown links, JSON parsing, and `git diff --check` passed. The Safetensors-specific test could not collect because this environment lacks the optional `safetensors` package; no weights were downloaded and no GPU/model execution was performed.
- **Remaining work:** Validate the documented SmolLM2 Safetensors and GGUF paths with immutable model files on an idle GPU, record exact model/tokenizer hashes, and verify generated prompt framing and output parity before enabling strict fast paths or tool envelopes.
- **Evidence:** `README.md`, `serving_models.example.json`, `serve.py`, `mini_llm/model_sources.py`, `mini_llm/model_loading.py`, `mini_llm/serving_api.py`, `benchmarks/try_external_gguf.py`, serving and external-loader tests under `tests/`.

## 2026-11-04 — Consolidate durable change history

- **Author:** GPT-5.6-Sol coding agent, at user request
- **Commit:** `uncommitted (working tree)`
- **Status:** accepted documentation policy and completed report consolidation
- **Subsystem:** documentation and project governance
- **Behavior:** Established this file as the sole future change log, consolidated historical status and negative results, added a deletion-safe migration ledger, and removed superseded standalone reports. Registered `tests/test_unified_benchmark_0103g.py` directly in the permanent CPU regression runner, then removed its one-shot registration helper. Removed unlinked generic repository-plan TeX/PDF/SVG assets that predated and did not describe current Model B. Cleared regenerable caches and editable-install metadata while preserving runtime data and the virtual environment.
- **Compatibility:** No model, optimizer, checkpoint, data, inference, or training hot-path code changed. The CPU regression gate gained one existing test suite; runtime/checkpoint/data directories were untouched.
- **Validation:** After direct 0103G registration, the bounded CPU runner passed **252 tests with 4 expected skips** across nine isolated suites. Focused checkpoint/profile tests passed 18 tests. The production audit matched all 36 environment values and 27 CLI options. All local Markdown links resolved, `git diff --check` passed, and the registry reported 348,261,376 parameters. No GPU test was run.
- **Remaining work:** Keep the ledger current and reject future standalone patch-report Markdown files; recover full historical reports from Git only when needed. GPU throughput and the unavailable external step-165 checkpoint remain unverified.
- **Evidence:** `docs/DEVELOPMENT_LOG.md`, `docs/README.md`, `docs/FUTURE_DEVELOPMENT.md`, `docs/research/README.md`, `scripts/run_0103a_regressions.py`, `tests/test_regression_runner_0103a.py`.

## 2026-11-03 — 0107–0108 production specialist hardening

- **Author:** Askael
- **Commit:** `eb61df0` (`specialist training is now up`)
- **Status:** accepted implementation; external run success remains user-reported/unverified
- **Subsystem:** specialist training, Model B compatibility, logging, checkpoint resume
- **Behavior:** Added fail-closed production provenance checks; required the sparse Model B foundation; derived specialists from exact foundation block geometry; supported positive arbitrary microbatch sizes with rowwise masks; added accumulation length regrouping, stage profiling, source-weighted CE, optional same-minibatch foundation CE, production-style logging, exact sparse-phase coverage, and prepared-RoPE reuse through checkpoint replay.
- **Compatibility:** The production launcher now requires CuPy and exact Model B geometry and rejects mismatched specialist IDs/layer counts on resume. Legacy tiny/reference specialist geometry remains for tests and old reference checkpoints. A 16-layer specialist maps exactly once across all 16 stride phases. Foundation parameters remain frozen. Repacking is target-equivalent but can change floating-point/order behavior; resumed sampler batch size must match.
- **Validation:** Commit includes CPU contract tests, but no tests were rerun for this documentation migration. The external `step165` claim has no inspected repository artifact and is **user-reported/unverified**.
- **Remaining work:** Verify the external run with immutable checkpoint/config/tokenizer hashes, logs and resume; run idle-GPU full Model B/specialist parity; measure 12-layer context-8192 VRAM and throughput; validate long-range quality and composed incremental generation.
- **Evidence:** `mini_llm/specialists/production_contract_0107.py`, `mini_llm/specialists/train_logging_0107c.py`, `mini_llm/specialists/residual_specialist.py`, `mini_llm/specialists/training_run.py`, `scripts/train_0105_specialist.py`, `tests/test_specialist_production_contract_0107a.py`, `tests/test_specialist_profile_0107b.py`, `tests/test_specialist_train_logging_0107c.py`, `tests/test_specialist_multibatch_0107d.py`, `tests/test_specialist_model_b_shape_0108.py`, `tests/test_specialist_prepared_rope_0108.py`.

## 2026-11-02 — 0106 external SmolLM2 retrieval control

- **Author:** Askael
- **Commit:** `68844dd`
- **Status:** accepted benchmark adapter; GPU/model result unverified
- **Subsystem:** external-model evaluation and long-context retrieval
- **Behavior:** Added a SmolLM2 GGUF runner for the same 0097 synthetic retrieval bank, with shared case/scoring definitions, explicit unsupported cases, own-tokenizer positions, optional counterfactuals, and chunked external scoring coverage.
- **Compatibility:** Does not modify Model B or extend the external model's native context. Prompts beyond native context are skipped, never silently truncated. Full-causal external prefill can have materially different memory complexity from Model B, so throughput is not directly comparable.
- **Validation:** Historical report says CPU contract tests were used; GPU GGUF inference was not run during development. No tests were rerun here.
- **Remaining work:** Run on a free GPU, record exact GGUF/tokenizer hashes, native context, unsupported cases, confusion matrix and answer bias; do not infer 16K–32K capability from an 8K-native model.
- **Evidence:** `benchmarks/evaluate_external_long_retrieval.py`, `tests/test_0106a_external_long_retrieval_control.py`, `tests/test_0106b_chunked_external_retrieval.py`, commit `68844dd`.

## 2026-11-01 — 0105 specialist training gates

- **Author:** Askael
- **Commit:** `8586f79` (with model-centered groundwork in `bf8f6e2` and `4baae24`)
- **Status:** accepted CPU/reference plumbing; superseded in part by 0107–0108; not production-validated by this report
- **Subsystem:** training API, frozen residual specialists, production profile
- **Behavior:** Added a read-only Model B profile audit, CPU trajectory parity between the model-centered `TrainingRun` and legacy path, a document-first specialist sampler, specialist-only Hybrid-Muon training/checkpoint/resume, and a guarded dry-run-first launcher.
- **Compatibility:** Explicitly preserves Model B's 348,261,376-parameter schema, sparse head, legacy foundation trainer, optimizer and active checkpoint. Specialist checkpoints are independent and bind to a foundation hash. Initial rows are single-document because packed multi-document specialist isolation was not yet implemented in 0105.
- **Validation:** Historical report records CPU testing only and states no GPU hardware was available in its authoring environment. No tests were rerun here.
- **Remaining work:** 0107–0108 addressed exact production geometry, multibatch masks, logging and RoPE reuse. GPU full-model parity, full 12-layer performance/VRAM, long-range quality and composed serving remain open.
- **Evidence:** `mini_llm/production_profile_0105.py`, `mini_llm/specialists/`, `scripts/train_0105_specialist.py`, `tests/test_model_b_gpu_contract_0105b.py`, `tests/test_training_path_parity_0105c.py`, `tests/test_specialist_sampler_0105d.py`, `tests/test_specialist_training_run_0105e.py`, `tests/test_specialist_cli_0105f.py`.

## 2026-10-30 to 2026-11-01 — 0103 model-centered API, execution, serving and evaluation

- **Author:** Lucka and Askael
- **Commits:** `7bc41a2` (WIP), `ae248a9` (explicit revert), `a15f664` (accepted cleanup/model facade), `fd20236` (0103K fine buckets), plus merge/continuation commits `4baae24` and `bf8f6e2`
- **Status:** accepted additive compatibility foundation; WIP remodel superseded/reverted; GPU gates remain open
- **Subsystem:** public model API, loading, checkpoint safety, generation sessions, serving, evaluation, sequence bucketing
- **Behavior:** Introduced strict `Model` loading/inspection, regression gates for protected Model B, canonical native/external loading, explicit execution policies and observed CUDA-Graph reporting, request-scoped unified generation, mixed local/remote OpenAI-compatible serving, incremental unified evaluation, and opt-in fine sequence-length buckets.
- **Compatibility:** Existing CLIs and legacy imports remain compatibility paths. Missing or shape-incompatible native weights fail closed. Remote providers do not expose native logits/KV/training. Runtime backend flags remain process-global. `strict-fast` requires observed graph execution rather than a requested flag. Fine buckets are opt-in and do not alter sampled targets/checkpoints, but increase possible compiled shapes. Model B remains exactly 348,261,376 parameters with the stride-16 head.
- **Validation:** Historical reports record CPU regression suites. 0103A2 explicitly classified low-VRAM GPU runs as blocked, not passed, and states GPU parity remained unverified in its authoring environment. No tests were rerun here.
- **Important negative results:** a production-width Triton gate hit CUDA OOM while another 348M job may have occupied VRAM; this was not a numerical mismatch. CUDA Graph support was not established for every model. Per-instance backend isolation and append-only conversational KV continuation were not implemented. Unified evaluation did not replace specialist scorers.
- **Remaining work:** Run GPU gates on an idle device; verify Model B graph capture, long prefill/decode, full-scale loading and checkpoint transaction paths; complete per-model backend isolation and remaining benchmark migration only with golden-result parity.
- **Evidence:** `mini_llm/model_api.py`, `mini_llm/model_loading.py`, `mini_llm/model_sources.py`, `mini_llm/generation.py`, `scripts/run_0103a_regressions.py`, `benchmarks/run_unified.py`, `tests/test_model_api_0103_0.py`, `tests/test_baseline_contract_0103a.py`, `tests/test_regression_runner_0103a.py`, `tests/test_canonical_model_loading_0103c.py`, `tests/test_inference_policy_0103d.py`, `tests/test_generation_session_0103e.py`, `tests/test_remote_serving_0103f.py`, `tests/test_dual_resolution_fine_shape_buckets_0103k.py`.

## 2026-10-28 to 2026-10-29 — 0102 dual-resolution attention and protected Model B

- **Author:** Askael
- **Commits:** `7b64c77`, `9dbd60c`, `5dec566`, `a6cb830`, `52b8ae3` (with benchmark correction `edfcb72`)
- **Status:** mixed: accepted architecture/correctness work, accepted defaults, and retained experimental/default-off alternatives
- **Subsystem:** architecture, document-first training, Triton kernels, memory/throughput, inference
- **Behavior:**
  - **0102 base–G — accepted correctness foundation:** added dual-resolution local attention plus completed-block compression/global summaries, bounded inference, full-model smoke tools, document-first full-sequence supervision, efficient metadata reuse/short paths, and the unequal-length summary-RoPE sentinel fix.
  - **0102B–C — experimental/default-off at introduction:** optional Triton local/global/compressor kernels; authoring reports did not claim GPU compilation or production readiness.
  - **0102F negative constraint:** 65K backward could OOM on 16 GB; short runs were not evidence of 65K correctness. Long documents must not be silently skipped/truncated.
  - **0102H–J — accepted after a negative first measurement:** the first 0102H single cold run showed a reported 3.62× slowdown. 0102I identified cold/JIT confounding and added warmed A/B. 0102J reported warmed gains and promoted packed/BF16 projections as the CuPy default; NumPy retained exact reference behavior. Fused pooling remained opt-in.
  - **0102K–M — fixes/experiments:** fixed Triton pointer launch; compressor RMSNorm and local/global gate fusion remained default-off pending GPU parity and end-to-end evidence.
  - **0102N — experimental opt-in:** BF16 output-projection `dX`; preserved FP32 parameter-gradient accumulation and exposed global backward as the next bottleneck.
  - **0102O — accepted Model B:** introduced matrix-pooled Model B as a separate architecture with 348,261,376 parameters; the prior 406,199,296-parameter dual-resolution model remained available.
  - **0102P–T — mixed:** benchmark ablation fix accepted; Q contiguous view and R metadata reuse default-on with guarded fallbacks; S fused matrix-pool forward remained off. Its later measured +0.34% was within variation and was not promoted on that evidence.
  - **0102U–W — experimental/default-off:** fused matrix-pool backward, chunked packed-projection backward, and empty-global-tile skipping retained fallbacks and required GPU parity/memory/throughput measurements.
  - **0102X — opt-in:** source-preserving accumulation regrouping; target-equivalent but not bitwise resume-equivalent because document order within an optimizer step changes.
  - **0102Y — accepted default optimization:** prepared local RoPE reuse; no unmeasured production speedup was claimed.
  - **0102Z — accepted protected architecture extension:** added the deterministic stride-16 original-token head with no new learned tensors, established phases and same-document causality. This became the protected Model B contract in 0103.
- **Compatibility:** Most performance work preserved parameter names/checkpoint layouts and retained reference fallbacks. Model B is a separate architecture, not an in-place reinterpretation of the 406M model. Weight initialization from a prior Model B checkpoint is weights-only; optimizer state must not be silently reused across architecture changes. Data-policy changes apply only to dual-resolution training. Environment flags and effective defaults must be logged with runs.
- **Validation:** Individual historical reports record CPU and selected RTX results; several explicitly say GPU tests were not run in their authoring environment. No tests were rerun here.
- **Remaining work:** Preserve Model B exactly; verify any active/external checkpoint manifest; run idle-GPU parity and full optimizer-step measurements; keep rejected/default-off paths off unless a new logged gate promotes them; do not infer 65K fit from reduced-tail smoke benchmarks.
- **Evidence:** `mini_llm/ops/dual_resolution_attention.py`, `mini_llm/ops/matrix_pool_dual_resolution_attention.py`, `mini_llm/ops/dual_resolution_attention_inference.py`, `mini_llm/ops/triton_dual_resolution_local.py`, `mini_llm/ops/triton_dual_resolution_dense.py`, `mini_llm/config.py`, `mini_llm/train_extended.py`, `tests/test_dual_resolution_0102.py`, `tests/test_dual_resolution_document_sampling_0102e.py`, `tests/test_dual_resolution_multibatch_rope_0102g.py`, `tests/test_dual_resolution_packed_projections_0102h.py`, `tests/test_matrix_pool_dual_resolution_0102o.py`, `tests/test_matrix_pool_contiguous_0102q.py`, `tests/test_matrix_pool_metadata_0102r.py`, `tests/test_matrix_pool_fused_0102s.py`, `tests/test_matrix_pool_fused_backward_0102u.py`, `tests/test_packed_projection_chunk_0102v.py`, `tests/test_global_skip_empty_tiles_0102w.py`, `tests/test_dual_resolution_accum_length_buckets_0102x.py`, `tests/test_matrix_pool_prepared_local_rope_0102y.py`, `tests/test_sparse_stride_history_0102z.py`.

## 2026-10-27 — 0100 independent full-history heads

- **Author:** Lucka
- **Commit:** `f98bbe5`
- **Status:** accepted opt-in research implementation; superseded as the current baseline by Model B; target-GPU production gate incomplete
- **Subsystem:** historical attention, checkpoint migration, training/inference
- **Behavior:** Added independent per-layer full-history heads with zero-output initialization, rectangular streaming Triton attention, valid-row compaction, exact positions, handwritten backward, cache integration, and transactional checkpoint/optimizer migration.
- **Compatibility:** Adds 2,097,152 parameters for the documented eight-layer layout and therefore requires explicit conversion to a new checkpoint directory. Source checkpoints are preserved; conversion is atomic and validates optimizer groups/birth steps. Arbitrary dense CUDA masks and provisional/shared-context routes remain unsupported. It is a historical alternative, not Model B.
- **Validation:** Historical report records CPU coverage and functional tests on a 4 GiB Quadro P1000. It explicitly states RTX 5080 4K×65K throughput, 16 GiB fit, microbatch 3, real migration and two-step resume were not executed there. No tests were rerun here.
- **Important negative results:** P1000 timings cannot be extrapolated to RTX 5080; numerical differences versus a square baseline were not evidence of a 4.7–15× intrinsic speedup. CUDA-Graph capture and strict refresh parity remained open.
- **Remaining work:** Treat as historical unless revived deliberately; if revived, use copied checkpoints and rerun target-GPU migration/resume/parity/memory gates.
- **Evidence:** `mini_llm/ops/full_history_head.py`, `mini_llm/ops/triton_full_history_attention.py`, `examples/migrate_full_history_heads.py`, `tests/test_full_history_upgrade_0100.py`, `tests/test_full_history_triton_0100.py`, commit `f98bbe5`.

## 2026-10-26 — 0092 live benchmark and supervised-token instrumentation

- **Author:** Lucka
- **Commit:** `58e8799`
- **Status:** accepted instrumentation; benchmark quality results not implied
- **Subsystem:** training-time evaluation and token accounting
- **Behavior:** Added optional live benchmark stops, exact supervised-token tracking, per-benchmark result schemas and resumable reporting.
- **Compatibility:** Instrumentation is operationally optional and can add evaluation cost. `steps × batch × context` is not a valid substitute for supervised-token counts. Branching from a benchmark result was not implemented in 0092.
- **Validation:** Historical report records CPU validation before GPU handoff; no tests were rerun here.
- **Remaining work:** Keep benchmark definitions/version hashes stable and distinguish model quality from instrumentation completion.
- **Evidence:** `benchmarks/live_suite.py`, `benchmarks/configs/`, benchmark tests under `tests/`, commit `58e8799`.

## 2026-10-26 — 0090 residual specialist prototype

- **Author:** Lucka
- **Commit:** `41c8fa2`
- **Status:** accepted CPU prototype; superseded by 0105 and 0107–0108 for current specialist training
- **Subsystem:** frozen-foundation residual specialists and research design
- **Behavior:** Established a frozen foundation plus zero-initialized hidden residual specialist, strict no-foundation-gradient behavior, specialist-only serialization concepts, and a two-layer CPU smoke implementation. The intended scientific specialist was 12 layers; two layers were never a capacity claim.
- **Compatibility:** Historical prototype targeted the earlier 338M historical-hybrid foundation and did not implement private specialist history, 12-layer GPU execution, BF16 production validation, or a trained ensemble combiner. The 0090B LaTeX note is a design/evidence record, not proof of a specialist ensemble.
- **Validation:** Historical 0090A report records `43 passed, 1 skipped` on NumPy and explicitly says no BF16/64K production test was performed. Not rerun here.
- **Remaining work:** Current work should use the 0105/0107/0108 Model B path and revalidate all scientific conclusions against protected Model B.
- **Evidence:** `mini_llm/specialists/`, current specialist tests, commit `41c8fa2`; historical design source `docs/research/0090B_modular_residual_specialist_combination.tex`.

## 2026-10-24 to 2026-10-26 — 0089 profiling, accepted fusions, rejected kernels and historical-read optimization

- **Author:** Askael
- **Commits:** `96ed2c3`, `4488335`, `33fceba`
- **Status:** mixed; see decisions below
- **Subsystem:** GPU performance, numerical parity, historical-read kernels, framework migration analysis
- **Behavior and decisions:**
  - **0089A–E:** accepted benchmark/profiling infrastructure. 0089C residual/RMSNorm fusion was promoted in GPU-fast profiles after reported parity and modest complete-block gains. Compiled-Torch block/LM-head/stack comparisons were references, not approval to replace the trainer. A packed gate/up GEMM ablation was rejected.
  - **0089J–K:** accepted fair comparison and correlation-correct profiling infrastructure; historical work, not host launch starvation, dominated the heavy-history case.
  - **0089L:** vectorized BF16 SwiGLU was **promoted** despite a small reported complete-training gain because direct output was bitwise, paired runs were consistently positive, and no durable allocation was added.
  - **0089M:** inverse-RoPE vectorization result was **rejected as confounded**; measured apparent gain exceeded the physical scope bound. Static depth-1 CUDA Graph work was **rejected** because the measured ideal host-side bound was about 0.12% with high semantic risk.
  - **0089N:** fused packed-QKV RoPE forward was **accepted/promoted** for CuPy fast profiles. Full PyTorch migration was **rejected/deferred**: historical custom ops, 61K fit, Hybrid-Muon offload and exact resume were not demonstrated.
  - **0089O:** accepted correlation-correct historical audit and staged optimization plan.
  - **0089P:** first streaming deterministic block forward was **rejected and removed**. It was fast and reduced reservation, but failed complete-model output/gradient parity (`body rel-L2 5.611e-3`, global-gradient rel-L2 `1.3895e-2`, worst cosine `0.998038`).
  - **0089R:** corrected two-pass forward remained **rejected for default training**; full-model parity still failed (`body rel-L2 5.5648e-3`, gradient rel-L2 `1.3625e-2`, worst cosine `0.9923513`) despite small isolated error and scalar-loss difference.
  - **0089S:** accepted generic kernel/prewarm plumbing only; it did not promote the forward.
  - **0089T:** retained the deterministic historical forward as an **explicit opt-in**, not numerical parity and not a silent default. Direct production calls were near `1e-6`, but full-model differences remained. Reported ~25k→slightly above 26k supervised tok/s was explicitly **user-observed**, not a controlled artifact.
  - **0089U:** historical failed shared-memory prototype, **superseded by 0089V**.
  - **0089V:** retained streaming deterministic historical backward as an **explicit opt-in**. Historical report measured a 10.46% no-optimizer F+B reduction for the fixed workload; complete BF16 gradient rel-L2 `0.003442` still failed the old strict threshold, so it was not made a silent default.
  - **Unnumbered fusion optimization:** accepted several probability-fusion GPU changes; rejected duplicate `dpre` recomputation (large slowdown despite ~1 GiB saving), `float4` elementwise specialization (noise), and one large SGEMM for scalar gradients. One broad-suite failure was unrelated argument validation, not silently counted as a pass.
- **Compatibility:** Promoted CuPy fusions preserved parameter/checkpoint formats and FP32 parameter-gradient accumulation. Historical forward/backward optimizations change floating-point reduction order and must be recorded in run metadata; do not switch them silently mid-run. PyTorch model weights could be mapped, but optimizer/checkpoint semantics were not compatible with stock `torch.optim`.
- **Validation:** Results above are historical report records, not tests run during this migration.
- **Remaining work:** Preserve negative results; do not resurrect removed kernels without full-model gates. Any framework migration must first pass compact historical correctness, memory, optimizer and native-resume gates. Validate opt-in historical kernels over controlled training/validation trajectories.
- **Evidence:** `mini_llm/ops/triton_historical_block.py`, `mini_llm/ops/historical_read.py`, `benchmarks/profile_0087a_training.py`, `benchmarks/analyze_0089o_historical_nsys.py`, `tests/test_0089a_optimized_training.py`, `tests/test_0089b_block_contract.py`, `tests/test_0089c_residual_rmsnorm.py`, `tests/test_0089d_lm_head_contract.py`, `tests/test_0089e_block_stack_contract.py`, `tests/test_0089j_optimization_comparison.py`, `tests/test_0089k_block_stack_nsys.py`, `tests/test_0089n_fused_qkv_rope.py`, `tests/test_0089o_historical_nsys.py`, commits `96ed2c3`, `4488335`, `33fceba`.

## 2026-10-20 to 2026-10-22 — 0084–0088 fusion, external models, attention and serving

- **Author:** Askael
- **Commits:** `801167d`, `68ba347`, `69c5826`, `5516c61`, `1a54a06`, `647c280`
- **Status:** mixed accepted infrastructure and rejected/default-off kernel candidates
- **Subsystem:** probability fusion, GGUF, attention performance, inference and serving
- **Behavior and decisions:**
  - **0084B:** accepted online frozen-teacher probability fusion with bounded teacher windows and partition/provenance controls. It did not prove fused quality or make cross-tokenizer comparisons valid.
  - **0085H:** accepted WikiText-2 adapter for supported GGUF checkpoints; the report left full real-model WikiText execution pending.
  - **0086A:** accepted attention baseline/oracle audit; no production files changed.
  - **0086B:** fused local forward candidate **rejected for production** because complete-step throughput regressed; grouped cuBLAS remained default and custom backward was not pursued.
  - **0086C:** cuDNN SDPA integration was a **no-go** in the audited environment; version strings alone did not establish required support.
  - **0087A:** profiling concluded ordinary Python launch starvation was not a material bottleneck for measured F+B; a mechanical C++ rewrite was not justified.
  - **0087B:** compiled FlexAttention established a useful fused-backward reference and Amdahl bound, but did not justify replacing the whole model/framework.
  - **0088A attention:** accepted opt-in Triton mixed-attention F+B after reported parity/performance; grouped remained the code default at that point and no six-fold full-model claim was made.
  - **0088A server / 0088B / 0088C:** accepted experimental native OpenAI-compatible serving, historical-read inference support and per-model sampling. Optimized incremental historical K/V updates, multi-user scheduling and routed-prefix support were not implemented.
- **Compatibility:** Fusion preserves frozen teachers and requires matching data/tokenizer interpretation. External loaders support specific Llama-family GGUF/Safetensors contracts, not arbitrary formats. Server APIs preserve common OpenAI fields but tool-call format was experimental. Attention alternatives remained selectable/fallback-safe rather than silently replacing semantics.
- **Validation:** Historical reports include CPU and RTX measurements; no tests were rerun here.
- **Remaining work:** Keep rejected 0086 candidates rejected; benchmark supported external models with exact hashes; verify serving concurrency/state ownership; use full-model parity and optimizer-step evidence before changing attention defaults.
- **Evidence:** `mini_llm/collective/online_training.py`, `mini_llm/collective/probability_fusion.py`, `mini_llm/external/`, `mini_llm/ops/triton_mixed_attention.py`, `mini_llm/serving_api.py`, `serve.py`, `tests/test_0084b_core_online_fusion.py`, `tests/test_0085h_gguf_wikitext.py`, `tests/test_0086a_attention_oracle.py`, `tests/test_0086b_fused_local_forward.py`, `tests/test_0086c_cudnn_sdpa.py`, `tests/test_0087a_profile_ranges.py`, `tests/test_0087b_attention_contract.py`, `tests/test_0088a_custom_attention.py`, `tests/test_0088a_openai_server.py`, `tests/test_0088c_serving_sampling.py`.

## 2026-10-13 to 2026-10-19 — 0080 persistent soft-plan/block-proposal research

- **Author:** Barbeau, Lucka and Askael
- **Commits:** `e97fafe`, `2e07d49`, `122d707`, `ac9d529`, `056077e`, `1e5ffe3`, `d359a29`, `a27487b`
- **Status:** historical research line; mixed implemented references, experimental paths, inconclusive audits and open production gates
- **Subsystem:** block proposals, soft-vocabulary interfaces, prefix decoding, MoE routing and checkpoints
- **Behavior and decisions:**
  - Implemented K-block proposal/prefix-decoder research with strict causal alignment, handwritten gradients, checkpoint/optimizer state, production data integration and read-only benchmark tools.
  - Production live parity found a **negative horizon-1 validation result** and weaker later gains; this is preserved as evidence against assuming proposal benefit.
  - The decoder checkpoint discontinuity investigation was **inconclusive**; reconstruction could neither prove nor rule out a save/load discontinuity. A before-save/after-load same-process test remained the definitive next check.
  - Checkpoint round-trip work accepted authoritative packed-QKV state and warned not to restore initialization-era packed matrices.
  - 0080AE–AJ3 built bounded CPU/full-vocabulary references, streamed/all-source reads and exact source-mixture factorization. The production CLI and cached generation intentionally rejected unsupported variants until later gates; several reports explicitly made no RTX/full-training speed claim.
  - 0080AJ2 batching and 0080AJ3 CUDA fusion remained experimental; correctness did not imply faster performance.
  - 0080AM learned probability embedding introduced versioning/initialization implications and remained a research option.
  - 0080AO/AP changed deep/top-1 MoE routing. Unified routing was not bitwise or optimization-equivalent to prior top-1 behavior, and strict old config readers could reject new fields. Do not overwrite old checkpoints merely to adopt it.
  - 0080AQ diagnostics were observational and did not alter training objective.
  - Full 64K routed main-model cached generation and broad production acceptance remained explicitly open in the implementation summary.
- **Compatibility:** Architecture/checkpoint variants require explicit version gates. CPU tiny-vocabulary references are not safe substitutes for 65,280-vocabulary BF16/CuPy training. Read-only production benchmarks require consistent frozen checkpoint copies and exact dataset/sample fingerprints. This research line is not protected Model B.
- **Validation:** Historical reports contain their own recorded CPU/CuPy results and known missing-test-module caveats. No tests were rerun here.
- **Remaining work:** Treat as historical unless a new experiment is explicitly approved. Preserve negative/inconclusive evidence and compare against current Model B using matched data/token/compute definitions.
- **Evidence:** `mini_llm/predictive_refinement/`, `mini_llm/ops/hierarchical_memory.py`, block-proposal tests matching `tests/test_block_proposal_0080*.py`, `tests/test_0080ao_deep_top1_moe.py`, `tests/test_0080ap_unified_expert_routing.py`, `tests/test_0080aq_expert_usage_logging.py`, commits listed above.

## 2026-10-02 to 2026-10-05 — 0058 hierarchical routed memory

- **Author:** Askael
- **Commits:** implementation history around `5da3846`, `75f66b4`, `7512e6d`, `7b2fc48`, `730c9d2`, `e6332d9`, `6d5f1a8`, `9d97a1b`; current report path recorded by `a27487b`
- **Status:** accepted historical architecture; superseded as current baseline
- **Subsystem:** long-context routing and target-only training
- **Behavior:** Added hierarchical same-document memory routing: shallow block selection over long history, exact selected-token reopening with chronological/original positions, fixed deep active length, target-only LM projection/loss, and router diagnostics. Mean pooling was selected for the production memory lifecycle; learned pooling retained more memory.
- **Compatibility:** The 16K/32K/64K presets were shape-compatible with each other, but `--context-length` was deliberately rejected when hierarchical memory was enabled to avoid confusing source horizon with deep context. Compatible with optimizer offload, chunked head and block checkpointing. It is not Model B's matrix-pooled/stride-16 architecture.
- **Validation:** Historical validator covered reference and optional CuPy/BF16 paths. It was not rerun here.
- **Remaining work:** Preserve as research history; use current Model B tests for present development rather than assuming routed-prefix behavior.
- **Evidence:** `mini_llm/ops/hierarchical_memory.py`, `mini_llm/model/decoder_lm.py`, `mini_llm/data/token_shards.py`, `validate_0058_hierarchical_memory.py`, Git commits above.

## 2026-09-29 — Legacy extended-training foundation

- **Author:** Askael
- **Commit:** `6119842` (later reorganized by `a27487b`/`4baae24`)
- **Status:** accepted historical foundation; operational details partly superseded by current guides
- **Subsystem:** tokenizer, data shards, trainer and checkpoints
- **Behavior:** Recorded BPE merge-rank correctness, separate tokenizer training, deterministic multiprocessing shard generation, streaming corpus handling, accumulation, validation and checkpoint basics.
- **Compatibility:** The old model sizes, commands and VRAM estimates are historical and must not override current Model B or current CLI documentation.
- **Validation:** Legacy report states deterministic worker-count checks; not rerun here.
- **Remaining work:** Keep current operational guidance in `USER_GUIDE.md` and `DATA_PIPELINE.md`; use this entry only for provenance.
- **Evidence:** `mini_llm/tokenizer/`, `mini_llm/data/`, `mini_llm/train_extended.py`, `tests/test_tokenizer.py`, commit `6119842`.

---

## Migration ledger for old Markdown reports

This ledger is intentionally filename-oriented. Entries above are authoritative summaries; old report contents remain recoverable with `git show <listed-commit>:<path>` after deletion.

| Old report file or grouped range | Consolidated entry | Git provenance / durable evidence |
|---|---|---|
| `docs/IMPLEMENTATION_SUMMARY.md` (originally root `IMPLEMENTATION_SUMMARY.md`) | Legacy extended-training foundation | Recover as `git show 6119842:IMPLEMENTATION_SUMMARY.md` or `git show a27487b:docs/IMPLEMENTATION_SUMMARY.md`; current code under `mini_llm/tokenizer/`, `mini_llm/data/`, `mini_llm/train_extended.py` |
| `docs/research/HIERARCHICAL_MEMORY_0058.md` | 0058 hierarchical routed memory | current path at `a27487b`; implementation history listed in entry |
| `docs/research/README_persistent_soft_plan_0080.md`, `docs/research/IMPLEMENTATION_0080.md` | 0080 | `1e5ffe3`, `a27487b`; `mini_llm/predictive_refinement/` and `tests/test_block_proposal_0080*.py` |
| `docs/research/PRODUCTION_LIVE_PARITY_AUDIT_0080AA.md`, `docs/research/DECODER_CHECKPOINT_DISCONTINUITY_0080AC.md`, `docs/research/PRODUCTION_CHECKPOINT_ROUNDTRIP_0080AD.md` | 0080 negative/inconclusive/checkpoint evidence | `a27487b`; matching `tests/test_block_proposal_0080aa_live_audit.py`, `...0080ac_checkpoint_discontinuity.py`, `...0080ad_roundtrip.py` |
| `docs/research/0080AE_full_soft_interface_audit.md`, `docs/research/0080AF_cpu_full_vocab_reference.md`, `docs/research/0080AG_teacher_forced_soft_vocab_integration.md`, `docs/research/0080AH_AI_combined_own_all_streamed_read.md`, `docs/research/0080AJ_exact_source_mixture_optimization.md`, `docs/research/0080AJ2_batched_full_vocabulary_gemm.md`, `docs/research/0080AJ3_fused_vocab_cuda.md` | 0080 full-vocabulary research | `a27487b`; matching `tests/test_block_proposal_0080a*.py` |
| `docs/research/0080AM_learned_probability_embedding.md`, `0080AO_32_layer_two_expert_top1.md`, `0080AP_unified_moe_expert_routing.md`, `0080AQ_expert_utilization_diagnostics.md` | 0080 later architecture/routing work | `a27487b`; matching `tests/test_block_proposal_0080am*.py`, `tests/test_0080ao*.py`, `tests/test_0080ap*.py`, `tests/test_0080aq*.py` |
| `docs/research/PRODUCTION_BLOCK_PROPOSAL_BENCHMARK.md` | 0080 read-only benchmark | `a27487b`; `benchmarks/evaluate_block_proposal_production.py` and its tests |
| `docs/0084B_CORE_ONLINE_FUSION.md` | 0084–0088 | `801167d`; `tests/test_0084b_core_online_fusion.py` |
| `docs/0085H_EXTERNAL_GGUF_WIKITEXT.md` | 0084–0088 | `68ba347`; `tests/test_0085h_gguf_wikitext.py` |
| `docs/0086A_ATTENTION_BASELINE_AUDIT.md`, `0086B_FUSED_LOCAL_FORWARD.md`, `0086C_CUDNN_SDPA_FWD_BWD_BENCHMARK.md` | 0084–0088 | `68ba347`, `69c5826`; matching `tests/test_0086*.py` |
| `docs/0087A_TRAINING_TIMELINE.md`, `0087B_COMPILED_TORCH_ATTENTION.md` | 0084–0088 | `5516c61`; matching `tests/test_0087*.py` |
| `docs/0088A_CUSTOM_TRITON_MIXED_ATTENTION.md`, `0088A_NATIVE_OPENAI_SERVER.md`, `0088B_HISTORICAL_READ_NATIVE_INFERENCE.md`, `0088C_PER_MODEL_SAMPLING.md` | 0084–0088 | `1a54a06`, `647c280`; matching `tests/test_0088*.py` |
| `docs/0089A_OPTIMIZED_PRODUCTION_REPROFILE.md`, `docs/0089B_COMPILED_TORCH_BLOCK.md`, `docs/0089C_TRAINING_RESIDUAL_RMSNORM.md`, `docs/0089D_COMPILED_TORCH_LM_HEAD.md`, `docs/0089E_COMPILED_TORCH_BLOCK_STACK.md` | 0089 | `96ed2c3`; matching `tests/test_0089a*`–`test_0089e*` |
| `docs/0089J_OPTIMIZATION_VALIDATION.md`, `0089K_COMPARATIVE_GPU_BOTTLENECK_AUDIT.md`, `0089L_VECTORIZED_BF16_SWIGLU.md`, `0089M_FAIR_FLAGS_ROPE_AND_GRAPH_AUDIT.md`, `0089N_FUSED_QKV_ROPE_AND_MIGRATION_GATE.md` | 0089 | `4488335`; profiler/benchmark code and matching tests |
| `docs/0089O_HISTORICAL_BOTTLENECK_AUDIT.md`, `0089O_HISTORICAL_OPTIMIZATION_PLAN.md`, `0089P_STREAMING_HISTORICAL_BLOCK_FORWARD_NEGATIVE.md`, `0089R_TWO_PASS_HISTORICAL_BLOCK_FORWARD_NEGATIVE.md`, `0089S_HISTORICAL_BLOCK_TRITON_PREWARM.md`, `0089T_HISTORICAL_BLOCK_ACCEPTANCE_AUDIT.md` | 0089 historical read, including rejected forward results | `4488335`; `mini_llm/ops/triton_historical_block.py`, profiler/analyzer tests |
| `docs/0089U_STREAMING_HISTORICAL_BLOCK_BACKWARD_PROTOTYPE.md`, `0089V_STREAMING_HISTORICAL_BLOCK_BACKWARD.md` | 0089 superseded prototype and retained opt-in backward | `33fceba` |
| `docs/FUSION_OPT_01_COMPREHENSIVE_GPU_OPTIMIZATION.md` | 0089 unnumbered fusion optimization | `4488335`; fusion code/tests in that commit |
| `docs/0090A_RESIDUAL_SPECIALIST_PHASE0_PHASE2.md` | 0090 | `41c8fa2`; superseded by current `mini_llm/specialists/` work |
| `benchmarks/README_0092_LIVE_BENCHMARKS.md` | 0092 | `58e8799`; current benchmark code/configs/tests |
| `FULL_HISTORY_HEADS_0100.md` | 0100 | `f98bbe5`; full-history code/tests listed in entry |
| `docs/DUAL_RESOLUTION_0102.md`, `docs/PATCH_0102B_TRITON_LOCAL.md`, `docs/PATCH_0102C_TRITON_GLOBAL_AND_COMPRESSION.md`, `docs/PATCH_0102D_FULL_MODEL_TRAINING_AND_MEMORY.md`, `docs/PATCH_0102E_DOCUMENT_FIRST_FULL_SEQUENCE.md`, `docs/PATCH_0102F_FAST_DOCUMENT_EXECUTION.md`, `docs/PATCH_0102G_VARIABLE_BATCH_SUMMARY_ROPE.md`, `docs/PATCH_0102H_PACKED_BF16_PROJECTIONS.md`, `docs/PATCH_0102I_STEADY_STATE_ABLATION.md`, `docs/PATCH_0102J_DEFAULT_PACKED_AND_FUSED_POOLING.md`, `docs/PATCH_0102K_POOLING_TRITON_POINTER_FIX.md`, `docs/PATCH_0102L_COMPRESSOR_RMSNORM_FUSION.md`, `docs/PATCH_0102M_FUSED_LOCAL_GLOBAL_GATE.md`, `docs/PATCH_0102N_ATTENTION_BACKWARD_PROFILE.md` | 0102 base–N | `9dbd60c`; dual-resolution code and matching `tests/*0102*.py` |
| `docs/PATCH_0102O_MATRIX_POOL_MODEL_B.md` | 0102 Model B | `9dbd60c`; `mini_llm/ops/matrix_pool_dual_resolution_attention.py`, `tests/test_matrix_pool_dual_resolution_0102o.py` |
| `docs/PATCH_0102P_MODEL_B_SCATTER_AND_ABLATION.md`, `PATCH_0102Q_TO_0102T_MODEL_B_OPTIMIZATIONS.md`, `PATCH_0102U_FUSED_MATRIX_POOL_BACKWARD.md`, `PATCH_0102V_PACKED_PROJECTION_BACKWARD_CHUNKS.md`, `PATCH_0102W_GLOBAL_SKIP_EMPTY_SUMMARY_TILES.md` | 0102 P–W | `5dec566`; matching matrix-pool/packed/global tests |
| `docs/PATCH_0102X_DUAL_LENGTH_AWARE_ACCUMULATION.md`, `PATCH_0102Y_MODEL_B_PREPARED_LOCAL_ROPE.md` | 0102 X–Y | `a6cb830`; matching tests |
| `docs/MODEL_B_STRIDE16_SPARSE_0102Z.md` | 0102Z protected Model B | `52b8ae3`; `tests/test_sparse_stride_history_0102z.py` |
| `docs/0103_MODEL_CENTRIC_FOUNDATION.md`, `0103A_BASELINE_REGRESSION.md`, `0103A2_GPU_MEMORY_AWARE_PARITY.md`, `0103C_UNIFIED_MODEL_LOADING.md`, `0103D_AUTOMATIC_INFERENCE_EXECUTION.md`, `0103E_UNIFIED_GENERATION_SESSIONS.md`, `0103F_UNIFIED_SERVING_AND_REMOTE_MODELS.md`, `0103G_UNIFIED_EVALUATION.md` | 0103 | `a15f664`; model/loading/generation/serving/evaluation code and tests |
| `docs/PATCH_0103K_ADAPTIVE_FINE_BUCKETS.md` | 0103K | `fd20236`; `tests/test_dual_resolution_fine_shape_buckets_0103k.py` |
| `docs/0105_MODEL_CENTERED_SPECIALIST_GATES.md` | 0105 | `8586f79`; 0105 specialist/profile tests |
| `docs/0106A_SMOLLM2_RETRIEVAL_CONTROL.md` | 0106 | `68844dd`; external retrieval runner/tests |
| No standalone 0107/0108 Markdown report existed | 0107–0108 | `eb61df0`; code/tests listed in entry. Future work must be logged here rather than creating one. |

## Research artifact cross-reference

The retained LaTeX/PDF research artifacts are indexed in `docs/research/README.md`. They are historical design/evidence documents and do not replace this change log or the protected current baseline.