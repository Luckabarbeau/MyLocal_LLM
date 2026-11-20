# Reproduce 0307 safely

Run from `MyLocal_LLM` with no other large model process. Every output directory must be new. The wrapper refuses to run outside the verified 0290 cgroup service. It uses the existing checkpoint and pinned CPU library; neither is copied or modified.

## Controlled long baseline

```sh
.venv/bin/python scripts/run_0290_isolated_benchmark.py \
  --memory-high-mib 32768 --memory-max-mib 36864 --timeout 200 \
  --output docs/qwen4exp/my_0307_control_guard_v1 -- \
  .venv/bin/python scripts/probe_0302_decode_configuration.py \
    --output docs/qwen4exp/my_0307_control_v1
```

This explicitly selects 19 native workers plus the participating native waiter, task factor3, Q8 GPU dense/expert arithmetic, BF16 HC storage with FP32 arithmetic, request-bound recurrence, GPU graph copies, native service, input-only graph copies, deferred checked QSA status, and 7.25GiB sized expert slots. BLAS/OpenMP remain1 independently of native worker count. No manual affinity, staged misses, bulk file warming or anonymous expert-bank copy. A new configuration record is saved with each result.

Controlled128warmup/359response IDs come from the retained0284result. Full head/sampler execute, but this is **forced replay**, not natural conversation speed. Latest adjacent baseline/shared-grid pair40.06/40.17forwards/s; earlier single42.64/s is not guaranteed. **50/s not met.**

## Isolate shared-grid candidate

Repeat the command with new output names and add `--shared-grid` to the configuration wrapper. Leave all other flags unchanged. Real packed GPU operators show1.085–1.857x, but no robust whole-model gain is established. The flag is not default-enabled.

## Natural generation

Use new output names and add `--natural` (and optionally `--shared-grid`). This removes forced tokens. Actual retained runs emit371tokens,370decode forwards,31.96/35.16forwards/s; both have identical IDs but substantial disk reads/major faults. Never quote the hotter forced rate as this rate.

## Optional diagnostic variants

- `--workers 15|19|23`: tested15/23did not beat the earlier19-worker control.
- `--task-factor 1..16`: default3; factor8lost in the model despite small operator gains.
- `--qsa-frontend`: earlier single42.64/s; no robust incremental gain claim.
- `--fused-shared`: captures shared expert/combination inside the native graph.
- `--kernel-publish`: GPU stores exact activation/metadata into existing pinned slabs and publishes readiness; requires native service, configured by this wrapper.
- `--telemetry`: bounded timestamped `nvidia-smi` snapshots, recorded separately. Avoid comparing a telemetry run with a nontelemetry control for small gains.
- `--trace --expert-cache-gib 7`:16-token node-level Nsight diagnostic. A7.25GiBprofiler attempt was rejected by VRAM headroom. Do not weaken admission; use7GiBfor the trace and do not compare its placement/rate directly to7.25GiBunprofiled results. Nsight collection stops but allows the application to finish. Inspect captures as diagnostics, not uninstrumented rate results.

All variants remain experimental/off unless requested. The wrapper does not launch a conversation server. No background worker is intentionally left behind after completion.

Detailed contracts/results/numerical and memory scopes: `0295_0307_critical_path_optimization.md`, `0295_0307_optimization_results_v2.json`. Current numerical verification includes471CPU/201GPUpassing tests and exact two-request logits/state hashes in the CPU-expert full-model fixture; resident GPU arithmetic has separate exact operator checks.
