# Reproduce the current Q8/BF16 sparse decode measurements

Run from the repository with no other large model server using RAM/VRAM. Preserve the checkpoint and use a **new output directory**. This launches the existing bounded benchmark, not an HTTP server.

## Recommended measured configuration

```sh
MINI_LLM_EXPERT_GPU_Q8=1 \
MINI_LLM_LINEAR_GPU_Q8=1 \
MINI_LLM_HC_BF16_WEIGHTS=1 \
MINI_LLM_NATIVE_MOE_HANDOFF=1 \
MINI_LLM_NATIVE_MOE_GRAPH=1 \
MINI_LLM_NATIVE_MOE_SERVICE=0 \
MINI_LLM_SIZED_EXPERT_SLOTS=0 \
MINI_LLM_DEVICE_PLAN_SPECIALIZED=0 \
MINI_LLM_QSA_FRONTEND_GRAPH=0 \
MINI_LLM_GRAPH_READONLY_INPUTS=0 \
MINI_LLM_QSA_DEFER_VALIDATION=0 \
MINI_LLM_GGML_POOL_FUSED_HIDDEN=0 \
MINI_LLM_GGML_POOL_SCHEDULER=tasks \
MINI_LLM_GGML_POOL_HOST_HELP=1 \
MINI_LLM_GGML_POOL_CPUS= MINI_LLM_HOST_CPU= \
MINI_LLM_GGML_POOL_SPIN_US=50 \
MINI_LLM_GGML_Q2_SIMD=1 \
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/probe_0254_strata_workload.py \
  --output docs/qwen4exp/diagnostics_0288_reproduce_v1 \
  --expert-cache-gib 7 --host-warm-gib 50 --mem-available-floor-gib 4 \
  --victim-policy heap --cpu-backend ggml_pool --cpu-workers 19 \
  --promotion-limit 0 --overlap-shared --pre-moe-graph \
  --device-expert-plan --shared-graph-decode --defer-shared
```

The existing `diagnostics_0284_bf16_hc_cache7_v1` measured **31.42545 decode forwards/s**,359 output IDs/358 post-prefill forwards. The optional `MINI_LLM_NATIVE_MOE_SERVICE=1` final safe implementation measured29.55954/s with identical IDs and remains opt-in. Its earlier31.71163/s iteration had a timeout-copy edge case subsequently corrected; do not deploy the historical source to chase that observation. Final `diagnostics_0288_reproduce_v1` measured29.29410/s with the same359 IDs, demonstrating run-to-run variability. Do not promise the best observation on every run or claim50/s. An8GiB attempt was safely rejected by physical headroom checks; keep7GiB rather than bypassing the planner.

No native CPU arithmetic, top10 completeness, request state or checkpoint precision is omitted. GPU activations are quantized with the tested native Q8_1 producer; CPU uses per-format GGML Q8_K/Q8_0. HC stores original BF16 values and computes in FP32.19 native row workers plus a participating waiter are independent of BLAS/OpenMP thread limits; the optional service adds one native service thread. No manual affinity is used.

## Controlled longer trajectory

For an identical retained warmup/response token sequence, append:

```sh
--force-tokens-from docs/qwen4exp/diagnostics_0284_bf16_hc_cache7_v1/result.json
```

Use a new output directory. Full logits/output head/sampler still execute. Results explicitly report `forced_trajectory=true` and the source path; this is a controlled replay, **not natural generation throughput**. To test sized fixed slots, change only `MINI_LLM_SIZED_EXPERT_SLOTS=1` and the output directory. The retained pair measured30.19602/s uniform versus31.14912/s sized, not50/s.

Do not enable cumulative warmup-frequency placement, hidden-phase fusion or extra graph flags merely because they exist; retained longer measurements do not show a useful speed gain. Direct mapped-host expert kernels are standalone diagnostics, not an enabled model mode. Staged CPU/PCIe splitting is not integrated.

## Safety and comparison limits

The runner uses128 warmup tokens, up to512 response tokens, exact captured Strata prompts/template/sampling settings, CPU/RAM/time guards and explicit GPU budgets. It frees unused CuPy pool blocks only at synchronized untimed request seams; it does not bypass memory planning or free live model/cache arrays.

This is context1024/FP32KV, frozen7GiB GPU expert placement after warmup and reclaimable mmap warming. It does not match Strata's context262144/INT8KV, suffix drafting, eligible tail-miss omission or pinned complement. Report the timing convention and token IDs rather than an unqualified emitted-token comparison.

Tests/artifacts/source hashes and per-patch contracts: `0281_0288_execution_optimization.md`, `0281_0288_performance_v3.json`. Latest selected inference suites:1969NumPy passed/1124skipped;1367actualCuPy passed/14skipped, including Qwen3-Next GPU components. No complete training-suite claim.
