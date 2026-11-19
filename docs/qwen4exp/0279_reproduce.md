# Reproduce the best measured longer workload

Run from the repository with no other model server occupying RAM/VRAM. This is a benchmark, not an HTTP server or interactive chat. Use a **new exclusive output directory** every time; the runner refuses to overwrite a capture.

```sh
MINI_LLM_NATIVE_MOE_HANDOFF=1 \
MINI_LLM_NATIVE_MOE_GRAPH=1 \
MINI_LLM_DEVICE_PLAN_SPECIALIZED=0 \
MINI_LLM_QSA_FRONTEND_GRAPH=0 \
MINI_LLM_GRAPH_READONLY_INPUTS=0 \
MINI_LLM_GGML_POOL_SCHEDULER=tasks \
MINI_LLM_GGML_POOL_HOST_HELP=1 \
MINI_LLM_GGML_POOL_CPUS= MINI_LLM_HOST_CPU= \
MINI_LLM_GGML_POOL_SPIN_US=500 \
MINI_LLM_GGML_Q2_SIMD=1 \
MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/probe_0254_strata_workload.py \
  --output docs/qwen4exp/diagnostics_0279_reproduce_v1 \
  --host-warm-gib 50 --mem-available-floor-gib 4 \
  --victim-policy heap --cpu-backend ggml_pool --cpu-workers 23 \
  --promotion-limit 0 --overlap-shared --pre-moe-graph \
  --device-expert-plan --shared-graph-decode --defer-shared
```

The retained single run `diagnostics_0277_workers23_v1` measured **412 decode forwards / 15.286669 seconds = 26.95159/s**, 413 tokens including the first prefill-produced token, 18.904584 seconds including prefill. This is an observation, not a promised lower bound or 50/s result. All later experiments preserve these output IDs but do not outperform that run.

`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` only prevent unrelated BLAS/OpenMP oversubscription: the native pool still owns23workers and its waiting callback thread participates. No manual affinity is used because copying Strata affinity regressed here.

The benchmark retains the original checkpoint, complete top10 execution, FP32 KV/context1024, CPU GGMLQ8_K/Q8_0 and GPU FP32 activations. It does not silently enable Strata suffix drafts, route-tail omission or INT8 KV. Warming touches canonical mmap pages, not a complete private/pinned bank. The parent/child time/memory guards remain enabled.

For a controlled timing investigation, retain these settings and capture node-level CUDA graph detail; default Nsight graph tracing hides kernel nodes inside captured segments. Correlate CPU native spans, graph launches and blocking copy API calls rather than adding overlapping totals or interpreting CUDA API time as DMA/kernel time.
