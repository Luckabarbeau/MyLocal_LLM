# PATCH 0086A — Attention baseline audit and benchmark harness

## Patch metadata

- **PATCH ID:** `0086A`
- **Source tree SHA:** `39d91e963562376f9d1ed5530f634708ffb539e4`
- **Prior attention patches:** `PATCH_0085I_ATTENTION_MEMORY_MOVES_CPU_AUDIT.patch` and `PATCH_0085J_GROUPED_BF16_RECOMPUTE_STRIDE_FIX.patch` are present as untracked patch files but are not applied. Both pass `git apply --check`; neither passes reverse-check.
- **Production files changed:** none
- **New files:** NumPy oracle, oracle tests, standalone benchmark, this audit, and generated JSON/CSV baseline reports
- **Default behavior:** unchanged

Apply from a clean copy of the recorded source tree:

```bash
git apply --check PATCH_0086A_ATTENTION_BASELINE_ORACLE_BENCHMARK.patch
git apply PATCH_0086A_ATTENTION_BASELINE_ORACLE_BENCHMARK.patch
```

The worktree was already dirty before 0086A. Existing tracked edits in
`mini_llm/diagnostics/gradient_noise.py`, `mini_llm/train_extended.py`, and
`train_model.py`, plus pre-existing untracked patch/diagnostic files, were not
modified or reset.

## Verified model and call path

The active preset resolves through
`train_model.model_config_from_name("ensemble-1024-20l-dense-historical-hybrid-64k")`
to `ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()` with:

- 20 layers, `d_model=1024`, `d_ff=4096`
- 16 Q heads, 4 native KV heads, `D=64`
- head 0: causal current-context window 4096
- heads 1–15: causal sliding window 512 unless valid history replaces a scheduled head

The verified path is:

```text
TransformerBlock.forward
  -> GQAAttention.forward
     -> packed QKV projection and RoPE
     -> GQAAttention._mixed_context_forward
        -> local_window_attention_forward
           -> _local_grouped_forward when BF16 pipeline/layout/cuBLAS gates pass
     -> HistoricalReadMemory.apply
     -> Wo projection

TransformerBlock.backward
  -> GQAAttention.backward
     -> HistoricalReadMemory.backward_reads
     -> GQAAttention._mixed_context_backward
        -> local_window_attention_backward
           -> _local_grouped_backward_recomputed when recompute is enabled
     -> inverse RoPE / packed projection backward
```

`historical_skip_heads` is populated only when
`historical_state.all_rows_have_valid` is true. Head 0 remains current-only.
The 20-layer schedule in the handoff matches source, including layer 0 heads
12/13, layer 4 heads 10–15, layer 6 heads 10/11/14/15, and no slots at layer 8.

## Exact local-attention semantics and mapping

For query `i`, valid keys are inclusive:

```text
max(0, i - W + 1) <= j <= i
```

The local subset must retain original head identities. For heads 1–15 the
verified map is:

```text
original heads: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
native KV:      0,0,0,1,1,1,1,2,2, 2, 2, 3, 3, 3, 3
```

For the irregular test subset `(1,2,3,8,12,15)`, the required map is
`(0,0,0,2,3,3)`, not a mapping based on subset-relative indices.

## Current grouped BF16 buffers and precision

With the supplied training flags plus `gpu-fast`, the existing grouped path is
the runtime baseline:

- Q/K/V are BF16 post-RoPE inputs. The grouped kernels consume their real byte
  strides; positive noncompact B/T/H strides are accepted when D is contiguous.
- Forward allocates BF16 `scores` and BF16 `probs` for every query chunk,
  dispatches grouped QK GEMMs, then fused FP32 max/sum/exp softmax arithmetic
  with BF16 probability storage.
- Scores are deleted after softmax. Context GEMM temporaries are BF16.
- With `MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1`, forward probabilities are
  deleted before returning. The cache retains Q/K/V, window, scale, mapping and
  chunk geometry, not full probability matrices.
- Under the supplied environment, `MINI_LLM_BF16_MIXED_CONTEXT` is unset, so
  the public local-attention output is FP32 even though grouped GEMM
  temporaries are BF16. Backward returns FP32 dQ/dK/dV.
- Recomputed backward first packs `dcontext` to BF16, allocates FP32 dQ/dK/dV,
  and processes one chunk at a time. Per chunk it reconstructs BF16 scores and
  probabilities, computes BF16 dProbs/dScores around an FP32 softmax Jacobian,
  forms BF16 dQ/dK/dV temporaries per Q head, then accumulates/scatters into
  native FP32 gradients. Shared-KV dK/dV contributions are summed by the
  existing gradient-store kernel.
- NumPy/non-grouped execution uses the public fallback implementation and the
  same mask and explicit mapping contract.

No host synchronization was added to production code. Synchronization exists
only in the standalone benchmark around controlled CUDA event and memory
measurements.

## 0085I isolation result

The current source uses tuple indexing for Q and dcontext subsets, producing a
copy for the benchmarked CuPy groups. The 0085I-style `basic-slice` run correctly
produces a zero-copy Q view, but the full forward+backward benchmark cannot
complete on the current source:

```text
ValueError: all GEMMs in a grouped descriptor must be uniform
```

The failure occurs in `_local_grouped_reconstruct_chunk`. A zero-copy 15-head Q
slice retains the original 16-head token stride (`2048` bytes for BF16 D64),
while `_local_grouped_backward_recomputed` packs dcontext to a compact 15-head
stride (`1920` bytes). Score and dProbs GEMMs are currently interleaved in one
`RowMajorGemmGroup`, but `grouped_bf16_gemm` requires uniform
`(m,n,k,lda,ldb,ldc,trans_a,trans_b)` within each group.

The unapplied 0085J patch explicitly separates score and dProbs GEMMs into two
uniform groups when Q/K and dcontext/V row strides differ. Its documented
invariant exactly matches this observed failure. Therefore **no 0085I speedup
is claimed**, and neither patch was applied during 0086A.

## Installed GPU and cuDNN feasibility

Verified locally:

- NVIDIA GeForce RTX 5080, compute capability 12.0, 16,303 MiB
- CuPy 14.2.0
- CUDA runtime 13.2, NVIDIA driver API 13.0, driver 580.178.04
- no importable `cudnn` Python frontend
- no discoverable `libcudnn` via `ldconfig` or `ctypes.util.find_library`
- no cuDNN files under the installed CuPy package

Consequently, an installed cuDNN SDPA forward/backward smoke cannot be run in
this environment. Version strings do not establish support; the required
frontend/library is absent. A future custom-CUDA fallback can remain
self-contained through CuPy RawModule/native CUDA and should use separate
D64/W512 local and Hq=1 full-causal kernels, explicit element strides and
original-head `kv_map`, FP32 LSE, query-parallel dQ, and key/KV-head-parallel
dK/dV accumulation. That is an 0086B/0086C activity, not part of this patch.

## Correctness results

Commands:

```bash
MINI_LLM_BACKEND=numpy python benchmarks/reference_fused_attention_oracle.py

MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_0086a_attention_oracle.py \
  tests/test_0083c_hybrid_current_attention.py \
  tests/test_local_window_attention.py \
  tests/test_mixed_attention.py \
  tests/test_bf16_attention_probability_cache.py \
  tests/test_attention.py
```

Results:

- standalone FP64 oracle self-test: PASS
- focused 0086A tests: 5 passed
- combined requested attention suite plus 0086A: **46 passed in 2.97 s**

The tests cover Q/K/V central differences at epsilon `1e-4`, `1e-5`, and
`1e-6`; W=1 gradients; future and left-window exclusion; exact inclusive left
boundary; inactive heads; FP32/FP64 public-path parity; and irregular
original-head mapping with shared native KV accumulation.

RTX 5080 smoke (`B=1,T=33,Hq=1/9/13/15,D=64`, BF16) selected grouped cuBLAS,
the fused BF16 pipeline, and recomputation for every case. Against the FP64
oracle, worst values across cases were:

| Tensor | max abs | RMSE | relative L2 | minimum cosine |
|---|---:|---:|---:|---:|
| output | 1.4945e-3 | 1.4962e-4 | 2.1261e-3 | 0.99999774 |
| dQ | 8.9756e-5 | 7.5440e-6 | 3.4442e-3 | 0.99999407 |
| dK | 1.0145e-4 | 1.4127e-5 | 3.5832e-3 | 0.99999359 |
| dV | 3.7156e-3 | 2.9947e-4 | 2.2882e-3 | 0.99999739 |

All compared arrays were finite. These are arithmetic baseline observations,
not acceptance thresholds for a future backend.

## Existing GPU baseline: B=3, T=2304, D=64

Complete public forward+backward, BF16 grouped cuBLAS, probability
recomputation, automatic query chunks, three warmups and ten measured repeats:

| Case | Hq | Auto chunk | Device median | p10 | p90 | retained live-byte delta |
|---|---:|---:|---:|---:|---:|---:|
| full head 0, W=2304 | 1 | 1024 | 0.914 ms | 0.883 ms | 1.027 ms | 17,695,232 |
| all local heads, W=512 | 15 | 256 | 5.094 ms | 4.942 ms | 5.828 ms | 67,240,448 |
| layer 0 history-valid local subset | 13 | 256 | 4.668 ms | 4.601 ms | 5.562 ms | 60,162,560 |
| layer 4 history-valid local subset | 9 | 384 | 3.049 ms | 3.002 ms | 4.499 ms | 46,006,784 |

For a no-history layer, separately dispatched full plus 15-head local attention
sum to about 6.007 ms median in this isolated harness. This is not a claim
about layer or optimizer-step time. Pool reserved bytes reached 767,843,840
after sequential cases; reserved storage is allocator cache and not equivalent
to live or peak attention memory. The report records before/retained-live
allocated and reserved bytes; transient kernel peak is not inferred from these
values.

Generated reports:

- `benchmarks/results/0086a_cpu_baseline.json` / `.csv`
- `benchmarks/results/0086a_gpu_smoke.json` / `.csv`
- `benchmarks/results/0086a_gpu_b3_t2304_current.json` / `.csv`

No matched optimizer-step run was performed and no supervised-token throughput
claim is made. Checkpoints, logs, sampler state and the active training run were
not touched.

## Reproduction

Current baseline:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM=1 \
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT=0 \
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1 \
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1 \
MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1 \
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=auto \
python benchmarks/benchmark_0086a_attention.py \
  --runtime-profile gpu-fast --batch 3 --seq 2304 --d-head 64 \
  --dtype bfloat16 --head-selection current --warmup 3 --repeats 10 \
  --oracle-max-seq 0 \
  --json-out benchmarks/results/0086a_gpu_b3_t2304_current.json \
  --csv-out benchmarks/results/0086a_gpu_b3_t2304_current.csv
```

0085I-style diagnostic (currently expected to expose the 0085J stride issue):

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM=1 \
MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT=0 \
MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD=1 \
MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS=1 \
MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1 \
MINI_LLM_LOCAL_ATTN_QUERY_CHUNK=auto \
python benchmarks/benchmark_0086a_attention.py \
  --runtime-profile gpu-fast --batch 3 --seq 2304 --d-head 64 \
  --dtype bfloat16 --head-selection basic-slice --warmup 3 --repeats 10 \
  --oracle-max-seq 0
```

## Regression risk and fallback

0086A changes no production path. The benchmark imports the existing public
API and synchronizes only in its own process. The old grouped implementation
remains the sole/default backend.

## Single next action

Before starting a fused kernel, apply **0085I and 0085J together in an isolated
worktree**, run their CPU tests and the B=3/T=2304 paired benchmark above, and
accept or reject the zero-copy change on complete forward+backward latency and
numerical parity. 0085I alone is not a valid GPU benchmark state under grouped
probability recomputation because the observed stride-uniformity failure blocks
backward.
