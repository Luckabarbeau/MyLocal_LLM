# 0088A — Opt-in Triton mixed-attention forward/backward

## Scope and decision

0088A implements the validated experimental mixed-attention kernel as an **opt-in production backend**. It replaces only the current-window mixed-attention component after Q/K RoPE and before the attention output projection. It does not change model mathematics, the historical reader, sampling, optimizer, checkpoint format, or the default training path.

The production default remains:

```bash
export MINI_LLM_TRAIN_ATTN_BACKEND=grouped
```

The Triton backend is selected only with:

```bash
export MINI_LLM_TRAIN_ATTN_BACKEND=triton
export MINI_LLM_TRAIN_ATTN_STRICT=1
```

Strict mode is recommended for an intentional trial because it fails rather than silently falling back when the optional dependency or validated geometry is unavailable. Without strict mode, unsupported configurations and an unavailable Triton import use the existing grouped implementation.

No full-model or optimizer-step throughput result is claimed by this patch. The measured result is for complete mixed-attention forward plus numerical backward only.

## Mathematical contract

For each original query head `h`, the native KV head remains:

```text
kv_head = h // 4
```

The topology is:

- original head 0: full causal attention;
- original heads 1–15: inclusive causal local window 512;
- local query position `q` admits keys `max(0, q - 511) <= k <= q`;
- heads substituted by historical attention retain their original identity but contribute exact-zero current output and exact-zero current dQ;
- shared dK/dV accumulate contributions from all active original query heads associated with each native KV head.

The implemented graph is:

```text
S     = (Q K^T) * scale
P     = softmax(S)
O     = P V
delta = sum(O * dO, axis=-1)
dS    = P * (dO V^T - delta)
dQ    = scale * dS K
dK    = scale * dS^T Q
dV    = P^T dO
```

Q/K/V are BF16. Output, dQ, dK, dV, softmax state, and reductions are FP32. The FP32 upstream gradient is loaded and rounded to BF16 for tensor-core dot products, matching the validated BF16 numerical contract rather than promising bitwise equality to a different reduction order.

## Kernel architecture

`mini_llm/ops/triton_mixed_attention.py` accepts CuPy allocations directly through Triton's tensor pointer protocol and has no Torch dependency. It contains:

1. an online-softmax forward traversal;
2. a delta preprocessing kernel for `sum(O * dO)`;
3. a query-parallel dQ traversal;
4. a native-KV-parallel dK/dV traversal with one writer per gradient element and no atomics.

The dK/dV grid includes the runtime native KV head. Only the four query-head offsets belonging to each KV head are statically unrolled. This reduced the earlier four sequential dK/dV launches to one launch and was the material backward optimization.

Contiguous 15-, 13-, and 9-head patterns avoid a device head-index load. Irregular historical substitutions use a compact list of active **original** head numbers; heads are never renumbered. Aligned local key tiles entirely to the left of a query's active window leave online-softmax state unchanged, avoiding an invalid `-inf - -inf` update.

## Production dispatch and isolation

`GQAAttention._mixed_context_forward` dispatches to 0088A only when all of the following hold:

- `MINI_LLM_TRAIN_ATTN_BACKEND=triton`;
- the active backend is CuPy;
- `Hq=16`, `Hkv=4`, and head dimension 64;
- Q/K/V are BF16 and the mixed-attention context is FP32;
- sequence length is at most 4096;
- the static pattern is exactly original head 0 full causal and heads 1–15 local-512;
- no current retrieval group or terminal-memory group is active;
- at least one local current head remains after historical substitution.

The forward cache is tagged `triton_mixed_0088a`; backward recognizes that tag and executes the matching custom backward once over the full original tensor shapes. Existing `performance_scope` instrumentation records `attention.triton_mixed.forward` and `attention.triton_mixed.backward`.

Triton is an optional dependency:

```toml
triton-attention = ["triton==3.8.0"]
```

The ordinary project dependencies and `gpu` extra are unchanged. Importing MyLocal_LLM with the default grouped backend does not import or require Triton.

## Environment used

| Component | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, SM120, 16 GiB |
| Python | 3.12.3 |
| CuPy | 14.2.0 |
| CUDA runtime reported by CuPy | 13.2 |
| Triton | 3.8.0 |

Triton was tested in a disposable virtual environment while loading the established production CuPy packages. Torch was not installed and is not required.

A reproducible isolated setup is:

```bash
python -m venv .venv-0088a-triton
.venv-0088a-triton/bin/python -m pip install 'triton==3.8.0'
```

With the repository's production environment at `.venv`, run isolated validation with:

```bash
MINI_LLM_BACKEND=cupy \
PYTHONPATH=.venv/lib/python3.12/site-packages \
.venv-0088a-triton/bin/python -m pytest -q \
  tests/test_0088a_custom_attention.py \
  tests/test_0088a_production_attention.py
```

Only after the bounded validation succeeds should the optional dependency be installed into the actual training environment, for example:

```bash
.venv/bin/python -m pip install -e '.[triton-attention]'
```

## Correctness

The declared acceptance gates were:

- relative L2 error <= 0.02 for output, dQ, dK, and dV;
- cosine similarity >= 0.999 for each tensor;
- exactly zero current output and dQ for inactive/substituted heads.

### Production shape: B3/T2304, 15 active local heads

Custom Triton versus production grouped CuPy:

| Tensor | Relative L2 | Cosine | Max abs |
|---|---:|---:|---:|
| Output | 0.002537 | 0.99999678 | 0.001924 |
| dQ | 0.004313 | 0.99999070 | 0.001210 |
| dK | 0.004213 | 0.99999112 | 0.000260 |
| dV | 0.002413 | 0.99999709 | 0.004339 |

The exact local-window boundary test at query 519 passed:

- local key 7 excluded;
- local key 8 included;
- future key 520 excluded;
- full-attention head key 7 included.

The scattered active-original-head pattern `(1, 2, 4, 5, 7, 8, 10, 12, 15)` passed. Production dispatch also passed with all 15 current local heads and with historical substitutions `(2, 5, 7, 11, 14, 15)`. Inactive current output and dQ were exactly zero.

## Performance methodology

All steady-state values below are warmed CUDA GPU-event medians. Forward, backward, and combined F+B are independently measured distributions; their medians must not be added. Compilation, fixture I/O, and host/device output copies are outside the timed region. The combined measurement includes FP32 output/gradient allocation, full/local custom launches, probability recomputation, native shared-KV dK/dV accumulation, and inactive-head handling.

The grouped and compiled FlexAttention numbers are the matched 0087B references using the same B3/T shapes, BF16 Q/K/V, FP32 effective outputs/gradients, original 16:4 GQA mapping, masks, and upstream gradient.

### Production-shaped B3/T2304, 15 active local heads

| Candidate | Forward | Backward | Combined F+B |
|---|---:|---:|---:|
| Production grouped CuPy | 2.028 ms | 4.330 ms | 6.094 ms |
| Compiled FlexAttention | 0.237 ms | 0.860 ms | 1.078 ms |
| Custom Triton 0088A | 0.265 ms | 0.781 ms | 1.011 ms |

The custom combined path is approximately 6.0x faster than grouped CuPy. Its backward latency is approximately 9.2% below the FlexAttention reference, while its forward remains slightly slower. Its combined latency is approximately 6.2% below FlexAttention for this dense production pattern.

### Shape matrix, 15 active local heads

| Shape | Grouped CuPy F+B | Compiled Flex F+B | Custom Triton F+B |
|---|---:|---:|---:|
| B3/T1024 | 3.062 ms | 0.494 ms | 0.445 ms |
| B3/T2048 | 5.301 ms | 0.945 ms | 0.926 ms |
| B3/T2304 | 6.094 ms | 1.078 ms | ~1.01 ms |
| B3/T4096 | 12.086 ms | 1.933 ms | 1.928 ms |

At T4096, custom Triton and FlexAttention are effectively tied within likely measurement noise.

### Historical-substitution patterns, B3/T2304

| Active local heads | Compiled Flex F+B | Custom Triton F+B |
|---:|---:|---:|
| 15 | 1.078 ms | ~1.01 ms |
| 13 | 0.899 ms | 0.965 ms |
| 9 | 0.738 ms | 0.762 ms |

The custom implementation wins the dense 15-head case and remains close, but does not beat FlexAttention for every sparse substitution pattern.

### Memory and compilation

For representative B3/T2304 combined execution, the custom process reported:

```text
CuPy pool used after combined:     149,078,528 bytes
CuPy pool reserved after combined: 156,598,784 bytes
```

The matched Flex process reported approximately 139.6 MiB peak allocated and 148.0 MiB peak reserved. These allocator metrics have different semantics, but both candidates are well below the 16 GiB device limit.

A representative cold compile plus first F+B took approximately 640 ms of host wall time. Another process observed approximately 181 ms with compiler caching. Compilation is excluded from steady-state timings and must be completed during warmup before a production measurement.

## Nsight attribution

An earlier Nsight Systems diagnostic, before combining native KV heads into one grid, attributed summed kernel time as follows:

| Kernel family | Calls | Summed GPU time |
|---|---:|---:|
| dK/dV | 4 | 0.613 ms |
| dQ | 2 | 0.293 ms |
| forward | 2 | 0.208 ms |
| delta | 1 | 0.061 ms |

Combining dK/dV into one runtime-KV launch reduced warmed backward from roughly 1.011 ms to 0.782 ms. This identifies shared-KV dK/dV partitioning and launch structure as the principal resolved bottleneck in the experimental backward.

The diagnostic trace predates the final runtime-KV optimization and therefore must not be presented as the final kernel count or final latency. Its generated files are intentionally kept outside the patch under:

```text
benchmarks/results/0088a_custom_profile.nsys-rep
benchmarks/results/0088a_custom_profile.sqlite
benchmarks/results/0088a_custom_profile_stats_cuda_gpu_kern_sum.csv
benchmarks/results/0088a_custom_profile_stats_cuda_gpu_mem_time_sum.csv
benchmarks/results/0088a_custom_profile_stats_cuda_api_sum.csv
```

No Nsight Compute hardware-counter claim is made.

## Reproduction

### CPU/default-dependency tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_0088a_custom_attention.py \
  tests/test_0088a_production_attention.py
```

Expected in the production environment without Triton: CPU contract tests pass and optional GPU Triton tests skip.

### Syntax validation

```bash
.venv/bin/python -m py_compile \
  mini_llm/ops/triton_mixed_attention.py \
  mini_llm/ops/attention.py \
  benchmarks/benchmark_0088a_custom_attention.py \
  tests/test_0088a_custom_attention.py \
  tests/test_0088a_production_attention.py
```

### Production-shape attention benchmark

The fixture and grouped reference are produced by PATCH 0087B. From the repository root:

```bash
MINI_LLM_BACKEND=cupy \
PYTHONPATH=.venv/lib/python3.12/site-packages \
.venv-0088a-triton/bin/python \
  benchmarks/benchmark_0088a_custom_attention.py \
  --fixture benchmarks/results/0087b_fixture_b3_t2304.npz \
  --heads 15 \
  --warmup 10 \
  --repeats 30 \
  --reference-tensors benchmarks/results/0087b_cupy_b3_t2304_h15_tensors.npz \
  --output-tensors benchmarks/results/0088a_custom_b3_t2304_h15_final_tensors.npz \
  --output-json benchmarks/results/0088a_custom_b3_t2304_h15_final.json
```

The runner refuses to accept timings as valid if output/dQ/dK/dV parity fails.

### Exact production-preset compatibility probe

The existing 0087A no-optimizer probe is intentionally locked to the grouped baseline and rejects another backend, so it is not a valid Triton smoke command without a separate future profiling patch.

Instead, 0088A was checked directly against `checkpoints/0083_intermediate_20L_1024_fullmix_100k/config.json` through the production `GQAAttention` dispatch. The checkpoint declares BF16, T=4096, 20 layers, Hq=16, Hkv=4, D=64, original head 0 window 4096, and original heads 1–15 window 512. With strict mode enabled, forward and backward returned cache kind `triton_mixed_0088a` for every layer and every exact historical substitution pattern:

```text
skipped heads                 active current local heads
()                            15
(10, 11)                      13
(12, 13)                      13
(14, 15)                      13
(10, 11, 12, 13)             11
(10, 11, 14, 15)             11
(10, 11, 12, 13, 14, 15)      9
```

All 20 layers had no legacy retrieval group and no terminal-memory module, as required by this backend. This probe instantiated the checkpoint's exact attention configurations but used bounded synthetic tensors; it did not load optimizer state, update parameters, or write a checkpoint. It establishes architectural dispatch compatibility, not full-model throughput.

## Engineering conclusion

The matched attention benchmark demonstrates a substantial and reproducible optimization opportunity in the production grouped attention backward. A geometry-specific fused traversal removes the grouped small-GEMM and shared-KV reduction overhead and reduces mixed-attention F+B from 6.094 ms to approximately 1.01 ms at B3/T2304/15 heads, while satisfying the declared numerical and original-head contracts.

This does **not** imply a six-fold full-model training speedup: attention is only one part of each layer, historical attention remains separate, and FFN, projections, LM head, optimizer/offload, data preparation, and accumulation are unchanged. The exact 20-layer checkpoint architecture passed strict production dispatch, but a future backend-neutral no-optimizer full-model probe is still required to quantify end-to-end throughput and peak VRAM. Grouped remains the code default; enabling Triton is an explicit operator decision.
