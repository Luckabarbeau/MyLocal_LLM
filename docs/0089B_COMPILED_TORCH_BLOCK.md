# PATCH 0089B — Complete compiled-PyTorch Transformer-block comparison

## Scope

0089B compares one complete production dense Transformer block, including full numerical backward, against PyTorch eager and `torch.compile(fullgraph=True)` with FlexAttention.

No production model or training hot path is modified by this patch. Historical reads, the LM head/loss, and the optimizer remain separate benchmark scopes.

The matched block graph is:

```text
x FP32 [B,T,1024]
→ RMSNorm1 statistics/multiply FP32 → branch BF16
→ one packed BF16 QKV GEMM
→ absolute-position BF16 RoPE
→ head 0 full causal + heads 1..15 causal local-512
→ context FP32 → BF16 → Wo BF16 GEMM
→ FP32 residual add
→ RMSNorm2 statistics/multiply FP32 → branch BF16
→ separate BF16 gate/up GEMMs
→ BF16 SwiGLU
→ BF16 down GEMM
→ FP32 residual add
→ y FP32
```

Backward preserves the production contract: residual gradients and parameter-gradient accumulation are FP32, while branch gradients are rounded to BF16 before projection/FFN GEMMs.

## Files

- `benchmarks/export_0089b_block_fixture.py`
- `benchmarks/reference_0089b_torch_block.py`
- `benchmarks/benchmark_0089b_block.py`
- `tests/test_0089b_block_contract.py`
- `docs/0089B_COMPILED_TORCH_BLOCK.md`

## Fixture and parity contract

Fixtures are derived from layer 0 of:

```text
checkpoints/0083_intermediate_20L_1024_fullmix_100k
```

Each fixture contains:

- all nine exact checkpoint parameter tensors as portable BF16 `uint16` bit patterns;
- deterministic FP32 residual input `x`;
- deterministic FP32 upstream gradient `d_out`;
- contiguous absolute positions beginning at 61,440;
- parameter/config metadata and dtype/shape/content SHA-256 hashes.

Compared tensors:

1. output;
2. input gradient;
3. norm1 gamma gradient;
4. Wq gradient;
5. Wk gradient;
6. Wv gradient;
7. Wo gradient;
8. norm2 gamma gradient;
9. W_gate gradient;
10. W_up gradient;
11. W_down gradient.

For every tensor the report records maximum absolute error, RMSE, relative L2, cosine similarity, mean bias, and the worst value pair. Gates were declared before timing:

```text
relative L2 <= 0.02
cosine >= 0.999
```

The local-window boundary is explicitly checked at query 519:

```text
local key 7: rejected
local key 8: accepted
local future key 520: rejected
full-head key 7: accepted
full-head future key 520: rejected
```

## Environments

Production CuPy:

```text
CuPy 14.2.0
CUDA runtime 13.2
Triton 3.8.0
RTX 5080, SM120
```

Isolated Torch environment:

```text
.venv-0089b-torch
Torch 2.14.1+cu130
CUDA 13.0
Triton 3.8.0
RTX 5080, SM120
```

Torch was not installed into production `.venv`.

The CuPy process uses:

```bash
MINI_LLM_BACKEND=cupy
MINI_LLM_TRAIN_ATTN_BACKEND=triton
MINI_LLM_TRAIN_ATTN_STRICT=1
MINI_LLM_BF16_MIXED_CONTEXT=0
MINI_LLM_FUSED_EXPERT_GEMM=0
```

Torch TF32 is disabled and float32 matmul precision is set to `highest`.

## Correctness results

All required correctness shapes passed for compiled Torch against production CuPy:

| Shape | Worst tensor | Relative L2 | Cosine | Result |
|---|---|---:|---:|---|
| B1/T17 | norm1 gamma gradient | < 0.02 | > 0.999 | Pass |
| B1/T129 | norm1 gamma gradient | 0.008304 | 0.9999655 | Pass |
| B1/T520 | Wv gradient | 0.007647 | 0.9999708 | Pass |
| B3/T1024 | Wv gradient | 0.007956 | 0.9999684 | Pass |
| B3/T2048 | Wv gradient | 0.008025 | 0.9999678 | Pass |
| B3/T2304 | Wv gradient | 0.008104 | 0.9999672 | Pass |
| B3/T4096 | Wv gradient | 0.007600 | 0.9999711 | Pass |

At B1/T17, Torch eager also passed every tensor. Its worst relative-L2 value was `0.007559` for Wk, with cosine `0.9999716`.

## Timing method

Each backend reports independent forward, backward, and complete F+B measurements. Medians from separate phases are never added.

- five warmups;
- twenty GPU-event samples;
- twenty outer synchronized host-wall samples;
- raw samples retained;
- median, p10, p90, standard deviation, and MAD retained;
- checkpoint loading, mask construction, and compilation excluded;
- backward rebuilds its forward cache outside the event;
- gradient clearing occurs outside the event;
- combined timing includes forward and full backward;
- Torch timing includes BF16 parameter-gradient to FP32 export conversion;
- mask construction and first compiled F+B are reported separately.

## Primary B3/T2304 result

GPU-event medians:

| Backend | Forward | Backward | Complete F+B |
|---|---:|---:|---:|
| production CuPy + strict Triton | 3.4843 ms | 6.7346 ms | 10.6903 ms |
| PyTorch eager + FlexAttention | 40.6638 ms | 54.2629 ms | 94.1063 ms |
| compiled PyTorch + FlexAttention | 2.8662 ms | 6.5074 ms | 9.2205 ms |

Compiled Torch versus production CuPy:

```text
speedup:           1.1594x
latency reduction: 13.75%
forward reduction: 17.74%
backward reduction: 3.37%
```

Compiled Torch setup costs at this shape were separate from steady-state timing:

```text
BlockMask build:       1696.66 ms
first compiled F+B:    4450.31 ms
```

These setup costs explain why compilation is not automatically beneficial for short-lived processes, but they are amortizable in long training runs.

## Scaling results

Twenty-sample GPU-event medians:

| Shape | CuPy F | CuPy B | CuPy F+B | compiled Torch F | compiled Torch B | compiled Torch F+B | Torch speedup | Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B3/T1024 | 1.6591 | 3.1308 | 5.0510 | 1.0843 | 2.8233 | 3.8116 | 1.3252x | 24.54% |
| B3/T2048 | 3.0943 | 6.3808 | 9.7325 | 2.6829 | 5.9900 | 8.3021 | 1.1723x | 14.70% |
| B3/T2304 | 3.4843 | 6.7346 | 10.6903 | 2.8662 | 6.5074 | 9.2205 | 1.1594x | 13.75% |
| B3/T4096 | 6.6078 | 12.9252 | 19.7750 | 4.9799 | 10.8713 | 16.1761 | 1.2225x | 18.20% |

The compiled advantage is repeatable across all tested production lengths; it is not a T2304 anomaly.

## Memory

Peak/high-water measurements use backend-specific allocator semantics and therefore should not be interpreted as byte-exact allocator parity.

| Shape | CuPy pool high-water | Torch peak allocated | Torch peak reserved |
|---|---:|---:|---:|
| B3/T1024 | 405 MiB | 283 MiB | 438 MiB |
| B3/T2048 | 707 MiB | 462 MiB | 682 MiB |
| B3/T2304 | 778 MiB | 509 MiB | 722 MiB |
| B3/T4096 | 1296 MiB | 821 MiB | 1144 MiB |

## Nsight diagnosis

Matched B3/T2304 F+B captures:

| Metric | CuPy | compiled Torch |
|---|---:|---:|
| GPU projected range | 11.263 ms | 9.341 ms |
| GPU operations | 89 | 22 |
| GEMM projected time | ~6.988 ms | ~7.084 ms |
| Non-GEMM remainder | ~4.275 ms | ~2.257 ms |

The key conclusion is that compiled Torch does **not** win through faster GEMMs. Its GEMM time is slightly higher. The approximately 2 ms advantage comes from fewer elementwise/cast/allocation operations and cross-operation fusion.

Attention is no longer the dominant discrepancy. Approximate traced attention-kernel totals were similar:

```text
CuPy custom Triton attention: ~0.943 ms
compiled FlexAttention:       ~0.910 ms
```

Production RMSNorm kernels are also already competitive. The remaining gap is concentrated around projection/FFN integration and residual/cast traffic.

The largest individual CuPy non-GEMM event was a BF16 add reaching about `0.792 ms`, consistent with dense FFN backward:

```python
dx_gate = dg @ W_gate.T
dx_up = du @ W_up.T
dx = dx_gate + dx_up
```

## Rejected ablation: packed gate/up GEMM

The existing `MINI_LLM_FUSED_EXPERT_GEMM=1` mode was tested at B3/T2304 with the same fixture and passed parity, but regressed performance:

| Phase | Separate GEMMs | Packed mode | Change |
|---|---:|---:|---:|
| Forward | 3.4843 ms | 3.7828 ms | 8.57% slower |
| Backward | 6.7346 ms | 7.7187 ms | 14.61% slower |
| F+B | 10.6903 ms | 11.5656 ms | 8.19% slower |

Therefore the production default must remain disabled. Packing all gate/up work is not the right remedy.

## Recommended next optimization

The strongest evidence-supported next patch is a narrow FFN-backward accumulation benchmark:

```text
1. compute dg @ W_gate.T into dx;
2. compute du @ W_up.T into the same dx with GEMM beta=1;
3. remove dx_up allocation and the standalone BF16 add.
```

This preserves the two efficient GEMM geometries and avoids the regression from packed gate/up matrices. The repository already has BF16 grouped GEMM machinery with a `beta` argument; the lower-level `bf16_matmul_into` wrapper currently hardcodes beta zero and could be extended in a separate patch.

The traced `0.792 ms` add gives a useful upper bound of roughly 7.4% of current block F+B at T2304. Actual gain must be measured because changing GEMM dispatch can alter algorithm selection.

Secondary candidates, after that experiment:

1. fuse residual addition, BF16 cast, and the following RMSNorm boundary where legal;
2. fuse or reuse QKV RoPE packing buffers more aggressively;
3. reduce remaining BF16/FP32 copy kernels around branch boundaries;
4. keep historical-read optimization separate—the 0089A heavy-history trace already showed that subsystem dominates history-heavy batches.

Python launch starvation remains low priority; the compiled advantage is primarily device-side operation fusion, not host gaps.

## Reproduction

Export a fixture:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python \
  benchmarks/export_0089b_block_fixture.py \
  --checkpoint checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --layer 0 --batch 3 --seq 2304 \
  --output benchmarks/results/0089b_b3_t2304_fixture.npz
```

Run production CuPy:

```bash
.venv/bin/python benchmarks/benchmark_0089b_block.py \
  --backend cupy \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --warmup 5 --repeats 20 \
  --output-tensors benchmarks/results/0089b_b3_t2304_cupy_tensors.npz \
  --output-json benchmarks/results/0089b_b3_t2304_cupy.json
```

Run compiled Torch:

```bash
.venv-0089b-torch/bin/python benchmarks/benchmark_0089b_block.py \
  --backend torch-compile \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --warmup 5 --repeats 20 \
  --reference-tensors benchmarks/results/0089b_b3_t2304_cupy_tensors.npz \
  --output-tensors benchmarks/results/0089b_b3_t2304_torch_compile_tensors.npz \
  --output-json benchmarks/results/0089b_b3_t2304_torch_compile.json
```

## Validation

```text
.venv/bin/python -m pytest -q tests/test_0089b_block_contract.py
9 passed
```

All edited-file diagnostics and `git diff --check` are clean. Raw reports, tensors, fixtures, and Nsight captures are under ignored `benchmarks/results/`.
