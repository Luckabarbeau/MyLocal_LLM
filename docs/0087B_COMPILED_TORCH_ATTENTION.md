# 0087B — Production grouped CuPy versus compiled FlexAttention

## Scope

0087B isolates the complete post-RoPE mixed-attention component. It compares the production grouped-cuBLAS path with an independent compiled PyTorch implementation and a compatible vendor hybrid. It does not port a Transformer block, modify the production path, implement a custom backward kernel, instantiate an optimizer, or write a checkpoint.

The compared mathematical component is:

- `q`: BF16 `[B,T,16,64]`
- `k/v`: BF16 `[B,T,4,64]`
- FP32 upstream `dOut`: `[B,T,16,64]`
- original GQA mapping `kv_head = original_query_head // 4`
- original head 0: full causal
- active original heads 1–15: inclusive causal local window 512
- substituted historical heads: exactly zero current-attention output and exactly zero current-attention dQ
- shared dK/dV reduction over all original query heads that map to each KV head
- FP32 composed context and effective FP32 dQ/dK/dV outputs

Inputs are deterministic post-RoPE tensors. No RoPE work, output projection, FFN, LM head, historical reader, or optimizer is timed.

## Implementations

### Production CuPy baseline

The CuPy runner uses the public production functions:

- `local_window_attention_forward`
- `local_window_attention_backward`

It applies `apply_runtime_defaults("gpu-fast", training=True, precision="bf16-mixed")` before importing the indexed-attention implementation and retains:

- grouped-cuBLAS backend
- grouped pointer upload and compact descriptors
- fused BF16 local pipeline
- local probability recomputation
- local query chunk `auto`

The effective F+B timing includes original-head selection/packing, separate full/local public dispatch, FP32 context allocation/scatter, dQ scatter, and shared dK/dV summation.

### Compiled FlexAttention

The optimized reference uses PyTorch 2.14.1+cu130 and Triton 3.8.0 in a separate `.venv-0087b-torch` process. It uses:

- `torch.compile(..., fullgraph=True, dynamic=False)`
- one logical 16-query-head/4-KV-head `flex_attention` call
- `enable_gqa=True`
- a sparse `BlockMask` with original head identity
- no dense T×T mask
- BF16 Q/K/V and FP32 context cast
- PyTorch autograd backward
- timed BF16→FP32 conversions for dQ/dK/dV

The block mask is prebuilt outside steady-state timing, just as CuPy’s warmed plans/descriptors are outside timing. Compilation and block-mask construction are reported separately.

### Vendor candidate

PyTorch SDPA exposes a causal FlashAttention path but no sparse per-query-head sliding-window bias. A vendor-only exact complete implementation would require a forbidden dense mask and is therefore **unavailable**.

The compatible vendor hybrid uses:

- PyTorch Flash SDPA for original full-causal head 0
- compiled FlexAttention for local heads
- exact output composition and shared gradients

Nsight confirmed `pytorch_flash::flash_fwd_*` and `pytorch_flash::flash_bwd_*` kernels executed. The hybrid is a valid complete candidate, but it is not vendor-only.

## Isolated environment

The production CuPy environment was not modified. The measured reference environment was created with:

```bash
python -m venv .venv-0087b-torch
.venv-0087b-torch/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.14.1+cu130 numpy
```

Measured versions:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, SM120, 16,303 MiB |
| Driver | 580.178.04 |
| Production Python | 3.12.3 |
| CuPy | 14.2.0 |
| CuPy CUDA runtime | 13.2 |
| Isolated PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| Triton | 3.8.0 |
| cuDNN shipped with Torch | 9.24.0 |

Installing this environment downloads several GiB of CUDA/PyTorch packages. It is optional benchmark infrastructure and is not a project or production dependency.

## Reproduction

### Contract tests

```bash
python -m py_compile \
  benchmarks/reference_0087b_torch_attention.py \
  benchmarks/export_0087b_attention_fixture.py \
  benchmarks/benchmark_0087b_attention.py \
  tests/test_0087b_attention_contract.py

python -m pytest -q tests/test_0087b_attention_contract.py
```

### Export byte-exact BF16 fixture

```bash
python benchmarks/export_0087b_attention_fixture.py \
  --batch 3 --seq 2304 --seed 20261022 \
  --output benchmarks/results/0087b_fixture_b3_t2304.npz
```

The fixture stores Q/K/V as portable `uint16` BF16 bit patterns, FP32 `dOut`, a manifest, tensor hashes, and the original 16:4 mapping. No I/O or host/device transfer occurs inside timed regions.

### CuPy production baseline

```bash
MINI_LLM_BACKEND=cupy \
python benchmarks/benchmark_0087b_attention.py \
  --backend cupy \
  --fixture benchmarks/results/0087b_fixture_b3_t2304.npz \
  --heads 15 --warmup 5 --repeats 20 \
  --output-tensors benchmarks/results/0087b_cupy_b3_t2304_h15_tensors.npz \
  --output-json benchmarks/results/0087b_cupy_b3_t2304_h15.json \
  --output-csv benchmarks/results/0087b_cupy_b3_t2304_h15.csv
```

### Compiled FlexAttention

```bash
TORCHINDUCTOR_CACHE_DIR=benchmarks/results/.torchinductor_0087b \
.venv-0087b-torch/bin/python benchmarks/benchmark_0087b_attention.py \
  --backend torch-flex-compile \
  --fixture benchmarks/results/0087b_fixture_b3_t2304.npz \
  --heads 15 --warmup 5 --repeats 20 \
  --reference-tensors benchmarks/results/0087b_cupy_b3_t2304_h15_tensors.npz \
  --output-tensors benchmarks/results/0087b_torch_flex_b3_t2304_h15_tensors.npz \
  --output-json benchmarks/results/0087b_torch_flex_b3_t2304_h15.json \
  --output-csv benchmarks/results/0087b_torch_flex_b3_t2304_h15.csv
```

Use `--heads 13`, `--heads 9`, or `--heads scattered` without renumbering heads.

### Nsight kernel capture

Both backends call the CUDA profiler API only around one warmed combined F+B when `--cuda-profiler-capture` is set:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=benchmarks/results/0087b_cupy_profile \
  python benchmarks/benchmark_0087b_attention.py \
    --backend cupy \
    --fixture benchmarks/results/0087b_fixture_b3_t2304.npz \
    --heads 15 --warmup 2 --repeats 1 \
    --output-tensors benchmarks/results/0087b_cupy_profile_tensors.npz \
    --output-json benchmarks/results/0087b_cupy_profile.json \
    --cuda-profiler-capture

TORCHINDUCTOR_CACHE_DIR=benchmarks/results/.torchinductor_0087b \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=benchmarks/results/0087b_torch_flex_profile \
  .venv-0087b-torch/bin/python benchmarks/benchmark_0087b_attention.py \
    --backend torch-flex-compile \
    --fixture benchmarks/results/0087b_fixture_b3_t2304.npz \
    --heads 15 --warmup 2 --repeats 1 \
    --reference-tensors benchmarks/results/0087b_cupy_b3_t2304_h15_tensors.npz \
    --output-tensors benchmarks/results/0087b_torch_flex_profile_tensors.npz \
    --output-json benchmarks/results/0087b_torch_flex_profile.json \
    --cuda-profiler-capture
```

## Numerical equivalence

Tolerances were declared before production-shape timing:

- relative L2 ≤ 0.02
- cosine ≥ 0.999
- substituted historical-head output and dQ exactly zero

The boundary test explicitly verifies for query 519:

- local key 7 excluded
- local key 8 included
- future key 520 excluded
- full-head key 7 included

### B1/T520 boundary case

Compiled Flex versus grouped CuPy, 15 local heads:

| Tensor | Relative L2 | Cosine | Max abs |
|---|---:|---:|---:|
| Output | 0.001894 | 0.99999821 | 0.001361 |
| dQ | 0.003551 | 0.99999370 | 0.000126 |
| dK | 0.003472 | 0.99999398 | 0.000148 |
| dV | 0.002374 | 0.99999719 | 0.004294 |

The 13-head, 9-head, and scattered cases also passed, with exact zero inactive output/dQ.

### B3/T2304 production case

| Tensor | Relative L2 | Cosine | Max abs | Bias |
|---|---:|---:|---:|---:|
| Output | 0.002812 | 0.99999605 | 0.001953 | -2.38e-9 |
| dQ | 0.004640 | 0.99998924 | 0.001210 | 8.75e-10 |
| dK | 0.004611 | 0.99998937 | 0.000256 | 4.63e-9 |
| dV | 0.002916 | 0.99999575 | 0.005424 | 8.72e-11 |

All measured shapes and head patterns passed the declared thresholds.

### Precision qualification

The production CuPy API directly returns FP32 dQ/dK/dV. FlexAttention’s BF16 input leaves produce BF16 gradients internally; the effective benchmark includes explicit timed FP32 upcasts. Numerical parity is strong, but the internal gradient rounding point is not byte-identical to CuPy. Therefore these results establish **numerical attention equivalence under BF16 tolerances**, not identical internal FP32 dataflow.

## Performance methodology

- GPU-event and synchronized host-wall timings are reported separately.
- Forward, backward, and combined F+B are independent runs; their medians must not be added.
- Every short result uses at least 20 measured samples after warmup.
- The primary B3/T2304 claim uses three independent processes in alternating framework order.
- Q/K/V transfer, fixture I/O, block-mask construction, compilation, and autotuning are outside steady-state timing.
- Head selection, context scatter, shared dK/dV summation, autograd dispatch, and FP32 gradient conversion are inside effective timing.
- TF32 is disabled and PyTorch float32 matmul precision is `highest`.

## Primary result: B3/T2304, 15 local heads

Three independent process medians:

| Run | CuPy F+B | Flex F+B | CuPy backward | Flex backward |
|---:|---:|---:|---:|---:|
| 1 | 6.191 ms | 1.078 ms | 4.378 ms | 0.860 ms |
| 2 | 5.955 ms | 1.075 ms | 4.206 ms | 0.857 ms |
| 3 | 6.094 ms | 1.132 ms | 4.330 ms | 1.118 ms |
| Median of run medians | **6.094 ms** | **1.078 ms** | **4.330 ms** | **0.860 ms** |

Median-of-run-median speedups:

- forward: 2.367 / 0.237 = **10.0×**
- backward: 4.330 / 0.860 = **5.03×**
- complete F+B: 6.094 / 1.078 = **5.65×**

These are attention-component speedups, not whole-model throughput.

## Shape coverage

Single-process 20-sample medians with 15 local heads:

| Shape | CuPy forward | CuPy backward | CuPy F+B | Flex forward | Flex backward | Flex F+B | F+B speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| B3/T1024 | 1.136 | 2.000 | 3.062 | 0.120 | 0.578 | 0.494 | 6.20× |
| B3/T2048 | 2.070 | 3.751 | 5.301 | 0.215 | 0.771 | 0.945 | 5.61× |
| B3/T2304 | 2.394 | 4.378 | 6.191 | 0.233 | 0.860 | 1.078 | 5.74× |
| B3/T4096 | 4.214 | 7.875 | 12.086 | 0.460 | 1.741 | 1.933 | 6.25× |

The T1024 independent backward and combined medians illustrate why independent timings must not be arithmetically combined: different warmed executions and compiler scheduling can make combined lower than the separately measured backward median.

## Historical-head substitution coverage

B3/T2304:

| Active original local heads | CuPy F+B | Flex F+B | Speedup | Parity |
|---:|---:|---:|---:|---|
| 15 | 6.094 ms | 1.078 ms | 5.65× | passed |
| 13 | 5.825 ms | 0.899 ms | 6.48× | passed |
| 9 | 3.932 ms | 0.738 ms | 5.33× | passed |

Inactive heads retain their original indices and original Q→KV associations. They are not compactly renumbered.

## Vendor hybrid and eager references

| Candidate, B3/T2304 | Forward | Backward | F+B | Verdict |
|---|---:|---:|---:|---|
| Compiled Flex | 0.237 ms | 0.860 ms | 1.078 ms | optimized reference |
| Flash head 0 + Flex local | 0.261 ms | 1.099 ms | 1.224 ms | valid, ~13.5% slower than pure Flex |
| Eager Flex | 34.150 ms | 44.719 ms | 77.355 ms | dense-score fallback; not competitive |

Replacing only full head 0 with vendor Flash SDPA does not explain the speedup. The important improvement is in compiled local sparse execution and backward fusion.

## Memory and compilation

B3/T2304/15 heads:

| Metric | CuPy | Compiled Flex |
|---|---:|---:|
| Live/peak allocated after effective F+B | 141.8 MiB pool-used | 139.6 MiB peak allocated |
| Reserved/high-water | 443.5 MiB CuPy pool | 148.0 MiB peak reserved |

CuPy pool reservation is a process high-water proxy rather than an exact allocator peak counter, so the reservation figures are not strictly equivalent.

Cold isolated compilation:

- sparse block-mask construction: **3.144 s**
- first compiled forward + backward: **2.171 s**

These costs are excluded from steady-state timing. A warmed persistent compiler cache reduced subsequent first-execution overhead substantially.

## Kernel attribution

One warmed Nsight-captured B3/T2304 F+B:

| Backend | GPU kernels | Memcpy | Memset | Summed kernel duration |
|---|---:|---:|---:|---:|
| CuPy grouped | 119 | 44 | 8 | 4.456 ms |
| Compiled Flex + FP32 grad conversion | 7 | 0 | 0 | 0.985 ms |
| Flash/Flex vendor hybrid | 13 | 0 | 0 | 1.076 ms |

CuPy top work:

| Kernel family | Instances | Summed duration | Share of summed kernel time |
|---|---:|---:|---:|
| CUTLASS BF16 128×256 TN | 21 | 1.001 ms | 22.5% |
| `local_causal_softmax_recompute_bwd_bf16` | 12 | 0.910 ms | 20.4% |
| `local_causal_softmax_fwd_bf16` | 9 | 0.612 ms | 13.7% |
| CUTLASS BF16 64×64 NN | 14 | 0.524 ms | 11.8% |
| grouped CUTLASS BF16 128×64 | 12 | 0.508 ms | 11.4% |
| CUTLASS BF16 64×64 NT | 14 | 0.386 ms | 8.7% |
| local shared-gradient stores | 12 | 0.183 ms | 4.1% |

Compiled Flex work:

| Kernel | Instances | Duration |
|---|---:|---:|
| fused Flex backward | 1 | 0.696 ms |
| fused Flex forward | 1 | 0.185 ms |
| three BF16→FP32 gradient upcasts | 3 | 0.046 ms total |
| backward conversion/permutation | 1 | 0.042 ms |
| output conversion/permutation | 1 | 0.016 ms |

The operation-level advantage is clear: Flex replaces separate QK/PV/dQ/dK/dV GEMMs, probability recomputation/softmax derivative, shared-KV reduction/store work, and more than 100 launches with one fused forward and one fused backward kernel plus small conversion kernels.

## Relation to 0087A and practical ceiling

0087A’s Nsight projection attributed 205.262 ms of a 668.443 ms microbatch to the combined current-attention range, a measured fraction `p = 0.3071`. Applying the primary component speedup `k = 5.65` gives:

```text
1 / ((1 - p) + p / k) = 1.338×
```

This **1.34×** is a conditional Amdahl ceiling, not a predicted production speedup. It assumes the standalone kernel can replace all current-attention work with the same benefit while preserving CuPy’s FP32 gradient behavior, historical integration, memory lifetime, and full-model scheduling. The 0087A phase is nested and not an exclusive timer.

## Decision and optimization angle

Attention backward is a substantial optimization opportunity. The evidence does **not** support a C++ framework rewrite or replacing the whole model with PyTorch. It supports a narrow investigation of the local-attention backward algorithm and work partition:

1. Fuse probability recomputation, softmax backward, dQ, dK/dV, and original-GQA shared-KV reduction more aggressively.
2. Reduce the current ~119 kernel/memory-operation sequence toward a small number of shape-specialized kernels.
3. Preserve original head numbers and irregular historical substitutions in the kernel schedule.
4. Preserve FP32 effective gradient outputs; benchmark the cost of matching CuPy’s exact internal accumulation policy.
5. Validate the candidate in the full model before using the 1.34× Amdahl ceiling as a forecast.

A vendor-only PyTorch SDPA replacement is not available for this per-head mixed sparse mask. The Flash/Flex hybrid is slower than pure Flex. Therefore the useful reference is the generated fused Flex local backward, not vendor fusion of head 0.

The next decision should be a targeted feasibility/design experiment for a fused local backward (or a compatible generated-kernel integration), with 0087C used only to separate the remaining grouped GEMM efficiency from softmax/reduction fusion. This patch deliberately does not implement that kernel.

## Raw artifacts

Generated under `benchmarks/results/` and intentionally excluded from the Git patch:

- fixtures: `0087b_fixture_b{1,3}_t*.npz`
- per-backend JSON and tensor NPZ outputs
- three independent B3/T2304 run outputs
- CuPy/Flex/vendor Nsight `.nsys-rep`, `.sqlite`, and stats CSV files
- isolated TorchInductor caches

Primary traces:

- `0087b_cupy_profile.nsys-rep`
- `0087b_cupy_profile_stats.csv`
- `0087b_torch_flex_profile.nsys-rep`
- `0087b_torch_flex_profile_stats.csv`
- `0087b_torch_vendor_profile.nsys-rep`
- `0087b_torch_vendor_profile_stats.csv`

## Validation

Passed:

```text
python -m py_compile ...                                passed
python -m pytest -q tests/test_0087b_attention_contract.py
                                                        6 passed
B1/T520 15/13/9/scattered parity                       passed
B3/T1024, T2048, T2304, T4096 parity                   passed
three independent B3/T2304 timing repetitions          completed
compiled fullgraph Flex execution                       completed
Flash/Flex vendor hybrid                                completed
CuPy/Flex/vendor Nsight captures                        completed
```

No production source file, checkpoint, optimizer, sampler, or training flag was modified.
