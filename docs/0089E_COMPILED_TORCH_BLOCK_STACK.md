# PATCH 0089E — Consecutive compiled-Torch Transformer-block stack

## Scope

PATCH 0089E extends the PATCH 0089B one-block comparison to fixed consecutive depths:

```text
1, 2, 4, 8, 20
```

The benchmark compares production CuPy with compiled PyTorch at the existing B3/T2304 geometry. It measures complete forward plus complete numerical backward only. It does not include an optimizer, activation checkpointing, historical reads, the LM head, or loss computation.

No PATCH 0089B or production file is modified. The implementation consists of:

- `benchmarks/benchmark_0089e_block_stack.py`
- `tests/test_0089e_block_stack_contract.py`
- `docs/0089E_COMPILED_TORCH_BLOCK_STACK.md`

The existing `0089B-block-fixture-v1` fixture and helpers in `benchmarks/reference_0089b_torch_block.py` remain the source of truth.

## Numerical graph

Every layer preserves the 0089B mixed-precision contract:

```text
FP32 residual
→ FP32 RMSNorm statistics/multiply → BF16 branch
→ packed BF16 QKV projection
→ absolute-position BF16 RoPE
→ head 0 full causal + heads 1..15 causal local-512
→ FP32 attention context → BF16 output projection
→ FP32 residual
→ FP32 RMSNorm statistics/multiply → BF16 branch
→ BF16 gate/up/SwiGLU/down
→ FP32 residual
```

Residual activations and residual gradients remain FP32. Parameter and branch storage remains BF16. CuPy parameter-gradient accumulation is FP32. Torch BF16 leaf gradients are converted to FP32 inside the timed range.

The same absolute `position_ids` and read-only attention geometry are used in every layer.

## Independent layer storage from one fixture

The fixture parameter arrays are loaded once. Every layer receives the same numerical values in independent device storage.

CuPy constructs one independent production `TransformerBlock` per layer, copies fixture values into each block's existing `Parameter.data`, and refreshes each layer's packed QKV compute buffer. It never assigns one shared fixture allocation as a layer parameter.

Torch transfers one base copy of each fixture parameter and creates independent per-layer leaves with `detach().clone().requires_grad_(True)`. Wq/Wk/Wv are represented by one independent packed QKV leaf per layer, matching PATCH 0089B.

Before execution the benchmark verifies:

- distinct parameter objects across layers;
- non-overlapping parameter device ranges across layers;
- non-overlapping gradient device ranges across layers;
- seven independent Torch leaves per layer;
- nine logical parameter gradients per layer.

CuPy Wq/Wk/Wv gradient views may share one packed backing allocation within a layer, but no gradient range may overlap another layer.

## CuPy execution

CuPy is required to use:

```text
MINI_LLM_TRAIN_ATTN_BACKEND=triton
MINI_LLM_TRAIN_ATTN_STRICT=1
MINI_LLM_FUSED_BF16_RMSNORM=1
MINI_LLM_FUSED_BF16_RMSNORM_STRICT=1
MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM=1
MINI_LLM_BF16_MIXED_CONTEXT=0
MINI_LLM_FUSED_EXPERT_GEMM=0
```

Forward stores one cache dictionary per layer. Backward traverses the block/cache lists in reverse order.

An untimed preflight fails unless every layer reports:

- `triton_mixed_0088a` attention;
- PATCH 0089C fused attention-residual/RMSNorm2 execution;
- FP32 residual caches;
- BF16 attention branch inputs and parameters;
- FP32 parameter gradients;
- finite output and gradients.

The shared RMSNorm CUDA module, Triton binaries, and read-only RoPE tables may be reused across layers. Activations, block caches, packed QKV buffers, parameters, and gradients remain layer-local.

## Torch execution

The benchmark builds the existing 0089B one-block Torch callable once, then invokes it repeatedly from a fixed-depth Python closure with independent layer leaves. The complete chain is compiled once:

```python
torch.compile(
    stack_forward,
    fullgraph=True,
    dynamic=False,
    mode=compile_mode,
)
```

The benchmark does not compile one block and call it from an external Python loop. The closed depth is unrolled into one Dynamo/Inductor graph, giving Torch visibility across block boundaries.

Mask construction, fixture transfer, and the first compiled F+B are reported separately. The first F+B includes lazy AOTAutograd backward compilation.

TF32 is disabled and float32 matmul precision is `highest`, matching PATCH 0089B.

## Correctness contracts

### Depth 1

Depth 1 exports the unchanged 11-tensor PATCH 0089B contract:

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

### Depth 2

Depth 2 exports 22 tensors:

1. final output;
2. initial input gradient;
3. block-0 output / block-1 input;
4. gradient with respect to that inter-block activation;
5. all nine logical parameter gradients for layer 0;
6. all nine logical parameter gradients for layer 1.

The inter-block activation and gradient are both FP32. Torch retains the returned block-0 activation gradient; CuPy records the gradient produced by block-1 backward before passing it into block-0 backward.

Every compared tensor must satisfy both predeclared gates:

```text
relative L2 <= 0.02
cosine >= 0.999
```

Shape, missing-name, extra-name, NaN, and non-finite metric cases fail closed.

### Depths 4, 8, and 20

Deeper runs are timing-only. They perform untimed storage, dtype, execution-path, and finite-value checks, but do not copy complete gradients to the host or write tensor archives.

Passing `--reference-tensors` or `--output-tensors` at a timing-only depth is an error. The benchmark never silently substitutes partial parity.

## Timing

Only complete F+B is measured.

- gradient clearing occurs outside timing;
- forward, backward, and all Torch BF16 logical parameter-gradient FP32 conversions occur inside timing;
- five warmups and twenty samples are the defaults;
- raw CUDA-event and synchronized host-wall samples are retained;
- median, p10, p90, standard deviation, and MAD are reported;
- median milliseconds per block are reported as `combined_ms_per_block`;
- compile/setup and memory probes are separate from steady-state timing.

Each backend/depth should run in a fresh process. This prevents CuPy pools, Torch allocator reservations, or depth-specific Inductor modules from carrying into another result.

## Memory gate

Before timing, one live complete F+B is measured against `--memory-limit-fraction`, default `0.90`.

CuPy uses the maximum of:

- CuPy pool high-water;
- total device memory minus free memory after the live preflight.

Torch uses the maximum of:

- Torch peak reserved memory;
- total device memory minus free memory after the live preflight.

The device-used measurement includes allocations owned by other processes, intentionally making the gate conservative. If the threshold is exceeded, the report status is `invalid_memory_gate`, timing is skipped, and the process exits nonzero.

Depth 20 may not fit on a 16 GB GPU. The benchmark does not enable checkpointing, offload, recomputation, or a different cache policy after an OOM because that would change the measured graph.

## Cross-block fusion boundary

PATCH 0089C fuses the attention residual with RMSNorm2 inside each CuPy block. It does not fuse:

```text
block i final FFN residual add
→ block i+1 RMSNorm1
```

Whole-chain Torch compilation can optimize elementwise operations, casts, gradient additions, buffer lifetimes, and scheduling around this edge. The FP32 boundary activation remains required for residual and backward paths, and GEMMs/FlexAttention remain hard kernel boundaries. `fullgraph=True` therefore means one captured graph, not one CUDA kernel.

## Reproduction

The fixture is the existing PATCH 0089B B3/T2304 fixture:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python \
  benchmarks/export_0089b_block_fixture.py \
  --checkpoint checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --layer 0 --batch 3 --seq 2304 \
  --output benchmarks/results/0089b_b3_t2304_fixture.npz
```

Generate a depth-2 CuPy correctness reference:

```bash
.venv/bin/python benchmarks/benchmark_0089e_block_stack.py \
  --backend cupy \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 2 --warmup 5 --repeats 20 \
  --output-tensors benchmarks/results/0089e_depth2_cupy_tensors.npz \
  --output-json benchmarks/results/0089e_depth2_cupy.json
```

Run compiled Torch against it:

```bash
.venv-0089b-torch/bin/python benchmarks/benchmark_0089e_block_stack.py \
  --backend torch-compile \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 2 --warmup 5 --repeats 20 \
  --reference-tensors benchmarks/results/0089e_depth2_cupy_tensors.npz \
  --output-tensors benchmarks/results/0089e_depth2_torch_tensors.npz \
  --output-json benchmarks/results/0089e_depth2_torch.json
```

Run a deeper timing-only sample in a fresh process:

```bash
.venv/bin/python benchmarks/benchmark_0089e_block_stack.py \
  --backend cupy \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 8 --warmup 5 --repeats 20 \
  --output-json benchmarks/results/0089e_depth8_cupy.json

.venv-0089b-torch/bin/python benchmarks/benchmark_0089e_block_stack.py \
  --backend torch-compile \
  --fixture benchmarks/results/0089b_b3_t2304_fixture.npz \
  --depth 8 --warmup 5 --repeats 20 \
  --output-json benchmarks/results/0089e_depth8_torch.json
```

Repeat with `--depth 1`, `4`, and `20` in separate processes. Depth 1 also requires `--output-tensors`; depths greater than 2 reject it.

## Results — depth 2 (B3/T2304)

GPU: NVIDIA GeForce RTX 5080, CUDA 13.0, cuBLAS 13020, Triton 3.8.0

### Timing — median per block

| Backend | GPU-event median | Host-wall median |
|---|---:|---:|
| CuPy | 12.514 ms | 12.605 ms |
| Compiled Torch | 10.047 ms | 10.207 ms |

**Speedup: ~1.246× (19.7% reduction)**

CuPy 90th percentile (25.475 ms) exceeds Torch 10th percentile (19.641 ms); the distributions are cleanly separated.

### Parity — 22 tensors, all passed

| Tensor | Relative L2 | Cosine |
|---|---:|---:|
| output | 0.00327 | 0.99999 |
| d_input | 0.00685 | 0.99998 |
| boundary_output | 0.00250 | 0.999997 |
| boundary_grad | 0.00418 | 0.99999 |
| layer0_grad_norm1_gamma | 0.00913 | 0.99996 |
| layer0_grad_norm2_gamma | 0.00761 | 0.99997 |
| layer0_grad_w_down | 0.00585 | 0.99993 |
| layer0_grad_w_gate | 0.00635 | 0.99993 |
| layer0_grad_w_up | 0.00630 | 0.99993 |
| layer0_grad_wk | 0.00940 | 0.99996 |
| layer0_grad_wo | 0.00812 | 0.99996 |
| layer0_grad_wq | 0.00881 | 0.99994 |
| layer0_grad_wv | 0.00970 | 0.99995 |
| layer1_grad_norm1_gamma | 0.00764 | 0.99997 |
| layer1_grad_norm2_gamma | 0.00649 | 0.99998 |
| layer1_grad_w_down | 0.00615 | 0.99993 |
| layer1_grad_w_gate | 0.00589 | 0.99993 |
| layer1_grad_w_up | 0.00559 | 0.99994 |
| layer1_grad_wk | **0.01173** | 0.99993 |
| layer1_grad_wo | 0.00576 | 0.99997 |
| layer1_grad_wq | 0.01137 | 0.99992 |
| layer1_grad_wv | 0.00756 | 0.99997 |

**Worst tensor:** `layer1_grad_wk` with relative L2 = 0.01173, cosine = 0.99993 (worst pair: index `[240, 45]`, actual 0.322265625, reference 0.3359375)

All tensors pass the declared gates (relative L2 ≤ 0.02, cosine ≥ 0.999).

### Memory — depth 2 CuPy

| Metric | Value |
|---|---:|
| Device total | 16.6 GB |
| Observed peak | 2.93 GB (17.6% of device) |
| Gate accepted | Yes (limit fraction 0.90) |

### Setup notes

- **CuPy runtime flags:**
  ```text
  MINI_LLM_TRAIN_ATTN_BACKEND=triton
  MINI_LLM_TRAIN_ATTN_STRICT=1
  MINI_LLM_FUSED_BF16_RMSNORM=1
  MINI_LLM_FUSED_BF16_RMSNORM_STRICT=1
  MINI_LLM_FUSED_TRAIN_RESIDUAL_RMSNORM=1
  MINI_LLM_BF16_MIXED_CONTEXT=0
  MINI_LLM_FUSED_EXPERT_GEMM=0
  ```
- **Torch compile flags:** `fullgraph=True`, `dynamic=False`, `mode=default`
- **Warmup/repeats:** 5 / 20
- **First F+B (incl. compile):** CuPy 439.1 ms, Torch 381.9 ms
- **Fixture transfer + stack setup:** CuPy 225.6 ms, Torch 176.8 ms
- **Torch cross-block boundary probe:** untimed eager single-block VJP performed only when AOTAutograd returned the auxiliary boundary without a retained `.grad`

## Validation

CPU-fast validation:

```bash
.venv/bin/python -m py_compile \
  benchmarks/benchmark_0089e_block_stack.py \
  tests/test_0089e_block_stack_contract.py

.venv/bin/python -m pytest -q tests/test_0089e_block_stack_contract.py
```

The optional depth-2 CuPy-versus-compiled-Torch smoke is disabled by default. Enable it explicitly with:

```bash
RUN_0089E_GPU_SMOKE=1 \
PATCH_0089E_GPU_FIXTURE=benchmarks/results/0089b_b3_t2304_fixture.npz \
.venv/bin/python -m pytest -q \
  tests/test_0089e_block_stack_contract.py::test_optional_depth_two_cupy_vs_compiled_torch_smoke
```
