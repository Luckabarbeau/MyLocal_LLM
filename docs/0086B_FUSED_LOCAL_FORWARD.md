# PATCH 0086B — Experimental fused local-attention forward

## Patch metadata

- **PATCH ID:** `0086B`
- **Source revision:** `39d91e963562376f9d1ed5530f634708ffb539e4`
- **Required prior patches:** `0085I` and `0085J`
- **Default backend:** unchanged (`grouped`)
- **Experimental opt-in:** `MINI_LLM_TRAIN_ATTN_BACKEND=fused`
- **Strict diagnostics:** `MINI_LLM_TRAIN_ATTN_STRICT=1`

Apply after 0085I and 0085J:

```bash
git apply --check PATCH_0086B_EXPERIMENTAL_FUSED_LOCAL_FORWARD.patch
git apply PATCH_0086B_EXPERIMENTAL_FUSED_LOCAL_FORWARD.patch
```

No checkpoint, optimizer, tokenizer, sampler, architecture, historical schedule,
head assignment, inference path or running training process was modified.

## Why this patch exists

The grouped-cuBLAS baseline materializes BF16 score and probability chunks.
0086B tests whether a tiled online-softmax forward can reduce that global
workspace while retaining Tensor-Core dot products and the exact causal local
mask. It is an experiment, not a default replacement.

The old grouped forward and backward remain intact. Unsupported fused calls
fall back to grouped unless strict mode is requested.

## Implementation

`mini_llm/ops/fused_local_attention.py` provides templated CUDA/NVRTC kernels
for `BN=16` and the measured `BN=32` experiment. `BN=16` remains the default;
`MINI_LLM_FUSED_LOCAL_TILE_N=32` selects the alternative explicitly.

Geometry and ownership:

- one CTA owns `(batch, selected Q head, 32 query rows)`
- two warps per CTA; each warp owns 16 query rows
- K/V tiles contain 16 or 32 positions
- D is fixed at 64
- QK and P×V use BF16 WMMA with FP32 accumulators
- running row maximum, denominator, output numerator and LSE are FP32
- output and LSE are contiguous FP32 arrays
- no global score or probability tensor is allocated

Both variants use 55 registers per thread and zero compiler-reported local
memory on SM120. Static shared memory is 19,840 bytes for BN16 and 27,008 bytes
for BN32.

Inputs use their actual positive B/T/H element strides. D must be contiguous.
The normalized device `kv_map` is read directly, preserving original-head GQA
mapping for arbitrary selected-head subsets. The exact mask is:

```text
max(0, i - window + 1) <= j <= i
```

The implementation explicitly guards fully masked intermediate K tiles so
`exp(-inf - -inf)` cannot produce NaN.

## Integration and cache contract

The dispatch seam remains `local_window_attention_forward`.

```bash
# Existing default
MINI_LLM_TRAIN_ATTN_BACKEND=grouped

# Experimental local forward
MINI_LLM_TRAIN_ATTN_BACKEND=fused
```

A cached fused forward stores Q/K/V, FP32 output, FP32 LSE, explicit mapping,
window, scale and grouped chunk geometry. Training backward dispatches to the
existing validated `_local_grouped_backward_recomputed`; 0086B does not claim a
fused backward.

Fused cached training therefore requires:

- CuPy BF16 Q/K/V
- D=64
- effective window <=512
- positive strides and contiguous D
- FP32 mixed-context output
- grouped BF16 recompute backward availability
- `MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS=1`

Head 0 at W>512, unsupported dtypes/layouts and other model families fall back
to grouped mode when strict mode is off. The full-current kernel remains future
work.

## Correctness

Focused GPU command:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_0086b_fused_local_forward.py -vv
```

Result:

```text
4 passed in 0.58s
```

Covered behavior:

- strided zero-copy Q input
- explicit uneven `kv_map`
- FP64 output and LSE oracle comparison
- public opt-in dispatch and tagged cache
- exact reuse of grouped recompute backward gradients
- W=512 exclusion of key 0 at query 519
- inclusion of exact left-boundary key 8 at query 519
- strict rejection of D != 64

For the deterministic `B=2,T=33,Hq=3,Hkv=4,D=64` strided-Q case:

| Value | max abs vs FP64 | RMSE | relative L2 | signed bias |
|---|---:|---:|---:|---:|
| output | 4.5165e-4 | 6.4362e-5 | 9.3049e-4 | -5.9023e-7 |
| LSE | 2.8556e-7 | 1.0135e-7 | 3.7352e-8 | 8.9215e-9 |

The public backward comparison used identical Q/K/V/dOut and was exactly equal
to grouped recompute for dQ/dK/dV in the focused test.

CPU regression command:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_0085i_attention_memory_moves.py \
  tests/test_0085j_grouped_stride_uniformity.py \
  tests/test_0086a_attention_oracle.py \
  tests/test_0083c_hybrid_current_attention.py \
  tests/test_local_window_attention.py \
  tests/test_mixed_attention.py \
  tests/test_bf16_attention_probability_cache.py \
  tests/test_attention.py
```

Result: **52 passed in 3.25 s**.

## Forward-only RTX 5080 measurements

Command:

```bash
MINI_LLM_BACKEND=cupy python \
  benchmarks/benchmark_0086b_fused_local_forward.py \
  --batch 3 --seq 2304 --warmup 10 --repeats 30 \
  --json-out benchmarks/results/0086b_gpu_b3_t2304_forward.json
```

Calls were alternated grouped/fused to reduce ordering bias. Timings include
public dispatch and output allocation but exclude JIT warmup.

| Active local heads | Grouped median | Fused median | Grouped / fused |
|---:|---:|---:|---:|
| 15 | 2.151 ms | 1.954 ms | **1.101×** |
| 13, layer-0 pattern | 1.861 ms | 1.727 ms | **1.078×** |
| 9, layer-4 pattern | 1.213 ms | 1.250 ms | **0.971×** |

Worst fused-versus-grouped max absolute output error was 1.882e-3. Worst
relative L2 error was 2.525e-3. All outputs were finite.

At `B=1,T=129`, the fused kernel measured 2.39–2.56× faster, showing that it
substantially reduces small-shape dispatch/workspace overhead. That does not
translate into the same production-geometry gain.

## Combined forward and backward

The unchanged grouped recompute backward dominates the overall attention call:

| Case | Grouped fwd+bwd | Fused-fwd + grouped-bwd | Ratio |
|---|---:|---:|---:|
| full head 0 | 0.936 ms | 0.920 ms | 1.018×; grouped fallback |
| 15 local heads | 5.204 ms | 5.157 ms | **1.009×** |
| 13 local heads | 4.522 ms | 4.429 ms | **1.021×** |
| 9 local heads | 2.984 ms | 3.148 ms | **0.948×** |

These isolated measurements did not justify enabling fused mode globally.

## BN32 experiment

Both BN16 and BN32 passed the same four GPU correctness tests. At
`B=3,T=2304`, BN32 was decisively slower despite halving the K-tile loop count:

| Heads | Grouped / BN16 | Grouped / BN32 |
|---:|---:|---:|
| 15 | **1.126×** | 0.791× |
| 13 | **1.136×** | 0.740× |
| 9 | **1.009×** | 0.739× |

BN32 raises static shared memory from 19,840 to 27,008 bytes and reduces
resident-CTA flexibility. It is retained only as an explicit experiment and is
rejected for production use.

## Matched production training benchmark

Both modes resumed the immutable production checkpoint at step 6800 with the
same seed, sampler state, batch size 3, accumulation 44, Hybrid-Muon state and
runtime flags. Validation was disabled for the smoke horizon. Temporary output
checkpoints used separate directories and were deleted after parameter parity
was measured.

Three matched optimizer steps processed the same 851,823 supervised tokens:

| Mode | Total step time | Duration-weighted supervised tok/s |
|---|---:|---:|
| Grouped | 51.613 s | **16,504** |
| Fused BN16 | 54.106 s | **15,744** |

Fused BN16 increased step time by 4.83% and reduced supervised throughput by
4.61%. All three fused steps were slower than their grouped counterparts. The
production backend is therefore rejected, not selectively enabled.

One-step optimizer parity across all 338,134,308 saved parameters:

- relative L2 difference: 3.1845e-4
- cosine: 0.9999999493
- RMSE: 8.3784e-6
- maximum difference: 0.00390625 (one BF16 step near magnitude one)
- differing elements: 2.73%

The matched one-step losses differed by 5.7e-6 and gradient norms by 4.64e-5.
These differences are consistent with the measured BF16 forward/reduction
change, but performance—not correctness—is the production rejection reason.

Generated reports now also include:

- `benchmarks/results/0086b_gpu_b3_t2304_bn16.json`
- `benchmarks/results/0086b_gpu_b3_t2304_bn32.json`
- `benchmarks/results/0086b_production_grouped_1step.csv`
- `benchmarks/results/0086b_production_fused_bn16_1step.csv`
- `benchmarks/results/0086b_production_grouped_3step.csv`
- `benchmarks/results/0086b_production_fused_bn16_3step.csv`
- `benchmarks/results/0086b_production_comparison.json`

## Limitations and fallback

- Forward only; backward is the existing grouped recompute implementation.
- No fused full-causal head.
- No BF16 public context-output mode.
- No shapes with D != 64 or W >512.
- No selective production dispatch has been added.
- The default remains grouped cuBLAS.

## Decision and next action

Keep `MINI_LLM_TRAIN_ATTN_BACKEND=grouped` for production. Do not proceed to a
custom backward on the current design: the measured forward savings are too
small to overcome integration effects, and whole-step throughput regressed.
The next useful experiment would require a materially different forward work
partition—such as sharing K/V tiles across Q heads mapped to the same KV head—
and must first beat grouped forward by enough margin to satisfy an Amdahl-law
whole-step target before any additional production integration.
