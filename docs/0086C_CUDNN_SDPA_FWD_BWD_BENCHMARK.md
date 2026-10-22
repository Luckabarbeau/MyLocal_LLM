# PATCH 0086C — cuDNN SDPA forward/backward feasibility benchmark

## Scope and decision

0086C is a benchmark-only investigation. It does not edit production attention,
training, model, optimizer, checkpoint, or runtime-default code. The existing
`grouped` backend remains the reference/fallback and 0086B remains opt-in.
No package was installed and no training or optimizer step was run.

**Decision for a future 0086D integration patch: NO-GO in the audited
environment.** The active Python environment has no importable NVIDIA cuDNN
Frontend package (`import cudnn` fails), no loader-visible cuDNN shared library,
and no optional SM120 DSL dependencies. Consequently no real `cudnn.pygraph`
forward/backward graph, SM120 execution plan, workspace, mask contract, or
latency could be validated. 0086C does not emulate cuDNN with another backend
and does not report baseline timings as vendor timings.

The benchmark remains useful as a reproducible capability probe, as a complete
F+B baseline harness, and as correctness-locked planning for the two required
regular-GQA strategies.

## Source audit

Source revision at audit time: `68ba347` with the prerequisite 0085I, 0085J,
0086A and 0086B working-tree changes present.

| Area | Current implementation and 0086C use |
|---|---|
| `mini_llm/ops/attention.py` | `_mixed_context_forward` selects original Q heads and passes explicit `original_head // 4` KV mappings. `_mixed_context_backward` scatters dQ and sums branch dK/dV. Unchanged. |
| `mini_llm/ops/indexed_attention.py` | `local_window_attention_forward/backward` are the public benchmark baseline. Inputs are BTHD; positive noncompact strides with D contiguous are accepted by the grouped path. BF16 grouped forward emits FP32 context in this configuration and backward emits FP32 dQ/dK/dV. |
| `mini_llm/ops/indexed_attention.py` | `_local_grouped_forward` is the default grouped-cuBLAS forward. `_local_grouped_backward_recomputed` reconstructs BF16 scores/probabilities and sums shared-KV gradients. |
| `mini_llm/ops/cublas_grouped.py` | Grouped BF16 GEMM uses FP32 accumulation but requires BF16 C output. 0085J's uniform-stride reconstruction constraint remains untouched. |
| `mini_llm/ops/fused_local_attention.py` | 0086B is D64/W<=512 BF16 fused forward with FP32 output/LSE. It has no fused backward and uses grouped recomputation. It remains opt-in through `MINI_LLM_TRAIN_ATTN_BACKEND=fused`. |
| `mini_llm/ops/historical_read.py`, `historical_schedule.py` | Historical retrieval is outside 0086C. The benchmark models the ordinary current-head sets remaining after valid-history substitutions. |
| `benchmarks/reference_fused_attention_oracle.py` | FP64 BTHD forward/backward oracle used to lock masking, original-head mapping, dummy-head gradients, and dK/dV sums. |
| `benchmarks/benchmark_0086c_cudnn_sdpa.py` | New isolated complete-F+B benchmark. Measures grouped and 0086B through public APIs; reports real cuDNN candidates unavailable rather than substituting an implementation. |
| `benchmarks/cudnn_sdpa_0086c.py` | New benchmark-only capability and GQA-strategy helper. It has no production imports. |
| `tests/test_0086c_cudnn_sdpa.py` | New CPU gates for the exact mask and both GQA integration strategies. |

## Audited environment

| Item | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB |
| Compute capability | SM120 (`12.0`) |
| Driver | `580.178.04`; CUDA driver API reported `13000` |
| CuPy | `14.2.0` |
| CUDA runtime through CuPy | `13020` |
| Python | `3.12.3` |
| `cudnn` Python module | absent: `ModuleNotFoundError: No module named 'cudnn'` |
| `nvidia-cudnn-frontend` distribution | absent |
| `nvidia-cudnn-cu12` / `nvidia-cudnn-cu13` | absent |
| Torch | absent; not added as a dependency |
| `nvidia-cutlass-dsl` | absent |
| `apache-tvm-ffi` | absent |
| loader-visible `libcudnn` | none (`ctypes.util.find_library("cudnn")` and `ldconfig` found none) |

A MATLAB-private cuDNN 8.9.5 library exists outside the active loader path. It
is not the cuDNN Frontend Python API, was not loaded into the training process,
and is not a validated SM120 SDPA route. Pointing the active process at that
private copy would be an invasive and unsafe dependency substitution, so 0086C
did not do so.

Exact capability failure recorded in JSON:

```text
cuDNN Frontend Python is not importable; real pygraph SDPA plan construction
cannot be attempted: ModuleNotFoundError: No module named 'cudnn'
```

Because `import cudnn` is the first required operation, there is no honest
minimal `pygraph.sdpa`/`sdpa_backward` execution call for this installed
environment. Build signatures, variant-pack pointer acceptance, stream
protocol, BF16 GQA plans, and mask bounds therefore remain unverified. The
benchmark exits successfully and records these cases as `unavailable`.

## Mathematical and layout contract

The benchmark uses deterministic post-RoPE tensors:

- Q: `[B,T,16,64]` BF16 BTHD;
- K/V: `[B,T,4,64]` BF16 BTHD;
- dO: FP32, matching the current mixed-context gradient path;
- scale: `1/sqrt(64) = 0.125`;
- local window: 512 unless clamped by T;
- full head: original Q head 0 with original KV head 0;
- local mapping: `kv_head = original_query_head // 4`.

A logical BHSD view is represented by `transpose(0,2,1,3)` and its actual byte
strides are reported. Python does not copy for that transpose. Whether an
installed cuDNN plan accepts those strides, or requires timed packing, is not
claimed.

The hard mask contract is:

```text
max(0, i - W + 1) <= j <= i
```

For `i=519`, `W=512`, key 7 is excluded and key 8 is included. The installed
API could not be checked, so the documented pygraph candidate
`diagonal_band_left_bound=512` is recorded as required but
`installed_api_verified=false`. Vendor timing is invalid until this sentinel
passes through the actual selected cuDNN plan.

## Regular-GQA strategies

Both required strategies are represented without changing original head IDs.

### A. Full 16-head local call

`cudnn_regular16` executes original Q heads 0 through 15 against all four KV
heads with map `[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]`. Only requested local
heads are scattered into context. dO is exactly zero for head 0 and every
historically inactive head, so those dummy heads contribute zero dQ/dK/dV.
Head 0's separate full-causal dK/dV contribution is then added.

For the 13-head pattern `(1..11,14,15)`, dummy local heads are `0,12,13`.
The extra dummy work remains chargeable when a real vendor path becomes
available.

### B. Split by original KV group

`cudnn_kv_split` emits at most four calls. Each call receives one original KV
head and only the eligible original Q heads belonging to it; its relative map
is all zeros. Outputs and dQ are scattered back to original head positions,
and each call's dK/dV is accumulated into its original KV slot. Noncontiguous
Q groups require timed packing if an installed plan rejects their strides.

The scattered audit set `(1,3,4,7,8,10,12,15)` becomes groups
`(1,3)`, `(4,7)`, `(8,10)`, `(12,15)` without relabelling cross-group
ownership.

## Correctness gates

Command:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q \
  tests/test_0086c_cudnn_sdpa.py -vv
```

Result: **5 passed in 0.15 s**.

The tests cover:

1. 15/13/9 and scattered original-head plans and exact GQA maps;
2. full-16 dummy dO masking and exact zero dummy dQ;
3. full-head plus local shared-KV dK/dV summation;
4. four-way KV splitting, output/dQ scatter, and dK/dV accumulation;
5. D64 `i=519,j=7/8` forward and backward sentinel behavior;
6. noninvasive capability reporting when optional packages are absent.

The actual cuDNN numerical matrix is blocked, not passed. No vendor output,
LSE, dQ, dK, or dV exists to compare. The benchmark declares prospective
acceptance thresholds before any future vendor timing:

- output: `rtol=8e-3`, `atol=4e-3`;
- gradients: relative L2 <= 0.02 and cosine >= 0.999;
- exact sentinel exclusion and nonzero included-key gradient flow.

Those thresholds may reject a future plan; they must not be weakened after
seeing results.

## RTX 5080 B3/T2304 baseline measurements

Command used:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python \
  benchmarks/benchmark_0086c_cudnn_sdpa.py \
  --batch 3 --seq 2304 --dtype bf16 --heads 15 \
  --window 512 --warmup 20 --repeats 50 \
  --compare grouped,0086b,cudnn_regular16,cudnn_kv_split \
  --output-json benchmarks/results/0086c_b3_t2304_fwd_bwd.json \
  --output-csv benchmarks/results/0086c_b3_t2304_fwd_bwd.csv
```

CUDA events were recorded on the current stream. Candidate order alternated on
each repeat. JIT/first execution was excluded from timed samples and recorded
separately. JSON contains every raw device and synchronized wall sample plus
median, p10, p90, standard deviation and median absolute deviation.

| Effective operation | Grouped median | 0086B median | grouped / 0086B |
|---|---:|---:|---:|
| local forward | 2.041 ms | 1.969 ms | 1.037x |
| local backward | 3.899 ms | 3.730 ms | 1.045x |
| local F+B in one call | 5.486 ms | 5.493 ms | 0.999x |
| full-head F+B | 0.938 ms | 0.927 ms | 1.012x; both use grouped fallback |
| full+local composite forward | 2.549 ms | 2.378 ms | 1.072x |
| full+local composite backward | 4.497 ms | 4.574 ms | 0.983x |
| **complete full+local composite F+B** | **6.431 ms** | **6.509 ms** | **0.988x** |

0086B was about **1.2% slower** for the complete composite F+B measurement,
despite the isolated forward improvement. This is consistent with the prior
0086B conclusion and is not evidence for production adoption.

Observed 0086B-versus-grouped complete-composite numerical metrics:

| Tensor | Max abs | Relative L2 | Cosine |
|---|---:|---:|---:|
| output | 0.007198 | 0.002550 | 0.9999969 |
| dQ | 0.000538 | 0.002239 | 0.9999976 |
| dK | 0.000191 | 0.002169 | 0.9999976 |
| dV | 0.003906 | 0.002140 | 0.9999977 |

The benchmark used identical resident Q/K/V/dO and seeds. Differences reflect
the fused forward's BF16 reduction path and the backward cache/recompute path;
they are reported, not treated as cuDNN evidence.

Both candidates showed the same process-level steady-state snapshot after a
live complete result: 402,503,680 pool-used bytes and 880,815,616 pool-reserved
bytes. This is not a transient peak measurement and the JSON labels it as such.
cuDNN workspace and plan memory are unavailable.

## Why the requested shape matrix stopped here

The missing frontend is a hard blocker before graph construction, mask
validation, or plan discovery. Running B1/B2/B3, T1024/2048/4096, 13/9/scattered
baseline-only sweeps cannot answer whether cuDNN supports or accelerates those
shapes. 0086C therefore records the priority B3/T2304 baseline and stops rather
than producing a large matrix with no vendor candidate. Once a compatible
frontend is available in an isolated environment, the same CLI can be invoked
per shape after a version-specific adapter and mask sentinel are validated.

## Artifacts

- `benchmarks/cudnn_sdpa_0086c.py`
- `benchmarks/benchmark_0086c_cudnn_sdpa.py`
- `tests/test_0086c_cudnn_sdpa.py`
- `benchmarks/results/0086c_b3_t2304_fwd_bwd.json`
- `benchmarks/results/0086c_b3_t2304_fwd_bwd.csv`
- `docs/0086C_CUDNN_SDPA_FWD_BWD_BENCHMARK.md`
- `PATCH_0086C_CUDNN_SDPA_FWD_BWD_BENCHMARK.patch`

## Reproduction commands

Capability-only smoke, with no training:

```bash
.venv/bin/python benchmarks/benchmark_0086c_cudnn_sdpa.py \
  --probe-only \
  --compare grouped,0086b,cudnn_regular16,cudnn_kv_split
```

Focused CPU correctness tests:

```bash
MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest -q \
  tests/test_0086c_cudnn_sdpa.py -vv
```

Focused existing GPU regression:

```bash
MINI_LLM_BACKEND=cupy .venv/bin/python -m pytest -q \
  tests/test_0086b_fused_local_forward.py -vv
```

Patch validation and application from the repository root:

```bash
git apply --check PATCH_0086C_CUDNN_SDPA_FWD_BWD_BENCHMARK.patch
git apply PATCH_0086C_CUDNN_SDPA_FWD_BWD_BENCHMARK.patch
```

## Interpretation and next useful experiment

No Amdahl step-time claim can be made: cuDNN saved time is unmeasured and the
real attention fraction of a training step was not measured in 0086C. The
20–30% complete-attention F+B screening threshold was not reached or missed;
it was **not testable**.

The single evidence-producing next step, outside this patch, is to create an
isolated environment containing a Python-3.12/CUDA-13/SM120-compatible NVIDIA
cuDNN Frontend and backend, then implement against that installed version's
actual examples/signatures. The first gate must be the D64 W512 forward and
backward `j=7/8` sentinel. Only after it passes should regular16 and KV-split
plans be timed. No production 0086D backend should be started before those
measurements show a material complete-F+B win.
