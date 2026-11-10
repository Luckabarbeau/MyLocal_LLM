# 0106M — General recurrent primitives and a thin Qwen3-Next mixer adapter

Date: 2026-11-09. This increment implements the recurrent token-mixing part
of the installed Qwen3-Next structure, without building a separate engine or
hard-coding that model into the low-level operators. Existing training and
native model computations are unchanged.

**Verified:** generic CPU operators, independent synthetic FP64 parity, explicit
state ownership/reset/copy, and actual-checkpoint CPU streaming parity for one
recurrent layer. **Pending:** GPU execution and complete decoder/generation.

## 1. Architecture boundary

```text
External Qwen3-Next adapter
  metadata / tensor shapes / projection storage
  grouped BA layout / pretransformed decay parameters
                    |
                    v
General inference operators in mini_llm/ops
  causal depthwise convolution
  sum-based L2 normalization
  gated delta state update
  RMS normalization followed by a configurable gate
                    |
                    v
Caller-owned state; optional GenerationState per-layer slots
```

No model name, GGUF metadata key, checkpoint tensor suffix, or fixed head
count appears inside the new general operators. Key and value dimensions
can differ. The recurrence accepts an explicit arbitrary key-head mapping;
its default is repeat-interleaved sharing. Other architectures can supply a
different mapping, coefficient transforms, channel layout, and gate activation
through their own adapters.

### General components

- `ops/causal_conv_inference.py`
  - `CausalDepthwiseConv1DInference(channels, kernel_size, ...)`.
  - `[C,K]` weights, oldest-to-current cross-correlation taps; no tap reversal.
  - Optional per-channel bias and optional shared SiLU activation.
  - `forward([B,T,C], state)` and `step([B,C], state)`.
  - State stores **raw inputs**, not convolved or activated values.
- `ops/gated_delta_inference.py`
  - `GatedDeltaRuleInference`: already-formed Q/K/V and coefficients.
  - `l2_normalize`: sum-based epsilon, at least FP32 arithmetic.
  - `GatedDeltaMixerInference`: composition over projected QKV, gate and
    control inputs; it does not own model projections.
- `ops/gated_norm_inference.py`
  - `GatedRMSNormInference`: wraps the existing `RMSNormInference` and shared
    activation helpers. Normalizes the last feature axis, uses direct gamma,
    then applies SiLU or sigmoid gating. No implicit `1 + gamma`.

These are inference components, not newly trainable layers: no backward
implementation or optimizer/checkpoint changes are introduced. The portable
NumPy/CuPy graph is a correctness-first path, not a fused recurrent GPU kernel.

## 2. Mathematical contract

The recurrence state is logically `[B,Hv,Dk,Dv]`, with **key-by-value** axes.
For each token and value head, select Q/K using the configured head map:

```text
S_minus = exp(log_decay) * S_old
prediction = key.T @ S_minus
correction = beta * (value - prediction)
S_new = S_minus + outer(key, correction)
output = (query * query_scale).T @ S_new
```

Decay occurs before prediction. The new correction is not decayed again.
Readout uses the updated state. Default query scale is `1/sqrt(Dk)`, applied
exactly once. There is no implicit sigmoid, normalization, clipping, or model
parameter transformation in this recurrence operator.

The independent unit reference instead uses the affine matrix form:

```text
S_new = exp(log_decay) * (I - beta * outer(key,key)) @ S_old
        + beta * outer(key,value)
```

This provides a different computational implementation of the same graph.
Tests include nonzero initial memory, zero write strength, decay-only updates,
custom sharing maps, unequal Dk/Dv and both FP32/FP64 state.

The projected mixer composes:

```text
raw QKV -> causal convolution -> SiLU -> split Q/K/V
Q/K -> L2 normalization with sum epsilon
Q/K/V + coefficients -> delta recurrence
readout -> per-value-head RMSNorm -> activated output gate
```

L2 normalization is `x / sqrt(sum(x*x) + eps)`, not mean-based RMSNorm. Its
RMSNorm equivalent would use epsilon `eps / width` and a final `1/sqrt(width)`.
The recurrent operator expects normalization to have already happened; the
composition normalizes it once, not twice.

## 3. Explicit state ownership

Operators retain weights and immutable configuration, not hidden conversation
state. Callers create and pass state explicitly. States support:

- In-place `reset()` preserving buffer identity.
- `copy()` for independent session forks.
- Empty chunks without state changes.
- Position counts and metadata validation before updates.
- Bounded storage independent of sequence length.

`GenerationState` gains an optional `recurrent_states` dictionary, appended to
its dataclass fields. Its `reset()` resets the attached operator states without
replacing their buffers. Native KV-only callers leave this dictionary empty;
existing positional fields and KV reset behavior are preserved.

This is an integration hook, not automatic Qwen decoder wiring. Callers still
need to attach the appropriate per-layer states when the full model adapter
is implemented. Padding masks, variable-length batch-row resets, complete
session cloning integration and speculative-state transactions are not added.
States require exclusive caller ownership while executing.

## 4. Verified Qwen3-Next contract

`external/qwen3next_mixer.py` contains the model-specific mapping:

- `Qwen3NextMixerSpec.from_index(index)` derives dimensions from exact metadata.
- `Qwen3NextRecurrentMixerInference.from_index(index, layer, ...)` validates all
  required shapes/types before accessing payloads, rejects full-attention
  layers, and preserves the owning index.
- Packed projections remain packed; CPU execution can use the existing native
  kernel. Device copies are lazy in the code, but are **not GPU-validated** in
  this increment. Dense F32/F16/BF16 projections load only when first used.
- Small convolution/norm/control tensors are loaded explicitly.
- The first adapter contract is **FP32 hidden inputs only**. It rejects other
  hidden dtypes rather than silently claiming low-precision model parity.

Actual installed-file dimensions:

| Quantity | Metadata / value |
|---|---|
| D / layers | 2048 / 48 |
| Key heads | `ssm.group_count` = 16 |
| Value heads | `ssm.time_step_rank` = 32 |
| Key dimension | `ssm.state_size` = 128 |
| Value dimension | `ssm.inner_size / ssm.time_step_rank` = 4096 / 32 = 128 |
| Convolution width | `ssm.conv_kernel` = 4 |
| QKV channels | 8192 |
| Full-attention interval | 4; recurrent layers exclude 3,7,...,47 |

The installed optimized GGUF stores global contiguous Q/K/V sections in
`attn_qkv.weight`; `attn_gate.weight` supplies the separate recurrent output
gate. The BA projection is **group-interleaved**:

```text
[b0,b1,a0,a1, b2,b3,a2,a3, ...]  # actual ratio=2
```

The adapter reshapes `[Hk,2*ratio]`, separates the beta and alpha portions of
each group, then flattens to value heads. It computes:

```text
beta = sigmoid(b)
log_decay = ssm_a * softplus(a + dt_bias)
```

`ssm_a` is already `-exp(A_log)` in this GGUF conversion path. It is not
exponentiated or negated again. Beta and softplus use stable backend
`logaddexp` formulas. Output normalization precedes **SiLU**, not sigmoid-only,
gating. Stored `ssm_norm.weight` is direct gamma. Value head h uses Q/K head
`h // ratio`, not `h % Hk`.

These choices are adapter semantics. The inspected Strata default Flash-Next
path has different head ordering/gating and must not be copied into this GGUF
adapter. Likewise, legacy combined-QKVZ GGUF layouts are not silently accepted
by this split-projection mapper.

### Source references

- Official Transformers revision:
  `86fdf1595ab819a7baa1d06f2902d00aa94feeb6`.
  `src/transformers/models/qwen3_next/modeling_qwen3_next.py`:
  recurrence 531–575; gated norm 58–74; projection/layout and mixer 628–768.
  Source SHA-256:
  `015adb95dd1de62c23af10a635ebe5b53e72e8f4cdf2b1c4d04bfcacb6254b88`.
  Reviewed source only; Transformers/PyTorch inference was not installed or run.
- Local llama.cpp source revision:
  `171e8846b4af9766c354064cb776cb34a50f053f`.
  `conversion/qwen.py:394–432`, `src/models/qwen3next.cpp:314–556`,
  `src/models/models.h:14–17`, and CUDA recurrence/convolution sources.
  The exact historical converter revision of the GGUF itself is not established.
- Previously recorded Strata source revision:
  `504b143342a9fe1aa8320a7b31e11f5c0ac265c3`; native GDN and preprocessing
  sources were reviewed to distinguish its model-specific wiring.

## 5. Validation

New test files:

- `tests/test_0106m_causal_conv.py`: 78 convolution/state/API cases.
- `tests/test_0106m_gated_delta.py`: recurrence, L2/gated normalization,
  composition validation, and generation-state reset.
- `tests/test_0106m_qwen3next_mixer.py`: independent FP64 full mixer graph,
  projection laziness, packed dispatch and metadata validation.

The synthetic adapter fixture uses **ratio 3 and Dk != Dv**. It demonstrably
detects wrong BA layouts, reversed convolution taps, a second A transformation,
and an incorrect norm-weight offset; passing is not caused by an insensitive
all-zero fixture.

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106m_gated_delta.py tests/test_0106m_causal_conv.py \
  tests/test_0106m_qwen3next_mixer.py
# 159 passed

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_0106*.py tests/test_0085b_external_llama.py \
  tests/test_0085c_external_safetensors.py tests/test_0085f_external_gguf.py \
  tests/test_0085g_gguf_tokenizer.py tests/test_moe_experts.py \
  tests/test_moe_router.py tests/test_topk.py tests/test_generation_session_0103e.py \
  tests/test_inference_policy_0103d.py tests/test_inference_rope_base.py \
  tests/test_rmsnorm_inference.py tests/test_native_dual_cuda_graph_decode.py
# 483 passed, 51 skipped in 1.41s
```

No failing mathematical test required a kernel rewrite or tolerance change.
A final packed-empty-chunk check exposed an existing CPU GEMV wrapper mismatch:
the C++ entry point rejects tokens=0. A localized guard in
`ops/native_packed_cpu.py` now returns the correctly shaped/dtyped empty array
without calling C++; nonempty execution is unchanged. The new regression
checks that no native call occurs and warmed mixer state remains unchanged.
The exact reproduction and the full selections above passed after this guard.

### Reproducible CPU smoke checks

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106m_recurrent_mixer.py \
  --output benchmarks/results/0106m_synthetic_mixer_cpu.json

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_0106m_recurrent_mixer.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --layer 0 --tokens 4 --cpu-backend native \
  --output benchmarks/results/0106m_real_qwen_mixer_cpu.json
```

| Check | Synthetic mixer | Actual checkpoint, layer 0 |
|---|---:|---:|
| Prefix / tested tokens | 3 / 16 | 3 / 4 |
| Batch | 2 | 1 |
| Maximum output error vs independent FP64 graph | 3.21217e-7 | Not run |
| Maximum chunk output error vs sequence | 8.94070e-8 | 0 |
| Maximum step output error vs sequence | 1.93715e-7 | 0 |
| Maximum step recurrent-memory error vs sequence | 6.51926e-9 | 0 |
| Total state bytes | 864 | 2,195,456 |
| Packed projection bytes retained | 0 (tiny F32 fixture) | 35,725,312 |

Both cases start from a **nonzero** warmed state. All comparisons pass
`rtol=3e-4, atol=5e-6`. Actual checkpoint outputs, raw convolution history and
recurrent memory match exactly between sequence/chunk/step for this sample.
This is streaming consistency, **not independent full-model/HF logits parity**.
Only one mixer's small core weights and packed projections are accessed;
no routed expert bank or whole model is expanded.

The real four-token sequence took approximately 50.8 ms in this CPU smoke.
That is not a controlled benchmark: a user Strata inference process was active,
and no full decoder, attention or MoE was measured. No throughput improvement
or model tokens/second is claimed.

### GPU boundary

At the initial resource check, the existing Strata process used **14,210 MiB**
of GPU memory. It was left running. All execution in this increment explicitly
used `MINI_LLM_BACKEND=numpy`. There were no CUDA allocations, GPU tests,
new release-binary executions, or model downloads. Generic code uses the
backend array API, but GPU correctness/performance remains pending.

## 6. Remaining architecture work

`full_model_decode_available` remains false. The index report now describes
recurrent **decoder integration**, rather than an absent recurrent primitive,
as the remaining first blocker.

Next priorities:

1. Validate this portable mixer on GPU when resources are idle; then consider
   fused kernels, not before correctness.
2. Generalize the full-attention projection/head geometry and partial RoPE path
   needed by Qwen, preserving existing native attention defaults.
3. Wire norms/residuals, recurrent versus attention layers, routed/shared MoE,
   output head, per-layer state creation and generation into the existing backend.
4. Add tokenizer/prompt/stop handling and full-prefill versus cached-logit parity.

The requested Qwen3.8 IQ1_M target still needs its own verified metadata and
mapping. This adapter was established for the installed Qwen3-Coder-Next GGUF;
it does not infer the target's architecture from its name.
