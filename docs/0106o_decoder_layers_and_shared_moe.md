# 0106O — Hybrid decoder layers and shared/routed MoE composition

Date: 2026-11-10.

**Verified:** complete individual Qwen3-Next decoder layers on CPU, a synthetic
four-layer hybrid chain against independent FP64 graphs, and real-checkpoint
sequence/chunk/step parity for layers 0, 3 and 46. **Pending:** new full-layer
CUDA validation and whole-model embeddings/logits/generation integration.

This extends the existing backend. No training or optimizer code, native
inference defaults, serving API, or previous kernel equations are replaced.
All user staging is preserved; no commit or branch was created.

## 1. Source-supported contract

The reference is local llama.cpp revision
`171e8846b4af9766c354064cb776cb34a50f053f`. Its inspected model/graph/conversion
files had no working-tree modifications. Relevant source regions:

- `src/models/qwen3next.cpp:155–193`: decoder norms, branch selection and residuals.
- `src/models/qwen3next.cpp:570–609`: routed SwiGLU, selected-probability
  renormalization, shared SwiGLU and shared sigmoid admission gate.
- `src/llama-graph.cpp:2033–2053,2115–2156,2313–2357`: router probabilities,
  selected-weight normalization, down projection, weighting and aggregation.
- `conversion/qwen.py:394–402`: normalization `+1` conversion, except the
  recurrent gated norm. Stored GGUF gamma is used directly, never offset again.

Prior source pins are retained, not freshly verified upstream HEAD:
Transformers `86fdf1595ab819a7baa1d06f2902d00aa94feeb6`, Qwen source SHA256
`015adb95dd1de62c23af10a635ebe5b53e72e8f4cdf2b1c4d04bfcacb6254b88`;
Strata `504b143342a9fe1aa8320a7b31e11f5c0ac265c3`. No external runtime was
used as the implementation, and no Strata release binary was downloaded/run.

Installed development checkpoint:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`.
Its actual metadata describes 48 layers, width 2048, 512 experts, top-10,
512-wide routed experts and a 512-wide shared FFN. **Do not use the separate
`feed_forward_length=5120` as either individual FFN width.** No remapping,
router-correction bias or routing-scale metadata was present. This is not the
requested 256-expert Qwen3.8 IQ1_M checkpoint.

For already-loaded effective GGUF norm weights:

```text
u = RMSNorm(x, attn_norm)
r = x + token_mixer(u)
z = RMSNorm(r, post_attention_norm)
output = r + routed(z) + gated_shared(z)
```

Each mixer already includes its output projection; the block does not add a
second projection, mixer norm or residual. Both FFN branches consume the same
`z`, and the second residual is `r`, not `z` or original `x`.

```text
p = softmax(router(z), across all E experts)
ids = topk(p)
w = gather(p, ids) / max(sum(gather(p, ids)), 2**-14)
routed = sum_e w_e * down_e(SiLU(gate_e(z)) * up_e(z))

shared_raw = down_s(SiLU(gate_s(z)) * up_s(z))
gated_shared = sigmoid(dot(shared_admission, z)) * shared_raw
```

The existing router ranks logits (monotonically equivalent for finite
nondegenerate probabilities), preserving its existing tie behavior. There is
no approximate routing or replacement of missing expert IDs. The shared
admission gate is one scalar per token, applied after down projection; it is
not an eleventh routed expert and is not jointly renormalized with routed weights.

## 2. Reuse and API boundaries

### General inference-only router option

`mini_llm/ops/router_inference.py` adds keyword-only:

- `renormalize_selected=False` — the historical native default remains unchanged.
- `normalization_floor=1e-30` — positive finite configurable denominator floor.

Normalization, when enabled, happens in softmax compute precision before a
low-precision storage cast. Qwen explicitly opts in with floor `2**-14`.
Training `expert_topk_forward`, backward behavior and native inference calls
are unchanged. Tests check the legacy result with exact array equality for
FP16/FP32/FP64 and top-1/top-2/all-expert cases.

### Thin MoE adapter

`mini_llm/external/qwen3next_moe.py` provides `Qwen3NextMoESpec` and
`Qwen3NextMoEInference.from_index(index, layer_id, *, expert_executor,
cpu_packed_backend="reference")`.

- Header validation precedes this branch's payload reads.
- Reuses `RouterInference`, `ExpertsInference`, `HeterogeneousExpertExecutor`,
  existing SiLU, and the shared lazy GGUF projection helper.
- Routed banks stay packed and are retrieved only for selected expert IDs.
  Both existing separate and fused gate/up bank layouts are accepted.
- Router/admission weights are eager, small non-expert tensors. Shared
  F32/F16/BF16 or supported packed projections are lazy and cached.
- Shared width is explicit metadata, distinct from routed and dense widths.
- The caller supplies and owns the executor; the adapter never closes it.
- FP32 hidden and FP32 executor compute are the initial parity contract.
  Unsupported IQ formats, FFN layouts/biases and explicit alternative routing
  metadata fail rather than silently assuming standard Qwen semantics.

No new cache policy, worker pool or expert execution engine is introduced.
Existing optional routing instrumentation, CPU workers, promotions and in-flight
protection are reached through the same managed dispatcher. CUDA integration of
this complete composition remains untested in this increment.

### Decoder-layer adapter and explicit state

`mini_llm/external/qwen3next_decoder.py` provides:

```python
layer = Qwen3NextDecoderLayerInference.from_index(
    index, layer_id, expert_executor=executor, cpu_packed_backend="native")
state = layer.create_state(batch_size, max_length)
output = layer.forward(hidden, state)
next_output = layer.step(next_hidden, state)
fork = layer.copy_state(state)
state.reset()
```

The caller can share one executor across all layers. The interval metadata
selects the existing recurrent or attention adapter. Norms and residuals reuse
existing primitives; no Qwen branch was added to generic attention/recurrence.

`Qwen3NextLayerState` is a bounded handle around existing `GenerationState` KV
storage or `DeltaMixerState`, with layer ID, batch size and horizon checks. It
is not a replacement top-level generation session. Copies have independent
native storage; reset delegates in place. Empty chunks make no projections or
expert requests. Explicit RoPE IDs apply only to attention layers.

Inputs and state require exclusive ownership while executing. These calls are
not transactional: a runtime failure after mixer execution can advance state;
reset/discard that state before retrying. Cross-layer session orchestration,
variable-length batch masks and speculative state transactions remain future work.

Review found a construction-order issue: incompatible executors were rejected
after reading mixer payloads. A regression reproduced this for both branches.
One localized reorder now rejects executor dtype/dimension errors before those
reads. No numerical operation changed.

## 3. Numerical discrepancy and established verified state

Initial focused tests passed 26 cases before one synthetic chain comparison
failed near cancellation at `[0,1,26]`:

| Item | Recorded value |
|---|---:|
| Runtime, FP32 | -0.019347863271832466 |
| Reference, FP64 | -0.019367861736830555 |
| Absolute error | 1.999846499808943e-5 |
| Relative error | approximately 0.00103256 |
| Requested rtol / atol | 4e-4 / 8e-6 |
| Allowed error there | approximately 1.57471447e-5 |
| Normalized tolerance ratio | 1.269974 |
| Finite-difference step | not applicable |

The reference graph was checked first. A matched-input FP64 oracle for each
individual layer passed, with maximum local tolerance ratios 0.07086, 0.04112,
0.05453 and 0.01361. Only accumulated cross-layer FP32/FP64 differences exceeded
the final near-zero tolerance; rounding just layer boundaries did not cure it.
The reused single-mixer fixture produced residual magnitudes up to 29.3 from
input scale 0.2. This did not establish an incorrect equation or gradient path.

A conditioning experiment scaled **only synthetic mixer-output weights** by
1/16: output magnitude fell to 1.91; full/chunk/step maximum errors were
1.05e-6/1.49e-6/1.23e-6 at the unchanged tolerance. That single localized fixture
correction was retained. Production equations and existing tolerances were not
changed. The entire focused selection then passed.

## 4. Tests and measurements

Latest verified selections:

- **46 passed**: new router, full-layer, hybrid-chain, lifecycle, negative-contract,
  worker-pool, instrumentation and benchmark CLI tests.
- **728 passed, 72 skipped**: broad 0106/external/generation/MoE regression,
  including native unified-routing/top-1 checks. GPU-guarded tests were skipped.
- Real layers **0**, **3**, **46**: native packed CPU sequence/chunk/token-step
  checks all passed. Layer 46 covers BF16 shared/mixer projections and routed
  Q5_K/Q5_K/Q6_K weights.

Synthetic four-layer benchmark, with a nonzero two-token prefix and four-token
continuation, batch two:

- Maximum final output error vs independent FP64: **2.33412e-6** across modes.
- Maximum output chunk/step difference vs sequence: **2.44379e-6**.
- Maximum error including FP64/state/cache checks: **1.37389e-5**;
  maximum tolerance ratio **0.75794**, passing the fixed mixed tolerance.
- Summary/detailed traces preserve exact selected IDs and selected weighting.
  Each measured mode contains 64 routed assignments across four layers.

Real-checkpoint measurements (synthetic hidden inputs, not tokenized prompts):

| Layer | Prefix / continuation | CPU workers | Max output streaming error | Max including state | Sequence / chunks / steps wall ms |
|---|---|---:|---:|---:|---|
| 0 recurrent | 2 / 4 | 0 | 1.19209e-7 | 1.19209e-7 | 281.494 / 172.330 / 175.705 |
| 3 attention | 2 / 4 | 0 | 1.90735e-6 | 1.90735e-6 | 211.274 / 161.295 / 164.362 |
| 46 mixed BF16 | 2 / 2 | 2 | 1.90735e-6 | 3.57628e-6 | 106.958 / 60.115 / 56.638 |

These are uncontrolled CPU correctness-smoke wall timings, fixed mode order,
under concurrent Strata use. They are **not** generation tokens/s, controlled
worker-count comparisons, independent real-model logits parity, or Strata
throughput comparisons. Layer-46's two-token chunk case has one nonempty chunk.
All routing traces report zero uploads in these CPU-only runs.

### Resource discipline

At this turn's check, Strata was again active: GPU utilization 100%, total used
15,497 MiB, Strata compute allocation 14,166 MiB. No process was stopped and no
CUDA allocation/test was requested for 0106O. Prior 0106N CUDA results remain
historical evidence for its primitives, not validation of this new composition.

The benchmark checks a header-based working-set estimate against the configured
512 MiB estimate budget before layer construction. **This is not a hard RSS
limit**: interpreter, GGUF metadata, allocator/compiler/library overhead and OS
page accounting are excluded. Actual Linux peak process RSS is also recorded:

| Run | Header estimate bytes | Actual peak process RSS bytes |
|---|---:|---:|
| Four-layer synthetic | 701,488 | 137,113,600 |
| Real layer 0 | 508,420,864 | 840,159,232 |
| Real layer 3 | 471,787,520 | 767,750,144 |
| Real layer 46 | 439,083,776 | 947,240,960 |

The GGUF is mmap-backed. No complete routed bank/model is dequantized or copied
into RAM. RSS is not model weight size or an isolated expert-cache measurement.

## 5. Reproduction and next step

CPU commands (complete invocations are also in `0106o_validation.json`):

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0106o*.py
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_0106o_decoder_layer.py --cpu-workers 2 --profile detailed --output benchmarks/results/0106o_synthetic_decoder_cpu.json
```

For a real checkpoint, pass `--model`, `--layer`, `--tokens`, and optionally
`--cpu-workers`. The CLI is CPU-only even when invoked from a CuPy-configured
parent process. Detailed mode adds bounded routing IDs/weights and execution
records; summary mode retains aggregate counts/times with less profiling work.

Changed production files: the two new external adapters,
`ops/router_inference.py`, and updated `external/qwen3next_gguf.py` blocker text.
New tests are `test_0106o_router_inference.py`, `test_0106o_qwen3next_decoder.py`,
`test_0106o_decoder_benchmark.py`; the benchmark is
`benchmarks/benchmark_0106o_decoder_layer.py`. Raw result JSON is under ignored
`benchmarks/results/`; the durable validation summary is tracked as an incremental
repository file, without staging/committing user work.

**Next:** model-level orchestration using these layers, embeddings/final norm/
output head, existing tokenizer/generation interfaces, and full prefill versus
incremental/logit parity on a small compatible fixture. Validate the complete
layer on CUDA once GPU resources are free. Keep `full_model_decode_available`
false until full inference actually works. Requested target metadata/remapping,
IQ quantization coverage, n-gram lookup, whole-backbone VRAM budgets and SSD
storage remain separate unresolved requirements.
