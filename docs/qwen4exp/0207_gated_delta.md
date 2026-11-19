# Patch0207 — Qwen4Exp Gated DeltaNet operator/adapter

## Scope

New operator-only adapter: `mini_llm/external/qwen4exp_mixer.py`.
New synthetic tests: `tests/test_0207_qwen4exp_mixer.py`.
No model loading, GGUF payload access, decoder, residency policy, parent status
updates, or existing-file edits are included.

The actual generic implementation is `mini_llm/ops/gated_delta_inference.py`
(not `gated_delta_mixer_inference.py`). It already supports
`gate_activation="sigmoid"`; no optional edit was necessary. Qwen3-Next and the
generic default remain SiLU-Z. Existing fused decode eligibility explicitly
rejects non-SiLU gates, so this adapter uses the portable graph rather than
silently running the SiLU-only decode kernel.

## Source contract

Inspected llama.cpp HEAD:
`171e8846b4af9766c354064cb776cb34a50f053f`, at
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp`.

Relevant sources:

- `src/models/qwen4exp.cpp`: `build_qkvz`, `build_norm_gated`,
  `build_layer_attn_linear` (separate beta/alpha projections, sigmoid Z).
- `src/models/delta-net-base.cpp`: autoregressive, chunking and fused recurrence.
- `src/models/models.h`: `build_gdn_l2_norm` uses RMSNorm with epsilon/D,
  then scale 1/sqrt(D), equivalent to x/sqrt(sum(x*x)+epsilon).
- `conversion/qwen.py`: `_LinearAttentionVReorderBase`, inherited by
  `conversion/qwen4exp.py`, converts grouped value heads to tiled order.
- Repository `docs/qwen4exp/reference_contract.md`, Gated DeltaNet section.

`Qwen4ExpMixerSpec.from_metadata` reads exact `qwen4exp` metadata keys without
payload access. Observed geometry is width 2560, Hk=16, Hv=48, D=128, K=4,
6144 value channels and 10240 QKV convolution channels. Smaller valid geometry
is allowed for isolated numerical tests. Key/value head dimensions are equal,
as in the pinned graph. The metadata RMS epsilon is used for **both** Q/K L2
normalization and output RMSNorm.

For each token:

```text
Qraw, Kraw, V = split(SiLU(causal_depthwise_conv(project_qkv(x))))
Q = Qraw / sqrt(sum(Qraw**2) + eps)
K = Kraw / sqrt(sum(Kraw**2) + eps)
beta = sigmoid(project_beta(x))
g = A * softplus(project_alpha(x) + dt_bias)   # A already -exp(A_log)
Mdecay = exp(g) * Mprev
correction = beta * (V - K @ Mdecay)
M = Mdecay + outer(K, correction)             # physical KEY-by-VALUE
read = (Q / sqrt(D)) @ M
y = project_out(RMSNorm_gamma(read) * sigmoid(project_z(x)))
```

Beta and softplus use logaddexp-based stable transforms. The existing output
sigmoid may overflow its internal exp at extreme negative Z, yielding the
correct saturated zero; warning-free evaluation under `np.errstate(over="raise")`
is not promised. Extreme-coefficient tests allow that overflow and compare
finite results to the float64 graph. Gamma is already converted and is applied
directly, without adding one. A must be finite and negative and is never
exponentiated again.

### Head order

All projection inputs are **already converted**. Q/K channels are global
`[all Q | all K | all V]`. Converted tiled value head h shares key head
`h % Hk`, not `h // (Hv/Hk)`. For Hk=2, Hv=6, the grouped-to-tiled head
permutation is `[0,3,1,4,2,5]`. The conversion must consistently cover V rows,
V convolution channels, Z rows, beta/alpha rows, A, dt and output columns.
The adapter does not reorder or materialize projection weights. The independent
grouped-graph test applies this entire converter permutation and verifies
output, convolution history and recurrent matrix.

## API, projection boundary and state

```python
spec = Qwen4ExpMixerSpec.from_metadata(metadata)
mixer = Qwen4ExpRecurrentMixerInference(
    spec, qkv=qkv_projection, z=z_projection,
    beta=beta_projection, alpha=alpha_projection, out=out_projection,
    conv_weight=conv, norm_weight=gamma, ssm_a=negative_a, dt_bias=dt,
)
state = mixer.create_state(batch_size)
y = mixer.forward(hidden_btd, state)
y_one = mixer.step(hidden_bd, state)
state.reset()
```

Each projection is either a backend FP32 output-first matrix `[output,input]`
or a pure callable accepting FP32 `[...,input]` and returning FP32
`[...,output]`. Native packed dispatch can live entirely inside a callable;
no large matrix is converted, copied, decoded or selected by the operator.
Dense projection references are retained. Small A/dt arrays and generic
convolution/norm weights are copied during initialization. Inputs, outputs,
convolution history and matrix state are FP32. There are no persistent token
buffers on the adapter.

The caller owns the existing `DeltaMixerState`:

- `state.conv.history`: `[B,K-1,10240]` at checkpoint geometry, raw projected
  QKV, oldest first (not convolved/activated history).
- `state.delta.memory`: `[B,Hv,Dkey,Dvalue]`, key-by-value.
- `state.conv.length` and `state.delta.length`: equal processed-position counts.
- `state.copy()` makes independent histories; `state.reset()` zeros both arrays
  and both counts. Empty sequence chunks do not advance them.

The float64 oracle uses value-by-key matrices and transposes explicitly when
comparing. GGML's contiguous native memory is likewise interpreted explicitly
as Python `[B,Hv,Dvalue,Dkey]`; it is not copied blindly into generic state.

Callbacks must not mutate hidden/state and must implement the exact linear
projection, including checkpoint ordering. Callback output shape and dtype are
checked. Input-projection failures precede state mutation. An output-projection
failure may leave the explicit state advanced; it propagates without retry.
As with the existing primitive, callers must serialize access to a session's
state and not mutate shared weights during inference.

## Validation — synthetic numerical scope, not full reference parity

Exact successful command, run from the repository root:

```sh
MINI_LLM_BACKEND=numpy QWEN4EXP_LLAMA_SOURCE=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp .venv/bin/python -m pytest -q -s tests/test_0207_qwen4exp_mixer.py tests/test_0106m_gated_delta.py tests/test_0106m_causal_conv.py tests/test_0106m_qwen3next_mixer.py tests/test_0106u_gated_delta_decode.py
```

Result: **206 passed, 3 skipped in 15.17s**. The three skips were in the
existing CPU-run regression selection; the two new pinned-GGML cases ran.
The initial run exposed test-harness issues (strict overflow handling and a
missing standalone GGML packaging template). One localized correction pass
adjusted the harness; production code was not changed after implementation.

Synthetic independent float64 graph comparisons, nonzero initial convolution
and matrix state, batch=2, sequence length=9, K=4, projection input width=7:

| Geometry / comparison | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|
| Hk=2 Hv=2 D=4 output | 6.41120801e-07 | 2.65159908e-07 |
| Hk=2 Hv=6 D=4 output | 9.38700208e-07 | 2.41349669e-07 |
| Hk=16 Hv=48 D=128 output | 1.15209755e-05 | 3.20512027e-07 |
| Hk=16 Hv=48 D=128 final matrix | 2.23251505e-07 | 1.67418853e-07 |
| Hk=16 Hv=48 D=128 final conv history | 4.56977783e-07 | 5.20014253e-08 |
| Hk=16 Hv=48 D=128 sequence vs step output | 2.09808350e-05 | 4.51055494e-07 |
| Grouped source graph vs tiled adapter output | 8.00163064e-07 | 2.31358993e-07 |
| Extreme coefficients output | 1.28032821e-06 | 2.29171708e-07 |

Graph comparisons use `rtol=3e-5, atol=3e-6`; streaming comparisons additionally
check every prefix's matrix/history/counters and use `rtol=2e-5, atol=2e-6`.
The nonzero sequence/step differences come from FP32 projection shapes and
reduction ordering, not different update graphs. Chunking, empty chunks,
reset/replay, independent copies, callback equivalence and callback validation
are covered. The explicit sigmoid-vs-SiLU-Z synthetic output gap is
**2.46787157**, protecting against accidental Qwen3 gate reuse.

### Actual pinned GGML primitive

The optional test verifies the source SHA, builds shared CPU GGML in pytest's
temporary directory, and executes `ggml_gated_delta_net(..., K=1)` through
ctypes for batch=2, Hk=2, Hv=6, D=128, from nonzero state. A temporary parent
CMake project embeds the unmodified vendor `ggml` subtree: standalone configure
cannot run because this vendor checkout omits `ggml/ggml.pc.in`. No reference
source or repository build configuration was edited; no network is needed.

| Comparison | T=1 max absolute difference | T=7 max absolute difference |
|---|---:|---:|
| GGML raw read vs float64 recurrence | 4.47943631e-09 | 7.83490239e-09 |
| GGML final matrix vs float64 recurrence | 5.69424752e-08 | 4.34575278e-08 |
| Adapter final matrix vs GGML | 5.96046448e-08 | 5.96046448e-08 |
| Adapter output vs float64 full operator | 2.09357965e-06 | 4.29971480e-06 |

The native primitive receives oracle-generated normalized Q/K, V, g and beta
rounded to FP32. Thus it independently validates recurrence, tiled head sharing,
query scaling and final-state layout, **not** native convolution/projection/norm
execution. The full adapter is checked separately against the float64 source
graph. Default test runs without `QWEN4EXP_LLAMA_SOURCE` skip the native build.

Unavailable/not claimed: real-checkpoint GDN outputs, packed projection numerical
parity through this adapter, native full-Qwen4Exp graph execution, GPU validation,
full decoder/model/logits parity or production throughput. The metadata geometry
test is not a real-weight layer test; synthetic full-head geometry still uses
input/output width 7, not checkpoint width 2560.
