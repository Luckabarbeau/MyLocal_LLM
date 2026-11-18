# Persistent-state validation — short-text gate passed, full equivalence unclaimed

## Scope

Reference: pinned no-prefetch CPU build, context32, one sequence/thread, F32 KV, lazy PLE and no repacking. Production remains its original F32 path. Identical fixed IDs `[1,2,3,4,5]` were consumed sequentially in separate processes. Token5 is an explicit sentinel exposing the recurrent/convolution **pre-state after the four-token prefix**; this is not an independently sampled continuation.

Sampling targets:

- Layer0 DeltaNet matrix state before each token.
- Layer0 and layer1 raw recurrent convolution history.
- Layer1 normalized PLE convolution history and selected n-gram row IDs.
- Layer3 current rotated K, V and raw indexer key used to populate QSA caches.
- Logits at prefixes of length1,2,4.

The QSA capture is of verified cache-input operations, not a complete persistent KV-buffer export. Whole-model state equality cannot be inferred from sampled layers.

## What ran successfully

The initial C++ state harness compiled and both five-token model processes completed:

| Process | Seconds | Sampled peak RSS bytes |
|---|---:|---:|
| Independent reference |3.72 |7,467,274,240 |
| Production, including local repeat/prefill check |82.28 |10,646,818,816 |

Both were below the sampled12GiB guard. Raw copied vectors and pointer/dimension/stride metadata remain in `diagnostics_0223_v1/`; the initial blocked report is `0223_state_evolution_v1.json`. No full expert banks or PLE table were expanded.

The production child then created a fresh request and prefills fixed IDs `[1,2,3,4]`. Its final logits and sampled matrix/conv/PLE states match the corresponding incremental results **bit for bit**. Evidence: `diagnostics_0223_v1/production/self_consistency.json`. This establishes local sequential-prefill/reset self-consistency, not independent state equivalence.

## Layout contracts

- GGML recurrent state `ne=[128,128,48,1]` is value-by-key physically. The local logical state is `[1,48,key,value]`; transpose the final two axes. This convention is independently exercised by the existing compiled recurrence tests and the new diagnostic layout test.
- GGML convolution history `ne=[3,10240,1,1]` is time-fastest within each channel; convert to local `[1,3,10240]` by transposing `[10240,3]`, not by reshaping directly.
- PLE history `ne=[9,10240,1,1]` similarly converts to local `[9,4,2560]`.
- Rotated K uses `ne=[256,2,1,1]`. The reference names both raw V projection `ne=[512,1,1,1]` and its head reshape `ne=[256,2,1,1]` as `Vcur-3`. Their source identities must prove the reshape association before selecting one boundary. This is a genuine duplicate-name case, distinct from the earlier scheduler suffix aliases.
- Raw indexer key `ne=[128,1,1,1]` maps to local `[128]`.

All values are copied immediately at evaluation. Comparing a reshaped projection with a state cache without checking the graph relationship is forbidden.

## Historical reporting failures (resolved by retained-capture reanalysis)

The initial analysis stopped at **`KV projection layout mismatch`**, rather than silently comparing different shapes. Retained metadata showed the two V nodes and the head reshape's source pointer. A localized correction added pointer-verified alias filtering and a retained-capture reanalysis mode. That correction introduced excess indentation:

```text
scripts/probe_0223_state_evolution.py:91
    ref=vectors/'reference';actual=vectors/'production'
IndentationError: unexpected indent
```

Latest command:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0223_state_evolution.py
```

At that earlier stop point, collection failed and no v2 report was produced. That historical result remains preserved; the current successful correction and reanalysis are described below.

The follow-up task authorized exactly that three-line indentation fix. **4 focused tests now pass** and `--reanalyze` successfully created `0223_state_evolution_v2.json`, reusing every existing capture without re-executing either large model. Pointer-verified V projection/reshape association is retained.

## Current sampled results

All35 expected comparisons are present with checked layouts. All5 PLE selected-index sets match exactly; fresh-state zero recurrence/convolution/PLE histories match exactly. Prefix1/2/4 logit relative errors are0.0616434 /0.0506754 /0.0410662. Layer0 recurrent-state relative errors after the same prefixes are0.00548179 /0.00540416 /0.00500166. The discrepancy does not monotonically grow over this short range.

| Prefix length | Layer0 raw convolution | Layer1 raw convolution | PLE normalized history |
|---|---:|---:|---:|
|1 |0.00314654 |0.00672520 |0.00154468 |
|2 |0.00409740 |0.00814450 |0.00318735 |
|4 |0.00415664 |0.01122788 |0.00384177 |

These are relative-L2 differences between different arithmetic trajectories, not failed equality assertions. For the fourth input token, current layer3 K/V/raw-indexer values differ by0.0328451 /0.0341913 /0.0211221; their inputs already differ. They are not complete cache-buffer exports. Local four-token prefill/incremental logits and sampled states remain bit-exact. No state-layout, PLE-hash or reset defect was demonstrated. The subsequent identical-input and continuous-state experiments below separate representative state-update equations from projection rounding.

The isolated MoE and perturbation experiments (`0224_isolated_moe_v1.json`, `0225_rounding_sensitivity_v1.json`) show that tiny activation differences can produce percent-level differences under quantized arithmetic. This makes state-path differences plausible but does not by itself prove every state update correct.

## Independent block-state reference over 16 tokens — 0226

`0226_identical_block_states_v2.json` and `0226_continuous_block_states_v1.json` each pass236 comparisons over IDs1–16. The actual reference graph executes decoder prefix0–3, stopping after QSA layer3; it does not provide unfinished full-model logits. Captured projection operands isolate state equations from quantized projection arithmetic.

Two policies are validated: identical reference pre-state at each step, and continuous local state from zero without per-step resets. The continuous report's explicit `state_policy` is authoritative; its historical scope string still mentions reference pre-states. Maximum relative-L2 error for continuous DeltaNet matrix state is8.57284e-7, normalized/gated output2.21826e-7, QSA gated output3.26062e-7, rotated K1.08275e-7, completed indexer pools8.46406e-8 and indexer scores1.82212e-7. Identical-input GDN/QSA output projection errors are1.32276e-6/1.99285e-6. Shifts, V copies and raw indexer keys match exactly where expected.

Causal selected cells match at each position, accounting for reference padded/masked physical cache entries versus contiguous local text storage. All16 causal cells fit the2051 selection width. This is not an independent long-context sparse-selection test. The initial diagnostic API failure is preserved in `0226_identical_block_states_v1.json`; successful replay reused the reference captures.

## Whole-model active-state self-consistency — 0227

`0227_short_sequences_profile_v1.json` passes at prefixes1/2/4/8/16 for identical fixed IDs1–16. Separate sequential CuPy incremental and chunked-prefill processes produce bit-exact logits and identical SHA256 hashes of **all active** DeltaNet matrix state, convolution history, PLE history/predecessors, QSA KV, raw indexer and completed pooled-index state. Unwritten allocated KV capacity is intentionally excluded. Each mode starts with fresh state; earlier0223/0210 tests separately cover same-process reset.

Both processes use FP32 compute/state, four native expert workers, max_length16, no GPU expert residents,120-second deadlines and a sampled12GiB RSS guard. Elapsed times67.99/65.30s; sampled peak RSS10,369,241,088/10,391,388,160 bytes. CUDA runs omit CPU-style virtual-address caps; runtime VRAM planning remains bounded. No expert bank or whole PLE table is expanded.

## Remaining limits and decision

- Full independent16-token model logits and all-layer persistent-state equivalence are not established; independent16-token evidence covers representative prefix blocks, while whole-model evidence is local self-consistency.
- No element-by-element attribution of every later state/logit discrepancy to quantization is claimed.
- Long contexts, multimodal positions and arbitrary token sequences require separate validation.

No mathematical defect was exposed. The requested short-text state gate is passed, permitting measured performance work while preserving these tests and FP32/reference oracles. No production mathematical defect was corrected during that gate.

## Optimized execution lifecycle — 0233

After adding switchable packed GPU and fused FP32 execution, `0233_fast_lifecycle_v1.json` passes16-token incremental/prefill equality for logits and SHA256 hashes of all active persistent state. Each run starts from fresh state; two subsequent independent greedy requests both reproduce `[220,100561,198]`. This uses eager bounded cache admission and QSA/HC/router/sigmoid-GDN fusion without changing model equations.

The new three-launch sigmoid-GDN kernel is additionally checked against retained0226 reference operands and continuous portable-state replay; convolution history matches exactly and measured16-token memory relative error versus portable is1.20e-7. Fused short-context QSA preserves cache/pooling state and explicitly falls back to reference selection beyond the all-causal-cell window or when diagnostics are requested.

These optimized-path checks do not supersede the independent-reference limits above. No claim of full independent16-token model equivalence or arbitrary long-context certification is made.

## Follow-up0238 optimized lifecycle

`0233_fast_lifecycle_v2.json` repeats the gate using read-only global codebooks, warp-per-row IQ kernels, grouping after admission and fixed slots.16-token incremental/prefill logits and SHA256 hashes of **all active** state are identical. Two fresh greedy requests both return `[220,100561,198]`. Dynamic-warp and fixed-slot forced-prefix artifacts also match each other bit-for-bit at1/2/4/8/16, establishing that slot ownership does not alter state trajectories.

This run keeps HC graph capture disabled, uses FP32 state/activations and full top10, and has no speculative decoding. It retains2,332/2,419 slots, no quarantines, sampled RSS11,180,748,800 bytes and pool12,270,584,320 reserved bytes. Context remains16; no long-context/reference equivalence is inferred.
