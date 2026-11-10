# 0106N — General projected attention, partial RoPE, and bounded CUDA validation

Date: 2026-11-10.

**Verified:** both Qwen3-Next token-mixer branches execute on the existing
NumPy/CuPy backend. This increment adds generic projected grouped-query attention
and partial rotary embedding, maps the installed GGUF full-attention branch,
and validates the preceding recurrent implementation on the RTX 5080.
**Not yet implemented:** complete Qwen decoder blocks or autoregressive generation.

This report supersedes 0106M's GPU-pending status, without altering that historical
validation record. No training, optimizer, checkpoint, or serving API is replaced.
The requested Qwen3.8-Flash-Next GSQ-RCO IQ1_M target is still not identified from
its actual metadata; the available fixture is Qwen3-Coder-Next UD-Q4_K_XL.

## 1. General operators, thin architecture adapter

### `mini_llm/ops/rotary_inference.py`

`RotaryEmbeddingInference(rotary_dim, *, base=10000, layout="interleaved")`
accepts `[B,T,H,D]` inputs and rotates only the requested even prefix. Both
interleaved and half/NEOX pairing are supported. The suffix is copied exactly.
`forward(x, start_pos=0, position_ids=None)` accepts implicit host offsets,
explicit `[T]` or `[B,T]` positions, or existing prepared position factors.

It reuses the repository's RoPE helpers, whose angles/trigonometry are FP32 even
for FP64 inputs. Rotation arithmetic and factor storage follow input dtype.
Tests account for that precision contract rather than changing tolerances.
Implicit table capacity grows geometrically per dtype; it does not perform a
device-scalar position query on each decode step. Raw explicit positions retain
the existing validation reduction; prepared positions avoid repeating it.
Existing training RoPE is unchanged.

### `mini_llm/ops/projected_attention_inference.py`

`ProjectedGQAAttentionInference(n_q_heads, n_kv_heads, key_dim, value_dim=None,
*, dtype="float32", scale=None)` consumes already-projected Q/K/V.

- Q: `[B,T,Hq,Dk]`; K/V input: `[B,T,Hkv,Dk/Dv]`.
- Native-head caches: `[B,Hkv,capacity,Dk/Dv]`, never repeated across query heads.
- Model width, projection width, key width and value width are independent.
- Contiguous query-head groups map to one KV head.
- `create_state(B, capacity)` uses existing `GenerationState` storage.
- `forward(q,k,v,state)` appends chronological causal KV entries.
- `copy_state` copies only this operator's KV state, not an entire session.
- Capacity, geometry and dtype checks precede KV writes; empty chunks are no-ops.
- `GenerationState.reset()` preserves cache storage and resets length; stale
  cache contents are not visible after reset.

`grouped_attention_forward` is also reused by the existing native
`GQAAttentionInference._attention`. Its default FP32 softmax, BF16 compute policy,
and caller-selected probability casting preserve the previous native math.
The new FP64 projected operator explicitly requests FP64 softmax. The native
constructor, optimized decode paths, projection layouts and training code are
not generalized or replaced. Each nonempty query must have at least one visible
key; arbitrary all-masked rows are outside this primitive's contract.

### `mini_llm/external/gguf_projection.py`

Shared lazy projection dispatch is used by both external mixer adapters.
Packed host views remain packed; device upload occurs on first use and is cached.
Dense F32/F16/BF16 payloads are materialized once as FP32. Existing recurrent
projection instrumentation is preserved through optional callable injection.

Review exposed an ownership bug: NumPy `asarray` can return the index's original
F32 array, so the attention adapter's read-only flag leaked to its owner. The
regression failed specifically on owner writeability. One localized correction
creates a zero-copy array view with independent flags. The F32/F16/BF16 regression
now passes, with unchanged contents, equations and numerical tolerances.

### `mini_llm/external/qwen3next_attention.py`

`Qwen3NextAttentionInference.from_index(index, layer_id,
cpu_packed_backend="reference")` validates metadata and tensor headers before
payload access. `create_state`, `forward` and `step` use caller-owned native KV.
Input is **already input-normalized FP32 hidden**. The adapter deliberately does
not own block residuals or input/post-attention normalization.

The model-specific adapter handles query/gate unpacking, direct-gamma Q/K
normalization, half-prefix RoPE, output gating and GGUF tensor names. None of
these checkpoint conventions are embedded into the generic attention operator.
Unsupported scaling, projection layouts and formats fail explicitly. It does
not accept a recurrent layer as a full-attention layer.

## 2. Installed GGUF full-attention contract

Model used:
`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`

| Quantity | Verified metadata/layout |
|---|---|
| Model width / layer count | 2048 / 48 |
| Full-attention layers | 3, 7, ..., 47; interval 4 |
| Query / KV heads | 16 / 2 |
| Key / value head width | 256 / 256 |
| Query/context width | 4096, not model width |
| Rotary prefix / base | 64 / 5,000,000 |
| Query+gate weight | `[8192,2048]`, Q8_0 |
| K and V weights | each `[512,2048]`, Q8_0 |
| Output weight | `[2048,4096]`, Q8_0 |
| Q/K gamma | each `[256]`, F32 |

Forward dependency chain:

```text
normalized hidden
  -> packed Q+gate projection -> split [Q_head, gate_head] per query head
  -> packed K/V projections
  -> Q/K direct-gamma RMSNorm over all 256 features
  -> half-RoPE on first 64 features (j paired with j+32)
  -> append full-width K and projected V to native-head KV cache
  -> causal GQA, query head h uses KV head h//8, scale 1/sqrt(256)
  -> context * sigmoid(raw gate)
  -> packed output projection
```

Q/K rows are not given Llama's row-unpermutation transform. Sigmoid gating here
is distinct from the recurrent branch's SiLU gate. Cache chronology remains
independent of explicitly supplied RoPE positions.

## 3. Sources and relationship to Strata

Previously inspected source revisions, retained for reproducibility (not a
claim that these are today's upstream HEAD):

- Transformers: `86fdf1595ab819a7baa1d06f2902d00aa94feeb6`.
  `src/transformers/models/qwen3_next/modeling_qwen3_next.py`, attention 229–303,
  recurrence 531–575, RMSNorm 126–140. Source SHA256:
  `015adb95dd1de62c23af10a635ebe5b53e72e8f4cdf2b1c4d04bfcacb6254b88`.
- Local llama.cpp: `171e8846b4af9766c354064cb776cb34a50f053f`.
  `src/models/qwen3next.cpp` and `conversion/qwen.py:394–432` establish the
  GGUF/half-RoPE layout; no full-attention Llama row permutation.
- Official `Niko1221/Strata`: `504b143342a9fe1aa8320a7b31e11f5c0ac265c3`.
  See `0106i_runtime_audit_and_routing_trace.md` for the hierarchy/scheduler audit.

Transferable Strata techniques remain permanent non-expert backbone/state
residency, packed hot experts, concurrent CPU misses versus selected GPU
promotions, routing-profile-guided placement, and explicit transfer/compute
stream dependencies with in-flight protection. This increment supplies the
missing attention operator boundary needed for that GPU backbone; it does not
claim whole-backbone placement, GPU-side routing, or a Strata throughput match.
Strata's default Flash-Next head ordering/gating is not a substitute for this
GGUF's contract. No release binary or external inference runtime was used.

## 4. Validation and resource discipline

Initial GPU check: RTX 5080, 16,303 MiB reported, about 1,257 MiB in use,
Zed the only reported compute process. No Strata process was running. No
existing process was stopped. CUDA smokes use a private **256 MiB** CuPy pool
and header-based preflight estimates, not a fraction of currently free VRAM.
The pool does not include CUDA context/library allocations or host memory.

| Latest selection | Result |
|---|---|
| Broad NumPy external/MoE/generation regression | 664 passed, 72 skipped |
| N recurrent + attention + projected-core CuPy tests | 38 passed |
| Existing packed-kernel / async GPU selection | 51 passed, 56 deselected |
| New dense-owner flags regression | 3 passed, 61 deselected |
| Actual recurrent layer 0 CUDA smoke | passed |
| Actual full-attention layer 3 CUDA smoke | passed |

Exact commands, results and tolerances are in `0106n_validation.json`.
The GPU selection covers Q3_K/Q4_K/Q5_K/Q6_K and the other existing packed
formats exercised by its parameterization, fused gate/up, residency and async
mixed CPU/GPU buffer ownership. It does not implement IQ formats.

Synthetic tests use independent FP64 attention/recurrence oracles with matched
input/storage precision, unequal dimensions, nonzero initial state, sequence /
chunk / step equivalence, causality, head mapping, reset/fork ownership, empty
chunks, lazy reads and default/nondefault streams. CPU-only fixture modules
remain CPU-only: NumPy references are exported in isolated subprocesses to
pickle-free NPZ files. The process-global backend is never switched in place.

### Real-checkpoint CPU/GPU parity

| Metric | Recurrent layer 0 | Attention layer 3 |
|---|---:|---:|
| Maximum output absolute error across CPU comparisons | 2.793968e-9 | 7.152557e-7 |
| Maximum error including state/cache checks | 2.980232e-7 | 1.668930e-6 |
| Maximum normalized tolerance ratio over all comparisons | 0.030762 | 0.171341 |
| Peak private pool reserved bytes | 46,500,352 | 29,682,688 |

Tolerance ratio is `abs(actual-reference)/(atol+rtol*abs(reference))`; ≤1 passes.
Recurrent uses rtol=3e-4, atol=5e-6. Attention output uses rtol=3e-5, atol=6e-6;
KV uses rtol=2e-5, atol=3e-6. Ownership/exact-reset checks use zero tolerance.
All arithmetic tolerances and kernel equations are preserved.

### Observed CUDA event timings, not throughput claims

| Smoke operation | Event milliseconds |
|---|---:|
| Recurrent full sequence | 5.015 |
| Recurrent chunks (includes empty chunk) | 9.356 |
| Recurrent token steps | 5.861 |
| Recurrent repeated chunks | 3.037 |
| Attention cold prefix | 317.375 |
| Attention continuation full sequence | 2.944 |
| Attention continuation chunks | 2.056 |
| Attention continuation token steps | 5.015 |

`--tokens 4` specifies a recurrent 4-token input **repeated twice** from a
nonzero 3-token-prefix state: each timed recurrent mode processes 8 tokens.
Attention measures a 4-token continuation after a 2-token prefix; the cold-prefix
measurement includes lazy first-use work. The modes run in a fixed order, not
randomized repeated trials. Host dispatch and synchronization affect these
measurements; do not compare them as kernel speedups, full-model tokens/s,
CPU/GPU throughput, or Strata performance. Per-comparison diagnostics and both
CUDA-event and synchronized wall timings are saved under ignored
`benchmarks/results/0106n_real_{recurrent,attention}_gpu.json`.

## 5. Remaining blockers and next increment

`Qwen3NextGGUFIndex.report()` still sets `full_model_decode_available=False`.
Its blocker descriptions now distinguish absent decoder wiring from the
implemented attention/recurrent primitives.

Next correctness-first increment:

1. Compose existing RMSNorm, residual additions and the two token-mixer adapters
   into hybrid decoder blocks, without adding model-specific math to generic ops.
2. Wire shared gated FFN + routed experts through the existing executor, preserving
   exact expert IDs, router normalization/weighting and protected allocations.
3. Integrate embeddings, final norm, output head, tokenizer and existing generation
   state/session interfaces; establish complete logits and incremental parity.

Still separate: exact requested pruned-model metadata/remapping and n-gram
semantics; IQ1_M's actual mixed quantization formats; full backbone/working-state
VRAM planning; complete-model trace/throughput benchmarks; sparse n-gram reads
and optional SSD tiers. This increment neither loads the entire model nor claims
that the installed 512-expert model represents the requested 256-expert target.

All changes remain incremental and uncommitted. Existing user staging is retained.
