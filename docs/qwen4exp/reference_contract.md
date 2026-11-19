# Patch 0200 — Qwen4Exp reference contract

## Scope and provenance

This audit describes the **local, original two-shard** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` checkpoint, not a pruned surrogate. No model weights were downloaded or modified. Tensor headers were inspected before implementation. Subsequent Patches0205–0210 assembled and smoke-tested the graph; independent full-model numerical validation remains blocked (see implementation_status.md). Full metadata (including tokenizer arrays and template) is in `gguf_metadata.json`; every tensor and byte offset is in `tensor_inventory.json`.

Pinned source revisions, inspected before implementation:

| Reference | Exact commit | Source scope |
|---|---|---|
| Strata | `c77572806996349b49995e4f98d41b5294d36a21` | All requested cache/source/pool/IQ/prefill/layer/QSA/PLE/quant inventory and DETAILS/MODELS/QUANTS files |
| Radiance | `a555ef392a48fbdac3df4a0eeb1197ec3f6e6100` | `spec.md`, planner/heat/mover/xprefetch, HC wrapper, MOE-W4/OPS; targeted implementations to verify lifecycle |
| llama.cpp | `171e8846b4af9766c354064cb776cb34a50f053f` | `src/models/qwen4exp.cpp`, DeltaNet base, MoE graph, conversion/tensor mapping, GGML blocks/codebooks/reference dequantizers |

All three checkouts were clean when audited. Source roots are respectively `/home/lucka/Documents/my_local_LLM/qwen4exp-references/{Strata,radiance}` and `/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/vendor/llama.cpp`. Strata separately pins GGML `3cf03257f219afbe7334045ff7c6a06ac68c627d`; this is **not** our llama.cpp revision. The pre-existing `/home/lucka/softwares/strata` contained only an installer and was not used as source evidence.

**Checkpoint revision is unknown.** Embedded provenance names `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, but neither shard embeds an immutable model revision. A `.done` sidecar claims shard-1 SHA256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`; payload hashing has not verified this. Header SHA256s identify only headers, not whole weights. Do not equate a source revision, filename or timestamp with a model revision. An official model-definition revision was not independently obtained; converter and checkpoint contracts are used explicitly, not labelled official provenance.

## Geometry and graph order

Observed metadata: 48 blocks, width D=2560, four streams C=4, HC rank R=320, 512 experts with top-10 and routed/shared width 640. Full/QSA layers are zero-based `3,7,...,47`; the other 36 layers are recurrent. PLE is at layer 1. Vocabulary 248320; context metadata 262144. GGML axes are fastest-first; canonical Python axes reverse them. No HF Q/K row unpermutation is appropriate for this Qwen graph.

Exact dependency order from `src/models/qwen4exp.cpp:353–457`:

1. Token embedding, duplicated into four residual streams.
2. At each block: optional PLE update **before** HC attention read; HC read; recurrent or QSA mixer; paired HC write; separate HC FFN read; routed plus gated shared FFN; paired HC write.
3. Final HC read/collapse; output projection. There is **no separate conventional output RMSNorm** in this graph.

The four-stream activation is per token and crosses blocks; it is not cross-token history. Persistent state is separate: recurrent matrix, recurrent conv, PLE normalized-value conv and token predecessors, QSA KV/raw-indexer keys, positions and sequence identity.

## Mathematical contract

All following equations are activation-FP32 reference contracts. Stored quantization is unchanged. RMSNorm epsilon comes from checkpoint metadata. Stored norm gamma already includes converter zero-centering adjustments: never add 1 again.

### Hyper-connections

From llama.cpp `qwen4exp.cpp:267–350`, independently checked against Radiance `arch/common/rad_block_hc.h` and Strata GR kernels. For streams X_c:

```
n_c = RMSNorm_gamma_c(X_c)              # reduce over D, independently per stream
n = concat(n_0,...,n_3)
a = W_up @ SiLU((W_down @ n) / C)
m = sum_c(n_c * sigmoid(a_c)) / C       # per-feature read gates
injection_c = 2 * sigmoid((W_inject @ n)_c / C)
X'_c = X_c + injection_c * block(m)     # scalar gain per stream; original residual
```

Read and write must share their own gains, not another subblock's. Final collapse uses read only. Three independent `/C` factors (before SiLU, before injection sigmoid, stream mean) are required. Radiance raw gamma handling (`wadd`) must not be copied onto already-converted GGUF gamma. Its BF16/FP8 storage boundaries are not our FP32 oracle.

### PLE

From `qwen4exp.cpp:1038–1126,1185–1293`, checked against Strata `ple.hpp` and n-gram implementation. N=3; eight heads per n-gram length; 16 concatenated rows of width160 = D. Integer multipliers, row offsets and modulus values are retained exactly in metadata.

```
h_n = XOR_j(uint64(token[p-j]) * uint64(multiplier[j]))  # wrap multiplication mod 2**64
row_head = h_n % vocab_size_head + offset_head
embedding = concat(selected rows for bigram then trigram heads)
k_c = RMSNorm_gamma_k_c((W_k @ embedding)_c)
q_c = RMSNorm_gamma_q_c(X_c)
v = W_v @ embedding
s_c = dot(k_c, q_c) / sqrt(D)
g_c = sigmoid(sign(s_c) * sqrt(max(abs(s_c), 1e-6)))
u_c = g_c * v
z_c = RMSNorm_gamma_conv_c(u_c)
conv_t = sum_j(w_j * z_[t-(K-1-j)*N])  # K=4, dilation N=3
X'_c = X_c + u_c + SiLU(conv_t)
```

The convolution input is **normalized gated values**, and history length is nine. GGML convolution weights are `[K,CD]`, Python `[CD,K]`; history advancement must occur once per position. Missing predecessors are PLE EOS 248044. A predecessor EOS cuts older context; current-token EOS does not erase its predecessor context. This PLE EOS differs from tokenizer generation EOS 248046. PLE image token is 248056. No full-table allocation is allowed.

### Gated DeltaNet

From `qwen4exp.cpp:460–485,861–985` and `delta-net-base.cpp`. Key heads=16, value heads=48, head width=128; convolution K=4 over 10240 QKV channels. Converter already tiles/reorders unequal value-head groups and transforms A_log to **A=-exp(A_log)**.

```
(q_raw,k_raw,v) = SiLU(causal_depthwise_conv(W_qkv @ m))
q = q_raw / sqrt(sum(q_raw**2) + eps)
k = k_raw / sqrt(sum(k_raw**2) + eps)
beta = sigmoid(W_beta @ m)
g = A * softplus(W_alpha @ m + dt_bias)
S_decay = exp(g) * S_prev
correction = beta * (v - S_decay @ k)
S = S_decay + outer(correction,k)       # mathematical value-by-key notation
o = S @ (q / sqrt(128))
y = W_out @ (RMSNorm_gamma(o) * sigmoid(W_z @ m))
```

**Sigmoid Z**, not the Qwen3-Next SiLU output gate. Physical state layouts differ across references; never copy state without layout conversion. Convolution still uses SiLU. Existing tested primitives are reusable only where normalization, gates and state layout are equivalent.

### QSA

From `qwen4exp.cpp:542–858` and hybrid-indexer memory. Q heads=24, KV heads=2, head width256, rotary width64, sections `[11,11,10,0]`, base 1e7. Query projection stores `[query|gate]` **per head**. Q/K receive per-head RMSNorm, then IMRoPE; V does not. Patch0208's additional CPU source inspection established that interleaving describes axis assignment: lane pairs are the two halves of the rotary prefix, not adjacent lanes. For scalar text positions repeated on t/h/w axes, the existing half-layout partial RoPE is mathematically equivalent. Sigmoid output gate precedes final projection.

Indexer has four Q heads of width128 and compression r=4:

```
raw_key_j = W_index_k @ m_j
pooled_b = mean(raw_key_j for j in block_b)  # pool BEFORE norm/RoPE
K_b = RoPE_at_block_start(RMSNorm_gamma_k(pooled_b))
Q_t,h = RoPE_at_t(RMSNorm_gamma_q((W_index_q @ m_t)_h))
score_t,b = sum_h(ReLU(dot(Q_t,h,K_b)))      # no extra learned head weights/scale
```

Expand scores to token/cell scores. Tail begins at `floor((t+1)/4)*4`, receives finite `+1e9`; apply causal/original attention mask. Select **token** top-k with width `min(n_kv,2048+4-1)`, not indivisible block-top-k. Main GQA softmax uses main Q/K scores and both selected and original masks; indexer scores are not probabilities. Patch0208 established a reference disagreement: GGML CPU uses value-only std::partial_sort (no stable lower-index tie contract); Strata specifies lower-index ties. The default ggml_cpu policy reproduces the tested local libstdc++ membership, while the explicit strata policy selects lower-index ties. This is not portable tie parity across STL vendors. Sorting selected cells ascending changes reduction order, not membership. Short-context dense attention is only equivalent when selection retains every allowed cell. All dynamic positions, widths and masks must stay runtime values.

### Router and FFN

From `qwen4exp.cpp:988–1035` and generic MoE graph:

```
p = softmax(W_router @ m)
I = topk(p,10)
w_e = p_e / max(sum(p_j for j in I), 2**-14)
f_e = W_down_e @ (SiLU(W_gate_e @ m) * (W_up_e @ m))
y = sum_e(w_e * f_e) + sigmoid(w_shared @ m) * f_shared(m)
```

Respect any nontrivial metadata expert-weight scale; no selection bias is passed by this graph. Expert outputs are **unweighted**, router weights apply exactly once after down projection. Same normalized HC input feeds shared and routed branches. No expert may disappear on a miss.

## Disagreements and nonportable behavior

- Strata can skip missed routing ranks >=7 without renormalization (`expert_source.cpp`, generation defaults). Disable `STRATA_ROUTE_TAIL_SKIP=0` for full-top-10 parity. This optimization is **not adopted**.
- Strata uses compiled geometry and fixed PLE wiring; ours must validate metadata. Strata can skip unready PLE; ours must fail rather than silently omit a required operation.
- Strata's BF16 HC/projection modes and QSA KV modes change arithmetic. Its incremental pooled-key state and key/value recurrence layout differ physically from llama.cpp.
- Radiance's served W4/FP8 Hadamard pipeline is **not IQ3_S GGUF**. Its fixed-slot/publication contracts are architectural references, not compatible kernels.
- Strata `admit()` publishes a slot table entry before filling; caller synchronization is essential. New code must publish complete bundles only after readiness.
- Reference source agreement alone is not numerical model validation. Subsequent operator tests exercise compiled-source hash, recurrence, text RoPE and tie selection. Multimodal positions remain unvalidated. Subsequent full independent-model logit comparison runs but reports ~5–6% relativeL2 disagreement; its first boundary association is suspect and must be resolved before a production correction.

## Existing-code reuse matrix

| Existing component | Reuse / required separation |
|---|---|
| `qwen3next_gguf.py` | Preserve single-file Qwen3-Next adapter; add generic multi-shard store and separate qwen4exp index |
| `quantized_weight.py`, `native_packed_cpu.*` | Extend per-projection formats, immutable packed views, independent reference decoder |
| `quantized_linear.py`, grouped experts | Reuse old-format packed GPU path; CPU format registration must not imply GPU capability |
| `expert_residency.py`, `experts_inference.py` | Reuse sparse selected dispatch, CPU workers, one-time deterministic combine, byte budget, protected in-flight handles |
| `router_inference.py`, `qwen3next_moe.py` | Reuse softmax/top-k/renormalization and shared SwiGLU where exact; keep model-specific graph separate |
| `qwen3next_mixer.py`, attention operators | Reuse primitives only; new Z gate/QSA/indexer/state contract |
| `qwen3next_decoder.py` | Conventional two residual adds are **not** reusable as HC decoder graph |
| `qwen3next_model.py` | Reuse lifecycle design/ownership guards, not its conventional output normalization graph |
| `serving_api.py`, generation session | Reuse HTTP/streaming/session system later; do not create another server |
| `model_api.py` | Native training facade is not a drop-in external model adapter; no training edits |

## Audit exit and limitations

All 1224 local tensor declarations are accounted for against the pinned qwen4exp loader with no missing names/shape mismatches/undeclared tensors in the recorded audit. Required mathematical operations are identified above; none may be approximated silently. No vision/MTP weights occur. The **tensor/operator compatibility audit is established**, but exact checkpoint revision/payload integrity and official-definition provenance remain unresolved. These are recorded provenance limitations, not a claim of an end-to-end-valid model. The complete original checkpoint now executes on NumPy and CuPy with sparse native CPU experts. Full-model numerical validation is still not established.
