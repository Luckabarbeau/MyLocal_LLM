# 0209 — first Qwen4Exp decoder runtime

## Scope

`mini_llm/external/qwen4exp_model.py` composes the existing 0205–0208 inference
operators. `qwen4exp_weights.py` owns only factory-opened multi-shard stores.
No operator, shared loader, training, HTTP or status files are changed.

The source contract is `reference_contract.md`, checked against local llama.cpp
`171e8846b4af9766c354064cb776cb34a50f053f` (`src/models/qwen4exp.cpp`).
Tensor names/shapes/formats were inspected from the actual `tensor_contract.families`
in `tensor_inventory.json` and metadata from `gguf_metadata.json` before coding.
The original model has 48 layers, D=2560, four streams, rank 320, 512 routed
experts/top-10, recurrent layers plus every-fourth-layer QSA, and PLE at layer 1.

## Graph

For each token:

1. Decode its selected IQ4_XS embedding row and duplicate into four streams.
2. At the configured layer, PLE updates streams and returns a new PLE state.
3. Attention HC read → GDN or QSA → paired HC write to original streams.
4. Separate FFN HC read → softmax/top-k router and sparse experts plus
   sigmoid-gated shared SwiGLU → paired HC write.
5. Final HC collapse → output projection. **No output_norm and no gamma +1.**

GDN is wrapped as `[1,1,D]`; QSA consumes a vector and contiguous text position.
HC uses `[...,4,D]`; PLE consumes `[4,D]`. Router selected mass is normalized
with floor `2**-14`; metadata expert scale multiplies routed weights once.
`ExpertsInference` is constructed with `HeterogeneousExpertExecutor(index.provider)`
from the outset: no resident list of 512 dense experts is created. Native expert
compute is FP32; host dequant-cache budget is zero. CPU worker count must be
positive and at most min(32, affinity CPU count). Native packed kernels are
single-threaded per worker; set BLAS thread limits in the launch environment
rather than allowing unrelated library pools to oversubscribe the host.

QSA defaults to 0208 `ggml_cpu` tie membership. This is a reference baseline,
not a fast QSA implementation. Multimodal input, batching, alternate RoPE,
unknown tensors/layouts, grouped routing and non-four-stream HC are rejected.

## Storage and reservations

Header inspection and planning occur before any tensor payload access or operator
construction. Dense F32/F16/BF16 backbone weights are expanded to FP32; the
recorded original inventory totals 3,066,007,040 bytes after expansion. HC
constructors copy matrices; transient inputs are not retained in a host cache.
The plan conservatively reserves constructor temporaries, recurrent/PLE state,
requested-length QSA caches, activation/selection scratch, staging, expert cache
and headroom separately. Factory `max_length` is mandatory. Metadata context
262144 is an upper bound, **not** an implicit allocation size.

CPU defaults: 8 GiB host budget, 512 MiB headroom, at least 64 MiB staging and
64 MiB activation reservation. These are explicit runtime reservations, not an OS
RSS guarantee: mapped file page residency, Python metadata, allocator pools and
external-library workspaces are OS/library controlled. No all-packed-RAM copy,
pre-touch, PLE-table expansion or dense expert-bank allocation is performed.

With CuPy, the dense backbone must fit the explicit static device budget; otherwise
construction fails and recommends the NumPy path. Supported old-format packed
projections upload compressed bytes only while that static budget permits.
Unsupported/unplaced packed projections copy only activations through native CPU
GEMV and return FP32 activations to GPU. There is no catch-and-full-dequant path.
Free VRAM and total device reservations are checked before construction/state
allocation. CPU is the first baseline; GPU integration needs separate execution
validation on the target device.

The canonical mmap is read-only. Token embedding gathers one row; PLE gathers
sixteen rows. Packed projections/expert slices retain zero-copy views. The owner
must remain open through outstanding operations. Closing a borrowed index model
does not close its store or borrowed executor. Closing an owned model drops its
operator/projection references and closes its own executor and source; exported
views can defer actual OS unmapping, per `MultiShardTensorStore.close`.

## Diagnostics and validation boundary

Optional `diagnostic_callback(layer_id, stage, copied_array)` receives
`hc_attention_read`, `post_mixer`, `hc_ffn_read`, `post_moe`, and final
`hc_output_read`/`logits`. Arrays are copies for first-divergence collection;
retaining them uses caller memory outside the runtime reservation. Callback errors
propagate and poison partially advanced states; no fallback/retry occurs.

Run the synthetic suite:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0209_qwen4exp_model.py
```

Tests cover a tiny combined HC/GDN/QSA/sparse-MoE factory graph, an FP64 synthetic
composition oracle, native packed/no-full-decode dispatch, header/budget rejection,
request ownership/capacity/reset/failure, mock PLE ordering, and close ownership.
**Synthetic results do not establish full-model parity.** The real original-shard
smoke probe is skipped unless an explicit path is supplied; independent pinned
llama.cpp layer/logit comparisons are subsequent work, not a claim of this patch.

## Subsequent actual-model validation

The original two-shard factory executes on CPU and CuPy/RTX5080. The real lifecycle test `tests/test_0210_qwen4exp_generation.py` passes on both, including three greedy decisions and exact within-backend request-reset/prefill parity. Consolidated CPU regressions for all new patches, existing0106 suite and listed training components:1488 passed,184 skipped; command in implementation_status.md.

Independent reference execution subsequently produced logit maxabs differences0.958172 and1.006777, relativeL2~6.16% and5.07% at fixed prefixes[1] and[1,2]. This patch therefore remains **implemented but not full-model numerically validated**. Callback association at the first reported HC boundary must be verified before diagnosing a production defect. No unsupported corrective edit was made. See0210_reference_diagnostics.md.
