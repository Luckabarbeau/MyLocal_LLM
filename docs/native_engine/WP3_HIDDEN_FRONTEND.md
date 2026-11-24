# WP3 — bounded native hidden-vector QSA frontend

Date: 2026-11-23. Isolated component milestone, **not complete WP3, full-layer parity or model acceptance**. Parent-owned `IMPLEMENTATION_STATUS.md` and `native_reference_manifest.json` were not edited. Existing uncommitted work was preserved. No production execution, training, selection, attention, cache/session commit or model loader policy changed.

## Actual descriptor inspection

Used `Runtime.gguf` native descriptors, with automatic shard discovery from `WP2_GGUF_INVENTORY.json` → `shards[0].path`:

`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`

All 60 projection descriptors were inspected. All are on shard 0. GGML shape is `[columns,rows]`, Python shape `[rows,columns]`. All have 2560 columns. Rows: Q=12288, K=512, V=512, indexQ=512, indexK=128. Q rows reshape head-major `[24,512]`, each row `[query256, gate_logits256]`.

| Layer | Q | K | V | indexQ | indexK |
|---:|---|---|---|---|---|
| 3 | IQ4_XS | Q6_K | Q6_K | BF16 | BF16 |
| 7 | IQ4_XS | IQ4_XS | Q6_K | BF16 | BF16 |
| 11 | Q4_K | Q6_K | Q5_K | BF16 | BF16 |
| 15 | IQ4_XS | Q6_K | Q6_K | BF16 | BF16 |
| 19 | Q5_K | Q6_K | Q5_K | BF16 | BF16 |
| 23 | Q5_K | Q6_K | Q6_K | BF16 | BF16 |
| 27 | Q5_K | Q6_K | Q6_K | BF16 | BF16 |
| 31 | Q6_K | Q6_K | Q6_K | BF16 | BF16 |
| 35 | Q6_K | Q6_K | Q6_K | BF16 | BF16 |
| 39 | Q6_K | Q6_K | Q6_K | BF16 | BF16 |
| 43 | Q6_K | Q6_K | Q6_K | BF16 | BF16 |
| 47 | Q6_K | Q6_K | Q6_K | BF16 | BF16 |

Names under `blk.{layer}.`: `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `indexer.q_proj.weight`, `indexer.k_proj.weight`.

| Format | GGML type | Elements/block | Bytes/block | Bytes/2560-column row |
|---|---:|---:|---:|---:|
| F32 (synthetic) | 0 | 1 | 4 | 10240 |
| BF16 | 30 | 1 | 2 | 5120 |
| Q4_K | 12 | 256 | 144 | 1440 |
| Q5_K | 13 | 256 | 176 | 1760 |
| Q6_K | 14 | 256 | 210 | 2100 |
| IQ4_XS | 23 | 256 | 136 | 1360 |

**Q5_K alone was insufficient: IQ4_XS is required by actual Q/K weights.** Other formats explicitly fail rather than using another decode or fallback.

Reuse/provenance:

- `mini_llm/ops/quantized_linear.py::_load_gguf_lanes` formats 2/4/5 supply Q4_K/Q5_K/Q6_K equations, including all shared scale/min packing and Q5 high-bit planes.
- `mini_llm/ops/iq_packed_gpu.py` supplies IQ4_XS lane indexing and signed six-bit scales. The immutable 16-value table comes from `mini_llm/ops/gguf_codebooks.py`, which records llama.cpp `171e8846b4af9766c354064cb776cb34a50f053f`. Its ggml-authors copyright and full MIT permission/disclaimer are retained beside the copied table in `qsa_projection.cu`. Repository license remains `LICENCE.TXT`.
- Unchanged `PackedGGUFWeight.dequantize_reference()` is the packed-weight numerical authority. BF16 reference expands stored bits to F32. No GGML/Q8 activation arithmetic is substituted.

## Computational graph and API

```text
uploaded F32 hidden[2560]
  -> five native matvecs, packed weights decoded per lane in registers
  -> owned F32 qg[24,512], K[2,256], V[2,256], raw[128], IQ[4,128]
  -> SAME ml_qsa_frontend_launch kernel used by projected-input API
  -> normalized/rotated Q,K,IQ; gate logits; dead key; conditional new pool
```

Projection arithmetic is `reference_fp32`: decoded F32 weight times original unquantized F32 hidden, lane-local accumulation then a shared-memory FP32 tree reduction. No Q8 activation quantization, CPU fallback, Python numerical intermediate, dequantized full matrix or per-run temporary device allocation. Kernel arithmetic flags and normalization/RoPE/pool source are unchanged.

```python
with engine.qsa_frontend() as frontend:
    frontend.from_hidden(
        hidden, query_gate_weight, key_weight, value_weight,
        index_query_weight, index_key_weight,
        gamma_q, gamma_k, gamma_iq, gamma_ik,
        first_raw, partial, n,
        epsilon=1e-6,
    )
    # Existing six-output diagnostic API is unchanged:
    frontend.readback_for_test(q, rotated_k, gate, rotated_iq, dead, pool)
    # Optional explicit diagnostics only; all five projections remain native:
    frontend.hidden_readback_for_test(qg, k, v, raw, iq)
```

All inputs are `UploadedTensor`s, same Runtime/engine. Hidden is exact F32[2560]; weights rank 2 with the exact shapes above. Converted gammas are F32 widths 256/256/128/128, used directly, not `1+gamma`. History is exact F32 `first_raw[128]`, `partial[3,128]`, consumed only as specified by the existing frontend. n includes current token. Epsilon must be finite positive; n must be positive uint64. Hidden is the already prepared caller vector: no HC read or hidden normalization is added here.

`ml_qsa_projection_launch` now accepts explicit columns and checks format, nonzero bounded rows/columns, packed block alignment, exact row bytes, and required pointers before launching. The existing output-projection public API remains restricted to width 6144 and its original four formats; it simply passes 6144 to the generalized helper. Hidden projections pass 2560. The helper does not silently interpret an unknown format as Q6_K.

Native ABI additions: `ml_qsa_frontend_from_hidden`, `ml_qsa_frontend_hidden_readback_for_test`, declared in `runtime.h`, bound in `mini_llm/native/__init__.py`. Raw and V are owned retained native outputs available for later native state integration; no cache mutation is performed and no device pointer escapes this ABI. Diagnostic D2H is explicit and is not used to compute native outputs.

## Allocation, ownership and invalidation

- Existing projected-only frontend stays **54288 device + 16 pinned bytes**, preserving its exact admission tests.
- First valid-shape hidden call lazily admits/allocates exactly **55808 device bytes**: `4*(12288+512+512+128+512)`. No extra pinned allocation. Thereafter storage is stable through runs/reset/set_n; total frontend payload is **110096 device + 16 pinned bytes**.
- Engine admission precedes scratch allocation; a budget failure does not change counters. Scratch is freed and its bytes subtracted on frontend destruction. Inputs/uploads remain separately accounted.
- Uploaded inputs are retained natively for the entire synchronized call, including failure unwinding. Upload completion is waited. Stream completion is drained before local input owners can release their buffers. Results are independent snapshots owned by frontend and survive closing public uploads/engine handles.
- Every native hidden/projected run attempt, reset and set_n invalidates the hidden snapshot. A hidden run publishes both snapshot flags only after successful scalar-status transfer/synchronization. Nonfinite hidden/decoded weight/arithmetic (including V, even though V bypasses normalization) rejects the result. Old scratch cannot be read as a newly valid output after failure.
- Python argument validation that rejects a closed/wrong-class handle before entering the ABI does not promise native snapshot invalidation; native stale handles do invalidate after workspace lookup. Tests distinguish these cases, as with existing APIs.
- Pool flag, unused history behavior, stable device n and left-to-right raw pooling additions remain the existing frontend contract. Unready pool storage is not newly valid data.

## Tests and verified state

Files involved: `mini_llm/native_cpp/qsa_projection.{h,cu}`, targeted additions in `runtime.{cpp,h}`, binding `mini_llm/native/__init__.py`, new `tests/test_native_engine_qsa_hidden_frontend.py`. Existing frontend source and Python dequantization/reference implementations were not edited.

Build: `cmake --build /tmp/mini-llm-wp1-build` succeeded. Existing three misleading-indentation warnings in unchanged `gguf_loader.cpp` remain.

Common environment:

```sh
export MINI_LLM_BACKEND=numpy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Commands actually completed:

1. `.venv/bin/python -m pytest -q tests/test_native_engine_qsa_hidden_frontend.py tests/test_native_engine_qsa_projection.py tests/test_native_engine_qsa_frontend.py` — **106 passed, 2 skipped in 5.95s**.
2. `MINI_LLM_QSA_HIDDEN_CHECKPOINT=all .venv/bin/python -m pytest -q -s tests/test_native_engine_qsa_hidden_frontend.py::test_actual_checkpoint` — **1 passed in 2.34s**, all 12 layers × 12 boundaries. Captured stdout is `/tmp/mini-llm-wp3-hidden-all.log`; this transient local raw log is not a committed artifact. Descriptor inventory and summarized diagnostics are recorded here.
3. `.venv/bin/python -m pytest -q tests/test_native_engine_*.py` — **353 passed, 3 skipped in 9.61s** (previous 337 + 16 new default tests). Default skips: attention benchmark, actual output projection, actual hidden frontend.
4. `MINI_LLM_QSA_HIDDEN_CHECKPOINT=1 MINI_LLM_QSA_PROJECTION_CHECKPOINT=1 .venv/bin/python -m pytest -q tests/test_native_engine_qsa_hidden_frontend.py::test_actual_checkpoint tests/test_native_engine_qsa_projection.py::test_actual_checkpoint` — **2 passed in 1.60s**. Independently checks one-layer hidden mode and preserves all-12 actual output-projection behavior.
5. `git diff --check` succeeded.

No verification failure or corrective source edit was needed. Partial edit-tool matches during initial implementation were resolved by more specific edit contexts before building; no verified implementation was replaced. No source/test edits after these successful validations.

Synthetic tests use all six supported storage types, full exact projection shapes, all packed bit fields, zero/negative/subnormal FP16 scales, unmodified packed bytes, all boundaries, exact scratch accounting/reuse, ownership after closes, unsupported F16, wrong shapes/engines/stale handles, nonfinite hidden and V, invalid n/epsilon, budget failure, reset/set_n/projected-run invalidation and diagnostic dtype/capacity. Checkpoint tests use actual F32 converted norm weights from all four norm descriptors; no artificial replacement gamma is used for actual cases.

### Separate tolerance mathematics

All computations are F32. No finite differences or tolerance adjustment after failure.

**1. Projection boundary:** for Python reference `y = dequantize_reference(W) @ hidden`, each native output must satisfy BOTH:

```text
abs(got-y) <= 2e-4 + 5e-5*abs(y)
abs(got-y) <= 2e-4 + 2e-6*sum_j(abs(decoded_W[j])*abs(hidden[j]))
```

The second is a cancellation-aware forward-error envelope; the first also constrains ordinary outputs. These are declared test bounds, not bitwise BLAS/reduction equivalence or a proof for all F32 magnitudes.

**2. Frontend kernel alone:** feed diagnostic native projections into the unchanged Python frontend reference, using the existing `test_native_engine_qsa_frontend.compare` without modification. Gate logits must match exactly; dead key has bound `3e-6*(1+abs(reference))`. For rotated Q/K/IQ/pool, add the following per-pair angle allowance to that base:

```text
u = finfo(float32).eps
f_i = float32(1e7)**(-float32(i)/32), i=0..31
A_i = 8*u*position*f_i*(abs(reference_i)+abs(reference_(i+32)))
```

Add `A_i` to each half-layout RoPE pair coordinate; suffix >=64 gets only the base. Position is n-1 (Q/K/IQ), n-4 (pool); dead has no RoPE. This separates matvec error from the existing powf/NumPy frequency and angle-rounding discrepancy.

**3. End-to-end:** Python-dequantized projections → unchanged Python normalization/RoPE/pool. Base bound `5e-5*(1+abs(reference))`, plus the SAME angle allowance for rotated outputs. Gate uses the base without angle allowance; it is only bit-exact against native projected logits, not against a different matvec reduction. Dead uses base only. n and pool flag are exact. Unready pools are excluded. This predeclared empirical envelope includes projection/RMS roundoff for these fixtures; it is not a uniform relative-error claim near cancellation or for ill-conditioned normalization.

### Actual diagnostics and memory

Hidden RNG seed **2026112304**, one F32 normal vector per layer; deterministic synthetic first_raw/partial from existing frontend fixture. Boundaries: **1,2,3,4,5,127,2048,2051,2052,2053,4096,16384**. These are independent component inputs, not consecutive cache evolution or full-model hidden states. Every projection row is checked. Host reference decodes only 32 rows at a time (**327680 decoded bytes**), with no full decoded matrix on host or GPU. One layer's five packed uploads at a time.

| Layer | Max projection abs error | Max end-to-end abs error | Max error / end-to-end bound | Accounted device bytes |
|---:|---:|---:|---:|---:|
| 3 | 2.14576721e-6 | 0.000654339790 | 0.09246624 | 22264336 |
| 7 | 2.86102295e-6 | 0.000564932823 | 0.09216996 | 21885456 |
| 11 | 1.90734863e-6 | 0.000385642052 | 0.09202643 | 23073296 |
| 15 | 2.62260437e-6 | 0.000661611557 | 0.09180500 | 22264336 |
| 19 | 2.14576721e-6 | 0.000517010689 | 0.09183793 | 27005456 |
| 23 | 2.38418579e-6 | 0.000882983208 | 0.09065259 | 27179536 |
| 27 | 2.02655792e-6 | 0.000718355179 | 0.09220985 | 27179536 |
| 31 | 2.14576721e-6 | 0.000617027283 | 0.09030371 | 31357456 |
| 35 | 1.90734863e-6 | 0.000490367413 | 0.09211266 | 31357456 |
| 39 | 2.38418579e-6 | 0.000861048698 | 0.09443044 | 31357456 |
| 43 | 2.02655792e-6 | 0.000487565994 | 0.09032198 | 31357456 |
| 47 | 3.33786011e-6 | 0.000708937645 | 0.09191472 | 31357456 |

Largest frontend-only component discrepancy: **0.0008829832077026367**, layer 23 at n=16384, also the largest end-to-end absolute discrepancy. Long-position frequency rounding dominates that error; it is not a projection failure. Largest error/bound ratio across all actual outputs was **0.09443043917417526**, below 1. No tolerance was relaxed.

Peak actual accounted payload: **31357456 device + 16 pinned = 31357472 bytes (~29.91 MiB)**, including five packed uploads, hidden/gammas/history and frontend storage. Maximum synthetic fixture uses a **256 MiB admission budget**, actual tests **128 MiB**, both below the **512 MiB extra payload bound**. Counters measure requested payload, not total VRAM: CUDA context/driver, stream/event metadata, allocator rounding and mmap/page cache are not included. No total-VRAM sampling or sanitizer claim is made.

## Remaining boundary

This is a synchronous serialized component API, not graph capture or a fused complete QSA layer. No attention consumption, raw/V cache commit, HC integration, sequence ownership, multi-GPU validation, throughput comparison, training run or full decoder execution is introduced. Broader hidden-state fixtures and integrated cache/state semantics remain later milestones; parent status remains parent-owned.
