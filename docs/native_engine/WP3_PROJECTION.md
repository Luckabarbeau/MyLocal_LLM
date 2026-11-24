# WP3 — bounded native reference_fp32 QSA output projection

Date: 2026-11-23. Isolated component milestone, **not complete WP3 or full-layer/model acceptance**. Parent-owned `IMPLEMENTATION_STATUS.md` and reference manifest are intentionally untouched. Existing uncommitted runtime/selection/attention work is preserved; no production Qwen paths changed.

## Inspection and format decision

Queried the actual checkpoint through `Runtime.gguf` native descriptors, using the first shard path in `WP2_GGUF_INVENTORY.json` (automatic two-shard discovery):

`/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`

All 12 QSA `blk.{3,7,...,47}.attn_output.weight` descriptors are GGML shape `[6144,2560]`, i.e. Python `[out=2560,in=6144]`, on shard 0. Layer 3 is Q4_K (type 12, 256 elements / 144 bytes per block, 3456 bytes/row, 8847360 total bytes). The other eleven are Q6_K (type 14, 256 / 210, 5040 bytes/row, 12902400 total bytes). No BF16 occurs among these twelve. F32 and BF16 (type 30) are also supported and synthetically tested; no other formats silently substitute.

Inspected reuse boundaries:

- `mini_llm/ops/mixed_q8_gpu.py` has packed Q4_K/Q6_K matvec, but its `linear` quantizes FP32 activations to Q8_1. This is **not** the chosen arithmetic.
- `mini_llm/ops/quantized_linear.py::_load_gguf_lanes`, formats 2 and 5, already defines validated lane-local FP32 Q4_K/Q6_K bit indexing and decode arithmetic. Those exact equations are ported to CUDA, not replaced with a new quantization convention. CUDA source identifies this repository source explicitly. No external mixed-Q8/Strata code was copied or stripped of its license.
- `mini_llm/external/gguf_projection.py` provides lazy packed-weight resolution and projection dispatch, not a native consumer of attention workspace pointers.
- `mini_llm/external/qwen4exp_weights.py::projection` dispatches dense, device packed, or CPU packed implementations depending on residency/capture settings. This new explicit native reference_fp32 API does not inherit its residency fallback or GGML activation arithmetic.
- Numerical authority in the new tests is existing `PackedGGUFWeight.dequantize_reference()` followed by NumPy FP32 matvec, in bounded tiles for actual weights.

## Graph and API

```text
native attention valid gated[24,256] buffer
    -> flatten head-major, j=0..6143 (no device/host copy)
packed uploaded weight[row,j]
    -> lane-local FP32 decode in registers
    -> FP32 decoded_weight * original FP32 gated[j]
    -> FP32 per-thread sums + shared-memory tree reduction
    -> native projected FP32 buffer[rows]
```

Explicit arithmetic name: `reference_fp32`. FMA contraction remains disabled by the existing CMake flags; no fast math, Q8 activation substitution, permanent dequantized matrix, CuPy/Python numerical operation, or CPU fallback exists in this graph. Q4_K decode is `(d*scale)*quant - dmin*minimum`; Q6_K is `(d*signed_scale)*signed_quant`, retaining the reference's FP32 operation grouping. BF16 expands its stored bits into the upper half of FP32. Reduction order differs from BLAS: bitwise output identity is not claimed.

```python
with attention.projection(uploaded_weight, arithmetic="reference_fp32") as projection:
    milliseconds = projection.run(repeats=20)
    # Optional explicit diagnostic only:
    projection.readback_for_test(writable_float32_output)
```

Native ABI: `ml_qsa_projection_{create,destroy,run,readback_for_test}`. Implementation: `mini_llm/native_cpp/qsa_projection.{h,cu}`, `runtime.{h,cpp}`, `CMakeLists.txt`; binding: `mini_llm/native/__init__.py`. Tests: `tests/test_native_engine_qsa_projection.py`.

- Creation requires same-engine attention and uploaded weight; native descriptor is authoritative. Exact rank 2, GGML width 6144, rows 1..2560, and F32/BF16/Q4_K/Q6_K types only. Bounded smaller row counts permit synthetic tests without a 60 MiB F32 fixture for every case.
- Projection owns shared attention, weight and indirectly selection/engine lifetimes; closing public upload/attention/selection/input/engine handles does not invalidate it. Destruction drains the stream before releasing dependencies. Upload completion is waited before use.
- Attention must have a successful valid snapshot. Every native run first invalidates the previous projection result; bad repeats, unavailable attention, nonfinite decoded weight/input or nonfinite output cannot expose a newly trusted result. Selection reset alone does not erase valid attention or projection snapshots.
- Projection outputs are independent snapshots. A later attention update does not retroactively erase an already completed projection snapshot; the next projection run consumes the latest valid gated output. No cache/session/state commit occurs.
- All GPU work is on the existing engine stream and under the existing runtime lock. A scalar status is copied to host and synchronized before publishing success, as in attention. **No gated activation, selected ID, or projected tensor readback is needed to compute projection.** Python still orchestrates separate attention/projection calls; this is not a fused asynchronous complete-layer API or CUDA graph capture.
- Exact added payload is `4*rows+4` device bytes and 4 pinned bytes, no per-run allocation. At 2560 rows: **10244 device + 4 pinned bytes**, excluding the independently uploaded packed weight. No raw device pointer escapes the ABI. Readback requires explicit writable native contiguous F32 storage and sufficient capacity.

## Validation and verified state

Build succeeded:

```sh
cmake --build /tmp/mini-llm-wp1-build
```

Existing three misleading-indentation warnings in unchanged `gguf_loader.cpp` remain. CUDA architecture 120, Release build, RTX 5080 device 0; no exclusive-GPU or locked-clock claim.

Common environment for commands below:

```sh
export MINI_LLM_BACKEND=numpy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

1. Focused initial run: **9 passed, 1 failed, 1 optional skipped**. Failure was test setup, not projection arithmetic: a 2561-row F32 rejection fixture could not upload into the test's 32 MiB engine budget. One localized correction increased only that engine budget to 128 MiB. No production corrective edit.
2. Exact focused retest: `.venv/bin/python -m pytest -q tests/test_native_engine_qsa_projection.py` — **10 passed, 1 skipped in 0.57s**.
3. Actual checkpoint validation/benchmark, separately:
   `MINI_LLM_QSA_PROJECTION_CHECKPOINT=1 .venv/bin/python -m pytest -q -s tests/test_native_engine_qsa_projection.py::test_actual_checkpoint` — **1 passed in 1.32s**.
4. Full native suite: `.venv/bin/python -m pytest -q tests/test_native_engine_*.py` — **257 passed, 2 skipped in 4.27s**. Includes all previous 247 tests unchanged plus 10 new default tests. Optional attention benchmark and actual-weight test are skipped by default; actual-weight test was explicitly run in step 3.

The verified state covers all source/binding files named above and the new test file. No source changes after these successful commands.

Synthetic coverage: F32/BF16 zero and random rows; both packed quant formats with random packed codes, all scale/min fields, zero/negative/subnormal/max finite FP16 scales, signed Q6 scales -128/127; unchanged packed bytes; repeated allocation stability; retained ownership after public closes; stale/null ABI handles; shape/type/engine/budget rejection; readback capacity/layout/type rejection; snapshots/reset; failed-run invalidation; NaN/Inf weights for all four formats.

### Numerical tolerances

All arithmetic is FP32; no finite-difference test or epsilon is involved. Actual weights and ordinary dense synthetic cases use predeclared `rtol=5e-5, atol=2e-4`. Different parallel reduction orders require an absolute allowance near cancellation/zero. Tolerance was not relaxed after a failure. Extreme synthetic quant scale cases use the cancellation-aware bound

`abs(got - reference) <= 2e-6 * sum_j(abs(decoded_weight[j])*abs(input[j])) + 2e-4`.

The scale-independent factor covers differing FP32 reductions of 6144 terms, including maximal finite-half scales; it is not a claim of relative accuracy for nearly cancelled outputs. Quant decoded layouts are inherited from the existing equations, not tuned to this bound.

### Actual checkpoint full-output results and raw CUDA-event samples

Deterministic input: existing fixture RNG seed 363386; n=1, gate=0, hence attention context is exactly current V and gated input is `0.5 * repeat(V,12,axis=0)`, width 6144. This isolates matvec from softmax reduction noise. All **2560 output rows for all 12 weights** compared to existing Python dequant+FP32 matvec. Host reference decoded 32 rows at a time (786432 decoded bytes), one packed weight on GPU at a time. No model construction or full decoded matrix on either host or GPU.

Each timing sample is CUDA-event ms/iteration for 20 repeated projections after one warmup run. Includes status clear and matvec; excludes weight/input uploads, attention/selection, explicit diagnostics, final scalar status transfer and host synchronization. Three actual raw samples per weight from this task's test output are transcribed below; no inferred samples.

| Layer | Format | Max absolute error | Raw event samples (ms/iteration) |
|---:|---|---:|---|
| 3 | Q4_K | 1.0728836059570312e-6 | 0.05551040172576904, 0.0554112009704113, 0.05540480092167854 |
| 7 | Q6_K | 1.5497207641601562e-6 | 0.037032000720500946, 0.03704959899187088, 0.03707999736070633 |
| 11 | Q6_K | 8.940696716308594e-7 | 0.03702720254659653, 0.037059199064970016, 0.037064000964164734 |
| 15 | Q6_K | 9.5367431640625e-7 | 0.03699519857764244, 0.03705599904060364, 0.037057600915431976 |
| 19 | Q6_K | 9.5367431640625e-7 | 0.03696800023317337, 0.03706879913806915, 0.03708320111036301 |
| 23 | Q6_K | 1.2516975402832031e-6 | 0.03704319894313812, 0.03702399879693985, 0.03708159923553467 |
| 27 | Q6_K | 9.5367431640625e-7 | 0.037057600915431976, 0.037033598870038986, 0.03706879913806915 |
| 31 | Q6_K | 1.2516975402832031e-6 | 0.03707840293645859, 0.037064000964164734, 0.03702880069613457 |
| 35 | Q6_K | 1.1473894119262695e-6 | 0.03704319894313812, 0.03708479925990105, 0.037062399089336395 |
| 39 | Q6_K | 1.0281801223754883e-6 | 0.03702079877257347, 0.03705280274152756, 0.05883679911494255 |
| 43 | Q6_K | 8.642673492431641e-7 | 0.03707519918680191, 0.03707680106163025, 0.03707519918680191 |
| 47 | Q6_K | 1.1920928955078125e-6 | 0.03704800084233284, 0.03706560283899307, 0.03703039884567261 |

Largest absolute discrepancy: **1.5497207641601562e-6**, layer 7. Largest raw relative discrepancy: **0.010461538098752499**, also layer 7 near a cancelled result; maximum error divided by `(atol + rtol*abs(reference))` across every row/layer was **0.006748745683580637**, far below 1. No numerical test failed.

Tracked payload during the actual test:

- Q4_K: 8979524 device + 24 pinned bytes; combined peak 8979548 bytes.
- Q6_K: 13034564 device + 24 pinned bytes; combined peak **13034588 bytes** (about 12.43 MiB).

This is well below the <512 MiB extra GPU payload target even including the one packed weight, selection/attention inputs/workspaces and projection. It is **not total VRAM/RSS**: CUDA context, stream/event/driver allocations, allocator rounding, mmap/page cache and caller host buffers are not measured by these counters. Largest synthetic shape-rejection upload is about 60 MiB, with 128 MiB engine admission budget, also below target.

## Limitations / next boundary

Only fixed width 6144 and at most 2560 rows, four explicit storage types, synchronous serialized run calls, one deterministic actual-weight vector, synthetic attention inputs. Actual tests use native leases/descriptors/payloads and Python reference decode from those same payload bytes; they do not independently authenticate upstream checkpoint conversion. No full-model inference, complete QSA layer comparison, HC write, cache mutation/sequence commit, graph capture, sanitizer/race checker, multi-GPU, broad fuzzing, throughput speedup or matched production benchmark is claimed. BF16 is optional synthetic coverage, not an actual QSA checkpoint requirement. Event samples are component timing evidence, not a comparison against the Python/Strata engine.

The next boundary is integration with native pre-attention projections/normalization/RoPE and controlled layer/cache ownership; it is not implemented by this output-only consumer.
