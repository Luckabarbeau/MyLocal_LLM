# WP6 — populated full-capacity layer evidence (GPU1)

## Acceptance scope

The parent full-capacity suite completed **6 passed in 22.61s** (retained JUnit suite time **22.604s**, zero failures/errors/skips). All four format/policy result files passed milestones **65,536 / 131,072 / 262,144**. This establishes bounded populated-layer transport, selection, attention and commit behavior for a **prepared controlled snapshot**, using **actual checkpoint layer-3 weights**. It is **not a 262K-token model run**, sequential checkpoint prefill, multi-layer decode, perplexity evaluation, or full WP6 quality acceptance. Strict WP7 remains open.

Raw full-capacity artifacts remain untouched in `logs/wp6_full_capacity_gpu1/`: each of `reference_fp32_{ggml_cpu,strata}` and `strata_int8_per64_fp16_{ggml_cpu,strata}` has `.config.json`, `.events.jsonl`, and `.result.json`. Each event stream contains 64 imports (at most 4,096 rows each) and three milestones. The parent JUnit was copied verbatim from `/tmp/wp6-full-capacity-gpu1.xml` to `logs/wp6_memcheck_gpu1_capacity64_20261124_evidence/parent_full_capacity.xml` for retention.

[`WP6_FULL_CAPACITY_EVIDENCE.json`](WP6_FULL_CAPACITY_EVIDENCE.json) is compact machine-readable evidence: SHA-256 of every config/event/result file, recorded source and library hashes, expanded documentation-time source hashes, exact used checkpoint payload hashes/descriptors, checkpoint identities, milestone metrics, memory snapshots, and sanitizer/JUnit hashes. Selected-ID arrays are omitted from the summary but retained in raw results and bound by their recorded hashes. Event milestones were checked against result records; checkpoint identities before/after match. All run-recorded source hashes and the library hash match the current files. Expanded source hashes do not prove reproducible binary compilation; checkpoint payload hashes cover the used layer-3 weights, not the entire checkpoint.

Capture library: `/tmp/mini-llm-wp1-build/libmini_llm_native.so`, SHA-256 `f8f6efcb3e99dbb171fb83a99dc72b7bb015bf4b9097c1517cbd07d9acd7c983`. Source/library-current comparisons below describe evidence-generation time. Subsequent sampler/generation integration and the GGML loader-lifetime fix rebuilt that library; these immutable artifacts remain evidence of the capture build, not an assertion that the current binary retains this hash.

## Computational contract and oracle boundaries

Targeted implementation: `tests/native_qsa_full_capacity.py` (`Prepared`, `cpu_front`, `attention`, `run_case`) and `tests/test_native_engine_qsa_full_capacity.py`.

- Historical K/V are deterministic dyadic arithmetic sequences, **not checkpoint hidden-state projections**. Historical K rotary prefixes are zero. One-hot raw index windows produce consistent normalized pools and controlled positive ties. Windows end at each milestone; their width is 4,096 in the full run.
- The GGUF checkpoint supplies actual `blk.3` projection/output weights and normalization tensors. At each milestone, a deterministic hidden input is projected, normalized, rotated, selected, attended and committed through the actual native complete-layer path. There are **three actual steps**, interleaved with imported prepared history, not 262,144 sequential steps.
- CPU frontend/selection use NumPy. Selection IDs must exactly equal CPU `select_cells` for each named tie policy, with 2,051 selected IDs. Full-capacity selections exclude historical token zero without a compulsory token-zero fetch.
- Independent CPU frontend checks use the existing position-dependent FP32 long-angle envelope for q/k/index-query/pool; gate/value use `rtol=2e-4, atol=2e-4`. The native standalone frontend provides a factorized same-frontend oracle for committed bytes and INT8 attention. This is not wholly independent end-to-end CPU validation of INT8 projections.
- Imports are checked exactly against FP32 or decoded INT8 K/V. Last committed K/V bytes, full pooled table, unchanged pooled prefix, position, stats and output are checked. A step beyond capacity must reject without changing committed state, stats or output. All four cases passed these checks; checkpoint inode/size/mtime identities remained unchanged.

### Numerical results: do not conflate two errors

Max absolute output errors below are ordered **65,536 / 131,072 / 262,144**; exact values are in JSON.

| Format / policy | Native vs asserted oracle | Native vs factored unquantized FP32 |
| --- | --- | --- |
| FP32 / ggml_cpu | 1.49012e-7 / 7.89762e-7 / 3.44217e-6 | 8.94070e-8 / 8.94070e-8 / 1.04308e-7 |
| FP32 / strata | 1.19209e-7 / 8.34465e-7 / 3.33786e-6 | 1.19209e-7 / 8.56817e-8 / 9.68575e-8 |
| INT8 / ggml_cpu | 8.94070e-8 / 1.19209e-7 / 9.68575e-8 | 0.0010967031 / 0.0010628849 / 0.0009119138 |
| INT8 / strata | 8.19564e-8 / 1.04308e-7 / 1.49012e-7 | 0.0010965616 / 0.0010619387 / 0.0009108121 |

For FP32, the asserted oracle is independent CPU attention including CPU frontend (`rtol=2e-4, atol=2e-4`). For INT8, it is CPU attention over independently decoded per-64 INT8/FP16-scale storage using native frontend values (`rtol=4e-5, atol=3e-6`). Thus ~1e-7 INT8 decoded-oracle agreement measures implementation correctness; ~1e-3 differences from unquantized FP32 measure storage-policy discrepancy and **are not decoded-oracle errors or quality acceptance**. `independent_fp32_output_max_abs` additionally includes independent frontend differences; the JSON retains it separately from `factored_fp32_policy_difference_max_abs`. These discrepancy metrics are observations, not a model-quality threshold.

## Working set, accounting and bounds

Full-run configuration: capacity 262,144; page size 1,024; **one physical GPU history slot**; raw historical diagnostics off. Host backing budgets are 1,200,000,000 bytes (FP32) and 350,000,000 bytes (INT8); engine requested GPU budget is 268,435,456 bytes.

The current **complete-layer** `QSAPagedState::gather` gathers all selected resident hits first, then stages/gathers each miss, waiting for its reader to retire before reusing a slot. Selected decoded K/V remain in bounded scratch. Therefore distinct selected historical pages **need not fit simultaneously in the slot cache**. This differs from the older standalone minimum-working-set contract described in `WP6_QUANTIZED_STREAMED_KV.md`; that older restriction must not be applied to this complete-layer evidence. No full-resident historical K/V fallback is used.

At the three milestones, distinct requested historical pages were **4/5/6** for ggml_cpu and **3/4/5** for strata, for both formats. Each step recorded zero hits and a miss per requested page. Post-commit residency was zero (the last staged page is invalidated by its new commit), pending zero; residency is asserted at most one. This demonstrates multi-page streaming through one slot, not cache-hit performance or arbitrary selection-distribution throughput.

Let C = capacity, P = page rows, S = physical slots, L = ceil(C/P), R = 4,096 FP32 or 1,056 INT8 bytes/token. With diagnostics off:

- Request structure: `24 + 2*2051*8 = 32,840` bytes. Metadata counter increases by exactly this once per attention call (98,520 total); it counts the request D2H copy, **not all CUDA transfers or page-table updates**.
- Paging device allocation: `8*L + 32840 + 8 + 2051*8 + 2051*4096 + (S+1)*P*R`. The 8,400,896-byte selected K/V scratch holds at most 2,051 decoded rows; the extra page is staging.
- Complete-layer stats add the layer workspace, including two capacity-sized pooled tables, capacity-sized scores and bounded raw tail, not full historical device raw/K/V. `stats.device_bytes` is **not** total engine or process VRAM.
- Host accounting: `32840 + 8*(L+2*S) + allocated_host_pages*P*R`. Full-capacity overhead is 34,904 bytes. This is tracked backing/metadata allocation, not total RSS, Python/NumPy storage, allocator/container overhead, or mapped checkpoint pages.
- `populated_bytes = n*R` counts committed payload, distinct from allocated host-page capacity. `h2d_bytes` counts staged historical payload only; it excludes fixture uploads, weights, page-table updates and other transfers.

| Full-run bound / final value | FP32 | INT8 |
| --- | ---: | ---: |
| Complete-layer `stats.device_bytes` | 85,186,704 | 78,960,784 |
| Maximum engine requested peak | 135,727,808 | 129,501,888 |
| Final host-accounted bytes | 1,073,776,728 | 276,858,968 |
| Final populated bytes | 1,073,741,824 | 276,824,064 |
| Cumulative staged H2D, ggml_cpu | 62,902,272 | 16,216,992 |
| Cumulative staged H2D, strata | 50,319,360 | 12,972,960 |

Full-run milestone process high-water RSS ranged from 1,000,264 to 2,020,820 KiB across sequential cases; high-water marks are cumulative process measurements, not isolated per-layer allocation. Observed CUDA free memory ranged from 5,714,018,304 to 5,720,309,760 bytes. Free-memory values are device-wide snapshots and can include other processes; they do not define this engine's allocation bound. Preflight requires exactly the GPU1 UUID mask, one visible device, NumPy backend, at least 15 GiB host available and 1.75 GiB GPU headroom. Runtime checks require requested peak below 256 MiB and more than 512 MiB GPU free at milestones.

## Commands and validation

### Parent full-capacity run (already completed)

The configs retain the actual argv:

```sh
.venv/bin/python -m pytest -q -s --junitxml=/tmp/wp6-full-capacity-gpu1.xml tests/test_native_engine_qsa_full_capacity.py -x
```

Recorded environment: `CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631`, `MINI_LLM_BACKEND=numpy`, `MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so`, `MINI_LLM_WP6_FULL_CAPACITY=1`, `MINI_LLM_WP6_FULL_OUTPUT=logs/wp6_full_capacity_gpu1`, `OPENBLAS_NUM_THREADS=1`; capacity defaults to 262,144. The config environment capture does not retain OMP or pytest-plugin settings, so those settings cannot be independently certified for the parent run from these artifacts. The parent reports **6 passed in 22.61s**; retained JUnit independently confirms six cases and no skips/failures/errors. This documentation task did not rerun or overwrite that directory.

### New focused memcheck, reduced capacity 64

The following command ran from the repository root after creating a fresh output directory:

```sh
env CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so MINI_LLM_WP6_FULL_CAPACITY=1 MINI_LLM_WP6_TEST_CAPACITY=64 MINI_LLM_WP6_FULL_OUTPUT=logs/wp6_memcheck_gpu1_capacity64_20261124_evidence OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout --signal=TERM --kill-after=5s 120s /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 --log-file logs/wp6_memcheck_gpu1_capacity64_20261124_evidence/memcheck.log .venv/bin/python -m pytest -q -s --junitxml=logs/wp6_memcheck_gpu1_capacity64_20261124_evidence/pytest.xml tests/test_native_engine_qsa_full_capacity.py -x > logs/wp6_memcheck_gpu1_capacity64_20261124_evidence/pytest.log 2>&1
```

Result: **exit code 0; 6 passed in 10.93s; `ERROR SUMMARY: 0 errors`**. Two transport/diagnostic format cases plus all four format/policy checkpoint cases ran, with checkpoint milestones 16/32/64. No timeout occurred. The timeout sends TERM at 120s with a 5s forced-kill grace. GPU1 was addressed only through the exact UUID mask; inside the mask it is logical device 0, never device 1. No GPU0 workload was requested. NumPy CPU reference, BLAS/OMP single-threading and disabled plugin autoload were explicitly set for this new run.

This is reduced-capacity memory-safety evidence, **not a full-capacity sanitizer run**. Its logs and all four config/event/result sets are retained separately in the new directory. Evidence-generation checks also passed: eight case results passed, all 24 milestone event/result records agree, recorded source/library hashes match current files, and checkpoint before/after identities match.

## Open gates, unchanged

The known open strict test is `tests/test_native_engine_decoder.py::test_actual_all48_two_steps_reset`. `WP7_DIVERGENCE.md` records token 0 / ID 1 first violating layer 7 `hc_ffn_read` (max abs 0.0015584230422973633, relative L2 0.0005356750911888023, 719/2560 violations) and token 1 / ID 7 layer 8 `hc_ffn_read` (max abs 0.000667273998260498, 287/2560 violations), with FP32 `rtol=2e-4, atol=2e-4`. That failure was not rerun, suppressed, marked xfail, relaxed or repaired here. Passing this WP6 suite does not close strict whole-model WP7, long-context model quality, or full WP6 quality acceptance. The parent status separately closes the WP2 descriptor/loading/resource gate based on whole-model binding; this is not a claim of whole-model numerical parity.

No runtime/test source or `IMPLEMENTATION_STATUS.md` edits were made by this documentation/validation task. Existing parent-owned source changes were preserved.
