# WP9 — native fixed-trajectory critical-path measurement (GPU1)

## Disposition

**PARTIAL: full-checkpoint baseline and separate C++ wall-timer diagnostic passed; Nsight capture finalization failed.** The dominant **measured host critical-path component is exposed CPU-expert waiting**, not a launch-count inference. There is **no usable Nsight report/SQLite, no measured GPU hot-kernel ranking, and no OSRT wait attribution**. Those requested trace results remain blocked, not green. No kernel, precision, equation, placement, normal-runtime instrumentation, or parent-owned implementation status was changed.

The existing HTTP results in `WP9_NATIVE_HTTP_BASELINE.md` remain unchanged: repeated 6.857810 emitted pieces/request-second, 8.742991 samples/native-decode-second, cold 33.898822-second prefill for the 128-token request (127 inputs consumed before decode).

## Workload and complete-step contract

New probe: `scripts/profile_native_qwen_decode.py`. No server, indefinitely running command, Python model, numerical fallback, or GPU0 workload is involved.

- Opens the actual two-shard **83,617,662,656-byte** checkpoint through existing `NativeDecoder.open`, using the same native/bridge/GGML libraries and budgets as the HTTP baseline.
- Reuses the exact retained HTTP `request.json`, GGUF tokenizer, and `render_chat(..., thinking=False)`. No generated approximation of the source prompt.
- Request-file SHA256: `b0fee04ba7a614c038e2db729c7718053d367d07d48d76d6472744d23f958a61` (matches retained HTTP metadata).
- Rendered prompt UTF-8 SHA256: `e7e392dba6fde81b7f1fc21a6204fea85e927c2ae41b9bf901672060c9ce94b6`.
- Full 128-ID prompt SHA256, compact JSON integer array: `f4581f508c17892c17f95b3a47f04d2b92811997e7fc623e4dc555306c8f4040`. Actual IDs are retained in raw JSON.
- Capacity **256**, page size 16, two physical pages, four CPU expert threads, sampler seed 0. Arithmetic/placement unchanged: reference-FP32 HC/PLE/recurrent/KV; QSA GGML CPU; routed experts CPU mmap GGML Q8.
- Every measured step is `step_device(input_id)` **plus** `sample_decoder(..., temperature=0)`: embedding, all **48 layers** (36 recurrent / 12 QSA), PLE where configured, HC, full routed/shared MoE, final HC, full **248,320-logit vocabulary projection**, and native GPU greedy sampling. Logits remain on device; no logits D2H, HTTP/SSE, queue, or detokenization included.
- Each process independently performs literal decoder/sampler reset, consumes the 127-token prefix, runs four complete warmup steps, then **resets and replays that same prefix again** before measuring four steps. No inherited history or checkpoint shortcut.
- Full-workload active committed history: **127 before, 131 after**, step positions 127–130. The first measured step consumes the final prompt token, just as the HTTP decode interval does. This is not four exclusively generated-token inputs.
- Fixed input IDs: `[271, 760, 4125, 369]`; sampled IDs: `[760, 4125, 369, 303]`. Warmup, baseline replay, and separate diagnostic replay samples matched exactly for this short workload. This is not whole-trajectory numerical acceptance.

## Unprofiled measurements — only these are the speed baseline

All are four-step observations, not confidence intervals or a benchmark campaign.

| Uninstrumented process | Complete-step walls, ms | Median, ms |
| --- | --- | ---: |
| Initial 128-token attempt | 119.269217, 120.673166, 119.094923, 119.005492 | 119.182070 |
| Final 128-token control paired with C++ diagnostic | 115.325605, 113.641515, 114.877550, 114.271948 | **114.574749** |

Final control total: **458.116618 ms**, reciprocal median **8.727927 complete steps/s**. This is not emitted/SSE throughput and is not a replacement for the HTTP baseline. Final control setup 5.901650 s, first prefix 14.910559 s, reset/replay prefix 14.320382 s; page-cache state differs from historical cold prefill. Process lifetimes: control **36.672 s**, diagnostic **37.454 s**; combined comfortably below 180 s.

## Nsight attempts and concrete blocker

`nsys` version **2026.3.2.476-263238834031v0**, installed at `/usr/local/cuda/bin/nsys`. Both experiments used synchronous owned launches, shared **172 s** deadline, cleanup, and an outer terminal **180 s** bound.

Initial full-prompt capture command prefix:

```sh
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
  --output=.../decode .../python scripts/profile_native_qwen_decode.py ... --worker profile
```

The full attempt timed out without a usable trace. Retained artifacts: `benchmarks/native_critical_path_gpu1/` (including the successful unprofiled result). Its initial log warns that CPU context-switch tracing is unavailable.

The requested bounded retry used the **first 16 actual HTTP prompt IDs**, not a claim of full-prompt equivalence, with active history **15 → 19**, capacity still 256. Both its unprofiled and profile processes used that identical shortened input. Its unprofiled median was **117.315510 ms**. Added `--cpuctxsw=none` because the environment check reports kernel paranoid level 4 and failed `perf_event_open`/CPU profiling support. CUDA/NVTX/OSRT tracing remained enabled, sampling remained disabled.

The retry's flushed progress log establishes:

1. Setup completed; prefill/warmup completed; literal reset/replay completed.
2. `ctypes` loaded the same `libcudart.so.13` used by the native library.
3. `cudaProfilerStart()` returned successfully at monotonic 303054.796880752.
4. Four complete native steps plus sampling executed; entry into `cudaProfilerStop()` was logged at 303055.324171918.
5. **`cudaProfilerStop()` did not return before the deadline.** No finalized trace/SQLite was emitted.

No CUDA permission-denied error was observed. The CPU permissions warning is **not proof of the CUDA finalization hang's cause**. CUDA stop/finalization interoperability remains unresolved. This CUDA installation lacks `libnvToolsExt.so`, so no optional NVTX step labels were available; capture boundaries were the profiler API calls themselves.

Raw streams were preserved from Nsight's temporary directory as `decode_incomplete.qdstrm` in both attempt directories. Bounded offline import of the 16-token stream returned code 1:

```text
QdstrmFile.cpp(221): QdstrmFile::verifyHeader()
boost::wrapexcept<QuadDCommon::IncompleteFileException>
Qdstrm file is incomplete.
```

Details: `benchmarks/native_critical_path_gpu1_16/nsys_failure.json`, `profile_launch.log`, and `profile_launch.json`. There is no synthetic SQLite file or fabricated kernel statistic.

### Cleanup correction

The initial Nsight target called `setsid`, escaping the launcher process group. The post-run GPU check exposed its still-owned PID 606495. Its full command line was verified against this probe/output/worker before killing **only that owned session**; `orphan_cleanup.json` records the action. The probe now records worker PID **and `/proc` start ticks** before native setup, and cleans that separately owned PID as well as its launcher group. The 16-token retry exercised this timeout cleanup successfully. Final GPU1 checks show **no compute processes**, 18 MiB used / 15,822 MiB free. No external job was stopped.

## Separate C++ wall-timer fallback — diagnostic, not speed baseline

`build_native_decode_diagnostic.py` creates a separate expanded `runtime_diagnostic.cpp`, instruments host stages and existing `cudaStreamSynchronize` calls, compiles **only a new host-runtime object**, and links it with the **15 unchanged existing objects**, including all CUDA kernel objects. It never edits the original source tree, rebuilds the baseline object/library, or adds synchronization, device events, precision changes, or new math. Original headers and compiler optimization conventions are reused. The generated source, object hashes, compile/link commands, logs, and library hash are retained in `benchmarks/native_decode_diagnostic_build/`.

Timers are enabled explicitly through diagnostic-only C exports immediately around the four steps, after setup/prefill/warmup/reset/replay. The separate timer library is never used as the unprofiled control. It is a measurement fallback, **not numerical fallback**.

Measured diagnostic total **485.340090 ms**, median **121.122665 ms**. These instrumented values must not be presented as the speed baseline, nor their difference from the control as a causal estimate of timer overhead (sequential processes also vary).

| Host stage | Calls / 4 steps | Aggregate ms | ms / step | Diagnostic wall |
| --- | ---: | ---: | ---: | ---: |
| HC read | 384 | 24.178166 | 6.044542 | 4.982% |
| MoE submit | 192 | 61.785286 | 15.446322 | 12.730% |
| **MoE finish** | **192** | **301.405399** | **75.351350** | **62.102%** |
| Recurrent branch | 144 | 49.361186 | 12.340297 | 10.170% |
| QSA branch | 48 | 31.619442 | 7.904861 | 6.515% |
| HC write | 384 | 5.706484 | 1.426621 | 1.176% |
| PLE | 4 | 0.784812 | 0.196203 | 0.162% |
| Final HC/head/completion tail | 4 | 9.912606 | 2.478151 | 2.042% |
| Other outer work, including sampling/bindings | — | 0.586709 | 0.146677 | 0.121% |

Nested measurements, **do not add to the stage rows**:

| Nested region | Calls / 4 steps | Aggregate ms | ms / step | Diagnostic wall |
| --- | ---: | ---: | ---: | ---: |
| **Exposed CPU-expert `cpu->finish()` wait inside MoE finish** | **192** | **288.321252** | **72.080313** | **59.406%** |
| Existing host-runtime `cudaStreamSynchronize` calls | 2,192 | 89.024508 | 22.256127 | 18.343% |

There are **548 measured stream barriers/step**; mean host duration **40.613 microseconds/barrier**. The duration, not this count alone, supports their significance. This is the instrumented host-runtime call set, not a CUPTI-wide count of every driver/device barrier or kernel. Some synchronization is in multiple enclosing stages. The CPU wait and synchronization totals cannot be added to stage totals or interpreted as GPU utilization.

### Evidence-supported interpretation and limits

The exact forward dependency chain is HC read → route/input preparation and CPU expert submission → shared/device work → CPU job completion → combine → HC write, repeated across the stack. Existing `Decoder::mix` calls `job->finish(false)` before its HC write, and `MoeJob::finish` calls `cpu->finish()` before uploading the expert rows and combining. Thus the measured **288.321 ms of exposed host wait is on the serial decoder dependency path**: CPU jobs must complete before the next layer can consume that result. MoE submit+finish occupies **74.832%** of measured diagnostic wall; the explicitly timed CPU wait is **59.406%** by itself.

This identifies the dominant measured component for this workload. It does **not** establish total expert CPU compute time, disk/page-fault time, CPU scheduler delay, precise GPU/CPU overlap, GPU idle duration, a particular expert dot-product bottleneck, or a specific kernel to optimize. The remaining recurrent/QSA/head host-stage measurements include launch overhead and existing waits, not isolated GPU execution durations. OSRT and GPU kernel ranking remain unavailable. No speculative optimization was made.

**Single next useful experiment:** reproduce the `cudaProfilerStop` finalization hang with a tiny CUDA/profiler-API program under this same Nsight/driver/runtime combination, without the model or OSRT first. That would distinguish a tooling-level blocker from this workload's capture interaction before another expensive model profile. No reason to change kernels or precision on the present trace evidence.

## Reproduction and artifacts

GPU-capable probe/build/profile launches used only:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
```

Device: physical GPU1, **NVIDIA GeForce RTX 5070 Ti**, driver **580.178.04**. Preflight refuses existing compute processes or >64 MiB usage rather than controlling jobs or using GPU0. CPU-only tests use an empty GPU mask.

Final passing timer experiment, working directory repository root, outer bound 180 s:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 \
  .venv/bin/python scripts/build_native_decode_diagnostic.py \
  --output benchmarks/native_decode_diagnostic_build

CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 \
  MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=. \
  .venv/bin/python scripts/profile_native_qwen_decode.py \
  --output benchmarks/native_critical_path_gpu1_timers --steps 4 --prompt-tokens 128 \
  --deadline 172 \
  --diagnostic-library benchmarks/native_decode_diagnostic_build/libmini_llm_native_diagnostic.so
```

Output directories must be new; existing artifacts are not overwritten. Native libraries/model paths are taken from the retained baseline provenance, not guessed cache filenames. To run Nsight instead, omit `--diagnostic-library`; the current finalization failure is expected to remain unresolved. `--prompt-tokens 16` is explicitly a shortened-history diagnostic.

Primary raw results: `benchmarks/native_critical_path_gpu1_timers/{unprofiled,diagnostic,analysis}.json`. Launch argv, logs, GPU pre/postflight, worker receipts, `integrity.json`, before/after provenance, and `validation.json` are alongside them.

Native baseline library SHA256 remains `fd72e97da4d0976a1a8f79fd23def26d2c33883eb135905e563be04bebabd493`. All recorded baseline source/library/model identities matched before and after each run. Large checkpoint hashes are the baseline's prior **full-file** SHA256 values, with current size/mtime verified; the 83.6 GB was **not rehashed** during the bounded probe. Other recorded files were fully rehashed. Parent-owned dirty native `decoder_runtime.inc` and other pre-existing working-tree changes were preserved, not reverted. Diagnostic build/library identity is separate from the baseline.

## Validation and numerical status

- Separate host diagnostic compile/link: **passed**, existing CUDA objects reused unchanged.
- Actual full-checkpoint 128-token unprofiled and separate diagnostic executions: **passed**, complete 48-layer/logits/sampling path, fixed inputs/samples equal, both process exit codes 0, cleanup verified.
- Nsight attempts: **failed to finalize**, raw incomplete streams retained; offline import **failed**. No claim of completed trace validation.
- CPU regression command, 120 s bound:

```sh
CUDA_VISIBLE_DEVICES='' MINI_LLM_BACKEND=numpy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=.:tests .venv/bin/python -m pytest -q \
  tests/test_profile_native_qwen_decode.py \
  tests/test_run_qwen4exp_serving_benchmark.py tests/test_native_chat.py \
  tests/test_native_http.py --tb=short
```

**49 passed in 1.39 s.** New tests cover rejected GPU masks, externally occupied preflight without launching work, detached worker ownership/start-time cleanup, and unique-source guards for diagnostic edits.

The established **48 conditioned layer checks remain passed** with no known equation defect. Strict whole-trajectory equality remains an independent diagnostic, **not green**, and was neither relaxed nor rerun. No parent status file or numerical acceptance criterion was changed.
