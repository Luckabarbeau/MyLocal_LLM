# WP9 — actual-checkpoint native HTTP baseline, GPU1 only

## Result

**PASS: full two-shard checkpoint startup, two bounded HTTP/SSE requests, exact-prompt checkpoint rollback, and owned-server cleanup.** This is an experimental native baseline, not a Strata comparison or model-quality acceptance. No native kernel, equation, tolerance, training process, parent-owned implementation status, or native-core source was changed by this task.

Artifacts: `benchmarks/native_http_gpu1_new/`. Machine-readable results are `baseline.json`; complete file hashes/build provenance are `provenance.json`; `launch.json` and `client_launch.json` retain executed argument vectors and selected environment. HTTP request payloads, raw SSE bodies/events, parsed responses, before/after health/models/metrics, and resource samples are under `http/0000-*` and `http/0001-*`. Existing paths were not overwritten.

## GPU, placement, arithmetic

Only `CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631` was used by GPU-capable commands and their children. UUID-selected `nvidia-smi` identified physical GPU1, NVIDIA GeForce RTX 5070 Ti, driver `580.178.04`, 16,303 MiB total, 16 MiB used / 15,825 MiB free before and after. CPU-only tests hid GPUs with an empty mask. No GPU0 work/control or training control was performed.

Native policy, reported by actual `/health` and `/metrics`:

- Reference-FP32 HC/PLE/recurrent/KV; QSA GGML CPU; **all routed experts CPU mmap GGML Q8**; native GPU sampler.
- Shared decoder/sampler device allocation budget 10 GiB; dense budget 7 GiB; mapped-file budget 96 GiB; host QSA budget 1 GiB.
- Capacity 256; page size 16; two physical pages; four CPU expert threads; seed 0; queue capacity 1. No all-expert resident bank and no Python tuning/cache policy applied.
- Actual checkpoint size: 54,817,524,224 + 28,800,138,432 = **83,617,662,656 bytes**. Mapping limits are not RSS caps; configured device limits exclude CUDA/context overhead.

Native CUDA build flags include `-O3 -DNDEBUG -std=c++20`, `compute_120/sm_120`, `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. `cmake --build /tmp/mini-llm-wp1-build -- -n` reported **ninja: no work to do**; the existing built library was reused, not rebuilt or replaced. Source hashes include pre-existing parent-owned native changes. The bridge was resolved once using `mini_llm.ops.ggml_packed_cpu._load_bridge()._name`; no cache hash filename was added to configuration or source. The resolved path below is the historical executed argument, not a portable default.

## Executed experiment

Working directory: `/home/lucka/Documents/my_local_LLM/MyLocal_LLM`. Terminal bound: **180 seconds**. Startup bound: **90 seconds**. Each HTTP timeout: **90 seconds**. Experiment deadline: **120 seconds shared by startup and client**, followed by bounded owned-process cleanup.

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=. .venv/bin/python scripts/run_qwen4exp_serving_benchmark.py \
  --output benchmarks/native_http_gpu1_new \
  --gpu-uuid GPU-e961516f-d77d-813c-98e9-2f5403f71631 --backend native \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --native-library /tmp/mini-llm-wp1-build/libmini_llm_native.so \
  --native-bridge-library /home/lucka/.cache/mylocal_llm/native/ggml_packed_cpu_e231d10a715274f9e9e6.so \
  --native-ggml-library /home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so \
  --native-budget-gib 10 --native-dense-budget-gib 7 \
  --native-mapped-budget-gib 96 --native-host-budget-gib 1 --native-threads 4 \
  --context 256 --sizes 128 --max-output 8 --repeat 2 --turns 1 --max-requests 2 \
  --startup-timeout 90 --timeout 90 --experiment-timeout 120 --port 18089 --no-thinking
```

Client children additionally use `MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, retaining the same GPU UUID mask. Greedy temperature 0, stream enabled, thinking disabled; generated synthetic-code workload `production-recovery-coding-v1`. Both request SHA256 values are `b0fee04ba7a614c038e2db729c7718053d367d07d48d76d6472744d23f958a61`.

## Measured results

Startup launch-to-ready file timestamp difference: **7.019 s** (derived timestamp estimate, not a dedicated timer). Health verified the actual loaded model and capacity 256. Both POSTs returned **HTTP 200**, `finish_reason=length`, 128 prompt tokens, eight completion tokens, and SSE `[DONE]`.

| Metric | First submission | Identical repeat |
| --- | ---: | ---: |
| Client request duration | 35.844944 s | 1.166553 s |
| Client first-content TTFT | 34.320745 s | 0.338053 s |
| Native adapter TTFT | 34.263636 s | 0.270017 s |
| Native prefill duration | 33.898822 s | 0 s |
| Native decode duration | 1.656880 s | 0.915019 s |
| Sampled / emitted pieces | 8 / 8 | 8 / 8 |
| Sampled tokens / client request second | 0.223183 | 6.857810 |
| Sampled tokens / native decode second | 4.828351 | 8.742991 |
| Emitted pieces / native decode second | 4.828351 | 8.742991 |
| Actually consumed prefill inputs / second | 127 / 33.898822 = 3.746443 | N/A (no prefill) |
| Reported new / reused prompt tokens | 128 / 0 | 1 / 127 |
| Checkpoint restored | false | true |
| Checkpoint restore duration | 0 s | 0.006825 s |

The final prompt token is processed inside native decode, so the initial `decoder.consume` prefill count is 127, not 128. Native decode timing includes that final prompt-token forward and sampling; these eight-token rates are **not pure steady-state decode throughput**. SSE pieces are not generally tokens; this run separately reports eight samples and eight emitted pieces. Queue-wait time overlaps worker execution and must not be added as an independent compute phase.

Both actual assistant texts were exactly `The issue is in \`compute_1` (deliberately truncated at eight tokens). Equal greedy text is not a model-quality or broad parity test. The repeat explicitly reported `reset_reason=prompt_checkpoint_restore`, `checkpoint_restore=true`, `prefill_seconds=0`, `reused_tokens=127`, `new_prompt_tokens=1`, `require_reset=false`, committed position 135, and **119,065,880 checkpoint host bytes**. Thus this is observed native rollback reuse, not a cache hit inferred from latency.

### Resource samples and cleanup

One-second UUID-selected GPU and selected-PID `/proc` sampling captured 36 first-request samples and two repeat samples:

- GPU device-global used memory: **8,189,378,560 bytes = 7,810 MiB**, stable over sampled requests. This is not native allocator accounting or PID-attributed memory.
- First-request RSS: 4,999,987,200–28,444,028,928 bytes; repeat RSS: 28,373,876,736–28,527,755,264 bytes. RSS includes file-backed mmap pages and does not imply a decoded/resident expert bank.
- Process virtual size: approximately 109.4–109.6 GB, including the mapped checkpoint.
- Process CPU (one core = 100%): first 73.99–183.98%; repeat 203.98% for its one interval. Global GPU utilization first 0–28%, repeat 19–38%; these coarse samples do not isolate a critical path.
- First-request host available memory: 57,688,068,096–58,376,605,696 bytes. Device/host budgets are allocation-domain limits, not whole-process guarantees.

The harness started a new session, signaled only its owned process group, waited/drained, and recorded `server_returncode=0`. Post-run checks confirmed owned PID `589642` absent and port `18089` closed. No existing listener was killed. All nine artifact-level acceptance checks in `baseline.json` passed. Optional benchmark probes `/props` and `/slots` returned 404 as expected; required health/models/metrics returned 200.

## Changes and CPU regressions

- `scripts/run_qwen4exp_serving_benchmark.py`: explicit `--gpu-uuid`, inheritance only from a single full UUID mask, no GPU-index/implicit-GPU0 default; Python remains default backend. Forward native libraries, four budgets, page/thread/seed/queue options, backend and model. Share experiment deadline across startup/client; retain launch/error/exit artifacts and existing owned-process cleanup.
- `scripts/benchmark_qwen4exp_serving.py`: positive `--max-requests` now applies to live workloads as well as replay, so the request bound is enforced rather than merely documented.
- `scripts/serve_qwen4exp.py`: after observing the real log's misleading Python cache/19-worker banner, corrected only native startup reporting to show CPU mmap experts, native threads and native budgets. The measured run predates this banner-only correction; health/metrics were already correct. No generation/backend change.
- `tests/test_run_qwen4exp_serving_benchmark.py`: CPU-only mocked subprocess command/env checks, explicit/inherited UUID, reject index/multi-device masks, Python default, native forwarding/model sizing, client timeout cleanup, existing-listener refusal, live request cap.
- `tests/test_native_http.py`: finite real-handler/mock-worker repeated-prompt checkpoint metrics and accurate native banner/lifetime regression.

Executed before real checkpoint: focused runner/client/server tests **53 passed, 7 subtests passed, 4.10 s**. The first attempt exposed the runner's module-import path issue; one localized import correction resolved it. After adding HTTP regressions, an initial test-only duplicate `max_tokens` argument was corrected by using the fixture's existing default; production generation was unaffected.

Final broad command (120-second bound):

```sh
CUDA_VISIBLE_DEVICES='' MINI_LLM_BACKEND=numpy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:tests .venv/bin/python -m pytest -q tests/test_run_qwen4exp_serving_benchmark.py tests/test_benchmark_qwen4exp_serving.py tests/test_native_http.py tests/test_native_chat.py tests/test_qwen4exp_serving_state.py tests/test_0239_qwen4exp_chat.py --tb=short
```

**PASS: 120 passed, 1 skipped, 7 subtests passed, 19.40 s.** No known new failing validation remains.

## Hashes and numerical status

Full streamed SHA256 over both complete checkpoint shards (not header/sample hashes):

- Shard 1: `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
- Shard 2: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
- Native library: `fd72e97da4d0976a1a8f79fd23def26d2c33883eb135905e563be04bebabd493`
- CPU bridge: `9be971e4091c0eda183e635d41d2bdb8fc24e3ac17b86b4ed90060bf7f7b6f5b`
- GGML CPU library: `924ac13f3e949cb1b62eddff20221600a6efcb1a2b99eb4abd390e75c23df839`

Repository HEAD: `eeeeff9a59a9884162941b3ea8f2915f937ce507`, with preserved pre-existing parent changes. Individual native sources, bindings, runner/client/server, CMake cache and Ninja build hashes are recorded in `provenance.json`, which distinguishes executed and post-banner server source.

`WP7_CONDITIONED_ALL_LAYERS.md` remains established: all 48 same-input conditioned checks passed at the unchanged operator envelope, with no identified decoder equation bug. The independent strict whole-trajectory diagnostic **remains historically failed**, consistent with FP32 reduction differences amplified through discontinuous Q8 quantization/routing/history. It was not hidden, relaxed, replaced, or rerun here. Health and metrics retain that warning and the actual arithmetic policy.

No matched Strata run was executed: **no fair-comparison or competitive-performance claim**. This is only two short requests, not a variance/quality study. The next performance task should profile the **measured critical path of the native repeated-request/decode baseline** before changing kernels; low coarse utilization does not justify an unmeasured optimization or a diversion into batched prefill.
