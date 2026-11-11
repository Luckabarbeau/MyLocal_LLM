# 0106R — Guarded generation benchmarking and measured cache tuning

## Scope and preserved state

Continues 0106P/Q on baseline `f133d3f2bafd47d7edc21b3fdb452a90a347a2db`.
This incremental change adds diagnostics, tests and a generation-only benchmark.
It does not change model equations, routing, quantization, tolerances, training,
native inference, serving APIs, or executor policy defaults. No commits or
branches were created. Existing untracked P/Q work is preserved.

**Verified:** installed Qwen3-Coder-Next UD-Q4_K_XL generated two serial
16-token completions on the RTX 5080, at each of two bounded expert-cache sizes.
All four completions had identical token IDs. This is a short raw-completion
baseline, not a claim of production readiness or strict real-model reference parity.

## Implementation

- `mini_llm/diagnostics/generation_profile.py`: retained surviving profiler;
  wraps existing per-block normalization, mixer, MoE, router and expert calls.
  Inclusive host wall intervals are nested and **must not be added together**.
  No per-layer synchronization or CUDA events are introduced by this profiler.
- `mini_llm/diagnostics/host_memory.py`: backend-independent Linux observations
  and fail-closed checks for missing availability/RSS, insufficient available RAM,
  or current/reported peak RSS above the configured ceiling. Loading preflight
  also reserves the remaining planned RSS allowance. Swap is observed, not used
  as RAM headroom. No cache clearing, swap changes, RLIMIT_AS, or full-file hashing.
- `benchmarks/benchmark_0106r_generation.py`: existing `ServingModel` and
  `GenerationSession`, greedy sampling, serial requests with fresh generation
  state and retained model/cache. No reference subprocess or full-forward replay.
  Separates actual prefill/decode routing; asserts per-layer token counts and
  state lengths. Bounded prompt, context, request/token count and detailed traces;
  refuses output overwrite. Saves completed requests and failure diagnostics.
- Three `tests/test_0106r_*.py` files cover profiler restoration and exceptions,
  memory parsing/guards, safe import/help/preflight, CLI bounds, repeated tiny
  generation, routing counts, latency denominators, and one state-release sample.

One local benchmark bookkeeping correction removed the duplicate final memory
observation. No production implementation was replaced.

### Observation and safety limitations

The guard is **soft**, not an OS memory limit. It samples before/after model
forwards and state creation, not every layer/allocation, and cannot prevent a
sudden competing application allocation. Linux RSS includes resident mmap pages
and is not equivalent to unreclaimable anonymous RAM. Finite cgroup-v2 **leaf**
headroom is included; ancestor limits are not traversed. Unresolvable membership
is explicitly reported as host-only; malformed detected leaf budgets fail closed.

`memory_plan` in these benchmark reports is the model's **single-token decode**
estimate at the requested state horizon; multi-token prefill has additional
workspace and is subject to the model's existing runtime checks and private pool
limit. Pool usage is not process/global VRAM, CUDA context usage, or a sampled
whole-device peak. The pool limit does not constrain all driver allocations.

Forward timings include current-stream completion at forward boundaries.
Stage timings are host attribution, not isolated GPU kernel times; executor
worker intervals overlap. Summary profiling avoids additional per-layer event
profiling. `memory_observation_ms` covers checks inside request timing, excluding
the final post-release observation/GC. TTFT is the first sampled token after the
request starts, **excluding model loading**. First visible text is separate.

The final sampled token is not forwarded: 16 generated tokens use 15 decode
calls. Decode tokens/s uses these actual calls, not 16 divided by their latency.
End-to-end completion tokens/s includes prefill and the request's observations.

## Reproduce

Use one real-model process at a time; confirm user workloads and RAM headroom first.
From the repository root:

```sh
MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  benchmarks/benchmark_0106r_generation.py \
  --model /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --max-new-tokens 16 --requests 2 --context 64 \
  --cpu-workers 2 --expert-cache-mib 256 \
  --device-budget-mib 4096 --device-reserve-mib 2048 \
  --ram-reserve-mib 16384 --max-rss-mib 12288 \
  --output benchmarks/results/0106r_real_generation_256.json
```

For the second configuration change only `--expert-cache-mib 1024` and use a
new output path. Recorded raw outputs are in `benchmarks/results/` (ignored
local artifacts); `0106r_validation.json` retains compact measured results.
Use a fresh filename for subsequent runs; the CLI deliberately refuses overwrite.

Tiny CPU/CUDA generation needs no downloaded model:

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  benchmarks/benchmark_0106r_generation.py --synthetic --context 32 \
  --max-new-tokens 4 --requests 2 --prefill-chunk-tokens 2 \
  --expert-cache-mib 1 --output benchmarks/results/0106r_synthetic_cpu.json
```

For tiny CUDA use `MINI_LLM_BACKEND=cupy`, `--device-budget-mib 512` and a new
output filename. Context 32 is explicit because the fixture's metadata horizon
is smaller than the real-workload CLI default. No generated code is executed.

## Real workload results

Hardware: RTX 5080 16 GB, installed mixed GGUF, FP32 activations, native packed
CPU experts, two CPU workers. Prompt: `def add(a, b):\n    return` (8 tokens).
Context 64; 16 output tokens; 48 layers, 10 routed experts per token per layer.
GPU planning/pool budget 4 GiB, free-VRAM reserve 2 GiB, RAM reserve 16 GiB,
RSS ceiling 12 GiB. The 256 MiB process finished before the 1024 MiB process began.

| Cache | Request | First sampled token | Prefill tokens/s | Decode forward tokens/s | Completion tokens/s, end-to-end | Decode cache hits | CPU assignment fraction | Expert upload MiB/decode call |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 MiB | First use | 9.149 s | 0.875 | 1.393 | 0.803 | 13.14% | 73.47% | 116.61 |
| 256 MiB | Warm | 4.303 s | 1.860 | 1.438 | 1.085 | 19.46% | 77.47% | 26.73 |
| 1024 MiB | First use | 2.235 s | 3.585 | 1.764 | 1.488 | 40.33% | 36.29% | 204.00 |
| 1024 MiB | Warm | 3.652 s | 2.192 | 1.746 | 1.305 | 49.65% | 42.39% | 69.52 |

Each decode phase records 720 layer dispatches, 7,200 expert assignments and
15 forwarded tokens per layer. Decode hit rates are hits / 7,200; unlike decode,
prefill groups multiple token assignments into one request for an expert, so its
grouped-request hit rate is not a per-token assignment hit rate. Upload figures
are expert weights only; activation staging bytes are separately recorded.

Warm decode throughput improved **21.43%** in this single short comparison.
The bigger cache reduced CPU assignments but also increased promotion traffic.
Do not infer a universal optimum or isolated cache-caused speedup from one pair.
No page cache or JIT cache was cleared. In particular, the first-use TTFT values
are **not comparable disk-cold starts**. Warm means retained weights, executor
routing history and cache, but fresh request state—not converged cache residency.
The 1 GiB configuration's first request being faster than its second is recorded,
not discarded; scheduling/cache history and normal run variation remain factors.

| Resource observation | 256 MiB | 1024 MiB |
|---|---:|---:|
| Retained backbone estimate | 2,415,209,472 bytes | same |
| Decode memory plan | 2.827 GiB | 3.577 GiB |
| Final private pool reserved | 2.664 GiB | 3.464 GiB |
| Reported process peak RSS | 10.628 GiB | 10.675 GiB |
| Minimum RAM availability at request observations | 52.848 GiB | 52.607 GiB |
| Final process swap | 0 bytes | 0 bytes |

System swap was already in use before the benchmark; it was not changed.

All four completions extended the prompt to:

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

```

This demonstrates successful raw generation, not semantic evaluation over a
coding suite, longer contexts, chat templates, concurrency, or a serving load.

## Bottleneck attribution and next experiment

For warm 256 MiB decode, expert `forward` host attribution was 8,937.79 ms
out of 10,432.60 ms total forward wall (~85.7%). Mixer attribution was 787.69 ms.
At 1024 MiB, experts still accounted for 7,367.07 / 8,591.47 ms (~85.7%).
The summed CPU worker time decreased from 13,449.99 to 7,928.10 ms;
these intervals overlap and are **not** layer critical-path times.
Resolve wait decreased from 6,142.10 to 3,498.71 ms; selection host time
increased from 888.89 to 1,717.32 ms. Promotion host intervals include enqueue
and conversion, not DMA-only time or a transfer-bandwidth measurement.

Evidence supports profiling/tuning expert execution and selection before the
already-small attention/recurrent backbone. Next: a matched-input, bounded
worker-count/cost-calibration comparison (2 versus 4 workers with the same
1 GiB cache), then investigate measured selection overhead/admission decisions.
Do not assume promoting every miss will help: larger cache already increased
uploads/token. Keep the 1 GiB value opt-in; defaults are unchanged. Larger-cache
sweeps, profile seeding, SIMD CPU work and GPU dispatch changes are separate
incremental patches, not speculative additions to this verified state.

## Strata relation and remaining blockers

The prior source-only audit is in `0106i_runtime_audit_and_routing_trace.md`,
with examined Strata revision `504b143342a9fe1aa8320a7b31e11f5c0ac265c3`.
No new upstream checkout or binary was executed for 0106R. Applicable lessons
remain GPU backbone residency, simultaneous CPU/GPU expert work, adaptive
frequency placement, bounded/event-owned staging, and keeping adaptation off
critical paths where possible. This patch supplies generation measurements to
choose among those techniques. No matched Strata throughput comparison is claimed.

Complete Qwen3-Coder-Next raw inference is no longer architecture-blocked.
Remaining work includes practical expert throughput, longer/more representative
requests, validated Qwen chat framing, and broader production/serving behavior.
The actual quantized llama.cpp discrepancy characterized in 0106Q is still open:
no strict real-model numerical-equivalence claim is added here.
Qwen3.8 pruning/remapping, its actual IQ formats, n-gram semantics and SSD backing
remain separate deferred targets; installed Coder-Next metadata does not establish
their support.

## Validation

- New R CPU tests: **137 passed** (profiler 30, host memory 76, benchmark 31).
- Broad CPU/external/native/generation regression: **1,166 passed, 79 skipped**.
  Optional source-linked llama reference integration was not enabled; no real
  reference/model subprocess ran concurrently with real generation.
- Selected CUDA regression: **79 passed, 193 skipped**; CPU-only fixtures skip
  under CuPy. Includes asynchronous ownership, packed numerics/Q6_K, attention,
  recurrence and embedding checks. Skips are not claimed as executed GPU tests.
- Tiny R CPU/CUDA generation: four identical greedy IDs `[47, 219, 210, 204]`
  for both requests on both backends; phase/state assertions passed.
- Existing 0106P synthetic CUDA model benchmark rerun independently: passed,
  including non-default-stream checks; maximum full-logit error versus CPU FP32
  **2.1457672119140625e-6**, versus independent FP64 oracle
  **1.2486889634510945e-6**, existing `rtol=4e-4, atol=8e-6` unchanged.
  That tiny oracle benchmark may replay full forwards; the **real generation
  benchmark does not**. No new real-model logit replay was run.
- Both real generation configurations: completed, all token IDs identical,
  no guard rejection, no allocation failure, no failed correctness tests.

Commands and compact measurements are recorded in `0106r_validation.json`.
