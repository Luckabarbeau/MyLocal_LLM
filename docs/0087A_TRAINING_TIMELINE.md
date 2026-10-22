# 0087A — CuPy training GPU/CPU timeline

## Scope and safety

Patch 0087A asks whether Python/CuPy launch orchestration leaves enough avoidable GPU starvation to justify a native/C++ rewrite. It does **not** benchmark or modify the optimizer, save a checkpoint, update weights, change the sampler, or change the production attention backend.

The standalone probe calls the exact production autoregressive training seam:

1. `DecoderLanguageModel.forward_body`
2. `DecoderLanguageModel.chunked_lm_head_loss_forward`
3. `DecoderLanguageModel.chunked_lm_head_backward`
4. `DecoderLanguageModel.backward_body`

It loads model tensors with `skip_optimizer=True`, clears gradients outside measured windows, and never constructs an optimizer. Samples come from the existing weighted mixed-shard manifest using `sampling_policy="document_first"`, the packed EOS document index, and `ar_compact_metadata`. No unrelated-document packing or dense 64K mask is created.

This report covers **0087A only**. It does not contain a PyTorch/Torch-compile result or a GEMM-efficiency verdict.

## Audited production settings

The user-provided production invocation establishes these settings for this audit:

- `MINI_LLM_BACKEND=cupy`
- grouped production attention (`MINI_LLM_TRAIN_ATTN_BACKEND=grouped`)
- activation checkpointing `none`
- historical encoder recomputation enabled
- history query chunk 1024
- local probability recomputation enabled
- local query chunk `auto`
- LM-head chunk 8192
- packed grouped-cuBLAS pointer upload and compact descriptors enabled
- prepared RoPE positions enabled
- AR compaction and accumulation-length buckets enabled
- fused packed FFN gate/up disabled

The probe applies these only with `os.environ.setdefault`, so explicit process-level A/B overrides still win. The optimizer-offload flags are not operational because no optimizer is constructed.

## Machine and tool audit

Measured 2026-10-22 on:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, SM 12.0 |
| VRAM | 16,303 MiB |
| Driver | 580.178.04 |
| Python | 3.12.3 |
| CuPy | 14.2.0 |
| CUDA runtime reported by CuPy | 13.2 (`13020`) |
| Nsight Systems | 2026.3.2.476-263238834031v0 |
| Model | 338,134,308 parameters, 20×1024, FFN 4096, 16:4 GQA, BF16 |
| Checkpoint | `checkpoints/0083_intermediate_20L_1024_fullmix_100k` |
| Dataset | `token_shards_ensemble_32k_mixed` |

Nsight Compute hardware counters were not collected. These results do not establish SM occupancy, Tensor Core active percentage, DRAM bandwidth, or achieved hardware FLOP/s.

## Implementation

### `benchmarks/profile_0087a_training.py`

- Loads the checkpoint without optimizer/training state.
- Samples real EOS-bounded data and reports source, shard, document spans, original/compacted rows, supervised rows, history lengths, and configured active-head patterns.
- Supports naturally occurring compact lengths through `--working-length auto` or a requested 128-token bucket. Requested lengths condition real document-first samples; they do not truncate or synthesize examples.
- Measures complete F+B with CUDA events and outer synchronized host wall time. Setup, warmup, checkpoint I/O, sampling, and `model.zero_grad()` are outside timing.
- Optionally translates existing synchronized `performance_scope` names to low-overhead CuPy NVTX push/pop ranges only in the standalone process. Production modules are not edited.
- Builds a tiny header-only NVTX3 C helper in `/tmp` to emit the **registered** capture range `PROFILE_0087A_MICROBATCH@MyLocal_LLM_0087A`. Ordinary CuPy `RangePush` strings do not trigger the installed Nsight registered-string capture filter.
- Writes capture JSON before closing the registered range because `--capture-range-end=stop-shutdown` terminates the target immediately. The analyzer excludes this serialization tail.

### `benchmarks/analyze_0087a_nsys.py`

- Discovers SQLite tables and columns from `sqlite_master` and `PRAGMA table_info`.
- Resolves both inline NVTX text and `textId → StringIds` values.
- Selects CUDA activities by correlation IDs whose CUDA API submissions overlap `microbatch_total`.
- Extends the CPU NVTX measurement boundary through the final associated GPU operation. This matters because queued GPU work finishes after the Python push/pop range closes.
- Computes a union of kernel, memcpy, and memset intervals rather than summing overlapped stream durations.
- Separates already-queued work, correlated candidate host-starvation time, synchronizations, API time, and enqueue delay.
- Emits JSON plus interval, kernel, phase, and API CSV files.

### Marker limitation

The checkpoint represents both head 0 (window 4096, effectively full causal at this shape) and heads 1–15 (window 512) as attention `kind="local"`. The existing production scope exposes both as `attention.local.*`. Therefore the emitted `attention_local` range is a **combined full-current + sliding-local current-attention range**. No trustworthy separate `attention_full` marker was fabricated. Separating it requires a future guarded marker carrying group window/head identity in `mini_llm/ops/attention.py`; that was intentionally left out of this no-hot-path-change patch.

All other requested coarse labels were emitted when active. `route_validity` is reported as `not_applicable_historical_read` because this checkpoint uses scheduled read-only historical heads rather than routed-prefix selection.

## Commands

Run from the repository root. The probe contains the supplied production defaults, but retaining explicit backend and grouped-attention settings makes the baseline unambiguous.

### Tests

```bash
python -m py_compile \
  benchmarks/profile_0087a_training.py \
  benchmarks/analyze_0087a_nsys.py \
  tests/test_0087a_profile_ranges.py

python -m pytest -q tests/test_0087a_profile_ranges.py

# Optional registered-NVTX/CuPy smoke
MINI_LLM_RUN_0087A_GPU_SMOKE=1 \
python -m pytest -q \
  tests/test_0087a_profile_ranges.py::test_registered_nvtx_helper_gpu_smoke
```

### Unprofiled timing

Use at least 20 warmed samples for short bounded runs:

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_TRAIN_ATTN_BACKEND=grouped \
python benchmarks/profile_0087a_training.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./token_shards_ensemble_32k_mixed \
  --batch-size 3 --microbatches 1 --working-length 2304 \
  --warmup 2 --repeats 20 --no-optimizer \
  --out benchmarks/results/0087a_b3_t2304_unprofiled.json
```

Repeat with `--working-length 1024`, `2048`, and `4096`. Add `--annotations` in an independent matched process to measure annotation overhead.

### Nsight Systems capture

The domain suffix is required by Nsight 2026.3. A plain unregistered CuPy marker does not trigger this capture.

```bash
rm -f benchmarks/results/0087a_microbatch.nsys-rep \
      benchmarks/results/0087a_microbatch.sqlite

MINI_LLM_BACKEND=cupy MINI_LLM_TRAIN_ATTN_BACKEND=grouped \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=PROFILE_0087A_MICROBATCH@MyLocal_LLM_0087A \
  --capture-range-end=stop-shutdown \
  --output=benchmarks/results/0087a_microbatch \
  python benchmarks/profile_0087a_training.py \
    --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --batch-size 3 --microbatches 1 --working-length 4096 \
    --seed 42 --warmup 2 --no-optimizer --capture \
    --out benchmarks/results/0087a_profile.json
```

`stop-shutdown` may make the target command return 143 after Nsight has generated the report. The JSON is durably written before the registered capture range ends.

```bash
nsys stats --force-export=true \
  --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  benchmarks/results/0087a_microbatch.nsys-rep \
  > benchmarks/results/0087a_nsys_stats.csv

nsys stats --force-export=true \
  --report nvtx_gpu_proj_sum --format csv \
  benchmarks/results/0087a_microbatch.nsys-rep \
  > benchmarks/results/0087a_nvtx_gpu_proj_sum.csv

python benchmarks/analyze_0087a_nsys.py \
  benchmarks/results/0087a_microbatch.nsys-rep \
  --sqlite benchmarks/results/0087a_microbatch.sqlite \
  --output-json benchmarks/results/0087a_timeline.json \
  --output-prefix benchmarks/results/0087a_timeline
```

## Unprofiled measurements

Each row is one fixed real document-first B=3 batch repeatedly executed after two warmups. Conditioning on a compact bucket is for shape coverage and is not a claim about its corpus frequency. Times include full model forward, chunked LM-head loss, numerical LM-head backward, and all 20 block backward passes. They exclude zeroing and optimizer work.

| Compact shape | Computed rows | Supervised rows | GPU event median / p10 / p90 | Synchronized wall median | History |
|---|---:|---:|---:|---:|---|
| B3/T1024 | 3,072 | 2,671 | 137.465 / 135.785 / 138.803 ms | 137.625 ms | none in selected batch |
| B3/T2048 | 6,144 | 3,669 | 278.747 / 276.742 / 280.343 ms | 278.759 ms | none in selected batch |
| B3/T2304 | 6,912 | 3,489 | 312.990 / 311.466 / 315.370 ms | 313.164 ms | none in selected batch |
| B3/T4096 | 12,288 | 6,657 | 662.723 / 652.053 / 674.785 ms | 662.903 ms | `[0, 0, 9284]` valid historical tokens |

The T4096 case was found after six document-first sampling attempts. Its active-local-head counts by layer were:

| Active current local heads | Layer count |
|---:|---:|
| 15 | 8 |
| 13 | 7 |
| 11 | 4 |
| 9 | 1 |

Head 0 remained the original full-current causal head. Historical heads retained their original head numbers and GQA associations.

### Instrumentation overhead

Matched B3/T4096, 20 samples per independent process:

| Mode | GPU-event median | Wall median |
|---|---:|---:|
| annotations off | 662.723 ms | 662.903 ms |
| annotations on | 663.696 ms | 663.795 ms |
| relative difference | +0.147% | +0.135% |

The single Nsight-profiled sample measured 668.652 ms by GPU event and 668.775 ms by synchronized wall, about 0.9% above the unprofiled median and inside the unprofiled p10–p90 spread. Profiled timing is diagnostic; the unprofiled table is authoritative.

## Timeline results

### Boundaries and interval union

The registered capture range lasted 763.965 ms because it includes durable JSON serialization before `stop-shutdown`. That tail is excluded.

- CPU `microbatch_total` push/pop duration: 636.786 ms.
- Nsight correlation-aware projected duration: 668.443 ms.
- Analyzer correlation-projected boundary: 668.710 ms (0.04% from Nsight projection).
- Associated CUDA activities: 15,897 total: 13,988 kernels, 1,477 memcopies, 432 memsets.
- Union of associated GPU kernel/memory intervals: **624.103 ms**.
- GPU-execution-gap proxy: **6.671%** (44.608 ms total gaps).
- Internal gap median / p90 / max: **1.440 / 2.528 / 1,340.021 µs**.

The 6.671% value is not an occupancy or utilization number. It contains device-side dependencies, stream scheduling, library behavior, launch latency, and host starvation. It is not all recoverable by C++.

### Host submission evidence

- Correlated GPU events: 15,897.
- Events whose work was already submitted before execution: **15,890**.
- Candidate host-starvation time above the 10 µs attribution threshold: **1.494 ms**.
- Candidate avoidable host-starvation fraction: **0.2235%**.
- Fixed-GPU-work speed upper bound from removing only that measured starvation:

```text
1 / (1 - 0.00223485) = 1.00224×
```

That is approximately a **0.224%** upper bound under this capture’s workload and boundaries, not a whole-step guarantee. The median enqueue lead was 28.283 ms: Python/CuPy generally kept substantial work queued rather than feeding the GPU one kernel at a time.

Synchronization records are diagnostic, not additive: runtime API records and CUPTI synchronization activities can describe the same synchronization from different views. The analyzer observed 4,434 such records totaling 33.031 ms in their respective CPU/activity intervals; this must not be subtracted from wall time or labeled avoidable without a causal dependency analysis.

### Correlation-aware NVTX GPU projection

Nsight `nvtx_gpu_proj_sum` associates GPU operations with the CPU NVTX stack. These ranges are nested and **must not be added**.

| Range | Total projected GPU span/work attribution | Instances | Interpretation |
|---|---:|---:|---|
| `microbatch_total` | 668.443 ms | 1 | Complete measured graph |
| `backward` / `gradient_accumulation` | 460.738 ms | 1 each | Same nested backward region |
| `attention_backward_recompute` | 255.282 ms | 20 | Block replay/attention-backward envelope |
| `attention_local` | 205.262 ms | 80 | Combined full-window head and local-512 groups |
| `forward` | 197.697 ms | 1 | Complete forward envelope |
| `ffn_backward` | 146.513 ms | 20 | FFN backward submissions |
| `historical_retrieval` | 74.521 ms | 26 | Historical read/recompute submissions |
| `ffn` | 71.812 ms | 20 | FFN forward submissions |
| `lm_head_loss` | 41.961 ms | 2 | Chunked projection/loss F+B |
| `rmsnorm_residual` | 11.987 ms | 82 | Norm/residual scopes |

“Projected time” is profiler-derived attribution, not an independent CUDA-event latency. Parent and child ranges overlap, and attention ranges overlap replay/backward envelopes.

### Kernel ranking

Top individual kernel symbols by summed GPU duration in the one profiled sample include:

| Kernel/group | Calls | Summed GPU duration |
|---|---:|---:|
| `nvjet_sm120...mma_256x128x64...NNNN` | 40 | 41.391 ms |
| `nvjet_sm120...mma_128x80x64...NTNN` | 38 | 39.810 ms |
| CUTLASS BF16 128×256 TN | 812 | 38.915 ms |
| `nvjet_sm120...mma_128x80x64...TNNN` | 38 | 38.724 ms |
| `nvjet_sm120...mma_256x128x64...TNNN` | 23 | 38.325 ms |
| CUTLASS BF16+ReLU 64×64 | 109 | 33.784 ms |
| `local_causal_softmax_recompute_bwd_bf16` | 418 | 33.169 ms |
| `local_causal_softmax_fwd_bf16` | 440 | 27.054 ms |
| FP32 CuPy add | 740 | 23.307 ms |
| grouped CUTLASS BF16 128×64 | 344 | 21.290 ms |

Summed kernel durations are useful for ranking but are not wall time when streams overlap. Kernel names and existence do not prove Tensor Core utilization or arithmetic efficiency.

## Raw artifacts

Generated locally under `benchmarks/results/`:

- `0087a_b3_t1024_unprofiled.json`
- `0087a_b3_t2048_unprofiled.json`
- `0087a_b3_t2304_unprofiled.json`
- `0087a_b3_t4096_unprofiled.json`
- `0087a_b3_t4096_annotated.json`
- `0087a_profile.json`
- `0087a_microbatch.nsys-rep` (1.6 MiB)
- `0087a_microbatch.sqlite` (4.1 MiB)
- `0087a_nsys_stats.csv`
- `0087a_nvtx_gpu_proj_sum.csv`
- `0087a_timeline.json`
- `0087a_timeline_intervals.csv`
- `0087a_timeline_kernels.csv`
- `0087a_timeline_phases.csv`
- `0087a_timeline_apis.csv`

The binary trace and generated result files are deliberately not embedded in the Git patch. Their paths are stable and the direct commands above reproduce them.

## Validation

Passed:

```text
python -m py_compile ...                                  passed
python -m pytest -q tests/test_0087a_profile_ranges.py   4 passed, 1 skipped
MINI_LLM_RUN_0087A_GPU_SMOKE=1 ...                       1 passed
B3/T1024, T2048, T2304, T4096 full F+B                  completed
registered-domain Nsight capture                         completed
SQLite analysis and CSV/JSON export                      completed
```

The default skip is the opt-in GPU helper smoke; it passed when explicitly enabled.

## Gate A decision

**Gate A favors GPU kernel/work-partition investigation, not a mechanical C++ rewrite.**

Evidence:

1. GPU work was already queued for 15,890 of 15,897 correlated activities.
2. Only 1.494 ms, 0.2235% of the projected microbatch boundary, was attributable to measured host-submission starvation under the conservative correlation rule.
3. Eliminating that measured starvation has an upper bound of about 1.00224× for fixed GPU work.
4. The larger 6.671% execution-gap proxy cannot be assigned wholesale to Python; most gaps occurred despite already-queued work.
5. Attention backward/recomputation, BF16 GEMMs, local softmax recomputation, conversion/add kernels, and shared scheduling dominate the trace attribution and kernel ranking.

This does **not** prove that CUDA Graphs or native scheduling can never help, and it does not extrapolate one microbatch to optimizer offload. It does show that ordinary Python launch starvation is not a significant bottleneck in this measured full-model F+B graph. If subsequent work continues, 0087B should compare a numerically equivalent compiled layer and 0087C should measure the weighted GEMM shapes. A CUDA Graph feasibility study is lower priority unless a future full optimizer-step trace reveals host-starved intervals absent from this no-optimizer microbatch.

## Remaining uncertainty

- No optimizer/offload/update range was measured, by design. The conclusion applies to forward + numerical backward only.
- Exact accumulation-length rebucketing across all 44 production microbatches was not replayed; multiple real compact shapes were measured independently.
- A standalone `attention_full` phase is unavailable from current production scopes; current-attention attribution combines head 0 with local-512 groups.
- Phase projections are nested and profiler-perturbed; they are not independently additive timings.
- No Nsight Compute hardware counters were collected.
- One trace cannot characterize thermal/clock variability across long optimizer steps. The unprofiled 20-sample distributions are the timing authority.
