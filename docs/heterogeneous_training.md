# Native heterogeneous training setup — experimental, two-GPU smoke verified

This is an opt-in setup, not a replacement for `train_model.py` or an approved
production launcher. It does not use PyTorch. CUDA and HIP workers use separate
Python environments and the existing native CuPy/Triton model, loss, handwritten
backward, gradient clipping, and Hybrid Muon/AdamW optimizer.

## Protocol

`train_heterogeneous.py` is a CPU coordinator with one subprocess per worker.
Worker zero owns the optimizer; every worker holds a complete model replica.
Only worker zero initializes from an optional checkpoint. Its weights are
broadcast before optimizer construction, so all replicas start identically.

Before training, each worker runs a fixed, discarded warmup and all workers
meet at a barrier. Sampling RNG state is restored, gradients/counters cleared,
and no optimizer update occurs. Warmup compiles only the sampled shapes; real
data can still introduce unseen shapes later.

For each optimizer round:

1. Every worker starts at the same weights, with independent sampling RNGs.
2. The named `anchor` worker performs its current target number of microbatches.
   The configured `target_microbatches` is the base/minimum target.
3. Other workers autonomously process their own batch size/context until the
   anchor finishes. No per-microbatch coordinator reply is required. A worker
   already processing a microbatch completes and includes it. The anchor signals
   completion before exporting, allowing peer exports to overlap. An optional
   worker cap stops extra work but waits for the same shared update boundary.
4. Each worker exports FP32 **loss-sum gradients**, plus supervised-token count
   and loss sum. Native token-mean CE backward is scaled by the microbatch's
   valid-token count before accumulation. Padding contributes no loss weight.
5. The coordinator sums gradients in bounded host chunks and divides once by the
   global supervised-token count. Nonfinite inputs/aggregate abort the update.
6. Worker zero clips once, applies one native optimizer update, and publishes
   weights. Peers reload them and refresh derived compute buffers before another
   round begins. The optimizer owner retains its already-updated weights rather
   than reloading the same host copy. There are no stale-gradient updates.

The objective is the token-weighted mean over all samples actually executed in
that round. Different context caps and hardware speed change that data mixture;
this is not trajectory-equivalent to fixed single-GPU accumulation. Independent
sampling streams can sample overlapping documents; data are not deduplicated.
Faster workers do not work during the final reduction/update/broadcast barrier.

## Storage and process isolation

Unix socket pairs carry JSON control messages only. Local memory-mapped files
carry FP32 gradients/weights, without GPU peer access, NCCL/RCCL, or shared device
pointers. With N workers and P parameters, bulk files occupy `(N + 2) * P * 4`
bytes: per-worker gradients, one aggregate, and one weights buffer. For two
348M-parameter replicas this is approximately 5.2 GiB, in addition to resident
model/optimizer state and host copies/staging. File-backed storage can incur disk
I/O; a RAM-backed directory consumes host RAM. Choose an existing scratch path
with sufficient space using `--work-dir`; temporary buffers are removed on exit.

Each worker receives common `env` settings then its own `env` settings before
backend import. Inherited `MINI_LLM_*` settings are removed, preventing accidental
reuse of an incompatible production profile. Specify desired fast paths explicitly.
Other inherited environment variables are retained; GPU library paths and Triton
driver selection must be set per worker. Worker logs are written to
`logs/heterogeneous-worker-0.log`, etc. (or configured `log_dir`) and overwritten
on each invocation. EOF or timeout aborts the run and terminates sibling workers.
`timeout_s` bounds a round, including compilation; this is not an indefinite job.

## Configuration

`configs/heterogeneous_smoke.json` contains this machine's working runtime paths,
per-device batch/context caps, a named AMD anchor, and explicit Triton selectors.
The reduced model preserves production's 16 query / 4 KV heads of width 64,
512-token local window, and aligned pooling blocks. It reduces depth to two
layers, vocabulary to 64, and FFN size to 128. Declared context capacity is 512;
workers execute 128 tokens × batch 2 on CUDA and 256 tokens × batch 1 on HIP.
This is a passing synthetic GPU example, not a production configuration.

Runtime-only preflight (CuPy matmul, not model training):

```sh
.venv/bin/python scripts/preflight_heterogeneous.py --config configs/heterogeneous_smoke.json
```

Verified two-update BF16 CUDA+HIP model smoke invocation:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python train_heterogeneous.py --config configs/heterogeneous_smoke.json --steps 2
```

Production mode requires `smoke: false`, explicit `train_shards` (a list of packed
shard paths), and the tokenizer's actual `eos_token_id`. The coordinator currently
uses document-first sampling from a uniformly selected shard, not the production
mixture/source scheduler. The inspected `token_shards_ensemble_32k_mixed` directory
was empty; no production data paths or EOS IDs were guessed. Context caps may
not exceed the model's configured context. GPU device ordinals are local to each
runtime, so both workers can legitimately use device `0`.

Optional `init_from` is **weights-only initialization**, not optimizer continuation;
it requires `acknowledge_nonexact_init: true` and an exactly matching config/dtype.
There is intentionally no exact-resume option. `optimizer` accepts the native
Hybrid Muon constructor options; defaults use constant LR 1.5e-4, not the
production warmup/cosine schedule. There is no automated validation, source
metrics, periodic saving, or automatic per-device batch/context selection.

Optional `checkpoint_dir` must be a **new** output path. The final save includes
native model weights, optimizer state, config and round metrics. It does not
include a tokenizer, worker RNG states, data cursors, or a resume contract, and
is explicitly marked `exact_resume_supported: false`. Never point it at an
existing production checkpoint. Do not use this path for unattended production
training until full-size/long-context validation and the required production
lifecycle are added.

## Warmup, scheduling controls, and timings

- `warmup_microbatches`: nonnegative integer, default **1 per worker**; the GPU
  example uses **2**. Set 0 to disable. Warmup is not counted toward the round's
  loss, tokens, gradients, or optimizer steps. A fixed-count CPU regression proves
  identical model and optimizer updates with warmup 0 versus 2.
- `target_compute_fraction`: optional fraction strictly between 0 and 1. Omit it
  to retain fixed anchor targets. This targets training-path wall time, **not
  hardware GPU occupancy or utilization**.
- `max_anchor_microbatches`: required explicit ceiling when adaptation is enabled;
  must be at least the base target. The GPU example uses fraction **0.8**, base
  **2**, maximum **16**. Targets use a smoothed estimate (EMA weight 0.2) of
  anchor microbatch time and shared-update overhead, including export/tail wait:

  ```text
  desired_compute_s = overhead_s * fraction / (1 - fraction)
  next_target = clamp(ceil(desired_compute_s / microbatch_s), base, maximum)
  ```

  This is a bounded heuristic, not an occupancy optimizer or guaranteed achieved
  fraction. The maximum can prevent attaining the requested fraction.
- Per-worker `max_microbatches`: optional positive cap to bound a fast worker's
  contribution. A capped peer waits for the anchor rather than taking an early
  update. The anchor's cap, if specified, must cover its maximum target. Caps
  trade utilization for tighter bounds on the training mixture.

Round JSON retains the previous loss/token/count fields and adds `anchor_target`,
`next_anchor_target`, per-worker `workers` measurements, and `timing_s` phases:

- `compute_s`: sampling, forward, loss, backward, and device synchronization;
- `export_s`: worker-to-host gradient export;
- `wait_for_stop_s`: waiting for the finish request after a worker cap;
- `compute_fraction`: worker compute time divided by end-to-end round time;
- `supervised_tokens_per_compute_s`: worker supervised-token rate while computing;
- `timing_s.accumulation`: all accumulation and gradient exports;
- `timing_s.reduce`, `.update`, `.sync`: host reduction, owner update/publication,
  and peer weight synchronization respectively.

The first result also contains `warmup` timings/counts, outside round timing.
Phase timings are wall-clock measurements; export can overlap peer computation,
and per-worker timings must not be added together as serial phase costs.

### Measured reduced-model comparison

Both modes used native BF16 CUDA+HIP/Triton, discarded two warmup microbatches per
worker, and ran five optimizer rounds. The summary below excludes the first
update (which can still compile optimizer kernels). Fixed mode used target 2;
adaptive mode started at 2 and then used its cap of 16. Batch/context caps were
unchanged: CUDA 2×128 and HIP 1×256.

| Steady-round measurement | Fixed target | Adaptive target |
|---|---:|---:|
| Supervised tokens/second | 8,912.7 | 21,571.3 |
| NVIDIA training-path wall-time fraction | 34.2% | 78.8% |
| AMD training-path wall-time fraction | 29.2% | 76.6% |

This is approximately **2.42× synthetic token throughput**, not a production
benchmark or convergence comparison. Adaptation intentionally increases effective
batch size and decreases optimizer-update frequency per token. It can change the
relative short-/long-context sample mixture. A learning-rate/token-budget policy
must account for that before production training. No claim of faster time to
quality follows from this measurement. Per-step results are saved locally in
`logs/heterogeneous_scheduling_comparison.json`.

## Verified state and remaining work

- Independent CUDA (RTX 5080) and HIP (`gfx1201`) CuPy 14.2 runtimes passed exact
  16×16 FP32 matrix probes. No package installations were needed; see
  `heterogeneous_gpu_environment.md` for library paths and Triton metadata.
- CPU verification passed:

  ```sh
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest tests/test_heterogeneous_training.py -q
  ```

  The original numerical/protocol suite remains **17 passed**. This includes two real subprocess replicas with different
  contexts/batches, two updates, checkpoint optimizer-state saving, chunked
  unequal-token reduction, nonfinite rejection, and gradient agreement against
  serial native full-logit training with masked targets.
- Before warmup/autonomous/adaptive scheduling was added, the revised BF16
  CUDA+HIP/Triton smoke completed **two shared optimizer updates** successfully:

  | Round | NVIDIA microbatches | AMD microbatches | Supervised tokens | Loss | Gradient norm |
  |---|---:|---:|---:|---:|---:|
  | 1 | 244 | 2 | 62,976 | 4.459728 | 2.207063 |
  | 2 | 2 | 2 | 1,024 | 4.429168 | 5.135115 |

  Round times were 4.573 s and 0.244 s. These are synthetic smoke observations,
  not production throughput measurements. The cold first-round imbalance is
  consistent with asymmetric compilation/startup time. That observation motivated
  the now-implemented discarded warmup and optional worker contribution caps.
- Earlier fixture failures (8-wide sparse heads, four-query-head Triton geometry,
  and context capacity below the local window) were configuration issues. The
  smoke now retains the production attention ABI; no attention kernels changed.
- `tests/test_heterogeneous_gpu.py` makes the two-update physical-GPU check opt-in:

  ```sh
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_BACKEND=numpy MINI_LLM_TEST_HETEROGENEOUS_GPU=1 .venv/bin/python -m pytest tests/test_heterogeneous_training.py tests/test_heterogeneous_scheduling.py tests/test_heterogeneous_gpu.py -q
  ```

The current combined opt-in command above passed: **69 passed in 11.64 s**,
including both physical-GPU subprocesses, numerical/protocol tests, bounded
scheduler validation, and exact warmup-state preservation. A read-only review
also found no blocking event-buffering, stop/export, or capped-worker races.

No existing kernels, single-device training implementations, environments, or
production checkpoints were modified. Still required before a production run:
full-size/long-context memory and numerical validation, per-device context/batch
tuning, production data-mixture and LR scheduling, validation/periodic saving, and an
explicit optimizer/RNG/data-state continuation contract. A passing reduced smoke
does not establish those properties.
