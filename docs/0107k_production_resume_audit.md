# Real-data R9700 Model B resume and fast-path audit

## Run contract and safety

Executed the user's bounded production command: CuPy HIP, Model B sparse20L,
AR BF16-mixed, consumer-gpu, mixed local token shards, context8192, batch1,
accumulation10, supervised-token weighting, Hybrid Muon/AdamW, BF16 Muon with
five Newton–Schulz iterations, total schedule100000, warmup750, peakLR1.5e-4,
gradclip1.0, decay0.1, max-run-steps1. All other supplied validation, monitoring,
seed and benchmark-suite options were retained.

No active training process was found. Source checkpoint:
`checkpoints/model_b_batch1_8k_norecompute_long`, saved trainer step15150.
The earlier `...r9700_resume_test` checkpoint/logs already existed and were not
overwritten. Each test used fresh output paths and read the same original
source; these are **two independent one-update runs**, not consecutive updates.

## Audited production run

Refactored `scripts/probe_0107j_amd_fastpaths.py` to expose `install_audit()`;
its existing synthetic entry point is preserved. New
`scripts/audit_0107k_production_resume.py` installs the same process-local
fail-on-fallback hooks, then executes `train_model.py` with the supplied CLI.
It requires a fresh destination and max-run-steps1, records failure or success,
and additionally compares pooling dispatches against fused pooling executions.
Production model, optimizer and training-loop code were not edited in this task.

Output checkpoint: `checkpoints/model_b_batch1_8k_r9700_production_audit_01`.
CSV: `logs/model_b_batch1_8k_r9700_production_audit_01.csv`.
Audit: `logs/model_b_batch1_8k_r9700_production_audit_01_fastpaths.json`.
The audit JSON includes the complete exact training argument list.

**Passed**: restored state, processed ten real microbatches, applied one optimizer
update, saved trainer step15151, and observed every required optimized path:

| Executed operation | Count |
|---|---:|
| Packed QKV/RoPE forward | 200 |
| BF16 global / sparse forward | 200 / 160 |
| Fused SwiGLU forward/recompute | 400 |
| Fused SwiGLU backward | 200 |
| Fused RMSNorm backward with eligible fused caches | 410 |
| Inverse RoPE backward | 800 |
| FP32-output HIP projection wgrad | 400 |
| Separate local KV backward | 200 |
| Segmented global/sparse KV backward | 360 |
| Pooling dispatch / fused pooling backward | **200 / 200** |
| Actual fused CE loss / gradient kernels | 10 / 10 |

The audit rejects non-None-required fusion fallbacks. These counters describe
actual real-batch executions, not merely enabled flags. Audit interval wall time
was11.24s including training launch work and final save. Its CSV cumulative
steps/sec implies approximately8.03s/update; audit timing is instrumented and
is not a warmed production throughput estimate.

## Uninstrumented matched production run

Ran the same CLI without hooks, changing only output paths to fresh
`...r9700_production_timing_01` names. `script -q -e -c` recorded the complete
console and preserved the child exit code. Training exited successfully.

- Checkpoint: `checkpoints/model_b_batch1_8k_r9700_production_timing_01`
- CSV: `logs/model_b_batch1_8k_r9700_production_timing_01.csv`
- Console: `logs/model_b_batch1_8k_r9700_production_timing_01_console.txt`

| Metric | Result |
|---|---:|
| Trainer step | 15150 -> 15151 |
| Training loss | 1.9450486898422241 |
| Gradient norm | 1.4613425731658936 |
| LR | 0.00014310849780979997 |
| Supervised tokens this update | 40,836 |
| Target fill | 49.8% |
| Logged step time | **7,291.3 ms** |
| Supervised tokens/sec | **5,601** |
| Executed tokens/sec | **5,925** |
| Capacity tokens/sec | **11,235** |

Versus the user's supplied earlier log (22,531.9ms,1812 supervised tok/s,
1917 executed tok/s), this is approximately3.09x faster. That is a historical
comparison, not a controlled multi-step warmup study. The existing earlier test
CSV at the time of inspection already contained a ~7.09s optimized update;
it was preserved, not relabeled as the original slow baseline.

The audited and uninstrumented CSVs have identical loss, LR, gradient norm,
processed/supervised token counters for this source state and data selection.
Both updates passed the training loop's finite-gradient checks. No full
parameter-by-parameter comparison of their saved checkpoints was performed.

The production rate is **not19k supervised tok/s**. The earlier18.9k result was
fixed-length fully scored synthetic forward/backward excluding optimizer/data
sampling. This run includes the real training/optimizer work and variable-length
documents, with49.8% target fill. Capacity tokens are configured slots, not
executed or supervised work. One update does not establish steady-state speed
or isolate the exact contributions of sampling, optimizer offload, JIT and
monitoring; profiling a warmed multi-update production run is the next experiment.

## Saved optimized policy and resume warnings

Both output checkpoints contain config, tokenizer, mixture manifest,
model_params, optimizer_state and training_state. Saved contract verifies
context8192, accumulation10 and hybrid-muon, with all new HIP policies:

```text
MINI_LLM_HIP_SPLIT_KV_BACKWARD=2
MINI_LLM_HIP_DQ_LOOP_STAGES=1
MINI_LLM_HIP_DENSE_BACKWARD_TILES=1
MINI_LLM_HIP_DENSE_KV_BACKWARD=1
MINI_LLM_HIP_FP32_PROJECTION_WGRAD=1
```

The original source still reports trainer step15150; both destinations report
15151. The original and previous test output paths were not written by these runs.

The command intentionally used `--allow-nonexact-resume`. Warnings include the
requested accumulation change1024->10, fine->geometric shape buckets, local
probability recomputation0->1, and newly recorded HIP policies. These changes
were accepted explicitly, not silently ignored.

The pre-existing warning also remains:

```text
preserving checkpoint trainer/optimizer step offset: 15150 != 15200
```

This task did not reset or reconcile the saved50-step offset. Review its history
before claiming schedule-exact continuation; the same warning was present in
the user's original run. For further continuation, resume from the chosen new
step15151 checkpoint rather than repeating the old step15150 source.

Validation did not fire at step15151 with val-interval100. Component-update CSV
was not created: interval10 was not due for this one-update run. This is not a
monitor failure. No multi-step stability/convergence or validation-quality claim
is made. The100000-step schedule horizon was retained while max-run-steps1
bounded execution.

## Reuse

For another audit, prefix the user's train options with:

```sh
sh scripts/run_r9700_preflight.sh -u scripts/audit_0107k_production_resume.py \
  --audit-report ./logs/NEW_UNIQUE_fastpaths.json
```

Supply all ordinary `train_model.py` options after this prefix, including a new
checkpoint/log destination and max-run-steps1. The runner refuses overwriting
prior outputs. The complete argument list used here is saved in the audit JSON;
choose fresh paths before replaying it.

For ordinary training use the original `train_model.py` entry point and AMD
wrapper. No extra optimization exports are needed for the tested consumer-gpu
path. Remove the one-update bound only when intentionally starting a longer run.

`git --no-pager diff --check` passed after the audit-harness changes.
