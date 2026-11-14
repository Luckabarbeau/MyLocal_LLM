# Per-GPU native production diagnostics

Detailed logging is enabled by default when `ProductionPeerSession` attaches.
It does not require a change to the peer JSON, model, optimizer, or saved
heterogeneous contract. Existing running processes are not retrofitted: logging
starts on the next process launch. Do not start a second GPU job alongside an
active training run just to enable these records.

## Files

For the configured production peer log in `logs/`, the default directory is:

```text
logs/gpu_diagnostics/
```

Each invocation creates separate files, with a shared unique run ID:

```text
owner-cuda.<run-id>.jsonl
owner-cuda.<run-id>.manifest.json
peer-hip.<run-id>.jsonl
peer-hip.<run-id>.manifest.json
```

The owner prints its diagnostic filename to the terminal. The peer prints its
filename to the existing peer stdout log. NumPy integration tests use
`owner-numpy` and `peer-numpy` roles. Neither worker truncates another run's files.

The manifests contain GPU/runtime identity, PID, Python/kernel/package versions,
full model configuration, sampling settings and shard paths, and filtered
numerical runtime flags. Secret/token credential environment variables are not
included. CPU-only `_TOKENS` execution-size flags remain visible.

## Recorded information

Each microbatch logs:

- Weight version, target update number, local microbatch ordinal, role, PID,
  run ID, wall-clock nanoseconds and monotonic nanoseconds.
- Full pre-sampling NumPy RNG state, shard cursors, batch/source counters, and
  processed-token count.
- Source and shard path; sampled document/window positions and valid lengths.
- CPU input/target SHA256 hashes, shapes, dtypes, strides and token-ID ranges.
- Effective loss-mask descriptor/range, valid-token count, microbatch weight,
  attention/block metadata descriptors and activation-checkpoint policy.
- Forward, loss, backward, layer and attention method begin/end events, including
  layer indices and tensor descriptors.
- CPU-side CuPy allocator pool accounting at microbatch entry, when available.

Update/boundary events record local and global supervised-token counts, actual
peer contribution, gradient export/reduction, weight publication/import/sync,
owner loss/gradient norm/LR, peer checkpoint sampling state, and transport errors.
Warmup is explicitly tagged `phase: warmup`; its sampling state is restored and
its gradients remain discarded as before.

Exceptions inside instrumented stages log their Python traceback, exception type,
message, active stage stack and microbatch context before propagation. A GPU reset
or abrupt process kill may leave an unmatched stage-begin record; that is useful
as a last breadcrumb. The logger cannot guarantee execution of an exception
handler after a fatal OS signal.

## Timing and data limits

This is **host-side execution logging**, not a complete GPU kernel trace.
`stage_end` means the host call exited (possibly with a separately logged error),
not that all asynchronous device work completed. Timings are explicitly labelled
`host_return_not_gpu_completion`. The first Python synchronization that reports
an error can still be later than the offending kernel.

Logging adds no GPU serialization, `.item()`, device-array downloads, gradient
scans, or extra device synchronization. Existing training transfers and scalar
reads are retained; the owner update log reuses the existing final gradient-norm
read. No hardware-utilization, temperature or power monitoring is claimed: the
per-microbatch memory figures are allocator accounting, not full device VRAM use.

Log records are written directly without a Python output buffer or background
queue, so they do not depend on normal process exit to reach the OS. They are not
`fsync`ed per record: this does not promise durability across host power loss.
There is still CPU and disk-I/O overhead; normal timing-dependent peer membership
can therefore differ with logging enabled. Numerical equivalence is tested with
fixed membership, not claimed for arbitrary wall-clock-scheduled runs.

Raw training tokens, activations, weights and gradients are not dumped. Selected
metadata arrays are limited to the first 32 values; shapes/counts identify any
truncation. Input hashes and sampling state help identify/recreate batches when
the corresponding data and checkpoint are available. These logs alone do not
constitute a full deterministic replay of GPU firmware state or all prior updates.

## Controls

These process environment variables are inherited by both workers and are not
part of the scientific peer contract:

| Variable | Default | Meaning |
|---|---|---|
| `NATIVE_GPU_LOG` | `1` | Set `0` to disable these diagnostic logs. |
| `NATIVE_GPU_LOG_DIR` | peer stdout log directory + `/gpu_diagnostics` | Override the output directory. |
| `NATIVE_GPU_LOG_LAYERS` | `1` | Set `0` to omit per-layer/attention hooks while retaining batch and training-stage logs. |
| `NATIVE_GPU_LOG_MAX_MB` | `64` | Rotate each JSONL stream at approximately this size. |
| `NATIVE_GPU_LOG_BACKUPS` | `4` | Number of rotated files per stream; `0` retains only the active file. |

With the defaults, each worker retains its active stream plus four backups,
approximately 320 MiB per worker / 640 MiB per invocation, excluding manifests.
Individual records are kept whole and can exceed the threshold. Different runs
have separate files, so older invocations are not automatically deleted.

For normal-speed logging, first clear the earlier serialization diagnostics if
those were exported in your shell:

```bash
unset AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY
```

Then launch your existing Python production command unchanged. Optional explicit
settings, before the same command, are:

```bash
export NATIVE_GPU_LOG=1
export NATIVE_GPU_LOG_DIR=./logs/gpu_diagnostics
```

Do not remove the NVIDIA/ROCm environment separation from your established launch
command. A running job retains its existing environment; unsetting variables in
a different shell does not alter it.

## Validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest tests/test_gpu_events.py tests/test_heterogeneous_gpu_logs.py tests/test_heterogeneous_production.py tests/test_train_extended.py -q
```

Result: **40 passed in 1.96 seconds**. Coverage includes fake-device objects that
reject all array/scalar materialization, rotation and error records, real CPU
owner/peer subprocess logs, matching run IDs, sampling details, injected forward
failure layer/traceback capture, exact logged/unlogged fixed-membership update
parity, and existing production resume/trainer tests. No real-GPU performance or
stability validation was run for this logging change because the user's GPUs
were already occupied. This instrumentation does not fix the previous AMD reset.
