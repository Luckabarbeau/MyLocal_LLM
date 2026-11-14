# AMD prolonged-training investigation

## Status

Two different observations are established:

1. At **2026-11-12 23:28:41**, the AMD driver reported MES failure to respond to
   `REMOVE_QUEUE`, failed queue eviction, and a GPU reset. The reset explicitly
   lost VRAM contents. The peer's HIP launch error and the owner's connection
   reset are consistent with that event. The journal does not identify the
   initiating kernel or distinguish kernel corruption from driver/firmware fault.
2. The later logged run rejected at least two optimizer rounds with `norm=inf`.
   One first attempt at update 15,715 was followed by a successful retry with
   norm **0.298002** and loss **2.267292**. Another rejection was reported while
   attempting 15,717. Rejected rounds did not advance weights/optimizer steps.
   An infinite FP32 sum-of-squares norm alone cannot distinguish actual Inf/NaN
   gradient elements from very large finite gradients.

The user is running a real GPU job. This investigation did **not** interrupt it,
run competing GPU tests, change power/driver settings, or modify checkpoints.

## Confirmed buffer-ownership defect and localized fix

`mini_llm/ops/triton_dual_resolution_dense.py` returned immediately for no keys
(or a bounded global operation with no visible history), leaving provided output
and gradient buffers unchanged.

The concrete production integration chain is:

```text
local_attention_forward:
  allocate empty output/LSE; skip the head owned by sparse retrieval
matrix-pool attention:
  pass that skipped output/LSE head as strided destination views
sparse retrieval:
  phase >= sequence length -> zero selected keys
old dense helper:
  return without writing supplied destinations
projection/backward:
  consume prior allocator contents instead of mathematical zeros
```

The same issue existed for supplied dQ in the zero-work backward return. Local
backward intentionally skips that dQ head, expecting sparse backward to overwrite
it. Empty history must have zero output and zero dQ. Shared K/V derivative
addition must preserve existing contributions from other heads.

The fix is restricted to the two early-return regions:

- Forward clears output and LSE destination views.
- Backward clears dQ; clears dK/dV only when `add_dkdv=False`.
- Additive K/V destinations and neighbouring head storage remain unchanged.
- Nonempty-history kernel launches, tiles, reductions and rounding are unchanged.

Regression tests run the actual Python dispatch/early-return functions using a
NumPy-backed CuPy facade, with sentinel-filled strided head views and neighbouring
canaries. They do not execute compiled Triton GPU kernels. New forward cases use
length/phase pairs `(1,1)`, `(5,8)`, `(8,8)`, `(15,15)`; backward covers empty keys
and bounded-no-history with overwrite/additive destinations.

**Attribution limit:** retained production logs contained 2,374 owner and 2,119
peer batch records when inspected. Minimum retained executed length was **48** on
both workers; no retained batch reached the zero-key sparse phase condition.
Rotation has removed older records, so this does not rule out earlier short
batches. This is a proven defect, but **not a proven explanation of these
particular rejected rounds or the AMD reset**. Large finite garbage/NaNs can
explain numerical corruption; this branch does not itself demonstrate an invalid
address or a reason for the MES reset.

## Failure-only gradient audit

With detailed diagnostics attached, `ExtendedTrainer` now invokes its existing
scale-stable gradient audit automatically **only after a rejected norm and before
clearing gradients**. Successful rounds do not gain gradient scans or device
synchronizations.

The owner JSONL gains `event: gradient_rejected` with:

- completed and attempted update numbers;
- ordinary norm, loss and global supervised-token weight;
- `failure_kind`: `norm_reduction_overflow` or `nonfinite_elements`;
- gradient parameter/element counts;
- exact parameter names and NaN/Inf counts for nonfinite tensors;
- scale-stable norm and largest finite parameter norms/magnitudes.

The original explicit `MINI_LLM_GRAD_NORM_AUDIT=1` path remains supported when
per-GPU diagnostics are disabled. Automatic logging does not depend on exporting
that variable, which checkpoint environment restoration could otherwise remove.
No learning-rate, clipping, loss-scaling, optimizer, or skip policy was changed.

The audit sees the owner's **globally combined** gradients. The peer export is
already finite-element checked before combination. A passing export check does
not rule out extreme finite peer values, and the owner's raw local gradients are
not retained separately after reduction. Do not label a rank as responsible from
a combined-gradient audit alone.

Two injected CPU integration cases establish that the audit:

1. distinguishes `[3e20, 4e20]` finite gradients with an overflowing ordinary norm
   from actual infinite gradient elements;
2. records evidence before clearing;
3. leaves model and optimizer state unchanged on rejection;
4. releases the waiting peer without advancing weights;
5. allows a subsequent normal update and performs no audit on that success.

## Other inspected paths

HIP local/global/sparse KV kernels were inspected for partial scratch ownership,
GQA reduction addresses, segment coverage and explicit skipped-head writes.
CPU index/ownership algebra checks passed:

- 38 local reduction cases (batch 1/2; partial/tile-boundary lengths through 8K).
- 240 dense/sparse-style reduction layout cases.
- Query segmentation coverage for every length from 1 through 8,192.

These checks do **not** prove compiler-generated GPU code is safe or numerically
correct. No new defect was established in those indexed reductions.

The HIP projection helper's three-part BF16 residual split, FP32-output GEMMs,
host scalar lifetimes, current-stream binding and pointer-mode restoration were
reviewed. Its documented extreme-range limitation remains relevant if the audit
finds very large finite gradients; ordinary-range parity tests do not establish
arbitrary-range equivalence to FP32 SGEMM. No evidence justified changing it.

Existing AMD full-model performance validation used a fixed 8K synthetic document
with two warmups and five measured iterations. It did not establish prolonged,
variable-shape, resumed-training stability. The live run's variable-length kernel
specializations and long-lived allocator/runtime state are a materially wider
workload. Driver, firmware, resource-lifetime, and shape-dependent kernel issues
remain plausible explanations for the reset, not established causes.

Observed runtime: gfx1201, 32 GiB-class AMD card, CuPy 14.2.0,
Triton `3.8.0+rocm7.2.4.git669b31ac`, Linux `7.0.0-42-generic`. No new matching GPU
reset was found in the inspected kernel log during this check.

## Validation and preserved evidence

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_BACKEND=numpy .venv/bin/python -m pytest tests/test_attention_empty_destinations.py tests/test_rejected_gradient_diagnostics.py tests/test_gradient_norm_audit.py tests/test_heterogeneous_production.py tests/test_heterogeneous_gpu_logs.py tests/test_sparse_stride_history_0102z.py -q --tb=short
```

Most recent result: **27 passed, 2 skipped**. The warning in the existing ordinary
FP32-overflow test is intentional. Actual CuPy/Triton device regression and a
prolonged GPU rerun were **not** performed while the production job was active.

Available per-GPU log rotations/manifests and the reset journal were copied to:

```text
logs/amd_long_run_investigation_20261113/
```

`snapshot.json` identifies the files and explicitly marks the copies best-effort
live-file snapshots, not quiescent replay state. `index_coverage.json` records the
CPU address/segment checks. At snapshot time the last saved checkpoint was 15,715;
the active job may subsequently save later checkpoints.

## Next discriminating experiment

At a controlled checkpoint boundary, restart with the same normal-speed logged
command so the buffer fix and failure-only audit are actually loaded. Do not
change contexts, tile policies, LR or serialize the whole run at the same time.
Inspect the **first `gradient_rejected` audit** if a rejection recurs. Parameter
names, stable norm, and nonfinite counts will distinguish numerical reduction
overflow from genuine corrupt gradient elements before any further kernel edit.
This is the specific evidence missing from the already-cleared failed rounds.
