# WP4 HC runtime integration — independently validated component

Date: 2026-11-23. HC runtime component gates **PASS**. This is not a complete
WP4/decoder milestone; parent owns acceptance, orchestration and shared status.
`IMPLEMENTATION_STATUS.md`, the reference manifest, production Python reference,
QSA implementation and existing tests were not changed by this work.

## Scope and implementation

- `mini_llm/native_cpp/hyperconnection_runtime.inc`: private weight/state ownership,
  budget admission, finite/status gates, read/write tickets and C ABI implementation.
- `mini_llm/native_cpp/runtime.cpp/.h`: targeted include and ABI declarations.
- `mini_llm/native_cpp/hyperconnection.h/.cu`: reuse the already validated read,
  write and collapse unchanged; add exact device BF16/F32 expansion helper.
- `mini_llm/native_cpp/CMakeLists.txt`: compile the HC CUDA module with existing
  `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true` flags.
- `mini_llm/native/__init__.py`: ownership/handle-only binding; no Python tensor
  arithmetic, weight decoding, runtime fallback or production dispatch changes.
- `tests/test_native_engine_hc.py`: 52 cases including an always-executed bounded
  checkpoint case with a real CuPy reference.

The initial primitive-only boundary in `WP4_HC_PRIMITIVES.md` is historical.
Its mathematics, source/licensing provenance and low-level launch contract still
apply. No third-party kernels or model weights were copied into this implementation.
The repository license/provenance statement is unchanged.

## Authority and actual checkpoint formats

Inspected authoritative sources:

- `mini_llm/external/qwen4exp_weights.py`: `inspect_headers` defines exact output-first
  HC shapes; converted gamma aliases accept flat `[4D]` or grouped `[4,D]` in the
  same memory order. `dense` expands BF16 by zero-extending the bits, without
  adding one to gamma. `hc_matrix`/`configure_hc_bf16` distinguish storage from
  arithmetic and are not a license to introduce activation BF16 rounding.
- `mini_llm/external/qwen4exp_model.py`: attention HC stems `blk.i.hc_attn_`, FFN
  HC stems `blk.i.hc_ffn_`, read-only final stem `output_hc_`. Separate read/write
  pairs precede/follow the attention/mixer and FFN branches respectively.
- `mini_llm/external/qwen4exp_hyperconnection.py`, existing HC CUDA modules and
  the low-level WP4 primitive verification described in `WP4_HC_PRIMITIVES.md`.

Actual native descriptors were queried from the checkpoint path pinned by
`WP2_GGUF_INVENTORY.json` (not guessed from filenames):

| Suffix | Python shape | GGML shape | Stored type |
| --- | --- | --- | --- |
| `norm.weight` | `[10240]` | `[10240]` | F32, id 0 |
| `down.weight` | `[320,10240]` | `[10240,320]` | BF16, id 30 |
| `up.weight` | `[10240,320]` | `[320,10240]` | BF16, id 30 |
| `inject.weight` | `[4,10240]` | `[10240,4]` | BF16, id 30 |

Final HC has no injection tensor. Metadata: four streams, low rank 320,
`qwen4exp.attention.layer_norm_rms_epsilon=9.999999974752427e-7`.
The actual-checkpoint test asserts these formats/shapes for layers 0, 3, 47 and
final HC. Both F32 and BF16 are supported for any HC weight, including gamma;
other formats (including F16) are explicitly rejected, not silently converted.

## Public ownership API

```python
weights = engine.hc_weights(gamma_upload, down_upload, up_upload, inject_upload,
                            epsilon=epsilon, arithmetic='reference_fp32')
final = engine.hc_weights(final_gamma, final_down, final_up,
                          epsilon=epsilon, arithmetic='reference_fp32')
state = engine.hc_state(width=D, rank_capacity=R)
state.initialize(embedding_upload)  # F32[D], GPU-replicated into four streams
# Alternative: initialize exact F32[4,D] streams, e.g. an explicit reference import.
snapshot = state.read(weights)     # only a scalar identity ticket crosses the ABI
state.write(snapshot, block_output_upload)  # F32[D], original-residual write
# More paired attention-HC/FFN-HC reads and writes as required by the caller.
state.collapse(final)              # terminal until reset
state.reset()                     # empty state, must initialize again
```

The C ABI is declared in `runtime.h`: `ml_hc_weights_create/destroy`,
`ml_hc_create/destroy/reset/initialize/read/write/collapse/readback_for_test`.
Arithmetic enum 0 means explicit `reference_fp32`; every other value is rejected.
Handle wrappers never return a device pointer, numerical activation or gain from
read/write/collapse. Diagnostics alone copy numerical arrays to caller buffers.

Weight creation retains all exact-byte `UploadedTensor` owners. F32 buffers are
aliased internally read-only; BF16 buffers expand **on GPU** into persistent owned
FP32 storage using `uint32(bits)<<16` and `__uint_as_float`. Every weight is finite
checked before its owner becomes available. No CPU fallback, full-model load,
activation precision change, BF16 rounding or extra gamma `+1` occurs.

The state is engine-owned by shared native ownership; the engine is retained even
after public engine close. All inputs and weights must belong to that exact engine
(not merely its device). Native checks enforce shapes, types, rank capacity,
scalar parameters and live handles. Public weight/upload/lease/store closure does
not revoke an existing native owner. A pending read retains its weight owner even
when its public HCWeights handle closes. Destructors drain the engine stream before
freeing allocations/dependencies and restore the caller's device.

All HC kernels execute on the engine's existing nonblocking stream. There are no
per-read/write/reset allocations. Calls synchronize only for scalar finite/error
completion and successful commits; **no HC numerical intermediate is computed in
Python or copied to host during an operation**. Initialization copies embeddings
four times on GPU or copies explicitly supplied streams. Branch block outputs in
this component API are immutable F32 uploads, not diagnostic readback arrays.

## Graph and state invariant

```text
empty -> initialize -> idle(original streams)
idle -> read(weights) -> pending(ticket, original streams, owned gains/hidden)
pending -> write(exact ticket, output) -> finite gate -> staged result
        -> copy staged to stable original-stream buffer -> idle
idle -> collapse(read-only weights) -> finite gate -> terminal hidden
any failed native mutating attempt -> invalid -> reset -> empty
```

Each read obtains a globally unique process-local ticket. Reset does not reuse
identities. A ticket is valid only for its state and single pending read. Wrong,
stale, cross-state or consumed tickets cannot use another set of gains. A pending
read cannot be replaced by a second read or collapse. No separate gain parameter
exists in write: it always uses the exact native gains paired with that state's
unchanged original streams. The caller still chooses semantic attention/FFN
ordering; tickets enforce read/write pairing, not model-layer naming.

State buffer addresses are stable, including committed streams. Write computes
into disjoint staging, checks nonfinite source/intermediate/output status and CUDA
completion **before** copying to streams. Until commit synchronization succeeds the
state is poisoned. Read and collapse similarly publish no valid hidden/gains until
status validation succeeds. NaN/Inf and finite overflow are errors, not sanitized.
Failed state-changing native calls (including argument/order errors) require
reset/initialize; there is no silent resume using the old pending gains. Reset
itself clears invalid only after successful zeroing and stream completion.

Binding-local argument errors (e.g. wrong Python class/runtime or closed wrapper)
that prevent any C ABI call do not mutate native state; this is distinct from a
rejected native transaction. Actual CUDA driver-fault injection was not performed;
the conservative failure/commit logic is implemented but not a fault-recovery claim.

`readback_for_test('streams'|'hidden'|'gains', destination)` requires writable
contiguous native F32 and sufficient element capacity. Streams require initialized,
noninvalid state; hidden requires a successful pending read or final collapse;
gains require a successful pending read. Successful write invalidates hidden/gain
snapshots, reset clears all snapshots, and invalid state cannot be diagnosed as a
valid previous result. Diagnostic validation failures do not poison/consume a valid
ticket. There is no device-pointer escape.

## Budget and ownership accounting

All requested device/pinned payload shares the engine budget; context, allocator
and CUDA driver overhead remain outside that established accounting convention.

| Owner | Persistent device bytes | Persistent pinned bytes |
| --- | --- | --- |
| Exact uploads | unchanged descriptor `nbytes` | 0 |
| HC weight owner | `4 + 4 * number_of_BF16_elements` | 4 |
| HC state | `4 * (13*D + rank_capacity + 5)` | 4 |

Weight overhead includes one device status int and exact expanded BF16 storage;
F32 aliases do not charge duplicated matrix storage. State contains streams,
staging and normalized arrays each `[4,D]`, low `[rank_capacity]`, hidden `[D]`,
gains `[4]`, status int. At D2560/rank320 the state payload is **134,420 device +
4 pinned bytes**. Init/read/write/collapse/reset do not reallocate or change the
state's own payload accounting or addresses. Releasing a pending weight owner can
separately reduce total engine usage when its public handle has already closed.
Leased/retained uploads remain accounted until their final native consumer releases them. Tests cover exact budget admission, one-byte-short state
and BF16 conversion rejection, stable run/reset accounting, failure cleanup and
release-to-zero after final owners close.

Actual checkpoint tests hold three HC groups (attention, FFN, final) for **one**
selected layer at a time, not 48 layers or a full model. Recorded payload in each
selected-layer case: **118,785,312 device +16 pinned =118,785,328 peak bytes**,
less than the explicit 192 MiB engine budget. Oracle CuPy allocations are separate
from the native engine budget and are not represented as native payload.

## Exact executed validation

Build, bounded to 120 seconds:

```sh
cmake --build /tmp/mini-llm-wp1-build -j 4
```

**PASS.** Existing three misleading-indentation warnings in unchanged
`gguf_loader.cpp` remain; no new HC build warning was reported.

All test commands below ran from the repository root, bounded to 180 seconds.
The common environment was:

```sh
MINI_LLM_BACKEND=numpy
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MINI_LLM_QSA_NATIVE_SELECT=0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
NVIDIA_TF32_OVERRIDE=0
```

These names are command environment assignments, not a sourced shell script.
Exact invocations:

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 .venv/bin/python -m pytest -q -s tests/test_native_engine_hc.py -x

MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 .venv/bin/python -m pytest -q tests/test_native_engine*.py -x

MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 MINI_LLM_QSA_LAYER_CHECKPOINT=1 .venv/bin/python -m pytest -q tests/test_native_engine*.py -x

MINI_LLM_BACKEND=cupy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q -s tests/test_native_engine_hc.py -x
```

| Gate | Most recent result |
| --- | --- |
| Focused HC integration | **52 passed in 1.97 s**, child checkpoint **1 passed in 1.07 s** |
| Default full native suite | **424 passed, 27 skipped in 18.87 s** |
| Full native suite, all existing QSA layer checkpoint cases enabled | **448 passed, 3 skipped in 52.62 s** |
| Full HC suite, actual CuPy backend, CUDA memcheck | **52 passed in 4.20 s; ERROR SUMMARY: 0 errors** |

The checkpoint-enabled regression preserves the previous **396 passing existing
cases** and adds 52 HC cases. Default skips are precisely 24 optional complete-QSA
checkpoint parametrizations plus the three existing optional timing/projection/
hidden-frontend tests. With the QSA checkpoint flag enabled, only those three
existing optional tests remain skipped. **HC checkpoint validation is not skipped
in either configuration**. Under the NumPy suite, only that checkpoint test runs
in a bounded (120 second) child with `MINI_LLM_BACKEND=cupy`; no backend substitute
is accepted. Sanitizer execution uses CuPy directly, covering the checkpoint GPU
reference and native operations in the same process. Integration tests ran on
RTX 5080 device 0; prior standalone primitive validation also covered device 1.

Focused coverage:

- F32 and BF16 gamma/matrices, flat/grouped gamma aliases, D/R 1/1, 7/5, 33/35,
  257/257. Six alternating synthetic attention/FFN HC pairs, final collapse,
  reset and repeat. Exact hand fixture verifies converted gamma, four groups,
  zero injection gain 1 and original-residual write.
- Diagnostic reads do not supply any native numerical input. Synthetic block
  outputs are pre-uploaded independently of diagnostic values. These tests compare
  HC pairs, not actual attention/FFN branch execution.
- Ticket equality/provenance: duplicate read, duplicate write, wrong/stale/reset/
  other-state identity; pending collapse; read with final weights; collapse with
  injecting weights; terminal-collapse restrictions; null ticket pointer.
- Wrong activation shapes, cross-engine weights/output, stale native upload handle,
  repeated initialization, rank capacity mismatch; poison gating and reset recovery.
- F16/shape/epsilon/arithmetic rejection; NaN F32 and infinite BF16 weights;
  nonfinite initialization/output and finite write overflow; successful cleanup.
- Public upload/weight/engine close ownership, pending weight retention, stale
  state handles, no-op zero destruction, exact accounting/admission and snapshots.

Actual checkpoint coverage:

- Selected recurrent layer 0, attention layer 3, final layer 47: actual attention-HC
  and FFN-HC matrices plus actual read-only output HC.
- Assert exact stored formats/shapes and actual epsilon; both native and reference
  BF16 expansion occur on GPU. Checkpoint bytes for the oracle travel through host
  as exact bytes only, without CPU numerical decoding. No CPU execution fallback.
- Each case initializes a deterministic embedding, runs six alternating paired
  HC writes using independent pre-uploaded F32 branch outputs and performs final
  collapse. Every read/gain, committed streams and final hidden are compared to
  unchanged `Qwen4ExpHyperConnection` running on real CuPy.
- Predeclared tolerance: **rtol=2e-6, atol=2e-6**, FP32 throughout. Maximum observed
  hidden/final-output absolute error in each selected-layer case:
  **1.9073486328125e-6**. Relative tolerance also applies; no threshold was changed
  after observing results. Exact integer/identity/write fixtures remain exact.

No test failed in these integration executions and no kernel equation was changed
to satisfy a test. This records the working state of the files listed above, not
an assertion that all repository tests or all model operations are implemented.

## Remaining WP4 work / explicit nonclaims

1. Direct private device-buffer chaining of HC hidden into native QSA/GDN/FFN/MoE
   and native branch outputs into HC write. The current public HC API consumes
   uploaded F32 branch outputs; it does **not** expose raw pointers or automatically
   call those branches. No production path reads diagnostics then reuploads them.
2. Decoder-level orchestration of attention-HC and FFN-HC groups for every layer,
   token/session lifecycle, final HC to output head, plus PLE interaction if needed.
   Tickets prove paired gain/original-stream identity, not complete model ordering.
3. End-to-end actual branch-output/layer/decoder/logit parity and long trajectories.
   This validation covers actual checkpoint HC tensors but synthetic block outputs,
   not real attention/GDN/MoE execution inside the HC trajectory.
4. Full-model resident-conversion budget/placement decisions (three groups at a
   time were validated here), graph capture/asynchronous orchestration, prefill/
   batching, performance benchmarks and any future explicitly named precision mode.
5. Injected CUDA-fault recovery testing and parent acceptance/shared-status update.

No full-WP4 completion, production dispatch, speedup, BF16 arithmetic mode,
external-engine execution parity or full-model memory feasibility is claimed.
