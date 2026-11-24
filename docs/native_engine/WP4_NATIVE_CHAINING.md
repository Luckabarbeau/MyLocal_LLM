# WP4 native device chaining / integration gate

## Acceptance

**PASS: bounded WP4 device chaining**, not a full decoder. Native HC read now
feeds native QSA or recurrent/GDN directly, and its branch output feeds the paired
HC write directly. Existing PLE accepts HC streams before mixing; final collapse
remains native. No FFN/MoE computation, production dispatch selection, logits or
full-model parity is claimed. Parent-owned `IMPLEMENTATION_STATUS.md` and
`native_reference_manifest.json` were not changed.

The authority's order is `Qwen4ExpGGUFModel._step` and the existing pre-MoE and
QSA block graphs:

1. Initialize this token's HC streams from its uploaded embedding or streams.
2. If present, PLE consumes/replaces HC streams, updating its own causal history.
3. Attention HC read -> QSA or recurrent/GDN -> **the same read's** HC write.
4. FFN HC read -> **explicitly pre-uploaded synthetic branch output until WP5**
   -> paired FFN HC write.
5. Further layer boundaries may follow; native final HC collapse is terminal
   until HC reset. Persistent branch/PLE positions span tokens; HC is reset and
   initialized for each new token in the integration tests.

This is a caller-ordered operator API, not an architecture-aware scheduler: it
cannot detect omission/reordering of separate PLE/FFN calls. The integration tests
exercise the required ordering. `mix` itself enforces its read/branch/write pair.

## API and implementation

New C ABI:

```c
int ml_hc_mix(ml_handle state, ml_handle weights, ml_handle branch,
              int kind, uint64_t position, uint64_t* committed);
/* kind = 0: QSA; kind = 1: recurrent/GDN */
```

Binding (control only, no Python numerical work):

```python
hc.initialize(uploaded_embedding)
ple.step(hc, token, position)       # optional, before attention HC
committed = hc.mix(branch, attention_hc_weights, position)
ticket = hc.read(ffn_hc_weights)
hc.write(ticket, uploaded_synthetic_ffn_output)  # not an implemented FFN
hc.collapse(final_hc_weights)
```

`HCState.mix` returns only committed position. It makes one ABI call, which uses
private pointer-based helpers under the parent runtime's `boundary`, registry
lock, device guard, stream completion guard and existing error/status mechanism.
There is no nested ABI call/lock and no new status subsystem.

Files:

- `mini_llm/native_cpp/chaining_runtime.inc`: private branch orchestration.
- `hyperconnection_runtime.inc`: extracted existing read/write launch/commit
  bodies into shared helpers, used by both old uploaded API and chaining.
- `qsa_layer_runtime.inc`, `recurrent_runtime.inc`: extracted existing step bodies
  into shared device-hidden helpers. Old uploaded APIs retain their validation,
  upload wait and completion guards; chaining validates its device owner instead.
- `mini_llm/native_cpp/runtime.cpp`, `runtime.h`: include and ABI declaration.
- `mini_llm/native/__init__.py`: signature and ownership/type-checked wrapper.
- `tests/test_native_engine_chaining.py`: integration acceptance tests.

No kernel, dequantizer, projection equation, normalization or recurrence equation
was added, copied or changed. PLE and final-collapse implementations are reused
unchanged. No new allocation occurs inside mix; hidden/output never cross host.
Existing scalar status checks and stream synchronizations remain (this is not an
asynchronous/captured execution or performance claim). QSA still transfers its
scalar length metadata; PLE still stages the selected immutable table row bytes.
Neither is an activation readback/reupload.

## Ownership, tickets and failure contract

- Native shared owners retain HC weights, branch weights and engine independently
  of public upload/store/engine handles. Local shared owners remain alive through
  stream completion on success and exception unwinding. No raw device pointer is
  exported and allocation ownership is not exchanged by chaining.
- HC must be initialized, uncollapsed, valid and have no pending read. Injecting
  weights must match its engine, width and rank capacity. Branch engine and width
  must match; QSA requires D=2560. Exact committed position and capacity are checked
  before launching HC read.
- The native read generates the normal globally unique one-use ticket, retaining
  the exact HC weight owner and gains. Mix checks the same ticket and consumes it
  with branch output; it never publishes a ticket that could be reused. On success
  no gains or hidden diagnostic remains available, as for ordinary HC write.
- Committed output is zero on ABI failure when its output pointer is non-null.
  All mutating mix failures poison HC, including metadata, pending ticket, stale
  handle and invalid kind failures. Reset and initialize before reusing HC.
- Preflight failures do not advance/poison an otherwise valid branch. Any exception
  during read/branch/write execution quarantines **both HC and branch** until reset.
  Branch cache may have committed before an HC-write failure; mix does not claim
  cross-owner rollback. Poison prevents resuming a partially completed graph.
  Reset both and replay the sequence. A separately committed PLE step likewise is
  not rolled back by a later mix failure; reset/replay PLE as part of the sequence.
- Existing standalone branch/PLE numerical-failure retry semantics are unchanged.
  Final collapse remains terminal and rejects pending HC reads.

## Executed verification / verified state

Build and commands ran from repository root using the existing CUDA build and
`.venv`. Build completed successfully:

```sh
cmake --build /tmp/mini-llm-wp1-build -j 4
```

Three existing misleading-indentation warnings in `gguf_loader.cpp` appeared on
the first rebuild. An extraction-introduced unused local was removed; the final
build emitted no new warning. No numerical production correction was needed.
A control-path review also restored the standalone recurrent wrapper's original
CUDA-poison catch around uploaded-input wait (the extracted helper has no upload).
The final rebuild, full native suite and memcheck below were rerun after that
localized exception-semantics preservation.

Focused acceptance (180-second bound):

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_TEST_PLE_CUDA=1 .venv/bin/python -m pytest -q -s tests/test_native_engine_chaining.py -x --tb=short
```

**15 passed in 4.32 s**, including mandatory bounded CuPy child checkpoint cases.
The first test-development runs found only test API/oracle issues: omitted explicit
QSA policy; a vector callback passed to a matrix-only GPU packed projection API;
and an imported NumPy-only diagnostic comparison used against CuPy arrays. Local
test corrections supplied the explicit policy, batch-axis view and explicit
reference diagnostic conversion. Tolerances were not loosened in response to a
numerical failure, and native math was not changed.

Coverage:

- QSA and recurrent/GDN, each with and without preceding PLE, eight successive
  tokens and two full reset/replays. PLE includes repeated EOS tokens. Compare PLE
  HC streams/history, branch output/persistent caches, attention HC streams,
  FFN HC hidden/gains/streams, and final collapse to unchanged Python references.
- All embeddings and synthetic FFN outputs are uploaded before execution.
  Diagnostic arrays never become native inputs. Tripwires forbid public Python
  branch `step`, HC `read`/`write`, and Engine upload during mix. Synthetic reference
  numerical work is test-only, not production control or fallback.
- Pending read, position, capacity, cross-engine, width, branch kind, stale handle,
  null completion and converted-A numerical failure cases. HC poison, branch
  position preservation or execution poison, reset, unavailable consumed gains,
  and retained native ownership after public source/engine closes are checked.
- Engine accounting is unchanged over all steps and reset/replay.
- Mandatory selected actual checkpoints: recurrent layer 0 and QSA layer 3, with
  their **actual attention HC, FFN HC and final HC** weights. Eight tokens, then
  reset and the same eight tokens again. The FFN result remains independent,
  explicitly pre-uploaded synthetic output. Both checkpoint oracles require real
  CuPy and existing GPU packed FP32 projection/delta implementations; no CPU
  numerical dequantization fallback. Native diagnostics are never fed back.

Comparison envelope fixed at rtol=2e-4, atol=2e-4, FP32, matching the existing
checkpoint projection/branch envelope; these are not finite-difference tests.

| Checkpoint chain | Maximum final-collapse absolute error | Native device bytes | Pinned bytes |
| --- | ---: | ---: | ---: |
| GDN layer 0 + HC | 3.814697265625e-6 | 169,491,364 | 20 |
| QSA layer 3 + HC | 9.5367431640625e-6 | 169,992,008 | 32 |

Each native engine has a hard 256 MiB budget. These figures are Engine accounting,
not whole-process/oracle GPU memory measurements. Only selected layers are tested;
no all-layer or all-checkpoint trajectory claim.

Full native regression (240-second bound), including QSA checkpoints and PLE CUDA
primitives explicitly enabled:

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 MINI_LLM_QSA_LAYER_CHECKPOINT=1 MINI_LLM_TEST_PLE_CUDA=1 .venv/bin/python -m pytest -q tests/test_native_engine*.py -x --tb=short
```

**Latest: 495 passed, 3 skipped in 69.95 s.** This preserves the prior 480 passing native
cases and adds 15 chaining cases. The three pre-existing optional cases remain
skipped; no new chaining test skips or native fallback. This is the full native
suite, not the entire repository's unrelated training/serving suites.

Focused CUDA memory validation (180-second bound):

```sh
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NVIDIA_TF32_OVERRIDE=0 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 .venv/bin/python -m pytest -q -s tests/test_native_engine_chaining.py -x --tb=short
```

**Latest: 15 passed in 39.27 s; ERROR SUMMARY: 0 errors**, including GPU checkpoint child
processes. Also passed Python compilation of binding/new tests and scoped
`git diff --check` on the tracked runtime/header/binding changes.

## Remaining / nonclaims

- WP5: actual FFN/shared experts/MoE computation and ownership/scheduling, replacing
  the explicit synthetic FFN branch boundary; no duplicated FFN kernels here.
- Architecture-aware full decoder/session orchestration, logits/generation and
  end-to-end model/reference trajectories. Existing Python model dispatch is not
  switched to this separate Engine API.
- A combined *actual-checkpoint* PLE+HC+branch trajectory is not added; existing
  actual PLE coverage remains in the passing native suite, and PLE-first combined
  ordering is covered by the synthetic integration cases.
- All layers, long contexts/trajectories, batching/prefill, capture/asynchronous
  scheduling and performance validation remain separate work.
- Injected CUDA commit faults and a failure specifically after successful branch
  commit are not tested. The explicit quarantine contract covers this path in
  code; no cross-owner rollback/fault-recovery acceptance claim is made.
