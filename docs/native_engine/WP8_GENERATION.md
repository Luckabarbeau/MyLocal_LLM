# WP8 independent native generation — EXPERIMENTAL, WP7 OPEN

This is a native worker/transport gate, **not** a WP7 numerical-parity gate.
WP7 remains **OPEN**. The parent owns implementation-status decisions; this work
does not change `IMPLEMENTATION_STATUS.md`, enable HTTP, or edit server code.
The known strict Python/native decoder failure and its test remain untouched.

## API and implementation

- C ABI: `mini_llm/native_cpp/generation.h`; implementation included from
  `generation_runtime.inc` after decoder and sampler private helpers.
- Python: `mini_llm.native.generation.NativeGeneration` and `GenerationEvent`.
- Existing decoder and sampler entry points call the same extracted private
  helpers as the worker. Arithmetic, launch parameters, and RNG are unchanged.
- One C++ thread executes decode → device sampler → compact-result readback →
  bounded native token ring. Python starts, polls, cancels, and closes only;
  it does not schedule decoder or sampler work per token. There is no fallback.

```python
from mini_llm.native.generation import NativeGeneration

# decoder and sampler are existing same-runtime, same-engine native owners.
with NativeGeneration.start(decoder, sampler, input_token=1, max_tokens=32,
                            eos_ids=(2,), temperature=.7, top_p=.9,
                            top_k=50, queue_capacity=1) as generation:
    while True:
        event = generation.wait(timeout_ms=1000)
        if event is None:             # bounded timeout, not necessarily EOF
            continue
        if event.kind == 1:
            token = event.sample.token
            # Application/tokenizer work only; no decode/sample math here.
        else:                         # terminal event, delivered once
            final = event
            break
# Same owner handles are restored by close(), including retained RNG state.
# final.pending_token is NOT already consumed by the decoder.
```

`start(decoder, sampler, input_token, max_tokens, eos_ids, temperature, top_p,
top_k, queue_capacity)` validates before transferring registry ownership.
`max_tokens=0` is valid. Queue capacity is 1..4096, poll timeout 0..120000 ms,
EOS IDs must be in vocabulary, and EOS list length is bounded by vocabulary.
C `top_k=0` means full vocabulary; Python follows the existing sampler's
`None`/positive-k convention for stochastic sampling. Supplied-uniform test
sampling is not part of generation; use the existing sampler diagnostic API.

`poll(0)` is nonblocking, `wait(ms)` is a bounded poll; both return an event or
`None`. C returns `kind=0` for timeout or after terminal delivery. `state` returns
`(last_published_event, queued_count, done)` without waiting for GPU completion.
Its `in_flight` flag reports dispatched work; position/pending fields remain the
last published snapshot, not a view of partially updated layers.

`cancel()` signals an atomic flag and wakes a full-queue wait. `close()` cancels,
joins/drains running GPU/CPU work, discards unread events, restores both owner
handles, and returns final metadata even if poll already delivered it. Python
close is idempotent. C stale handles are rejected; C close with handle zero is a
no-op with a zeroed output. Close cannot impose a wall-clock bound on a stalled
CUDA context or CPU library; it must drain running work rather than free it.

## Ownership and locking

Start temporarily removes the decoder **and sampler** from their ordinary native
registries and retains stable shared owners. Their original handles are rejected
by ordinary step/reset/sample/readback/destroy APIs until close restores them.
No other worker can claim those same owners. Python retains the wrappers while
running; explicit owner close during generation is rejected, not deferred.

GPU math retains the existing runtime-wide mutex for each complete decode/sample
pair. No cross-engine parallel throughput claim is made. Generation has a
separate short-held registry mutex and queue mutex; neither cancellation nor
poll needs the global runtime mutex or Python `Runtime._lock`. Worker queue waits
release their lock and never hold the runtime mutex. Close releases registry
locks before joining, then reacquires the runtime mutex only to restore owners.
Concurrent native close calls serialize and reject the loser. Existing public
engine/store/weight destruction is safe because retained native owners keep their
stream, maps, weights, and workspaces alive. Native static teardown cancels and
joins outstanding workers; final owner destruction/accounting is serialized
with other workers using the engine.

## Sampled versus consumed state

For each successful iteration:

1. Decoder consumes the current input; committed `position` increases by one.
2. Sampler chooses the next ID from device-resident logits.
3. Event reports `pending=1`, `pending_token=sample.token`, and a relative
   `sampled` count. That ID has **not** been consumed.
4. Only the next successful decoder step consumes the pending ID.

`position` is absolute, including a prefix committed before start; `sampled` is
relative to this generation. The caller supplies the first input explicitly.
At max-token/EOS/capacity/cancellation termination, the final sampled ID remains
pending unless it has subsequently been consumed without a successful sample
(e.g. sampler error). Resume by passing the pending ID to the restored decoder
or to a new generation, not by assuming all emitted IDs are already cached.
Generation does not advance the final sampled EOS through the decoder.

Cancellation is a step-boundary signal, **not** a GPU-kernel interrupt. A running
decode/sample pair completes and its token is enqueued even if cancel races it.
Space is reserved logically before dispatch: a capacity-one full queue prevents
the next decode, rather than allowing one extra hidden pending token.

Termination reasons are exact and explicit:

- Before dispatch: cancellation, then zero/exhausted max tokens, then capacity.
- After a successful sample: sampled EOS, then max tokens, then decoder capacity;
  otherwise cancellation is observed before the next dispatch.
- A thrown decode/sample error becomes `reason=error`, with native status/text.

Terminal state is separate from the bounded token ring. Poll drains queued token
events first, then delivers one terminal event; a full ring cannot block error
publication or cancellation. The ring is preallocated before worker launch, so
enqueue is a fixed-size assignment with no allocation/fallible growth. Allocation
or configuration failure at start is synchronous and leaves owners untouched.
Close intentionally discards unread events, but returns final counts/pending ID.
On a late-layer decoder failure, global position does not advance, `failed=1`,
and reset is required: partial layer mutation is not claimed to be rolled back.
A previously pending ID remains explicitly pending if its decode failed.

## Transfers and resource bounds

No full vocabulary logits readback occurs in the worker. Sampler D2H is the
existing 40-byte `ml_sample_result`; decoder status transfers remain unchanged.
Worker execution suppresses optional test trace D2H and invalidates stale trace
readback. Existing hybrid CPU GGML expert input/output transfers still occur;
this is not a claim that total decoder traffic is only 40 bytes/token.

A token event is 600 bytes on the tested ABI. Host queue payload is
`queue_capacity * sizeof(ml_generation_event)` (maximum 2,457,600 bytes), plus two
fixed state snapshots, bounded EOS-set metadata, thread/registry/control overhead.
These host control allocations are not included in the engine's device/pinned
payload budget. No extra device generation workspace is allocated. Existing
sampler and decoder budgets, capacity limits, and CPU expert pools apply.
SplitMix64 seed/counter semantics are unchanged: stochastic successful samples
advance once, greedy does not. Closing generation does not reset the sampler.

## Executed validation — 2026-11-24

All GPU commands used **only**:

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Preflight queried only that UUID: 10,044 MiB used, 5,797 MiB free, 98% GPU
utilization. It was already busy; no process/device reconfiguration, full-model
load, other-GPU query, or training command was issued. The reused tiny fixture
has a 512 MiB native engine budget (48 reduced-width layers, seven-token vocab,
four-position capacity, one CPU thread per existing expert owner). No training
interference/throughput measurement is claimed.

Every GPU invocation was bounded to 120 seconds. Final verified source state:

| Command (after environment above) | Result |
| --- | --- |
| `cmake --build /tmp/mini-llm-wp1-build -j 2` | PASS |
| `timeout 120s .venv/bin/python -m pytest -q tests/test_native_engine_generation.py --durations=5` | **12 passed, 24.31 s** |
| `timeout 120s .venv/bin/python -m pytest -q tests/test_native_engine_sampler.py -k 'not sampler_only_benchmark' --durations=5` | **46 passed, 1 deselected, 5.55 s** |
| `timeout 120s .venv/bin/python -m pytest -q tests/test_native_engine_decoder.py::test_last_layer_failure_global_position tests/test_native_engine_runtime.py --durations=5` | **9 passed, 2.68 s** |
| `timeout 120s /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q tests/test_native_engine_generation.py -k 'full_queue or late_layer or cancel_while_waiting or close_unread' --durations=5` | **4 passed, 8 deselected, 39.72 s; ERROR SUMMARY: 0 errors** |

Generation tests cover byte-for-byte `ml_sample_result` replay versus manual
**native** decode/sample for greedy and seeded temperature/top-p/top-k sampling;
bitwise next-step logits after consuming the final pending ID; queue-one
backpressure; cancellation while full, at initial dispatch, and during an observed
in-flight step while another thread polls; EOS, zero max, max, capacity; late
layer-47 MoE failure; restoring poisoned/resettable session state; unread close;
wrong/stale handles/geometry/engine; ordinary concurrent owner operations rejected;
public engine close; Python GC; and a child process exiting with an outstanding
native full-queue worker and no generation close. In-flight cancellation is gated
at <250 ms; full-queue cross-thread cancellation/join at <1 s. These are test
bounds, not production latency guarantees. Process teardown has a 30-second
subprocess deadline. The memcheck selection does not cover the child process,
all numerical cases, or constitute a CPU thread-race detector.

### Failures encountered and resolved locally

- Initial two-case replay process aborted on the second fixture's external GGML
  unload/reload: `GGML_ASSERT(prev != ggml_uncaught_exception)` in
  `_GLOBAL__sub_I_ggml.cpp`. The first greedy case passed separately. A test-only
  module fixture retains the selected shared-library loader reference across
  fixtures; production loader behavior is unchanged.
- Initial memcheck selection: 3 passed, 1 test assertion failed, zero reported
  CUDA memory errors. Under instrumentation an in-flight step exceeded the
  five-second bounded poll, which correctly returned `None`. The test now accepts
  timeout and continues bounded draining. The exact memcheck selection was rerun
  successfully, including after the final owner-teardown serialization change.
- No command-level 120-second timeout occurred in these runs. No timeout or
  aborted/failed run is counted as a pass.

## Limits / gate still open

This proves native-worker replay and bounded control/lifetime behavior on the
existing tiny fixture, not full-model quality, strict Python-model equivalence,
QSA/PLE generation replay at actual geometry, performance improvement, fairness
between competing workers, or recovery from a poisoned CUDA context. The late
layer failure fixture fails on its first step; it does not synthesize a failure
after a previously successful emitted prefix. Queued-event drain ordering is
exercised separately by full-queue cancellation. Larger numerical/throughput,
exhaustive multigeneration race testing, and WP7 resolution remain separate gates.
No HTTP/production integration is enabled or justified by these results.
