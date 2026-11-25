# WP8 — opt-in native HTTP integration

## Scope

`scripts/serve_qwen4exp.py` retains Python as the default (`--backend python`).
`--backend native` explicitly selects `NativeDecoder.open`, a native `Sampler`
and `NativeChatSession`; it never falls back to the Python model. The same HTTP
handler, routes, authentication, WebUI, and serial `Service` lock serve both owners.
The handler was mechanically extracted into `make_handler` for finite HTTP tests.
No native runtime, binding, decoder, or chat implementation was edited.

Native setup runs before applying `BEST_ENV`; the Python GPU-Q8/BF16 tuning policy
is not installed for native serving. Native sampling parameters are passed directly
to `NativeChatSession.stream`, not through a Python sampler closure. Existing GGUF
tokenizer/template code is reused, with EOS/EOT/EOM IDs read from model metadata.
Native nonstream output is raw adapter text, without Python ChatSession's reasoning
split/rewrite; callers must preserve the returned text in explicit message history.
Native usage counts all sampled tokens, including a sampled EOS.

## Explicit configuration

Native requires existing files for all three options:

- `--native-library`: rebuilt CUDA runtime library compatible with WP8 event ABI.
- `--native-bridge-library`: CPU expert bridge library.
- `--native-ggml-library`: compatible GGML CPU library.

It also requires finite positive GiB limits for each allocation domain:

- `--native-budget-gib`: shared decoder/sampler engine device allocation budget.
- `--native-dense-budget-gib`: decoder dense weight allocation limit.
- `--native-mapped-budget-gib`: mapped checkpoint shard limit (not an RSS cap).
- `--native-host-budget-gib`: host QSA page budget.


These are not total-process memory limits; driver/context overhead and other
allocations remain outside these limits. No actual-model budget is recommended or
certified by these tests. Do not copy the Python cache budget as a native policy.

Native config also exposes `--native-page-size` (16), `--native-physical-pages` (2),
`--native-threads` (4), `--native-seed` (0), `--native-queue-capacity` (1), and shared
`--context`. Native context must be divisible by four and at most 262144; physical
pages cannot exceed the logical page count. Invalid/missing native setup fails
before binding/listening. Native has no automatic warmup, Python expert cache,
Python prefill policy, or hidden automatic enablement. Python-specific tuning
options retain their existing defaults but are not applied to native setup.

## Lifetime and transport

Native generation requires decoder and sampler to belong to the **same engine**.
The serving-only Runtime subclass retains the factory-created engine registry owner
in an outer ExitStack; the factory's inner engine scope borrows that owner through
`nullcontext`. The sampler is created from that exact engine, with allocations
charged against the shared `--native-budget-gib` limit. Cleanup order is sampler,
decoder, engine, including startup/serving exceptions. No borrowed native handles,
second engine, or native factory/binding modification is needed. CPU setup tests
assert same-engine identity, budget forwarding and owner lifetime explicitly.

Every output iterator is closed while holding `Service.lock`. A stream write or
flush failure exits the generation loop instead of continuing to finish the turn.
The adapter's close path cancels/drains/joins the native worker and invalidates
prefix reuse on interruption; the next request resets according to the adapter's
existing contract. Cancellation is not kernel preemption. Native server shutdown
joins request threads before closing sampler/decoder owners; idle native HTTP
connections have a five-second socket timeout. Tests bind loopback port 0, close
connections, call shutdown/server_close, and join the serving/request threads.

Nonstream generation completes before its response is written, so disconnects
that occur while computing an entirely buffered response cannot be detected until
the response write. This is not proactive disconnect monitoring.

`/health` and `/metrics` expose backend, arithmetic policy, native experimental
status and diagnostic trajectory differences in `config`. Native request stats
retain the adapter's phase timings, queue wait, committed/reused token counts,
termination and reset requirement. Native metrics do not fabricate Python expert
cache statistics (`experts` is null). Native logs report phase stats without a
competitive throughput claim.

## Numerical status

All-48 conditioned operator checks passed at roundoff. Independent strict
whole-trajectory diagnostics remain failed, as documented in
`WP8_NATIVE_CHAT.md` and the WP7 reports. Native reference-FP32 HC/PLE/recurrent/KV,
QSA GGML CPU policy, CPU mmap routed experts using GGML Q8 arithmetic, and native
GPU sampling are not the Python server's GPU-Q8/BF16 mixed policy. Logits and
sampled trajectories can differ. HTTP tests do not establish quality, production
readiness, competitive performance, or whole-model parity. No tolerances changed.

## Executed validation

All commands were bounded to 120 seconds. Final broad CPU regression command,
with GPUs hidden:

```sh
CUDA_VISIBLE_DEVICES='' MINI_LLM_BACKEND=numpy PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:tests .venv/bin/python -m pytest -q tests/test_native_http.py tests/test_native_chat.py tests/test_qwen4exp_serving_state.py tests/test_0239_qwen4exp_chat.py tests/test_benchmark_qwen4exp_serving.py --tb=short
```

**PASS: 106 passed, 1 skipped, 7 subtests passed, 20.29 s.**
Focused `tests/test_native_http.py::test_eos_http`: **2 passed, 0.24 s**
(nonstream and SSE).

Passing coverage includes native SSE/nonstream text and native sampling forwarding,
health/metrics, deterministic transport-failure cancellation (false write and
ConnectionResetError), reset/recovery after cancellation, concurrent request
serialization, generation errors and recovery, malformed requests, Python
stream/nonstream reasoning behavior, default CLI and native-unavailable-before-listen
with environment unchanged, plus existing adapter/serving-state regressions.
`tests/test_native_http.py` uses the real NativeChatSession/tokenizer with a mocked
native worker. It also verifies native startup owner wiring using CPU-only owners,
including serving-error cleanup and unchanged Python tuning environment.

### Real native tiny-fixture validation (GPU1 only)

Executed with these explicit process settings (inherited by all children):

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631 MINI_LLM_BACKEND=cupy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=.:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_native_engine_http.py tests/test_native_engine_chat.py tests/test_native_engine_generation.py --tb=short
```

**PASS: 17 passed, 42.35 s.** Before the broad native run, the existing generation
EOS close case and chat disconnect/EOS case passed together: **2 passed, 3.84 s**.

`test_native_engine_http.py` uses the existing `small_setup` synthetic fixture:
48 narrow recurrent layers, width 256, vocabulary 7, context capacity 4, and a
512 MiB engine budget. It checks real native HTTP nonstream/SSE EOS finish reasons,
sampled-EOS usage, disconnect cancel/drain, reset before the next request, and
accounted device allocation within the fixture budget. Servers use loopback port 0,
shutdown and thread joins. No indefinite server process remains.

No full checkpoint was loaded; no actual-model budget/resource acceptance is
claimed. Production `NativeDecoder.open` startup with a full model remains
unvalidated; its server-side engine retention and cleanup were tested with mocks.
The real tiny HTTP test uses the existing fixture factory, not the actual-checkpoint
factory. GPU0 was not used for tests or allocations.

### Local corrections and terminal-close contract

Initial runs exposed an HTTPConnection import alias error, then an EOS mock error:
the shared historical FakeWorker returns `cancelled` on close whenever sampled is
less than maximum, even after an EOS terminal event. A **new-test-local** worker
now preserves its observed terminal snapshot through close. EOS assertions remain
strict (`stop`, one sampled token, empty text); no shared fixture or adapter was
changed. Native source confirms close cancels/joins and returns the saved terminal
snapshot; the existing real native chat EOS test asserts last_stats termination
remains `eos` after the adapter's finally cleanup. Both real tests passed. There
is no evidence here of a production chat-finally termination bug.

Source inspection also caught the initial server's invalid separate sampler engine:
`ml_generation_start` requires engine identity. The localized serving-runtime
retention correction described above ensures same-engine ownership and retains
the engine until after sampler/decoder cleanup. It does not change native math,
APIs, assertions, or tolerances.

Parent-owned implementation status files are intentionally untouched.
