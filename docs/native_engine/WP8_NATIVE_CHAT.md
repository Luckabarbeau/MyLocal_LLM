# WP8 — opt-in native chat/session adapter

## Scope and status

Implemented `mini_llm/native/chat.py`, backed by the existing `NativeDecoder`, `NativeGeneration`, and native `Sampler`. This is actual opt-in session development following the all-48 conditioned operation evidence in `WP7_CONDITIONED_ALL_LAYERS.md`, **not** a replacement for the failed strict whole-trajectory diagnostic or a quality/performance acceptance claim.

This adapter/runtime work does not change production defaults, server files, training, decoder equations, GPU kernels, reference tests or fixed tolerances. No full model checkpoint was loaded. Parent-owned status files are untouched; concurrent server integration is owned by the other agent.

**Prompt-boundary rollback is now implemented:** one bounded native host snapshot at `prompt[:-1]` enables exact repeated-request prefix caching without re-prefill. Public chat method signatures remain unchanged. See `WP8_PROMPT_CHECKPOINT.md` for the new API, replay evidence, pager fencing and scope.

## APIs

### Sequential native prompt consumption

```c
int ml_decoder_consume(ml_handle decoder, const int32_t *tokens,
                      uint64_t count, uint64_t *committed);
```

Python: `decoder.consume(tokens) -> absolute_committed_length`.

- One ABI call validates the complete vector and remaining capacity, then loops over the **existing `decoder_step_native` body** with null output logits and diagnostic tracing disabled.
- All original layer operations and final device vocabulary projection remain present. No Python per-token/per-layer prompt execution and no full logits D2H. Only the committed length returns.
- Empty vector is a no-op returning the existing length. Invalid tokens/capacity are rejected before mutation. Failed decoder owners are rejected.
- An execution failure preserves the last successful committed length and poisons the decoder. Reset is mandatory; this is not partial-prompt recovery. `decoder.state` remains the diagnostic source after a Python exception.
- This is a **sequential baseline, not WP10 batched prefill**; it retains the original step's synchronization and CPU-expert behavior.

### Chat owner

```python
from contextlib import closing
from mini_llm.native.chat import NativeChatSession

# decoder and sampler already exist; their native allocation lifetimes remain
# caller-owned. reader is existing GGUF metadata, not a second model load.
chat = NativeChatSession.from_gguf(
    decoder, sampler, reader, eos_ids=stop_token_ids, queue_capacity=1)
with closing(chat.stream(
    [{"role": "user", "content": "Hello"}],
    max_new_tokens=32, temperature=0.7, top_p=0.9, top_k=40,
    thinking=False,
)) as output:
    for piece in output:
        send_text(piece)
# closing is required on disconnect/early break, not just on normal exhaustion.
stats = chat.last_stats
```

- `NativeChatSession(decoder, sampler, tokenizer, *, eos_ids=None, queue_capacity=1)` accepts the existing `GGUFTokenizer`. Vocabulary sizes must match. `from_gguf` reuses `load_gguf_tokenizer`; no new tokenizer algorithm or model factory.
- `stream(messages, max_new_tokens, *, thinking=True, reasoning_effort=None, temperature=0., top_p=1., top_k=None)` uses the existing `qwen4exp_chat.render_chat` and GGUF template, then existing `encode(...).ids`. Text-only system/user/assistant validation and unsupported-template behavior are inherited. History is explicit on every request; no hidden reasoning reinterpretation or automatic assistant-message rewriting.
- `stream_tokens(prompt_tokens, max_new_tokens, *, temperature=0., top_p=1., top_k=None)` is the exact-ID completion/session entry point. Positive output limit and a nonempty, in-vocabulary prompt are required. The worst-case committed length `len(prompt)+max_new_tokens-1` must fit native capacity.
- Sampling parameters are passed directly to `NativeGeneration.start`. There is no Python logits access, sampling loop, fallback, or first-token Python sample. Native validation governs temperature/top-p/top-k; a native start error invalidates session reuse.
- `eos_ids=None` means only `tokenizer.spec.eos_id`, if present. Callers must supply the model's complete stop-ID set where additional stop markers apply. EOS counts toward the output limit and remains a pending unconsumed sample, but is not emitted as text.
- `committed_tokens` is a read-only tuple; `pending_token`, `require_reset`, and `last_stats` expose session bookkeeping. `reset()` resets decoder and native sampler RNG to its configured seed, clears the prefix and pending token, and leaves allocations warm. Normal prefix reuse advances the existing native RNG without reseeding.

### Exclusive ownership and prefix invariant

For prompt `P` and samples `S = [s0, ..., sn]`, a normal successful request ends with:

```text
native committed tokens = P + S[:-1]
native position         = len(P) + len(S) - 1
pending unconsumed ID   = S[-1]          (including EOS)
```

For a reusable prefix of length `r`, consume `P[r:-1]` in the native prompt loop and start the worker with `P[-1]`. The worker consumes that last prompt token before sampling. **Do not consume all of P and then pass its final ID again.** A continuation containing the prior pending sample consumes it exactly once. A newly rendered template that does not contain that ID never gets an implicit pending-token insertion.

Ordinary continuation reuse requires the actual incoming IDs to begin with the entire committed token sequence, with an additional tail. The additional exact-request cache recognizes only the **entire last prompt's token vector** and restores its checkpoint at `prompt[:-1]`, discarding the generated suffix. It skips `consume`, then sends the final prompt token to the native worker exactly once. Other shorter/different histories reset/replay; there is no arbitrary-prefix rollback or multi-request cache. A prompt equal to committed tokens can use the cache only if it is also that exact saved request; otherwise it still resets. No rule depends on conversation labels, message object equality, or caller claims.

The session holds a nonblocking serialization lock from first iterator advance through completion/finally. Concurrent generation/reset is rejected, not interleaved. Decoder and sampler must be exclusively borrowed by **one** session, including between requests; do not construct competing sessions on the same owners. A changed native position/failed flag is detected and resets the session, but an external reset-and-replay to the same position cannot prove token identity. The exclusive-owner API contract is necessary, not optional.

### Output and cleanup

`IncrementalDetokenizer` reuses the installed `tokenizers.decoders.DecodeStream(skip_special_tokens=False)` against the existing tokenizer backend. It preserves merges, added/special token boundaries and split UTF-8. Streaming requires the existing ByteLevel decoder; unsupported/replaced decoder formats fail rather than approximate output. It requires a tokenizers release providing `DecodeStream` (executed with 0.23.2).

Because DecodeStream has no EOF flush, `finish()` performs **one** exact batch decode of output IDs, verifies already emitted text is a prefix, and emits the remaining suffix. This preserves incomplete UTF-8 replacement characters and literal U+FFFD at EOF. There is no full-history decode for each output token. Output ID/text bookkeeping is linear in the bounded output limit; this is not constant-memory detokenization.

The consumer polls a bounded native output queue in 100 ms waits. Native backpressure, not a Python producer, bounds unread samples. Every worker path has `finally: cancel(); close()`: native close drains/joins in-flight work, discards queued events, and returns owners. Any interrupted iterator, native error, detokenizer error, inconsistent terminal metadata, or cleanup error marks `require_reset=True` and clears the pending ID. Unread/discarded IDs are never inferred from the final position. The next request resets before executing. Even if a disconnect happens after a final text piece but before iterator exhaustion, reuse is conservatively invalidated. Native close may wait for running kernels; cancellation is not kernel preemption.

## Metrics (no speed claim)

`last_stats` separates:

- `prefill_seconds`: wall time for the one sequential `consume` call over all **but the final prompt token**. Empty uncached input still incurs ABI overhead. A prompt-checkpoint cache hit skips the call and reports zero. Reset and checkpoint capture/restore are outside this measurement.
- `checkpoint_restore`, `checkpoint_restore_seconds`, `checkpoint_host_bytes`: cache-hit status, native rollback call wall time, and the retained snapshot's bounded requested ordinary-host storage. Capture overhead is included in request time, not a separate metric. Cache restore preserves the sampler's advancing RNG; it does not cache sampled responses.
- `decode_seconds`: cumulative native successful decoder-step **plus sample** wall time, including the first worker step on the final prompt token. Excludes native queue backpressure and runtime-mutex wait. This is not pure GPU kernel time; incomplete failing steps are not included.
- `output_queue_wait_seconds`: cumulative Python `poll` call wall time (may overlap decode time). It is not native producer queue-full wait and must not be added to decode as disjoint phases.
- `ttft_seconds`: time from activation of the token-ID request iterator to the first nonempty text piece yielded; `None` for EOS-only/no emitted text. Template rendering/encoding done by `stream` before this iterator starts is excluded. It measures adapter emission, not transport delivery.
- `request_seconds`: whole active iterator lifetime, including consumer pauses; `prompt_tokens`, `reused_tokens`, `new_prompt_tokens`, `sampled_tokens`, `emitted_pieces`, `termination`, `reset_reason`, `require_reset`.

The experimental `ml_generation_event` C/ctypes layout gains a trailing `double decode_seconds`. **Rebuild the native library and restart callers together**; old external struct consumers must be rebuilt. No compatibility with an old binary layout is claimed.

## Initial adapter validation (before prompt-checkpoint follow-up)

All terminal test/build invocations were bounded to **120 seconds**. All GPU test processes used only:

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH=.:tests
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Only the existing `small_setup` synthetic fixture was loaded: 48 narrow recurrent layers, width 256, vocabulary 7, capacity 4, CPU mmap routed experts and a **512 MiB native engine budget**. This is not an actual 48-layer checkpoint allocation. Its integration test also asserts accounted device allocation remains below that budget. Initial free memory on the selected GPU was 4399 MiB, lower than the anticipated ~5.8 GiB; no other workload was stopped. GPU0 was not used for tests or allocations. No full-model or all-48 conditioned-memory run was attempted.

| Executed command | Result |
| --- | --- |
| `cmake --build /tmp/mini-llm-wp1-build -j 2` | PASS; rebuilt runtime/library; existing CUDA CUB deprecation warnings on first build |
| `CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_native_chat.py --tb=short` | Final **10 passed, 0.16 s** |
| `.venv/bin/python -m pytest -q tests/test_native_engine_chat.py tests/test_native_engine_generation.py --tb=short` | Initial 16 passed; final **16 passed, 37.39 s** |
| `compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q tests/test_native_engine_chat.py::test_native_consume_replay_and_atomic_validation tests/test_native_engine_chat.py::test_native_chat_disconnect_and_eos tests/test_native_engine_chat.py::test_native_prefill_and_worker_error_reset --tb=short` | **3 passed, 108.77 s; ERROR SUMMARY: 0 errors** |
| `git diff --check` | PASS |

CPU coverage: incremental merged output, UTF-8 bytes split over four tokens, partial UTF-8 EOF, literal replacement characters, custom/special tokens, one final batch decode, fake exact-prefix continuation/repeated prompts, no-tail replay, reset and serialization, disconnect, EOS, worker/cleanup errors, external position changes, native-only sampling argument forwarding, template/message/decoder-format and context/token validation.

Native coverage: sequential consume equals manual native replay, no Python step/sampler callback, empty consume, atomic token/capacity rejection, queue/pending off-by-one, repeated requests and continuation, owner position changes, bounded queue disconnect, EOS, explicit reset, late-layer failure both during consume and in the worker. Existing worker suite covers stochastic/greedy native replay, backpressure, full-queue cancellation, native ownership, terminal drains, failure states, and process teardown. These tests do not establish full-model quality or independent Python-model parity.

### Strict decoder failures remain separate and visible

The unchanged tiny strict diagnostic was executed **separately**:

```sh
.venv/bin/python -m pytest -q \
  tests/test_native_engine_decoder.py::test_synthetic_all48_reference_replay_lifetime --tb=short
```

**FAIL, 2.40 s**, first boundary `layer=15 stage=post_moe`, at the unchanged `rtol=2e-4, atol=2e-4`: 135/1024 mismatched components, max absolute difference among violations `0.0569191`, max relative difference among violations `0.2926828`. Example analytical/native value `48.707828521728516`, Python reference `48.69479751586914` (FP32 tensors; no finite-difference step). This independent-trajectory assertion is not the WP8 native-vs-native session oracle. No corrective math edit or tolerance relaxation was made.

The historical actual-checkpoint strict whole-trajectory failure in `WP7_CONDITIONED_ALL_LAYERS.md` also remains **FAIL**, not rerun under this memory budget. Token-0/token-1 logit maxabs remain `0.1547901630` / `0.4191079140`, relative-L2 `0.0115831571` / `0.0378313720`. The all-48 conditioned operator result remains separately PASS. The repository is **not globally green**; passing WP8 tests do not relabel either strict failure.

## Next gate and limitations

1. **Implemented checkpoint gate:** exact last-request rollback at the pre-final-input boundary; see `WP8_PROMPT_CHECKPOINT.md`. Arbitrary shortened prompts and resume-after-unknown-drain are still unsupported. Sampler RNG/response replay is not a checkpoint feature.
2. Parent/other-agent server lifecycle integration must reliably close generators on disconnect and keep decoder/sampler ownership exclusive. No server files were edited by this adapter/runtime task.
3. Longer-context, actual mixed recurrent/QSA/PLE chat, actual stop-ID sets, model quality, transport timing and model-scale resource acceptance remain separate gates. The tiny fixture cannot certify them.
4. WP10 batched prefill is not implemented. No throughput or latency improvement is claimed.
