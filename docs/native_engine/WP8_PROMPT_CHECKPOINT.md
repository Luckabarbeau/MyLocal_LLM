# WP8 — native prompt-boundary checkpoint and exact-request cache

## Result and scope

Implemented one bounded, ordinary-host-memory prompt checkpoint per native decoder and exact-last-request token-prefix caching in `NativeChatSession`. Public chat constructor/stream/reset signatures are unchanged. Existing generation, pending-sample, output queue and cancellation contracts remain intact. There is no model-math, GPU-kernel, training, fixed-tolerance, server-file or parent-status change in this task.

**Executed final focused native validation: 26 passed. CPU adapter validation: 12 passed. Memcheck subset: 3 passed, 0 errors.** These are native bitwise replay/session checks, not full-model precision/quality acceptance. The strict independent-trajectory diagnostic remains historically FAIL as documented in `WP8_NATIVE_CHAT.md` / `WP7_CONDITIONED_ALL_LAYERS.md`; it was not changed or rerun here. No full model was loaded. This is not WP10 batched prefill or a speed claim.

## API

```python
checkpoint = decoder.checkpoint(host_budget_bytes=256 * 1024**2)
checkpoint.host_bytes       # bounded requested ordinary-host storage, not GPU RAM
length = checkpoint.restore(decoder)  # absolute committed position
checkpoint.close()
```

Native ABI, declared in `mini_llm/native_cpp/runtime.h` and bound in `mini_llm/native/__init__.py`:

```c
int ml_decoder_checkpoint_capture(ml_handle decoder, uint64_t host_budget,
                                 ml_handle *out, uint64_t *host_bytes);
int ml_decoder_checkpoint_restore(ml_handle decoder, ml_handle checkpoint,
                                 uint64_t *committed);
int ml_decoder_checkpoint_destroy(ml_handle checkpoint);
```

- Successful capture replaces the decoder's previous checkpoint. Only **one complete state payload** is retained. Replacement drops the old data buffer before allocating the new one. Metadata/budget validation occurs first, so a rejected budget preflight leaves an existing checkpoint usable.
- Default Python host budget is explicitly **256 MiB per decoder checkpoint**; callers can configure a smaller/larger bound. Storage is ordinary pageable host RAM, not a new GPU allocation or pinned reservation. `host_bytes` counts the requested state buffer, native checkpoint object, and vector metadata capacity; allocator/control-block/registry-node overhead and whole-process RSS are not claimed as exact accounting. Snapshot size is checked before the large buffer allocation.
- The decoder owns the snapshot. Snapshot handles and the checkpoint's owner reference are weak: a checkpoint cannot keep a closed decoder, engine, or mmap alive. The live decoder already retains its required engine/weights/session resources safely. Closing an engine wrapper while the decoder is live remains supported.
- Reset, decoder close, or successful checkpoint replacement invalidates prior checkpoint handles. An epoch plus owner identity prevents restoring across reset-and-replay even at the same position. Wrong-decoder/runtime handles are rejected.
- Decoder binding transfers mutable GDN/QSA/PLE owners and invalidates their original handles. There is **no decoder import API**, so incompatible standalone imports cannot mutate checkpoint-owned decoder state through old aliases. Any future decoder import must invalidate the checkpoint/epoch just like reset; it must not bypass that contract.
- Capture/restore/destroy of a live checkpoint are rejected while native generation exclusively owns the decoder. This uses the existing registry ownership transfer and native runtime mutex, not unsynchronized access through a separate snapshot handle. Stale-handle destruction still releases registry metadata after reset/close.
- Capture is permitted at position zero; its logits-valid flag is false. At a completed positive position, the last device logits are included and restored as valid. There is no full-logit Python readback in normal checkpoint operations.

## Forward dependency contract

Snapshot only the committed dependencies needed for the next token:

| State | Snapshot/restore policy |
| --- | --- |
| Decoder | Committed position, epoch/owner identity, last-logits validity and device logits if valid |
| Each GDN | Committed convolution history and recurrence matrix; reset the layer length and invalidate scratch/output diagnostics |
| Each PLE | Committed 9-row normalized history, two predecessor IDs, row metadata, position and history-buffer allocation roles |
| Paged QSA | Five raw-index rows (first row plus four-slot ring); truncate logical KV length and invalidate resident mappings |
| Monolithic QSA | Truncate logical raw/K/V/pool visibility; committed prefix rows remain append-only |
| Completed QSA pools | **Never copied**; restore `floor(position/4)` visibility, and let the next normal step rebuild staging/new pools |
| HC / MoE / transient branch outputs | Invalidate output/trace state; next step reinitializes HC from its embedding and recomputes scratch |

The native PLE implementation swaps `history` and `next_history` at each commit. A snapshot's host copy targets the captured allocation, so rollback restores those allocation roles as well as its bytes. Copying bytes to that allocation without restoring roles is wrong after an odd number of suffix steps. Output/staged pointers need no restoration because their contents are scratch and diagnostics are invalidated.

The checkpoint does **not** contain sampled output IDs, EOS/pending samples, sampler RNG, an HC trace, routed expert results, full QSA raw/K/V arrays, or a copy of completed QSA pooled history. Direct sampling from restored resident logits is supported through the unchanged native sampler. The sampler's seed/counter remain under its existing owner; no response/RNG replay is implied by prompt caching.

## Paged-QSA rollback and readers

`QSAPagedState::truncate(length)` is the private rollback helper in `qsa_paged_runtime.inc`:

1. Validate truncation is backward within committed length and the generation can advance.
2. Wait for the pager completion event and synchronize the engine stream, retiring all existing page readers/copies.
3. Zero the device page table and fence its completion before changing/freeing host backing metadata.
4. Invalidate all host table entries and resident slot identities/generations. No stale tail can be reached through a resident mapping.
5. Keep the existing prefix backing pages; free only pages whose logical index is at or beyond `ceil(length/page_size)`. Update explicit host-page accounting. A partial tail page is retained in place; rows beyond the restored length are inaccessible until overwritten by append.
6. Advance pager generation, set logical length and populated-byte count, and report zero resident pages. Future gather remaps valid prefix rows normally.

There is **no KV prefix copy, prefix-page duplication, or full resident fallback**. All checkpoint operations run with native runtime serialization, no worker-owned decoder handle, and the existing engine-stream ordering. Prefix backing lifetime is protected until readers are fenced. Resident cache contents, generation and cache counters are not required to return to their old physical values; logical committed state and subsequent native results must replay bit-for-bit.

## Restore failure behavior

All owner/epoch/position/branch-state checks occur before mutation. After preflight the decoder is marked failed/unusable before the first host-to-device restore copy. Committed position is published only after every state copy, PLE role/metadata restore, pager truncation, length update and completion fence succeeds.

A partial copy/truncation failure leaves the session **quarantined**, invalidates its checkpoint/epoch, and requires reset. It does not pretend to undo a partially failed CUDA operation. Tests use a narrow test-only injected restore failure after a real copy is submitted; they verify failed-state rejection, checkpoint rejection, and recovery only through reset. Diagnostic/fault-injection symbols ending in `_for_test` are internal test hooks, not a supported serving API.

## Chat exact-request cache

The adapter captures immediately after consuming the prompt tail **except its final input token**. For full prompt `P`, the checkpoint position is `len(P)-1`. Its cache key is the complete actual token tuple `P`, not message text, conversation identity or a caller-supplied cache label.

On a fully known completed session and the same full token tuple:

1. Restore the saved native boundary.
2. Set Python committed bookkeeping to `P[:-1]` and clear the previous pending sample.
3. **Skip `decoder.consume` entirely**, including an empty consume call.
4. Start the existing native worker with `P[-1]` exactly once.

Thus the saved prefix is not re-prefilled, while the final prompt token is still the worker's first normal native decoder step. One-token prompts use the empty-state boundary. No generation API change or special first-token Python sampling path is needed.

After samples `S`, the unchanged invariant is:

```text
committed_tokens = P + S[:-1]
pending_token    = S[-1]       # not consumed, including EOS
position         = len(P) + len(S) - 1
```

A normal exact-prefix continuation still reuses all committed tokens and consumes the old pending token only if it is actually present in the incoming prompt. A different/shortened prompt that is not the exact cached request resets/replays. Each new request boundary replaces the sole old checkpoint: this is an **exact-last-request prompt cache**, not an LRU or arbitrary prefix branch cache.

Sampling arguments still go only to the native worker. Checkpoint restore preserves the advancing sampler RNG; stochastic repeated requests need not return identical responses. Explicit session reset retains the prior behavior of resetting the sampler to its configured seed. Greedy repeated requests and native state/logit replay are bitwise tested.

Cancellation/unknown unread queue state remains conservative: `require_reset=True` prohibits chat cache restoration. The next request resets and discards the checkpoint rather than guessing which samples were consumed. A caller using the native checkpoint API directly may restore after a successfully drained, nonfailed cancellation; that is separately tested and is **not** an automatic chat resume feature. A failed native decoder cannot be restored and must reset.

Additive `last_stats` fields:

- `checkpoint_restore`: whether the exact-request native rollback path was taken.
- `checkpoint_restore_seconds`: rollback ABI wall time.
- `checkpoint_host_bytes`: retained snapshot requested host storage.
- `reset_reason='prompt_checkpoint_restore'` identifies cache rollback without an actual reset.

A cache hit reports `reused_tokens=len(P)-1`, `new_prompt_tokens=1`, and `prefill_seconds=0`. Existing decode timing still includes the final prompt-input step plus sampling. Checkpoint capture overhead is included in `request_seconds`; it is not included in `prefill_seconds` and has no separate timing field. No timing number is presented as a performance claim.

## Executed tests and resources

Every GPU process used only:

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

Commands were bounded to 120 or 180 seconds by the tool runner. No GPU0 test/allocation, workload termination, training change, or full-model allocation occurred.

Fixtures:

- Existing `small_setup`: width 256, 48 narrow GDN layers, vocabulary 7, capacity 4, 512 MiB engine budget.
- `tests/native_prompt_fixture.py`: two synthetic decoder layers (QSA+PLE and narrow GDN), width 2560, vocabulary 8, capacity 16, 512 MiB engine budget. Reuses existing QSA/PLE/recurrent/HC fixture helpers. CPU experts are zero-valued sparse immutable packed files, not decoded/resident expert banks. This is **not** the actual checkpoint or a new model implementation.
- Measured mixed fixture device allocations before its sampler: **188,255,252 bytes FP32 QSA** / **188,230,932 bytes INT8 QSA** (about 180 MiB). Checkpoint creation adds no GPU allocations.
- Mixed host snapshot at prefix zero: **372,664 bytes**; at all tested positive prefixes: **372,792 bytes**, independent of prefix length. The extra 128 bytes at positive positions comprise 8 FP32 logits plus one copy-descriptor vector capacity expansion, not QSA history growth. Native test asserts device allocation remains below 512 MiB and snapshot below its explicit 1 MiB test budget.

| Executed validation | Final observed result |
| --- | --- |
| `cmake --build /tmp/mini-llm-wp1-build -j 2` | PASS, native library rebuilt |
| CPU-only `CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_native_chat.py --tb=short` | **12 passed, 0.15 s** |
| `.venv/bin/python -m pytest -q -s tests/test_native_engine_prompt_checkpoint.py tests/test_native_engine_chat.py tests/test_native_engine_generation.py --tb=short` | **26 passed, 9.67 s** |
| `compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q 'tests/test_native_engine_prompt_checkpoint.py::test_mixed_pool_page_tail_ple_eos_bitwise[3-reference_fp32]' 'tests/test_native_engine_prompt_checkpoint.py::test_mixed_pool_page_tail_ple_eos_bitwise[3-strata_int8_per64_fp16]' tests/test_native_engine_prompt_checkpoint.py::test_restore_partial_failure_quarantines_and_wrong_owner --tb=short` | **3 passed, 93.95 s; ERROR SUMMARY: 0 errors** |
| `python -m py_compile` for adapter, decoder binding, mixed fixture and new native tests | PASS |
| `git diff --check` | PASS |

Native checkpoint tests cover:

- GDN prompt consumption in chunks `(3)`, `(1,2)`, `(1,1,1)`, resident-logit native sampling replay, bounded state bytes, checkpoint replacement and stale handles.
- Mixed QSA/GDN/PLE prefixes **0, 3, 4, 5, 7**, odd/even suffix lengths **3/4**, both FP32 and INT8 QSA policies, repeated EOS ID 7 in PLE predecessor history, and alternate suffixes checked against reset+prefill.
- Bitwise suffix logits and bounded committed-state buffer replay, not loosened `allclose` checks.
- Pager truncation length/populated bytes, generation advancement, suffix-page host accounting, zero resident mappings and drained completion state. Raw-ring overwrite, newly completed four-token pools and partial-page tails are exercised.
- Repeated native chat requests forbid reset and prefill on cache hits, for both narrow GDN and mixed QSA/GDN/PLE; final pending sample, EOS-only output, continuations and disconnect reset behavior are checked.
- Host-budget rejection without losing a valid checkpoint, wrong decoder, reset-and-replay at the same position, engine-wrapper close, decoder close, full-queue cancellation and capture/restore/destroy rejection while worker-owned.
- Partial restore failure quarantines the decoder and invalidates the checkpoint; reset restores usability.

### Local failure found and corrected

Initial even-suffix tests passed. Inspecting the existing PLE commit path revealed pointer swaps, motivating an odd-suffix test. The first odd FP32 run failed the exact saved-state comparison at byte offset 248320. The localized correction retained/restored PLE history buffer allocation roles. Odd/even FP32 and INT8 tests then passed. No PLE equation, original fixture tolerance or strict diagnostic was changed. Memcheck subsequently covered the corrected odd-suffix mixed path and injected failure path.

## Limits / next gates

- Actual-checkpoint/long-context quality, strict whole-trajectory agreement and real serving-scale memory acceptance remain independent unresolved gates. The synthetic result is not a precision claim for the full model.
- Only the last exact request is cached. Other shortened prefixes recompute. No response cache, sampler RNG checkpoint, arbitrary branching, multi-checkpoint cache, shared-session concurrency or resume from unknown unread tokens.
- Current replay coverage is paged QSA in both formats plus GDN/PLE. The monolithic QSA path uses logical truncation but does not have a dedicated complete-decoder rollback fixture in this task.
- Snapshot transfer is synchronous, ordinary-host and bounded; no latency/throughput improvement is claimed. It must not be relabeled as batched prefill.
- Server implementation/new server tests belong to the other agent. This task edits only adapter/runtime/binding, its tests/fixture and WP8 documentation. Parent status files are untouched.
- Rebuild/restart native callers for the new checkpoint symbols; no chat API migration is required. The prior generation-event timing ABI extension is unchanged by this checkpoint task.
