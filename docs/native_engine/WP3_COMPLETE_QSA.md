# WP3 complete QSA layer integration — incomplete verification

Date: 2026-11-23. **Not an accepted/completed WP3 milestone.** Parent-owned
`IMPLEMENTATION_STATUS.md` and `native_reference_manifest.json` were not edited.
Pre-existing uncommitted work was preserved. No production dispatch or fallback
was added.

## Implemented contract

`Engine.qsa_layer(capacity, query_gate, key, value, index_query, index_key,
output, gamma_q, gamma_k, gamma_iq, gamma_ik, *, policy, epsilon=1e-6,
attention_scale=1/16)` creates a native layer owner. Capacity is exactly 4096 or
16384. Explicit policy is `ggml_cpu` or `strata`. All ten inputs are retained
uploaded tensors from the same native engine; the frontend/output weights are
bound explicitly, not looked up by name or substituted. Hidden input is an
uploaded F32 vector of width 2560. Fixed geometry is Q24/KV2 x 256 and index Q4 x
128, rotary width 64/base 1e7. Converted gamma is used directly.

`layer.step(hidden, position)` requires position equal committed length and
less than capacity, returning only the new committed length. It runs all five
matvecs, frontend, scoring/selection, selected GQA, gate, and output projection
inside C++ on the engine stream. No Python intermediate, IDs readback, host
activation computation, or full matrix decoding occurs in step. Output remains
in owned GPU storage. This is the attention layer only: no HC/residual/MLP.

Forward dependency chain:

```text
hidden -> Q/gate,K,V,IQ,raw projections
       -> normalized/rotated Q,K,IQ; gate logits; dead; conditional pool
       -> staged historical pools + current completed pool
       -> block scores, incomplete dead + 1e9, policy-specific IDs
       -> selected attention (history plus current frontend K/V before commit)
       -> sigmoid gate -> output matvec -> scalar status validation
       -> append current raw/K/V and completed pool -> commit synchronization
       -> publish length and output-valid flag
```

`n=position+1`. Pooling remains chronological left-to-right FP32. First raw and
partial history pointers refer directly to owned full raw history. Committed raw,
KV and pooled arrays are not changed before output/status validation. Current
pool is staged separately and used for current scoring. The selection helper now
has an internal `clear_status` argument, defaulting to true for existing users;
complete-layer execution passes false so frontend failures cannot be erased.

State includes full raw history, full KV, committed pools and a separate staged
pool array. No rolling-window/history optimization was attempted. All arrays and
scratch are admitted once and stable through steps/reset. Native shared ownership
retains weights and engine independently of public handles. Registry handles are
monotonic; same-engine, shape, supported format, position and capacity checks are
native. Reset clears storage and snapshots without reallocation.

Before cache-copy commit, state is marked invalid; only successful stream
completion clears invalid and publishes length. A CUDA error in commit therefore
requires reset and does not expose resumable partial state. CUDA failures in the
compute phase also poison state; finite-status rejection leaves committed caches
unchanged and invalidates output. Actual CUDA fault injection has **not** been
executed.

## Explicit import and diagnostics

`layer.import_reference_state(length, raw_index, keys, values, pooled)` accepts
only an empty valid layer and length in [4,capacity]. Inputs must be same-engine
F32 uploads of shapes `[length,128]`, `[length,512]`, `[length,512]`,
`[length//4,128]`. Native validation checks shape, dtype, bounds and all consumed
values for finiteness before copying. Import commit is also poisoned until
synchronization succeeds. It does **not** prove semantic consistency of arbitrary
caller history with weights/gamma/positions. The test fixture establishes that
provenance by constructing every raw/K/V row and chronological pool with the
unchanged reference operators, avoiding thousands of attention steps.

`position` returns scalar committed length. `readback_for_test(which, destination)`
is explicit diagnostics only: `output`, `raw`, `keys`, `values`, `pooled`, `ids`,
`scores`. State copies only committed prefixes. Output/IDs/scores require a valid
successful step snapshot. IDs are int64; other destinations are contiguous native
F32. Diagnostic arrays are never fed into native step.

## Files

- `mini_llm/native_cpp/qsa_layer_runtime.inc`: private native state owner and ABI
  implementation, included after runtime helpers.
- `mini_llm/native_cpp/qsa_layer.cu`: native finite validation for imported arrays.
- Targeted additions in `runtime.cpp`, `runtime.h`, `CMakeLists.txt` and
  `mini_llm/native/__init__.py`.
- `qsa_selection.{h,cu}`: status-preserving internal launch option.
- `tests/test_native_engine_qsa_layer.py`: synthetic reference trajectories,
  lifecycle/import/rejection/accounting cases, optional bounded checkpoint test.

## Executed validation and stop condition

Build completed successfully:

```sh
cmake --build /tmp/mini-llm-wp1-build -j 4
```

Three existing misleading-indentation warnings remain in unchanged
`gguf_loader.cpp`.

The following exact command was executed twice:

```sh
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest -q tests/test_native_engine_qsa_layer.py -x
```

Both runs completed **16 trajectory cases successfully**, then failed in
`test_invalid_inputs_import_and_budget`. Most recent result:
**1 failed, 16 passed in 8.00s**. This is not a passing test command.

Verified trajectory coverage (unchanged between successful case executions):

- Both explicit policies and flat/nonflat controlled scoring fixtures.
- Fresh sequence of nine tokens, crossing n=4 and n=8 pool completion.
- Imported length 2050 followed by five steps, including n=2051/2052/2053.
- Imported length 4092 followed by four steps, reaching full 4K capacity.
- Imported length 16380 followed by four steps, reaching full 16K capacity.
- Exact selected IDs against unchanged `Qwen4ExpAttention.step` diagnostics.
- Every output and complete committed state prefix compared at every step.
- Position/capacity rejection with exact unchanged committed arrays, reset and
  replay, weight/engine public-close ownership, stable step accounting, and
  accounted peak less than 512 MiB in each passing trajectory case.

Predeclared output tolerance: rtol=2e-4, atol=2e-4. Scores: rtol=5e-5,
atol=2e-4 (IDs still exact). State base bound: 5e-5*(1+abs(reference)), plus
per-coordinate long-position angle allowance on rotated K/pools:
`8*eps_f32*position*frequency*(abs(first_half)+abs(second_half))`.
All arithmetic is F32. No finite differences or post-failure tolerance adjustment.

First failure was a test expectation, not evidence of a production defect: after
nonfinite-step rejection, a short-output diagnostic expected `capacity`, but output
was correctly unavailable. One localized correction produced another successful
reference/native step before checking short-buffer capacity.

That corrected check passed on rerun. A later line in the same test then failed:

```text
small = stack.enter_context(r.engine(budget_bytes=1))
mini_llm.native.NativeError: invalid engine configuration
```

The fixture intended to check cross-engine rejection before layer allocation, but
its 1-byte engine is invalid under the existing engine API. Concrete remaining
cause is an invalid test fixture budget. No numerical discrepancy was observed in
passing trajectory cases. The single next change to investigate is replacing that
fixture's budget with a valid engine budget (the test already creates a valid
2**20-byte other-engine instance), followed by rerunning the same command.

**Stopped after the one localized correction and failing rerun**, per the user's
execution limit. No second correction was made. Exact-admission tests after this
failure were not reached. Optional actual-checkpoint repeated-step parity was
written but not executed. Existing 353-pass/3-skip baseline was not rerun in this
turn; it is prior evidence, not proof for this modified state. No sanitizer,
CUDA fault injection, broader regression or complete-layer performance result is
claimed. A benchmark was not published because the requested correctness gate
has not passed. Parent owns final acceptance/status.
