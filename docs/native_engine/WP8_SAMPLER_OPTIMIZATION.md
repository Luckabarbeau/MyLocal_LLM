# WP8 isolated sampler optimization — measured implementation slice

Date: 2026-11-24. **WP8 completion is not claimed. WP7 strict acceptance remains
OPEN.** No parent status, manifest, server, model equation, precision flag,
reference tolerance, or existing strict decoder test was changed. Existing
`WP8_SAMPLER.md` describes the original implementation/evidence; this document
records the subsequent optimization, not a retroactive rewrite of that evidence.

## Source-supported change and preserved invariants

The pre-edit `sampling.cu` always sorted for greedy. Its `finish` kernel scanned
V entries for both cumulative sums, nucleus filtering, counting and normalization,
even when only k entries could carry mass. This was measured before editing.

Only production files changed in this slice:

- `mini_llm/native_cpp/sampling.cu`
- `mini_llm/native_cpp/sampling.h` (workspace comment)

No C ABI, Python sampler binding, decoder API or private runtime layout changed.

### Greedy

`prepare` still visits **all V logits**, validates finiteness and canonicalizes
signed-zero keys. It runs before either sort or reduction, including for greedy.
Greedy now calls **CUB DeviceReduce::ArgMax** on these prepared original keys,
then a scalar result kernel. ArgMax selects the smallest input offset for equal
maxima, preserving lowest-token-ID ties. There is no greedy sort. Nonfinite
status cannot be overwritten by the reduction/result path; arbitrary maxima of
sanitized invalid inputs never become successful tokens. Greedy still does not
advance the sampler's seed counter.

The retained CUDA Toolkit 13 KeyValuePair ArgMax overload is supported but
**deprecated** in favor of the newer separate value/index iterator API. The build
succeeds with deprecation warnings. No warning suppression was added.

### Stochastic top-k

Stable sorting, selected candidate IDs/order, temperature division, exp,
strict nucleus cutoff, renormalization and right-sided sampling are unchanged.
The serial prefix sums, nucleus pass and kept count now visit only indices
`0..k-1`, in **exactly the old order**. Binary search is bounded by k as well.

Before the change, every weight at index >=k was +0. The first pass has positive
mass because a finite selected maximum contributes exp(0)=1; appending +0 cannot
change that finite FP32 sum. The same holds after nucleus retention. Omitting
those trailing additions therefore does not change either cumulative sum, the
threshold or the chosen CDF interval. There is no reassociation, compensated sum,
tree scan, altered accumulation dtype or relaxed tolerance.

A separate parallel post-normalization kernel performs the same per-element
FP32 divisions for active entries. It initializes the entire diagnostic padding:
ID=-1, weight=+0 and normalized CDF=1 for every index >=k. No stale scratch from
an earlier call is observable in valid diagnostics. The exponentiation launch
now visits only k active entries; global finite validation still visits V.

**Full-vocabulary stochastic CDF remains serial.** When k=V, both prefix sums
and nucleus/count passes still execute exactly the old V additions/operations
in order. Only elementwise normalization moves to parallel threads; there is no
silent policy change to the full-vocabulary summation.

### Allocation accounting

One stable engine allocation still holds the same six V-element arrays and
40-byte result, plus aligned temporary storage. Creation now queries both:

- S(V): CUB stable sort scratch;
- R(V): CUB ArgMax scratch.

The admitted temporary requirement is `max(S(V), 256 + R(V))`. In the greedy
branch, the first 256-byte aligned region contains the KeyValuePair result and
reduction scratch starts at +256; stochastic sort instead uses the whole region.
These branches never overlap. The private runtime's existing admission and
accounting already use this queried maximum, so no additional allocation owner
or runtime API was needed.

Total requested bytes are:

```text
align_up(24*V + sizeof(ml_sample_result), 256) + max(S(V), 256 + R(V))
```

No per-sample allocation, hidden CUB allocator, source mutation, CPU sampling,
full-vocabulary production D2H or public device pointer was added. At V=248320,
sort scratch dominates, so workspace remains **7,980,799 bytes**, unchanged.
Original exact-budget and one-byte-short tests continue to pass unmodified.

## Before/after measurement

Every GPU/build/test command used the mask:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
```

Physical GPU1 was the sole visible CUDA device, ordinal 0, NVIDIA GeForce RTX
5070 Ti, CuPy 14.2.0. Training was not stopped or modified; the training process
was still present at the final check and physical GPU1 again showed 5797 MiB free.

Before any production source edit, the current binary was copied to
`/tmp/mini-llm-wp1-build/libmini_llm_native_wp8_before.so` and measured. Its SHA256,
original source SHA256 and original 47-test-file SHA256 are in the BEFORE JSON.
The original binary was also retained for later direct bitwise comparisons.

Both benchmark runs used the same pre-uploaded V=248320 F32 input generated by
NumPy default_rng(8), with identical input hashes, seed=8, 2 warmups and **7 timed
calls per mode**. Timings are synchronous **host-wall** milliseconds including
Python/C ABI dispatch, GPU work, compact result transfer and completion fence.
Input upload, full diagnostics, decoder/model execution and source setup are
excluded. Each call transfers only the existing 40-byte compact result.

| Mode | Before median ms | After median ms | Before / after |
| --- | ---: | ---: | ---: |
| Greedy, temperature=0 | 2.298592 | 2.243857 | 1.024x |
| Nucleus, temperature=.7 / top_k=50 / top_p=.9 | 197.598367 | 2.303404 | 85.785x |
| Full stochastic, temperature=1 / top_p=1 / no top-k | 104.251559 | 52.125156 | 2.000x |

The greedy implementation avoids sorting, but its observed host-wall improvement
is small; **no substantial greedy latency improvement is claimed**. No kernel
profiler was used to attribute the remaining approximately 2.2 ms. Concurrent
training and host/CUDA synchronization can affect wall timings; the available
evidence does not isolate those costs. Top-k's large improvement is consistent
with eliminating serial work on V-k padding. Full stochastic benefits from
parallel normalization without changing its serial prefix sums.

These are **sampler-only microbenchmarks**, not model tokens/s, serving latency,
full-model performance, isolated kernel timings or a WP8 performance acceptance
closure.

### Raw JSON and reproducible probe

- `WP8_SAMPLER_OPTIMIZATION_BEFORE.json`: all seven raw host timings and token IDs
  per mode, medians, environment, binary/source/input hashes, workspace, tied-input
  diagnostic snapshots and seeded compact result bytes.
- `WP8_SAMPLER_OPTIMIZATION_AFTER.json`: corresponding actual post-edit capture.
- `WP8_SAMPLER_OPTIMIZATION_BITWISE.json`: direct old/new binary comparison on the
  **random non-tied** input at k=50 and k=V, for top_p=.9 and 1. Includes complete
  array hashes, raw compact result hex and up to 64 active diagnostic entries as
  raw integer/FP32 bit patterns, rather than a huge full-vocab JSON array.
- `WP8_SAMPLER_OPTIMIZATION_RESULTS.json`: computed timing ratios, completed JUnit
  group counts, timeout records, memcheck scope and explicit non-green status.

The new probe is `tests/native_sampler_optimization_probe.py`. Executed commands
used the following common environment (also used for focused/regression tests):

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

```sh
# BEFORE: executed before editing sampling.cu
cp /tmp/mini-llm-wp1-build/libmini_llm_native.so /tmp/mini-llm-wp1-build/libmini_llm_native_wp8_before.so
PYTHONPATH=.:tests .venv/bin/python tests/native_sampler_optimization_probe.py \
  --library /tmp/mini-llm-wp1-build/libmini_llm_native_wp8_before.so \
  --output docs/native_engine/WP8_SAMPLER_OPTIMIZATION_BEFORE.json

# AFTER: same probe, input, seed, warmup count and repeat count
cmake --build /tmp/mini-llm-wp1-build -j 2
PYTHONPATH=.:tests .venv/bin/python tests/native_sampler_optimization_probe.py \
  --library /tmp/mini-llm-wp1-build/libmini_llm_native.so \
  --output docs/native_engine/WP8_SAMPLER_OPTIMIZATION_AFTER.json
```

Do not overwrite the BEFORE artifact with a rebuilt/new binary and call that a
baseline. The supplemental direct comparison checked the preserved old binary's
hash against the original BEFORE artifact before running it.

## Verification and bitwise evidence

`tests/test_native_engine_sampler.py` was **not edited**: all original 47 cases
remain unchanged, including numerical tolerances and Python-reference checks.
The new `tests/test_native_engine_sampler_optimization.py` verifies its original
SHA256 as well as the additional optimization invariants.

Latest focused command (120s limit):

```sh
.venv/bin/python -m pytest -q -rs \
  --junitxml=/tmp/mini-llm-wp1-build/wp8-opt-sampler.xml \
  tests/test_native_engine_sampler.py tests/test_native_engine_sampler_optimization.py
```

**64 passed in 7.18s**: original 47 plus 17 new cases.

Additional coverage:

- Full V with top-k 1/50/500, top_p .9/1/just below 1; source candidate order,
  right-sided CDF search and every padding slot verified.
- Six tied full-V snapshots (k=1/50/500, top_p=.9/1) match the pre-edit binary's
  complete ID/probability/CDF hashes and raw result bytes **bit-for-bit**.
- Four random-input snapshots (k=50/V, top_p=.9/1) likewise match the original
  binary bit-for-bit across all V entries, including nontrivial FP32 cumulative
  sums. The full-V comparisons explicitly guard unchanged serial-CDF behavior.
- Original/optimized seeded compact metadata bytes and benchmark token sequences
  match exactly; this is old/new native compatibility, not a backend RNG promise.
- All-equal +0/-0/lowest finite inputs choose lowest ID via parallel ArgMax.
- Nonfinite values at a discarded tail position still reject both greedy and
  stochastic; selected temperature overflow, invalid supplied uniforms and reset
  do not consume seeded draws. Seed+counter unsigned wrap is exercised.
- Workspace usage remains constant over repeats; source logits remain unchanged.

Full-CDF/ID hashes cover all elements, not only the short active arrays displayed
in JSON. Diagnostic calls are outside benchmarks and do not feed production
sampling. No production readback/flag/server seam was introduced.

### Focused memcheck

Executed with the same GPU mask and a 120s bound:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q \
  'tests/test_native_engine_sampler_optimization.py::test_bitwise_before_snapshot[0.9-50]' \
  'tests/test_native_engine_sampler_optimization.py::test_bitwise_before_snapshot[1.0-50]' \
  tests/test_native_engine_sampler_optimization.py::test_argmax_all_equal_and_tail_invalid \
  tests/test_native_engine_sampler.py::test_nonfinite_and_recovery
```

**8 passed in 0.95s; ERROR SUMMARY: 0 errors.** This covers full-V top-k-50
padding/normalization and full-V greedy reduction, plus error recovery; it is not
an exhaustive sanitizer run of every stochastic mode.

## Split native regression — partial, not a green full suite

Every group below had a separate **120-second** limit. Common command was
`.venv/bin/python -m pytest -q -rs --durations=3 --junitxml=...` with file scopes:

| Group | Files under `tests/` | Result |
| --- | --- | --- |
| transport | `test_native_engine_runtime.py`, `..._gguf.py`, `..._payload.py`, `..._upload.py`, `..._qwen_contract.py`, `..._cpu_experts.py` | 119 passed, 1 skipped, 1.27s |
| blocks | `test_native_engine_chaining.py`, `..._hc.py`, `..._ple.py`, `..._ple_primitives.py`, `..._recurrent.py` | 98 passed, 1 skipped, 18.29s |
| moe-decoder | `test_native_engine_moe.py`, `..._moe_chaining.py`, `..._decoder.py`, `..._divergence.py` | 52 passed, 23 skipped, 1 deselected, 6.80s |
| qsa-primitives | `test_native_engine_qsa.py`, `..._qsa_attention.py`, `..._qsa_frontend.py`, `..._qsa_projection.py` | 227 passed, 2 skipped, 7.78s |
| qsa-hidden | `test_native_engine_qsa_hidden_frontend.py` | 16 passed, 1 skipped, 8.26s |
| kv | `test_native_engine_kv_pages.py`, `..._kv_streamed.py` | 29 passed, 6.41s |
| qsa-full-capacity | `test_native_engine_qsa_full_capacity.py` | 2 passed, 4 skipped, 0.96s |
| sampler | Original sampler and new optimization test files | 64 passed, 7.18s |

Completed groups total **607 passed, 32 skipped, no failures/errors in those
completed groups**. Full file lists and parsed JUnit counts are in RESULTS JSON.
Skips retain existing opt-in behavior: second CUDA device unavailable under the
required mask, optional PLE compile, actual MoE/chaining/decoder and QSA checkpoint
or timing gates. No unmasking or full decoder checkpoint execution was attempted.

The MoE/decoder group explicitly used:

```sh
MINI_LLM_DECODER_CHECKPOINT=0 .venv/bin/python -m pytest -q -rs --durations=3 \
  --junitxml=/tmp/mini-llm-wp1-build/wp8-opt-moe-decoder.xml \
  tests/test_native_engine_moe.py tests/test_native_engine_moe_chaining.py \
  tests/test_native_engine_decoder.py tests/test_native_engine_divergence.py \
  --deselect=tests/test_native_engine_decoder.py::test_synthetic_all48_reference_replay_lifetime
```

**That known WP7 strict synthetic failure remains unresolved and unchanged.** Its
deselection is not a passed test, numerical acceptance closure or evidence that
the full suite is green. No model precision gate or tolerance was modified.

Two other group attempts did **not** finish:

1. `test_native_engine_qsa_layer.py`, `test_native_engine_qsa_paged.py`,
   `test_native_engine_qsa_full_capacity.py`: timed out after 120s with ten
   completion dots, no final summary/JUnit file. The queued paging/full-capacity
   files had not been reached.
2. The not-yet-reached `test_native_engine_qsa_paged.py` plus
   `test_native_engine_qsa_full_capacity.py`: timed out after 120s with eleven
   completion dots, again without a final summary. Full-capacity was subsequently
   run on its own and passed as listed above.

Neither timeout is counted as a pass. **QSA layer and paging regression remain
incomplete.** No longer timeout, unrelated QSA edit or tolerance change was
attempted. A useful next experiment is caller-approved finer shards of those
specific parameterized trajectory tests (or a longer bound); there is no evidence
here supporting an implementation change in QSA.

Build and `git diff --check` passed. The build's supported-but-deprecated CUB
ArgMax API warning remains visible. Parent status files remain parent-owned and
unchanged. No full-model speedup, model numerical parity or WP8/WP7 completion is
claimed.
