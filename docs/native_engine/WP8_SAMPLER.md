# WP8 independent GPU sampler — implementation/evidence slice, acceptance OPEN

Date: 2026-11-24. **Not a claim of WP8 completion. WP7 strict acceptance remains
OPEN.** No server integration, production flags, generated-model quality claim,
or change to the existing strict synthetic decoder test. Parent-owned
`IMPLEMENTATION_STATUS.md` and `native_reference_manifest.json` are unchanged.
Existing WP6/WP7 and training work was preserved. In particular the known failing
`test_synthetic_all48_reference_replay_lifetime` was neither edited nor relaxed.

## Authority and exact candidate ordering

The import chain actually inspected is
`scripts/serve_qwen4exp.py` -> `mini_llm.generation.sample_token`, not the separate
PyTorch inference sampler. This slice supports its temperature/top-k/top-p graph
without request-scoped repetition/frequency/presence penalties.

The active reference is **CuPy 14.2.0**. Inspection of installed
`cupy.argpartition` confirms that it fully sorts using `argsort`; installed
`cupy.argsort` documents stable sorting. A GPU probe confirmed stable ascending
IDs for equal values and reversed IDs after `argsort(...)[::-1]`.
This matters: NumPy's partial-partition order is **not** the contract here.

For a finite F32 vector with vocabulary V (1..248320):

1. Validate temperature >= 0 and `0 < top_p <= 1` before choosing a branch.
   Greedy temperature 0 returns the lowest ID among maximal original logits,
   matching first-occurrence `argmax`. Top-k does not affect greedy.
2. Stochastic `k=min(V, top_k)` or V for `None`. Explicit Python top-k is an
   integer >=1; values >V clamp. The C ABI uses top_k=0 as the `None` sentinel.
3. If k<V, stable ascending sort of **original** logits, then take the final k
   entries. Thus equal-logit top-k boundary ties choose **higher IDs**. Candidate
   order at this stage is ascending score, ascending ID.
   If k=V, candidate order instead starts as ascending token ID (no partition).
4. Divide selected logits by FP32 temperature. If `top_p < 1`, stable ascending
   sort of these scaled values followed by reversal, including ties. For k<V,
   positive division is monotone, so reversing the original sorted selected tail
   exactly preserves the stable scaled-sort semantics, including new rounding
   ties. For k=V, sort scaled values themselves: ties then reverse by ID.
5. FP32 `weight=exp(scaled-max(scaled))`. Compute cumulative mass and retain
   **strictly** `(cum - weight) < float32(top_p) * total`. No <= substitution,
   no top-p-before-top-k, no threshold divided by full unfiltered-vocab mass.
6. Recompute the retained cumulative mass (renormalization). Search with
   `side='right'` at `double(uniform)*double(FP32_total)`. Return that candidate ID.
   With top_p=1 and k<V this CDF is ascending selected-score/ID order, **not**
   full token-ID order. With top_p=1 and k=V it is token-ID order.

The ABI accepts double scalar temperature/top_p so branch decisions are not
prematurely rounded to F32. Scaling and the nucleus threshold remain F32 like
the reference. A top_p just below 1 that rounds to F32 1 still takes the nucleus
branch. Signed-zero keys are canonicalized so radix stability implements numeric
comparison ties, not an artificial -0/+0 distinction.

This is a filtering/distribution/CDF contract, **not bitwise CuPy arithmetic
parity**: CUDA exp and sequential FP32 scan can differ from backend exp/parallel
scan by roundoff. A uniform arbitrarily close to a rounded CDF boundary can
therefore yield a different token. Tests use fixed `rtol=2e-6, atol=2e-7` for
probabilities/CDF, supplied uniforms, and an exactly representable CDF boundary.
No tolerance was loosened to make a failing numerical comparison pass.

### Explicit safety restrictions

All V input logits, including top-k-discarded ones and greedy input, must be
finite. Positive temperature must remain positive and finite when converted to
F32. Selected scaled values must remain finite. Violations return ML_INVALID,
never an arbitrary token or a CPU fallback. This is intentionally stricter than
the Python source's lack of explicit finite checks (which may produce NaN mass).
The binding accepts integers only for explicit top-k rather than Python's broad
`int(...)` coercion. Penalty arguments are not silently ignored; they are absent.

Compact device status: 1 nonfinite input, 2 selected scaling overflow,
3 nonpositive/nonfinite total mass. The C call returns ML_INVALID and preserves
the device status in its result on such failures. Python raises NativeError.
Failed attempts invalidate diagnostics; a later valid sample can recover without
reallocation or reset. No sampler failure poisons the source decoder.

## Implementation and provenance

- `mini_llm/native_cpp/sampling.h` / `sampling.cu`: new native CUDA sampling code.
  Uses installed CUDA Toolkit 13 CUB `DeviceRadixSort::SortPairs`, stable ascending
  keys with initial ascending IDs, to implement an explicit score/ID total order.
  No Thrust default allocator, downloaded dependency, copied Strata sampling
  kernel, CPU math fallback, or Python numerical execution seam.
- `sampler_runtime.inc`: private engine-retaining owner, same registry mutex,
  exception/status boundary, DeviceGuard and stream-completion discipline as the
  existing runtime. Included after `decoder_runtime.inc` for private logits access.
- `runtime.h`, `runtime.cpp`, `CMakeLists.txt`, `mini_llm/native/__init__.py`:
  thin C ABI / ctypes signatures / engine factory and owner binding only.
- `decoder_runtime.inc` / `mini_llm/native/decoder.py`: localized opt-in suppression
  of the existing full-logit copy. Original `step(token)` still returns host
  logits and retains the same computational graph and commit/reset semantics.
- `tests/test_native_engine_sampler.py`: new independent NumPy FP32 probability/CDF
  oracle that reproduces **CuPy's stable** sorting contract, plus tests exercising
  the actual imported `generation.sample_token` with CuPy and supplied uniforms.
  Existing GGUF upload and synthetic decoder fixtures are reused, not replaced.

All sampling math, ordering, input validation, seed mixing and token choice occur
on GPU. A normal sample transfers **only `sizeof(ml_sample_result)=40` bytes D2H**:
token, finite-validation status, positive-mass candidate count, uniform, raw retained
total, chosen probability and normalized CDF interval. `metadata=False` returns
only the Python integer token, but transport remains this compact struct.
No source logit is written or copied to host by sampling.

The probability/CDF pass currently runs on one GPU thread after parallel exp and
CUB sorting. This is simple/deterministic but slow at full V; performance is an
explicit remaining limitation, not evidence of production readiness.

## API, seed and lifetime

```python
with runtime.engine(budget_bytes=32 * 2**20) as engine:
    with engine.sampler(248320, seed=123) as sampler:
        token = sampler.sample(uploaded_f32_logits, temperature=.7,
                               top_k=50, top_p=.9)
        result = sampler.sample(uploaded_f32_logits, temperature=.7,
                                top_k=50, top_p=.9, uniform=.25, metadata=True)
        sampler.reset()            # same initial seed/counter
        sampler.reset(seed=456)    # new explicit seed/counter
```

`sample` requires immutable UploadedTensor F32 `[V]` or `[1,V]` on this exact
engine, not merely the same CUDA device. Inputs are retained through synchronized
completion. Closing public engine/store/lease handles does not invalidate the
native sampler/upload owners. Closed/stale handles, wrong engine/type/shape and
insufficient budgets fail before launching sampling work. No device pointer
escapes the C ABI.

For an existing valid same-engine decoder of matching vocabulary:

```python
position = decoder.step_device(input_token)  # one native decoder call, no logit D2H
next_token = sampler.sample_decoder(decoder, temperature=.7, top_k=50, top_p=.9)
```

`sample_decoder` alone does not advance or retain a mutable decoder between calls;
it samples its latest successfully committed device logits. Before first step,
after reset, after failed step or after close, it rejects stale/unavailable logits.
`step_device` uses the existing `ml_decoder_step` with null logits/count=0 and returns
only the committed position. Existing `step()` still explicitly copies all logits;
that host path must not be confused with the device-resident path.

Seed stream: GPU SplitMix64 finalizer on unsigned `(seed + counter)` modulo 2^64,
including the standard initial addition `0x9e3779b97f4a7c15`; high 53 bits times
2^-53 produce a double uniform in [0,1). Seed is an explicit uint64. Counter starts
at zero and advances only on successful stochastic calls without supplied uniform.
Greedy, invalid calls and supplied-uniform diagnostics do not advance it. Reset
sets counter=0 and invalidates diagnostics. Tests verify this algorithm against an
independent integer oracle, not against reference RNG token sequences. There is
**no promise of identical CuPy/NumPy/PyTorch RNG or seeded token sequences**.

`sampler.readback_for_test()` is the only full-array sampler diagnostic. It copies
IDs/probabilities/normalized CDF of length V. If k<V, the final V-k slots have ID=-1,
probability=0 and final CDF=1; these are padding, not extra candidates. Available
only after a valid stochastic call. There is no production flag enabling full
logit diagnostics or server sampler selection.

## Stable workspace and budget

Creation queries CUB scratch size without allocating it. One cudaMalloc contains:

- two F32 sort-key arrays and two int32 ID arrays (16*V bytes);
- F32 weights and CDF arrays (8*V bytes);
- the 40-byte compact result, followed by 256-byte alignment padding;
- exactly the queried CUB temporary storage.

Requested device bytes = `align_up(24*V + 40, 256) + cub_temp_bytes(V)`.
No sampler-owned pinned payload, per-sample cudaMalloc, hidden sort allocator,
full-vocab host staging or mutation of original logits. The engine admits the
**entire** allocation before cudaMalloc, tracks peak/current bytes and releases
it after the completion fence. Driver/context/library overhead is outside the
existing requested-payload budget, as with other owners. On this GPU/toolkit,
V=248320 requested **7,980,799 bytes** sampler workspace, plus 993,280 bytes for
an uploaded F32 input. Exact-budget success and one-byte-short rejection pass.

An intentional invalid-device test revealed CUDA's already-reported last-error
slot leaking into CUB's next query. The CUB entry points clear that per-thread
slot before their own checked operations; failed asynchronous completion is still
checked through stream synchronization. No global runtime error policy changed.

## Executed validation — physical GPU1 only

All GPU/build/test commands used:

```sh
CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
```

Observed CuPy visible count=1, device ordinal=0, NVIDIA GeForce RTX 5070 Ti.
This mask selects physical GPU1; `nvidia-smi` itself lists physical indices despite
the mask. Observed physical GPU1 free memory before work: 5797 MiB; a later masked
CUDA probe reported 5,839,847,424 free bytes. No command stopped or modified the
existing training workload. Full-model checkpoint execution was not enabled.

Common focused environment (literal tested paths):

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_BACKEND=cupy
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build /tmp/mini-llm-wp1-build -j 2
.venv/bin/python -m pytest -q -s tests/test_native_engine_sampler.py
```

Most recent final-source result: **47 passed in 12.81s**. Build succeeded;
existing `gguf_loader.cpp` misleading-indentation warnings were not changed.
Coverage includes V=1/9/513/248320, ties, signed zeros, extreme finite temperatures,
top-k/p combinations, p just below 1, right-CDF boundaries, unchanged original
logits, seeded replay/uniform non-consumption, nonfinite input/overflow and
recovery, exact/insufficient budgets, close/retention, stale/null handles,
wrong type/size/shape/engine, invalid device and decoder validity/reset/replay.
There are 324 direct CuPy-source stochastic comparisons in the tied-case test
alone. No full-model sampler quality claim follows from the synthetic decoder.

Most recent final-source bounded memcheck (120s command limit):

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest -q \
  tests/test_native_engine_sampler.py -k 'greedy_ties or nonfinite_and_recovery'
```

**4 passed, 43 deselected in 0.75s; ERROR SUMMARY: 0 errors.** An earlier larger
memcheck subset including large-V probability/CDF cases timed out after 120s,
showing two completed cases; it is **not a memcheck pass**, and full-V memcheck
coverage remains open. No longer timeout was attempted.

Most recent final-source broad native command, also bounded to 120s:

```sh
MINI_LLM_DECODER_CHECKPOINT=0 .venv/bin/python -m pytest -q -rs \
  tests/test_native_engine*.py \
  --deselect=tests/test_native_engine_decoder.py::test_synthetic_all48_reference_replay_lifetime
```

**TIMED OUT after 120s beyond 67% progress**, with passes/skips shown and no failure
shown before termination, but no final summary. An earlier broad attempt likewise
timed out. This is **incomplete validation**, not a passed broad suite or green
full suite. The known strict synthetic decoder failure was explicitly deselected;
the full checkpoint decoder gate was disabled to preserve training headroom.
Other existing opt-in tests retain their own defaults/skips. The next useful
broader experiment is a caller-approved longer run or bounded shards of this same
broad suite; no unrelated implementation or tolerance should be changed because
of the timeout.

`git diff --check` passed. Parent status and strict WP7 acceptance are unchanged.

### Sampler-only microbenchmark (not model performance)

Final focused run, physical GPU1 concurrent with existing training, V=248320,
pre-uploaded input, one warm-up and 10 synchronous calls per mode. Host-wall median
includes Python/C ABI overhead, sort, exp/CDF, compact result transfer and fence;
excludes input upload, decoder/model execution and full diagnostics:

| Mode | Median milliseconds |
| --- | ---: |
| Greedy, temperature=0 | 2.296 |
| Nucleus, temperature=.7 / top_k=50 / top_p=.9 | 204.852 |
| Full stochastic, temperature=1 / top_p=1 / no top-k | 112.303 |

This unoptimized deterministic scan is a known performance limitation. Timings
under concurrent training are not isolated CUDA kernel latency, model tokens/s,
end-to-end serving performance or an acceptance closure.
