# WP3 projected-input QSA frontend

Date: 2026-11-23. **Bounded experimental frontend verified; complete WP3 / full
attention / production acceptance is not claimed.** Parent
`IMPLEMENTATION_STATUS.md` remains exclusively parent-maintained and was not
edited by this task. Existing dirty work was preserved.

## Scope and API

Integrated the existing `mini_llm/native_cpp/qsa_frontend.{h,cu}` into CMake,
`runtime.{h,cpp}`, and the experimental `mini_llm/native/__init__.py` binding.
The kernel files and production Python attention/norm/RoPE implementations were
not modified. The kernels previously had standalone compilation evidence only;
this milestone adds runtime numerical tests, not model-level validation.

```python
with engine.qsa_frontend() as frontend:
    frontend.set_n(n)  # INCLUDING-current length, text position n-1
    frontend.run(qg, k, raw, iq, first_raw, partial,
                 gamma_q, gamma_k, gamma_iq, gamma_ik, epsilon=1e-6)
    device_n, pool_ready = frontend.readback_for_test(q, k_out, gate, iq_out, dead, pool)
```

All ten inputs are existing immutable `UploadedTensor` handles from the same
runtime **and engine**, exact F32 descriptors with Python row-major shapes:

| Input | Shape |
|---|---|
| qg (per-head query then gate logits) | `[24,512]` |
| k | `[2,256]` |
| current raw index key | `[128]` |
| iq | `[4,128]` |
| historical first raw key | `[128]` |
| partial raw keys, chronological | `[3,128]` |
| converted gamma Q / K | `[256]` each |
| converted gamma IQ / IK | `[128]` each |

Outputs remain device-resident: q `[24,256]`, K `[2,256]`, copied gate logits
`[24,256]` (not sigmoid), IQ `[4,128]`, dead `[128]`, pool `[128]`, plus an
integer pool-ready flag. Readback is explicit diagnostics into caller-provided
writable contiguous native F32 buffers, with native element-capacity checks
before any copy. NumPy arrays are used only as test/reference/diagnostic storage;
no device address escapes. Diagnostic n is copied from the actual device scalar,
not returned from the host shadow.

The internal fixed allocation is **54,288 device bytes + 16 pinned bytes**.
Device layout: uint64 n, 13,568 F32 output elements, int pool-ready, int status.
Pinned storage holds stable n staging and scalar status staging. Admission and
accounting reuse the shared engine budget; no allocations during run, set_n or
reset. Tests verify exact admission boundary 54,303 rejected / 54,304 accepted,
unchanged counters across repeated runs/resets, and release on workspace close.
Allocator/context overhead is excluded, as in the existing runtime.

`set_n` synchronizes before changing pinned staging, copies the uint64 scalar to
the stable workspace device address, and synchronizes before return. It rejects
zero; `reset` explicitly uploads zero and invalidates output without reallocating.
Run refuses an unset/reset n. All positive uint64 values are representable by the
API: tests exercise UINT64_MAX for finite output and exact device-n transport,
not reference numerical parity at that extreme. RoPE position conversion remains
FP32 and loses integer precision at large positions.

Each native run attempt first invalidates the old snapshot, validates arguments,
retains shared ownership of all uploads, waits for uploads, clears device status,
launches on the engine stream, and synchronizes scalar status before permitting
readback. The completion guard drains before input owners can be released on
error paths. Workspace ownership retains the engine independently of its public
handle. Binding-only rejections (wrong Python object/runtime, already-closed
Python object) occur before the native run attempt; they do not execute a kernel
or promise invalidation of an existing native snapshot. Native invalid-handle,
shape/type/engine/epsilon/status failures do invalidate it.

## Established computational graph

- RMS: `(x * (1/sqrt(mean_fp32(x*x) + epsilon))) * converted_gamma`;
  **no additional `1+gamma`**. Epsilon is a finite positive C float.
- Q, K and IQ: normalize, then half-layout RoPE at text position `n-1`,
  rotary width 64, base 1e7; suffix `[...,64:]` remains normalized only.
- Gate: exact copied query-gate second half, no normalization or sigmoid.
- Dead: normalized current raw at n=1, otherwise normalized first raw;
  **never rotated**.
- Only at `n%4==0`: normalize
  `(((partial[0]+partial[1])+partial[2])+raw)*float32(0.25)`, then RoPE
  at `n-4`. Raw additions remain explicitly left-to-right FP32.
- Flag is overwritten on every launch. When false, pool storage is untouched
  (initially zero) and is **not a newly completed pool**. Reset invalidates access
  but need not erase storage.
- Nonfinite consumed sources, squares/reductions, sums, normalized values or
  outputs fail device status. Unused first_raw at n=1 and partial at non-pool
  positions are deliberately not checked. Current raw is checked even without
  pooling. A ready flag alone is not validity; status completion gates readback.

## Tests and numerical limitations

`tests/test_native_engine_qsa_frontend.py` adds **80 tests**. Reference calculations
instantiate unchanged `Qwen4ExpAttention`, use its four `norms` and `_rotate`, and
reproduce its left-to-right raw-pool equations. Projection callbacks deliberately
raise if called. Reference and device parameter agree at:

`1,2,3,4,5,127,2048,2051,2052,2053,4096,16384`.

Both epsilon `1e-6` and `0.125` are checked. Gamma fixtures include zero/negative
values and are used directly. Each output is individually compared, inputs remain
byte-identical, repeated runs have stable budgets, and pool-not-ready preserves
the previous pool exactly. Additional cases cover all ten shapes/types, all ten
consumed inputs with NaN/+Inf/-Inf, unused-history poison, finite-input square,
pool-sum and normalized-output overflow, epsilon overflow/underflow, n bounds,
reset, closed/stale handles, cross-engine/runtime rejection, diagnostic
type/contiguity/capacity rejection, and engine close with retained snapshots.

**Not bitwise parity for normalized/rotated floating outputs.** The existing
kernel uses a tree RMS reduction and CUDA `powf/sinf/cosf`; the NumPy reference
uses different reduction/transcendental arithmetic. Gate logits, device n, flag,
input immutability and untouched pool storage are tested exactly. All consumed
floating output elements must be finite and satisfy an explicit componentwise
error bound, not an indiscriminate large global tolerance:

- Base allowance: `3e-6 * (1 + abs(reference))`.
- Only rotary coordinates add
  `8 * eps_f32 * position * frequency_i * (abs(ref_i)+abs(ref_i+32))`,
  for each half-layout pair. Dead and nonrotary suffix get no angle allowance.
- `frequency_i = float32(1e7)**(-float32(i)/32)`.

The rotary bound accounts conservatively for a few ulps of frequency and angle
arithmetic growing with position. It is a test allowance, not a formally proved
universal CUDA error bound. Maximum observed absolute discrepancy across tested
positions was **0.000641822815** (n=16384, epsilon=1e-6); at n=1 it was
**1.43051147e-6**. Epsilon=0.125 maximum was **0.000601291656**. No tolerances were
relaxed after a failure, and no production/kernel numerical changes were made.

## Commands and verified results

From the repository root, using the existing external build directory:

```sh
cmake --build /tmp/mini-llm-wp1-build -j 4
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q -s tests/test_native_engine_qsa_frontend.py --maxfail=1
MINI_LLM_BACKEND=numpy MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_*.py
git --no-pager diff --check
```

- Build **passed**, including frontend CUDA compilation and shared-library link.
  Three existing `gguf_loader.cpp` misleading-indentation warnings remain.
- Initial focused execution: **79 passed / 1 failed**. The test incorrectly
  expected `ValueError` for an already-closed Python upload; established
  `_require_open()` correctly raises `NativeError(1, "object is closed")`.
  One localized test assertion correction only; no runtime correction.
- Exact focused retest: **80 passed in 0.63s**, no skips.
- Combined native-engine suite: **337 passed, 2 skipped in 6.14s**, preserving
  the prior 257 passing tests plus the 80 new cases. The two existing skips remain;
  frontend tests do not silently skip missing CUDA/library execution.
- `git diff --check`: **passed** (tracked diff, including pre-existing changes).
- Test GPU: NVIDIA GeForce RTX 5080, driver 580.178.04; CUDA toolkit as configured
  in the existing build directory. No alternate GPU/backend numerical claim.

## Remaining gates / exclusions

No packed input projections, conversion/dequantization, session state commits,
cache mutation, native frontend-to-selection/attention consumption, complete
attention layer, end-to-end logits/generation, CUDA graph capture, asynchronous
public completion API, performance acceptance, sanitizer/leak-detector validation,
or production backend integration. The tests establish bounded synthetic
projected-input behavior, not checkpoint/model compatibility. Prior standalone
kernel compilation was not treated as numerical validation. Parent WP3 remains
open until its separate complete-layer gates are met.
