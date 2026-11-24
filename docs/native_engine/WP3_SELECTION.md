# WP3 — first bounded computational milestone: GPU index scoring + selection

Date: 2026-11-23. **WP3 remains in progress.** This milestone implements a native
CUDA operator integrated with the experimental engine runtime, C ABI, Python
binding, allocation budget and tests. It is **not full QSA attention or model
inference**. Parent-owned implementation status and reference manifest were not
updated by this milestone. Existing uncommitted WP2 changes were preserved.

## Source contract

Authority: `mini_llm/external/qwen4exp_attention.py`, `select_cells` lines 71–127
and `step` lines 312–363; `docs/qwen4exp/reference_contract.md`, QSA section.

Inputs are exact stored-byte F32 uploads from the **same engine**:

| Input | Python row-major shape | GGML fastest-first shape |
|---|---|---|
| Already normalized/rotated index queries | `[4,128]` | `[128,4]` |
| Committed completed pooled index keys | `[capacity/4,128]` | `[128,capacity/4]` |
| Normalized dead index key | `[128]` | `[128]` |

Capacity must be a positive multiple of four, at most `INT32_MAX-3`.
`n` is a runtime scalar in `[1,capacity]`, initially unset. The caller explicitly
uploads host `n` into a stable device parameter via `set_n`; kernels read it from
device memory, not a compiled constant or host-generated ID list. Any newly
completed pool must **already be committed** in the pooled input. This operator
does not compute or mutate pools; fresh uploads can be supplied on later runs.
Only consumed query/key values must be finite; nonfinite input or arithmetic is
reported as an error, not silently substituted. Shape/type checks are native as
well as Python handle validation. No implicit casts or quantization decoding.

For `completed=floor(n/4)` and `tail=n%4`:

1. Score every completed block and, if needed, one dead-key block.
2. Each dot uses FP32 products followed by a left-to-right FP32 sum over 128
   lanes. FMA contraction is disabled. Apply ReLU independently to all four dots.
3. Add the four ReLU dots in head order, then repeat each block score four times
   and truncate to `n` tokens. The incomplete dead tail receives finite `+1e9`.
4. Select token width `min(n,2051)` on GPU, **not block top-k**.
5. Emit ascending int64 token IDs on GPU.

Policies are mandatory and explicit:

- `ggml_cpu`: the reference value-only libstdc++ `__heap_select`/`__adjust_heap`
  membership transcription. Strict `>` replacement, original-order scan; no
  lower-ID tie rule. The reference's chunk filter can omit only candidates that
  could not replace the current minimum; scanning all candidates in order has
  identical membership. This is not a portable claim for arbitrary STL vendors.
- `strata`: descending score, lower token ID on ties. The same heap algorithm
  uses a total comparator, then sorts the chosen IDs ascending.

Both execute on GPU, including ascending ID sorting. There is no host score/ID
roundtrip and no CPU fallback. Selection is intentionally a **single-thread GPU
reference heap**, not a performance-optimized selector.

## Implementation / API

- `mini_llm/native_cpp/qsa_selection.cu`: CUDA scoring, heap membership and ID sort.
- `mini_llm/native_cpp/qsa_selection.h`: internal device-pointer launch interface.
- `mini_llm/native_cpp/runtime.{h,cpp}`: shared engine ownership, workspace registry,
  same-engine uploaded input access, budget admission, stable storage, CUDA events,
  scalar status checks, explicit diagnostics, stale-handle rejection.
- `mini_llm/native/__init__.py`: `Engine.qsa_selection`, `QSASelection` binding;
  no NumPy/CuPy dependency in the new binding itself.
- `mini_llm/native_cpp/CMakeLists.txt`: CUDA language and `.cu` compilation;
  no fast math; `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.

Example, given three same-engine `UploadedTensor` objects:

```python
with engine.qsa_selection(capacity=16384) as workspace:
    workspace.set_n(2053)  # explicit host scalar -> device parameter
    milliseconds = workspace.run(query, pooled, dead, policy="ggml_cpu")
    # Results remain on GPU. Diagnostics only, with caller-owned storage:
    scores = np.empty(2053, dtype=np.float32)
    ids = np.empty(2051, dtype=np.int64)
    assert workspace.readback_for_test(scores, ids) == 2053
    workspace.reset()  # n=0, results invalid; same allocations
```

C ABI: `ml_qsa_create/destroy/set_n/reset/run/readback_for_test`; declarations and
pointer-capacity contracts are in `runtime.h`. Run validates handles and input
shapes, waits for uploads (retiring their mmap leases), enqueues kernels, checks
one device scalar status, and synchronizes before returning. It returns CUDA-event
milliseconds per iteration; `repeats>1` is explicitly benchmark mode, bounded at
10000. No scores or IDs are copied by run. Results are unavailable after creation,
reset, or `set_n`, until a successful run. Readback requires valid results and
adequate caller capacities; bounds errors copy no payload.

Native shared ownership retains the engine independently of its public handle.
Input shared ownership lasts through synchronized completion. Results remain
readable after public engine/input handles close. Destruction drains the stream,
then frees workspace storage; best-effort CUDA teardown follows existing runtime
conventions. Device selection is guarded/restored. Calls retain the runtime's
global serialization; this is not a concurrent serving API or CUDA graph API.

## Storage budget and stable buffers

For capacity `C`, `W=min(C,2051)`, one fixed device allocation contains:

- device `n`: 8 bytes;
- selected int64 IDs: `8*W` bytes;
- FP32 token scores: `4*C` bytes;
- nonfinite status: 4 bytes.

Device bytes: **`12 + 8*W + 4*C`**. One pinned allocation contains the host n scalar
and host diagnostic status: **12 bytes**. Total additional budget is therefore
**`24 + 8*W + 4*C`**. These allocations compete with sessions and uploads under
`Engine.admit/record`; capacity failures do not change live/peak counters. Uploads
retain their existing exact-byte accounting. No per-run device or pinned payload
allocation occurs. Update/reset/run reuse the same slab and pinned storage.

At C=16384: workspace device=81956, pinned=12 bytes. Benchmark inputs add 2099712
device bytes, yielding **2181668 device + 12 pinned bytes**, as captured in JSON.
Accounting excludes CUDA context/stream/event overhead, allocator rounding,
CUDA-internal staging, host bookkeeping, mmap/page cache and caller readback
buffers, consistent with the existing runtime budget. It is not total VRAM/RSS.

## Validation — observed results

Environment: NVIDIA GeForce RTX 5080, device 0; driver 580.178.04; nvcc
13.4.92; NumPy 2.5.3. Release build targeted CUDA architecture 120. Tests used
`MINI_LLM_BACKEND=numpy`; all native computation still ran on CUDA. No checkpoint
or CuPy installation was needed for synthetic input upload/scoring/selection.

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build /tmp/mini-llm-wp1-build
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MINI_LLM_QSA_NATIVE_SELECT=0 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest -q tests/test_native_engine_*.py
```

- Initial targeted execution: **60 passed in 0.73s**.
- Full native engine execution: **170 passed in 1.99s**, including the previous
  110 tests. No verification failures or corrective edits were needed.
- Build completed successfully. Existing misleading-indentation warnings in
  unchanged `gguf_loader.cpp` were not modified.
- `git diff --check` passed.

`tests/test_native_engine_qsa.py` covers both policies at
`n=1,2,3,4,5,127,1024,2048,2051,2052,2053,4096,8192,16384`, all-flat tie cases,
seeded controlled integer-valued random inputs, incomplete tails, exact scores
and exact selected membership against the existing reference selector. Controlled
integer products/sums are exactly representable in FP32; distinct scores have
integer separation rather than numerical near-ties. Gaussian random FP32 tests
at n=16383 separately exercise differing dot-reduction order and exact membership
against both GPU diagnostic scores and reference scores for the fixed fixture.

### Scoring tolerance rationale

The source's NumPy/BLAS `iq @ bk` can use a different reduction tree than the new
explicit left-to-right kernel. The four-head sum order is preserved, but bitwise
BLAS dot equality is not claimed. Gaussian tests use `rtol=3e-5, atol=3e-5` and an
additional input-dependent forward-error bound on non-tail scores:

`2 * gamma_128 * sum_h(sum_j(abs(q[h,j]*k[b,j]))) + 1e-5`,
where `gamma_128=128*eps_float32/(1-128*eps_float32)`.

The factor two allows the two FP32 reduction implementations. Tail addition is
checked by the full-vector allclose; its FP32 rounding at `1e9` is not a small-dot
error. No production code or tolerance was adjusted in response to a failure.
Selection comparisons are **exact**, not allclose. The fixed random fixture
agrees across reduction orders, but arbitrarily near-cutoff blocks may exchange
membership if FP32 dot roundoff changes ordering; this limitation is explicit.

Lifecycle/invalid tests cover dynamic n growth and shrinkage, crossing a completed
pool boundary, reset invalidation, unchanged allocation counters through updates,
exact budget admission/rejection and competition with session storage, result
lifetime after public input/engine close, invalid capacity/n/repeats/policy,
wrong rank/dimensions/F16, NaN/Inf, cross-engine uploads, wrong-kind/zero/stale
handles, required output pointers, insufficient readback capacities, and Python
buffer validation. Stability is exercised through constant accounting and the
implementation's single creation-only allocation; no raw device addresses are
exported for tests.

## Reproducible microbenchmark — observed, not a speedup claim

`benchmarks/benchmark_native_qsa_selection.py` writes a temporary synthetic F32
GGUF, uses exact uploaded input arrays, seed 20261123, capacity16384, three warmup
runs per case, seven samples of 100 repeated launches per policy/n. CUDA events
bracket status clear + index scores + heap selection + ascending-ID sort on the
engine stream. Events exclude input/n uploads, scalar status D2H, score/ID
readback, and host synchronization after the end event. Event intervals can
include stream idle gaps between host-enqueued launches; these are not isolated
kernel-cycle measurements or end-to-end API wall latency.

```sh
MINI_LLM_BACKEND=numpy \
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m benchmarks.benchmark_native_qsa_selection \
  --device 0 --repeats 100 --samples 7 \
  | tee docs/native_engine/WP3_SELECTION_BENCHMARK.json
```

Captured separately after the full tests completed, with no concurrent test run.
The GPU was also a desktop display device; no exclusive-device claim is made.
All raw samples are in `WP3_SELECTION_BENCHMARK.json`.

| n | ggml_cpu median ms | strata median ms |
|---:|---:|---:|
| 1 | 0.028690 | 0.028692 |
| 2051 | 1.773947 | 1.773685 |
| 2053 | 1.774029 | 1.776274 |
| 4096 | 3.063101 | 3.051897 |
| 8192 | 4.594069 | 4.564603 |
| 16384 | 6.481000 | 6.438637 |

These are reference implementation latencies, **not optimized or model decode
latencies**. No matched CPU/production baseline was timed and no speedup is
claimed. The single-thread heap/sort deliberately prioritizes source membership
parity over throughput. Scoring also uses one thread per block-score and launches
for capacity rather than host-computed active size, preserving device-n control.

## Limitations / next bounded work

- No index projection, RMSNorm/RoPE, pool creation/commit, raw-key history, KV cache,
  selected-key attention, GQA, gating/output projection, or sequence/session
  integration. Input tensors are already prepared; this is not full WP3 completion.
- No production Qwen source changes, checkpoint transformations or approximations.
- Synchronous run/status checking, globally serialized ABI, static capacity,
  immutable uploaded inputs; no graph capture or fully device-driven sequence
  advancement. n starts as an explicit host upload.
- Correct reference heap, but slow. A next isolated task can replace only the
  selection implementation with a parallel algorithm while retaining both exact
  membership contracts and this test suite; performance is not established here.
- Full-attention integration needs an internal consumer of these device IDs and
  runtime sequence ownership. Preserve this selector's verification as a separate
  boundary while adding that path; do not infer model parity from these tests.
- Multi-GPU QSA selection and randomized fuzzing beyond these fixtures are not
  claimed. Existing upload/device-restoration tests remain part of the 170 passes.
