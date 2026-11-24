# WP5 integrated experimental native MoE owner

## Scope and explicit arithmetic

Implemented in `mini_llm/native_cpp/moe_runtime.inc`, privately included by
`runtime.cpp`, with C declarations in `runtime.h`, thin ctypes owners in
`mini_llm/native/__init__.py`, and `moe.cu` linked by CMake.

This is a **single-token experimental complete MoE owner**, not a production
model replacement, adaptive cache, full decoder, or HC FFN chain. Production
Python model math, fallback paths, GGML worker arithmetic, and parent-owned
status/manifest files were not changed by this integration.

The only accepted policy string is:

`cpu_ggml_q8_gpu_reference_fp32`

- Nonresident routed experts use the existing registered CPU row pool, its
  GGML activation quantizers and integer dots (`ggml_q8`).
- Resident routed experts use the validated packed lane decoder, FP32 input,
  FP32 products/accumulators, and the existing Python IQ **warp** reduction graph
  (`reference_fp32`). No GPU Q8 activation approximation is substituted.
- Shared gate/up/down and the scalar gate use decoded FP32 arithmetic. Q8_0
  shared-down **weight storage** does not quantize activations.
- Moving an expert between CPU and GPU explicitly changes that expert's
  arithmetic. Pure-CPU, pure-GPU and mixed results must each be compared with
  the reference using the **same placement**, not with one another as though
  the CPU and GPU arithmetic were interchangeable.

CMake retains `--fmad=false`, precise divide/sqrt and no fast-math. The canonical
`moe_formats.cuh` decoder/codebooks and notices are unchanged. No new dependency
or scheduler was introduced.

## Setup API and descriptors

`Engine.moe(router, shared_gate, shared_up, shared_down, scalar_gate,
 gate_bank, up_bank, down_bank, *, d_model, d_ff, gpu_experts,
 bridge_library, ggml_library, threads=4, arithmetic=...)` creates an owner.

The first five arguments are existing uploaded tensor handles:

| Argument | Stored GGML shape | Allowed storage |
|---|---|---|
| router | `[D,512]` | F32/F16/BF16 |
| shared gate/up | `[D,SF]` | Q4_K/Q5_K/Q6_K/IQ4_XS |
| shared down | `[SF,D]` | IQ4_NL/Q8_0 |
| separate scalar gate | `[D]` | F32/F16/BF16 |

The three source arguments are existing mmap payload **leases**, not whole-bank
GPU uploads: gate/up `[D,F,512]`, down `[F,D,512]` in GGML order. The native owner
validates complete bank dimensions, format, tensor size, and matrix boundaries.
Each source matrix descriptor consists of its retained mapping, tensor data
base, expert-relative byte offset, row stride, and format. No expert bank is
materialized as floating weights. Every checkpoint bank remains packed.

Routed GU support IQ3_XXS/IQ2_S/IQ3_S/IQ4_XS; down supports IQ4_NL/Q2_0. All seven
checkpoint triples listed in `WP5_GPU_PRIMITIVES.md` are covered. Dimensions are
D=256..6144, divisible by 256, F/SF=256..12288, divisible by 64. The isolated
launcher previously imposed F%256; its row kernels now accept F%64, admitting
actual **D=2560, F=640, E=512, K=10** without changing projection equations.
The shared width SF is independently inferred/validated from uploaded weights.

Actual router/scalar checkpoint controls are BF16. Router projection uses the
existing native F32/F16/BF16 projection helper; the scalar uses the existing
lossless row decoder into a bounded FP32 `[D]` activation-sized buffer. No host
conversion or deployment-specific preprocessing is required.

`gpu_experts` is an explicit unique ID list in `[0,512)`, frozen at creation.
Only those experts receive GPU packed gate/up/down allocations. The owner
uploads a fixed 512-entry descriptor table plus placement flags; nonresident
entries have no device payload. Every other expert is registered once with a
private existing CPU row pool. New route IDs never trigger promotion, eviction,
allocation, omitted-expert dropping, or an unsupported-mode fallback.

An empty placement is supported. “Pure GPU” in the tests means all ten active
experts are resident, not that all 512 bank entries were unnecessarily uploaded.
A fully resident 512-entry placement is admitted only if the configured budget
allows it; budget-exhaustion rollback is tested.

Library configuration is explicit, as in `WP5_CPU_POOL.md`. For example, at setup
only, callers can obtain `bridge_library` from
`mini_llm.ops.ggml_packed_cpu._load_bridge()._name` and select `ggml_library` using
`os.environ['MINI_LLM_GGML_CPU_LIBRARY']`. There is no hardcoded deployment path,
library search, implicit compilation during per-expert dispatch, or silent
alternative arithmetic. If the placement has a CPU complement, library
resolution and GGML ABI/trait checks happen at owner creation, even if the first
route happens to hit only GPU experts.

## Native execution and graph

C ABI: `ml_moe_create`, `ml_moe_submit`, `ml_moe_reset`, `ml_moe_destroy`,
`ml_moe_job_wait`, `ml_moe_job_readback_for_test`, `ml_moe_job_destroy`.

`owner.submit(x, scale=...)` snapshots a contiguous FP32 `[D]` or `[1,D]` input
and returns a native job. All per-expert selection/dispatch is C++, not Python:

1. Upload the native-owned activation snapshot and clear per-epoch status.
2. Project router logits on the engine's native stream.
3. Run full-512 softmax; choose top10 logits, lower ID on ties; normalize selected
   full-softmax mass by `max(mass, 2^-14)`; apply routed scale **once**; sort IDs
   and associated weights into ascending ID order.
4. Copy 10 IDs, 10 diagnostic scaled weights, and status (84 bytes) to native
   host buffers and synchronize this **routing-only fence**.
5. Compact nonresident IDs to registered pool indices and route-slot positions;
   submit their unweighted FP32 CPU rows using the immutable shared input.
6. Enqueue the separate shared branch and resident GU/down work on the native
   stream while CPU workers can run. Resident kernels skip only explicit CPU
   positions; their old isolated entry point still rejects missing descriptors.
7. Only at job wait/readback/close, drain the CPU batch, scatter unweighted CPU
   rows into their original ten route slots, and stage the bounded row slab.
8. Multiply CPU rows by the already-scaled route weights exactly once. Add all
   ten CPU/GPU weighted terms sequentially in ascending ID order, with separate
   round-to-nearest multiply/add operations. Add the independent
   `sigmoid(scalar_gate*x) * shared_down((shared_gate*x * sigmoid(shared_gate*x))
   * shared_up*x)` output. Shared output is not routed-scaled.
9. Observe completion/status and copy a detached final result snapshot. Failed
   device results are never consumable.

The added internal `ml_moe_resident_launch` and `ml_moe_finish_launch` are
additive seams around the established kernels. The original isolated route,
expert/shared and combine APIs remain available. No CPU join occurs before
shared/resident work is enqueued. Launch overlap is an execution-order property,
not a throughput claim.

Current input is a host FP32 activation snapshot and explicit test readback
copies the final result to host. Host slabs are native-owned pageable memory;
no new pinned allocations are made. CUDA pageable-copy blocking behavior can
limit overlap. Device scratch is O(10*(D+F) + SF + 512), plus explicitly selected
packed payloads and retained uploaded controls/shared weights. Engine statistics
charge every native device allocation and roll it back on setup failure; CPU
pool/vector/mmap host storage is not device payload.

## Ownership, errors and reset

- The owner retains all five uploaded weights, all three bank leases, GPU slots,
  and its engine. The private CPU pool retains source mappings and both explicit
  `dlopen` library handles until after workers are destroyed.
- Jobs retain the complete owner and CPU job. Closing public leases, mappings,
  uploaded tensors or the engine cannot invalidate an existing job.
- One outstanding job per owner is allowed until completion. A second submission
  and reset while busy are rejected. Calls use the established serialized ABI
  boundary lock.
- Waiting snapshots the output independently of the reusable workspace. A later
  submission/reset does not change an older completed job's result or route.
- Reset is an idle-only operation: this single-token operator has no recurrent
  position/cache state. Each submit clears status; invalid submissions do not
  create a job. Handle outputs are zeroed before validation.
- Job close and owner close drain both branches. Closing discards failed results
  but still releases their handles; explicit wait/readback reports the failure.
  Owner close drains pending jobs but leaves already-public job snapshots usable.
- Nonfinite input, invalid scale/policy/shape/format/placement, wrong-kind or stale
  handles, busy use, inadequate diagnostic buffers, and exhausted budget are
  rejected. Partial launch/setup failure drains before releasing DMA inputs,
  mapping ownership or slots. Nonresident experts are never silently omitted.
- Source files must remain immutable while retained. This owner does not make
  external mutation of mmap files safe.

## Validation (2026-11-23, RTX 5080)

Build:

```sh
cmake -S mini_llm/native_cpp -B /tmp/mini-llm-wp1-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /tmp/mini-llm-wp1-build -j 4
```

Completed successfully. Existing `gguf_loader.cpp` misleading-indentation
warnings remain unchanged.

For reproduction, select the explicit GGML library as documented in
`WP5_CPU_POOL.md` and set `MINI_LLM_NATIVE_LIBRARY` to the built `.so`.
All runs below also set `PYTHONDONTWRITEBYTECODE=1`, `NVIDIA_TF32_OVERRIDE=0`,
`OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

- Integrated suite: `MINI_LLM_BACKEND=cupy MINI_LLM_MOE_CHECKPOINT=1`,
  `.venv/bin/python -m pytest -q -s -x -p no:cacheprovider tests/test_native_engine_moe.py`.
  Latest execution under `/usr/local/cuda/bin/compute-sanitizer --tool memcheck
  --error-exitcode 99`: **41 passed in 7.66 s; ERROR SUMMARY: 0 errors**.
- Full established suite: `MINI_LLM_BACKEND=numpy MINI_LLM_TEST_PLE_CUDA=1
  MINI_LLM_QSA_LAYER_CHECKPOINT=1`, `.venv/bin/python -m pytest -q
  -p no:cacheprovider tests/test_native_engine*.py`: **505 passed, 44 skipped in
  68.13 s**. Skips are the 41 new CuPy oracle cases (verified separately above)
  plus the original three optional skips. This preserves all 505 parent passes;
  it is not a claim that the new tests ran in the NumPy process.
- Isolated GPU primitive regression: `MINI_LLM_BACKEND=cupy .venv/bin/python
  tests/native_moe_primitives_harness.py --quick --report
  /tmp/wp5_moe_integrated_primitives.json`: **passed**, all seven triples,
  315 comparisons. Original decoder/product exactness and warp term/sum
  bitwise comparisons remain intact.
- `git diff --check`: passed (the tree already contained parent changes).

The 41 new tests cover all seven routed triples in CPU-only/GPU-only/mixed
modes; empty placement; all eight shared GU/down combinations; full router ties;
route changes into the nonresident complement without new allocation; busy jobs;
source/upload/engine close with a pending job; eventual mmap release; invalid
configuration/handles and diagnostic buffers; result snapshot reuse; reset;
nonfinite device-output invalidation and failed-job release; and allocation
budget rollback. They forbid subsequent Python `GGMLPackedCPUExecutor.expert`
dispatch in native execution after constructing the oracle.

### Actual checkpoint comparison

The inventory-selected checkpoint is opened read-only, one selected layer at a
time. Layers **0, 1, 47** use actual D=2560/F=640/E=512/K=10 banks and controls,
covering Q2_0 routed down and layer-47 Q8_0 shared down. Input is a seeded FP32
row per layer; scale is 0.37. The unchanged `Qwen4ExpMoE`, `RouterInference`,
`ExpertsInference` and `HeterogeneousExpertExecutor` form the oracle. Its setup
freezes `_prepare` to the exact same explicit placement and uses `ggml_pool`,
FP32 warp grouped GPU experts, and existing packed shared projections. No
production Python source is modified. Oracle-only expert copies are bounded to
selected IDs, not full banks; no full model is instantiated.

| Layer | Active placement | Native device bytes | Max output absolute error | Relative L2 |
|---|---|---:|---:|---:|
| 0 | CPU / empty | 5,682,776 | 1.86265e-9 | 2.39561e-7 |
| 1 | CPU / empty | 5,682,776 | 3.49246e-10 | 2.52128e-7 |
| 47 | CPU / empty | 6,757,976 | 2.61934e-10 | 2.07235e-7 |
| 0 | GPU / 10 selected | 27,442,776 | 1.86265e-9 | 2.40086e-7 |
| 1 | GPU / 10 selected | 20,786,776 | 3.49246e-10 | 2.50580e-7 |
| 47 | GPU / 10 selected | 33,381,976 | 2.91038e-10 | 2.06721e-7 |
| 0 | mixed / 5 selected | 16,562,776 | 1.86265e-9 | 2.37871e-7 |
| 1 | mixed / 5 selected | 13,234,776 | 3.49246e-10 | 2.51105e-7 |
| 47 | mixed / 5 selected | 20,069,976 | 2.61934e-10 | 2.09600e-7 |

Mixed placements were `[7,64,254,347,481]`, `[51,215,249,358,462]`, and
`[54,167,197,267,501]` respectively. Exact sorted route IDs and scaled weights
are checked and printed by the test. Comparison uses the existing primitive
relative/scaled bounds: relative L2 <1e-5 or absolute <2e-6, with max absolute
<2e-6 + 2e-5*max(abs(reference)); this is not a claim of bitwise router/shared
reduction parity. All calculations/comparisons here are FP32, not finite
differences; no tolerance was loosened during integrated validation.

Every actual-layer test enforces native peak <512 MiB and native payload plus
retained CuPy default memory-pool allocation <512 MiB. Observed maxima are
33,381,976 native bytes (31.84 MiB) and 49,618,944 oracle-pool bytes; their sum is
<80 MiB. This accounting excludes driver context, loaded kernel modules/constant
codebooks and external allocator overhead, and is not total physical VRAM/RSS.

The initial checkpoint test exposed BF16 controls incorrectly assumed F32 in the
new setup API/test reader. The localized fix reused existing lossless native
helpers and BF16 bit expansion in the oracle reader. A test callback also needed
the existing packed-linear 2D input contract; an edit indentation artifact was
removed locally. Final tests above supersede those failures; no worker,
projection or reference mathematical equation was changed to pass them.

## Remaining milestone / performance

**Direct native HC FFN read -> MoE -> HC write is deferred.** This owner currently
accepts a host activation and snapshots host results. The next integration must
use HC's internal native read-buffer ownership, perform only the necessary CPU
activation D2H staging after routing, and feed MoE's device result directly to HC
write with explicit failure/transaction semantics. No claim is made that calling
these operations separately through diagnostic host buffers is native chaining.

No whole-MoE latency benchmark was added or inferred from primitive timings.
A valid future measurement must time submit **through final job completion**,
including router projection/softmax, routing fence, CPU work, shared/resident
work, CPU row transfer, ordered combine and required output completion. Device
kernel time alone, or submit time excluding wait, is not whole-MoE latency.
Graph capture, concurrent requests, adaptive placement, multi-token prefill,
complete HC chaining and end-to-end generation remain outside this milestone.
Parent status/manifest ownership remains exclusive to the parent.
