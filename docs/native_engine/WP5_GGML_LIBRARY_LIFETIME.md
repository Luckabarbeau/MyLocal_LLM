# WP5: native GGML process-lifetime library residency

## Defect and scope — 2026-11-24

`CPUExpertPool` owned two ordinary `RTLD_NOW | RTLD_LOCAL` loader references,
for the explicit bridge and GGML paths. Closing the final native pool/job owner
balanced both references with `dlclose`, allowing GGML and its dependencies to
unmap. The selected external GGML build installs a process-global C++ terminate
handler. Unloading does not restore that handler; a subsequent load can encounter
the old handler address in its static initializer and abort. The previously
recorded signature is `GGML_ASSERT(prev != ggml_uncaught_exception)` in
`_GLOBAL__sub_I_ggml.cpp`; see `WP8_GENERATION.md:186-190` for the original finding.
That document's module-local `CDLL` fixture was a test workaround, not a production
fix, and did not protect transitions between test modules.

This change is entirely in native loader policy, not CPU worker arithmetic,
quantization, routing, sampling, events, or scheduling. MoE and decoder owners
reuse the existing `CPUExpertPool` and therefore receive the same loader policy.

## Contract and deliberate cleanup exception

`CPUSharedLibrary` has an optional private `process_resident` argument, false by
default. Only `CPUExpertPool`'s GGML member passes true:

- Bridge: `RTLD_NOW | RTLD_LOCAL`, unchanged.
- Explicit GGML: `RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE`.
- Both owned loader references still receive `dlclose` in the existing RAII
  destructor, after pool destruction. Partial setup uses the same RAII path.
- GGML's mapping/static state remains resident until process exit, so closing the
  last native owner cannot cause unsafe reinitialization on the next load.
  Loader-required dependencies may remain resident with it; in this external
  split GGML build the terminate-handler initializer belongs to that dependency
  graph. No explicit residency flag is added to the bridge or native runtime.
- Pool workers, jobs, registered source leases/weight mmaps, engine ownership,
  and stale-handle invalidation retain their existing cleanup rules. Residency
  is **not** a hidden pool/engine/weight retainer.
- Explicit paths and ABI/symbol checks are unchanged; there is no library search,
  replacement library, arithmetic fallback, or retry. If `RTLD_NODELETE` is not
  available at compile time, requesting residency fails with
  `GGML process residency requires RTLD_NODELETE` rather than silently reverting
  to the unsafe policy. This is the existing Linux `dlopen`/`mmap` runtime.

This retains library mappings for **each distinct GGML library loaded**, including
one that loads successfully but later fails symbol/traits validation. It is not
an unload/reload hot-swap facility. Replacing a loaded GGML build requires a new
process. Library/static allocations are not engine device/pinned payload budget.

## Explicit provenance

These are observed test inputs, not defaults or proof of arbitrary GGML-fork ABI
compatibility. No source revision is inferred solely from a shared-library hash.

- GGML selected path:
  `/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so`
- Selected library SHA256:
  `924ac13f3e949cb1b62eddff20221600a6efcb1a2b99eb4abd390e75c23df839`
  (unchanged from `WP5_CPU_POOL.md`).
- Rebuilt native library: `/tmp/mini-llm-wp1-build/libmini_llm_native.so`
- Rebuilt native SHA256:
  `a913957cf7513f9e9d353bc65c8d88f886365c65fec36092337730352c4463df`
- Bridge path is supplied explicitly using the existing `_load_bridge()._name`;
  no bridge implementation or CPU worker arithmetic was modified.

## Cheapest before/after proof

Every shell invocation used the GPU-1 UUID mask below, never GPU 0; reproducers
used `ulimit -c 0`. A fresh interpreter, without any GGML `CDLL` retainer, ran:

```python
import ctypes as C
import os
from mini_llm.native import Runtime
from mini_llm.ops.ggml_packed_cpu import _load_bridge
r = Runtime()
e = r.engine()
p = e.cpu_experts(d_model=256, d_ff=320, threads=1,
    bridge_library=_load_bridge()._name,
    ggml_library=os.environ['MINI_LLM_GGML_CPU_LIBRARY'])
print('native pool opened', flush=True)
p.close()
e.close()
print('all native owners closed; loading CDLL', flush=True)
lib = C.CDLL(os.environ['MINI_LLM_GGML_CPU_LIBRARY'])
print('CDLL reload survived', flush=True)
```

Executed as `timeout 30s .venv/bin/python -u -c '...'`, with the environment below:

- **Before:** shell exit **134**, abort at `CDLL` after closing pool/engine.
- **After:** exit **0**, all three messages printed, including
  `CDLL reload survived`.

The new regression was also run **before** the loader edit:

```sh
timeout 40s .venv/bin/python -m pytest -q tests/test_native_engine_cpu_experts.py -k process_residency -x --tb=short
```

Result: **1 failed, 10 deselected in 2.10 s**. The `ctypes` child's return code
was **-6 (SIGABRT)** after `native last owner closed; next=ctypes`. This is a
recorded red regression, not a green suite result.

## Regression and verified results

`test_ggml_process_residency_without_retainer` launches four independent fresh
interpreters with a 30-second deadline each. Children inherit `os.environ.copy()`
(including the GPU mask), disable core dumps with `RLIMIT_CORE`, and first create
and close a native engine/pool **before loading any GGML oracle/CDLL**. They then
exercise repeated same-process transitions to:

1. `ctypes.CDLL`, explicit `dlclose`, and a native engine/pool again, three times;
2. a new native engine/pool, three times;
3. the existing Python `GGMLPackedCPURowPool`, three times, interleaved with native
   engine/pool cycles;
4. native MoE owners, twice, interleaved with native engine/pool cycles.

Children check stale native pool handles, GGML mapping residency, and, in the MoE
case, retirement of the synthetic weight file mapping after all owners close.
No generation-module fixture or parent-process loader reference can protect these
fresh interpreters. The generation fixture remains unchanged; it is no longer
needed as the production lifetime mechanism.

All commands used:

```sh
export CUDA_VISIBLE_DEVICES=GPU-e961516f-d77d-813c-98e9-2f5403f71631
export MINI_LLM_BACKEND=cupy
export MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so
export MINI_LLM_GGML_CPU_LIBRARY=/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
ulimit -c 0
```

| Exact command after environment above | Observed result |
| --- | --- |
| `timeout 120s cmake --build /tmp/mini-llm-wp1-build -j 2` | PASS, rebuilt and linked native library |
| `timeout 120s .venv/bin/python -m pytest -q tests/test_native_engine_cpu_experts.py tests/test_native_engine_runtime.py tests/test_native_engine_payload.py --durations=5` | **46 passed in 2.82 s**, no deselections |
| `timeout 120s .venv/bin/python -m pytest -q tests/test_native_engine_sampler.py tests/test_native_engine_sampler_optimization.py tests/test_native_engine_generation.py tests/test_native_engine_divergence.py tests/test_native_engine_decoder.py --deselect=tests/test_native_engine_decoder.py::test_synthetic_all48_reference_replay_lifetime --deselect=tests/test_native_engine_decoder.py::test_actual_all48_two_steps_reset --durations=8` | **87 passed, 2 deselected in 27.19 s** |

The focused result includes all four subprocess regressions, exact packed CPU
rows, busy/pending ownership, stale handles, partial-setup/error behavior, payload
mapping cleanup, and process teardown. The combined command includes the sampler
benchmark (not excluded), sampler optimization snapshots, generation replay and
event/control/lifetime checks, divergence diagnostics, and the remaining decoder
tests. It completed without the former cross-module unload/reload abort.

**Not a full-suite green claim:** exactly the two named strict decoder parity
tests were deselected as requested, not fixed or reclassified here. No sanitizer,
full-model quality/performance, exhaustive concurrency, or arbitrary external
GGML build compatibility claim follows from these runs. No bounded command timed
out. No known failure was hidden or counted as passing.
