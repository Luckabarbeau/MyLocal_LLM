# Patches 0269–0273 — CPU scheduling and captured GPU segments

## Provenance and shared validation scope

Strata source `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`; GGML ABI/reference `171e8846b4af9766c354064cb776cb34a50f053f`. No checkpoint or pinned reference mathematics changed. CPU experts retain per-format Q8_K/Q8_0; GPU experts retain packed weights and FP32 activations. These are distinct arithmetic paths, not model-wide Strata Q8_1 equivalence.

All features below remain independently selectable. Newer final-tree regressions: **1939 CPU passed / 1016 skipped**, **1202 CuPy passed / 18 skipped**. Commands and results: `0279_{cpu,gpu}_regression_v1.xml`; source hashes and per-run measurements: `0268_0279_performance_v2.json`. These cover Qwen4Exp and Qwen3-Next inference, not the entire training repository. Historical focused counts below have the narrower scope indicated.

## 0269 — Participating host and optional affinity

- **Objective/files:** native `ggml_packed_cpu.{py,cpp}`, worker-capacity snapshot in `external/qwen4exp_weights.py`, workload runner, `tests/test_0269_host_pool_participation.py` and model tests.
- **Contract:** `host_help=True` / `MINI_LLM_GGML_POOL_HOST_HELP=1` lets the waiting thread claim row tasks. `cpu_ids` / `MINI_LLM_GGML_POOL_CPUS` binds persistent workers; `MINI_LLM_HOST_CPU` optionally pins the submitting thread after construction. Outstanding host tasks protect phase advancement. Default host assistance is off.
- Worker capacity is snapshotted before optional host pinning; replanning must not shrink a previously approved team to one worker. A later broad regression corrected the synthetic `__new__` planner fixture to provide that constructor-owned field.
- **Tests/numerics:** historical native group 76 passed / 4 skipped; model/workload 34 passed / 1 skipped. Row arithmetic unchanged. Longer responses have identical 512 output IDs across the ablations.
- **Performance:** unpinned host-help **17.72739/s**; exact Strata affinity plus help **13.48530/s**; pin-only **13.84512/s**. Do not enable affinity by default. Unpinned decode used ~10.55 active-core equivalents.
- **Memory/limits:** persistent worker scratch only; no additional packed copy. No isolated worker-memory peak measured. `diagnostics_0269_strata_cpu_v1` is the preserved failed planning attempt, not a throughput result.
- **Next dependency:** reduce launch/synchronization overhead and rendezvous tails, not blindly increase affinity constraints.

## 0270 — Pre-MoE recurrent graph

- **Objective/files:** `external/qwen4exp_pre_moe_graph.py`, model/router integration, `ops/inference_graph.py`, `tests/test_0270_pre_moe_graph.py`.
- **Contract:** `pre_moe_graph=True` captures attention HC read → recurrent mixer → residual write → FFN HC read → router. PLE/QSA remain outside. Tuple outputs are internal borrowed buffers until the next serialized replay. Persistent recurrence/convolution arrays are copied in/out; positions advance in runtime state. Standalone GDN capture is mutually exclusive.
- **Tests/numerics:** focused 29 passed / 1 skipped. `0270_pre_graph_parity_v1.json`: two requests, fixed prefixes 1/2/4/8/16, bit-exact logits and all active-state hashes against the same-GEMV uncaptured path.
- **Memory:** 16 MiB cap per recurrent graph; observed ~128 MB total graph allocation, not a full VRAM peak.
- **Performance:** **17.88268/s**, 512 output cap, no demonstrated material gain. Host profile still showed native completion waits, repeated descriptor preparation, and QSA work; cumulative profiles overlap and are not exclusive costs.
- **Next dependency:** remove repeated metadata construction and worker phase tails.

## 0271 — Registered native descriptors

- **Objective/files:** `ops/ggml_packed_cpu.{py,cpp}`, `device_expert_plan.py`, `tests/test_0271_registered_cpu_descriptors.py`.
- **Contract:** `pool.register(key, weights)` validates immutable projection descriptors once and retains canonical sources; internal `_submit_registered` consumes bounded int32 IDs and distinct owned input/output slabs. Maximum 32768 descriptors, approximately 4 MiB metadata, no packed payload copy.
- **Tests/numerics:** focused CPU 49 passed / 2 skipped, device plan 6 passed. Same native row arithmetic; output IDs unchanged in the long run.
- **Performance:** preparation improved in the isolated test but longer throughput regressed to **8.49598/s**. Both registered and ordinary paths exhibited large completion tails. Keep this result; do not advertise preparation speed as model speed.
- **Memory/limits:** metadata bound stated above; no separate peak measured. Runtime descriptor/source ownership remains until pool close.
- **Next dependency:** task-completion scheduling, below.

## 0272 — Generation-tagged task completion

- **Objective/files:** native pool and `tests/test_0272_task_completion_pool.py`.
- **Contract:** opt-in `MINI_LLM_GGML_POOL_SCHEDULER=tasks`, default `barrier`. Advance when all phase tasks complete, rather than waiting for every worker to rendezvous. Generation-tagged atomic claims, an exhausted sentinel, counted live writers and an explicit generation-overflow guard protect descriptor/scratch reuse. Adaptive chunks target roughly three row tasks per participant.
- **Tests/numerics:** native group 62 passed / 2 skipped, including 2400 varying mixed-format epochs. Controlled ordinary/registered output arithmetic retained.
- **Performance:** synthetic median ~0.15–0.18 ms; barrier p95 ~1.76–1.85 ms versus tasks ~0.19–0.20 ms. Long **18.96007/s**, identical 512-token trajectory to 0270; ~10.71 active-core equivalents. One sequential comparison, not confidence bounds.
- **Memory/limits:** fixed pool scratch and descriptor registry unchanged; no additional weight copy.
- **Next dependency:** shared/resident GPU overlap and the remaining Python handoff.

## 0273 — Shared-expert graph

- **Objective/files:** `external/qwen4exp_model.py`, workload/parity runners, `tests/test_0273_shared_expert_graph.py`.
- **Contract:** `shared_graph_decode=True` captures gate/up/SwiGLU/down and shared gating score. Existing Triton FP32 GEMV avoids capture-incompatible cuBLAS; graph output lifetime and dedicated-stream producer/completion events are explicit. Default off.
- **Tests/numerics:** focused 54 passed / 1 skipped. `0273_shared_graph_parity_v1.json` shows bit-exact same-arithmetic logits and state hashes at prefixes 1/2/4/8/16 over two requests.
- **Memory:** 2 MiB cap/layer; observed ~1.97 MB total.
- **Performance:** **18.88463/s**, 413 output tokens, a different trajectory from 0272. No demonstrated gain or identical-input ratio against 0272.
- **Next dependency:** queue CPU inputs before overlapping shared GPU work; patches 0274 onward are documented separately.
