# PATCH 0089U — Streaming historical block backward prototype

> Historical prototype record. The shared-memory failures documented below were resolved by smaller tiles, and the integrated measured result is documented in `docs/0089V_STREAMING_HISTORICAL_BLOCK_BACKWARD.md`.

## Prototype decision at the time

The deterministic historical block backward is the next significant measured
optimization target. A non-dispatched Triton prototype has been added to
`mini_llm/ops/triton_historical_block.py`, but it is **not production-ready**:
its key-major `dK/dV` kernel exceeds the RTX 5080 shared-memory limit.

No backward dispatch or environment flag was added to `HistoricalReadMemory`.
The accepted forward-only configuration remains the active implementation:

```bash
export MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD=1
export MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM=1
```

## Current accepted-path profile

Artifacts:

```text
benchmarks/results/0089u_seed48_triton_block.nsys-rep
benchmarks/results/0089u_seed48_triton_block.sqlite
benchmarks/results/0089u_seed48_triton_block_analysis.json
benchmarks/results/0089u_seed48_triton_block_{ranges,phases,kernels,events}.csv
```

Fixture:

```text
B=3, T=4096, seed=48
history validity=[0, 6162, 61440]
accepted Triton deterministic forward enabled
all required production fairness flags fixed
```

| Range | GPU union | Kernels | Complete-capture share |
|---|---:|---:|---:|
| Complete captured F+B | 928.573 ms | — | 100% |
| All historical work | 351.576 ms | 7,197 | 37.86% |
| Historical backward | 250.662 ms | 5,165 | 26.99% |
| Deterministic block backward | **172.393 ms** | **1,968** | **18.57%** |
| Deterministic block forward | 38.531 ms | 12 | 4.15% |

The block backward is 4.47 times larger than the next-largest historical range.

## Backward composition

The 172.393 ms target is fully accounted:

| Category | Time | Block share | Launches |
|---|---:|---:|---:|
| GEMMs | 44.299 ms | 25.70% | 288 |
| Non-GEMM | **128.093 ms** | **74.30%** | **1,680** |
| Elementwise/math | 89.096 ms | 51.68% | 816 |
| Copy/indexing | 29.349 ms | 17.02% | 576 |
| Reductions | 9.648 ms | 5.60% | 288 |

Each of the 48 CuPy backward tiles launches 41 kernels. This confirms that the
opportunity is fusion and elimination of score/probability materialization, not
replacement of already-competitive GEMMs in isolation.

## Amdahl bounds

| Block-backward reduction | Saved time | Capture reduction | Maximum throughput gain |
|---:|---:|---:|---:|
| 40% | 68.957 ms | 7.43% | 8.02% |
| 50% | 86.196 ms | 9.28% | 10.23% |
| 60% | 103.436 ms | 11.14% | 12.54% |
| 70% | 120.675 ms | 13.00% | 14.94% |

A 40–50% target reduction is plausible if the custom implementation removes
most non-GEMM work. Sixty percent is aggressive; seventy percent approaches the
limit unless GEMM execution also improves.

## Prototype design

The prototype extends the accepted forward with an optional cache:

```text
FP32 output:  [B,Q,64]
FP32 LSE:     [B,Q], base-2
geometry:     first_local, selected_tokens, period
```

Planned deterministic backward launches per slot:

1. Delta and gate-gradient kernel.
2. Key-major additive `dK/dV` kernel.
3. Query-major additive `dQ` plus active-row context clearing.

Across 12 slots this would reduce launch count from 1,968 to approximately 36.
The retained forward cache costs about 36.6 MiB across all slots and avoids much
larger per-chunk score/probability intermediates.

The prototype is not imported or called by `historical_read.py`.

## Current failure

Direct backward compilation reports:

```text
triton.runtime.errors.OutOfResources
shared memory required: 123,392 bytes
hardware limit:          101,376 bytes
```

Reducing the key-major kernel from three pipeline stages to two did not change
the reported 123,392-byte requirement. No second speculative correction was
made.

The concrete next change is:

```text
key-major dK/dV only:
BLOCK_M 64 -> 32
BLOCK_N remains 64
```

This halves the query tile resident in the key-major traversal while preserving
64-key ownership and the existing 64x64 forward/query-major kernels. After that
single change, the required sequence is:

1. Compile and run a direct one-slot backward.
2. Compare dQ, additive dK/dV, gate gradient, and context clearing with CuPy.
3. Test all-invalid rows, active rows with no phase-selected keys, nonzero crops,
   non-contiguous Q views, and prefilled gradient destinations.
4. Only if local gradients pass, add an explicit backward opt-in dispatch.
5. Reprofile complete seed-48 F+B and require at least a 40% block-backward
   reduction for promotion.

## Preserved state and validation

The accepted forward was revalidated after adding the undispatched prototype:

```text
production-shape FP32 relative L2: 9.75e-7
short-run median kernel latency:    3.07 ms
```

Regression suite:

```text
30 passed, 1 skipped
```

`git diff --check` passed.
