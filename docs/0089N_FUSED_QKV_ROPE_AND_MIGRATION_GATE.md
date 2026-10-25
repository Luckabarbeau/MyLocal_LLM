# PATCH 0089N — Fused packed-QKV RoPE and PyTorch migration gate

## Executive decision

The fused packed-QKV forward path is accepted and promoted for CuPy `gpu-fast` and `consumer-gpu` profiles.

A production PyTorch migration is **not yet approved**. PyTorch has a verified fullgraph custom-operator mechanism and is faster on the ordinary Transformer stack, but the actual historical forward/backward, 61K-history memory fit, Hybrid-Muon offload, and exact checkpoint-resume behavior have not been demonstrated in Torch. The history-heavy workload is dominated by custom historical execution rather than the ordinary block, so the standard-block speedup alone is insufficient evidence for migration.

The next migration action should be a bounded Torch historical custom-operator prototype on the fixed seed-48 fixture, not a rewrite of the full trainer.

## Fused forward implementation

### Established graph

The production packed-QKV path previously executed:

```text
Packed QKV BF16 GEMM
Q strided view -> 4 BF16 multiplies + subtract + add
K strided view -> 4 BF16 multiplies + subtract + add
Strided V materialization into compact contiguous storage
```

Across 20 layers this produced 240 RoPE arithmetic kernels plus a multi-kernel strided V materialization.

The new `packed_qkv_rope_forward_bf16` CUDA kernel reads the packed BF16 projection directly and writes three independent contiguous outputs:

```text
RoPE(Q), RoPE(K), compact V
```

One thread processes one adjacent pair. The kernel supports both shared `[T]` and batch-specific `[B,T]` prepared-position factors.

### Numerical contract

The old CuPy graph materialized BF16 multiplication results before the final BF16 addition/subtraction. The fused kernel explicitly reproduces those intermediate BF16 roundings; it does not silently replace them with one FP32/FMA expression.

Fallback conditions include:

- non-CuPy backend;
- non-BF16 or non-contiguous packed projection;
- non-prepared positions;
- incompatible shapes;
- odd head dimensions;
- kernel compilation or launch failure when strict packed-QKV mode is disabled.

Explicit rollback:

```bash
MINI_LLM_FUSED_QKV_ROPE_FORWARD=0
```

Promoted profiles set it to `1` through `os.environ.setdefault`, so explicit overrides win.

## Correctness

### Direct kernel parity

The focused GPU test covers shared and batch-specific positions.

| Tensor | Result |
|---|---|
| RoPE Q | BF16 bitwise equal |
| RoPE K | BF16 bitwise equal |
| compact V | BF16 bitwise equal |
| Q/K cache cosine/sine ownership | identical prepared factors |

### Complete depth-2 F+B

Compared against the established 22-tensor native CuPy reference:

| Metric | Result |
|---|---:|
| Accepted | yes |
| Bitwise-identical tensors | 18 / 22 |
| Worst relative L2 | 9.24e-8 |
| Maximum absolute difference | 7.63e-6 |
| Non-bitwise tensors | four RMSNorm gamma reductions |

Outputs, input gradients, boundary tensors, Q/K/V/Wo gradients, and FFN gate/up/down gradients remain bitwise identical.

## Performance

### Depth-20 block stack

Two independent alternating baseline/candidate process pairs, five warmups and 30 samples per process:

| Run | Baseline GPU | Candidate GPU | Saved |
|---:|---:|---:|---:|
| 1 | 190.927 ms | 187.537 ms | 3.390 ms |
| 2 | 191.269 ms | 187.868 ms | 3.401 ms |
| Median of process medians | **191.098 ms** | **187.703 ms** | **3.395 ms (1.78%)** |

Host-wall median of process medians improved from `192.068` to `188.664 ms`, saving `3.404 ms` (`1.77%`).

### Nsight Systems mechanism

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| GPU execution union | 189.714 ms | 188.409 ms | -1.304 ms |
| GPU kernels | 1,260 | 940 | **-320** |
| All GPU events | 1,380 | 1,060 | -320 |
| QKV/RoPE category | 4.520 ms / 280 kernels | 3.055 ms / 60 kernels | -1.465 ms / -220 kernels |
| Memory/data movement | 1.833 ms / 100 kernels | 0.722 ms / 0 kernels | -1.111 ms / -100 kernels |
| Internal execution gaps | 2.703 ms | 1.915 ms | -0.787 ms |

The new kernel executes 20 times for `0.999 ms` total. The backward inverse-RoPE pack remains unchanged at approximately `1.63 ms`.

The profiler-union delta is smaller than the unprofiled delta because separate capture processes had GEMM and residual kernel drift. The launch/category changes directly establish the intended mechanism; the alternating unprofiled pairs establish complete-stack latency.

### Complete production model, history-free B3/T2304

The first two-warmup run was bimodal and rejected. A corrected candidate-first/baseline-second run used ten warmups and 30 samples:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| GPU median | 240.831 ms | 236.497 ms | **-4.334 ms (-1.80%)** |
| Host median | 240.928 ms | 236.751 ms | **-4.177 ms (-1.73%)** |
| Loss | 2.8587992191 | 2.8587992191 | exact |
| Fixture hashes | reference | identical | pass |

### History-heavy B3/T4096

A current deterministic document-first fixture was found without fabricated padding:

```text
seed: 48
history lengths: [0, 6162, 61440]
document spans: [1033, 11236, 1170271]
target valid lengths: [1032, 4096, 4096]
source: finepdfs/file_00009/train_shard_00004.bin
sampling attempts: 2
input SHA256: 8bed9e8e4902ccf10b0635d43d6f05856ab62e9e43ede41a027c36d5e434d16d
```

Matched complete numerical F+B:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| GPU median | 935.343 ms | 928.372 ms | **-6.971 ms (-0.75%)** |
| GPU p10/p90 | 933.227 / 937.680 ms | 926.082 / 930.815 ms | non-overlapping |
| Host median | 936.340 ms | 928.545 ms | **-7.796 ms (-0.83%)** |
| Loss | 2.0288021564 | 2.0288021564 | exact |
| Fixture hashes | reference | identical | pass |
| CuPy pool reservation high-water | 14,091,812,864 B | 14,091,812,864 B | unchanged |
| Device used at boundary | 15,702,425,600 B | 15,696,527,360 B | other processes included |

The candidate adds no cache tensors and does not increase peak pool reservation.

## Updated standard-stack comparison

A same-state comparison after promotion measured:

| Backend | Depth-20 GPU median | MAD | Peak/device memory gate |
|---|---:|---:|---:|
| Compiled Torch | 202.721 ms | 0.914 ms | 9.147 GB observed |
| Promoted CuPy | 221.224 ms | 0.896 ms | 8.981 GB observed |

Remaining standard-path gap:

```text
18.503 ms
9.13% relative to compiled Torch
CuPy requires another 8.36% reduction to match Torch
```

The absolute values differ from earlier sessions because GPU run state changed; this table uses adjacent same-session processes.

# PyTorch migration gate

## Gate matrix

| Gate | Evidence | Status |
|---|---|---|
| Ordinary 20-layer Transformer fullgraph | Existing depth-20 compiled benchmark; current Torch lead is 18.503 ms | **Pass** |
| LM head + CE compilation | Existing 0089D complete F+B comparison | **Pass** |
| Torch custom op inside `fullgraph=True` | Torch 2.14.1+cu130 probe registered fake and autograd formulas; CUDA forward/backward compiled and gradient was exact | **Pass, mechanism only** |
| Actual historical forward custom op | No Torch implementation of `HistoricalReadMemory.open/apply` | **Not demonstrated** |
| Actual historical backward/custom autograd | No Torch equivalent of `backward_reads/close/_encode_backward` | **Not demonstrated** |
| Ragged `[0, short, 61440]` history | Native seed-48 fixture runs correctly; no Torch result | **Native pass / Torch open** |
| Compact selected context | Native learned routing gathers top-8 blocks into compact selected K/V; no dense 64K attention mask | **Native pass / Torch open** |
| Document/EOS causality | Native historical tests pass invalid-history isolation, future-token invariance, and cross-document mask tests | **Native pass / Torch open** |
| Historical recomputation | Native recompute/no-recompute outputs and parameter gradients are tested; seed-48 uses recomputation | **Native pass / Torch open** |
| Gradient fan-in | Native historical state accumulates shared K/V, learned K/V, and router-key gradients before exactly one encoder backward | **Native pass / Torch open** |
| Model checkpoint import | 217 stable named NPY tensors; actual BF16 Wq imported to Torch by raw uint16 view and round-tripped bitwise | **Partial pass** |
| Optimizer checkpoint/resume | NPY groups `master_weights`, `m`, `v`, `muon_m` exist, but Torch Hybrid-Muon semantics are unimplemented | **Not demonstrated** |
| Full CPU optimizer offload | Native pinned master/momentum states and bounded 16 MB staging are implemented; no Torch equivalent | **Not demonstrated** |
| 16 GB heavy-history memory fit | Native pool high-water is 14.092 GB and total device use is ~15.70 GB with other processes | **Native pass, Torch high risk/open** |
| Heavy-history complete F+B performance | Promoted native result is 928.372 ms | **Native baseline established; Torch open** |
| Heavy-history real optimizer step | Existing real-training evidence does not cover the seed-48 61K fixture | **Open for both matched migration test** |
| Full checkpoint compatibility after optimizer step | Native roundtrip tests pass; no Torch-produced checkpoint has resumed native training | **Torch open** |

## Why direct `torch.compile` of the current historical code will not work

The existing historical implementation contains source-level graph boundaries that are correct for handwritten CuPy but incompatible with a direct fullgraph conversion:

- GPU-to-host `.item()` decisions for common history cropping and validity;
- mutable `HistoricalState` dataclasses and per-layer dictionaries;
- Python loops over scheduled slots, route chunks, and learned heads;
- side-effect gradient fan-in into shared state;
- CuPy-specific advanced indexing, `add.at`, RawKernels, and memory-pool behavior;
- recomputation controlled by mutable object/environment state.

These are not arguments against Torch custom operators. They mean the historical subsystem must be exposed through explicit operator schemas rather than translated operation-by-operation and left for Dynamo to discover.

## Custom-op feasibility result

The installed environment supports the required integration mechanism:

```text
PyTorch: 2.14.1+cu130
CUDA: 13.0
GPU: RTX 5080
custom op under torch.compile(fullgraph=True): pass
registered autograd backward: pass
gradient exactness: pass
```

This proves that an opaque custom historical operator can coexist with a compiled surrounding graph. It does not prove that the production historical operator is correct, fast, or memory-safe.

## Checkpoint assessment

The checkpoint contains:

```text
217 parameter files
676,268,688 parameter bytes
BF16 raw storage: 676,268,544 bytes
FP32 storage: 144 bytes
optimizer groups: master_weights, m, v, muon_m
```

BF16 NPY tensors use raw two-byte `|V2` storage. An actual `blocks.0.attention.Wq` tensor was imported with:

```text
NPY raw bytes -> uint16 view -> torch.Tensor -> torch.bfloat16 view
```

and round-tripped bitwise. Model-weight conversion is therefore technically straightforward, but a complete name/shape manifest and save-back test are still required. Optimizer state is not compatible with stock `torch.optim` because this project uses name-based Hybrid-Muon grouping, FP32 masters, pinned CPU offload, bounded staging, and project-specific checkpoint metadata.

## Memory assessment

Memory is the strongest migration risk.

Standard depth-20 stack measurements showed similar allocator reservations:

```text
CuPy pool peak reserved:   7.333 GB
Torch peak reserved:       7.554 GB
Torch peak allocated:      7.345 GB
```

This does not establish historical fit. The native 61K fixture already reserves `14.092 GB` in the CuPy pool. Only about `2.51 GB` remains relative to physical capacity before accounting for non-pool/device users, compilation caches, Torch allocator fragmentation, Inductor workspaces, and historical custom-op temporaries. At the measured boundary, total device use was approximately `15.70 GB`.

A Torch historical prototype must therefore preserve compact selected K/V and recomputation. It must not create a dense 64K mask, a padded per-head context, or simultaneous CuPy and Torch copies of the historical store.

## Performance/Amdahl assessment

The archived history-heavy profile measured:

```text
Complete GPU execution union:      833.015 ms
Historical range GPU union:        511.745 ms
Historical range elapsed union:    529.930 ms
GPU kernels:                       9,765
Candidate host starvation:         0.712 ms
```

Historical ranges overlap parent backward/gradient ranges and must not be added to them, but they establish that historical work occupies most of the critical execution window.

Using the current standard-path CuPy reduction-to-match requirement (`8.36%`) only as a rough bound:

```text
native heavy F+B:                   928.4 ms
rough non-historical remainder:     ~416.6 ms
optimistic standard-only saving:    ~34.8 ms
optimistic total heavy saving:      ~3.7%
```

This estimate combines different captures and is not a prediction. Its purpose is to show that a Torch migration which leaves historical performance unchanged has only low-single-digit upside on history-heavy batches. A historical Torch path that regresses by roughly 35 ms would erase the estimated standard-block benefit.

## Required migration prototype

Do not port the complete trainer first. Implement a bounded prototype with these requirements:

1. Freeze the seed-48 source IDs, targets, masks, history starts, router seed, and retrieved block identities.
2. Keep document-first/EOS semantics and the compact top-8-block representation.
3. Define coarse Torch custom-op schemas for historical open/encode, per-layer route/read, and backward/close rather than exposing thousands of small Python/CuPy operations.
4. Register fake/meta implementations and explicit autograd formulas so the surrounding 20-layer stack remains `fullgraph=True`.
5. Avoid CuPy↔Torch DLPack crossings inside each layer; use one allocator/framework for persistent tensors.
6. Preserve recomputation and FP32 shared-gradient fan-in.
7. Import all 217 checkpoint tensors bitwise and preserve canonical names.
8. Implement Hybrid-Muon grouping and full pinned CPU offload before measuring real training.
9. Pass forward, input-gradient, historical parameter-gradient, retrieved-block identity, and two-microbatch accumulation parity.
10. Measure peak allocated and reserved VRAM; reject any result exceeding the native pool high-water by more than 5% or 90% of physical VRAM.
11. Measure complete seed-48 F+B and one optimizer step. A migration should deliver at least a repeatable few-percent end-to-end improvement to justify the maintenance and checkpoint risk.

## Migration recommendation

**Continue the custom CuPy backend for production today.** The new kernel demonstrates that material graph-level native improvements still exist, and it improves both history-free and 61K-history execution without additional memory.

At the same time, start the bounded Torch historical custom-op prototype described above. PyTorch is a credible long-term host framework because fullgraph custom autograd operators work and the standard stack remains faster. It is not yet a credible production replacement because every difficult gate—historical correctness, heavy-history memory, optimizer offload, and exact resume—remains unmeasured.

The decision point should be the seed-48 prototype result:

- if it fits below the memory limit, preserves exact retrieval/backward behavior, and improves complete heavy-history training by several percent, proceed toward a staged migration;
- if it OOMs, graph-breaks around historical state, or loses the standard-block gain inside historical execution, retain CuPy and continue targeted fusion work.

## Validation and artifacts

Focused tests:

```text
tests/test_0089n_fused_qkv_rope.py: 2 passed
tests/test_runtime_defaults.py + block contract: 23 passed, 1 skipped
historical/compaction/accumulation/checkpoint suite: 45 passed
profile/analyzer contracts: 10 passed, 1 skipped
```

Primary artifacts:

```text
benchmarks/results/0089n_depth2_fused_qkv_rope.json
benchmarks/results/0089n_depth20_qkv_rope_{baseline,candidate}_run{1,2}.json
benchmarks/results/0089n_depth20_qkv_rope_{baseline,candidate}_profile.nsys-rep
benchmarks/results/0089n_depth20_qkv_rope_{baseline,candidate}_analysis.json
benchmarks/results/0089n_full_model_qkv_rope_{baseline,candidate}_r2.json
benchmarks/results/0089n_migration_gate_history_{baseline,candidate}.json
benchmarks/results/0089n_migration_gate_depth20_{torch,cupy_promoted}.json
```
