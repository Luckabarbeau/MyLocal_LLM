# WP2 milestone: IQ descriptor coverage and actual checkpoint header parity

Date: 2026-11-23. **Descriptor-only milestone; not full WP2 completion, dense GPU loading, or model execution.** No decoding, weight upload, full payload scan or checkpoint modification was performed.

## Evidence before editing

`docs/qwen4exp/tensor_inventory.md` records the two-shard checkpoint and 13 observed formats. Existing diagnostic configuration `scripts/probe_0218_hc_identity.py:18` supplies its concrete `MODEL` path. No actual model path was found in `configs/*` or the example serving configurations; those are not claimed as the checkpoint configuration source. `strata_launch_bench_capture.sh` selects the `iq3_s` preset and accepts a `MODEL_PATH` override, but does not pin an absolute checkpoint path.

Checkpoint directory: `/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/`. Both documented files exist and were opened successfully.

Installed `gguf==0.19.0`, `.venv/lib/python3.12/site-packages/gguf/constants.py`, provides enum identities and `GGML_QUANT_SIZES` at lines 4249–4254. Constants file SHA256: `19e98ab20fb4d197efb4e23e5fc8de21a8c0d528af1dffc14695af84a0015f24`. Repository `mini_llm/ops/quantized_weight.py:24–38` independently agrees on the five byte/block sizes.

| Added format | Type ID | Elements/block | Bytes/block | Constants expression for bytes (QK_K=256) |
|---|---:|---:|---:|---|
| IQ3_XXS | 18 | 256 | 98 | 2 + QK_K/4 + QK_K/8 |
| IQ4_NL | 20 | 32 | 18 | 2 + 16 |
| IQ3_S | 21 | 256 | 110 | 2 + QK_K/4 + QK_K/8 + QK_K/32 + 4 |
| IQ2_S | 22 | 256 | 82 | 2 + QK_K/4 + QK_K/16 |
| IQ4_XS | 23 | 256 | 136 | 2 + 2 + QK_K/2 + QK_K/64 |

Only these five descriptor cases were added to `format()`; existing type identities remain distinct even where block sizes coincide. Unknown types still fail. No C ABI or ctypes changes were needed.

Each added parametrized test checks enum identity, installed constants, repository constants, and native/Python descriptor parity for a multiblock rank-3 tensor at a nonzero offset, then rejects an invalid row width and a truncated block. No corrective edit was required.

## Exact validation commands and results

```sh
cmake --build /tmp/mini-llm-wp1-build && MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m pytest -q tests/test_native_engine_gguf.py tests/test_native_engine_runtime.py
```

Build succeeded; **28 passed in 0.54s**, no skips (20 GGUF cases, 8 runtime cases). The three pre-existing misleading-indentation warnings remain (now lines 34, 59, 113). This supersedes the previous 23-test verified state without changing its tests' expectations.

```sh
MINI_LLM_NATIVE_LIBRARY=/tmp/mini-llm-wp1-build/libmini_llm_native.so .venv/bin/python -m docs.native_engine.probe_gguf_inventory /home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf docs/native_engine/WP2_GGUF_INVENTORY.json
```

Exited successfully, `status=passed`, elapsed **4.953515704997699 seconds**, within a 60-second command limit. All 1224 names and 13 descriptor fields matched `MultiShardTensorStore`; both per-shard on-disk metadata value dictionaries matched, excluding Python's synthetic `GGUF.*` fields. This does not assert metadata ABI type-tag parity independently of the values. Native parsing validates overlapping metadata type/encoded-value signatures itself.

`WP2_GGUF_INVENTORY.json` records source file SHA256s (including the exact probe), library hash, installed constants hash, Git HEAD, checkpoint stat identities, and per-format accounting. Git HEAD was `fb8de6d0dbe8c9e4197b8bc3c411b9b810a89728`; source hashes are necessary because the native files are untracked worktree changes, not contained in that HEAD. Built library SHA256: `ba46511a248dd79c805f8c968c36eaf4a6d8d77361e71b969a1eea2ca094791e`.

## Observed complete inventory

| Format | Tensors | Tensor bytes |
|---|---:|---:|
| BF16 | 484 | 1,527,889,920 |
| F16 | 1 | 81,920 |
| F32 | 292 | 10,063,360 |
| IQ2_S | 40 | 10,747,904,000 |
| IQ3_S | 20 | 7,208,960,000 |
| IQ3_XXS | 34 | 10,918,297,600 |
| IQ4_NL | 87 | 47,245,962,240 |
| IQ4_XS | 45 | 1,398,906,880 |
| Q2_0 | 9 | 2,123,366,400 |
| Q4_K | 47 | 198,696,960 |
| Q5_K | 35 | 346,931,200 |
| Q6_K | 129 | 1,877,836,800 |
| Q8_0 | 1 | 1,740,800 |
| **Total** | **1,224** | **83,606,638,080** |

| Shard suffix | File bytes | Tensors | On-disk metadata keys | Data start | Tensor bytes |
|---|---:|---:|---:|---:|---:|
| 00001-of-00002.gguf | 54,817,524,224 | 1223 | 67 | 11,024,384 | 54,806,499,840 |
| 00002-of-00002.gguf | 28,800,138,432 | 1 | 3 | 192 | 28,800,138,240 |

Total file bytes: **83,617,662,656**. File bytes minus tensor bytes equal the two aligned header regions: **11,024,576** bytes. The parser validates nonoverlap and bounds. File size/device/inode/mtime/ctime were unchanged across the probe.

Header SHA256s (read only `[0, data_start)`, capped at 64 MiB per shard):

- shard 1: `d6932b90e1b001f706ee5c8a1aebbab3a8a6ae35892df2fb09af459add1140ef`
- shard 2: `60a762da9127f76c0a895de009e7245566708ee638c2ae57f0c12b20ae948e4d`

Both match the previous recorded header inventory. **No whole-file hash was attempted**: scanning 83+ GB would read tensor payloads, outside this header-only IO milestone. Embedded immutable checkpoint revision remains unknown; header hashes/stat identity do not establish payload integrity or immutable model provenance.

## Remaining WP2 blockers

The actual checkpoint's descriptor-format coverage and full header inventory parity gaps are now closed for this source state. Remaining work is native architecture-specific metadata/tensor contracts and required roles/alias handling; bounded payload access with retained mapping leases; model-weight memory/admission accounting and stream/event ownership; dense/device conversion and upload policies where appropriate; architecture-level and numerical/composition validation. Full payload integrity and immutable checkpoint provenance remain open. Descriptor parity is not model correctness, GPU format support, or a WP2 sign-off.
