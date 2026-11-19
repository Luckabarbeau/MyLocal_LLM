# Patch 0200 — Complete local checkpoint inventory

## Authoritative machine-readable artifacts

- [`tensor_inventory.json`](tensor_inventory.json): **all 1224 tensors**, exact names, both GGML and canonical shapes, raw type IDs, per-format block geometry, element/byte counts, row bytes, shard number, relative/absolute offsets and end offsets. `tensor_contract.families` summarizes every observed tensor family and layer/type variants; `loader_comparison` records the pinned source declaration comparison.
- [`gguf_metadata.json`](gguf_metadata.json): complete typed field descriptors and metadata values for **both** shards, including all tokenizer tokens, merges, types and chat-template text. No arrays were truncated with `contents(0)`.

These JSON files are the complete enumeration, not a routed-expert-only sample. `tests/test_0200_qwen4exp_contract.py` verifies accounting, geometry, selected required tensor families and PLE row bounds without model files. Runtime metadata-only validation is independently tested by Patch0201.

## Local files

Directory: `/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/`

| File | Bytes | Tensors | Data start |
|---|---:|---:|---:|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | 1223 | 11,024,384 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | 1 | 192 |

GGUF v3, little endian, alignment32. Total file bytes83,617,662,656; payload83,606,638,080. Tensor ranges do not overlap; gaps/trailing payload bytes are zero. Split numbering/total count agree. Shard2 intentionally contains only split metadata and the PLE table; lack of repeated model metadata is valid.

Header-only SHA256 (not payload hashes):

- shard1 `d6932b90e1b001f706ee5c8a1aebbab3a8a6ae35892df2fb09af459add1140ef`
- shard2 `60a762da9127f76c0a895de009e7245566708ee638c2ae57f0c12b20ae948e4d`

Exact checkpoint revision is not recorded in the files. See `reference_contract.md` for the unverified sidecar and explicit provenance limitation.

## Every observed storage format

| Format | Tensor count | Payload bytes | Role examples |
|---|---:|---:|---|
| BF16 | 484 | 1,527,889,920 | HC matrices, routers, recurrent alpha/beta, PLE projections |
| F16 | 1 | 81,920 | PLE convolution |
| F32 | 292 | 10,063,360 | Norms, recurrent convolution/A/dt |
| IQ2_S | 40 | 10,747,904,000 | Routed gate/up |
| IQ3_S | 20 | 7,208,960,000 | Routed gate/up |
| IQ3_XXS | 34 | 10,918,297,600 | Routed gate/up |
| IQ4_NL | 87 | 47,245,962,240 | Expert down/shared down, PLE table |
| IQ4_XS | 45 | 1,398,906,880 | Token embedding, dense and routed/shared projections |
| Q2_0 | 9 | 2,123,366,400 | Routed down |
| Q4_K | 47 | 198,696,960 | Dense/shared projections |
| Q5_K | 35 | 346,931,200 | Dense/shared projections |
| Q6_K | 129 | 1,877,836,800 | Dense/shared/output projections |
| Q8_0 | 1 | 1,740,800 | Shared down |

The model's filename is a quantization **preset**, not a uniform block type. IQ3_S and Q3_K are distinct encodings even though both have110-byte blocks. Type42 is Q2_0 `(64,18)` in the pinned local definitions; installed `gguf==0.19.0` lacks its enum, so header indexing uses an explicit verified descriptor without monkeypatching the package.

## Required roles and Qwen3-Next differences

Routed gate/up banks have GGML `[2560,640,512]`, canonical `[512,640,2560]`; down `[640,2560,512]`, canonical `[512,2560,640]`. Slicing the slowest expert axis preserves contiguous packed rows. Down width640 cannot be encoded as whole 256-element IQ3_S blocks; its own IQ4_NL/Q2_0 encoding is required.

All layers have separate attention/FFN HC norm, down, up and injection tensors. Final `output_hc_{norm,down,up}.weight` replaces the conventional single-stream final norm graph. Each recurrent block has convolution, A, dt, norm and projections; each fourth block includes full attention and indexer projections/norms. Shared FFNs and routers are required text operations, not optional components. The complete family shapes/types are retained in the JSON.

`per_layer_token_embd.weight`: canonical `[320001536,160]`, IQ4_NL, 28,800,138,240 bytes in shard2. Six other PLE tensors bring PLE total to28,865,879,040 bytes. Sixteen selected rows concatenate to width2560. Hash constants are integer metadata, never rounded floats.

**Text-only classification:** all 1224 tensors belong to required text inference roles. No vision or MTP tensors occur; no tensor may be dropped as an optional vision/MTP component. The vocabulary includes image token IDs, which does not imply a vision encoder is present. Multimodal input support is not claimed.

The existing Qwen3-Next adapter recognizes only its own architecture and conventional residual graph. A separate qwen4exp index is required; expanding architecture acceptance alone would silently execute the wrong mathematics. See the source reuse matrix and equations in `reference_contract.md`.

## Validation boundary

The initial read-only exporter verified byte reconciliation, metadata preservation, tensor declarations against the pinned qwen4exp loader, and unchanged checkpoint size/mtime/ctime/inode. It did not read payloads. Header compatibility and checksum provenance are separate from numerical validation. Full logits/generation remain unvalidated.
