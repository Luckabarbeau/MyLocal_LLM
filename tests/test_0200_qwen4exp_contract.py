"""0200 recorded local headers: complete namespace, exact storage and geometry."""
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "docs" / "qwen4exp"


@pytest.fixture(scope="module")
def audit():
    return json.loads((ROOT / "tensor_inventory.json").read_text())


@pytest.fixture(scope="module")
def metadata():
    return json.loads((ROOT / "gguf_metadata.json").read_text())["shards"][0]["fields"]


def test_inventory_complete_byte_ranges_and_accounting(audit):
    tensors = audit["tensors"]
    assert len(tensors) == 1224 == len({t["name"] for t in tensors})
    assert sum(t["n_bytes"] for t in tensors) == 83606638080
    assert sum(t["n_elements"] for t in tensors) == 176943899520
    assert {t["type"] for t in tensors} == {
        "BF16", "F16", "F32", "IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL",
        "IQ4_XS", "Q2_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}
    for shard in audit["shards"]:
        end = shard["data_offset"]
        selected = sorted((t for t in tensors if t["shard_index"] == shard["shard_index"]),
                          key=lambda t: t["offset_absolute"])
        assert len(selected) == shard["tensor_count"]
        for t in selected:
            assert t["shape_logical_numpy"] == t["shape_ggml"][::-1]
            assert t["shape_ggml"][0] % t["block_elements"] == 0
            assert t["n_elements"] == math.prod(t["shape_ggml"])
            assert t["n_bytes"] == t["n_elements"] // t["block_elements"] * t["block_bytes"]
            assert t["offset_absolute"] >= end
            assert t["offset_absolute"] % shard["alignment"] == 0
            end = t["offset_absolute"] + t["n_bytes"]
            assert end == t["end_offset_absolute"] <= shard["stat_before"]["size_bytes"]
        assert shard["stat_unchanged"]


def test_all_layer_expert_hc_and_mixer_names(audit, metadata):
    assert metadata["general.architecture"] == "qwen4exp"
    d, f, experts = (metadata["qwen4exp." + name] for name in
                     ("embedding_length", "expert_feed_forward_length", "expert_count"))
    assert (d, f, experts) == (2560, 640, 512)
    assert metadata["qwen4exp.expert_used_count"] == 10
    tensors = {t["name"]: t for t in audit["tensors"]}
    c, rank = metadata["qwen4exp.hyper_connection.count"], metadata["qwen4exp.hyper_connection.low_rank"]
    assert (c, rank) == (4, 320)
    for layer in range(48):
        stem = f"blk.{layer}."
        expected = {"ffn_gate_exps.weight": [experts, f, d],
                    "ffn_up_exps.weight": [experts, f, d],
                    "ffn_down_exps.weight": [experts, d, f],
                    "ffn_gate_inp.weight": [experts, d]}
        for part in ("attn", "ffn"):
            expected.update({f"hc_{part}_norm.weight": [c * d],
                             f"hc_{part}_down.weight": [rank, c * d],
                             f"hc_{part}_up.weight": [c * d, rank],
                             f"hc_{part}_inject.weight": [c, c * d]})
        for suffix, shape in expected.items():
            assert tensors[stem + suffix]["shape_logical_numpy"] == shape
        if (layer + 1) % 4:
            assert tensors[stem + "attn_qkv.weight"]["shape_logical_numpy"] == [10240, d]
            assert tensors[stem + "ssm_conv1d.weight"]["shape_logical_numpy"] == [10240, 4]
            assert tensors[stem + "ssm_out.weight"]["shape_logical_numpy"] == [d, 6144]
        else:
            assert metadata["qwen4exp.attention.compress_ratios"][layer] == 4
            assert stem + "indexer.q_proj.weight" in tensors
            assert stem + "indexer.k_proj.weight" in tensors


def test_ple_and_text_only_contract(audit, metadata):
    assert metadata["qwen4exp.ple.layers"] == [1]
    assert metadata["qwen4exp.ple.ngram_size"] == 3
    assert metadata["qwen4exp.ple.eos_token_id"] == 248044
    assert metadata["tokenizer.ggml.eos_token_id"] == 248046
    assert len(metadata["tokenizer.ggml.tokens"]) == 248320
    offsets, sizes = (metadata["qwen4exp.ple." + k] for k in ("head_offsets", "head_vocab_sizes"))
    assert len(offsets) == len(sizes) == 16
    assert all(offsets[i] + sizes[i] == offsets[i + 1] for i in range(15))
    table = next(t for t in audit["tensors"] if t["name"] == "per_layer_token_embd.weight")
    assert table["shape_logical_numpy"] == [320001536, 160]
    assert offsets[-1] + sizes[-1] <= table["shape_logical_numpy"][0]
    assert table["type"] == "IQ4_NL" and table["shard_index"] == 1
    assert audit["categories"]["MTP"]["tensor_count"] == 0
    assert audit["categories"]["vision"]["tensor_count"] == 0
    assert audit["provenance"]["checkpoint_exact_revision"] is None
