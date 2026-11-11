"""Tiny, matched-weight Qwen3-Next GGUF pair for the external C++ harness.

Only fixture construction lives here: no model execution, CUDA, subprocesses, or
reference runtime. ``write_fixture(Path(...))`` returns paths, explicit raw token
IDs, and flat scalar provenance. Both files are deterministic for a fixed seed.

Source contract checked against llama.cpp 171e8846b4af9766c354064cb776cb34a50f053f:
* src/models/qwen3next.cpp:4-125 loads interval/SSM/expert metadata and tensors;
  attention Q rows are per-head [query, gate] (:250-265), recurrent BA rows are
  per-key-head [beta_group, alpha_group] (:424-446). ssm_a is already -exp(A_log)
  (:461), not A_log. Norm weights are direct gamma, not gamma-minus-one.
* src/models/delta-net-base.cpp:33-43,307-317,389-399 asserts S_k == S_v,
  H_v % H_k == 0, and square recurrent state [S_v,S_v,H_v,B]. Therefore the
  earlier Python-only dk=3,dv=32 fixture must NOT be reused for this pair.
* src/models/qwen3next.cpp:57-73 uses ssm.state_size for both loaded head sizes;
  :321-327 derives value size from inner_size/time_step_rank. Here both are 32.
* src/models/models.h:14-18 implements x/sqrt(sum(x*x)+eps) via RMSNorm;
  qwen3next.cpp:507-510 supplies the RMS epsilon, set here to 1e-6 to match
  MyLocal_LLM's recurrent l2_eps. Attention key/value lengths are explicit,
  independent of embedding_length/head_count (llama-model.cpp:1435-1444).

The original uses Q8_0 routed banks because MyLocal_LLM rejects dense banks.
The reference has ONLY F32 tensors: feeding Q8_0 weights to llama.cpp's CPU
matmuls would also quantize activations and confound a matched-weight check.
Actual llama.cpp execution/parity is deliberately left to the parent harness.
"""

from pathlib import Path

import numpy as np


LLAMA_CPP_REVISION = "171e8846b4af9766c354064cb776cb34a50f053f"
PROMPT_TEXT = "ab a\nab"
# Vocabulary IDs 0..255 represent bytes; the sole merge 'a b' has ID 256.
TOKEN_IDS = (256, 32, 97, 10, 256)
D_MODEL = 32
LAYERS = 4
INTERVAL = 4
HEAD_DIM = 32
Q_HEADS = 4
KV_HEADS = 2
KEY_HEADS = 2
VALUE_HEADS = 4
ROTARY_DIM = 8
CONV_KERNEL = 4
EXPERTS = 4
TOP_K = 2
EXPERT_WIDTH = 32
SHARED_WIDTH = 64
RMS_EPS = 1e-6
Q8_SCALE = 1 / 128


def _tokens():
    """GPT-2 ByteLevel alphabet in byte-ID order, plus merge and controls."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {byte: chr(byte) for byte in visible}
    extra = 0
    for byte in range(256):
        if byte not in alphabet:
            alphabet[byte] = chr(256 + extra)
            extra += 1
    return [alphabet[byte] for byte in range(256)] + [
        "ab", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    ]


def _pack_q8(rng, shape):
    """Return physical byte-row shape (NOT raw_shape) and exact F32 values."""
    blocks = int(np.prod(shape)) // 32
    quants = rng.integers(-7, 8, (blocks, 32), dtype=np.int8)
    scales = np.full((blocks, 1), Q8_SCALE, dtype="<f2")
    raw = np.concatenate((scales.view(np.uint8), quants.view(np.uint8)), axis=1)
    # GGUFWriter.add_tensor_info converts uint8 byte-row width 34 -> logical 32.
    # Passing raw_shape=shape would incorrectly convert that logical shape again.
    raw = raw.reshape(*shape[:-1], shape[-1] // 32 * 34)
    decoded = (scales.astype(np.float32) * quants.astype(np.float32)).reshape(shape)
    return raw, decoded


def _add_metadata(writer, gguf, tokens, seed, reference):
    writer.add_name("0106q tiny matched-weight Qwen3-Next")
    writer.add_file_type(int(gguf.LlamaFileType.ALL_F32 if reference
                             else gguf.LlamaFileType.MOSTLY_Q8_0))
    writer.add_context_length(32)
    writer.add_embedding_length(D_MODEL)
    writer.add_block_count(LAYERS)
    writer.add_vocab_size(len(tokens))
    writer.add_feed_forward_length(128)
    writer.add_expert_count(EXPERTS)
    writer.add_expert_used_count(TOP_K)
    writer.add_expert_feed_forward_length(EXPERT_WIDTH)
    writer.add_expert_shared_feed_forward_length(SHARED_WIDTH)
    writer.add_full_attention_interval(INTERVAL)
    writer.add_head_count(Q_HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    writer.add_rope_dimension_count(ROTARY_DIM)
    writer.add_rope_freq_base(5e6)
    writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    writer.add_ssm_conv_kernel(CONV_KERNEL)
    writer.add_ssm_inner_size(VALUE_HEADS * HEAD_DIM)
    writer.add_ssm_state_size(HEAD_DIM)
    writer.add_ssm_time_step_rank(VALUE_HEADS)
    writer.add_ssm_group_count(KEY_HEADS)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * 257 + [3, 3, 3])
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(257)
    writer.add_eos_token_id(259)
    writer.add_eot_token_id(259)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_string("fixture.id", "0106q")
    writer.add_uint64("fixture.seed", seed)
    writer.add_string("fixture.llama_cpp_revision", LLAMA_CPP_REVISION)
    writer.add_string("fixture.prompt_text", PROMPT_TEXT)
    writer.add_bool("fixture.all_f32_reference", reference)


def write_fixture(directory: Path, seed=10617) -> dict:
    """Write a small packed-original/all-F32-reference pair, without executing it.

    Returns ``{'model_path': Path, 'reference_path': Path, 'token_ids': list[int],
    'metadata': dict[str, scalar]}``. Existing output files are not overwritten.
    The reference is for an independent GGUF runtime, NOT MyLocal_LLM's model
    loader (which intentionally requires packed routed banks).
    """
    import gguf

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    directory = Path(directory)
    model_path = directory / "qwen3next-0106q-q8.gguf"
    reference_path = directory / "qwen3next-0106q-f32.gguf"
    for path in (model_path, reference_path):
        if path.exists():
            raise FileExistsError(path)
    directory.mkdir(parents=True, exist_ok=True)
    tokens = _tokens()
    rng = np.random.default_rng(seed)
    original = gguf.GGUFWriter(model_path, "qwen3next")
    reference = gguf.GGUFWriter(reference_path, "qwen3next")
    tensor_count = 0

    def add(name, shape, *, packed=False, values=None, std=0.03):
        nonlocal tensor_count
        if packed:
            raw, values = _pack_q8(rng, shape)
            original.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            if values is None:
                values = rng.normal(0, std, shape).astype(np.float32)
            values = np.ascontiguousarray(values, dtype=np.float32)
            original.add_tensor(name, values)
        reference.add_tensor(name, values)
        tensor_count += 1

    def norm(name):
        add(name, (D_MODEL,), values=(
            1 + rng.normal(0, 0.03, D_MODEL)).astype(np.float32))

    try:
        _add_metadata(original, gguf, tokens, seed, False)
        _add_metadata(reference, gguf, tokens, seed, True)
        add("token_embd.weight", (len(tokens), D_MODEL), packed=True)
        norm("output_norm.weight")
        add("output.weight", (len(tokens), D_MODEL), packed=True)
        for layer in range(LAYERS):
            stem = f"blk.{layer}."
            norm(stem + "attn_norm.weight")
            norm(stem + "post_attention_norm.weight")
            if (layer + 1) % INTERVAL == 0:
                for suffix, shape in {
                    "attn_q.weight": (2 * Q_HEADS * HEAD_DIM, D_MODEL),
                    "attn_k.weight": (KV_HEADS * HEAD_DIM, D_MODEL),
                    "attn_v.weight": (KV_HEADS * HEAD_DIM, D_MODEL),
                    "attn_output.weight": (D_MODEL, Q_HEADS * HEAD_DIM),
                }.items():
                    add(stem + suffix, shape)
                norm(stem + "attn_q_norm.weight")
                norm(stem + "attn_k_norm.weight")
            else:
                inner = VALUE_HEADS * HEAD_DIM
                channels = 2 * KEY_HEADS * HEAD_DIM + inner
                for suffix, shape in {
                    "attn_qkv.weight": (channels, D_MODEL),
                    "attn_gate.weight": (inner, D_MODEL),
                    "ssm_ba.weight": (2 * VALUE_HEADS, D_MODEL),
                    "ssm_out.weight": (D_MODEL, inner),
                }.items():
                    add(stem + suffix, shape)
                add(stem + "ssm_conv1d.weight", (channels, CONV_KERNEL), std=0.15)
                add(stem + "ssm_dt.bias", (VALUE_HEADS,), values=np.linspace(
                    -0.8, 0.2, VALUE_HEADS, dtype=np.float32))
                add(stem + "ssm_a", (VALUE_HEADS,), values=-np.linspace(
                    0.25, 1.0, VALUE_HEADS, dtype=np.float32))
                norm(stem + "ssm_norm.weight")
            for suffix, shape in {
                "ffn_gate_inp.weight": (EXPERTS, D_MODEL),
                "ffn_gate_inp_shexp.weight": (D_MODEL,),
                "ffn_gate_shexp.weight": (SHARED_WIDTH, D_MODEL),
                "ffn_up_shexp.weight": (SHARED_WIDTH, D_MODEL),
                "ffn_down_shexp.weight": (D_MODEL, SHARED_WIDTH),
                "ffn_gate_exps.weight": (EXPERTS, EXPERT_WIDTH, D_MODEL),
                "ffn_up_exps.weight": (EXPERTS, EXPERT_WIDTH, D_MODEL),
                "ffn_down_exps.weight": (EXPERTS, D_MODEL, EXPERT_WIDTH),
            }.items():
                add(stem + suffix, shape, packed="_exps.weight" in suffix)
        for writer in (original, reference):
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
    finally:
        original.close()
        reference.close()

    return {
        "model_path": model_path,
        "reference_path": reference_path,
        "token_ids": list(TOKEN_IDS),
        "metadata": {
            "fixture_id": "0106q", "seed": seed, "architecture": "qwen3next",
            "llama_cpp_revision": LLAMA_CPP_REVISION,
            "d_model": D_MODEL, "layers": LAYERS, "context_length": 32,
            "full_attention_interval": INTERVAL, "attention_q_heads": Q_HEADS,
            "attention_kv_heads": KV_HEADS, "attention_key_dim": HEAD_DIM,
            "attention_value_dim": HEAD_DIM, "rotary_dim": ROTARY_DIM,
            "recurrent_key_heads": KEY_HEADS, "recurrent_value_heads": VALUE_HEADS,
            "recurrent_key_dim": HEAD_DIM, "recurrent_value_dim": HEAD_DIM,
            "conv_kernel": CONV_KERNEL, "expert_count": EXPERTS, "top_k": TOP_K,
            "expert_width": EXPERT_WIDTH, "shared_width": SHARED_WIDTH,
            "vocab_size": len(tokens), "tensor_count": tensor_count,
            "original_quantization": "F32+Q8_0", "reference_quantization": "F32",
            "q8_scale": Q8_SCALE, "q8_quant_min": -7, "q8_quant_max": 7,
            "dense_projection_std": 0.03, "rms_epsilon": RMS_EPS,
            "tokenizer_model": "gpt2", "tokenizer_pre": "qwen2",
            "prompt_text": PROMPT_TEXT, "token_count": len(TOKEN_IDS),
        },
    }
