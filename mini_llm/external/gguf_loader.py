"""GGUF Llama F32/F16/Q8_0 loader for the native NumPy/CuPy inference engine.

Weights from llama.cpp's GGUF conversion have interleaved Q/K RoPE rows,
whereas ExternalLlamaModel expects original Hugging Face rotate_half rows.
Undo that permutation *after* dequantization, before passing weights onward.

The `gguf` package is used only for metadata and tensor directory parsing;
no llama.cpp/Torch model backend is used for inference.
"""

from collections.abc import Mapping
from pathlib import Path
import hashlib

import numpy as np

from .config import ExternalLlamaConfig
from .llama_model import ExternalLlamaModel


# Llama-only GGUF names to original Hugging Face Llama tensor names.
_BASE_TENSORS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_LAYER_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def _as_metadata(field):
    """Get the first scalar value using the public ReaderField interface."""
    return field.contents(0)


def _metadata(reader, name, default=None):
    field = reader.fields.get(name)
    return default if field is None else _as_metadata(field)


def _required(reader, name):
    if name not in reader.fields:
        raise ValueError(f"GGUF is missing required Llama metadata: {name}")
    return _metadata(reader, name)


def _config_from_gguf(reader, tensor_names):
    architecture = _required(reader, "general.architecture")
    if architecture != "llama":
        raise ValueError(f"Unsupported GGUF architecture {architecture!r}; expected 'llama'")
    for key in ("llama.rope.scaling.type",):
        scaling = _metadata(reader, key)
        if scaling not in (None, "none"):
            raise ValueError(f"Unsupported GGUF RoPE scaling: {key}={scaling!r}")
    width = int(_required(reader, "llama.embedding_length"))
    head_count = int(_required(reader, "llama.attention.head_count"))
    if head_count <= 0 or width % head_count:
        raise ValueError("GGUF embedding size must divide evenly across query heads")
    head_dim = width // head_count
    if int(_metadata(reader, "llama.rope.dimension_count", head_dim)) != head_dim:
        raise ValueError("Partial rotary GGUF models are not supported")
    for key in ("llama.attention.key_length", "llama.attention.value_length"):
        val = _metadata(reader, key)
        if val is not None and int(val) != head_dim:
            raise ValueError(f"GGUF {key}={val} differs from head_dim={head_dim}")
    if _metadata(reader, "llama.attention.layer_norm_rms_epsilon") is None:
        raise ValueError("GGUF is missing llama.attention.layer_norm_rms_epsilon")
    # Some files omit llama.vocab_size; the actual token embeddings then define it.
    embedding = tensor_names.get("token_embd.weight")
    if embedding is None:
        raise ValueError("GGUF lacks required token_embd.weight")
    gguf_vocab = int(embedding.shape[1])  # GGUF stores tensor dimensions reversed.
    metadata_vocab = _metadata(reader, "llama.vocab_size")
    if metadata_vocab is not None and int(metadata_vocab) != gguf_vocab:
        raise ValueError("GGUF vocab_size disagrees with token_embd.weight")
    return ExternalLlamaConfig.from_dict({
        "model_type": "llama",
        "hidden_size": width,
        "intermediate_size": int(_required(reader, "llama.feed_forward_length")),
        "num_hidden_layers": int(_required(reader, "llama.block_count")),
        "num_attention_heads": head_count,
        "num_key_value_heads": int(_metadata(reader, "llama.attention.head_count_kv", head_count)),
        "head_dim": head_dim,
        "vocab_size": gguf_vocab,
        "max_position_embeddings": int(_required(reader, "llama.context_length")),
        "rope_theta": float(_metadata(reader, "llama.rope.freq_base", 10000.0)),
        "rms_norm_eps": float(_required(reader, "llama.attention.layer_norm_rms_epsilon")),
        # GGUF omits output.weight for physically tied embeddings.
        "tie_word_embeddings": "output.weight" not in tensor_names,
    })


def _dequantize_q8_0(raw, expected_elements):
    """Decode 34-byte Q8_0 blocks: one little-endian FP16 scale + 32 int8s.

    The GGUFReader.data property is a byte view over the memory-mapped file.
    Return a fresh FP32 array; no bytes are modified on disk.
    """
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if expected_elements % 32 or raw.size != (expected_elements // 32) * 34:
        raise ValueError("Invalid GGUF Q8_0 block count/size")
    blocks = raw.reshape(-1, 34)
    scales = np.frombuffer(blocks[:, :2].copy().tobytes(), dtype="<f2").astype(np.float32)
    quantized = blocks[:, 2:].copy().view(np.int8).reshape(-1, 32)
    return (scales[:, None] * quantized).reshape(-1)


def _restore_hf_rope_rows(matrix, num_heads):
    """Reverse llama.cpp HF->GGUF Q/K row permutation for rotate_half RoPE.

    convert_hf_to_gguf takes [heads, 2, half_dim, in] and swaps its
    2 and half_dim axes, yielding [heads, half_dim, 2, in]. This reverses it.
    """
    rows, columns = matrix.shape
    if num_heads <= 0 or rows % (2 * num_heads):
        raise ValueError("Invalid Q/K head shape for GGUF inverse RoPE permutation")
    half_dim = rows // (num_heads * 2)
    return matrix.reshape(num_heads, half_dim, 2, columns).swapaxes(1, 2).reshape(rows, columns)


class GGUFWeightMap(Mapping):
    """Shape-first, lazy mapping of GGUF tensors to canonical HF names."""

    def __init__(self, reader):
        self.reader = reader
        tensor_names = {tensor.name: tensor for tensor in reader.tensors}
        if len(tensor_names) != len(reader.tensors):
            raise ValueError("Duplicate GGUF tensor name")
        self.config = _config_from_gguf(reader, tensor_names)
        names = {}
        for raw_name, tensor in tensor_names.items():
            canonical = _BASE_TENSORS.get(raw_name)
            if canonical is None and raw_name.startswith("blk."):
                pieces = raw_name.split(".", 2)
                if len(pieces) == 3 and pieces[1].isdecimal():
                    suffix = _LAYER_TENSORS.get(pieces[2])
                    if suffix is not None:
                        canonical = f"model.layers.{int(pieces[1])}.{suffix}"
            if canonical is None:
                raise ValueError(f"Unsupported GGUF tensor name: {raw_name}")
            if canonical in names:
                raise ValueError(f"Duplicate mapped GGUF tensor: {canonical}")
            if tensor.tensor_type.name not in {"F32", "F16", "Q8_0"}:
                raise ValueError(
                    f"Unsupported GGUF tensor type {tensor.tensor_type.name} for {raw_name}"
                )
            names[canonical] = tensor
        self._tensors = names
        self._dtypes = sorted({t.tensor_type.name for t in names.values()})

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        return iter(self._tensors)

    def shape_of(self, name):
        return tuple(int(d) for d in reversed(self._tensors[name].shape))

    @property
    def tensor_types(self):
        return list(self._dtypes)

    def __getitem__(self, name):
        tensor = self._tensors[name]
        shape = self.shape_of(name)
        if tensor.tensor_type.name == "Q8_0":
            values = _dequantize_q8_0(tensor.data, int(np.prod(shape))).reshape(shape)
        else:
            values = np.asarray(tensor.data)
            if values.shape != shape:
                raise ValueError(f"GGUF {name}: tensor data {values.shape} != metadata {shape}")
        if name.endswith("self_attn.q_proj.weight"):
            values = _restore_hf_rope_rows(values, self.config.num_attention_heads)
        elif name.endswith("self_attn.k_proj.weight"):
            values = _restore_hf_rope_rows(values, self.config.num_key_value_heads)
        return values


def load_gguf_model(path, *, dtype="float32", hash_weights=False):
    """Return (ExternalLlamaModel, provenance) for a supported GGUF checkpoint.

    Default FP32 permits checking the quantization error independently of BF16
    inference. For the RTX 5080, dtype='float16' reduces weights + KV-cache VRAM.
    """
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("GGUF loading requires 'pip install gguf'") from exc
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {path}")
    reader = GGUFReader(str(path))
    tensors = GGUFWeightMap(reader)
    model = ExternalLlamaModel(tensors.config, dtype=dtype)
    model.set_weights(tensors)
    file_hash = None
    if hash_weights:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for piece in iter(lambda: stream.read(2 * 1024 * 1024), b""):
                digest.update(piece)
        file_hash = digest.hexdigest()
    provenance = {
        "source_path": str(path),
        "architecture": "llama",
        "format": "gguf",
        "tensor_types": tensors.tensor_types,
        "runtime_dtype": dtype,
        "quantized_weights_dequantized": "Q8_0" in tensors.tensor_types,
        "file_sha256": file_hash,
    }
    return model, provenance
