"""Lazy, mmap-backed Qwen3-Next GGUF expert index for the shared backend.

This is a checkpoint/weight adapter, NOT a Qwen3-Next decoder. In particular,
DeltaNet, gated attention, recurrent state, shared experts, and tokenization
must be implemented/validated before full text generation is possible.

No tensor is dequantized while indexing. The GGUFReader must stay alive for
as long as any returned PackedGGUFWeight references its mmap-backed bytes.
"""

from collections import Counter
from pathlib import Path
import math

from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.expert_residency import ExpertKey


def _meta(reader, name, default=None):
    field = reader.fields.get(name)
    return default if field is None else field.contents(0)


def _shape(tensor):
    # GGUF stores tensor shapes in GGML order, i.e. input dimension first.
    return tuple(int(d) for d in reversed(tensor.shape))


def _packed(tensor):
    quant_type = tensor.tensor_type.name
    if quant_type not in BLOCK_FORMATS:
        raise NotImplementedError(
            f"{tensor.name}: GGUF {quant_type} is not implemented by the "
            "native packed expert backend; do not silently dequantize an 80B model"
        )
    return PackedGGUFWeight(tensor.data, _shape(tensor), quant_type)


def _row_range(weight, first, last):
    """Zero-copy contiguous rows of packed [out, in] GGUF weights."""
    if len(weight.shape) != 2 or not 0 <= first < last <= weight.shape[0]:
        raise ValueError("Invalid packed matrix row range")
    qk, qb = BLOCK_FORMATS[weight.quant_type]
    if weight.shape[1] % qk:
        raise ValueError("Cannot split an unaligned packed GGUF projection")
    row_bytes = weight.shape[1] // qk * qb
    return PackedGGUFWeight(
        weight.data[first * row_bytes:last * row_bytes],
        (last - first, weight.shape[1]), weight.quant_type,
    )


class Qwen3NextGGUFIndex:
    """Index GGUF tensors and expose routed expert weights on demand.

    Call with a GGUFReader (or use ``open``). The returned ``provider`` works
    directly with HeterogeneousExpertExecutor: gate, up, down are all
    PackedGGUFWeight matrices with shapes [F,D], [F,D], [D,F].
    """

    def __init__(self, reader, *, source_path=None):
        self.reader = reader
        self.source_path = str(source_path) if source_path else None
        self.architecture = str(_meta(reader, 'general.architecture', ''))
        if self.architecture not in ('qwen3next', 'qwen3_next'):
            raise ValueError(
                f"Expected Qwen3-Next GGUF, found {self.architecture!r}; "
                "the existing Llama loader is a different adapter"
            )
        self.tensors = {t.name: t for t in reader.tensors}
        if len(self.tensors) != len(reader.tensors):
            raise ValueError('Duplicate GGUF tensor names')
        prefix = self.architecture
        self.layers = int(_meta(reader, f'{prefix}.block_count', 0))
        self.d_model = int(_meta(reader, f'{prefix}.embedding_length', 0))
        self.expert_count = int(_meta(reader, f'{prefix}.expert_count', 0))
        self.experts_per_token = int(_meta(reader, f'{prefix}.expert_used_count', 0))
        if self.layers <= 0 or self.d_model <= 0:
            raise ValueError('Qwen3-Next GGUF lacks valid layer/embedding metadata')
        if self.expert_count <= 0:
            sample = self.tensors.get('blk.0.ffn_down_exps.weight')
            if sample is not None and len(_shape(sample)) == 3:
                self.expert_count = _shape(sample)[0]
        if self.expert_count <= 0:
            raise ValueError('Cannot determine expert count from GGUF')
        if self.experts_per_token < 0 or self.experts_per_token > self.expert_count:
            raise ValueError('Invalid number of active experts')
        sample_down = self.tensors.get('blk.0.ffn_down_exps.weight')
        if sample_down is None or len(_shape(sample_down)) != 3:
            raise ValueError('Missing blk.0.ffn_down_exps.weight [E,D,F]')
        down_shape = _shape(sample_down)
        if down_shape[:2] != (self.expert_count, self.d_model):
            raise ValueError(f'Unexpected routed down-projection shape {down_shape}')
        self.d_ff = down_shape[2]
        self._validate_layer_names()

    @classmethod
    def open(cls, path):
        """Index an on-disk GGUF; does not load whole-file tensor contents."""
        try:
            from gguf import GGUFReader
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional local GGUF reader with "
                "'.venv/bin/python -m pip install gguf'; the GGUF stays on disk"
            ) from exc
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Model not found: {path}')
        return cls(GGUFReader(str(path)), source_path=path)

    def _validate_layer_names(self):
        for layer in range(self.layers):
            stem = f'blk.{layer}.'
            required = (stem + 'ffn_down_exps.weight', stem + 'ffn_gate_inp.weight')
            missing = [n for n in required if n not in self.tensors]
            if missing:
                raise ValueError(f'Qwen3-Next layer {layer}: missing {missing}')
            fused = stem + 'ffn_gate_up_exps.weight' in self.tensors
            separate = all(stem + part + '.weight' in self.tensors
                           for part in ('ffn_gate_exps', 'ffn_up_exps'))
            if not (fused or separate):
                raise ValueError(f'Qwen3-Next layer {layer}: no gate/up expert tensors')
            down = _shape(self.tensors[stem + 'ffn_down_exps.weight'])
            if down != (self.expert_count, self.d_model, self.d_ff):
                raise ValueError(f'Qwen3-Next layer {layer}: unexpected down shape {down}')
            gate_name = stem + ('ffn_gate_up_exps.weight' if fused else 'ffn_gate_exps.weight')
            expect = (self.expert_count, self.d_ff * (2 if fused else 1), self.d_model)
            if _shape(self.tensors[gate_name]) != expect:
                raise ValueError(f'{gate_name}: expected {expect}, got {_shape(self.tensors[gate_name])}')
            if not fused:
                up_name = stem + 'ffn_up_exps.weight'
                if _shape(self.tensors[up_name]) != (self.expert_count, self.d_ff, self.d_model):
                    raise ValueError(f'{up_name}: unexpected shape {_shape(self.tensors[up_name])}')

    def expert_weights(self, layer_id: int, expert_id: int):
        """Return three zero-copy *packed* matrices; no full-file dequant."""
        layer_id = int(layer_id)
        expert_id = int(expert_id)
        if not 0 <= layer_id < self.layers or not 0 <= expert_id < self.expert_count:
            raise IndexError(f'Invalid expert ({layer_id}, {expert_id})')
        stem = f'blk.{layer_id}.'
        down = _packed(self.tensors[stem + 'ffn_down_exps.weight']).select_outer(expert_id)
        fused_name = stem + 'ffn_gate_up_exps.weight'
        if fused_name in self.tensors:
            # GGUF conversion concatenates [gate, up] along output rows.
            fused = _packed(self.tensors[fused_name]).select_outer(expert_id)
            gate, up = _row_range(fused, 0, self.d_ff), _row_range(fused, self.d_ff, self.d_ff * 2)
        else:
            gate = _packed(self.tensors[stem + 'ffn_gate_exps.weight']).select_outer(expert_id)
            up = _packed(self.tensors[stem + 'ffn_up_exps.weight']).select_outer(expert_id)
        return gate, up, down

    def provider(self, key: ExpertKey):
        if not isinstance(key, ExpertKey):
            raise TypeError('Expert provider expects ExpertKey')
        return self.expert_weights(key.layer_id, key.expert_id)

    def report(self, *, device_budget_gib=6.0, ram_budget_gib=32.0):
        """Metadata-only inventory; does not touch each packed expert's bytes."""
        formats = Counter()
        counts = Counter()
        unsupported = Counter()
        routed_bytes = 0
        for tensor in self.tensors.values():
            kind = tensor.tensor_type.name
            counts[kind] += 1
            try:
                size = int(tensor.data.nbytes)
            except AttributeError:
                size = 0
            formats[kind] += size
            if '.ffn_' in tensor.name and '_exps.weight' in tensor.name:
                routed_bytes += size
                if kind not in BLOCK_FORMATS:
                    unsupported[kind] += 1
        sample_sizes = None
        sample_error = None
        try:
            sample_sizes = [w.nbytes for w in self.expert_weights(0, 0)]
        except (NotImplementedError, ValueError) as exc:
            sample_error = str(exc)
        device_budget = int(float(device_budget_gib) * 1024**3)
        ram_budget = int(float(ram_budget_gib) * 1024**3)
        return {
            'source': self.source_path,
            'architecture': self.architecture,
            'layers': self.layers,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'routed_experts_per_layer': self.expert_count,
            'active_experts_per_token': self.experts_per_token,
            'tensor_count': len(self.tensors),
            'tensor_counts_by_format': dict(sorted(counts.items())),
            'tensor_bytes_by_format': dict(sorted(formats.items())),
            'routed_expert_bytes': routed_bytes,
            'unsupported_routed_formats': dict(sorted(unsupported.items())),
            'first_expert_projection_bytes': sample_sizes,
            'first_expert_error': sample_error,
            'device_cache_budget_bytes': device_budget,
            'host_budget_bytes': ram_budget,
            'estimated_device_cache_experts': (device_budget // sum(sample_sizes)
                                               if sample_sizes else None),
            'full_model_decode_available': False,
            'full_model_decode_blockers': [
                'Gated DeltaNet recurrent decode and state',
                'Gated attention and KV state',
                'shared MoE expert path and layer wiring',
                'quantized non-expert projection coverage',
                'tokenizer and end-to-end logits parity',
            ],
        }
