"""Descriptor-only architecture binding; reuse the existing synthetic Qwen graph."""
import ctypes as C
import struct

import pytest
from gguf import GGMLQuantizationType

from mini_llm.native import Runtime, NativeError, _TensorRequirement
from mini_llm.external.qwen4exp_weights import inspect_headers
from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
from test_0209_qwen4exp_model import TinyStore
import test_0201_gguf_store as binary


def fixture_file(tmp_path, monkeypatch, change=None, grouped=False):
    source = TinyStore()
    if change:
        change(source)
    original = binary.value_bytes
    def encode(kind, value, endian):
        if kind == 6:
            return struct.pack(endian + 'f', value)
        return original(kind, value, endian)
    monkeypatch.setattr(binary, 'value_bytes', encode)
    metadata = []
    for key, value in source.metadata.items():
        if isinstance(value, str):
            kind = binary.STRING
        elif isinstance(value, float):
            kind = 6
        elif isinstance(value, list):
            kind, value = binary.ARRAY, (binary.U32, value)
        else:
            kind = binary.U32
        metadata.append((key, kind, value))
    tensors, offset = [], 0
    for name, d in source.tensors.items():
        dims = (256, 4) if grouped and 'hc_' in name and name.endswith('norm.weight') else d.ggml_shape
        type_id = 42 if d.quant_type == 'Q2_0' else int(GGMLQuantizationType[d.quant_type])
        import math
        size = math.prod(dims) // d.format.block_elements * d.format.block_bytes
        payload = source.payload[name].tobytes()[:size].ljust(size, b'\0')
        tensors.append((name, dims, type_id, offset, payload))
        offset = (offset + len(payload) + 31) // 32 * 32
    path = tmp_path / 'qwen.gguf'
    binary.write_gguf(path, tensors, metadata)
    assert source.reads == []
    return path, source


@pytest.mark.parametrize('grouped', [False, True])
def test_required_layer_contract(tmp_path, monkeypatch, grouped):
    path, source = fixture_file(tmp_path, monkeypatch, grouped=grouped)
    expected = inspect_headers(Qwen4ExpGGUFIndex(source))
    with Runtime().gguf(path) as store:
        # Detached snapshots cannot bypass native-owner validation.
        store.metadata['general.architecture'] = 'wrong'
        store.tensors.clear()
        actual = store.validate_qwen4exp()
        assert actual.shapes == expected.shapes
        assert actual.recurrent == (True, False)
        assert actual.ple_layers == ()
        assert actual.dense_bytes == expected.dense_bytes
    with pytest.raises(NativeError, match='closed'):
        store.validate_qwen4exp()
    assert source.reads == []


@pytest.mark.parametrize('case,match', [
    ('architecture', 'architecture'), ('missing', 'expected shape'),
    ('extra', 'Unsupported tensors'), ('layers', 'expected shape'),
    ('ratios', 'ratio'), ('routing', 'exceeds'), ('streams', 'four HC'),
    ('ple', 'PLE'), ('control', 'control weight'), ('embedding', 'IQ4_XS'),
])
def test_architecture_failure(tmp_path, monkeypatch, case, match):
    from dataclasses import replace
    from mini_llm.external.gguf_store import GGUFFormat
    def change(s):
        if case == 'architecture': s.metadata['general.architecture'] = 'other'
        elif case == 'missing': del s.tensors['blk.1.attn_q.weight']
        elif case == 'extra': s.tensors['unknown'] = s.tensors['output.weight']; s.payload['unknown'] = s.payload['output.weight']
        elif case == 'layers': s.metadata['qwen4exp.block_count'] = 3
        elif case == 'ratios': s.metadata['qwen4exp.attention.compress_ratios'] = [4, 0]
        elif case == 'routing': s.metadata['qwen4exp.expert_used_count'] = 5
        elif case == 'streams': s.metadata['qwen4exp.hyper_connection.count'] = 3
        elif case == 'ple': s.metadata['qwen4exp.ple.layers'] = [1]
        else:
            name = 'output_hc_norm.weight' if case == 'control' else 'token_embd.weight'
            s.tensors[name] = replace(s.tensors[name], format=GGUFFormat(20, 'IQ4_NL', 32, 18))
    path, _ = fixture_file(tmp_path, monkeypatch, change)
    with Runtime().gguf(path) as store:
        with pytest.raises((ValueError, NotImplementedError), match=match):
            store.validate_qwen4exp()


@pytest.mark.parametrize('case,match', [
    ('missing', 'missing'), ('shape', 'shape'), ('type', 'type'),
    ('extra', 'outside'), ('duplicate', 'duplicate'), ('rank', 'invalid'),
    ('null', 'empty/null'),
])
def test_native_schema_failure(tmp_path, case, match):
    path = tmp_path / 'tiny.gguf'
    binary.write_gguf(path, [('x', (4,), 0, 0, bytes(16)), ('y', (4,), 0, 32, bytes(16))])
    runtime = Runtime()
    with runtime.gguf(path) as store:
        req = lambda name, rank=1, dim=4, kind=0: _TensorRequirement(name, rank, (C.c_uint64 * 4)(dim), kind)
        entries = [req(b'x'), req(b'y')]
        if case == 'missing': entries[0] = req(b'absent')
        elif case == 'shape': entries[0] = req(b'x', dim=8)
        elif case == 'type': entries[0] = req(b'x', kind=1)
        elif case == 'extra': entries.pop()
        elif case == 'duplicate': entries[1] = req(b'x')
        elif case == 'rank': entries[0] = req(b'x', rank=5)
        schema = None if case == 'null' else (_TensorRequirement * len(entries))(*entries)
        with pytest.raises(NativeError, match=match):
            runtime._call('ml_gguf_validate_schema', store._handle, schema, len(entries))
        # Failure does not destroy or mutate the mapping.
        assert set(store.tensors) == {'x', 'y'}
