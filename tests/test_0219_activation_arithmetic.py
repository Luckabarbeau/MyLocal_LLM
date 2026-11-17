"""Linked CPU arithmetic contracts, not FP32/GGML output equivalence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('activation_0219',ROOT/'scripts/probe_0219_activation_arithmetic.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

@pytest.fixture(scope='module')
def binary(tmp_path_factory):
    return probe.compile_reference(tmp_path_factory.mktemp('linked0219'))

@pytest.mark.parametrize('type_id,activation',[(12,'q8_K'),(14,'q8_K'),(20,'q8_0'),(42,'q8_0'),(30,'bf16')])
def test_linked_synthetic_arithmetic(binary,tmp_path,type_id,activation):
    # Upstream quantize_chunk creates synthetic weights; all decoding and graph
    # arithmetic are independently supplied by linked GGML, not Python codecs.
    x = (np.random.default_rng(219).normal(size=256)*np.logspace(-3,2,256)).astype(np.float32)
    r = probe.run_tile(binary,tmp_path,type_id,np.empty(0,np.uint8),x,synthetic=True)
    assert r['traits']['activation_type'] == activation
    assert r['finite']
    # Quantized integer dots and dequantized float64 products can differ by
    # scales/reduction order; this is a bounded smoke check, not parity proof.
    assert r['graph_vs_rounded_oracle']['relative_l2'] < 0.001
    assert r['activation_rounding_vs_fp64']['relative_l2'] > 0
    assert r['fp32_vs_fp64']['relative_l2'] < 1e-5


def test_input_provenance():
    cases = probe.inputs(2560)
    assert cases[0][0] == 'real_A'
    expected = np.load(ROOT/'docs/qwen4exp/diagnostics_0218_v1/hc_attention_read.npy').reshape(-1)
    np.testing.assert_array_equal(cases[0][1],expected)
    assert len(cases) == 3
    assert probe.inputs(640)[0][0] == 'deterministic_width_substitute'
