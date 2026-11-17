"""Upstream codec contracts for the diagnostic-only arithmetic intervention."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_0221_hc_arithmetic import bf16_roundtrip
from probe_0222_matched_activations import Codec, compile_codec


@pytest.fixture(scope='module')
def codec(tmp_path_factory):
    from probe_0218_hc_identity import COPY
    if not (COPY/'build-cpu/bin/libggml-cpu.so').exists():
        pytest.skip('pinned CPU reference library required')
    return Codec(compile_codec(tmp_path_factory.mktemp('activation-codec')))


def test_bf16_upstream_ties_and_varied_values(codec):
    values=np.array([0.,-0.,1.,-1.,1+2**-8,1+3*2**-8,2**-120,123.456],np.float32)
    np.testing.assert_array_equal(codec(values,30),bf16_roundtrip(values))


@pytest.mark.parametrize('weight_type',[21,20,42,14,0])
def test_quantized_activations_are_finite_and_shape_preserved(codec,weight_type):
    x=np.random.default_rng(222).normal(size=(2,256)).astype(np.float32)
    got=codec(x,weight_type)
    assert got.shape==x.shape and got.dtype==np.float32 and np.isfinite(got).all()
    assert np.linalg.norm(got-x)/np.linalg.norm(x)<.02
    if weight_type==0:
        np.testing.assert_array_equal(got,x)
