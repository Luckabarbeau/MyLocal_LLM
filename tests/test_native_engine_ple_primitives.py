"""Standalone PLE primitives only; no native runtime shared library needed."""
import os
from pathlib import Path
import runpy

import pytest


@pytest.fixture(scope="module")
def harness():
    return runpy.run_path(str(Path(__file__).with_name("native_ple_primitives_harness.py")))


def test_ple_hash_uint64_eos_state(harness):
    harness["hash_checks"]()


@pytest.mark.skipif(os.environ.get("MINI_LLM_TEST_PLE_CUDA") != "1",
                    reason="opt in with MINI_LLM_TEST_PLE_CUDA=1 (temporary nvcc build)")
def test_ple_cuda_original_shape(harness):
    pytest.importorskip("cupy")
    harness["cuda_checks"]()
