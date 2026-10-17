"""0080AJ3A: NVRTC kernels must not require unavailable host math headers.

This is a CPU-only static regression check; actual NVRTC compilation and
numerical parity still require the separate RTX 5080 CuPy test suite.
"""

import ast
from pathlib import Path


def test_fused_vocab_cuda_is_self_contained_for_nvrtc():
    source_file = (
        Path(__file__).resolve().parents[1]
        / "mini_llm/block_proposal/fused_vocab_cuda.py"
    )
    module = ast.parse(source_file.read_text(encoding="utf-8"))
    cuda = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CUDA"
                for target in node.targets)
    )
    assert "#include" not in cuda
    assert "INFINITY" not in cuda
    assert cuda.count("-3.402823466e+38F") == 3
    assert 'void mix_online_forward(' in cuda
    assert 'void mix_score_backward(' in cuda
    for function in ('fmaxf(', 'expf(', 'logf('):
        assert function in cuda
