"""Inspect the AMD assembly Triton emits for the Model-B local-attention dot shapes.

Purpose: decide whether Triton's ``tl.dot`` on ``gfx1201`` uses matrix/tensor
instructions (191 TFLOP/s class) or vector FMA loops (dual-issue FP32 class,
about 40 TFLOP/s).  This determines whether local-attention tiling or pass
fusion can ever approach the BLAS rate measured at the same shapes.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import triton
import triton.language as tl


@triton.jit
def _dot_probe(
    a_ptr, b_ptr, c_ptr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # Mirrors the production KV backward dot chain: scores from a resident key
    # tile, then the probability-operand accumulation into an FP32 tile.
    ns = tl.arange(0, BLOCK_N)
    ds = tl.arange(0, HEAD_DIM)
    ms = tl.arange(0, BLOCK_M)
    keys = tl.load(a_ptr + ns[:, None] * HEAD_DIM + ds[None, :])
    queries = tl.load(b_ptr + ms[:, None] * HEAD_DIM + ds[None, :])
    scores = tl.dot(keys, tl.trans(queries))
    probabilities = tl.exp2(scores)
    acc = tl.dot(probabilities.to(tl.bfloat16), queries, tl.zeros((BLOCK_N, HEAD_DIM), tl.float32))
    acc = tl.dot((scores * 0.5).to(tl.bfloat16), queries, acc)
    tl.store(c_ptr + ns[:, None] * HEAD_DIM + ds[None, :], acc + tl.sum(scores, axis=1)[:, None])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warps', type=int, default=4)
    parser.add_argument('--key-tile', type=int, default=64)
    args = parser.parse_args(argv)
    import cupy as cp
    from mini_llm.ops.triton_dual_resolution_local import _ptr

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('HIP required for this probe')
    block_n, block_m, head_dim = int(args.key_tile), 32, 64
    bf16 = cp.dtype('bfloat16')
    a = cp.ones((block_n, head_dim), dtype=bf16)
    b = cp.ones((block_m, head_dim), dtype=bf16)
    c = cp.empty((block_n, head_dim), dtype=cp.float32)
    compiled = _dot_probe[(1,)](
        _ptr(a), _ptr(b), _ptr(c), BLOCK_N=block_n, BLOCK_M=block_m, HEAD_DIM=head_dim,
        num_warps=args.warps, num_stages=1,
    )
    assembly_by_language = {}
    for name, text in compiled.asm.items():
        if name in ('amdgcn', 'ttgir', 'llir'):
            assembly_by_language[name] = text
    amdc = compiled.asm.get('amdgcn', '')
    patterns = {
        'mfma': r'mfma',
        'wmma': r'wmma',
        'matrix_mac': r'MASM_MACT|MFMAMAC|MAC_|_MAC|dot_4x4',
        'dual_issue_fma': r'v_dual_fmac|dual_fma',
        'vector_fma': r'v_fmad_f32|v_fma3_f32|v_fmac_f32|fmad_f32|fma_f32',
        'exp2_masm': r'MASM_MEX|MEX_|ex2',
        'fp32_mul': r'v_mul_f32|MULF',
        'fp32_cvt': r'cvt_f32|v_cvt',
    }
    counts = {}
    samples = {}
    for label, pattern in patterns.items():
        hits = re.findall(pattern, amdc, flags=re.IGNORECASE)
        counts[label] = len(hits)
        if hits:
            line_index = amdc.lower().find(pattern.split('|')[0].lower())
            samples[label] = [ln.strip() for ln in amdc.splitlines()
                              if re.search(pattern, ln, flags=re.IGNORECASE)][:2]
    record = {
        'hip': True,
        'device': str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'block_n': block_n, 'block_m': block_m, 'head_dim': head_dim,
        'num_warps': args.warps,
        'n_regs': getattr(compiled, 'n_regs', None),
        'n_spills': getattr(compiled, 'n_spills', None),
        'shared_bytes': getattr(compiled.metadata, 'shared', None),
        'asm_sections': sorted(compiled.asm),
        'amd_asm_lines': len(amdc.splitlines()),
        'instruction_counts': counts,
        'sample_lines': samples,
    }
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    raw_path = args.output.with_suffix('.amdgcn.s.txt')
    raw_path.write_text(amdc, encoding='utf-8')
    record['amd_asm_path'] = str(raw_path)
    print(json.dumps({k: record[k] for k in (
        'asm_sections', 'amd_asm_lines', 'n_regs', 'n_spills', 'shared_bytes',
        'instruction_counts')}, indent=2, allow_nan=False))
    for label, lines in samples.items():
        print(label, '::', *lines, sep='\n  ')


if __name__ == '__main__':
    main()
