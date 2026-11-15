#!/usr/bin/env python3
"""Decompose the HIP LM-head backward region at Model B shapes.

Production form (``DecoderLanguageModel.chunked_lm_head_backward`` with cached
logits) is:

    d_logits = chunked_bf16_cross_entropy_grad_inplace(logits, ...)   # in place
    W_grad  += x_chunk.T @ d_logits                                    # [D, V] FP32
    dx      = d_logits @ W.T                                         # [T, D]

This benchmark times each production statement in isolation, plus candidate
alternatives (``projection_wgrad`` BF16-gradient dispatch, a beta=1 FP32-accumulating
gemmEx that avoids the BF16 temporary and the extra elementwise add, and a
contiguous-transpose activation). Nothing here changes production code.

FLOPs count as two per FMA. Dense GEMM FLOPs = 2*T*D*V for each of the two
LM-head GEMMs. Timings use CUDA/HIP events on the current stream, and no
allocation or input conversion happens inside a timed region unless the
statement itself performs it (that is the point of the comparison).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.d_model, args.vocab) < 1:
        parser.error("shape arguments must be positive")
    if min(args.warmup, args.repeats, args.iterations) < 1:
        parser.error("warmup, repeats and iterations must be positive")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    return args


def main() -> int:
    args = _parse_args()

    # Match the production runtime profile, which sets MINI_LLM_FUSED_BF16_CE=1
    # (consumer-gpu), so the fused in-place CE kernel rather than the FP32
    # reference fallback is measured.
    os.environ.setdefault("MINI_LLM_FUSED_BF16_CE", "1")

    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError("This benchmark is restricted to AMD HIP")

    from mini_llm.ops import loss as loss_ops
    from mini_llm.ops.hip_projection_wgrad import (
        projection_wgrad,
    )
    from cupy_backends.cuda.libs import cublas

    t, d_model, vocab = args.tokens, args.d_model, args.vocab
    bf16 = cp.dtype("bfloat16")

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    info = {
        k: (v.decode() if isinstance(v, bytes) else v)
        for k, v in props.items()
        if k in ("name", "gcnArchName", "multiProcessorCount", "clockRate")
    }
    print("Device:", json.dumps(info), flush=True)

    cp.random.seed(12031)
    x = (cp.random.standard_normal((t, d_model), dtype=cp.float32) * 0.5).astype(bf16)
    # Production orientation: W is [d_model, vocab]; logits = x @ W.
    W = (cp.random.standard_normal((d_model, vocab), dtype=cp.float32) * 0.5).astype(bf16)
    W_grad = cp.zeros((d_model, vocab), dtype=cp.float32)
    W_grad_ref = cp.zeros((d_model, vocab), dtype=cp.float32)
    dx_out = cp.zeros((t, d_model), dtype=bf16)
    x_T_contig = cp.ascontiguousarray(x.T)

    # Cached-logits path: the CE gradient overwrites the logits tile in place,
    # so each measured CE call needs a pristine tile. The tile refill is done
    # outside the timed region.
    def fresh_logits():
        return x @ W

    logits_template = fresh_logits()
    targets = cp.random.randint(0, vocab, size=(t,), dtype=cp.int32)

    def run_ce():
        logits = logits_template.copy()
        return loss_ops.chunked_bf16_cross_entropy_grad_inplace(
            logits, targets, normalizer_count=float(t), grad_scale=1.0
        )

    d_logits = run_ce()
    if not bool(cp.all(cp.isfinite(d_logits))):
        raise RuntimeError("d_logits contains non-finite values")

    def measure(name, fn, flops=0):
        for _ in range(args.warmup):
            fn()
        cp.cuda.get_current_stream().synchronize()
        samples = []
        last = None
        for _ in range(args.repeats):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            for _ in range(args.iterations):
                last = fn()
            end.record()
            end.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start, end) / args.iterations)
        value = float(median(samples))
        row = {
            "name": name,
            "median_ms": value,
            "min_ms": float(min(samples)),
            "max_ms": float(max(samples)),
            "samples_ms": samples,
        }
        if flops:
            row["tflop_per_s"] = 2.0 * flops / (value * 1e-3) / 1e12
        if last is not None:
            row["out_dtype"] = str(getattr(last, "dtype", "scalar"))
            row["out_shape"] = [int(v) for v in getattr(last, "shape", ())]
        print(f"  {name:<34} {value:8.3f} ms", flush=True)
        return row, last

    rows = []

    # --- cross-entropy gradient pass (destructive; clone outside timing) -----
    row, _ = measure("tile_clone_only", lambda: logits_template.copy())
    rows.append(row)
    row, _ = measure("ce_grad_inplace(copy_first)", run_ce)
    rows.append(row)

    # --- production weight-gradient statement ---------------------------------
    row, prod_wgrad = measure(
        "wgrad_prod  W_grad += x.T @ d_logits",
        lambda: W_grad.__iadd__(x.T @ d_logits),
        flops=t * d_model * vocab,
    )
    rows.append(row)
    row, raw_wgrad = measure("wgrad_bf16_dot  x.T @ d_logits",
                             lambda: x.T @ d_logits,
                             flops=t * d_model * vocab)
    rows.append(row)
    row, _ = measure("wgrad_bf16_dot_contiguous_xT  xT @ d_logits",
                     lambda: x_T_contig @ d_logits,
                     flops=t * d_model * vocab)
    rows.append(row)

    # --- candidate: helper with BF16 gradient (one FP32-output gemmEx) -------
    row, helper_wgrad = measure("wgrad_helper  projection_wgrad(x, d_logits)",
                                lambda: projection_wgrad(x, d_logits),
                                flops=t * d_model * vocab)
    rows.append(row)
    row, _ = measure("wgrad_helper_plus_add  W_grad += projection_wgrad(...)",
                     lambda: W_grad.__iadd__(projection_wgrad(x, d_logits)),
                     flops=t * d_model * vocab)
    rows.append(row)

    # --- candidate: beta=1 accumulation straight into the FP32 parameter grad -
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    pointer_mode = cublas.getPointerMode(handle)
    alpha = np.array(1.0, dtype=np.float32)
    beta_one = np.array(1.0, dtype=np.float32)

    def wgrad_into():
        # Row-major C += A.T @ B becomes column-major C.T += B.T @ A.
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
            vocab, d_model, t, alpha.ctypes.data,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            x.data.ptr, cp.cuda.runtime.CUDA_R_16BF, d_model,
            beta_one.ctypes.data, W_grad_ref.data.ptr,
            cp.cuda.runtime.CUDA_R_32F, vocab,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    if pointer_mode != cublas.CUBLAS_POINTER_MODE_HOST:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    row, _ = measure("wgrad_beta1_into_param_grad", wgrad_into,
                     flops=t * d_model * vocab)
    rows.append(row)
    if pointer_mode != cublas.CUBLAS_POINTER_MODE_HOST:
        cublas.setPointerMode(handle, pointer_mode)

    # --- input-gradient (dX) statement ----------------------------------------
    W_T = W.T
    W_T_contig = cp.ascontiguousarray(W.T)
    row, prod_dx = measure("dx_prod  dx[...] = d_logits @ W.T",
                           lambda: dx_out.__setitem__((slice(None),), d_logits @ W_T),
                           flops=t * d_model * vocab)
    rows.append(row)
    row, raw_dx = measure("dx_bf16_dot  d_logits @ W.T",
                          lambda: d_logits @ W_T,
                          flops=t * d_model * vocab)
    rows.append(row)
    row, _ = measure("dx_bf16_dot_contiguous_WT  d_logits @ WT_contig",
                     lambda: d_logits @ W_T_contig,
                     flops=t * d_model * vocab)
    rows.append(row)

    # Explicit gemmEx orientations. Row-major dx[T,D] = d_logits[T,V] @ W.T
    # becomes column-major dx_cm[D,T] = (W stored row-major [D,V] seen as
    # column-major [V,D]) transposed @ (d_logits seen as column-major [V,T]).
    dx_ex = cp.zeros((t, d_model), dtype=bf16)
    dx_f32 = cp.zeros((t, d_model), dtype=cp.float32)
    beta_zero = np.array(0.0, dtype=np.float32)

    def dx_ex_TN():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            d_model, t, vocab, alpha.ctypes.data,
            W.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            beta_zero.ctypes.data, dx_ex.data.ptr,
            cp.cuda.runtime.CUDA_R_16BF, d_model,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_TN_bf16out", dx_ex_TN, flops=t * d_model * vocab)
    rows.append(row)

    def dx_ex_NN():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
            d_model, t, vocab, alpha.ctypes.data,
            W_T_contig.data.ptr, cp.cuda.runtime.CUDA_R_16BF, d_model,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            beta_zero.ctypes.data, dx_ex.data.ptr,
            cp.cuda.runtime.CUDA_R_16BF, d_model,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_NN_bf16out", dx_ex_NN, flops=t * d_model * vocab)
    rows.append(row)

    def dx_ex_f32out():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            d_model, t, vocab, alpha.ctypes.data,
            W.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            beta_zero.ctypes.data, dx_f32.data.ptr,
            cp.cuda.runtime.CUDA_R_32F, d_model,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_TN_fp32out", dx_ex_f32out,
                     flops=t * d_model * vocab)
    rows.append(row)

    # Transposed output orientation: row-major dxT[D,T] = W @ d_logits.T,
    # i.e. column-major C_cm[T,D] (lc=T) = d_logits(kept cm [V,T], opT) @ W
    # (kept cm [V,D], opN). The [D,T] buffer is then transposed into the
    # [T,D] activation gradient, which costs one 2*T*D pass.
    dxT = cp.zeros((d_model, t), dtype=bf16)
    dxT_f32 = cp.zeros((d_model, t), dtype=cp.float32)

    def dx_ex_transposed_out():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            t, d_model, vocab, alpha.ctypes.data,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            W.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            beta_zero.ctypes.data, dxT.data.ptr,
            cp.cuda.runtime.CUDA_R_16BF, t,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_TN_transposed_out_bf16", dx_ex_transposed_out,
                     flops=t * d_model * vocab)
    rows.append(row)
    row, _ = measure("dx_transposed_out_then_copy", lambda: dx_out.__setitem__(
        (slice(None),), dxT.T), flops=0)
    rows.append(row)

    def dx_ex_transposed_out_f32():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            t, d_model, vocab, alpha.ctypes.data,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            W.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            beta_zero.ctypes.data, dxT_f32.data.ptr,
            cp.cuda.runtime.CUDA_R_32F, t,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_TN_transposed_out_fp32",
                     dx_ex_transposed_out_f32, flops=t * d_model * vocab)
    rows.append(row)

    # V-split accumulation: replace one k=32768 GEMM with several k<vocab
    # GEMMs accumulating into an FP32 dx buffer, which tests whether the cost
    # is the long reduction itself or the operand traffic.
    d_logits_T_contig = cp.ascontiguousarray(d_logits.T)

    def dx_ex_NT_contigA():
        # opA=N with W_T_contig (cm [D,V]) and opB=T with d_logits_T_contig.
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
            d_model, t, vocab, alpha.ctypes.data,
            W_T_contig.data.ptr, cp.cuda.runtime.CUDA_R_16BF, d_model,
            d_logits_T_contig.data.ptr, cp.cuda.runtime.CUDA_R_16BF, t,
            beta_zero.ctypes.data, dx_ex.data.ptr,
            cp.cuda.runtime.CUDA_R_16BF, d_model,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_NT_contig_transposed_A",
                     dx_ex_NT_contigA, flops=t * d_model * vocab)
    rows.append(row)

    def dx_ex_TT_contigA():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_T,
            d_model, t, vocab, alpha.ctypes.data,
            W.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            d_logits_T_contig.data.ptr, cp.cuda.runtime.CUDA_R_16BF, t,
            beta_zero.ctypes.data, dx_ex.data.ptr,
            cp.cuda.runtime.CUDA_R_16BF, d_model,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("dx_gemmEx_TT_contig_transposed_A",
                     dx_ex_TT_contigA, flops=t * d_model * vocab)
    rows.append(row)

    def make_dx_split(v_chunk):
        def dx_split():
            for offset in range(0, vocab, v_chunk):
                cublas.gemmEx(
                    handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                    d_model, t, v_chunk,
                    alpha.ctypes.data,
                    W[:, offset:offset + v_chunk].data.ptr,
                    cp.cuda.runtime.CUDA_R_16BF, vocab,
                    d_logits[:, offset:offset + v_chunk].data.ptr,
                    cp.cuda.runtime.CUDA_R_16BF, vocab,
                    (beta_zero if offset == 0 else beta_one).ctypes.data,
                    dx_f32.data.ptr, cp.cuda.runtime.CUDA_R_32F, d_model,
                    cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
                )
        return dx_split

    for v_chunk in (2048, 4096, 8192, 16384):
        row, _ = measure(f"dx_gemmEx_splitV_{v_chunk}_fp32acc",
                         make_dx_split(v_chunk), flops=t * d_model * vocab)
        rows.append(row)
    row, _ = measure("dx_fp32acc_cast_to_bf16",
                     lambda: dx_out.__setitem__((slice(None),), dx_f32), flops=0)
    rows.append(row)

    # --- production tied-head memory layout --------------------------------
    # TiedOutputProjection exposes W.data and W.grad as fresh transpose views
    # of the registered embedding tensors, so physically the weight is E
    # ([Vocab, D] BF16 C-contiguous) and the gradient is E_grad ([Vocab, D]
    # FP32 C-contiguous). The shape-level rows above use the opposite
    # ([D, V] C-contiguous) layout, so these rows repeat the statements with
    # production layouts and both GEMM operand orientations.
    E = cp.ascontiguousarray(W.T)
    E_grad = cp.zeros((vocab, d_model), dtype=cp.float32)
    row, _ = measure("prod_layout_forward_x_at_E_T", lambda: x @ E.T,
                     flops=t * d_model * vocab)
    rows.append(row)
    row, _ = measure("prod_layout_wgrad_today_grad_view_iadd",
                     lambda: E_grad.T.__iadd__(x.T @ d_logits),
                     flops=t * d_model * vocab)
    rows.append(row)
    row, _ = measure("prod_layout_dx_d_logits_at_E",
                     lambda: dx_out.__setitem__((slice(None),), d_logits @ E),
                     flops=t * d_model * vocab)
    rows.append(row)

    def wgrad_into_embedding_grad():
        # Row-major C[d,v] = x.T @ d_logits where the FP32 gradient is stored
        # row-major [Vocab, D]: column-major C_cm[D, Vocab] (lc=D) = x
        # (cm [D, T], opN, lda=D) @ d_logits (cm [Vocab, T], opT, ldb=Vocab).
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
            d_model, vocab, t, alpha.ctypes.data,
            x.data.ptr, cp.cuda.runtime.CUDA_R_16BF, d_model,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            beta_one.ctypes.data, E_grad.data.ptr,
            cp.cuda.runtime.CUDA_R_32F, d_model,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )

    row, _ = measure("prod_layout_wgrad_gemmEx_into_embedding_grad",
                     wgrad_into_embedding_grad, flops=t * d_model * vocab)
    rows.append(row)

    grad_scratch = cp.zeros((d_model, vocab), dtype=cp.float32)

    def wgrad_scratch_then_transposed_add():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
            vocab, d_model, t, alpha.ctypes.data,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            x.data.ptr, cp.cuda.runtime.CUDA_R_16BF, d_model,
            beta_zero.ctypes.data, grad_scratch.data.ptr,
            cp.cuda.runtime.CUDA_R_32F, vocab,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )
        E_grad.T.__iadd__(grad_scratch)

    row, _ = measure("prod_layout_wgrad_scratch_then_transposed_add",
                     wgrad_scratch_then_transposed_add,
                     flops=t * d_model * vocab)
    rows.append(row)

    # The two elementwise add arrangements separately: a strided destination
    # (``E_grad.T += scratch``) versus a strided source (``E_grad += scratch.T``).
    # Only the second one keeps the destination coalesced.
    scratch_T = grad_scratch.T
    row, _ = measure("add_strided_destination_only",
                     lambda: E_grad.T.__iadd__(grad_scratch), flops=0)
    rows.append(row)
    row, _ = measure("add_strided_source_only",
                     lambda: E_grad.__iadd__(scratch_T), flops=0)
    rows.append(row)

    def wgrad_scratch_then_add_into_contiguous():
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
            vocab, d_model, t, alpha.ctypes.data,
            d_logits.data.ptr, cp.cuda.runtime.CUDA_R_16BF, vocab,
            x.data.ptr, cp.cuda.runtime.CUDA_R_16BF, d_model,
            beta_zero.ctypes.data, grad_scratch.data.ptr,
            cp.cuda.runtime.CUDA_R_32F, vocab,
            cp.cuda.runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT,
        )
        E_grad.__iadd__(grad_scratch.T)

    row, _ = measure("prod_layout_wgrad_scratch_then_add_into_contiguous",
                     wgrad_scratch_then_add_into_contiguous,
                     flops=t * d_model * vocab)
    rows.append(row)

    # Numerical check: the accumulating GEMM into the [Vocab, D] FP32 buffer
    # against the exact BF16 operands in FP64.
    E_grad[:] = 0.0
    wgrad_into_embedding_grad()
    cp.cuda.get_current_stream().synchronize()
    oracle64 = (cp.asnumpy(x).astype(np.float64).T
                @ cp.asnumpy(d_logits).astype(np.float64))
    direct64 = cp.asnumpy(E_grad.T).astype(np.float64)
    difference = direct64 - oracle64
    reference_norm = float(np.sqrt((oracle64 * oracle64).sum()))
    prod_layout_numerics = {
        "max_absolute": float(np.abs(difference).max()),
        "relative_l2": float(np.sqrt((difference * difference).sum())
                            / reference_norm),
    }

    # --- staged weight gradient: what production now does ------------------
    from mini_llm.ops import hip_projection_wgrad as hip_module
    from mini_llm.ops.hip_projection_wgrad import accumulate_transposed

    staged_grad_buffer = cp.zeros((vocab, d_model), dtype=cp.float32)
    staged_scratch = cp.empty((d_model, vocab), dtype=cp.float32)

    staged_calls = []
    original_bridge = hip_module.projection_wgrad

    def counting_bridge(activation_arg, gradient_arg, *, out=None,
                        accumulate=False):
        staged_calls.append((out is not None, accumulate))
        return original_bridge(activation_arg, gradient_arg, out=out,
                              accumulate=accumulate)

    hip_module.projection_wgrad = counting_bridge
    try:
        row, _ = measure("staged_wgrad_gemm_into_scratch",
                         lambda: hip_module.projection_wgrad(
                             x, d_logits, out=staged_scratch, accumulate=False),
                         flops=t * d_model * vocab)
        rows.append(row)
        row, _ = measure("staged_transpose_add_into_embedding_grad",
                         lambda: accumulate_transposed(staged_grad_buffer,
                                                       staged_scratch), flops=0)
        rows.append(row)
    finally:
        hip_module.projection_wgrad = original_bridge
    dispatch_proof = {"staged_bridge_calls": staged_calls}

    # Region-level A/B at the production tied-head layout: the same CE pass,
    # input-gradient GEMM and weight-gradient accumulation, staged versus today.
    def region_prod_layout_today():
        logits = logits_template.copy()
        grad_logits = loss_ops.chunked_bf16_cross_entropy_grad_inplace(
            logits, targets, normalizer_count=float(t), grad_scale=1.0
        )
        dx = cp.empty_like(x)
        E_grad.T.__iadd__(x.T @ grad_logits)
        dx[:, :] = grad_logits @ E
        del logits, grad_logits

    def region_prod_layout_staged():
        logits = logits_template.copy()
        grad_logits = loss_ops.chunked_bf16_cross_entropy_grad_inplace(
            logits, targets, normalizer_count=float(t), grad_scale=1.0
        )
        dx = cp.empty_like(x)
        hip_module.projection_wgrad(x, grad_logits, out=staged_scratch,
                                   accumulate=False)
        accumulate_transposed(staged_grad_buffer, staged_scratch)
        dx[:, :] = grad_logits @ E
        del logits, grad_logits

    row, _ = measure("region_prod_layout_today", region_prod_layout_today)
    rows.append(row)
    row, _ = measure("region_prod_layout_staged", region_prod_layout_staged)
    rows.append(row)

    # The staged and today forms must produce the same gradient buffer shape,
    # and the staged one must stay finite.
    staged_grad_buffer[:] = 0.0
    E_grad[:] = 0.0
    region_prod_layout_staged()
    cp.cuda.get_current_stream().synchronize()
    staged_numerics = {
        "shape_matches": bool(staged_grad_buffer.shape == E_grad.shape),
        "all_finite": bool(cp.all(cp.isfinite(staged_grad_buffer))),
    }

    # --- faithful replica of the whole production region, event-timed --------
    full_W_grad = cp.zeros((d_model, vocab), dtype=cp.float32)

    def prod_region_replica():
        logits = logits_template.copy()
        grad_logits = loss_ops.chunked_bf16_cross_entropy_grad_inplace(
            logits, targets, normalizer_count=float(t), grad_scale=1.0
        )
        dx = cp.empty_like(x)
        full_W_grad.__iadd__(x.T @ grad_logits)
        dx[:, :] = grad_logits @ W.T
        del logits, grad_logits

    row, _ = measure("region_replica_ce_wgrad_dx", prod_region_replica)
    rows.append(row)

    # --- correctness cross-checks (outside timing) ---------------------------
    W_grad_ref[:] = 0.0
    wgrad_into()
    cp.cuda.get_current_stream().synchronize()
    helper_ref = projection_wgrad(x, d_logits)
    delta = np.abs(
        cp.asnumpy(W_grad_ref).astype(np.float64)
        - cp.asnumpy(helper_ref).astype(np.float64)
    )
    rel = float(np.linalg.norm(delta) / max(np.linalg.norm(helper_ref), 1e-30))
    production = W_grad.copy()
    W_grad[:] = 0.0
    W_grad += x.T @ d_logits
    cp.cuda.get_current_stream().synchronize()
    prod_vs_helper = np.abs(
        cp.asnumpy(production).astype(np.float64)
        - cp.asnumpy(helper_ref).astype(np.float64)
    )
    rel_prod = float(
        np.linalg.norm(prod_vs_helper)
        / max(np.linalg.norm(helper_ref), 1e-30)
    )
    # Re-do the production write immediately before comparing, because later
    # candidate measurements reuse the same dx_out buffer.
    dx_out[:, :] = raw_dx
    cp.cuda.get_current_stream().synchronize()
    dx_ref = cp.asnumpy(raw_dx).astype(np.float64)
    dx_written = cp.asnumpy(dx_out).astype(np.float64)
    dx_match = bool(np.array_equal(dx_ref, dx_written))

    dx_ex_TN()
    dx_ex_NN()
    cp.cuda.get_current_stream().synchronize()
    dx_ex_TN_ref = cp.asnumpy(dx_ex).astype(np.float64)
    dx_TN_err = float(np.abs(dx_ex_TN_ref - dx_ref).max())
    dx_ex_NN()
    cp.cuda.get_current_stream().synchronize()
    dx_ex_NN_ref = cp.asnumpy(dx_ex).astype(np.float64)
    dx_NN_err = float(np.abs(dx_ex_NN_ref - dx_ref).max())

    dx_ex_f32out()
    cp.cuda.get_current_stream().synchronize()
    dx_f32_err = float(
        np.abs(cp.asnumpy(dx_f32).astype(np.float64) - dx_ref).max()
    )
    cp.cuda.get_current_stream().synchronize()
    transposed_host = np.ascontiguousarray(
        cp.asnumpy(dxT).astype(np.float64).T
    )
    dx_transposed_err = float(np.abs(transposed_host - dx_ref).max())

    dx_ex_TT_contigA()
    cp.cuda.get_current_stream().synchronize()
    dx_TT_err = float(np.abs(cp.asnumpy(dx_ex).astype(np.float64) - dx_ref).max())
    make_dx_split(4096)()
    cp.cuda.get_current_stream().synchronize()
    dx_split_err = float(np.abs(cp.asnumpy(dx_f32).astype(np.float64) - dx_ref).max())

    report = {
        "shapes": {"tokens": t, "d_model": d_model, "vocab": vocab},
        "device": info,
        "prod_layout_numerics": prod_layout_numerics,
        "staged_numerics": staged_numerics,
        "dispatch_proof": dispatch_proof,
        "counts": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "iterations": args.iterations,
        },
        "rows": rows,
        "correctness": {
            "beta1_vs_helper_rel_l2": rel,
            "beta1_vs_helper_max_abs": float(delta.max()),
            "production_vs_helper_rel_l2": rel_prod,
            "production_vs_helper_max_abs": float(prod_vs_helper.max()),
            "dx_write_matches_dot": dx_match,
            "dx_TN_max_abs_vs_dot": dx_TN_err,
            "dx_NN_max_abs_vs_dot": dx_NN_err,
            "dx_TN_fp32out_max_abs_vs_dot": dx_f32_err,
            "dx_TT_contig_transposed_A_max_abs_vs_dot": dx_TT_err,
            "dx_transposed_out_max_abs_vs_dot": dx_transposed_err,
            "dx_splitV_max_abs_vs_dot": dx_split_err,
        },
    }
    print(json.dumps(report["correctness"], indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
