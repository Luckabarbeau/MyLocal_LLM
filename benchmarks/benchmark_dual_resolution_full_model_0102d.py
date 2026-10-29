#!/usr/bin/env python3
"""0102D: real decoder/LM-head forward+backward, checkpoint and memory smoke.

Not a synthetic single-attention-layer benchmark. This exercises every decoder
layer, the final norm, chunked next-token cross entropy and all model gradients.
For long contexts, a *tail-scored* objective is used to keep vocabulary-head
compute bounded; token representations still traverse every decoder layer.
Memory figures are sampled diagnostics, not an exact device-allocation peak.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import median
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--model-context', type=int, default=None,
                   help='configured maximum context; defaults to --context. Use a larger '
                        'block-aligned value to exercise shorter production buckets')
    p.add_argument('--architecture', choices=('dual-resolution', 'matrix-pool'),
                   default='dual-resolution')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--layers', type=int, default=20)
    p.add_argument('--compression-dim', type=int, choices=(256, 512, 1024), default=512)
    p.add_argument('--disable-global', action='store_true',
                   help='matrix-pool ablation: execute the identical mixed-window trunk '
                        'without pooling/global retrieval')
    p.add_argument('--score-last-k', type=int, default=128,
                   help='number of final next-token predictions supervised (all context tokens still processed)')
    p.add_argument('--head-chunk-tokens', type=int, default=64)
    p.add_argument('--activation-checkpoint', choices=('block', 'none'), default='block')
    p.add_argument('--local-kernel', choices=('cupy', 'triton'), default='triton')
    p.add_argument('--global-kernel', choices=('cupy', 'triton'), default='triton')
    p.add_argument('--compression-kernel', choices=('cupy', 'triton'), default='triton')
    p.add_argument('--query-chunk', type=int, default=512)
    p.add_argument('--compression-block-chunk', type=int, default=64)
    p.add_argument('--seed', type=int, default=102)
    p.add_argument('--warmup-iterations', type=int, default=0,
                   help='complete untimed forward+loss+backward iterations to warm packed caches, '
                        'CuPy and Triton kernels; avoid judging a cold JIT run')
    p.add_argument('--repeat-iterations', type=int, default=1,
                   help='independent measured forward+loss+backward iterations; '
                        'report medians and per-iteration timings')
    p.add_argument('--runtime-profile',
                   choices=('auto', 'reference', 'gpu-fast', 'consumer-gpu'),
                   default='auto',
                   help='apply the same runtime defaults as train_model; explicit '
                        'MINI_LLM_* environment values still win')
    p.add_argument('--profile-global-branch', action='store_true',
                   help='record deferred CUDA-event and host timings for Model B '
                        'pooling, global projections, attention, and integration')
    p.add_argument('--alternate-fused-pool-backward', action='store_true',
                   help='alternate the Model B fused pooling backward off/on within '
                        'one initialized model and report per-mode timing medians')
    p.add_argument('--alternate-bounded-global', action='store_true',
                   help='alternate bounded Model B global attention off/on within '
                        'one initialized model and report per-mode timing medians')
    p.add_argument('--optimizer-step', action='store_true',
                   help='also construct a real AdamW optimizer and apply one update; '
                        'use MINI_LLM_OPTIMIZER_OFFLOAD=full for a 16GB GPU')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--tiny', action='store_true',
                   help='small 2-layer/32-wide CPU/GPU contract model for fast tests')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    if args.warmup_iterations < 0 or args.repeat_iterations < 1:
        p.error('warmup-iterations >= 0 and repeat-iterations >= 1 required')
    if args.context < 2 or args.batch_size < 1 or args.layers < 1:
        p.error('context >= 2, batch-size >= 1, layers >= 1 required')
    if args.score_last_k < 1 or args.head_chunk_tokens < 1 or args.query_chunk < 1:
        p.error('score-last-k, head-chunk-tokens, query-chunk must be positive')
    if args.compression_block_chunk < 1:
        p.error('compression-block-chunk must be positive')
    if args.tiny and args.context > 128:
        p.error('tiny mode requires context <= 128')
    if args.alternate_fused_pool_backward and args.architecture != 'matrix-pool':
        p.error('--alternate-fused-pool-backward requires --architecture matrix-pool')
    if args.alternate_bounded_global and args.architecture != 'matrix-pool':
        p.error('--alternate-bounded-global requires --architecture matrix-pool')
    if args.alternate_fused_pool_backward and args.alternate_bounded_global:
        p.error('only one alternating Model B ablation may be selected')
    model_context = args.context if args.model_context is None else args.model_context
    if model_context > 65536:
        p.error('model context exceeds configured maximum (65536)')
    if args.context > model_context:
        p.error('--context must not exceed --model-context')
    return args


def build_config(args, *, backend_name):
    from mini_llm.config import ModelConfig
    model_context = args.context if args.model_context is None else args.model_context
    base = (ModelConfig.dual_resolution_matrix_pool_1024_20l()
            if args.architecture == 'matrix-pool'
            else ModelConfig.dual_resolution_1024_20l())
    if args.tiny:
        dr = replace(base.dual_resolution, block_size=8, local_window=16,
                     long_local_window=(32 if args.architecture == 'matrix-pool' else 512),
                     global_head_dim=(8 if args.architecture == 'matrix-pool' else 64),
                     compression_dim=16, compression_heads=2,
                     max_blocks=16, global_query_chunk=min(16, args.query_chunk),
                     compression_block_chunk=min(16, args.compression_block_chunk),
                     global_enabled=(base.dual_resolution.global_enabled
                                     and not args.disable_global))
        return replace(base, context_length=model_context, n_layers=args.layers,
                       tokenizer_vocab_size=47, d_model=32, d_ff=64, n_q_heads=4,
                       n_kv_heads=2, d_head=8, dual_resolution=dr,
                       dtype='float32')
    dr = replace(base.dual_resolution, compression_dim=args.compression_dim,
                 compression_block_chunk=args.compression_block_chunk,
                 global_query_chunk=args.query_chunk,
                 global_enabled=(base.dual_resolution.global_enabled
                                 and not args.disable_global))
    return replace(base, context_length=model_context, n_layers=args.layers,
                   dual_resolution=dr,
                   dtype='bfloat16' if backend_name == 'cupy' else 'float32')


def _finite_array(x, xp):
    return bool(xp.all(xp.isfinite(x.astype('float32', copy=False))).item())


def run(args):
    # Set operator selection *before* constructing model objects.
    os.environ['MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL'] = args.local_kernel
    os.environ['MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL'] = args.global_kernel
    os.environ['MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL'] = args.compression_kernel
    if args.profile_global_branch:
        os.environ['MINI_LLM_MATRIX_POOL_DETAIL_PROFILE'] = '1'
    from mini_llm.backend import BACKEND_NAME, xp, synchronize
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
    from mini_llm.performance_profiler import (
        PERFORMANCE_PROFILER, configure_performance_profiler,
    )

    if BACKEND_NAME != 'cupy' and not args.tiny:
        raise RuntimeError('Production-width smoke requires MINI_LLM_BACKEND=cupy; '
                           'use --tiny for the NumPy reference test')
    if BACKEND_NAME != 'cupy' and any(x == 'triton' for x in (
            args.local_kernel, args.global_kernel, args.compression_kernel)):
        # CPU reference mode must never attempt to import/launch CUDA kernels.
        os.environ['MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL'] = 'cupy'
        os.environ['MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL'] = 'cupy'
        os.environ['MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL'] = 'cupy'
    config = build_config(args, backend_name=BACKEND_NAME)
    from mini_llm.runtime_defaults import apply_runtime_defaults
    runtime_profile = apply_runtime_defaults(
        args.runtime_profile, backend_name=BACKEND_NAME, config=config,
        training=True, precision='bf16-mixed' if BACKEND_NAME == 'cupy' else 'float32',
    )
    if args.context > config.dual_resolution.block_size * config.dual_resolution.max_blocks:
        raise ValueError('context exceeds dual-resolution block capacity')
    if args.context <= config.dual_resolution.block_size:
        raise ValueError('context must extend beyond the first complete compression block, '
                         'otherwise the global/compressor pathway receives no LM gradient')
    records = []
    start = time.perf_counter()
    def sample(label):
        synchronize()
        point = {'stage': label, 'elapsed_s': round(time.perf_counter()-start, 6)}
        if BACKEND_NAME == 'cupy':
            pool = xp.get_default_memory_pool()
            free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
            point.update(pool_used_mib=round(pool.used_bytes()/1048576, 2),
                         pool_reserved_mib=round(pool.total_bytes()/1048576, 2),
                         device_used_sample_mib=round((total_bytes-free_bytes)/1048576, 2),
                         device_total_mib=round(total_bytes/1048576, 2))
        records.append(point)
        return point

    sample('start')
    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    actual = sum(int(p.size) for p in model.parameters())
    expected = config.estimated_parameter_count()
    if actual != expected:
        raise AssertionError(f'parameter accounting mismatch: {actual} != {expected}')
    sample('model_initialized')

    rng = np.random.default_rng(args.seed + 1)
    # No EOS in this synthetic document. Targets are *shifted*, not equal to inputs.
    eos_id = int(config.vocab_size - 1)
    raw = rng.integers(0, eos_id, size=(args.batch_size, args.context + 1),
                       dtype=np.int64)
    inputs = xp.asarray(raw[:, :-1], dtype=xp.int64)
    k = min(args.score_last_k, args.context)
    targets = xp.asarray(raw[:, 1:][:, -k:], dtype=xp.int64)
    metadata = build_dual_resolution_metadata(raw[:, :-1], eos_token_id=eos_id,
                                              block_size=config.dual_resolution.block_size)
    # Use the same public APIs as ExtendedTrainer, with a bounded LM head.
    optimizer = None
    if args.optimizer_step:
        from mini_llm.optim import AdamW
        optimizer = AdamW(model.optimization_parameters(), lr=args.lr)
        sample('optimizer_initialized')
    # The 0102D default remains one cold pass for backward compatibility.
    # For performance comparisons use --warmup-iterations 1 --repeat-iterations 3;
    # patch 0102H changes the QKV dtype presented to Triton, so a *cold*
    # forward/backward may include expensive compilation for new specializations.
    if os.environ.get('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '1' if BACKEND_NAME == 'cupy' else '0').strip().lower() in ('1', 'true', 'yes'):
        model.refresh_compute_buffers()
    measured = []
    loss_value = None
    configure_performance_profiler(False, reset=args.profile_global_branch)
    for iteration in range(args.warmup_iterations + args.repeat_iterations):
        record = iteration >= args.warmup_iterations
        pool_backward_mode = None
        bounded_global_mode = None
        if args.alternate_fused_pool_backward:
            pool_backward_mode = iteration % 2
            os.environ['MINI_LLM_MATRIX_POOL_FUSED_BACKWARD'] = str(pool_backward_mode)
        elif args.alternate_bounded_global:
            bounded_global_mode = iteration % 2
            os.environ['MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED'] = str(
                bounded_global_mode
            )
        if args.profile_global_branch and record and not measured:
            configure_performance_profiler(True, reset=True)
        model.zero_grad()
        synchronize()
        if record and not measured:
            sample('before_forward')
        t0 = time.perf_counter()
        head, cache = model.forward_body(
            inputs, activation_checkpoint=args.activation_checkpoint == 'block',
            dual_resolution_metadata=metadata,
        )
        synchronize()
        forward_ms = (time.perf_counter() - t0) * 1000
        if record and not measured:
            sample('after_forward')
        t0 = time.perf_counter()
        loss, loss_cache = model.chunked_lm_head_loss_forward(
            head, targets, chunk_tokens=args.head_chunk_tokens,
            target_slice=(args.context-k, args.context), return_device_loss=True,
        )
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            raise AssertionError('nonfinite loss')
        synchronize()
        loss_ms = (time.perf_counter() - t0) * 1000
        del head
        if record and not measured:
            sample('after_chunked_loss')
        t0 = time.perf_counter()
        dx = model.chunked_lm_head_backward(loss_cache)
        model.backward_body(dx, cache)
        synchronize()
        backward_ms = (time.perf_counter() - t0) * 1000
        del dx, cache, loss_cache
        if record and not measured:
            sample('after_backward')
        if record:
            measured.append({
                'forward_ms': round(forward_ms, 5),
                'loss_ms': round(loss_ms, 5),
                'backward_ms': round(backward_ms, 5),
                'total_ms': round(forward_ms + loss_ms + backward_ms, 5),
                **({'fused_pool_backward': bool(pool_backward_mode)}
                   if pool_backward_mode is not None else {}),
                **({'bounded_global': bool(bounded_global_mode)}
                   if bounded_global_mode is not None else {}),
            })
    synchronize()
    profile_rows = PERFORMANCE_PROFILER.rows() if args.profile_global_branch else []
    configure_performance_profiler(False)
    forward_ms = median(row['forward_ms'] for row in measured)
    loss_ms = median(row['loss_ms'] for row in measured)
    backward_ms = median(row['backward_ms'] for row in measured)
    median_total_ms = median(row['total_ms'] for row in measured)
    def alternating_summary(field):
        summary = {}
        for mode in (False, True):
            selected = [row for row in measured if row[field] is mode]
            summary[str(int(mode))] = {
                'iterations': len(selected),
                'forward_ms': median(row['forward_ms'] for row in selected),
                'loss_ms': median(row['loss_ms'] for row in selected),
                'backward_ms': median(row['backward_ms'] for row in selected),
                'total_ms': median(row['total_ms'] for row in selected),
                'supervised_tokens_per_second': args.context * args.batch_size * 1000.0
                    / median(row['total_ms'] for row in selected),
            }
        return summary

    alternating_pool_backward = (
        alternating_summary('fused_pool_backward')
        if args.alternate_fused_pool_backward else {}
    )
    alternating_bounded_global = (
        alternating_summary('bounded_global')
        if args.alternate_bounded_global else {}
    )
    # Check representative parameters from the global/compressor/FFN paths,
    # rather than allocating a second flattened copy of every parameter.
    grads = {}
    for label, fragments in {
        'compress': ('.compress.', '.matrix_pool.'),
        'global': ('.global.',),
        'ffn': ('.dense_ffn.', '.moe.'),
        'embedding': ('embedding.',),
    }.items():
        matches = [p for p in model.parameters() if any(frag in p.name for frag in fragments)]
        grads[label] = {'parameters': len(matches),
                        'nonzero': any(bool(xp.any(p.grad != 0).item()) for p in matches),
                        'finite': all(_finite_array(p.grad, xp) for p in matches)}
    if not all(stat['finite'] for stat in grads.values()):
        raise AssertionError('nonfinite gradients: ' + repr(grads))
    if not grads['ffn']['nonzero']:
        raise AssertionError('no gradient reaches the dense SwiGLU FFN')
    if config.dual_resolution.global_enabled:
        if not grads['compress']['nonzero'] or not grads['global']['nonzero']:
            raise AssertionError('no gradient reaches global/compression branch; '
                                 'score predictions after at least one completed block')
    update_ms = None
    if optimizer is not None:
        t0 = time.perf_counter()
        optimizer.step()
        model.refresh_compute_buffers()
        synchronize()
        update_ms = (time.perf_counter()-t0)*1000
        sample('after_optimizer_update')
    result = {
        'architecture': ('dual-resolution-matrix-pool-1024-20l'
                         if args.architecture == 'matrix-pool'
                         else 'dual-resolution-1024-20l'),
        'backend': BACKEND_NAME, 'dtype': str(config.dtype),
        'runtime_profile': {
            'requested': runtime_profile.requested,
            'resolved': runtime_profile.resolved,
            'applied': dict(runtime_profile.applied),
        },
        'layers': config.n_layers, 'context': args.context,
        'model_context': config.context_length, 'batch_size': args.batch_size,
        'compression_dim': config.dual_resolution.compression_dim,
        'global_enabled': config.dual_resolution.global_enabled,
        'matrix_pool_grad_scatter_inplace': os.environ.get(
            'MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE', '0'
        ).strip().lower() in ('1', 'true', 'yes'),
        'matrix_pool_fused_forward': os.environ.get(
            'MINI_LLM_MATRIX_POOL_FUSED_FORWARD', '0'
        ).strip().lower() in ('1', 'true', 'yes'),
        'matrix_pool_fused_backward': os.environ.get(
            'MINI_LLM_MATRIX_POOL_FUSED_BACKWARD', '1'
        ).strip().lower() in ('1', 'true', 'yes'),
        'activation_checkpoint': args.activation_checkpoint,
        'local_kernel': os.environ['MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL'],
        'global_kernel': os.environ['MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL'],
        'bounded_global': os.environ.get(
            'MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED', '1'
        ).strip().lower() in ('1', 'true', 'yes', 'on'),
        'compression_kernel': os.environ['MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL'],
        'fast_projections': os.environ.get('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '1' if BACKEND_NAME == 'cupy' else '0').strip().lower() in ('1', 'true', 'yes'),
        'bf16_projection_grads': os.environ.get('MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS', '1').strip().lower() in ('1', 'true', 'yes'),
        'bf16_output_dx': os.environ.get('MINI_LLM_DUAL_RESOLUTION_BF16_OUTPUT_DX', '0').strip().lower() in ('1', 'true', 'yes'),
        'bf16_compressor_qkv': os.environ.get('MINI_LLM_DUAL_RESOLUTION_BF16_COMPRESSOR_QKV', '1').strip().lower() in ('1', 'true', 'yes'),
        'fused_pooling': os.environ.get('MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING', '0').strip().lower() in ('1', 'true', 'yes'),
        'fused_compressor_rmsnorm': os.environ.get('MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM', '0').strip().lower() in ('1', 'true', 'yes'),
        'fused_swiglu': os.environ.get('MINI_LLM_FUSED_SWIGLU', '0').strip().lower() in ('1', 'true', 'yes'),
        'swiglu_vec2': os.environ.get('MINI_LLM_SWIGLU_VEC2', '0').strip().lower() in ('1', 'true', 'yes'),
        'fused_expert_gemm': os.environ.get('MINI_LLM_FUSED_EXPERT_GEMM', '1').strip().lower() in ('1', 'true', 'yes'),
        'runtime_lm_head_chunk_tokens': int(os.environ.get('MINI_LLM_LM_HEAD_CHUNK_TOKENS', '0')),
        'parameters': actual, 'scored_token_rows': int(k*args.batch_size),
        'loss': loss_value,
        'warmup_iterations': args.warmup_iterations,
        'repeat_iterations': args.repeat_iterations,
        'iteration_timings_ms': measured,
        'global_branch_profile': profile_rows,
        'alternating_fused_pool_backward': alternating_pool_backward,
        'alternating_bounded_global': alternating_bounded_global,
        'head_chunk_tokens': args.head_chunk_tokens,
        'timing_note': 'Medians of full forward+LM loss+backward without optimizer; '
                       'first measured stage also sampled for memory',
        'forward_ms': forward_ms, 'loss_ms': loss_ms,
        'backward_ms': backward_ms, 'optimizer_ms': update_ms,
        'full_model_forward_backward_tok_per_s': args.context*args.batch_size/
            (median_total_ms/1000),
        'gradients': grads, 'memory_samples': records,
        'memory_note': 'Samples after synchronization; not a true peak within an operation.',
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        # Preserve the original exception and do not claim an OOM was recovered.
        print(f'0102D full-model smoke FAILED: {type(exc).__name__}: {exc}',
              file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
