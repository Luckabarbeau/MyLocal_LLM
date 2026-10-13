"""0079I: real packed-text, full-vocabulary refinement research trainer.

Examples (the SAME mixed shards/tokenizer as train_model.py):
  MINI_LLM_BACKEND=cupy python -m examples.train_predictive_refinement_text \\
      --mixed-shard-root ../pretraining_data/mixed_shards --steps 500 \\
      --checkpoint-dir checkpoints/refinement-1layer

This is a small 0079H experiment, NOT the production 4K/64K routed trainer.
Uses document-first EOS-bounded samples and measures UNIQUE supervised tok/s.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, asnumpy, synchronize, validate_bfloat16_backend
from mini_llm.config import ModelConfig
from mini_llm.predictive_refinement.scale_config import WIDE_TRUNK_NAME, wide_static_trunk_config
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.tokenizer.tokenizer import FastBPETokenizer
from mini_llm.predictive_refinement.real_data import PackedRefinementDocuments, sha256
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.predictive_refinement.checkpoint import save_refinement_checkpoint, load_refinement_checkpoint
from mini_llm.diagnostics.source_metrics import (
    empty_record, merge_record, row as source_metrics_row, write_source_rows,
)
from mini_llm.predictive_refinement.gradient_diagnostics import (
    TokenWeightedRefinementGradientMonitor,
)
from mini_llm.predictive_refinement.telemetry import (
    empty_counters, merge_counters, observe_microbatch, rates,
    source_summary, memory_snapshot, write_scalar_csv,
)
from mini_llm.performance_profiler import (
    configure_performance_profiler, performance_scope, performance_report,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--mixed-shard-root', type=Path)
    inp.add_argument('--shard-dir', type=Path)
    p.add_argument('--tokenizer-path', type=Path, help='defaults to the root tokenizer.json')
    p.add_argument('--token-dtype', choices=('uint16', 'uint32'), default='uint16')
    p.add_argument('--steps', type=int, default=500, help='TOTAL optimizer steps (also on resume)')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--prefix-length', type=int, default=32)
    p.add_argument('--block-size', type=int, default=4)
    p.add_argument('--window-blocks', type=int, default=8)
    p.add_argument('--refinement-group-size', type=int, default=2)
    p.add_argument('--max-window-tokens', type=int, default=256,
                   help='explicit causal window safety cap, up to 4096; default preserves 0079M')
    p.add_argument('--compact-recurrent-tape', action='store_true',
                   help='0079P1 compact CE rows and rolling feedback-gradient buffer')
    p.add_argument('--resident-history', action='store_true',
                   help='0079P2 shared causal K/V storage across refinement passes')
    p.add_argument('--compact-history', action='store_true',
                   help='pack static causal K/V selections across independent target blocks')
    p.add_argument('--max-history-keys', type=int, default=4096,
                   help='maximum distinct selected historical keys per provisional row')
    p.add_argument('--compact-active-blocks', action='store_true',
                   help='pack only nonempty document/block pairs per refinement pass; '
                        'opt-in until profiled on CUDA')
    p.add_argument('--max-active-queries', type=int, default=0,
                   help='optional hard B*G*K per-pass memory guard; required for groups >16')
    p.add_argument('--model-preset', choices=('custom', WIDE_TRUNK_NAME), default='custom',
                   help='500M static-trunk probe uses real MoE shape, not 64K memory/retrieval')
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-q-heads', type=int, default=4)
    p.add_argument('--n-kv-heads', type=int, default=2)
    p.add_argument('--d-ff', type=int, default=256)
    p.add_argument('--experts', type=int, default=2)
    p.add_argument('--top-k', type=int, default=1)
    p.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--initial-loss-weight', type=float, default=0.1)
    p.add_argument('--feedback-mode', choices=('full', 'unknown'), default='full',
                   help='unknown is a matched-pass, no-future-feedback ablation')
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--val-every', type=int, default=100)
    p.add_argument('--val-batches', type=int, default=32,
                   help='held-out microbatches; use >=32 for less noisy estimates')
    p.add_argument('--validation-mode', choices=('fixed', 'rolling'), default='fixed',
                   help='fixed uses the same independent held-out samples each evaluation')
    p.add_argument('--profile-after-step', type=int, default=10,
                   help='start optional synchronous profiling at this 1-based optimizer step')
    p.add_argument('--eval-only', action='store_true',
                   help='evaluate a --resume checkpoint without modifying weights')
    p.add_argument('--source-metrics', action=argparse.BooleanOptionalAction, default=True,
                   help='observational token-weighted source CE and document statistics')
    p.add_argument('--source-metrics-csv', type=Path, default=None,
                   help='defaults to <checkpoint-dir>/refinement_source_metrics.csv')
    p.add_argument('--gpu-cache-audit-interval', type=int, default=0,
                   help='optional non-invasive inventory of retained GPU caches')
    p.add_argument('--gpu-memory-interval', type=int, default=0,
                   help='optional allocator/device snapshots every N steps; 0 means log steps only')
    p.add_argument('--adamw-update-monitor', action='store_true',
                   help='observational token-weighted AdamW direction comparisons')
    p.add_argument('--adamw-update-output', type=Path, default=None)
    p.add_argument('--gradient-noise-monitor', action='store_true',
                   help='observational token-weighted disjoint-half gradients')
    p.add_argument('--gradient-noise-interval', type=int, default=100)
    p.add_argument('--gradient-noise-accum-points', default='auto',
                   help='even accumulation lengths, e.g. 2,4,8; auto selects valid')
    p.add_argument('--gradient-noise-parameter-groups',
                   default='attention,moe,moe_router,head')
    p.add_argument('--gradient-noise-output', type=Path, default=None)
    p.add_argument('--gradient-group-interval', type=int, default=0,
                   help='optional per-parameter-family gradient norms every N updates')
    p.add_argument('--numerical-debug', action='store_true',
                   help='check all parameter gradients for nonfinite values at every step')
    p.add_argument('--metrics-csv', type=Path, default=None,
                   help='legacy-compatible scalar CSV; defaults to checkpoint dir')
    p.add_argument('--metrics-jsonl', type=Path, default=None,
                   help='optional append-only log of training and validation metrics')
    p.add_argument('--profile-steps', type=int, default=0,
                   help='number of optimizer steps to profile; 0 disables sync overhead')
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--no-save', action='store_true', help='skip large checkpoints in sizing smoke tests')
    p.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/refinement-1layer-text'))
    p.add_argument('--resume', type=Path, help='resume this exact refinement checkpoint and RNG stream')
    p.add_argument('--generate', type=int, default=32)
    p.add_argument('--prompt', type=str, default='The ')
    args = p.parse_args(argv)
    max_window = args.prefix_length + args.window_blocks * args.block_size
    if min(args.batch_size, args.grad_accum, args.prefix_length, args.block_size,
           args.window_blocks, args.refinement_group_size, args.layers,
           args.d_model, args.d_ff, args.experts, args.n_q_heads, args.n_kv_heads) < 1:
        p.error('all model, batch, prefix, and block dimensions must be positive')
    if not 1 <= args.block_size <= 16 or not 1 <= args.refinement_group_size <= 64:
        p.error('block-size must be in [1,16] and refinement-group-size in [1,64]')
    if args.max_active_queries < 0:
        p.error('max-active-queries must be >=0')
    if args.refinement_group_size > 16 and args.max_active_queries == 0:
        p.error('group sizes >16 require explicit --max-active-queries safety guard')
    if not 1 <= args.max_window_tokens <= 4096 or max_window > args.max_window_tokens:
        p.error('window exceeds --max-window-tokens (supported cap: 4096)')
    if args.max_history_keys <= 0:
        p.error('--max-history-keys must be positive')
    if args.resident_history and not args.compact_history:
        p.error('--resident-history requires --compact-history')
    if args.compact_history and not args.compact_active_blocks:
        p.error('--compact-history requires --compact-active-blocks')
    if max_window > 256 and (not args.compact_history or args.max_active_queries <= 0):
        p.error('windows >256 require --compact-history, --compact-active-blocks, '
                'and explicit --max-active-queries')
    if args.d_model % args.n_q_heads or args.n_q_heads % args.n_kv_heads:
        p.error('d-model must divide by n-q-heads; n-q-heads by n-kv-heads')
    if args.steps < 0 or args.val_batches < 1 or args.log_every < 1 or args.save_every < 1 or args.val_every < 1 or args.profile_steps < 0 or args.profile_after_step < 1 or args.gpu_memory_interval < 0 or args.gradient_group_interval < 0 or args.gradient_noise_interval < 1 or args.gpu_cache_audit_interval < 0:
        p.error('invalid step/log/validation configuration')
    if (args.gradient_noise_monitor or args.adamw_update_monitor) and args.grad_accum < 2:
        p.error('--gradient-noise-monitor requires --grad-accum >= 2')
    if args.eval_only and args.resume is None:
        p.error('--eval-only requires --resume')
    if not math.isfinite(args.clip_grad_norm) or args.clip_grad_norm <= 0:
        p.error('clip-grad-norm must be finite and positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error('learning-rate must be finite and positive')
    return args


def _load_tokenizer(path):
    token = FastBPETokenizer.load(str(path))
    eos = token._tokenizer.token_to_id(token.eos_token)
    if eos is None:
        raise ValueError('BPE tokenizer is missing EOS token')
    return token, int(eos)


def _provenance(args, tokenizer_path, train_data, val_data):
    return {
        'experiment': '0079I_full_vocab_real_text',
        'tokenizer_sha256': sha256(tokenizer_path),
        'tokenizer_path': str(tokenizer_path.resolve()),
        'train_data': train_data.provenance(),
        'val_data': val_data.provenance(),
        'seed': args.seed,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'prefix_length': args.prefix_length,
        'block_size': args.block_size,
        'window_blocks': args.window_blocks,
        'group_size': args.refinement_group_size,
        **({'compact_active_blocks': True} if args.compact_active_blocks else {}),
        **({'max_window_tokens': args.max_window_tokens}
           if args.max_window_tokens != 256 else {}),
        **({'resident_history': True} if args.resident_history else {}),
        **({'compact_history': True, 'max_history_keys': args.max_history_keys}
           if args.compact_history else {}),
        **({'max_active_queries': args.max_active_queries}
           if args.max_active_queries else {}),
        'initial_loss_weight': args.initial_loss_weight,
        **({'model_preset': args.model_preset} if args.model_preset != 'custom' else {}),
        **({'feedback_mode': args.feedback_mode} if args.feedback_mode != 'full' else {}),
    }


def _sample(data, rng, args):
    kwargs = dict(batch_size=args.batch_size, prefix_length=args.prefix_length,
                  block_size=args.block_size, window_blocks=args.window_blocks)
    if isinstance(data, PackedRefinementDocuments) and args.source_metrics:
        return data.sample(rng, **kwargs, return_observations=True)
    return data.sample(rng, **kwargs)


def _unpack_sample(sample):
    if len(sample) == 4:
        tokens, valid, prefix, source_counts = sample
        return tokens, valid, prefix, source_counts, None
    if len(sample) == 5:
        return sample
    raise ValueError('unexpected packed-refinement sampler output')


def _train_one(ref, data, rng, args, *, collect_position=False, gradient_monitor=None):
    """Same global token-weighted objective as 0079I; diagnostics are read-only."""
    timers = {'sampling_wall_ms': 0.0, 'refinement_forward_backward_host_ms': 0.0}
    start = time.perf_counter()
    with performance_scope("refinement.data_sampling"):
        minibatches = [_unpack_sample(_sample(data, rng, args))
                        for _ in range(args.grad_accum)]
    timers['sampling_wall_ms'] = (time.perf_counter() - start) * 1000
    unique_tokens = sum(int(mask[:, prefix:].sum()) for _, mask, prefix, _, _ in minibatches)
    if unique_tokens <= 0:
        raise ValueError('no supervised tokens in sampled microbatches')
    ref.zero_grad()
    losses = {'commitment_ce': 0., 'initial_ce': 0., 'total_loss': 0.}
    counters = empty_counters()
    k = args.block_size
    pos = ({'counts': np.zeros(k, np.int64),
            'initial_sum': np.zeros(k, np.float64),
            'commit_sum': np.zeros(k, np.float64)}
           if collect_position else None)
    observed_tokens = 0
    for micro_index, (tokens, valid, prefix, counts, observations) in enumerate(minibatches, 1):
        t0 = time.perf_counter()
        result = ref.train_window(
            xp.asarray(tokens, dtype=xp.int64), prefix_length=prefix,
            valid_token_mask=xp.asarray(valid), group_size=args.refinement_group_size,
            loss_normalization_tokens=unique_tokens,
            record_row_ce=args.source_metrics,
            compact_active_blocks=args.compact_active_blocks,
            max_active_queries=args.max_active_queries,
            compact_history=args.compact_history,
            resident_history=args.resident_history,
            max_history_keys=args.max_history_keys,
            max_window_tokens=args.max_window_tokens,
        )
        for key in losses:
            losses[key] += float(result[key])
        timers['refinement_forward_backward_host_ms'] += (time.perf_counter()-t0)*1000
        if args.source_metrics:
            row = observe_microbatch(
                tokens, valid, prefix, result, configured_prefix=args.prefix_length,
                window_blocks=args.window_blocks, block_size=k,
                observations=observations, source_counts=counts,
                collect_sources=True,
            )
        else:
            # Still report structurally correct throughput without row NLL.
            row = observe_microbatch(
                tokens, valid, prefix, result, configured_prefix=args.prefix_length,
                window_blocks=args.window_blocks, block_size=k,
                observations=observations, source_counts=counts,
                collect_sources=False,
            )
        merge_counters(counters, row)
        observed_tokens += int(result['supervised_tokens'])
        if gradient_monitor is not None:
            gradient_monitor.capture(micro_index, observed_tokens)
        if collect_position:
            for key, field in [('position_counts', 'counts'),
                               ('initial_position_ce_sums', 'initial_sum'),
                               ('commitment_position_ce_sums', 'commit_sum')]:
                pos[field] += np.asarray(asnumpy(result[key]), dtype=pos[field].dtype)
    if counters['supervised_tokens'] != unique_tokens:
        raise AssertionError('accumulated supervision is inconsistent')
    return losses, unique_tokens, counters, timers, pos


def _clip_grads(parameters, limit):
    norm_sq = xp.asarray(0., dtype=xp.float64)
    for p in parameters:
        g = p.grad.astype(xp.float32, copy=False)
        norm_sq += xp.sum(g * g, dtype=xp.float64)
    norm = float(xp.sqrt(norm_sq))
    if not math.isfinite(norm):
        raise FloatingPointError('nonfinite refinement gradient norm')
    if norm > limit:
        scale = limit / (norm + 1e-12)
        for p in parameters:
            p.grad *= scale
    return norm


def _fixed_validation_windows(data, args):
    """Rebuild deterministic held-out samples independently of training RNG.

    Cache only bounded token windows on CPU; never cache model activations or
    GPU copies. Identical samples at every evaluation make checkpoint trends
    and first-position refinement comparisons interpretable.
    """
    rng = np.random.default_rng(args.seed + 10_079)
    return tuple(_sample(data, rng, args) for _ in range(args.val_batches))


def _evaluate(ref, data, rng, args, *, fixed_windows=None):
    """No-gradient, bounded grouped evaluation with correct per-token weights."""
    windows = fixed_windows if fixed_windows is not None else (
        _sample(data, rng, args) for _ in range(args.val_batches)
    )
    totals = {'commitment_ce': 0., 'initial_ce': 0.}
    k = args.block_size
    position_counts = np.zeros(k, dtype=np.int64)
    init_sums = np.zeros(k, dtype=np.float64)
    commit_sums = np.zeros(k, dtype=np.float64)
    n = 0
    counters = empty_counters()
    for sample in windows:
        tokens, valid, prefix, source_counts, observations = _unpack_sample(sample)
        result = ref.train_window(
            xp.asarray(tokens, dtype=xp.int64), prefix_length=prefix,
            valid_token_mask=xp.asarray(valid), group_size=args.refinement_group_size,
            compute_gradients=False, record_row_ce=args.source_metrics,
            compact_active_blocks=args.compact_active_blocks,
            max_active_queries=args.max_active_queries,
            compact_history=args.compact_history,
            resident_history=args.resident_history,
            max_history_keys=args.max_history_keys,
            max_window_tokens=args.max_window_tokens,
        )
        row = observe_microbatch(
            tokens, valid, prefix, result, configured_prefix=args.prefix_length,
            window_blocks=args.window_blocks, block_size=k,
            observations=observations, source_counts=source_counts,
            collect_sources=args.source_metrics,
        )
        merge_counters(counters, row)
        count = int(result['supervised_tokens'])
        n += count
        for key in totals:
            totals[key] += float(result[key]) * count
        position_counts += asnumpy(result['position_counts']).astype(np.int64)
        init_sums += asnumpy(result['initial_position_ce_sums']).astype(np.float64)
        commit_sums += asnumpy(result['commitment_position_ce_sums']).astype(np.float64)
    if n == 0 or int(position_counts.sum()) != n:
        raise ValueError('validation token accounting mismatch')
    initial_per_pos = np.divide(init_sums, position_counts,
                                out=np.full(k, np.nan), where=position_counts != 0)
    commit_per_pos = np.divide(commit_sums, position_counts,
                               out=np.full(k, np.nan), where=position_counts != 0)
    # Improvement at position 1 is not explained by new ground-truth tokens:
    # first refinement occurs before any position in the block is committed.
    gain = initial_per_pos - commit_per_pos
    return {
        **{key: value / n for key, value in totals.items()},
        'supervised_tokens': n,
        'position_counts': position_counts.tolist(),
        'initial_ce_by_pos': initial_per_pos.tolist(),
        'commit_ce_by_pos': commit_per_pos.tolist(),
        'refinement_gain_by_pos': gain.tolist(),
        'first_position_refinement_gain': float(gain[0]),
        'source_metrics': source_summary(counters),
        'token_accounting': {k: v for k, v in counters.items() if k != 'source_records'},
    }


def _format_position_metrics(val):
    def fmt(values):
        return '[' + ','.join(f'{v:.4f}' if math.isfinite(v) else 'NA' for v in values) + ']'
    return (f'valid_positions={val["position_counts"]} '
            f'initial_pos={fmt(val["initial_ce_by_pos"])} '
            f'commit_pos={fmt(val["commit_ce_by_pos"])} '
            f'gain_pos={fmt(val["refinement_gain_by_pos"])} '
            f'gain_first={val["first_position_refinement_gain"]:.5f}')


def _write_metrics(path, payload):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as stream:
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: clean(v) for key, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value
        stream.write(json.dumps(clean(payload), allow_nan=False) + '\n')


def _save(path, ref, opt, step, supervised, provenance, train_rng, val_rng,
          *, telemetry=None, telemetry_since_step=None):
    path.mkdir(parents=True, exist_ok=True)
    save_refinement_checkpoint(
        path, ref, optimizer=opt, training_step=step, supervised_tokens=supervised,
        tokenizer_fingerprint=provenance['tokenizer_sha256'], data_provenance=provenance,
    )
    # Checkpoint and RNG metadata carry the same optimizer step. A mismatch is
    # rejected on resume, rather than silently restarting a different stream.
    payload = {'step': step, 'train_rng': train_rng.bit_generator.state,
               'val_rng': val_rng.bit_generator.state}
    tmp = path / 'refinement_data_rng.json.tmp'
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf8')
    tmp.replace(path / 'refinement_data_rng.json')
    if telemetry is not None:
        snapshot = {'step': step, 'since_step': telemetry_since_step,
                    'counters': telemetry}
        sidecar = path / 'refinement_telemetry.json.tmp'
        sidecar.write_text(json.dumps(snapshot, indent=2), encoding='utf8')
        sidecar.replace(path / 'refinement_telemetry.json')


def _format_performance_message(counters, elapsed, step_seconds):
    r = rates(counters, elapsed, step_seconds=step_seconds)
    return (
        f"grad_weighting=supervised_tokens, target_fill={100*r['target_fill']:.1f}%, "
        f"{r['steps_per_second']:.2f} steps/sec, "
        f"step_time={step_seconds*1e3:.1f} ms, "
        f"source_tok/s={r['source_tok_per_second']:,.0f}, "
        f"valid_source_tok/s={r['valid_source_tok_per_second']:,.0f}, "
        f"valid_history_tok/s={r['valid_history_tok_per_second']:,.0f}, "
        f"active_tok/s={r['active_tok_per_second']:,.0f}, "
        f"supervised_tok/s={r['supervised_tok_per_second']:,.0f}, "
        f"active_queries/pass={r['active_queries_per_pass_mean']:.1f}, "
        f"active_position_evals/s={r['active_position_evals_per_second']:,.0f}, "
        f"position_evals/target={r['active_position_evals_per_unique_target']:.2f}, "
        f"block_row_fill={100*r['active_block_row_fill']:.1f}%"
    )


def _gradient_family_stats(parameters):
    """Optional read-only FP32 gradient norms; not gradient-noise estimation."""
    sums = {}
    for p in parameters:
        name = p.name.lower()
        if 'attention' in name:
            family = 'attention'
        elif 'moe' in name and 'router' in name:
            family = 'moe_router'
        elif 'moe' in name:
            family = 'moe_experts'
        elif 'embedding' in name:
            family = 'embedding'
        elif 'output_proj' in name or 'lm_head' in name:
            family = 'lm_head'
        elif 'refinement' in name or 'unknown' in name or 'soft' in name or 'fixed' in name:
            family = 'refinement_states'
        else:
            family = 'other'
        if p.grad is None:
            continue
        g = p.grad.astype(xp.float32, copy=False)
        term = xp.sum(g * g, dtype=xp.float64)
        sums[family] = term if family not in sums else sums[family] + term
    return {family: float(xp.sqrt(value)) for family, value in sorted(sums.items())}


def _check_gradient_finiteness(parameters):
    for p in parameters:
        if p.grad is not None and not bool(xp.all(xp.isfinite(p.grad))):
            raise FloatingPointError(f'nonfinite gradients in {p.name}')


def _source_report(stats, *, step, source_csv, cumulative, scope='interval',
                   since_step=1):
    if not stats['source_records']:
        return
    print('  Per-source commitment CE (valid-target weighted):', flush=True)
    for name, row in source_summary(stats).items():
        mean = 'NA' if row['commitment_ce'] is None else f'{row["commitment_ce"]:.4f}'
        span = ('' if row['sampled_document_span_mean'] is None else
                f', sampled_doc_mean={row["sampled_document_span_mean"]:.0f}')
        print(f'    {name}: CE={mean}, supervised={row["supervised_tokens"]:,}, '
              f'target_fill={100*row["target_fill"]:.1f}%{span}', flush=True)
    if source_csv is not None:
        rows = []
        for name in sorted(stats['source_records']):
            rows.append(source_metrics_row(name, stats['source_records'][name], step,
                                           scope, since_step))
            rows.append(source_metrics_row(name, cumulative['source_records'][name], step,
                                           'cumulative_since_telemetry_start', since_step))
        write_source_rows(source_csv, rows)


def main(argv=None):
    args = parse_args(argv)
    max_window = args.prefix_length + args.window_blocks * args.block_size
    root = args.mixed_shard_root or args.shard_dir
    tok_path = (args.tokenizer_path or (root / 'tokenizer.json')).expanduser().resolve()
    if not tok_path.exists():
        raise FileNotFoundError(f'tokenizer not found: {tok_path}')
    tokenizer, eos_id = _load_tokenizer(tok_path)
    vocab = len(tokenizer)
    if not 60000 <= vocab <= 70000:
        raise ValueError(f'expected the EXISTING ~65K-token BPE, got {vocab}; do not train a new tokenizer')
    dataset_args = dict(mixed_root=args.mixed_shard_root, shard_dir=args.shard_dir,
                        eos_token_id=eos_id, token_dtype=args.token_dtype)
    train_data = PackedRefinementDocuments(**dataset_args, split='train')
    val_data = PackedRefinementDocuments(**dataset_args, split='val')
    provenance = _provenance(args, tok_path, train_data, val_data)
    if args.dtype == 'bfloat16':
        validate_bfloat16_backend()
    if args.model_preset == WIDE_TRUNK_NAME:
        if vocab != 65280:
            raise ValueError('wide-500m-static-trunk requires the original 65280 tokenizer')
        config = wide_static_trunk_config(dtype=args.dtype, context_length=max(256, max_window))
        print('WARNING: 500M static trunk: two retrieval heads replaced with local heads; '
              'no 64K historical router. Long causal windows are experimental.', flush=True)
    else:
        config = ModelConfig(
            tokenizer_vocab_size=vocab,
            context_length=max(args.max_window_tokens if args.max_window_tokens != 256 else 256,
                               max_window), n_layers=args.layers, d_model=args.d_model,
            n_q_heads=args.n_q_heads, n_kv_heads=args.n_kv_heads,
            d_head=args.d_model // args.n_q_heads, d_ff=args.d_ff,
            n_experts=args.experts, top_k=args.top_k, dtype=args.dtype,
        )
    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=args.dtype)
    ref = WindowRefinementTrainer(
        model, RefinementReferenceConfig(
            block_size=args.block_size, initial_loss_weight=args.initial_loss_weight,
            feedback_mode=args.feedback_mode,
            enable_shared_context_training=True,
            compact_recurrent_tape=args.compact_recurrent_tape,
        ), rng_seed=args.seed + 1,
    )
    optimizer = AdamW(ref.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_rng, val_rng = np.random.default_rng(args.seed + 2), np.random.default_rng(args.seed + 3)
    # Fixed validation is generated from a separate seed and never consumes
    # train_rng or checkpoint val_rng. Existing 0079I optimizer resumes remain
    # valid; validation-mode/profile flags do not affect training updates.
    fixed_val = (_fixed_validation_windows(val_data, args)
                 if args.validation_mode == 'fixed' else None)
    start_step, supervised_tokens = 0, 0
    restored_telemetry = None
    telemetry_since_step = 1
    if args.resume:
        saved = load_refinement_checkpoint(args.resume, ref, optimizer=optimizer)
        if saved['architecture'].get('data_provenance') != provenance:
            raise ValueError('checkpoint data/tokenizer/settings provenance mismatch')
        start_step, supervised_tokens = int(saved['step']), int(saved['supervised_tokens'])
        run_state = json.loads((args.resume / 'refinement_data_rng.json').read_text(encoding='utf8'))
        if int(run_state['step']) != start_step:
            raise ValueError('checkpoint/RNG step mismatch; refusing ambiguous resume')
        train_rng.bit_generator.state = run_state['train_rng']
        val_rng.bit_generator.state = run_state['val_rng']
        sidecar = args.resume / 'refinement_telemetry.json'
        if sidecar.exists():
            observed = json.loads(sidecar.read_text(encoding='utf8'))
            if int(observed['step']) == start_step:
                restored_telemetry = observed['counters']
                telemetry_since_step = observed.get('since_step', 1)
            else:
                print('WARNING: stale telemetry sidecar; restarting diagnostics '
                      'without changing checkpoint or RNG state', flush=True)
        else:
            telemetry_since_step = start_step + 1
    nparams = sum(int(p.data.size) for p in ref.parameters())
    print(f'backend={BACKEND_NAME} precision={args.dtype} vocab={vocab} eos={eos_id} '
          f'layers={config.n_layers} d_model={config.d_model} experts={config.n_experts} '
          f'top_k={config.top_k} params={nparams:,} K={args.block_size} feedback={args.feedback_mode} '
          f'blocks={args.window_blocks} group={args.refinement_group_size} '
          f'prefix={args.prefix_length} batch={args.batch_size} accum={args.grad_accum} '
          f'active_queries_per_pass={args.batch_size * args.refinement_group_size * args.block_size}'
          f' recurrent_tape={"compact_0079p1" if args.compact_recurrent_tape else "full_reference"}')
    print(f'optimizer=AdamW beta1={optimizer.beta1:g} beta2={optimizer.beta2:g} '
          f'eps={optimizer.eps:g} lr={optimizer.lr:g} '
          f'weight_decay={args.weight_decay:g} '
          f'gradient_weighting=supervised_tokens lr_schedule=constant '
          f'effective_sequence_batch={args.batch_size * args.grad_accum} '
          f'token_capacity_per_optimizer_step={args.batch_size * args.grad_accum * args.window_blocks * args.block_size}',
          flush=True)
    print('history_mode=short_static_reference router=disabled '
          'router_entropy=NA history_64k=NA retrieval_stats=NA', flush=True)
    for name, weight, paths in train_data.sources:
        print(f'  train_source={name} sampling_weight={weight:.4f} shards={len(paths)}',
              flush=True)
    print(f'validation_mode={args.validation_mode} val_batches={args.val_batches} '
          f'profile_steps={args.profile_steps} profile_after={args.profile_after_step}')
    print(f'tokenizer_sha256={provenance["tokenizer_sha256"]} '
          f'train_sources={train_data.source_names} val_sources={val_data.source_names}')
    metrics_jsonl = args.metrics_jsonl
    if metrics_jsonl is None and not args.no_save:
        metrics_jsonl = args.checkpoint_dir / 'refinement_metrics.jsonl'
    metrics_csv = args.metrics_csv
    if metrics_csv is None and not args.no_save:
        metrics_csv = args.checkpoint_dir / 'refinement_training.csv'
    source_csv = args.source_metrics_csv
    if source_csv is None and not args.no_save and args.source_metrics:
        source_csv = args.checkpoint_dir / 'refinement_source_metrics.csv'
    print('observability=0079L source_metrics={} source_csv={} scalar_csv={} jsonl={} '
          'gradient_family_interval={} gpu_memory_interval={}'.format(
              args.source_metrics, source_csv, metrics_csv, metrics_jsonl,
              args.gradient_group_interval, args.gpu_memory_interval), flush=True)
    if args.eval_only:
        val = _evaluate(ref, val_data, val_rng, args, fixed_windows=fixed_val)
        print(f'evaluation checkpoint_step={start_step} '
              f'commitment_ce={val["commitment_ce"]:.5f} '
              f'initial_ce={val["initial_ce"]:.5f} '
              f'unique_tokens={val["supervised_tokens"]} '
              + _format_position_metrics(val), flush=True)
        for name, row in val['source_metrics'].items():
            print(f'  validation source={name} CE={row["commitment_ce"]:.5f} '
                  f'supervised={row["supervised_tokens"]} '
                  f'target_fill={100*row["target_fill"]:.1f}%', flush=True)
        _write_metrics(metrics_jsonl, {'kind': 'validation', 'step': start_step,
                       'mode': args.validation_mode, 'feedback_mode': args.feedback_mode,
                       'block_size': args.block_size, 'group_size': args.refinement_group_size,
                       **val})
        if metrics_csv is not None:
            write_scalar_csv(metrics_csv, step=start_step, supervised_tokens=supervised_tokens,
                             val_loss=val['commitment_ce'],
                             first_position_gain=val['first_position_refinement_gain'])
        return
    gradient_monitor = None
    if args.gradient_noise_monitor or args.adamw_update_monitor:
        gradient_monitor = TokenWeightedRefinementGradientMonitor(
            ref.parameters(), accum_steps=args.grad_accum,
            accum_points=args.gradient_noise_accum_points,
            categories=args.gradient_noise_parameter_groups,
            output=(args.gradient_noise_output or
                    args.checkpoint_dir / 'refinement_gradient_noise.csv'),
            interval=args.gradient_noise_interval, optimizer=optimizer,
            adamw_output=((args.adamw_update_output or
                           args.checkpoint_dir / 'refinement_adamw_update.csv')
                          if args.adamw_update_monitor else None),
            grad_clip=args.clip_grad_norm,
        )
    interval_t0, interval_tokens = time.perf_counter(), 0
    interval_steps = 0
    interval_counters = empty_counters()
    cumulative_counters = (restored_telemetry if restored_telemetry is not None
                           else empty_counters())
    interval_timers = defaultdict(float)
    interval_loss_numerators = defaultdict(float)
    for step in range(start_step, args.steps):
        last_step = step + 1
        step_start = time.perf_counter()
        monitored = gradient_monitor is not None and gradient_monitor.begin(step)
        profiling = (args.profile_steps > 0 and
                     args.profile_after_step <= last_step < args.profile_after_step + args.profile_steps)
        configure_performance_profiler(profiling, reset=profiling and last_step == args.profile_after_step)
        with performance_scope('refinement.training_microbatches'):
            losses, n, counters, timings, pos = _train_one(
                ref, train_data, train_rng, args,
                collect_position=(last_step == start_step + 1
                                  or last_step % args.log_every == 0),
                gradient_monitor=(gradient_monitor if monitored else None),
            )
        noise_rows = gradient_monitor.finish(n) if monitored else []
        t0 = time.perf_counter()
        with performance_scope('refinement.clip_gradient'):
            grad_norm = _clip_grads(ref.parameters(), args.clip_grad_norm)
        timings['grad_clip_wall_ms'] = (time.perf_counter()-t0)*1000
        if args.numerical_debug:
            _check_gradient_finiteness(ref.parameters())
        family_norms = None
        if args.gradient_group_interval and last_step % args.gradient_group_interval == 0:
            family_norms = _gradient_family_stats(ref.parameters())
        t0 = time.perf_counter()
        with performance_scope('refinement.optimizer_step'):
            optimizer.step()
        timings['optimizer_host_submission_ms'] = (time.perf_counter()-t0)*1000
        if monitored:
            gradient_monitor.commit(noise_rows)
            if noise_rows:
                print('  Gradient-noise (token-weighted, observational): '
                      + ', '.join(f'{r["parameter_group"]} '
                                  f'cos={r["gradient_cosine"]:.3f}'
                                  if r['gradient_cosine'] is not None else
                                  f'{r["parameter_group"]} cos=NA'
                                  for r in noise_rows), flush=True)
        if profiling:
            synchronize()
            print(performance_report(f'0079L step {last_step} (instrumented)'), flush=True)
        configure_performance_profiler(False)
        supervised_tokens += n
        interval_tokens += n
        interval_steps += 1
        merge_counters(interval_counters, counters)
        merge_counters(cumulative_counters, counters)
        for key, value in timings.items():
            interval_timers[key] += value
        for key, value in losses.items():
            interval_loss_numerators[key] += float(value) * n
        if family_norms is not None:
            print('  Gradient families (pre-update, after clipping): '
                  + ', '.join(f'{name}={value:.4g}' for name,value in family_norms.items()),
                  flush=True)
        mem = None
        if args.gpu_memory_interval and last_step % args.gpu_memory_interval == 0:
            mem = memory_snapshot()
            if mem:
                print(f'  GPU snapshot: CuPy live={mem["cupy_pool_live_gib"]:.2f} GiB '
                      f'reserved={mem["cupy_pool_reserved_gib"]:.2f} GiB '
                      f'device_used_now={mem["device_used_now_gib"]:.2f} GiB '
                      '(not peak)', flush=True)
        if args.gpu_cache_audit_interval and last_step % args.gpu_cache_audit_interval == 0:
            from mini_llm.diagnostics.gpu_cache_audit import persistent_gpu_cache_inventory
            inventory = persistent_gpu_cache_inventory(model)
            print('  Persistent GPU caches (nominal bytes; not peak): '
                  + ', '.join(f'{k}={v["entries"]} entries/{v["cuda_bytes"]} B'
                              for k,v in inventory.items()), flush=True)
            _write_metrics(metrics_jsonl, {'kind': 'gpu_cache_audit', 'step': last_step,
                                           'inventory': inventory})
        if last_step == start_step + 1 or last_step % args.log_every == 0:
            synchronize()
            elapsed = max(1e-9, time.perf_counter()-interval_t0)
            avg_step = elapsed / interval_steps
            r = rates(interval_counters, elapsed, step_seconds=avg_step)
            interval_losses = {
                key: value / interval_counters['supervised_tokens']
                for key, value in interval_loss_numerators.items()
            }
            memory = mem if mem is not None else memory_snapshot()
            print(f'Step {last_step}/{args.steps}: '
                  f'loss={interval_losses["commitment_ce"]:.4f}, '
                  f'initial_ce={interval_losses["initial_ce"]:.4f}, '
                  f'total={interval_losses["total_loss"]:.4f}, '
                  f'lr={optimizer.lr:.6f}, grad_norm={grad_norm:.4f}, '
                  + _format_performance_message(interval_counters, elapsed, avg_step),
                  flush=True)
            print(f'  Parallel work: unique_targets={interval_counters["supervised_tokens"]:,}, '
                  f'group_batches={interval_counters["refinement_batches"]:,}, '
                  f'refinement_passes={interval_counters["refinement_full_model_passes"]:,}, '
                  f'context_encodes={interval_counters["causal_context_encodes"]:,}, '
                  f'queries/pass_min={r["active_queries_per_pass_min"]}, '
                  f'queries/pass_max={r["active_queries_per_pass_max"]}, '
                  f'effective_batch={args.batch_size * args.grad_accum}, '
                  f'active_queries_per_full_group={args.batch_size * args.refinement_group_size * args.block_size}, '
                  f'skipped_empty_block_rows={interval_counters["skipped_empty_block_rows"]:,}, '
                  f'history_keys/group_max={r["context_keys_per_group_max"]}',
                  flush=True)
            _source_report(interval_counters, step=last_step, source_csv=source_csv,
                           cumulative=cumulative_counters,
                           since_step=telemetry_since_step)
            payload = {
                'kind': 'training', 'schema_version': 2, 'step': last_step,
                'feedback_mode': args.feedback_mode, 'block_size': args.block_size,
                'group_size': args.refinement_group_size,
                'batch_size': args.batch_size, 'grad_accum': args.grad_accum,
                'effective_sequence_batch': args.batch_size * args.grad_accum,
                'gradient_weighting': 'supervised_tokens',
                'learning_rate': optimizer.lr,
                'unique_supervised_tokens': supervised_tokens,
                'unique_tok_per_second': r['supervised_tok_per_second'],
                'gradient_norm': grad_norm,
                'step_time_avg_ms': avg_step * 1000,
                'throughput': r,
                'token_accounting': {k: v for k, v in interval_counters.items()
                                     if k != 'source_records'},
                'source_metrics': source_summary(interval_counters),
                'timings_ms': {k: v / interval_steps for k,v in interval_timers.items()},
                'gpu_memory_snapshot': memory,
                'gradient_family_norms': family_norms,
                'last_step_losses': {key: float(value) for key, value in losses.items()},
                **interval_losses,
            }
            if pos is not None:
                counts = pos['counts']
                initial_by_pos = np.divide(
                    pos['initial_sum'], counts,
                    out=np.full(args.block_size, np.nan), where=counts > 0)
                commit_by_pos = np.divide(
                    pos['commit_sum'], counts,
                    out=np.full(args.block_size, np.nan), where=counts > 0)
                payload['train_position_metrics'] = {
                    'counts': counts.tolist(),
                    'initial_ce_by_pos': initial_by_pos.tolist(),
                    'commit_ce_by_pos': commit_by_pos.tolist(),
                    'gain_by_pos': (initial_by_pos - commit_by_pos).tolist(),
                }
                print('  Training position CE: initial='
                      + repr(np.round(initial_by_pos, 4).tolist())
                      + ' commitment=' + repr(np.round(commit_by_pos, 4).tolist()),
                      flush=True)
            _write_metrics(metrics_jsonl, payload)
            if metrics_csv is not None:
                write_scalar_csv(
                    metrics_csv, step=last_step, supervised_tokens=supervised_tokens,
                    train_loss=interval_losses['commitment_ce'],
                    learning_rate=optimizer.lr, gradient_norm=grad_norm,
                    steps_per_second=r['steps_per_second'],
                    step_time_ms=avg_step*1000, throughput=r,
                )
            interval_t0 = time.perf_counter()
            interval_tokens = 0
            interval_steps = 0
            interval_counters = empty_counters()
            interval_timers.clear()
            interval_loss_numerators.clear()
        if last_step % args.val_every == 0:
            excluded_start = time.perf_counter()
            val = _evaluate(ref, val_data, val_rng, args, fixed_windows=fixed_val)
            print(f'validation step={last_step} commitment_ce={val["commitment_ce"]:.5f} '
                  f'initial_ce={val["initial_ce"]:.5f} '
                  f'unique_tokens={val["supervised_tokens"]} '
                  + _format_position_metrics(val), flush=True)
            for name, source in val['source_metrics'].items():
                print(f'  validation source={name} commitment_ce={source["commitment_ce"]:.5f} '
                      f'supervised={source["supervised_tokens"]} '
                      f'target_fill={100*source["target_fill"]:.1f}%', flush=True)
            _write_metrics(metrics_jsonl, {'kind': 'validation', 'schema_version': 2,
                           'step': last_step, 'mode': args.validation_mode,
                           'feedback_mode': args.feedback_mode,
                           'block_size': args.block_size,
                           'group_size': args.refinement_group_size,
                           **val})
            if metrics_csv is not None:
                write_scalar_csv(
                    metrics_csv, step=last_step, supervised_tokens=supervised_tokens,
                    val_loss=val['commitment_ce'],
                    first_position_gain=val['first_position_refinement_gain'],
                )
            # Exclude validation from the current training interval while
            # preserving all accumulated counts when cadence differs from log.
            interval_t0 += time.perf_counter() - excluded_start
        if not args.no_save and last_step % args.save_every == 0:
            excluded_start = time.perf_counter()
            _save(args.checkpoint_dir, ref, optimizer, last_step, supervised_tokens,
                  provenance, train_rng, val_rng,
                  telemetry=cumulative_counters,
                  telemetry_since_step=telemetry_since_step)
            interval_t0 += time.perf_counter() - excluded_start
    if not args.no_save and args.steps > start_step:
        _save(args.checkpoint_dir, ref, optimizer, args.steps, supervised_tokens,
              provenance, train_rng, val_rng,
              telemetry=cumulative_counters,
              telemetry_since_step=telemetry_since_step)
    if args.generate:
        from mini_llm.inference_model import InferenceModel
        from mini_llm.predictive_refinement.inference import (
            CachedPredictiveRefinementGenerator, RefinementGenerationConfig,
        )
        from mini_llm.backend import asnumpy
        prompt_ids = tokenizer.encode(args.prompt)
        if not prompt_ids:
            raise ValueError('generation prompt tokenized to zero tokens')
        if len(prompt_ids) + args.generate > 256:
            raise ValueError('prompt + generation exceeds reference 256-token cache')
        inference = InferenceModel(config, dtype=args.dtype)
        inference.set_weights(model)
        state = inference.create_generation_state(1, config.context_length)
        inference.prefill(xp.asarray([prompt_ids], dtype=xp.int32), state)
        generator = CachedPredictiveRefinementGenerator(
            inference, ref.states, RefinementGenerationConfig(
                block_size=args.block_size, eos_token_id=eos_id,
            ), seed=args.seed + 4,
        )
        output = generator.generate(state, args.generate)
        print(f'generation passes={output["passes"]} tokens={output["tokens"]}')
        print('generated_text=' + repr(tokenizer.decode(output['tokens'])))


if __name__ == '__main__':
    main()
