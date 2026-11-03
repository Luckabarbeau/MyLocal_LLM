#!/usr/bin/env python3
"""0105F: guarded experimental frozen-Model-B residual specialist trainer.

No implicit foundation optimizer, checkpoint overwrite, lexical injection or
specialist-owned history. Defaults are a SHORT smoke test, not a production run.
Use a separate worktree and *copied/immutable* completed foundation checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foundation-checkpoint', type=Path, required=True)
    p.add_argument('--mixed-shard-root', type=Path, required=True)
    p.add_argument('--specialist-checkpoint', type=Path, required=True)
    p.add_argument('--specialist-id', default='general')
    p.add_argument('--specialist-layers', type=int, default=2,
                   help='2-layer smoke; 12-layer scientific experiment after GPU validation')
    p.add_argument('--context-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=1,
                   help='Any positive microbatch size, subject to available device memory')
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--total-steps', type=int, default=10000)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--peak-lr', type=float, default=3e-4)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--muon-momentum', type=float, default=.95)
    p.add_argument('--muon-ns-steps', type=int, default=5)
    p.add_argument('--muon-matmul-precision', choices=('bf16','fp32'), default='bf16')
    p.add_argument('--head-chunk-tokens', type=int, default=512)
    p.add_argument('--runtime-profile',
                   choices=('auto', 'reference', 'gpu-fast', 'consumer-gpu'),
                   default=None,
                   help='GPU fast-path defaults; auto selects consumer-gpu for the Model B foundation')
    p.add_argument('--no-activation-checkpoint', action='store_true',
                   help='Disable whole-specialist activation checkpoint replay')
    p.add_argument('--no-attention-prob-recompute', action='store_true',
                   help='Retain Model B matrix-pool attention probabilities for backward; '
                        'independent of whole-specialist activation checkpointing')
    p.add_argument('--no-accum-length-buckets', action='store_true',
                   help='Disable source-preserving length regrouping across accumulation')
    p.add_argument('--shape-bucket', choices=('none','geometric','fine'), default='fine')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=1,
                   help='Optimizer updates to run in this invocation')
    p.add_argument('--validation-batches', type=int, default=2)
    p.add_argument('--log-interval', type=int, default=1,
                   help='Print foundation-style training and per-source CE every N optimizer steps')
    p.add_argument('--train-foundation-ce', action='store_true',
                   help='Evaluate the frozen foundation CE on every EXACT training minibatch '
                        'before each specialist update (extra frozen LM-head computation)')
    p.add_argument('--log-text-file', type=Path,
                   help='Optional append-only plain-text file containing the console step reports')
    p.add_argument('--profile-steps', type=int, default=0,
                   help='Profile the first N steps; inserts CUDA synchronizations')
    p.add_argument('--save-interval', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--run', action='store_true',
                   help='Actually load model, train and write specialist checkpoint')
    p.add_argument('--log-jsonl', type=Path)
    return p.parse_args(argv)


def validate_paths(args):
    foundation = args.foundation_checkpoint.expanduser().resolve()
    specialist = args.specialist_checkpoint.expanduser().resolve()
    if specialist == foundation or specialist in foundation.parents or foundation in specialist.parents:
        raise ValueError('foundation and specialist checkpoints must occupy independent directories')
    if not args.mixed_shard_root.is_dir():
        raise FileNotFoundError('mixed-shard root not found')
    if not foundation.is_dir():
        raise FileNotFoundError('completed foundation checkpoint not found')
    if args.resume:
        if not specialist.is_dir():
            raise FileNotFoundError('specialist resume checkpoint not found')
    elif specialist.exists():
        raise FileExistsError('refusing to replace an existing specialist checkpoint; use --resume')
    if args.batch_size <= 0:
        raise ValueError('batch size must be positive')
    if args.log_interval <= 0:
        raise ValueError('log interval must be positive')
    if args.steps < 0 or args.validation_batches < 0 or args.save_interval < 0 or args.profile_steps < 0:
        raise ValueError('steps/validation/save interval must be nonnegative')
    return foundation, specialist


def main(argv=None):
    args = arguments(argv)
    foundation_path, specialist_path = validate_paths(args)
    print('Specialist experiment: frozen foundation / hidden-only residual')
    print('Foundation:', foundation_path)
    print('Mixture:', args.mixed_shard_root.expanduser().resolve())
    print('Independent specialist checkpoint:', specialist_path)
    print('Backend:', os.environ.get('MINI_LLM_BACKEND','numpy'))
    print('Layers:', args.specialist_layers, 'context:', args.context_length,
          'microbatch:', args.batch_size, 'accumulation:', args.grad_accum_steps,
          'effective sequences/update:', args.batch_size * args.grad_accum_steps)
    if not args.run:
        print('DRY RUN: no model loaded and no checkpoint written. Add --run to execute.')
        return 0

    # Deliberately delay heavyweight imports until the explicit --run gate.
    from train_model import load_tokenizer_file, tokenizer_eos_id
    from mini_llm.specialists import (FrozenFoundation, ResidualSpecialist,
                                      SpecialistConfig, FrozenResidualComposition)
    from mini_llm.specialists.document_sampler import SpecialistDocumentSampler
    from mini_llm.specialists.training_run import (
        SpecialistTrainingConfig, SpecialistTrainingRun)
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from mini_llm.specialists.production_contract_0107 import (
        validate_model_b_specialist, validate_sparse_model_b_foundation,
        validate_resume_specialist)
    from mini_llm.specialists.train_logging_0107c import SpecialistIntervalLogger
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.config import ModelConfig
    from mini_llm.runtime_defaults import (
        apply_runtime_defaults, format_runtime_fast_paths, format_runtime_profile)
    if BACKEND_NAME != 'cupy':
        raise RuntimeError('Production specialist training requires MINI_LLM_BACKEND=cupy')

    # Apply the same validated CuPy/BF16 fast paths as train_model.py before
    # constructing either the frozen foundation or the specialist. Explicit
    # MINI_LLM_* environment settings remain authoritative through setdefault.
    with (foundation_path / 'config.json').open(encoding='utf-8') as source:
        cfg = ModelConfig(**json.load(source))
    runtime_profile = apply_runtime_defaults(
        args.runtime_profile, backend_name=BACKEND_NAME, config=cfg,
        training=True, precision='bf16-mixed')
    print(format_runtime_profile(runtime_profile))
    print(format_runtime_fast_paths(cfg))

    sources, manifest = load_weighted_shard_sources(args.mixed_shard_root)
    tokenizer = load_tokenizer_file(args.mixed_shard_root / 'tokenizer.json')
    eos_id = tokenizer_eos_id(tokenizer)
    foundation = FrozenFoundation.from_checkpoint(foundation_path)
    cfg = foundation.config
    if int(cfg.tokenizer_vocab_size) != int(manifest['vocab_size']) or len(tokenizer) != int(manifest['vocab_size']):
        raise ValueError('foundation vocabulary and mixture tokenizer must agree')
    validate_sparse_model_b_foundation(cfg)
    if args.context_length > cfg.context_length:
        raise ValueError('requested specialist context exceeds foundation configuration')
    sample = SpecialistDocumentSampler(args.mixed_shard_root,
        context_length=args.context_length, batch_size=args.batch_size,
        eos_token_id=eos_id, block_size=cfg.dual_resolution.block_size,
        seed=args.seed, shape_bucket=args.shape_bucket)
    training = SpecialistTrainingConfig(
        grad_accum_steps=args.grad_accum_steps, total_steps=args.total_steps,
        warmup_steps=args.warmup_steps, peak_lr=args.peak_lr,
        grad_clip=args.grad_clip, weight_decay=args.weight_decay,
        muon_momentum=args.muon_momentum, muon_ns_steps=args.muon_ns_steps,
        muon_matmul_precision=args.muon_matmul_precision,
        head_chunk_tokens=args.head_chunk_tokens,
        activation_checkpoint=not args.no_activation_checkpoint,
        accum_length_buckets=not args.no_accum_length_buckets)
    if args.resume:
        run = SpecialistTrainingRun.resume(specialist_path, foundation, sample, config=training)
        validate_resume_specialist(
            run.composition.specialist.config, requested_id=args.specialist_id,
            requested_layers=args.specialist_layers)
    else:
        spec = SpecialistConfig.from_foundation(
            cfg, specialist_id=args.specialist_id,
            n_layers=args.specialist_layers,
            recompute_attention_probs=not args.no_attention_prob_recompute)
        specialist = ResidualSpecialist(spec, rng_seed=args.seed)
        run = SpecialistTrainingRun(
            FrozenResidualComposition(foundation, specialist), sample,
            config=training, checkpoint_dir=specialist_path)
    validate_model_b_specialist(
        run.composition.specialist.config, cfg,
        recompute_attention_probs=not args.no_attention_prob_recompute)
    print('Foundation parameters (frozen):',sum(int(p.size) for p in foundation.model.parameters()))
    print('Specialist trainable parameters:',sum(int(p.size) for p in run.parameters))
    print('Starting supervised tokens:', run.supervised_tokens)
    if args.validation_batches:
        print('Paired baseline CE:',json.dumps(run.evaluate_paired(batches=args.validation_batches)))

    logger = SpecialistIntervalLogger(log_interval=args.log_interval,
                                      total_steps=args.total_steps)

    def emit(lines):
        for line in lines:
            print(line, flush=True)
        if args.log_text_file and lines:
            args.log_text_file.parent.mkdir(parents=True, exist_ok=True)
            with args.log_text_file.open('a', encoding='utf-8') as log:
                log.write('\n'.join(lines) + '\n')

    def on_step(record):
        if logger.observe(record):
            emit(logger.flush())
        if args.log_jsonl:
            args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.log_jsonl.open('a',encoding='utf-8') as log:
                log.write(json.dumps(record)+'\n')
        if args.save_interval and record['step']%args.save_interval==0:
            run.save_checkpoint()

    run.train_steps(
        args.steps, on_step=on_step, profile_steps=args.profile_steps,
        source_metrics=True, foundation_train_ce=args.train_foundation_ce,
        measure_step_time=True, retain_reports=False)
    emit(logger.flush())  # Print an incomplete final log interval as well.
    if args.validation_batches:
        print('Paired final CE:',json.dumps(run.evaluate_paired(batches=args.validation_batches)))
    if args.steps:
        run.save_checkpoint()
    print('Completed experimental specialist run:',run.step,run.supervised_tokens)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
