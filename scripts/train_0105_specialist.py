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
    p.add_argument('--batch-size', type=int, default=1)
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
    p.add_argument('--shape-bucket', choices=('none','geometric','fine'), default='fine')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=1,
                   help='Optimizer updates to run in this invocation')
    p.add_argument('--validation-batches', type=int, default=2)
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
    if args.steps < 0 or args.validation_batches < 0 or args.save_interval < 0:
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
          'accumulation:', args.grad_accum_steps)
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

    sources, manifest = load_weighted_shard_sources(args.mixed_shard_root)
    tokenizer = load_tokenizer_file(args.mixed_shard_root / 'tokenizer.json')
    eos_id = tokenizer_eos_id(tokenizer)
    foundation = FrozenFoundation.from_checkpoint(foundation_path)
    cfg = foundation.config
    if int(cfg.tokenizer_vocab_size) != int(manifest['vocab_size']) or len(tokenizer) != int(manifest['vocab_size']):
        raise ValueError('foundation vocabulary and mixture tokenizer must agree')
    if not cfg.dual_resolution.enabled or cfg.dual_resolution.compression_mode != 'matrix_pool':
        raise ValueError('0105 specialist trainer requires matrix-pool Model B foundation')
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
        head_chunk_tokens=args.head_chunk_tokens, activation_checkpoint=True)
    if args.resume:
        run = SpecialistTrainingRun.resume(specialist_path, foundation, sample, config=training)
    else:
        spec = SpecialistConfig(
            specialist_id=args.specialist_id, n_layers=args.specialist_layers,
            d_model=cfg.d_model, d_ff=cfg.d_ff, n_q_heads=cfg.n_q_heads,
            n_kv_heads=cfg.n_kv_heads, d_head=cfg.d_head,
            local_window=512, full_window=4096,
            full_heads=1, dtype=cfg.dtype)
        specialist = ResidualSpecialist(spec, rng_seed=args.seed)
        run = SpecialistTrainingRun(
            FrozenResidualComposition(foundation, specialist), sample,
            config=training, checkpoint_dir=specialist_path)
    print('Foundation parameters (frozen):',sum(int(p.size) for p in foundation.model.parameters()))
    print('Specialist trainable parameters:',sum(int(p.size) for p in run.parameters))
    print('Starting supervised tokens:', run.supervised_tokens)
    if args.validation_batches:
        print('Paired baseline CE:',json.dumps(run.evaluate_paired(batches=args.validation_batches)))

    def on_step(record):
        print(json.dumps(record), flush=True)
        if args.log_jsonl:
            args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.log_jsonl.open('a',encoding='utf-8') as log:
                log.write(json.dumps(record)+'\n')
        if args.save_interval and record['step']%args.save_interval==0:
            run.save_checkpoint()

    run.train_steps(args.steps, on_step=on_step)
    if args.validation_batches:
        print('Paired final CE:',json.dumps(run.evaluate_paired(batches=args.validation_batches)))
    if args.steps:
        run.save_checkpoint()
    print('Completed experimental specialist run:',run.step,run.supervised_tokens)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
