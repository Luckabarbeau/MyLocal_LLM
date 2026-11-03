"""0105E: experimental model-centered training of a frozen residual specialist.

Reuses production tokenizer/shard mixture, document sampler, chunked frozen
LM-head, clipping, Hybrid Muon and warmup/cosine schedule. The *only* optimizer
parameters belong to the residual specialist. Not yet validated at 12L/8K on
an RTX 5080; launch with short GPU smoke checks before production training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, synchronize
from mini_llm.optim.hybrid_muon import HybridMuonAdamW
from mini_llm.optim.grad_clip import clip_grad_global_norm, _array_to_float
from mini_llm.optim.schedule import WarmupCosineSchedule
from mini_llm.diagnostics.source_metrics import batch_coverage, empty_record, merge_record

from .checkpoint import save_specialist_checkpoint, load_specialist_checkpoint


@dataclass(frozen=True)
class SpecialistTrainingConfig:
    grad_accum_steps: int = 1
    total_steps: int = 10000
    warmup_steps: int = 100
    peak_lr: float = 3e-4
    grad_clip: float = 1.0
    weight_decay: float = .1
    muon_momentum: float = .95
    muon_ns_steps: int = 5
    muon_matmul_precision: str = 'bf16'
    head_chunk_tokens: int = 512
    activation_checkpoint: bool = True
    accum_length_buckets: bool = True

    def __post_init__(self):
        if self.grad_accum_steps <= 0 or self.head_chunk_tokens <= 0:
            raise ValueError('accumulation and head chunks must be positive')
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError('warmup must be shorter than total steps')
        if not math.isfinite(self.peak_lr) or self.peak_lr <= 0:
            raise ValueError('learning rate must be positive finite')
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError('gradient clipping must be positive finite')


class SpecialistTrainingRun:
    """Only train residual parameters; the foundation is strictly read-only."""

    def __init__(self, composition, sampler, *, config: SpecialistTrainingConfig,
                 checkpoint_dir=None):
        if composition.foundation.checkpoint_path is None:
            raise ValueError('specialist training requires a checkpoint-backed foundation')
        if not composition.foundation.config.dual_resolution.enabled:
            raise ValueError('0105 specialist sampler supports dual-resolution Model B only')
        if sampler.context_length > composition.foundation.config.context_length:
            raise ValueError('sampler exceeds foundation context capacity')
        if sampler.block_size != composition.foundation.config.dual_resolution.block_size:
            raise ValueError('sampler/model block size mismatch')
        if sampler.eos_token_id < 0:
            raise ValueError('EOS is required for document-first specialist training')
        self.composition = composition
        self.sampler = sampler
        self.config = config
        self.step = 0
        self.supervised_tokens = 0
        self.source_tokens = {name: 0 for name in sampler.names}
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir).expanduser().resolve()
        if self.checkpoint_dir is not None:
            frozen = composition.foundation.checkpoint_path.resolve()
            if self.checkpoint_dir == frozen or frozen in self.checkpoint_dir.parents:
                raise ValueError('specialist checkpoint must not live inside foundation checkpoint')
            if self.checkpoint_dir in frozen.parents:
                raise ValueError('specialist checkpoint must not contain foundation checkpoint')
        frozen_names = {p.name for p in composition.foundation.model.parameters()}
        self.parameters = list(composition.parameters())
        if not self.parameters or frozen_names.intersection(p.name for p in self.parameters):
            raise ValueError('specialist optimizer must contain only independent parameters')
        if any(p.grad is not None for p in composition.foundation.model.parameters()):
            raise AssertionError('foundation gradients must not be allocated')
        self.optimizer = HybridMuonAdamW(
            self.parameters, lr=config.peak_lr, beta1=0.9, beta2=0.999,
            eps=1e-8, weight_decay=config.weight_decay,
            muon_momentum=config.muon_momentum,
            muon_ns_steps=config.muon_ns_steps,
            muon_matmul_precision=config.muon_matmul_precision)
        self.schedule = WarmupCosineSchedule(
            config.peak_lr, config.warmup_steps, config.total_steps)

    def train_step(self, *, profile=False, source_metrics=False,
                   foundation_train_ce=False, measure_step_time=False):
        if self.step >= self.config.total_steps:
            raise ValueError('total training steps exhausted')
        cfg = self.config
        timings = {} if profile else None
        if profile or measure_step_time:
            synchronize()
            step_start = perf_counter()
        self.optimizer.zero_grad()
        # Keep all microbatch weights, including short-document masks, on CPU
        # until the exact denominator is known. A per-microbatch CE is already
        # normalized internally; its gradient must be weighted by valid tokens.
        total_valid = 0
        total_loss = 0.0
        total_foundation_loss = 0.0
        executed_tokens = 0
        capacity_tokens = 0
        source_records = {} if source_metrics else None
        prefetched_batches = None
        length_bucket_stats = None
        if (cfg.accum_length_buckets and self.sampler.batch_size > 1
                and cfg.grad_accum_steps > 1):
            sample_start = perf_counter() if profile else None
            sampled = []
            for _ in range(cfg.grad_accum_steps):
                inputs, targets, metadata = self.sampler.next_batch(split='train')
                sampled.append((metadata['source_name'], inputs, targets, metadata))
            from mini_llm.data.dual_resolution_length_buckets import (
                regroup_dual_resolution_document_batches)
            prefetched_batches, length_bucket_stats = (
                regroup_dual_resolution_document_batches(
                    sampled, batch_size=self.sampler.batch_size,
                    context_length=self.sampler.context_length,
                    eos_token_id=self.sampler.eos_token_id))
            del sampled
            # Repacking changes [B,T], so the old per-shape block arrays cannot
            # be reused. The rowwise target masks and provenance remain exact.
            from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata
            rebuilt = []
            for source, inputs, targets, metadata in prefetched_batches:
                dual = build_dual_resolution_metadata(
                    inputs, eos_token_id=self.sampler.eos_token_id,
                    block_size=self.sampler.block_size)
                if dual['block_indices'].shape[1]:
                    ends = dual['block_indices'][:, :, -1]
                    dual['block_valid'] &= (
                        ends < metadata['target_valid_lengths'][:, None])
                dual['valid_lengths'] = np.asarray(
                    metadata['target_valid_lengths'], dtype=np.int64)
                metadata['dual_resolution_metadata'] = dual
                metadata['source_name'] = source
                rebuilt.append((inputs, targets, metadata))
            prefetched_batches = rebuilt
            if profile:
                assert timings is not None and sample_start is not None
                timings['sample_s'] = (timings.get('sample_s', 0.0)
                                       + perf_counter() - sample_start)

        for microbatch_index in range(cfg.grad_accum_steps):
            if prefetched_batches is None:
                sample_start = perf_counter() if profile else None
                inputs, targets, metadata = self.sampler.next_batch(split='train')
                if profile:
                    assert timings is not None and sample_start is not None
                    timings['sample_s'] = (timings.get('sample_s', 0.0)
                                           + perf_counter() - sample_start)
            else:
                inputs, targets, metadata = prefetched_batches[microbatch_index]
            valid = int(np.asarray(metadata['target_loss_mask']).sum())
            executed_tokens += int(inputs.size)
            capacity_tokens += int(self.sampler.batch_size * self.sampler.context_length)
            if valid <= 0:
                raise ValueError('sampler produced a batch without supervised tokens')
            loss, cache = self.composition.loss(
                xp.asarray(inputs, dtype=xp.int32),
                xp.asarray(targets, dtype=xp.int32),
                loss_mask=metadata['target_loss_mask'],
                dual_resolution_metadata=metadata['dual_resolution_metadata'],
                chunk_tokens=cfg.head_chunk_tokens,
                activation_checkpoint=cfg.activation_checkpoint,
                return_cache=True, timings=timings,
                return_foundation_ce=foundation_train_ce,
            )
            baseline_ce = None
            if foundation_train_ce:
                baseline_ce = float(_array_to_float(cache.pop('foundation_loss')))
                total_foundation_loss += baseline_ce * valid
            self.composition.backward(cache, grad_scale=float(valid), timings=timings)
            combined_ce = float(_array_to_float(loss))
            total_loss += combined_ce * valid
            total_valid += valid
            name = metadata['source_name']
            if source_metrics:
                coverage = batch_coverage(
                    batch_size=int(inputs.shape[0]),
                    source_length=int(inputs.shape[1]),
                    target_shape=targets.shape,
                    loss_mask=np.asarray(metadata['target_loss_mask']),
                    metadata=metadata)
                # Generic coverage counts B*T source positions as valid; for
                # document-first microbatches, shorter EOS-padded rows are
                # executed but not valid source tokens. Keep that distinction
                # explicit without altering the shared foundation metrics.
                valid_source = int(np.asarray(metadata['target_valid_lengths']).sum())
                coverage['valid_active_tokens'] = valid_source
                coverage['valid_source_tokens'] = valid_source
                # Preserve the production source metrics' valid-target weighting.
                coverage['loss_nll_sum'] = combined_ce * coverage['supervised_token_weight']
                bucket = source_records.setdefault(name, empty_record())
                merge_record(bucket, coverage)
                if foundation_train_ce:
                    bucket['foundation_nll_sum'] = (bucket.get('foundation_nll_sum', 0.0)
                                                    + baseline_ce * coverage['supervised_token_weight'])
            self.source_tokens[name] += valid
        if total_valid <= 0:
            raise RuntimeError('zero supervised tokens in optimizer update')
        for p in self.parameters:
            if p.grad is not None:
                p.grad[...] /= total_valid
        if profile:
            synchronize()
            clip_start = perf_counter()
        norm_backend, _, finite = clip_grad_global_norm(
            self.parameters, max_norm=cfg.grad_clip)
        if profile:
            synchronize()
            timings['clip_s'] = perf_counter() - clip_start
        if not finite:
            self.optimizer.zero_grad()
            raise FloatingPointError('nonfinite specialist gradient; update rejected')
        lr = float(self.schedule(self.step))
        if profile:
            synchronize()
            optimizer_start = perf_counter()
        self.optimizer.step(lr=lr)
        self.composition.specialist.refresh_compute_buffers()
        if profile:
            synchronize()
            timings['optimizer_and_refresh_s'] = perf_counter() - optimizer_start
        self.step += 1
        self.supervised_tokens += total_valid
        report = {'step': self.step, 'supervised_tokens': self.supervised_tokens,
                  'supervised_tokens_step': total_valid,
                  'loss': total_loss / total_valid,
                  'grad_norm': float(_array_to_float(norm_backend)), 'lr': lr}
        if foundation_train_ce:
            report['foundation_ce'] = total_foundation_loss / total_valid
            report['foundation_gain_ce'] = (total_foundation_loss - total_loss) / total_valid
        if source_metrics:
            report['per_source'] = source_records
        if length_bucket_stats is not None:
            report['length_bucket'] = length_bucket_stats
        if profile or measure_step_time:
            synchronize()
            elapsed = perf_counter() - step_start
            report.update({
                'step_time_s': elapsed,
                'executed_tokens_step': executed_tokens,
                'capacity_tokens_step': capacity_tokens,
                'target_fill': total_valid / max(capacity_tokens, 1),
                'supervised_tokens_per_s': total_valid / max(elapsed, 1e-12),
                'executed_tokens_per_s': executed_tokens / max(elapsed, 1e-12),
                'capacity_tokens_per_s': capacity_tokens / max(elapsed, 1e-12),
            })
        if profile:
            profile_info = dict(timings)
            profile_info['total_step_s'] = elapsed
            profile_info['supervised_tokens_per_s'] = total_valid / elapsed
            if BACKEND_NAME == 'cupy':
                pool = xp.get_default_memory_pool()
                # Sampled endpoint values, NOT the maximum lifetime allocation.
                profile_info['gpu_pool_used_mib_at_step_end'] = pool.used_bytes() / 1048576
                profile_info['gpu_pool_reserved_mib_at_step_end'] = pool.total_bytes() / 1048576
            report['profile'] = profile_info
        return report

    def train_steps(self, steps, *, on_step=None, profile_steps=0,
                    source_metrics=False, foundation_train_ce=False,
                    measure_step_time=False, retain_reports=True):
        if steps < 0 or profile_steps < 0:
            raise ValueError('steps and profile_steps must be nonnegative')
        reports = []
        for i in range(steps):
            record = self.train_step(
                profile=i < profile_steps, source_metrics=source_metrics,
                foundation_train_ce=foundation_train_ce,
                measure_step_time=measure_step_time)
            if retain_reports:
                reports.append(record)
            if on_step is not None:
                on_step(record)
        return reports

    def evaluate(self, *, batches=4):
        if batches <= 0:
            raise ValueError('evaluation requires at least one batch')
        losses, total = 0.0, 0
        for _ in range(batches):
            inputs, targets, metadata = self.sampler.next_batch(split='val')
            valid = int(np.asarray(metadata['target_loss_mask']).sum())
            loss = self.composition.loss(
                xp.asarray(inputs, dtype=xp.int32),
                xp.asarray(targets, dtype=xp.int32),
                loss_mask=metadata['target_loss_mask'],
                dual_resolution_metadata=metadata['dual_resolution_metadata'],
                chunk_tokens=self.config.head_chunk_tokens,
                return_cache=False)
            losses += float(_array_to_float(loss)) * valid
            total += valid
        return losses / total

    def evaluate_paired(self, *, batches=4):
        """Teacher-forced CE for frozen and corrected models on *identical* batches.

        Validation RNG/cursors are restored after evaluation so repeated calls
        at subsequent checkpoints cover the same held-out sequences.
        """
        if batches <= 0:
            raise ValueError('evaluation requires at least one batch')
        snapshot = self.sampler.checkpoint_state()
        base_sum = corrected_sum = 0.0
        total = 0
        try:
            for _ in range(batches):
                inp, tgt, meta = self.sampler.next_batch(split='val')
                valid = int(np.asarray(meta['target_loss_mask']).sum())
                inp = xp.asarray(inp, dtype=xp.int32)
                tgt = xp.asarray(tgt, dtype=xp.int32)
                h, target_slice = self.composition.foundation.hidden(
                    inp, dual_resolution_metadata=meta['dual_resolution_metadata'])
                base, _ = self.composition.foundation.head_loss(
                    h, tgt, loss_mask=meta['target_loss_mask'],
                    target_slice=target_slice, chunk_tokens=self.config.head_chunk_tokens)
                correction = self.composition.specialist.forward_delta(
                    h, dual_resolution_metadata=meta['dual_resolution_metadata'],
                    return_cache=False)
                predicted = self.composition.specialist.corrected_hidden(h, correction)
                updated, _ = self.composition.foundation.head_loss(
                    predicted, tgt, loss_mask=meta['target_loss_mask'],
                    target_slice=target_slice, chunk_tokens=self.config.head_chunk_tokens)
                base_sum += float(_array_to_float(base)) * valid
                corrected_sum += float(_array_to_float(updated)) * valid
                total += valid
        finally:
            self.sampler.restore(snapshot)
        return {'foundation_ce': base_sum / total, 'specialist_ce': corrected_sum / total,
                'gain_ce': (base_sum-corrected_sum)/total,
                'evaluated_tokens': total}

    def save_checkpoint(self):
        if self.checkpoint_dir is None:
            raise ValueError('no specialist checkpoint directory configured')
        from dataclasses import asdict
        save_specialist_checkpoint(
            self.checkpoint_dir, self.composition,
            optimizer_state=self.optimizer.checkpoint_state(),
            training_state={'schema': 'specialist-training-v1',
                            'run_config': asdict(self.config),
                            'step': self.step,
                            'supervised_tokens': self.supervised_tokens,
                            'source_tokens': self.source_tokens,
                            'sampler': self.sampler.checkpoint_state()})

    @classmethod
    def resume(cls, checkpoint_dir, foundation, sampler, *, config):
        composition, optimizer_state, state = load_specialist_checkpoint(
            checkpoint_dir, foundation, verify_foundation=True)
        from dataclasses import asdict
        if state.get('schema') != 'specialist-training-v1' or state.get('run_config') != asdict(config):
            raise ValueError('specialist training configuration differs from checkpoint')
        # Validate sampler state before constructing a trainable run.
        sampler.restore(state['sampler'])
        run = cls(composition, sampler, config=config, checkpoint_dir=checkpoint_dir)
        if optimizer_state is None:
            raise ValueError('specialist resume requires optimizer state')
        run.optimizer.restore_checkpoint(optimizer_state, from_hybrid=True)
        if int(state['step']) != run.optimizer.step_index:
            raise ValueError('optimizer update count differs from training state')
        run.step = int(state['step'])
        run.supervised_tokens = int(state['supervised_tokens'])
        run.source_tokens = dict(state['source_tokens'])
        return run
