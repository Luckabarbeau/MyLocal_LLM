"""0092: in-process, configuration-driven evaluation for autoregressive training.

No model checkpoint is created. Full WikiText-2 windowed and production
TextGenerator evaluations share the live trained model. Any GPU exception is
fatal by default; partial JSON cannot be mistaken for a completed evaluation.
"""
from __future__ import annotations

import contextlib
import dataclasses
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from mini_llm.backend import xp, asnumpy


SUPPORTED = {'wikitext', 'generation', 'reference_continuation'}


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def _plain_4096_model(model):
    """Temporarily bypass historical_read without allocating a second backbone.

    Verified only for historical_read and dense models; do not silently alter
    other hierarchical architectures. Training mode is always restored.
    """
    memory = getattr(model, 'historical_memory', None)
    original_config = model.config
    if getattr(model, 'hierarchical_memory_enabled', False):
        cfg = original_config.memory_context
        if not (memory is not None and cfg.integration_mode == 'historical_read'):
            raise NotImplementedError('0092 plain live evaluation supports historical_read or no memory')
    try:
        if memory is not None:
            model.historical_memory = None
            model.config = dataclasses.replace(original_config,
                memory_context=dataclasses.replace(original_config.memory_context, enabled=False))
        yield
    finally:
        model.historical_memory = memory
        model.config = original_config


@contextlib.contextmanager
def _preserve_global_rng():
    """Isolate seeded evaluation: swap CuPy's global RandomState object.

    CuPy RandomState need not offer get_state/set_state. Saving the original
    generator *object* and switching the global pointer to a fresh generator
    means TextGenerator.generate(seed=...) mutates only the temporary state.
    """
    np_state = np.random.get_state()
    original_gpu_state = None
    if xp is not np:
        original_gpu_state = xp.random.get_random_state()
        xp.random.set_random_state(xp.random.RandomState(0))
    try:
        yield
    finally:
        np.random.set_state(np_state)
        if original_gpu_state is not None:
            xp.random.set_random_state(original_gpu_state)


def run_wikitext(model, tokenizer, *, data, context_length, stride=None,
                 document_mode='continuous', max_tokens=None):
    from benchmarks.evaluate_wikitext import tokenize_eval_streams, suffix_nll
    path = Path(data)
    raw = path.read_text(encoding='utf-8')
    context = int(context_length)
    stride = context if stride is None else int(stride)
    if not 1 <= stride <= context:
        raise ValueError('WikiText stride must be in [1,context]')
    streams, token_count, _ = tokenize_eval_streams(
        raw, tokenizer, document_mode=document_mode, max_tokens=max_tokens,
    )
    nll, scored, windows = 0.0, 0, 0
    with _plain_4096_model(model):
        for stream in streams:
            if len(stream) < 2:
                continue
            for target_start in range(1, len(stream), stride):
                target_end = min(target_start + stride, len(stream))
                begin = max(0, target_end - context - 1)
                arr = np.asarray(stream[begin:target_end], dtype=np.int64)
                inputs = xp.asarray(arr[:-1][None, :]); targets = xp.asarray(arr[1:][None, :])
                logits = model.forward(inputs, return_cache=False)
                current_nll, current_count = suffix_nll(logits, targets, target_start - (begin+1))
                nll += current_nll
                scored += current_count
                windows += 1
                del inputs, targets, logits
    if scored == 0:
        raise ValueError('WikiText benchmark had no scoreable tokens')
    ce = nll / scored
    return {
        'cross_entropy': ce, 'perplexity': math.exp(ce),
        'bits_per_byte': (nll / (len(raw.encode('utf-8')) * math.log(2))
                          if max_tokens is None else None),
        'scored_tokens': scored, 'tokenizer_tokens': token_count,
        'windows': windows, 'context_length': context, 'stride': stride,
        'memory_mode': 'plain', 'document_mode': document_mode,
        'dataset_sha256': _sha256(path),
    }


def _generator(model, tokenizer, *, max_context=None):
    from inference import TextGenerator
    memory_cfg = model.config.memory_context
    if memory_cfg.enabled and memory_cfg.integration_mode in {
        'historical_read', 'terminal_landmark', 'routed_prefix'
    }:
        horizon = int(memory_cfg.memory_length)
    else:
        horizon = int(model.config.context_length)
    return TextGenerator(model, tokenizer, max_context=min(horizon, int(max_context)) if max_context else horizon)


def run_generation(model, tokenizer, *, prompt_suite, prompt_ids=None, seeds=(42,),
                   strategies=('greedy', 'top-k'), max_new_tokens=128, temperature=0.8,
                   top_k=40, top_p=1.0, use_kv_cache=True):
    from benchmarks.evaluate_generation import select_prompts
    from benchmarks.generation_metrics import repetition_metrics, grouped_summary
    suite_path = Path(prompt_suite)
    suite = json.loads(suite_path.read_text(encoding='utf-8'))
    prompts = select_prompts(suite, ','.join(prompt_ids) if prompt_ids else None, None, None)
    generator = _generator(model, tokenizer)
    records = []
    with _preserve_global_rng():
        for prompt in prompts:
            for seed in seeds:
                for strategy in strategies:
                    text_out, timing = generator.generate(
                        prompt=prompt['prompt'], max_new_tokens=int(max_new_tokens),
                        strategy=strategy, temperature=float(temperature),
                        top_k=int(top_k) if strategy == 'top-k' else None,
                        top_p=float(top_p) if strategy == 'top-p' else None,
                        seed=int(seed), use_kv_cache=use_kv_cache,
                        return_token_ids=True,
                    )
                    raw_ids = timing['generated_token_ids']
                    ended = bool(raw_ids and raw_ids[-1] == timing['eos_token_id'])
                    ids = raw_ids[:-1] if ended else raw_ids
                    records.append({
                        'prompt': prompt, 'seed': int(seed),
                        'decoding': {'strategy': strategy, 'temperature': temperature,
                                     'top_k': top_k if strategy == 'top-k' else None,
                                     'top_p': top_p if strategy == 'top-p' else None},
                        'generated_text': text_out, 'generated_token_ids': ids,
                        'stopped_on_eos': ended, 'metrics': repetition_metrics(ids),
                    })
    # Force recreation at each future checkpoint: cached InferenceModel copies
    # must never accidentally carry earlier step's weights.
    del generator
    gc.collect()
    return {'summary': grouped_summary(records), 'records': records,
            'prompt_suite_sha256': _sha256(suite_path)}


def run_reference(model, tokenizer, *, references, top_k_predictions=10,
                  cutoffs=(0.25, 0.5, 0.75), generation_tokens=128,
                  seeds=(42,), strategy='top-k', temperature=0.8,
                  top_k=40, use_kv_cache=True, branch_top_k=0, max_context=4096):
    from benchmarks.reference_continuation import analyze_reference, load_references
    cases, digest = load_references(references)
    results = []
    with _preserve_global_rng():
        with _plain_4096_model(model):
            # Build a fresh plain-context KV inference model, backed by the
            # live training weights. Do not mutate the frozen training memory
            # configuration outside this context manager.
            generator = _generator(model, tokenizer, max_context=max_context)
            for ref in cases:
                result = analyze_reference(
                    model, tokenizer, ref['text'], top_k=int(top_k_predictions),
                    max_context=int(max_context), cutoffs=cutoffs,
                    max_new_tokens=int(generation_tokens), generator=generator,
                    seeds=seeds, strategy=strategy, temperature=float(temperature),
                    sampling_top_k=int(top_k), use_kv_cache=use_kv_cache,
                    branches=int(branch_top_k),
                )
                results.append({'id': ref['id'], 'category': ref.get('category'), **result})
            del generator
    gc.collect()
    return {'references_sha256': digest, 'cases': results}


class LiveBenchmarkScheduler:
    """Schedule each benchmark independently by cumulative supervised tokens."""

    def __init__(self, suite, tokenizer, initial_tokens, *, suite_path):
        self.suite = suite
        self.tokenizer = tokenizer
        self.suite_path = Path(suite_path)
        self.config_hash = _sha256(suite_path)
        self.output_root = Path(suite.get('output_root', './benchmarks/results/live'))
        self.last_step = None
        self.next_due = {}
        if suite.get('version') != 1:
            raise ValueError('Benchmark suite version must be 1')
        if suite.get('on_error', 'raise') not in ('raise', 'continue'):
            raise ValueError('on_error must be raise or continue')
        benchmarks = suite.get('benchmarks', [])
        if not isinstance(benchmarks, list) or not benchmarks:
            raise ValueError('benchmarks must be a nonempty list')
        seen = set()
        for benchmark in benchmarks:
            name = benchmark.get('id')
            kind = benchmark.get('type')
            if not name or name in seen or kind not in SUPPORTED:
                raise ValueError(f'Unsupported or duplicate benchmark: {benchmark}')
            seen.add(name)
            if not benchmark.get('enabled', True):
                continue
            token_interval = benchmark.get('every_supervised_tokens')
            step_interval = benchmark.get('every_optimizer_steps')
            if (token_interval is None) == (step_interval is None):
                raise ValueError(f'{name} needs exactly one scheduling interval')
            amount = int(token_interval if token_interval is not None else step_interval)
            if amount <= 0:
                raise ValueError(f'{name} needs a positive scheduling interval')
            params = benchmark.get('parameters', {})
            if not isinstance(params, dict):
                raise ValueError(f'{name} parameters must be an object')
            allowed = {
                'wikitext': {'data', 'context_length', 'stride', 'document_mode', 'max_tokens'},
                'generation': {'prompt_suite', 'prompt_ids', 'seeds', 'strategies',
                               'max_new_tokens', 'temperature', 'top_k', 'top_p', 'use_kv_cache'},
                'reference_continuation': {'references', 'top_k_predictions', 'cutoffs',
                    'generation_tokens', 'seeds', 'strategy', 'temperature', 'top_k',
                    'use_kv_cache', 'branch_top_k', 'max_context'},
            }[kind]
            if set(params) - allowed:
                raise ValueError(f'{name} unknown parameters: {sorted(set(params)-allowed)}')
            if kind == 'wikitext' and not {'data', 'context_length'} <= set(params):
                raise ValueError(f'{name}: wikitext requires data/context_length')
            if kind == 'generation' and 'prompt_suite' not in params:
                raise ValueError(f'{name}: generation requires prompt_suite')
            if kind == 'reference_continuation' and 'references' not in params:
                raise ValueError(f'{name}: reference_continuation requires references')
            if token_interval is not None:
                self.next_due[name] = {'unit': 'supervised_tokens',
                    'value': (int(initial_tokens) // amount + 1) * amount}
            else:
                self.next_due[name] = {'unit': 'optimizer_steps', 'value': amount}

    @classmethod
    def from_file(cls, path, tokenizer, trainer):
        path = Path(path)
        suite = json.loads(path.read_text(encoding='utf-8'))
        scheduler = cls(suite, tokenizer, trainer.supervised_tokens_total, suite_path=path)
        for item in suite['benchmarks']:
            name = item['id']
            if name in scheduler.next_due and scheduler.next_due[name]['unit'] == 'optimizer_steps':
                amount = int(item['every_optimizer_steps'])
                scheduler.next_due[name]['value'] = (int(trainer.step) // amount + 1) * amount
        return scheduler

    def maybe_run(self, trainer):
        total = float(trainer.supervised_tokens_total)
        for benchmark in self.suite['benchmarks']:
            name = benchmark['id']
            if name not in self.next_due:
                continue
            due = self.next_due[name]
            current = int(trainer.step) if due['unit'] == 'optimizer_steps' else total
            if current < due['value']:
                continue
            amount = int(benchmark['every_optimizer_steps'] if due['unit'] == 'optimizer_steps'
                         else benchmark['every_supervised_tokens'])
            crossed = due['value']
            due['value'] = (int(current) // amount + 1) * amount
            dest = self.output_root / f'step_{trainer.step:08d}' / name
            if (dest / 'result.json').exists():
                continue
            event = {
                'benchmark_id': name, 'benchmark_type': benchmark['type'],
                'optimizer_step': int(trainer.step), 'supervised_tokens_total': total,
                'supervised_tokens_provenance': trainer.supervised_tokens_provenance,
                'threshold_crossed': crossed, 'threshold_unit': due['unit'],
                'suite_hash': self.config_hash,
                'configuration': benchmark,
            }
            print(f'[0092] Live benchmark {name} at step={trainer.step} tokens={total:,.0f}', flush=True)
            before = time.perf_counter()
            try:
                kind = benchmark['type']
                fn = {'wikitext': run_wikitext, 'generation': run_generation,
                      'reference_continuation': run_reference}[kind]
                result = fn(trainer.model, self.tokenizer, **benchmark.get('parameters', {}))
                from benchmarks.live_report import (summary_metrics, reference_html,
                                                    rebuild_summary_csv)
                event.update(status='complete', results=result,
                             summary_metrics=summary_metrics(kind, result),
                             elapsed_seconds=time.perf_counter() - before)
                _atomic_json(dest / 'result.json', event)
                if kind == 'reference_continuation':
                    (dest / 'report.html').write_text(reference_html(result), encoding='utf-8')
                rebuild_summary_csv(self.output_root, self.config_hash)
            except Exception as exc:
                event.update(status='error', error=f'{type(exc).__name__}: {exc}',
                             elapsed_seconds=time.perf_counter() - before)
                _atomic_json(dest / 'error.json', event)
                print(f'[0092] Benchmark {name} failed: {exc}', flush=True)
                if self.suite.get('on_error', 'raise') == 'raise':
                    raise
            finally:
                gc.collect()
