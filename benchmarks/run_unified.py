#!/usr/bin/env python3
"""0103G unified generation / end-to-end speed benchmark for model registry entries.

Reuses generation_prompts_v1.json and the 0103E generation session; accepts
native, supported in-process GGUF/Safetensors, and remote OpenAI-compatible
registry models. Legacy teacher-forced/correctness suites remain under
run_checkpoint_suite.py, without changes to their scoring semantics.

Example:
 MINI_LLM_BACKEND=cupy python -m benchmarks.run_unified \\
   --registry serving_models.json --model native-348m \\
   --prompt-ids math_general_short --tokens 128 --output-dir benchmarks/results/0103g
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from benchmarks.generation_metrics import repetition_metrics
from mini_llm.evaluation import EvaluationEngine
from mini_llm.model_sources import ModelCatalog, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / 'benchmarks' / 'generation_prompts_v1.json'
FORMAT_VERSION = '0103G-unified-v1'


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def prompts_from_file(path: Path, selected: set[str] | None = None) -> tuple[dict, ...]:
    data = json.loads(path.read_text(encoding='utf-8'))
    prompts = data.get('prompts') if isinstance(data, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('Prompt suite must contain a nonempty prompts array')
    seen = set()
    chosen = []
    for record in prompts:
        if not isinstance(record, dict) or any(not isinstance(record.get(k), str) or not record[k]
                                                for k in ('id', 'prompt', 'domain')):
            raise ValueError('Each prompt needs nonempty id, prompt, and domain')
        if record['id'] in seen:
            raise ValueError(f'Duplicate prompt ID: {record["id"]}')
        seen.add(record['id'])
        if selected is None or record['id'] in selected:
            chosen.append(record)
    if selected is not None and (selected - seen):
        raise ValueError(f'Unknown prompt IDs: {sorted(selected - seen)}')
    if not chosen:
        raise ValueError('No prompts selected')
    return tuple(chosen)


def checkpoint_identity(spec: ModelSpec) -> dict:
    """Record source identity, without claiming cryptographic weight integrity.

    The saved file metadata detects ordinary mutations across --resume. It is
    NOT an independent hash of model tensor contents. Snapshot SHA256 manifest
    can be independently checked with verify_checkpoint_0103b.py.
    """
    descriptor = {'model_id': spec.id, 'format': spec.resolved_format(),
                  'execution_policy': spec.execution_policy,
                  'context_length': spec.context_length}
    if spec.resolved_format() == 'remote':
        descriptor['provider_endpoint'] = spec.api_url
        descriptor['provider_model'] = spec.remote_model
        descriptor['identity_strength'] = 'provider_alias_only; upstream weights may change'
    else:
        path = spec.path.resolve()
        descriptor['path'] = str(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
        # No model tensor loads / hashing multi-gigabyte checkpoints.
        stats = []
        for file in files:
            if file.is_symlink():
                continue
            s = file.stat()
            stats.append((str(file.relative_to(path) if path.is_dir() else file.name),
                          s.st_size, s.st_mtime_ns))
        descriptor['source_files_metadata_sha256'] = hashlib.sha256(
            json.dumps(stats, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        descriptor['identity_strength'] = 'file_size_mtime_not_content_hash'
        state = path / 'training_state.json' if path.is_dir() else None
        if state and state.is_file():
            try:
                value = json.loads(state.read_text(encoding='utf-8'))
                descriptor['optimizer_step'] = value.get('step')
                descriptor['supervised_tokens'] = value.get('supervised_tokens_total')
            except (ValueError, OSError):
                descriptor['training_state_readable'] = False
    return descriptor


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(s.strip()) for s in value.split(','))
    except ValueError as exc:
        raise ValueError('Seeds must be comma-separated integers') from exc
    if not result or len(result) != len(set(result)):
        raise ValueError('Seeds must be nonempty and unique')
    return result


def _rows(path: Path, valid: set[tuple[str, int]]) -> list[dict]:
    result = []
    seen = set()
    if not path.exists():
        return result
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            row = json.loads(line)  # reject torn writes; never silently skip
            key = (row['prompt_id'], row['seed'])
            if key in seen or key not in valid:
                raise ValueError('Duplicate/unexpected results in resume file')
            seen.add(key)
            result.append(row)
    return result


def _summary(rows: list[dict], expected: int) -> dict:
    grouped = {}
    for row in rows:
        name = row['domain']
        grouped.setdefault(name, []).append(row)
    groups = {}
    for name, items in sorted(grouped.items()):
        rates = [r['performance']['end_to_end_tokens_per_s'] for r in items
                 if r['performance']['end_to_end_tokens_per_s'] is not None]
        groups[name] = {'samples': len(items),
                        'mean_end_to_end_tokens_per_s': statistics.mean(rates) if rates else None,
                        'note': 'E2E includes prompt, sampling, decoding and remote network when applicable.'}
    return {'schema': FORMAT_VERSION, 'completed': len(rows), 'expected': expected,
            'complete': len(rows) == expected, 'domains': groups}


def _run(*, model_spec: ModelSpec, prompts_path: Path, output_dir: Path,
         prompt_ids: set[str] | None, seeds: tuple[int, ...], tokens: int,
         temperature: float, top_k: int | None, top_p: float,
         resume: bool, dry_run: bool, remote_mode: str = "completion") -> dict:
    if tokens <= 0 or temperature < 0 or not math.isfinite(temperature):
        raise ValueError('tokens must be positive; temperature must be finite and nonnegative')
    if not 0 < top_p <= 1 or not math.isfinite(top_p):
        raise ValueError('top_p must be finite in (0,1]')
    if top_k is not None and top_k < 1:
        raise ValueError('top_k must be positive')
    prompts = prompts_from_file(prompts_path, prompt_ids)
    if model_spec.resolved_format() == 'remote' and top_k is not None:
        raise ValueError('Remote providers do not share a guaranteed top-k API')
    if model_spec.resolved_format() == 'remote' and model_spec.execution_policy == 'strict-fast':
        raise ValueError('Strict GPU-fast execution cannot be verified for a remote provider')
    suite_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    settings = {'model': checkpoint_identity(model_spec), 'suite_file': str(prompts_path.resolve()),
                'suite_sha256': suite_hash, 'prompt_ids': [p['id'] for p in prompts],
                'seeds': list(seeds), 'max_new_tokens': tokens,
                'sampling': {'temperature': temperature, 'top_p': top_p, 'top_k': top_k},
                'backend_env': os.environ.get('MINI_LLM_BACKEND', 'numpy'),
                'remote_request_style': remote_mode if model_spec.resolved_format() == 'remote' else None,
                'note': 'CPU/GPU backend is process-global until runtime isolation is implemented.'}
    count = len(prompts) * len(seeds)
    if dry_run:
        return {'expected': count, 'settings': settings}
    manifest = output_dir / 'manifest.json'
    results = output_dir / 'results.jsonl'
    if resume:
        if not manifest.exists():
            raise ValueError('--resume requires a previous manifest')
        prior = json.loads(manifest.read_text(encoding='utf-8'))
        if prior.get('schema') != FORMAT_VERSION or prior.get('settings') != settings:
            raise ValueError('Resume settings or model source metadata changed; refuse to mix results')
    elif manifest.exists() or results.exists():
        raise FileExistsError('Output exists; choose a new output directory or use --resume')
    valid = {(p['id'], seed) for p in prompts for seed in seeds}
    existing = _rows(results, valid) if resume else []
    completed = {(r['prompt_id'], r['seed']) for r in existing}
    if len(existing) == count:
        summary = _summary(existing, count)
        atomic_json(output_dir / 'summary.json', summary)
        return summary

    # Load model only after validating resume compatibility / output paths.
    from mini_llm.serving_models import load_model
    engine = EvaluationEngine(load_model(model_spec), model_id=model_spec.id,
                              strict_fast=model_spec.execution_policy == 'strict-fast',
                              remote_mode=remote_mode)
    if not resume:
        atomic_json(manifest, {'schema': FORMAT_VERSION,
                               'created_utc': datetime.now(timezone.utc).isoformat(),
                               'settings': settings,
                               'metric_conventions': {
                                   'end_to_end_tokens_per_s': 'local count or provider-reported count divided by wall time; not GPU decode throughput',
                                   'prefill_s': 'available for local engine only',
                                   'token_repetition': 'local token IDs only; never inferred from remote text',
                                   'gpu_acceleration': 'only validated when execution report confirms it',
                               }})
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(existing)
    with results.open('a', encoding='utf-8', buffering=1) as fh:
        for prompt in prompts:
            for seed in seeds:
                if (prompt['id'], seed) in completed:
                    continue
                sample = engine.generate(prompt['prompt'], max_new_tokens=tokens,
                                         temperature=temperature, top_k=top_k,
                                         top_p=top_p, seed=seed)
                repetition = repetition_metrics(list(sample.token_ids)) if sample.token_ids is not None else None
                rate = (sample.completion_tokens / sample.wall_s if sample.completion_tokens is not None
                        and sample.wall_s > 0 else None)
                row = {'schema': FORMAT_VERSION, 'prompt_id': prompt['id'], 'domain': prompt['domain'],
                       'seed': seed, 'text': sample.text, 'sample': sample.to_dict(),
                       'quality': {'token_repetition': repetition},
                       'performance': {'end_to_end_wall_s': sample.wall_s,
                                       'end_to_end_tokens_per_s': rate,
                                       'prefill_s': sample.prefill_s,
                                       'decode_s': sample.decode_s,
                                       'decode_tokens_per_s': (sample.completion_tokens / sample.decode_s
                                         if sample.decode_s and sample.completion_tokens is not None else None)}}
                fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
                rows.append(row)
                print(f'[{len(rows)}/{count}] {prompt["id"]} seed={seed} '
                      f'wall={sample.wall_s:.3f}s count={sample.completion_tokens}', flush=True)
    summary = _summary(rows, count)
    atomic_json(output_dir / 'summary.json', summary)
    return summary



@contextmanager
def output_lock(directory: Path):
    """Prevent two local benchmark agents from interleaving JSONL writes.

    Linux/POSIX lock releases automatically on process termination. The lock
    file can remain on disk and does not indicate an active lock by itself.
    """
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.benchmark.lock'
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Benchmark output is already in use: {directory}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_legacy_suite(*, spec: ModelSpec, suite: Path, output_dir: Path,
                     benchmark_ids: tuple[str, ...] | None = None,
                     runtime_profile: str | None = None, dry_run: bool = False) -> dict:
    """Compatibility bridge to existing live suites without rewriting scorers.

    Existing WikiText, coding, retrieval, and other live-suite task semantics
    remain owned by LiveBenchmarkScheduler. Output files keep their historical
    schema; this only supplies a searchable index and common entrypoint.
    """
    if spec.resolved_format() != 'native':
        raise ValueError('Existing live checkpoint suites require a native checkpoint with weights and tokenizer')
    if spec.execution_policy != 'auto':
        raise ValueError('Legacy suite does not implement --execution-policy; use --runtime-profile or the native benchmark CLI')
    if not suite.is_file():
        raise FileNotFoundError(suite)
    config = json.loads(suite.read_text(encoding='utf-8'))
    jobs = config.get('benchmarks')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Legacy benchmark suite requires a benchmarks array')
    if benchmark_ids is not None:
        available = {row['id'] for row in jobs}
        if set(benchmark_ids) - available:
            raise ValueError(f'Unknown legacy benchmark IDs: {sorted(set(benchmark_ids)-available)}')
    identity = checkpoint_identity(spec)
    record = {'schema': FORMAT_VERSION, 'mode': 'existing-live-suite',
              'checkpoint': identity, 'suite_file': str(suite.resolve()),
              'suite_sha256': hashlib.sha256(suite.read_bytes()).hexdigest(),
              'selected_benchmarks': list(benchmark_ids) if benchmark_ids else None,
              'runtime_profile': runtime_profile,
              'note': 'Legacy evaluation output retains its native formats and scoring implementations.'}
    if dry_run:
        return record
    from benchmarks.run_checkpoint_suite import run_checkpoints
    events = run_checkpoints(checkpoints=[spec.path], suite_path=suite,
                             selected=benchmark_ids, output_root=output_dir,
                             runtime_profile=runtime_profile)
    record['events'] = len(events)
    record['completed_events'] = sum(e.get('status') == 'complete' for e in events)
    record['all_reported_events_complete'] = bool(events) and record['completed_events'] == len(events)
    atomic_json(output_dir / 'unified_legacy_index.json', record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--registry', type=Path, help='Registry containing native, pretrained and remote models')
    group.add_argument('--checkpoint', type=Path, help='Direct native checkpoint path')
    parser.add_argument('--model', help='Registry model id (required with --registry)')
    parser.add_argument('--execution-policy', choices=('auto', 'reference', 'balanced', 'fast', 'strict-fast'))
    parser.add_argument('--prompts', type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument('--legacy-suite', type=Path, help='Run existing LiveBenchmarkScheduler suite (native only)')
    parser.add_argument('--benchmark-ids', help='Comma-separated existing live benchmark IDs')
    parser.add_argument('--runtime-profile', help='Legacy suite runtime profile (existing scheduler only)')
    parser.add_argument('--prompt-ids', help='Comma-separated list of prompt IDs')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--remote-mode', choices=('completion', 'chat'), default='completion',
                        help='Text completions preserve raw continuation prompts; chat alters prompt format')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.registry:
        if not args.model:
            parser.error('--model is required with --registry')
        spec = ModelCatalog.from_manifest(args.registry).get(args.model)
        if args.execution_policy:
            from dataclasses import replace
            spec = replace(spec, execution_policy=args.execution_policy)
    else:
        if args.model:
            parser.error('--model requires --registry')
        spec = ModelSpec(id=args.checkpoint.name, path=args.checkpoint.resolve(),
                         format='native', execution_policy=args.execution_policy or 'auto')
    if args.legacy_suite:
        if args.prompt_ids:
            parser.error('--prompt-ids is only valid with generation prompts')
        if args.resume:
            parser.error('--resume applies to generation JSONL only; live suite handles its own outputs')
        if args.registry and spec.resolved_format() != 'native':
            parser.error('--legacy-suite only supports native models')
        if args.dry_run:
            record = run_legacy_suite(spec=spec, suite=args.legacy_suite.resolve(),
                                      output_dir=args.output_dir.resolve(),
                                      benchmark_ids=tuple(args.benchmark_ids.split(',')) if args.benchmark_ids else None,
                                      runtime_profile=args.runtime_profile, dry_run=True)
        else:
            with output_lock(args.output_dir.resolve()):
                record = run_legacy_suite(spec=spec, suite=args.legacy_suite.resolve(),
                                          output_dir=args.output_dir.resolve(),
                                          benchmark_ids=tuple(args.benchmark_ids.split(',')) if args.benchmark_ids else None,
                                          runtime_profile=args.runtime_profile, dry_run=False)
        print(json.dumps(record, indent=2))
        return 0 if args.dry_run or record['all_reported_events_complete'] else 1
    if args.benchmark_ids or args.runtime_profile:
        parser.error('--benchmark-ids and --runtime-profile require --legacy-suite')
    selected = set(args.prompt_ids.split(',')) if args.prompt_ids else None
    options = dict(model_spec=spec, prompts_path=args.prompts.resolve(),
                   output_dir=args.output_dir.resolve(), prompt_ids=selected,
                   seeds=_parse_seeds(args.seeds), tokens=args.tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   resume=args.resume, dry_run=args.dry_run, remote_mode=args.remote_mode)
    if args.dry_run:
        summary = _run(**options)
    else:
        with output_lock(args.output_dir.resolve()):
            summary = _run(**options)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
