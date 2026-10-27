#!/usr/bin/env python3
"""0093: evaluate one or more saved native checkpoints with the SAME JSON suite
used by the live training scheduler, without writing any model snapshots.

Run from the repository root:
  MINI_LLM_BACKEND=cupy .venv/bin/python -m benchmarks.run_checkpoint_suite \\
    --checkpoint ./checkpoints/foundation_16k --suite benchmarks/configs/foundation_capabilities_0093.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from benchmarks.live_suite import LiveBenchmarkScheduler


def _checkpoint_identity(path):
    """Fast disk-identity tag, including parameter-file sizes/mtimes.

    This is NOT a cryptographic model-weights hash; checkpoint directory must
    remain immutable while testing. The full path is also recorded.
    """
    path = Path(path).resolve()
    if not (path/'config.json').is_file():
        raise FileNotFoundError(f'Not a native model checkpoint: {path}')
    info = {'path': str(path), 'files': []}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix in ('.npz', '.npy', '.safetensors', '.json'):
            stat = item.stat()
            info['files'].append((str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]


def _token_total(checkpoint, token_map=None, explicit=None):
    state_path = checkpoint/'training_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    if explicit is not None:
        return float(explicit), 'explicit_checkpoint_override', state
    if token_map is not None:
        key = str(checkpoint.resolve())
        for candidate in (key, str(checkpoint), checkpoint.name):
            if candidate in token_map:
                return float(token_map[candidate]), 'supplied_checkpoint_token_map', state
    total = state.get('supervised_tokens_total')
    if total is not None:
        return float(total), state.get('supervised_tokens_provenance', 'checkpoint'), state
    return 0., 'unknown_legacy_checkpoint', state


def run_checkpoints(*, checkpoints, suite_path, selected=None, output_root=None,
                    token_map=None, supervised_tokens=None, runtime_profile=None):
    from benchmarks.evaluate_generation import load_generator
    from mini_llm.backend import BACKEND_NAME, xp

    if len(checkpoints) > 1 and supervised_tokens is not None:
        raise ValueError('--supervised-tokens requires a single checkpoint; use --token-map for multiple')
    events = []
    for checkpoint_arg in checkpoints:
        checkpoint = Path(checkpoint_arg).resolve()
        checkpoint_id = _checkpoint_identity(checkpoint)
        total, provenance, state = _token_total(checkpoint, token_map, supervised_tokens)
        if total < 0 or not (total < float('inf')):
            raise ValueError('Invalid supervised-token total')
        generator, info = load_generator(checkpoint, None, runtime_profile)
        try:
            step = int(info['step'] if info['step'] is not None else state.get('step', 0))
            scheduler = LiveBenchmarkScheduler(
                json.loads(Path(suite_path).read_text(encoding='utf-8')),
                generator.tokenizer, int(total), suite_path=suite_path)
            if output_root:
                scheduler.output_root = Path(output_root)
            events += scheduler.run_selected(
                generator.model, step=step, supervised_tokens_total=total,
                provenance=provenance, benchmark_ids=selected, checkpoint_id=checkpoint_id,
                checkpoint_path=str(checkpoint))
        finally:
            del generator
            gc.collect()
            if BACKEND_NAME == 'cupy':
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', type=Path, required=True,
                        help='Repeat to evaluate several checkpoints sequentially')
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--benchmarks', default=None,
                        help='Comma-separated benchmark IDs; default: all enabled')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--token-map', type=Path, default=None,
                        help='JSON mapping checkpoint directory names/paths to verified supervised-token totals')
    parser.add_argument('--supervised-tokens', type=float, default=None,
                        help='For ONE legacy checkpoint only; historical total supplied by user')
    parser.add_argument('--runtime-profile', default=None)
    args = parser.parse_args(argv)
    if not args.suite.is_file():
        parser.error(f'Suite file not found: {args.suite}')
    mapping = json.loads(args.token_map.read_text(encoding='utf-8')) if args.token_map else None
    if mapping is not None and not isinstance(mapping, dict):
        parser.error('Token map JSON must be an object')
    requested = args.benchmarks.split(',') if args.benchmarks else None
    rows = run_checkpoints(
        checkpoints=args.checkpoint, suite_path=args.suite, selected=requested,
        output_root=args.output_root, token_map=mapping,
        supervised_tokens=args.supervised_tokens, runtime_profile=args.runtime_profile)
    print(f'Completed {sum(event.get("status") == "complete" for event in rows)} / {len(rows)} benchmark results.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
