#!/usr/bin/env python3
"""Check a version-1 MyLocal_LLM streaming registry before enabling a new data phase.

Runs on the user's machine with network access. Does not train or download the full
corpora. Fetches a small streamed sample and validates pinned revisions/fields.
Usage: python preflight_stream_registry.py configs/streaming_sources.json
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', type=Path)
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--segment-tokens', type=int, default=524288)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error('--examples must be >= 1')
    data = json.loads(args.registry.read_text())
    if data.get('version') != 1:
        raise ValueError('expected version=1 registry')
    sources = data['sources']
    weights = sum(float(x['weight']) for x in sources)
    print(f'{len(sources)} remote sources, total sampling weight {weights:.1%}; local weight {1-weights:.1%}')
    if not (0 < weights < 1):
        raise ValueError('remote sampling weights must sum to (0,1)')

    from huggingface_hub import HfApi
    from datasets import load_dataset
    api = HfApi()
    failed = 0
    for source in sources:
        name = source['name']
        print(f'\n{name} ({source["dataset"]})', flush=True)
        try:
            pinned = source['revision']
            info = api.dataset_info(source['dataset'], revision=pinned)
            if info.sha != pinned:
                raise ValueError(f'commit mismatch: expected {pinned}, got {info.sha}')
            stream = load_dataset(
                source['dataset'], name=source.get('config'),
                split=source['split'], revision=pinned, streaming=True,
            )
            observed = 0
            for row in stream.take(args.examples):
                value = row.get(source['text_field'])
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'missing or empty text field {source["text_field"]!r}; keys={list(row)}')
                chars = len(value)
                if chars > 16 * args.segment_tokens:
                    raise ValueError(f'first examples include oversized doc {chars:,} chars (> text cap)')
                print(f'  example {observed+1}: {chars:,} chars; fields={list(row)[:8]}')
                observed += 1
            if observed == 0:
                raise ValueError('empty stream')
            print('  OK: pinned SHA, config, split, and selected field')
        except Exception as exc:
            failed += 1
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)
    if failed:
        print(f'\n{failed} source(s) failed; do not start main run.', file=sys.stderr)
        return 1
    print('\nAll sampled streams passed. Long-document lengths and steady-state rate remain unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
