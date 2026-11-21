#!/usr/bin/env python3
"""Offline, causal request-boundary placement diagnostic; never moves model weights.

CLI: --input result.json --output NEW.json [--max-swaps 8] [--max-transfer-mib 32].
Geometry defaults to 48 layers / 512 experts / top10; synthetic fixtures may use
--layers and --experts. Each A->B proposal starts from recorded residency, uses
only A's request delta, and is evaluated on B's unchanged recorded trajectory.
This is NOT a rolling simulated model, decode-only attribution or a rate estimate.

Unlike analyze_0295_route_heat.decode_frame, this decoder rejects noninteger
counters instead of coercing them and imports no NumPy/backend/model modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

MAX_SOURCE_BYTES = 64 * 1024**2
MAX_SWAPS = 8
MAX_TRANSFER_BYTES = 32 * 1024**2
TOP_K = 10
MAX_COUNTER = 2**64 - 1


def _integer(value, name, minimum=0, maximum=MAX_COUNTER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')
    return value


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'Missing/invalid {name} object')
    return value


def _layout(cache):
    slots = _object(cache.get('slots'), 'sized slots')
    classes = _object(slots.get('size_classes'), 'size_classes')
    if not classes or len(classes) > 64:
        raise ValueError('Require 1..64 sized-slot classes')
    result = {}
    for text, entry in classes.items():
        if not isinstance(text, str) or not text.isascii() or not text.isdecimal():
            raise ValueError('Slot class must be a canonical integer byte string')
        size = int(text)
        if str(size) != text or size <= 0 or size % 256:
            raise ValueError('Slot class must be positive and 256-byte aligned')
        entry = _object(entry, 'slot class')
        capacity = _integer(entry.get('capacity'), 'class capacity', 1)
        free = _integer(entry.get('free'), 'class free', 0, capacity)
        result[size] = (capacity, free)
    if _integer(slots.get('capacity'), 'slot capacity', 1) != sum(v[0] for v in result.values()):
        raise ValueError('Slot capacities disagree')
    reserved = sum(size * v[0] for size, v in result.items())
    if _integer(slots.get('device_reserved_bytes'), 'device_reserved_bytes', 1) != reserved:
        raise ValueError('Slot physical byte ledger disagrees')
    return dict(sorted(result.items()))


def _slot_class(size, layout):
    # sized_slot_layout rounds bundles to 256 bytes. reservation_size chooses
    # the smallest existing fitting class, never an arbitrary equal-stride slot.
    aligned = (size + 255) // 256 * 256
    for capacity in layout:
        if aligned <= capacity:
            return capacity
    raise ValueError('Complete expert bundle has no fitting slot class')


def decode_snapshot(cache, *, layers=48, experts=512):
    """Validate one complete cumulative native snapshot and its sized-slot ledger."""
    _integer(layers, 'layers', 1, 64)
    _integer(experts, 'experts', TOP_K, 512)
    cache = _object(cache, 'cache')
    layout = _layout(cache)
    rows = cache.get('native_handoffs')
    if not isinstance(rows, list) or len(rows) != layers:
        raise ValueError('Incomplete native layer profile')
    indexed = {}
    for row in rows:
        row = _object(row, 'native handoff')
        layer = _integer(row.get('layer'), 'layer', 0, layers - 1)
        if layer in indexed:
            raise ValueError('Duplicate layer profile')
        indexed[layer] = row
    decoded = []
    occupied = Counter()
    for layer in range(layers):
        row = indexed[layer]
        calls = _integer(row.get('calls'), 'calls')
        heat = row.get('route_heat')
        if not isinstance(heat, list) or len(heat) != experts:
            raise ValueError('Missing/incomplete route_heat geometry')
        heat = tuple(_integer(v, 'route_heat') for v in heat)
        if sum(heat) != TOP_K * calls or any(v > calls for v in heat):
            raise ValueError('Cumulative heat must equal top_k(10) * calls with unique routes')
        ids = row.get('profile_resident_ids')
        if not isinstance(ids, list):
            raise ValueError('Missing profile_resident_ids')
        ids = tuple(_integer(v, 'resident ID', 0, experts - 1) for v in ids)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate resident ID')
        size = _integer(row.get('expert_packed_bytes'), 'expert_packed_bytes', 1)
        slot_class = _slot_class(size, layout)
        formats = row.get('formats')
        if formats is not None:
            if (not isinstance(formats, list) or len(formats) != 3 or
                    any(not isinstance(v, str) or not v.strip() for v in formats)):
                raise ValueError('formats must name gate/up/down, or be absent')
            formats = tuple(formats)
        occupied[slot_class] += len(ids)
        decoded.append(dict(calls=calls, heat=heat, resident=frozenset(ids),
                            packed_bytes=size, slot_class_bytes=slot_class, formats=formats))
    for size, (capacity, free) in layout.items():
        if occupied[size] != capacity - free:
            raise ValueError('Recorded residents disagree with sized-slot occupancy')
    if len({r['calls'] for r in decoded}) != 1:
        raise ValueError('Layer call counts disagree')
    return dict(rows=decoded, layout=layout)


def _same_geometry(a, b):
    if a['layout'] != b['layout']:
        raise ValueError('Sized-slot layout/occupancy changed')
    for left, right in zip(a['rows'], b['rows']):
        for key in ('resident', 'packed_bytes', 'slot_class_bytes', 'formats'):
            if left[key] != right[key]:
                raise ValueError(f'Recorded residents/geometry changed: {key}')


def _delta(before, after):
    _same_geometry(before, after)
    counts = []
    for left, right in zip(before['rows'], after['rows']):
        calls = right['calls'] - left['calls']
        heat = tuple(y - x for x, y in zip(left['heat'], right['heat']))
        if calls < 0 or any(v < 0 for v in heat):
            raise ValueError('Native counters reset or regressed')
        if sum(heat) != TOP_K * calls or any(v > calls for v in heat):
            raise ValueError('Heat delta must equal top_k(10) * calls delta with unique routes')
        counts.append(heat)
    return counts


def _new_handoffs_before(cache, after_cache, after):
    """Initial [] means handoffs did not yet exist, not a missing partial profile."""
    if _layout(cache) != after['layout']:
        raise ValueError('Initial sized-slot layout changed')
    for name in ('promotions', 'evictions'):
        left = _integer(_object(cache.get('device'), 'device').get(name), name)
        right = _integer(_object(after_cache.get('device'), 'device').get(name), name)
        if left != right:
            raise ValueError('Cannot infer initial residents across placement changes')
    return dict(layout=after['layout'], rows=[dict(r, calls=0, heat=(0,) * len(r['heat']))
                                            for r in after['rows']])


def request_profiles(result, *, layers=48, experts=512):
    """Decode ordered cases, including repeated request IDs, without mutating input."""
    cases = _object(result, 'result').get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Missing cases')
    profiles, skipped = [], []
    previous = None
    for index, case in enumerate(cases):
        case = _object(case, 'case')
        name = case.get('request_id')
        if not isinstance(name, str) or not name:
            raise ValueError('Missing request_id')
        before_cache = _object(case.get('cache_before'), 'cache_before')
        after_cache = _object(case.get('cache_after'), 'cache_after')
        # The ordinary adaptive warmup has no native plans/counters. It cannot
        # train a native-placement proposal and is explicitly excluded.
        if (previous is None and name == 'warmup' and
                before_cache.get('native_handoffs') == [] and
                after_cache.get('native_handoffs') == []):
            skipped.append(dict(case_index=index, request_id=name, reason='unprofiled adaptive warmup'))
            continue
        after = decode_snapshot(after_cache, layers=layers, experts=experts)
        initialized = before_cache.get('native_handoffs') == []
        if initialized:
            if previous is not None:
                raise ValueError('Native profiles reset after a completed request')
            before = _new_handoffs_before(before_cache, after_cache, after)
        else:
            before = decode_snapshot(before_cache, layers=layers, experts=experts)
        if previous is not None:
            gap = _delta(previous, before)
            if any(any(row) for row in gap):
                raise ValueError('Unattributed native calls between request cases')
        counts = _delta(before, after)
        if not after['rows'][0]['calls'] > before['rows'][0]['calls']:
            raise ValueError('Request has no native observations')
        profiles.append(dict(case_index=index, request_id=name, counts=counts,
                             snapshot=after, inferred_initial_zero=initialized,
                             calls=after['rows'][0]['calls'] - before['rows'][0]['calls']))
        previous = after
    if len(profiles) < 2:
        raise ValueError('Require at least two complete profiled requests')
    return profiles, skipped


def propose_swaps(profile, *, max_swaps=MAX_SWAPS, max_transfer_bytes=MAX_TRANSFER_BYTES):
    """Stable greedy positive previous-request gain, same class, bounded raw bytes.

    Rank compatible pairs by saved previous-request occurrences, then incoming
    (layer,expert), then victim (layer,expert). No future counts, cost or format
    performance is used. Strictly positive gain avoids churn on equal heat.
    """
    _integer(max_swaps, 'max_swaps', 0, MAX_SWAPS)
    _integer(max_transfer_bytes, 'max_transfer_bytes', 0, MAX_TRANSFER_BYTES)
    groups = {}
    for layer, (row, heat) in enumerate(zip(profile['snapshot']['rows'], profile['counts'])):
        candidates, victims = groups.setdefault(row['slot_class_bytes'], ([], []))
        for expert, count in enumerate(heat):
            (victims if expert in row['resident'] else candidates).append((count, layer, expert))
    pairs = []
    for slot_class, (candidates, victims) in groups.items():
        candidates.sort(key=lambda v: (-v[0], v[1], v[2]))
        victims.sort()
        for hot, cold in zip(candidates, victims):
            gain = hot[0] - cold[0]
            if gain <= 0:
                break
            pairs.append((gain, hot[1:], cold[1:], slot_class))
    pairs.sort(key=lambda v: (-v[0], v[1], v[2]))
    selected, incoming_bytes = [], 0
    for gain, incoming, victim, slot_class in pairs:
        if len(selected) == max_swaps:
            break
        size = profile['snapshot']['rows'][incoming[0]]['packed_bytes']
        if incoming_bytes + size > max_transfer_bytes:
            continue
        selected.append(dict(incoming=list(incoming), victim=list(victim),
                             slot_class_bytes=slot_class, incoming_packed_bytes=size,
                             previous_request_gain=gain))
        incoming_bytes += size
    return selected


def _metrics(rows, counts, swaps):
    incoming = {tuple(s['incoming']) for s in swaps}
    victims = {tuple(s['victim']) for s in swaps}
    by_class, by_format = {}, {}
    total = Counter()
    missing_formats = []
    for layer, (row, heat) in enumerate(zip(rows, counts)):
        baseline = sum(v for e, v in enumerate(heat) if e not in row['resident'])
        saved = sum(heat[e] for l, e in incoming if l == layer)
        displaced = sum(heat[e] for l, e in victims if l == layer)
        values = dict(routes=sum(heat), baseline_misses=baseline,
                      proposed_misses=baseline - saved + displaced,
                      saved_misses=saved, displaced_hits=displaced,
                      net_saved_misses=saved - displaced)
        total.update(values)
        by_class.setdefault(row['slot_class_bytes'], Counter()).update(values)
        if row['formats'] is None:
            missing_formats.append(layer)
        else:
            by_format.setdefault(row['formats'], Counter()).update(values)
    return dict(total=dict(total),
                by_class=[dict(slot_class_bytes=s, **dict(v)) for s, v in sorted(by_class.items())],
                by_format=[dict(formats=list(f), **dict(v)) for f, v in sorted(by_format.items())],
                format_attribution=dict(complete=not missing_formats,
                    missing_layers=missing_formats,
                    caveat='Only explicitly supplied gate/up/down formats are attributed; none inferred from size.'))


def analyze_result(result, *, layers=48, experts=512, max_swaps=MAX_SWAPS,
                   max_transfer_bytes=MAX_TRANSFER_BYTES):
    """Return count-only independent previous-request -> next-request evaluations."""
    _integer(max_swaps, 'max_swaps', 0, MAX_SWAPS)
    _integer(max_transfer_bytes, 'max_transfer_bytes', 0, MAX_TRANSFER_BYTES)
    profiles, skipped = request_profiles(result, layers=layers, experts=experts)
    boundaries = []
    for train, following in zip(profiles, profiles[1:]):
        swaps = propose_swaps(train, max_swaps=max_swaps, max_transfer_bytes=max_transfer_bytes)
        boundaries.append(dict(
            training_case_index=train['case_index'], training_request_id=train['request_id'],
            evaluated_case_index=following['case_index'], evaluated_request_id=following['request_id'],
            training_calls_per_layer=train['calls'], evaluated_calls_per_layer=following['calls'],
            swaps=swaps, swap_count=len(swaps),
            incoming_transfer_bytes=sum(s['incoming_packed_bytes'] for s in swaps),
            evaluation=_metrics(following['snapshot']['rows'], following['counts'], swaps)))
    return dict(schema_version=1, diagnostic_only=True, model_movement=False,
                scope='Independent causal request-boundary count estimates on unchanged recorded trajectories',
                geometry=dict(layers=layers, experts_per_layer=experts, top_k=TOP_K),
                policy=dict(max_swaps=max_swaps, max_transfer_bytes=max_transfer_bytes,
                    starting_placement='Recorded residents at each boundary; not rolling simulated placement',
                    slot_class_rule='256-byte alignment, then smallest recorded fitting sized-slot class'),
                skipped_cases=skipped,
                inferred_initial_zero_cases=[p['case_index'] for p in profiles if p['inferred_initial_zero']],
                boundaries=boundaries,
                caveats=[
                    'Whole-request native counts include prefill; not decode-only attribution.',
                    'No trajectory changes, cross-site rounding, transfer cost or graph rebuild is simulated.',
                    'No throughput, latency, quality or realized speedup estimate is made.',
                    'Last completed request has no following request to evaluate.',
                    'Same-class exchanges preserve recorded class occupancy; actual lease retirement is not tested.'])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def analyze_file(path, **options):
    """Read <=64 MiB once, hash those exact bytes, and analyze without model imports."""
    path = Path(path)
    with path.open('rb') as stream:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError('Source exceeds 64MiB bound')
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('Source exceeds 64MiB bound')
    result = json.loads(raw, object_pairs_hook=_unique_object)
    report = analyze_result(result, **options)
    report['source'] = dict(path=str(path.resolve()), size_bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Existing result.json')
    parser.add_argument('--output', type=Path, required=True, help='New exclusive JSON output')
    parser.add_argument('--layers', type=int, default=48)
    parser.add_argument('--experts', type=int, default=512)
    parser.add_argument('--max-swaps', type=int, default=MAX_SWAPS)
    parser.add_argument('--max-transfer-mib', type=int, default=32)
    args = parser.parse_args(argv)
    try:
        report = analyze_file(args.input, layers=args.layers, experts=args.experts,
                              max_swaps=args.max_swaps,
                              max_transfer_bytes=args.max_transfer_mib * 1024**2)
        payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Wrote {len(report["boundaries"])} count-only boundaries; source SHA256 {report["source"]["sha256"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
