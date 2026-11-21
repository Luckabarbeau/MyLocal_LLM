"""Read-only request-count input for setup inside an already initialized model.

Importing this module uses only the standard library. No CLI, output writer,
model construction, weight reads, or GPU execution is provided here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MAX_SOURCE_BYTES = 64 * 1024**2


def _source_identities(records, integer, object_):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('Require nonempty checkpoint_sources provenance')
    identities = {}
    for record in records:
        record = object_(record, 'checkpoint source')
        if set(record) != {'path', 'size_bytes', 'mtime_ns'}:
            raise ValueError('Checkpoint source requires exactly path, size_bytes, mtime_ns')
        path = record['path']
        if not isinstance(path, str) or not path.strip() or '\x00' in path:
            raise ValueError('Checkpoint source path must be a nonempty string')
        path = str(Path(path).resolve())
        identity = dict(path=path,
                        size_bytes=integer(record['size_bytes'], 'checkpoint size_bytes', 1),
                        mtime_ns=integer(record['mtime_ns'], 'checkpoint mtime_ns'))
        if path in identities:
            raise ValueError('Duplicate canonical checkpoint source path')
        identities[path] = identity
    return [identities[path] for path in sorted(identities)]


def _invalid_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


def load_profile_counts(path, *, case_index, checkpoint_sources, model_id,
                        layers=48, experts=512):
    """Return ``({ExpertKey: int}, provenance)`` for one completed request.

    ``case_index`` is a zero-based position, never a request-ID lookup or an
    implicit last-case selection. Completion requires a complete cache_after
    with positive native observations. Only the prefix through that case is
    validated; later cases are merely JSON-parsed. Earlier unprofiled adaptive
    warmup cases are allowed, but cannot themselves supply counts.

    Defaults require all 48 x 512 counters. Smaller explicit geometry supports
    synthetic tests under the same decoder contract (top_k is always 10).
    Initial [] handoffs may imply zeros only before any profiled request and
    with unchanged promotion, eviction, and sized-slot evidence. Otherwise both
    snapshots and inter-case continuity must validate without resets or gaps.

    Checkpoint records must contain exactly path/size_bytes/mtime_ns. Their
    duplicate-free canonical identities must match, irrespective of list order.
    The caller supplies current stat provenance: this function neither stats
    checkpoint files nor reads/hashes weights. This is NOT cryptographic
    checkpoint verification. Only the bounded JSON source bytes are SHA-256
    hashed, once, and no files are written.

    Provenance contains source {path, size_bytes, sha256}, case_index,
    request_id, calls (per layer), layers, experts, top_k, totals
    {per_layer, all_layers}, inferred_initial_zero, checkpoint_sources, and
    checkpoint_verification. Counts include zeros and cover the full geometry.
    """
    from scripts.analyze_0325_request_placement import (
        TOP_K, _delta, _integer, _new_handoffs_before, _object,
        _unique_object, decode_snapshot,
    )

    _integer(case_index, 'case_index')
    _integer(layers, 'layers', 1, 64)
    _integer(experts, 'experts', TOP_K, 512)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError('model_id must be a nonempty string')
    current = _source_identities(checkpoint_sources, _integer, _object)
    path = Path(path).resolve()
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size > MAX_SOURCE_BYTES:
            raise ValueError('Source exceeds 64MiB bound')
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('Source exceeds 64MiB bound')
    source = dict(path=str(path), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    result = _object(json.loads(raw, object_pairs_hook=_unique_object,
                                parse_constant=_invalid_constant), 'result')
    recorded = _source_identities(
        _object(result.get('provenance'), 'provenance').get('checkpoint_sources'),
        _integer, _object)
    if recorded != current:
        raise ValueError('Checkpoint source stat identities do not match current provenance')
    cases = result.get('cases')
    if not isinstance(cases, list) or case_index >= len(cases):
        raise ValueError('case_index must select an existing completed case')

    previous = None
    for index in range(case_index + 1):
        case = _object(cases[index], 'case')
        request_id = case.get('request_id')
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError('Missing request_id')
        before_cache = _object(case.get('cache_before'), 'cache_before')
        after_cache = _object(case.get('cache_after'), 'cache_after')
        if (index < case_index and previous is None and request_id == 'warmup' and
                before_cache.get('native_handoffs') == [] and
                after_cache.get('native_handoffs') == []):
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
        calls = after['rows'][0]['calls'] - before['rows'][0]['calls']
        if calls <= 0:
            raise ValueError('Selected prefix request has no native observations')
        previous = after

    # The caller already owns an initialized model; do not pull its backend into
    # module import or reject invalid input only after importing model machinery.
    from mini_llm.ops.expert_residency import ExpertKey

    mapping = {ExpertKey(model_id, layer, expert): count
               for layer, row in enumerate(counts) for expert, count in enumerate(row)}
    totals = [sum(row) for row in counts]
    provenance = dict(source=source, case_index=case_index, request_id=request_id,
                      calls=calls, layers=layers, experts=experts, top_k=TOP_K,
                      totals=dict(per_layer=totals, all_layers=sum(totals)),
                      inferred_initial_zero=initialized, checkpoint_sources=recorded,
                      checkpoint_verification='stat_identity_only_not_cryptographic')
    return mapping, provenance
