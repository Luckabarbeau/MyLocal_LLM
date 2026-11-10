"""Offline analysis of schema-v1 detailed RoutingTrace snapshots.

Replay uses the retained route records in their supplied dispatch order, with
one request per distinct expert in ascending expert-ID order inside a dispatch.
All policies share a global cache keyed by (layer_id, expert_id). Capacities are
equal-sized expert slots, NOT predictions of VRAM bytes or execution latency.
No weights are materialized and no executor, GPU, or trace recorder is invoked.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
import json
import math
from pathlib import Path

from mini_llm.ops.expert_residency import AdaptiveExpertCache, ExpertKey


_DEFAULT_WINDOWS = (128, 512, 2048, 8192)
_DEFAULT_CAPACITIES = (0, 2, 4, 8, 16, 32, 64)
_COVERAGE_PERCENT = (50, 75, 90, 95, 99)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sizes(values, name, minimum):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integers") from exc
    for value in values:
        _integer(value, name, minimum)
    return tuple(dict.fromkeys(values))


def _record_schema(record, name, identity):
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be an object")
    if "schema_version" in record:
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError(f"{name}: mixed schema_version")
    if "mode" in record and record["mode"] != "detailed":
        raise ValueError(f"{name}: mixed trace mode")
    if "identity" in record and record["identity"] != identity:
        raise ValueError(f"{name}: mixed trace identity")


def _validate(snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError("trace must be one snapshot object, not merged snapshots")
    if type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise ValueError("trace schema_version must be 1")
    if snapshot.get("mode") != "detailed":
        raise ValueError("analysis requires a detailed trace; summary-only traces cannot be replayed")
    identity = snapshot.get("identity")
    if not isinstance(identity, dict) or any(
        not isinstance(identity.get(field), str)
        for field in ("model_id", "quantization", "expert_mapping", "workload")
    ):
        raise ValueError("identity requires model_id, quantization, expert_mapping, workload strings")
    routes = snapshot.get("routes")
    if not isinstance(routes, list):
        raise ValueError("routes must be a list")
    executions = snapshot.get("executions", [])
    if not isinstance(executions, list):
        raise ValueError("executions must be a list")
    # Execution records are not replay inputs; only guard against mixed traces.
    for record in executions:
        _record_schema(record, "execution", identity)
    ends = {}
    dispatch_ids = set()
    for index, route in enumerate(routes):
        name = f"routes[{index}]"
        _record_schema(route, name, identity)
        dispatch = _integer(route.get("dispatch_id"), f"{name}.dispatch_id")
        if dispatch in dispatch_ids:
            raise ValueError(f"{name}: duplicate dispatch_id")
        dispatch_ids.add(dispatch)
        layer = _integer(route.get("layer_id"), f"{name}.layer_id")
        start = _integer(route.get("token_start"), f"{name}.token_start")
        if start < ends.get(layer, 0):
            raise ValueError(f"{name}: overlapping or regressing token_start within layer")
        ids, weights = route.get("expert_ids"), route.get("routing_weights")
        if not isinstance(ids, list) or not isinstance(weights, list) or len(ids) != len(weights):
            raise ValueError(f"{name}: expert_ids/routing_weights must have matching token rows")
        top_k = None
        for token_ids, token_weights in zip(ids, weights):
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"{name}: expert_ids rows must be nonempty lists")
            if top_k is None:
                top_k = len(token_ids)
            if len(token_ids) != top_k:
                raise ValueError(f"{name}: ragged expert_ids top-k rows")
            if not isinstance(token_weights, list) or len(token_weights) != top_k:
                raise ValueError(f"{name}: routing_weights shape mismatch")
            for expert in token_ids:
                _integer(expert, f"{name}.expert_id")
            for weight in token_weights:
                if (type(weight) not in (int, float) or not math.isfinite(weight)
                        or weight < 0):
                    raise ValueError(f"{name}: routing_weights must be finite nonnegative numbers")
        # Nonzero starts and gaps are allowed: bounded retention need not include
        # the beginning (or every dispatch) of a layer's token history.
        ends[layer] = start + len(ids)
    return routes


def _metrics(requests, hits, evictions=0, admissions=0, resident=0):
    return {
        "requests": requests, "hits": hits, "misses": requests - hits,
        "hit_rate": hits / requests if requests else 0.0,
        "miss_rate": (requests - hits) / requests if requests else 0.0,
        "evictions": evictions, "admissions": admissions, "resident_slots": resident,
    }


def _adaptive_replay(requests, capacity, model_id):
    # One synthetic byte per expert maps the existing byte budget to slot count.
    # A non-None sentinel is essential: None means CPU fallback to the cache API.
    cache = AdaptiveExpertCache(capacity)
    sentinel = object()
    for layer, expert in requests:
        cache.request(ExpertKey(model_id, layer, expert), 1, lambda: sentinel)
    stats = cache.statistics()
    result = _metrics(len(requests), stats.get("hits", 0),
                      stats.get("evictions", 0), stats.get("promotions", 0),
                      len(cache.resident))
    result["cpu_fallbacks"] = stats.get("cpu_fallbacks", 0)
    return result


def _online_replay(requests, capacity, policy):
    resident = {}
    counts = Counter()
    hits = evictions = admissions = 0
    for step, key in enumerate(requests):
        counts[key] += 1
        if key in resident:
            hits += 1
            resident[key] = step
        elif capacity:
            if len(resident) == capacity:
                if policy == "lru":
                    victim = min(resident, key=lambda k: (resident[k], k))
                elif policy == "lfu":
                    victim = min(resident, key=lambda k: (counts[k], resident[k], k))
                else:
                    victim = min(resident, key=lambda k: (
                        counts[k] / (step - resident[k] + 1), resident[k], k))
                del resident[victim]
                evictions += 1
            resident[key] = step
            admissions += 1
    return _metrics(len(requests), hits, evictions, admissions, len(resident))


def _static_replay(requests, capacity, ranking):
    eligible = set(ranking[:capacity])
    resident = set()
    hits = 0
    for key in requests:
        if key in resident:
            hits += 1
        elif key in eligible:
            resident.add(key)
    return _metrics(len(requests), hits, admissions=len(resident), resident=len(resident))


def _belady_replay(requests, capacity):
    future = defaultdict(deque)
    for index, key in enumerate(requests):
        future[key].append(index)
    resident = set()
    hits = evictions = admissions = 0
    for index, key in enumerate(requests):
        future[key].popleft()
        if key in resident:
            hits += 1
        elif capacity:
            if len(resident) == capacity:
                # CPU fallback permits bypass: the current request may be the
                # farthest-next-use victim, unlike mandatory-admission paging.
                victim = max(resident | {key}, key=lambda k: (
                    future[k][0] if future[k] else math.inf, k))
                if victim == key:
                    continue
                resident.remove(victim)
                evictions += 1
            resident.add(key)
            admissions += 1
    return _metrics(len(requests), hits, evictions, admissions, len(resident))


def _coverage(counts):
    total = sum(counts.values())
    result = {}
    ordered = sorted(counts.values(), reverse=True)
    for percent in _COVERAGE_PERCENT:
        cumulative = 0
        slots = 0
        while 100 * cumulative < percent * total:
            cumulative += ordered[slots]
            slots += 1
        result[str(percent)] = slots
    return result


def _window_analysis(routes, width):
    buckets = {}
    for route in routes:
        layer = route["layer_id"]
        for offset, experts in enumerate(route["expert_ids"]):
            start = (route["token_start"] + offset) // width * width
            bucket = buckets.setdefault((layer, start), {"tokens": 0, "counts": Counter()})
            bucket["tokens"] += 1
            bucket["counts"].update(experts)
    layers = defaultdict(list)
    for (layer, start), bucket in sorted(buckets.items()):
        counts = bucket["counts"]
        selections = sum(counts.values())
        layers[layer].append({
            "token_start": start, "token_end": start + width,
            "observed_tokens": bucket["tokens"],
            "missing_tokens": width - bucket["tokens"],
            "partial": bucket["tokens"] != width,
            "selections": selections,
            "experts": [
                {"expert_id": expert, "selections": count,
                 "selection_share": count / selections}
                for expert, count in sorted(counts.items())
            ],
            "static_coverage_slots": _coverage(counts),
        })
    return {"window_size": width, "layers": [
        {"layer_id": layer, "windows": entries} for layer, entries in sorted(layers.items())
    ]}


def analyze_trace(snapshot, windows=_DEFAULT_WINDOWS, capacities=_DEFAULT_CAPACITIES):
    """Return a JSON-serializable summary without mutating the snapshot.

    Weights must match rectangular top-k ID rows and be finite/nonnegative;
    normalization is not assumed. Duplicate IDs and zero weights are retained
    as selections, matching grouped runtime routing's active expert IDs.

    Every replay starts empty at the retained trace boundary. Static-hot fits
    request frequencies on this same sample, fixes its eligible set, and loads
    only on first use (no free warm start). Belady is an offline optimal oracle
    with bypass allowed, reflecting the runtime's ability to execute on CPU.
    LFU counts all past requests, breaking ties by LRU. The recency-frequency
    hybrid evicts minimum lifetime frequency / (requests since last use + 1).

    Windows are nonoverlapping, per-layer absolute flattened-token intervals.
    Only observed windows are emitted, including partial leading/trailing or
    gapped windows. Coverage counts selections, not probability mass; it is
    an in-sample static oracle, NOT a measurement of LRU capacity. There is no
    per-window cache reset/replay: curves replay whole retained dispatches.
    """
    routes = _validate(snapshot)
    widths = _sizes(windows, "windows", 1)
    sizes = _sizes(capacities, "capacities", 0)
    requests = []
    tokens = Counter()
    for route in routes:
        layer = route["layer_id"]
        tokens[layer] += len(route["expert_ids"])
        requests.extend((layer, expert) for expert in sorted({
            expert for row in route["expert_ids"] for expert in row
        }))
    frequencies = Counter(requests)
    ranking = sorted(frequencies, key=lambda key: (-frequencies[key], key))
    curves = []
    for capacity in sizes:
        policies = {
            "adaptive": _adaptive_replay(requests, capacity, snapshot["identity"]["model_id"]),
            **{policy: _online_replay(requests, capacity, policy)
               for policy in ("lru", "lfu", "recency_frequency")},
            "static_hot": _static_replay(requests, capacity, ranking),
            "belady": _belady_replay(requests, capacity),
        }
        curves.append({"capacity_slots": capacity, "policies": policies})
    return {
        "schema_version": 1,
        "identity": deepcopy(snapshot["identity"]),
        "source_mode": "detailed",
        "scope": "retained routes only; cold start at retained boundary, not full workload",
        "capacity_unit": "equal-sized expert slots; not VRAM bytes",
        "request_order": "route record order; ascending distinct expert IDs within each dispatch",
        "cache_key": ["layer_id", "expert_id"],
        "limitations": [
            "No VRAM-byte, latency, transfer, asynchronous protection, or in-flight simulation.",
            "Token windows count flattened selections, not runtime expert requests. Batched prefill "
            "coalesces tokens into one request per active expert per dispatch; token-window "
            "request locality is only an approximation, not a token-by-token runtime replay.",
            "Static coverage slots are an in-sample optimistic oracle, not actual LRU capacities.",
            "Unretained history is unknown; wholly unobserved windows are omitted.",
        ],
        "policy_definitions": {
            "adaptive": {"implementation": "AdaptiveExpertCache.request with sentinel and size=1",
                         "promote_after": 3, "score_half_life": 8192,
                         "replacement_margin": 1.05, "idle_eviction_requests": 0,
                         "hits": "resident before request; promotions count as misses"},
            "lru": "least recently requested resident; admit every miss",
            "lfu": "lifetime request frequency, LRU tie-break; admit every miss",
            "recency_frequency": "minimum lifetime frequency / (request age + 1); admit every miss",
            "static_hot": "in-sample optimistic oracle: fixed hottest request keys, demand-loaded cold",
            "belady": "offline optimal farthest-next-use with bypass allowed; demand-loaded cold",
        },
        "retained": {"dispatches": len(routes), "requests": len(requests),
                     "unique_experts": len(frequencies),
                     "layer_tokens": [{"layer_id": layer, "tokens": count}
                                      for layer, count in sorted(tokens.items())]},
        "replay_curves": curves,
        "token_windows": [_window_analysis(routes, width) for width in widths],
    }


def main(argv=None):
    """Run with python -m mini_llm.diagnostics.routing_analysis --trace trace.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capacities", nargs="+", type=int, default=_DEFAULT_CAPACITIES)
    parser.add_argument("--windows", nargs="+", type=int, default=_DEFAULT_WINDOWS)
    args = parser.parse_args(argv)
    try:
        with args.trace.open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
        result = analyze_trace(snapshot, windows=args.windows, capacities=args.capacities)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
