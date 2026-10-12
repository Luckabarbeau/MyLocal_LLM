"""Pure-Python, token-level repetition measurements for fixed generation tests.

These metrics quantify *exact token sequence* repetition.  They do not measure
semantic redundancy, topical coherence, factuality, or program correctness.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean


def _ngram_counts(tokens: list[int], n: int) -> tuple[int, int]:
    if len(tokens) < n:
        return 0, 0
    seen = set()
    repeats = 0
    for i in range(len(tokens) - n + 1):
        item = tuple(tokens[i:i + n])
        if item in seen:
            repeats += 1
        seen.add(item)
    return len(tokens) - n + 1, repeats


def first_repeated_ngram_end(tokens: list[int], n: int = 8) -> int | None:
    """1-based position ending the first n-gram seen earlier in this output."""
    seen = set()
    for i in range(max(0, len(tokens) - n + 1)):
        gram = tuple(tokens[i:i + n])
        if gram in seen:
            return i + n
        seen.add(gram)
    return None


def longest_repeated_span(tokens: list[int]) -> int:
    """Longest contiguous exact substring occurring twice (overlap allowed).

    Uses O(N^2) time / O(N) memory.  N <= 512 by default.  Overlapping
    matches are intentional; the metric measures repetition, not duplicated
    non-overlapping substrings.
    """
    size = len(tokens)
    dp = [0] * (size + 1)
    longest = 0
    for i in range(1, size + 1):
        for j in range(size, i, -1):
            if tokens[i - 1] == tokens[j - 1]:
                dp[j] = dp[j - 1] + 1
                longest = max(longest, dp[j])
            else:
                dp[j] = 0
    return longest


def repetition_metrics(tokens: list[int]) -> dict:
    """Measure only generated token IDs, not the conditioning prompt or EOS."""
    values = {"generated_tokens": len(tokens)}
    for n in (3, 4, 5, 8):
        total, repeats = _ngram_counts(tokens, n)
        values[f"distinct_{n}"] = (total - repeats) / total if total else None
        values[f"repeated_{n}gram_rate"] = repeats / total if total else None
    values["first_repeated_8gram_end"] = first_repeated_ngram_end(tokens, 8)
    values["longest_repeated_span_tokens"] = longest_repeated_span(tokens)
    return values


def summarize(records: list[dict]) -> dict:
    """Equal-weighted mean of per-generation metrics; onset mean is conditional."""
    if not records:
        return {"count": 0}
    ms = [record["metrics"] for record in records]
    out = {"count": len(ms), "generated_tokens": sum(m["generated_tokens"] for m in ms)}
    keys = (
        "distinct_3", "distinct_4", "distinct_5", "distinct_8",
        "repeated_3gram_rate", "repeated_4gram_rate",
        "repeated_5gram_rate", "repeated_8gram_rate",
        "longest_repeated_span_tokens",
    )
    for key in keys:
        valid = [m[key] for m in ms if m[key] is not None]
        out[f"mean_{key}"] = mean(valid) if valid else None
    onsets = [m["first_repeated_8gram_end"] for m in ms if m["first_repeated_8gram_end"] is not None]
    out["repeat_8gram_onset_fraction"] = len(onsets) / len(ms)
    out["mean_repeat_8gram_onset_when_present"] = mean(onsets) if onsets else None
    return out


def grouped_summary(records: list[dict]) -> dict:
    """Summary across strategies and per strategy, domain, and prompt length."""
    groups = {
        "overall": summarize(records),
        "by_domain": {},
        "by_length": {},
        "by_domain_and_length": {},
        "by_strategy": {},
        "by_strategy_and_domain": {},
        "by_strategy_and_length": {},
    }
    domain = defaultdict(list)
    length = defaultdict(list)
    both = defaultdict(list)
    by_strategy = defaultdict(list)
    strategy_domain = defaultdict(list)
    strategy_length = defaultdict(list)
    for record in records:
        item = record["prompt"]
        strategy = record.get("decoding", {}).get("strategy", "unknown")
        domain[item["domain"]].append(record)
        length[item["length"]].append(record)
        both[(item["domain"], item["length"])].append(record)
        by_strategy[strategy].append(record)
        strategy_domain[(strategy, item["domain"])].append(record)
        strategy_length[(strategy, item["length"])].append(record)
    groups["by_domain"] = {key: summarize(items) for key, items in sorted(domain.items())}
    groups["by_length"] = {key: summarize(items) for key, items in sorted(length.items())}
    groups["by_domain_and_length"] = {
        f"{d}/{length}": summarize(items)
        for (d, length), items in sorted(both.items())
    }
    groups["by_strategy"] = {
        strategy: summarize(items)
        for strategy, items in sorted(by_strategy.items())
    }
    groups["by_strategy_and_domain"] = {
        f"{strategy}/{name}": summarize(items)
        for (strategy, name), items in sorted(strategy_domain.items())
    }
    groups["by_strategy_and_length"] = {
        f"{strategy}/{name}": summarize(items)
        for (strategy, name), items in sorted(strategy_length.items())
    }
    return groups
