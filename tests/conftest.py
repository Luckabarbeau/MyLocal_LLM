"""Shared pytest organization for the repository's historical flat test suite.

Markers are inferred from stable filename conventions so older tests do not need to
be rewritten merely to participate in focused test runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_GPU_HINTS = ("cupy", "cuda", "gpu")
_INTEGRATION_HINTS = (
    "benchmark",
    "checkpoint",
    "corpus",
    "generation",
    "integration",
    "preprocessing",
    "shard",
    "streaming",
    "train",
)
_SLOW_HINTS = ("audit", "benchmark", "extended", "performance", "production")


def _matches(filename: str, hints: tuple[str, ...]) -> bool:
    return any(hint in filename for hint in hints)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify tests without changing their execution or backend selection."""
    for item in items:
        filename = Path(str(item.fspath)).name.lower()

        if _matches(filename, _GPU_HINTS):
            item.add_marker(pytest.mark.gpu)
        if _matches(filename, _SLOW_HINTS):
            item.add_marker(pytest.mark.slow)

        if _matches(filename, _INTEGRATION_HINTS):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
