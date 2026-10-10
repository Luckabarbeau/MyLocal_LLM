import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "archive" / "experimental"))

import pytest

from generate_mixed_pretraining_shards import parallelism_plan


def test_parallelism_plan_uses_requested_cpu_budget():
    assert parallelism_plan(6, 24, 7) == (6, 4)
    assert parallelism_plan(7, 24, 7) == (7, 3)
    assert parallelism_plan(1, 24, 7) == (1, 24)


def test_parallelism_plan_is_bounded_by_pending_sources_and_threads():
    assert parallelism_plan(8, 24, 3) == (3, 8)
    assert parallelism_plan(8, 2, 7) == (2, 1)
    assert parallelism_plan(8, 24, 0) == (0, 0)


def test_parallelism_plan_rejects_invalid_controls():
    with pytest.raises(ValueError):
        parallelism_plan(0, 24, 1)
    with pytest.raises(ValueError):
        parallelism_plan(1, 0, 1)
