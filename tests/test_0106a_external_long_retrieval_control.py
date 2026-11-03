"""Fast contract tests for the external GGUF retrieval control (no model weights)."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.evaluate_external_long_retrieval import build_cases, evaluate


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'benchmarks' / 'tasks' / 'long_retrieval_0097.json'


class FakeTokenizer:
    bos_id = 1

    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        result = []
        for value in re.findall(r' ?[A-Za-z0-9_-]+|[^\w\s]|\s+', text):
            if value not in self.vocab:
                self.vocab[value] = len(self.vocab) + 2
            result.append(self.vocab[value])
        return result


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=8192)

    def create_generation_state(self, batch_size, max_length):
        assert batch_size == 1 and max_length <= 8192
        return SimpleNamespace(max_length=max_length)

    def prefill(self, ids, state):
        assert ids.shape == (1, state.max_length)
        return np.linspace(-8, 1, 4096, dtype=np.float32)[None, :]


def test_full_task_grid_is_96_and_exclusions_are_explicit():
    tok = FakeTokenizer()
    distances = (1000, 4000, 8000, 16000, 24000, 32000)
    runnable, skipped, digest = build_cases(tok, TASKS, distances, 2, 8192)
    assert len(runnable) + len(skipped) == 96
    assert {x['distance'] for x in runnable} == {1000, 4000, 8000}
    assert len(runnable) == len(skipped) == 48
    assert all('exceeds' in x['reason'] for x in skipped)
    assert len(digest) == 64
    assert all(len(row['ids']) - row['fact_end'] == row['distance'] for row in runnable)


def test_1k_4k_shared_scoring_and_counterfactual_metadata():
    tok = FakeTokenizer()
    result = evaluate(FakeModel(), tok, TASKS, (1000, 4000), 2,
                      include_counterfactual=True, progress=False)
    assert result['count'] == 32
    assert not result['unsupported_cases']
    assert {d: result['by_distance'][d]['count'] for d in ('1000', '4000')} == {
        '1000': 16, '4000': 16}
    assert all(len(r['choices']) == 4 for r in result['cases'])
    assert all(r['prompt_tokens'] <= 8192 for r in result['cases'])
    assert all(r['counterfactual']['measured_distance_tokens'] ==
               r['requested_distance_tokens'] for r in result['cases'])
    assert 'same 0097 task bank' in result['warning'].lower()
