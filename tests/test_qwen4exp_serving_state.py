"""CPU-only serving state contracts: no checkpoint, GPU, sockets or server launch."""
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from mini_llm.external.qwen4exp_chat import ChatSession, render_chat
from scripts.serve_qwen4exp import Service


class Characters:
    # A small template accepting both raw streamed and split reasoning histories.
    chat_template = '''{% for m in messages %}{{ m.role }}:{% if m.role == 'assistant' and m.get('reasoning_content') %}<think>{{ m.reasoning_content }}</think>{{ m.content }}{% else %}{% if m.role == 'assistant' and enable_thinking and not m.content.startswith('<think>') %}<think>{% endif %}{{ m.content }}{% endif %}\n{% endfor %}assistant:{% if enable_thinking %}<think>{% endif %}'''

    def encode(self, text):
        return SimpleNamespace(ids=list(map(ord, text)))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class FakeModel:
    stop_token_ids = {3}

    def __init__(self, reply='OK'):
        self.reply = list(map(ord, reply)) + [3]
        self.states = []
        self.prefills = []
        self.expert_reads = 0
        self.expert_executor = SimpleNamespace(statistics=self.statistics)

    def statistics(self):
        self.expert_reads += 1
        return {'device': {'hit_rate': 0.5, 'used_bytes': 123}}

    def create_generation_state(self, batch, context):
        state = SimpleNamespace(tokens=[])
        self.states.append(state)
        return state

    def logits(self):
        row = np.full((1, 256), -100, dtype=np.float32)
        row[0, self.reply[self.index]] = 100
        return row

    def prefill(self, ids, state, **kwargs):
        self.prefills.append((list(ids), kwargs))
        state.tokens.extend(ids)
        self.index = 0
        return self.logits()

    def step(self, token, state):
        state.tokens.append(token)
        self.index += 1
        return self.logits()


def service(thinking=False, reply='OK', chunk=0):
    return Service(ChatSession(FakeModel(reply), Characters(), 4096,
                               thinking=thinking, prefill_chunk_size=chunk),
                   {'qsa_state_storage': 'host', 'prefill_chunk_size': chunk})


def run(svc, messages, *, stream=False, limit=128):
    svc.prepare(messages)
    text = ''.join(svc.session.generate(messages[-1]['content'], limit, raw_response=stream))
    return text, dict(svc.session.messages[-1])


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('stream', [False, True])
def test_returned_history_reuses_verified_tokens(thinking, stream):
    svc = service(thinking, 'why</think>OK' if thinking else 'OK')
    request = [dict(role='user', content='Hi')]
    raw, assistant = run(svc, request, stream=stream)
    assert assistant['content'] == (raw if stream else 'OK')
    if thinking and not stream:
        assert assistant['reasoning_content'] == 'why'
    state, used = svc.session.state, list(svc.session.consumed)
    # Incoming history, not the previous Python representation, is authoritative.
    if thinking:
        assistant = (dict(role='assistant', content='<think>why</think>OK') if not stream
                     else dict(role='assistant', content='OK', reasoning_content='why'))
    run(svc, request + [assistant, dict(role='user', content='Again')], stream=stream)
    assert svc.session.state is state
    assert svc.session.last_stats['prefix_match']
    assert svc.session.last_stats['reused_tokens'] == len(used)
    assert svc.session.last_stats['reset_reason'] is None
    assert svc.session.state.tokens == svc.session.consumed


@pytest.mark.parametrize('thinking', [False, True])
def test_streamed_whitespace_is_stored_exactly(thinking):
    svc = service(thinking, 'why</think> OK \n' if thinking else ' OK \n')
    request = [dict(role='user', content='Hi')]
    text, assistant = run(svc, request, stream=True)
    assert assistant == dict(role='assistant', content=text)
    state = svc.session.state
    run(svc, request + [assistant, dict(role='user', content='Next')], stream=True)
    assert svc.session.state is state
    assert svc.session.last_stats['prefix_match']


def test_pending_last_generated_token_is_prefilled_once():
    svc = service(reply='OK')
    request = [dict(role='user', content='Hi')]
    _, assistant = run(svc, request, stream=True, limit=2)
    assert chr(svc.session.consumed[-1]) == 'O'
    assert svc.session.state.tokens == svc.session.consumed
    used = len(svc.session.consumed)
    run(svc, request + [assistant, dict(role='user', content='Next')], stream=True, limit=2)
    tail, _ = svc.session.model.prefills[-1]
    assert tail[0] == ord('K')
    assert svc.session.last_stats['reused_tokens'] == used
    assert svc.session.state.tokens == svc.session.consumed


@pytest.mark.parametrize('change', ['user', 'system', 'remove_system', 'reasoning'])
def test_prefix_changes_rebuild_and_system_does_not_leak(change):
    svc = service(True, 'why</think>OK')
    request = [dict(role='system', content='secret'), dict(role='user', content='Hi')]
    _, assistant = run(svc, request)
    state = svc.session.state
    messages = [dict(m) for m in request] + [assistant, dict(role='user', content='Next')]
    if change == 'user':
        messages[1]['content'] = 'Different'
    elif change == 'system':
        messages[0]['content'] = 'Other'
    elif change == 'remove_system':
        messages.pop(0)
    else:
        messages[2]['reasoning_content'] = 'Different'
    svc.prepare(messages)
    assert svc.session.state is state  # No string-history reset in prepare.
    list(svc.session.generate('Next', 128))
    stats = svc.session.last_stats
    assert svc.session.state is not state
    assert stats['state_reset'] and not stats['prefix_match']
    assert stats['reset_reason'] == 'token_prefix_mismatch'
    assert stats['reused_tokens'] == 0
    expected = svc.session.tokenizer.encode(render_chat(
        svc.session.tokenizer, messages, thinking=True)).ids
    assert svc.session.model.prefills[-1][0] == expected
    if change == 'remove_system':
        assert svc.session.system == ''
        svc.session.reset()
        assert svc.session.messages == []


@pytest.mark.parametrize('chunk', [0, 7])
def test_chunk_forwarding_and_request_metrics(chunk):
    svc = service(chunk=chunk)
    started = time.perf_counter()
    run(svc, [dict(role='user', content='Hi')])
    assert svc.session.model.prefills[-1][1] == ({'chunk_size': chunk} if chunk else {})
    stats = svc.record('request-test', started)
    assert stats['request_id'] == 'request-test'
    assert stats['cache']['total_tokens'] == stats['prompt_tokens']
    assert stats['cache']['new_tokens'] + stats['cache']['reused_tokens'] == stats['prompt_tokens']
    assert stats['cache']['reset_reason'] == 'initial_state'
    assert stats['prefill_seconds'] >= 0 and stats['decode_seconds'] >= 0
    assert stats['first_piece_seconds'] >= 0 and stats['total_seconds'] >= 0
    assert svc.session.model.expert_reads == 0
    metrics = svc.metrics()
    assert metrics['mylocal_stats'] == stats
    assert metrics['config']['prefill_chunk_size'] == chunk
    assert metrics['experts']['device']['used_bytes'] == 123
    json.dumps(metrics)
    assert svc.session.model.expert_reads == 1


@pytest.mark.parametrize('bad', [None, 12, [], {}])
def test_reasoning_must_be_optional_string(bad):
    svc = service()
    with pytest.raises(ValueError, match='text messages'):
        svc.prepare([dict(role='assistant', content='OK', reasoning_content=bad),
                     dict(role='user', content='Next')])
    with pytest.raises(ValueError, match='text system/user/assistant'):
        render_chat(Characters(), [dict(role='assistant', content='OK', reasoning_content=bad)])


def test_serving_help_is_import_safe():
    script = Path(__file__).resolve().parents[1] / 'scripts/serve_qwen4exp.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True,
                            text=True, check=True, timeout=10)
    for option in ('--qsa-state-storage', '--host-state-budget-gib', '--prefill-chunk-size'):
        assert option in result.stdout


@pytest.fixture(scope='module')
def checkpoint_tokenizer():
    # Header/tokenizer only: never import or instantiate the production model.
    from scripts.serve_qwen4exp import MODEL
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    if not Path(MODEL).exists():
        pytest.skip('local GGUF tokenizer header unavailable')
    with MultiShardTensorStore.open(MODEL) as store:
        return load_gguf_tokenizer(store.metadata, vocab_size=248320)


class TokenReplayModel(FakeModel):
    stop_token_ids = {248320}

    def __init__(self, tokens):
        super().__init__()
        self.reply = list(tokens) + [248320]

    def logits(self):
        row = np.full((1, 248321), -100, dtype=np.float32)
        row[0, self.reply[self.index]] = 100
        return row


def replay_service(tokenizer, tokens, *, thinking=True):
    return Service(ChatSession(TokenReplayModel(tokens), tokenizer, 4096, thinking=thinking))


@pytest.mark.parametrize('thinking,reply', [
    (True, 'We should check the divisor. The'),
    (True, 'We should check the divisor.\n</think>\n\nCount non-None samples.'),
    (False, 'Count non-None samples.'),
])
def test_actual_template_growing_raw_conversation(checkpoint_tokenizer, thinking, reply):
    tok = checkpoint_tokenizer
    tokens = tok.encode(reply).ids
    svc = replay_service(tok, tokens, thinking=thinking)
    messages = [dict(role='user', content='Review the average calculation.')]
    for turn in range(3):
        old_state = svc.session.state
        used = list(svc.session.consumed)
        text, assistant = run(svc, messages, stream=True, limit=len(tokens))
        assert text == reply and assistant == dict(role='assistant', content=reply)
        assert svc.session.state.tokens == svc.session.consumed
        if turn:
            assert svc.session.state is old_state
            assert svc.session.last_stats['reused_tokens'] == len(used)
            assert svc.session.last_stats['prefix_match']
            assert svc.session.model.prefills[-1][0][0] == tokens[-1]
        messages += [assistant, dict(role='user', content=f'Continue {turn}.')]


def test_saved_host_request_replay_with_explicit_fake_tokens(checkpoint_tokenizer):
    base = Path(__file__).resolve().parents[1] / (
        'benchmarks/production_recovery_20261121/mylocal_host_sequential/http')
    previous = base / '0001-128-identical_prefix_repeat/request.json'
    growing = base / '0002-128-multi_turn/request.json'
    if not previous.exists() or not growing.exists():
        pytest.skip('saved production HTTP requests unavailable')
    request = json.loads(previous.read_text())
    followup = json.loads(growing.read_text())
    raw = followup['messages'][1]['content']
    tok = checkpoint_tokenizer
    tokens = tok.encode(raw).ids
    assert len(tokens) == 128 and '</think>' not in raw
    svc = Service(ChatSession(BoundedTokenReplayModel(tokens), tok, 4096))
    text, _ = run(svc, request['messages'], stream=True, limit=128)
    assert text == raw
    state, consumed = svc.session.state, list(svc.session.consumed)
    assert len(consumed) == 254  # 127 prompt + 127 committed output tokens.
    unnormalized = tok.encode(render_chat(tok, followup['messages'])).ids
    assert len(unnormalized) == 280
    assert next(i for i, (a, b) in enumerate(zip(consumed, unnormalized)) if a != b) == 126
    repeated, _ = run(svc, request['messages'], stream=True, limit=128)
    assert repeated == raw and svc.session.state is state
    assert svc.session.last_stats['checkpoint_restore']
    assert svc.session.last_stats['reused_tokens'] == 127
    assert svc.session.last_stats['new_prompt_tokens'] == 0
    assert not svc.session.last_stats['state_reset']
    assert len(svc.session.model.prefills) == 1
    run(svc, followup['messages'], stream=True, limit=128)
    assert svc.session.state is state
    assert svc.session.last_stats['reused_tokens'] == 254
    assert svc.session.last_stats['new_prompt_tokens'] == 27
    assert svc.session.model.prefills[-1][0] == tokens[-1]


def test_identical_request_requires_rollback_not_reuse(checkpoint_tokenizer):
    tok = checkpoint_tokenizer
    svc = replay_service(tok, tok.encode('Some reasoning remains unfinished.').ids)
    messages = [dict(role='user', content='Hi')]
    run(svc, messages, stream=True)
    state, consumed = svc.session.state, list(svc.session.consumed)
    prompt_ids = tok.encode(render_chat(tok, messages)).ids
    assert len(consumed) > len(prompt_ids)
    run(svc, messages, stream=True)
    stats = svc.session.last_stats
    assert svc.session.state is not state and stats['reused_tokens'] == 0
    assert stats['reset_reason'] == 'token_prefix_mismatch'
    assert stats['prefix_mismatch'] == dict(
        index=len(prompt_ids), consumed_tokens=len(consumed), prompt_tokens=len(prompt_ids),
        consumed_token_id=consumed[len(prompt_ids)], prompt_token_id=None)


def test_reencoding_equal_text_does_not_authorize_reuse(checkpoint_tokenizer):
    tok = checkpoint_tokenizer
    # These are deliberately separate sampled tokens, not the canonical BPE encoding.
    tokens = tok.encode('O').ids + tok.encode('K').ids + tok.encode(' more reasoning.').ids
    raw = tok.decode(tokens, skip_special_tokens=False)
    assert tok.encode(raw).ids != tokens
    svc = replay_service(tok, tokens)
    messages = [dict(role='user', content='Hi')]
    _, assistant = run(svc, messages, stream=True)
    state = svc.session.state
    run(svc, messages + [assistant, dict(role='user', content='Next')], stream=True)
    stats = svc.session.last_stats
    assert svc.session.state is not state and stats['reused_tokens'] == 0
    assert stats['reset_reason'] == 'token_prefix_mismatch'
    mismatch = stats['prefix_mismatch']
    assert mismatch['consumed_token_id'] != mismatch['prompt_token_id']


def test_actual_template_trimming_committed_text_still_resets(checkpoint_tokenizer):
    tok = checkpoint_tokenizer
    raw = 'Check the divisor.\n</think>\n\n OK \n'
    svc = replay_service(tok, tok.encode(raw).ids)
    messages = [dict(role='user', content='Hi')]
    text, assistant = run(svc, messages, stream=True)
    assert text == raw and assistant['content'] == raw
    state = svc.session.state
    # All generated tokens were committed before stop; trimming cannot undo them.
    assert svc.session.last_stats['stopped']
    run(svc, messages + [assistant, dict(role='user', content='Next')], stream=True)
    stats = svc.session.last_stats
    assert svc.session.state is not state and stats['reused_tokens'] == 0
    assert stats['reset_reason'] == 'token_prefix_mismatch'
    assert stats['prefix_mismatch']['consumed_token_id'] != stats['prefix_mismatch']['prompt_token_id']


def test_unknown_raw_history_is_not_guessed_to_be_reasoning(checkpoint_tokenizer):
    tok = checkpoint_tokenizer
    svc = replay_service(tok, tok.encode('Known reasoning.').ids)
    messages = [dict(role='user', content='Hi')]
    _, assistant = run(svc, messages, stream=True)
    assistant['content'] = 'Different reasoning.'
    incoming = messages + [assistant, dict(role='user', content='Next')]
    svc.prepare(incoming)
    assert svc.session.messages[-1] == assistant
    state = svc.session.state
    list(svc.session.generate('Next', 128, raw_response=True))
    assert svc.session.state is not state
    assert svc.session.last_stats['reset_reason'] == 'token_prefix_mismatch'
    assert svc.session.model.prefills[-1][0] == tok.encode(render_chat(tok, incoming)).ids


@dataclass(eq=False)
class BoundedState:
    owner: object
    max_length: int
    recurrent: dict
    qsa: dict
    ple: dict
    length: int = 0
    failed: bool = False


class BoundedModel:
    """Small stateful CPU graph: output depends on every checkpointed component."""
    stop_token_ids = set()

    def __init__(self):
        self.states = []
        self.prefills = []
        self.steps = []

    def create_generation_state(self, batch, context):
        from mini_llm.external.qwen4exp_attention import QSAState
        from mini_llm.external.qwen4exp_ple import PLEState
        from mini_llm.ops.causal_conv_inference import CausalConvState
        from mini_llm.ops.gated_delta_inference import DeltaMixerState, DeltaRuleState
        recurrent = DeltaMixerState(CausalConvState(np.zeros((1, 2, 2), np.float32)),
                                    DeltaRuleState(np.zeros((1, 1, 2, 2), np.float32)))
        qsa = QSAState(object(), context, np.full((context, 1, 2), np.nan, np.float32),
                       np.full((context, 1, 2), np.nan, np.float32),
                       np.full((context, 2), np.nan, np.float32),
                       np.full(((context + 3) // 4, 2), np.nan, np.float32),
                       qsa_state_storage='host')
        qsa.staged_prefix_length = 0
        state = BoundedState(object(), context, {0: recurrent}, {1: qsa},
                             {0: PLEState((-1, -1), np.zeros((3, 1, 2), np.float32))})
        self.states.append(state)
        return state

    def advance(self, token, state):
        from mini_llm.external.qwen4exp_ple import PLEState
        r, q, p, t = state.recurrent[0], state.qsa[1], state.ple[0], state.length
        assert q.length == r.conv.length == r.delta.length == t
        r.conv.history[:, :-1] = r.conv.history[:, 1:]
        r.conv.history[:, -1] = np.float32(token)
        r.delta.memory *= np.float32(.5)
        r.delta.memory += np.float32(token + t)
        q.keys[t] = np.float32(token + t)
        q.values[t] = np.float32(token - t)
        q.raw_index[t] = np.float32(token)
        if (t + 1) % 4 == 0:
            q.pooled[t // 4] = q.raw_index[t-3:t+1].mean(axis=0)
        state.ple[0] = PLEState((p.predecessors[1], token),
                               np.concatenate((p.normalized_history[1:],
                                               np.full((1, 1, 2), token, np.float32))))
        state.length = q.length = r.conv.length = r.delta.length = t + 1
        q.staged_prefix_length = t + 1

    def logits(self, state):
        r, q, p = state.recurrent[0], state.qsa[1], state.ple[0]
        # Deliberately ignore every QSA row beyond logical length, including a
        # formerly completed pool slot that is now only a partial block.
        value = (r.conv.history.sum() + r.delta.memory.sum() + p.normalized_history.sum()
                 + q.values[:q.length].sum() + q.pooled[:q.length // 4].sum()
                 + sum(p.predecessors))
        row = np.full((1, 256), -100, np.float32)
        row[0, 65 + int(value) % 26] = 100
        return row

    def prefill(self, ids, state):
        self.prefills.append(list(ids))
        for token in ids:
            self.advance(token, state)
        return self.logits(state)

    def step(self, token, state):
        self.steps.append(token)
        self.advance(token, state)
        return self.logits(state)


class BoundedTokenReplayModel(BoundedModel):
    def __init__(self, tokens):
        super().__init__()
        self.reply = list(tokens)

    def prefill(self, ids, state):
        self.prompt_end = state.length + len(ids)
        return super().prefill(ids, state)

    def logits(self, state):
        row = np.full((1, 248321), -100, np.float32)
        row[0, self.reply[(state.length - self.prompt_end) % len(self.reply)]] = 100
        return row


def bounded_service():
    return Service(ChatSession(BoundedModel(), Characters(), 4096, thinking=False))


def assert_active_states_equal(a, b):
    assert a.length == b.length
    np.testing.assert_array_equal(a.recurrent[0].conv.history, b.recurrent[0].conv.history)
    np.testing.assert_array_equal(a.recurrent[0].delta.memory, b.recurrent[0].delta.memory)
    assert a.ple[0].predecessors == b.ple[0].predecessors
    np.testing.assert_array_equal(a.ple[0].normalized_history, b.ple[0].normalized_history)
    qa, qb = a.qsa[1], b.qsa[1]
    assert qa.length == qb.length == a.length
    for name in ('keys', 'values', 'raw_index'):
        np.testing.assert_array_equal(getattr(qa, name)[:qa.length], getattr(qb, name)[:qb.length])
    np.testing.assert_array_equal(qa.pooled[:qa.length // 4], qb.pooled[:qb.length // 4])


@pytest.mark.parametrize('limit', [1, 2, 6])
def test_prompt_checkpoint_identical_request_uses_saved_logits(limit):
    svc = bounded_service()
    messages = [dict(role='user', content='Hi')]
    first_text, _ = run(svc, messages, stream=True, limit=limit)
    state, checkpoint = svc.session.state, svc.session._prompt_checkpoint
    prompt_length = len(svc.session.model.prefills[0])
    bindings = (state.recurrent[0].conv.history, state.recurrent[0].delta.memory,
                state.qsa[1].keys, state.qsa[1].values, state.qsa[1].raw_index, state.qsa[1].pooled)
    for _ in range(2):
        text, _ = run(svc, messages, stream=True, limit=limit)
        stats = svc.session.last_stats
        assert text == first_text and svc.session.state is state
        assert svc.session._prompt_checkpoint is checkpoint
        assert len(svc.session.model.prefills) == 1
        assert stats['reused_tokens'] == prompt_length and stats['new_prompt_tokens'] == 0
        assert stats['checkpoint_restore'] and stats['prefix_match']
        assert not stats['state_reset'] and not stats['rebuilt_prefix']
        assert stats['reset_reason'] == 'prompt_checkpoint_restore'
        assert stats['state_tokens'] == prompt_length + limit - 1
    assert all(a is b for a, b in zip(bindings, (
        state.recurrent[0].conv.history, state.recurrent[0].delta.memory,
        state.qsa[1].keys, state.qsa[1].values, state.qsa[1].raw_index, state.qsa[1].pooled)))


def test_prompt_checkpoint_branch_matches_fresh_graph_and_replaces_checkpoint():
    svc = bounded_service()
    first = [dict(role='user', content='Hi')]
    run(svc, first, stream=True, limit=6)
    state, checkpoint = svc.session.state, svc.session._prompt_checkpoint
    branch = first + [dict(role='assistant', content='Alternate'), dict(role='user', content='Next')]
    expected = bounded_service()
    expected_text, _ = run(expected, branch, stream=True, limit=4)
    text, _ = run(svc, branch, stream=True, limit=4)
    assert text == expected_text and svc.session.state is state
    assert_active_states_equal(state, expected.session.state)
    stats = svc.session.last_stats
    assert stats['checkpoint_restore'] and not stats['state_reset']
    assert stats['reused_tokens'] == checkpoint.length
    assert svc.session.model.prefills[-1] == expected.session.model.prefills[-1][checkpoint.length:]
    assert svc.session._prompt_checkpoint is not checkpoint
    assert svc.session._prompt_checkpoint.length == stats['prompt_tokens']
    run(svc, first, stream=True, limit=2)
    assert svc.session.state is not state  # Only the newest checkpoint is retained.
    assert not svc.session.last_stats['checkpoint_restore']
    assert svc.session.last_stats['state_reset']


def test_checkpoint_does_not_interfere_with_growing_pending_last_token():
    svc = bounded_service()
    messages = [dict(role='user', content='Hi')]
    text, assistant = run(svc, messages, stream=True, limit=2)
    used, state = list(svc.session.consumed), svc.session.state
    assert used[-1] == ord(text[0])
    run(svc, messages + [assistant, dict(role='user', content='Again')], stream=True, limit=2)
    assert svc.session.state is state
    assert svc.session.last_stats['reused_tokens'] == len(used)
    assert svc.session.model.prefills[-1][0] == ord(text[-1])
    assert not svc.session.last_stats['checkpoint_restore']


@pytest.mark.parametrize('prefix_length', [3, 4, 5, 7])
def test_checkpoint_partial_qsa_pool_is_ignored_and_recomputed(prefix_length):
    from mini_llm.external.qwen4exp_prompt_checkpoint import PromptCheckpoint
    model = BoundedModel()
    state = model.create_generation_state(1, 128)
    tokens = list(range(10, 10 + prefix_length))
    logits = model.prefill(tokens, state)
    checkpoint = PromptCheckpoint.capture(state, tokens, logits)
    for token in (90, 91, 92, 93, 94):
        model.advance(token, state)
    assert state.qsa[1].staged_prefix_length > prefix_length
    restored = checkpoint.restore(state, tokens + [44], tokens + [90, 91, 92, 93, 94])
    np.testing.assert_array_equal(restored, logits)
    assert state.qsa[1].staged_prefix_length == 0
    assert state.length == prefix_length
    expected = model.create_generation_state(1, 128)
    model.prefill(tokens, expected)
    assert_active_states_equal(state, expected)
    model.advance(44, state)
    model.advance(44, expected)
    assert_active_states_equal(state, expected)
    np.testing.assert_array_equal(model.logits(state), model.logits(expected))


def test_checkpoint_memory_is_capacity_independent_and_does_not_pin_state():
    from mini_llm.external.qwen4exp_prompt_checkpoint import PromptCheckpoint
    model = BoundedModel()
    sizes = []
    for capacity in (32, 32768):
        state = model.create_generation_state(1, capacity)
        logits = model.prefill([10, 20, 30], state)
        checkpoint = PromptCheckpoint.capture(state, [10, 20, 30], logits)
        expected_bytes = (state.recurrent[0].conv.history.nbytes + state.recurrent[0].delta.memory.nbytes
                          + state.ple[0].normalized_history.nbytes + logits.nbytes)
        assert checkpoint.host_array_bytes == expected_bytes
        assert not np.shares_memory(checkpoint.recurrent[0][0], state.recurrent[0].conv.history)
        assert not np.shares_memory(checkpoint.ple[0][1], state.ple[0].normalized_history)
        sizes.append(checkpoint.host_array_bytes)
    assert sizes[0] == sizes[1]
    reference = weakref.ref(state)
    model.states.clear()
    del state
    gc.collect()
    assert reference() is None and checkpoint._state() is None


@pytest.mark.parametrize('change', ['tokens', 'consumed', 'state', 'qsa_array', 'failed'])
def test_checkpoint_rejects_invalid_provenance_without_mutation(change):
    from mini_llm.external.qwen4exp_prompt_checkpoint import PromptCheckpoint
    model = BoundedModel()
    state = model.create_generation_state(1, 128)
    tokens = [10, 20, 30]
    checkpoint = PromptCheckpoint.capture(state, tokens, model.prefill(tokens, state))
    model.advance(40, state)
    incoming, consumed = tokens + [50], tokens + [40]
    if change == 'tokens':
        incoming[0] = 11
    elif change == 'consumed':
        consumed[0] = 11
    elif change == 'state':
        state = model.create_generation_state(1, 128)
        model.prefill(consumed, state)
    elif change == 'qsa_array':
        state.qsa[1].keys = state.qsa[1].keys.copy()
    else:
        state.failed = True
    history = state.recurrent[0].conv.history.copy()
    assert checkpoint.restore(state, incoming, consumed) is None
    assert state.length == 4
    np.testing.assert_array_equal(state.recurrent[0].conv.history, history)


def test_reset_discards_prompt_checkpoint_and_unsupported_states_skip_capture():
    from mini_llm.external.qwen4exp_prompt_checkpoint import PromptCheckpoint
    assert PromptCheckpoint.capture({}, [1], np.zeros((1, 2), np.float32)) is None
    svc = service()
    run(svc, [dict(role='user', content='Hi')])
    assert svc.session._prompt_checkpoint is None
    svc = bounded_service()
    run(svc, [dict(role='user', content='Hi')], limit=2)
    assert svc.session._prompt_checkpoint is not None
    svc.session.reset()
    assert svc.session._prompt_checkpoint is None and svc.session.state is None
