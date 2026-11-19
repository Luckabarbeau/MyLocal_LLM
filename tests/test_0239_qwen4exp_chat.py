"""Qwen35 tokenizer, actual-template rendering, incremental chat and CLI contract."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import xp
from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
from mini_llm.external.qwen4exp_chat import ChatSession, render_chat
from test_0106p_qwen_tokenizer import _reader, _byte_encode

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S')
REFERENCE = Path('/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch')


@pytest.mark.parametrize('text,chunks', [
    ('e\u0301\u0308!', ['e\u0301\u0308', '!']),
    ('café 中文 123', ['café', ' 中文', ' ', '1', '2', '3']),
    ("I'm fine\r\n", ['I', "'m", ' fine', '\r\n']),
])
def test_qwen35_mark_runs_and_exact_byte_roundtrip(text, chunks):
    tokenizer = load_gguf_tokenizer(_reader(**{'tokenizer.ggml.pre': 'qwen35'}), vocab_size=262)
    actual = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    assert [piece for piece, _ in actual] == [_byte_encode(c) for c in chunks]
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    with pytest.raises(ValueError, match='chat framing is not validated'):
        tokenizer.chat_prompt(text)  # Generic framing is still fail-closed.


@pytest.fixture(scope='module')
def real_tokenizer():
    if not MODEL.exists():
        pytest.skip('local checkpoint header absent')
    from mini_llm.external.gguf_store import MultiShardTensorStore
    with MultiShardTensorStore.open(MODEL) as store:
        tokenizer = load_gguf_tokenizer(store.metadata, vocab_size=248320)
    return tokenizer


def test_actual_template_thinking_and_text_roles(real_tokenizer):
    no = render_chat(real_tokenizer, [dict(role='user', content='Hi')], thinking=False)
    assert no == '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    yes = render_chat(real_tokenizer, [dict(role='user', content='Hi')], thinking=True)
    assert yes.startswith('<|im_start|>system\nReasoning effort is set to xhigh.')
    assert yes.endswith('<|im_start|>assistant\n<think>\n')
    messages = [dict(role='user', content='Hi'),
                dict(role='assistant', content='Hello!', reasoning_content='A short greeting.'),
                dict(role='user', content='Continue.')]
    prompt = render_chat(real_tokenizer, messages, thinking=False)
    assert '<think>\nA short greeting.\n</think>\n\nHello!<|im_end|>\n' in prompt
    assert real_tokenizer.decode(real_tokenizer.encode(prompt).ids) == prompt
    with pytest.raises(ValueError, match='text system/user/assistant'):
        render_chat(real_tokenizer, [dict(role='tool', content='unsupported')])


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    if not (REFERENCE/'build-cpu/bin/libllama.so').exists() or not MODEL.exists():
        pytest.skip('pinned vocabulary-only oracle unavailable')
    binary = tmp_path_factory.mktemp('qwen35_vocab')/'oracle'
    subprocess.run(['c++', '-std=c++17', '-O2',
        str(ROOT/'tests/fixtures/qwen4exp_tokenizer_reference.cpp'),
        '-I'+str(REFERENCE/'include'), '-I'+str(REFERENCE/'ggml/include'),
        '-L'+str(REFERENCE/'build-cpu/bin'), '-Wl,-rpath,'+str(REFERENCE/'build-cpu/bin'),
        '-lllama', '-lggml', '-lggml-base', '-lggml-cpu', '-o', str(binary)],
        check=True, capture_output=True, text=True, timeout=30)
    return binary


@pytest.mark.parametrize('text', ['Hello, world!', 'e\u0301\u0308 café 中文 🙂',
    "I'm writing Python:\n\treturn 123\r\n", '<|im_start|>user\nHi<|im_end|>\n'])
def test_actual_token_ids_match_pinned_vocab_only_reference(real_tokenizer, oracle, text):
    first = sorted(MODEL.glob('*00001-of-*.gguf'))[0]
    result = subprocess.run([str(oracle), str(first)], input=text, text=True,
                            capture_output=True, check=True, timeout=30)
    expected = [int(v) for v in result.stdout.split()]
    assert real_tokenizer.encode(text).ids == expected


class Characters:
    chat_template = '{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}assistant:'
    def encode(self, text):
        return SimpleNamespace(ids=[ord(c) for c in text])
    def decode(self, ids, **kwargs):
        return ''.join(chr(t) for t in ids)


class FakeModel:
    stop_token_ids = {3}
    def __init__(self, reply='OK'):
        self.reply = list(map(ord, reply)) + [3]
        self.states = []
    def create_generation_state(self, batch, context):
        state = SimpleNamespace(length=0)
        self.states.append(state)
        return state
    def logits(self):
        result = xp.full((1, 256), -100, dtype=xp.float32)
        result[0, self.reply[self.index]] = 100
        return result
    def prefill(self, ids, state):
        state.length += len(ids)
        self.index = 0
        return self.logits()
    def step(self, token, state):
        state.length += 1
        self.index += 1
        return self.logits()


def test_conversation_reuses_exact_prefix_and_reset_isolated():
    model = FakeModel()
    chat = ChatSession(model, Characters(), 256, thinking=False)
    assert ''.join(chat.generate('Hi', 8)) == 'OK'
    first = chat.state
    used = len(chat.consumed)
    assert ''.join(chat.generate('Again', 8)) == 'OK'
    assert chat.state is first and chat.last_stats['reused_tokens'] == used
    assert not chat.last_stats['rebuilt_prefix'] and first.length == len(chat.consumed)
    assert chat.last_stats['decode_forward_tokens'] == 2
    chat.reset()
    assert not chat.messages and chat.state is None and not chat.consumed
    assert ''.join(chat.generate('Hi', 8)) == 'OK'
    assert chat.state is not first


def test_template_rewrite_rebuilds_instead_of_reusing_wrong_state():
    model = FakeModel(' OK ')
    chat = ChatSession(model, Characters(), 256, thinking=False)
    assert ''.join(chat.generate('Hi', 8)) == ' OK '
    first = chat.state
    assert ''.join(chat.generate('Again', 8)) == ' OK '
    assert chat.state is not first and chat.last_stats['rebuilt_prefix']
    assert chat.last_stats['reused_tokens'] == 0


def test_capacity_rejection_does_not_mutate_conversation():
    chat = ChatSession(FakeModel(), Characters(), 10, thinking=False)
    with pytest.raises(ValueError, match='Context needs'):
        list(chat.generate('This will not fit.', 10))
    assert not chat.messages and chat.state is None


def test_cli_help_does_not_load_model():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/chat_qwen4exp.py'), '--help'],
                            capture_output=True, text=True, check=True, timeout=10)
    assert '--interactive' in result.stdout and '--dry-run' in result.stdout
