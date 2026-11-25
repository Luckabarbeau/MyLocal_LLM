"""WP8 bitwise native rollback; no Python trajectory/quality acceptance claim."""
from contextlib import ExitStack
import ctypes as C

import numpy as np
import pytest

from mini_llm.native import NativeError
from mini_llm.native.chat import NativeChatSession
from mini_llm.native.generation import NativeGeneration
from native_prompt_fixture import mixed_setup
from test_native_engine_decoder import small_setup, forbidden
from test_native_engine_generation import collect, await_state
from test_native_engine_moe import libraries, selected_library
from test_native_engine_chat import tokenizer

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def retain_ggml(selected_library):
    library = C.CDLL(selected_library)
    yield library


def snapshot_bytes(cp):
    function = cp._runtime._lib.ml_decoder_checkpoint_readback_for_test
    function.argtypes = [C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)]
    function.restype = C.c_int
    size = C.c_uint64()
    assert function(cp._handle,None,0,C.byref(size)) == 0
    buffer = C.create_string_buffer(size.value)
    assert function(cp._handle,buffer,size.value,C.byref(size)) == 0
    return buffer.raw


@pytest.mark.parametrize('chunks', [(3,), (1,2), (1,1,1)])
def test_gdn_checkpoint_chunk_replay_and_resident_logits(tmp_path, libraries, chunks):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack,tmp_path,libraries)
        sampler = stack.enter_context(e.sampler(7,seed=19))
        start = 0
        prompt = [1,2,3]
        for chunk in chunks:
            d.consume(prompt[start:start+chunk]); start += chunk
        expected = sampler.sample_decoder(d,metadata=True)
        cp = stack.enter_context(d.checkpoint())
        saved = snapshot_bytes(cp)
        baseline = e.stats.device_bytes
        suffix = d.step(4).tobytes()
        assert cp.restore(d) == 3 and d.state == (3,False)
        assert bytes(sampler.sample_decoder(d,metadata=True)) == bytes(expected)
        assert d.step(4).tobytes() == suffix
        cp.restore(d)
        second = stack.enter_context(d.checkpoint())
        assert snapshot_bytes(second) == saved
        assert e.stats.device_bytes == baseline
        with pytest.raises(NativeError,match='stale'):
            cp.restore(d)
        second.close()
        with pytest.raises(NativeError):
            second.restore(d)


@pytest.mark.parametrize('fmt', ['reference_fp32','strata_int8_per64_fp16'])
def test_mixed_pool_page_tail_ple_eos_bitwise(tmp_path, libraries, fmt):
    with ExitStack() as stack:
        d, _, e = mixed_setup(stack,tmp_path,libraries,fmt)
        tokens = [1,7,7,2,3,4,5,6,1,2,3,4]
        sizes = []
        # Prefix 3 completes a pool during decode, 4 starts a fresh pool/page,
        # 5 preserves a partial page and overwritten raw-ring dependencies.
        for boundary in (0,3,4,5,7):
            d.reset()
            d.consume(tokens[:boundary])
            cp = stack.enter_context(d.checkpoint(host_budget_bytes=2**20))
            saved = snapshot_bytes(cp)
            sizes.append(cp.host_bytes)
            expected = [d.step(token).tobytes() for token in tokens[boundary:boundary+4]]
            assert cp.restore(d) == boundary
            # Capture directly after restore verifies committed GDN/PLe/raw-ring
            # and resident logits, then continue from the replacement checkpoint.
            restored = stack.enter_context(d.checkpoint(host_budget_bytes=2**20))
            assert snapshot_bytes(restored) == saved
            got = [d.step(token).tobytes() for token in tokens[boundary:boundary+4]]
            assert got == expected
            restored.restore(d)
            alternative = list(reversed(tokens[boundary:boundary+4]))
            got = [d.step(token).tobytes() for token in alternative]
            d.reset(); d.consume(tokens[:boundary])
            want = [d.step(token).tobytes() for token in alternative]
            assert got == want  # Different suffix never sees old pooled/page-tail rows.
        # Snapshot size is independent of QSA prefix length; only whether resident
        # logits exist changes payload size. Neither KV nor completed pools copied.
        assert len(set(sizes[1:])) == 1
        assert sizes[0] < sizes[1] < 2**20
        assert e.stats.device_bytes < 512*2**20
        print('WP8 mixed checkpoint host_bytes',sizes,'device_bytes',e.stats.device_bytes)


def test_checkpoint_budget_epoch_close_and_worker_exclusivity(tmp_path, libraries):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack,tmp_path,libraries)
        sampler = stack.enter_context(e.sampler(7,seed=1))
        d.consume([1])
        cp = stack.enter_context(d.checkpoint())
        with pytest.raises(NativeError,match='budget'):
            d.checkpoint(host_budget_bytes=1)
        assert cp.restore(d) == 1  # Failed budget preflight didn't discard it.
        with NativeGeneration.start(d,sampler,2,3,queue_capacity=1) as worker:
            await_state(worker,lambda event,queued,done:queued==1)
            for action in (d.checkpoint,lambda:cp.restore(d),cp.close):
                with pytest.raises(NativeError,match='handle'):
                    action()
            worker.cancel(); collect(worker)
        assert cp.restore(d) == 1  # Known native checkpoint unaffected by cancellation.
        d.reset(); d.consume([1])
        with pytest.raises(NativeError,match='stale'):
            cp.restore(d)  # Same length, different epoch must not authorize restore.
        cp = stack.enter_context(d.checkpoint())
        e.close()  # Decoder safely retains engine; checkpoint itself is weak.
        assert cp.restore(d) == 1
        d.close()
        with pytest.raises(NativeError):
            cp.restore(d)
        with pytest.raises(AssertionError):
            snapshot_bytes(cp)  # Decoder close released the sole host state buffer.


def test_restore_partial_failure_quarantines_and_wrong_owner(tmp_path,libraries):
    with ExitStack() as stack:
        first, second = tmp_path/'first', tmp_path/'second'
        first.mkdir(); second.mkdir()
        d, (r, _, _), _ = small_setup(stack,first,libraries)
        other, _, _ = small_setup(stack,second,libraries)
        d.consume([1])
        cp = stack.enter_context(d.checkpoint())
        committed = C.c_uint64()
        with pytest.raises(NativeError,match='stale'):
            r._call('ml_decoder_checkpoint_restore',other._handle,cp._handle,C.byref(committed))
        assert other.state==(0,False) and d.state==(1,False)
        inject = r._lib.ml_decoder_checkpoint_fail_restore_for_test
        inject.argtypes = [C.c_uint64]
        inject.restype = C.c_int
        assert inject(cp._handle)==0
        d.consume([2])
        with pytest.raises(NativeError,match='injected'):
            cp.restore(d)
        assert d.state==(2,True)
        with pytest.raises(NativeError,match='stale'):
            cp.restore(d)
        with pytest.raises(NativeError,match='unresumable'):
            d.consume([1])
        d.reset()
        assert d.consume([1])==1 and d.state==(1,False)


def test_chat_repeated_native_no_prefill_pending_eos_cancel(tmp_path,libraries,monkeypatch):
    with ExitStack() as stack:
        d, (_, e, _), _ = small_setup(stack,tmp_path,libraries)
        sampler = stack.enter_context(e.sampler(7,seed=3))
        chat = NativeChatSession(d,sampler,tokenizer(),eos_ids=())
        for prompt,maximum in (([1],3),([1,2],2)):
            text = ''.join(chat.stream_tokens(prompt,maximum))
            committed,pending = chat.committed_tokens,chat.pending_token
            with monkeypatch.context() as patch:
                patch.setattr(d,'consume',forbidden)
                patch.setattr(d,'reset',forbidden)
                for _ in range(2):
                    assert ''.join(chat.stream_tokens(prompt,maximum)) == text
                    assert chat.committed_tokens == committed and chat.pending_token == pending
                    assert d.state == (len(committed),False)
                    assert chat.last_stats['checkpoint_restore']
                    assert chat.last_stats['prefill_seconds'] == 0
            prefix = list(chat.committed_tokens)+[chat.pending_token]
            if len(prefix)<d.capacity:
                list(chat.stream_tokens(prefix,1))
                assert d.state == (len(prefix),False)
        stream = chat.stream_tokens([1],3)
        next(stream); stream.close()
        assert chat.require_reset
        list(chat.stream_tokens([1],1))
        assert not chat.last_stats['checkpoint_restore']
        assert chat.last_stats['reset_reason']=='require_reset'
        chat.reset()
        eos = NativeChatSession(d,sampler,tokenizer(),eos_ids=range(7))
        for _ in range(2):
            assert list(eos.stream_tokens([1,2],2)) == []
            assert eos.committed_tokens==(1,2) and eos.pending_token in range(7)
            assert d.state==(2,False)
        assert eos.last_stats['checkpoint_restore']
