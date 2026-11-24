"""WP4 device chaining. Readbacks are oracle diagnostics, never native inputs."""
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.native import Runtime, NativeError
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
from test_native_engine_hc import upload, make_weights, read as hc_read
from test_native_engine_recurrent import fixture, reference, create, STEMS as REC_STEMS
from test_native_engine_qsa_layer import synthetic, NAMES
from test_native_engine_qsa_hidden_frontend import STEMS, NORMS, KINDS
from test_native_engine_ple import synthetic as ple_fixture, read as ple_read

pytestmark = pytest.mark.gpu


def close(got, want):
    np.testing.assert_allclose(got, asnumpy(want), rtol=2e-4, atol=2e-4)


def branch_read(branch, name, shape):
    out = np.empty(shape, np.float32)
    branch.readback_for_test(name, out)
    return out


def check_branch(branch, ref, state, hidden, recurrent):
    if recurrent:
        want = ref.step(hidden[None], state)[0]
        close(branch_read(branch, 'history', state.conv.history.shape), state.conv.history)
        close(branch_read(branch, 'matrix', state.delta.memory.shape), state.delta.memory)
    else:
        want, diag = ref.step(hidden, state, position=state.length, diagnostics=True)
        n = state.length
        for name, expected in [('raw', state.raw_index[:n]), ('keys', state.keys[:n]),
                               ('values', state.values[:n]), ('pooled', state.pooled[:n//4])]:
            close(branch_read(branch, name, expected.shape), expected)
        ids = np.empty(min(state.length, 2051), np.int64)
        branch.readback_for_test('ids', ids)
        np.testing.assert_array_equal(ids, asnumpy(diag['selected_ids']))
    close(branch_read(branch, 'output', (2560,)), want)
    return want


@pytest.mark.parametrize('recurrent', [True, False])
@pytest.mark.parametrize('with_ple', [False, True])
def test_native_order_replay(tmp_path, monkeypatch, recurrent, with_ple):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=320*2**20))
        attn, ar, _ = make_weights(stack, tmp_path, r, e, 2560, 5)
        ffn, fr, _ = make_weights(stack, tmp_path, r, e, 2560, 5, seed=5)
        final, cr, _ = make_weights(stack, tmp_path, r, e, 2560, 5, inject=False, seed=6)
        if recurrent:
            spec = Qwen4ExpMixerSpec(2560, 2, 6, 5, 4, 1e-6)
            tensors, values = upload(stack, tmp_path, r, e, fixture(spec))
            ref = reference(spec, [xp.asarray(a) for a in values])
            branch = stack.enter_context(create(e, spec, tensors, capacity=8))
        else:
            tensors, ref = synthetic(stack, tmp_path, r, e, 'ggml_cpu')
            branch = stack.enter_context(e.qsa_layer(4096, *tensors, policy='ggml_cpu'))
        hc = stack.enter_context(e.hc_state(2560, 5))
        if with_ple:
            pw, pr, _, _, _, _, _ = ple_fixture(stack, tmp_path, r, e, rows=256)
            ple = stack.enter_context(pw.state(8))
        xs = [(np.sin(np.arange(2560, dtype=np.float32)*.03125+t*.125)*.2).astype(np.float32)
              for t in range(8)]
        fs = [(x*.17).astype(np.float32) for x in xs]
        us, _ = upload(stack, tmp_path, r, e, xs+fs)
        # No public per-op branch/HC dispatch or new upload may implement mix.
        def forbidden(*a, **k):
            raise AssertionError('Python per-op dispatch/upload in native mix')
        monkeypatch.setattr(branch, 'step', forbidden)
        monkeypatch.setattr(e, 'upload', forbidden)
        stable = bytes(e.stats)
        for replay in range(2):
            state = ref.create_state(1 if recurrent else 4096)
            ps = pr.fresh_state() if with_ple else None
            for t in range(8):
                hc.initialize(us[t]); streams = ar.initialize(xp.asarray(xs[t]))
                if with_ple:
                    streams, ps = pr.step([1, 7, 7, 2, 3, 0, 4, 5][t], streams, ps)
                    ple.step(hc, [1, 7, 7, 2, 3, 0, 4, 5][t], t)
                    close(hc_read(hc, 'streams', 2560), streams)
                    close(ple_read(ple, 'history'), ps.normalized_history)
                hidden, gains = ar.read(streams)
                with monkeypatch.context() as patch:
                    patch.setattr(hc, 'read', forbidden); patch.setattr(hc, 'write', forbidden)
                    assert hc.mix(branch, attn, t) == t+1
                mixed = check_branch(branch, ref, state, hidden, recurrent)
                streams = ar.write(streams, mixed, gains)
                close(hc_read(hc, 'streams', 2560), streams)
                # FFN/MoE is deliberately NOT implemented; independent uploaded output.
                fh, fg = fr.read(streams); ticket = hc.read(ffn)
                close(hc_read(hc, 'hidden', 2560), fh)
                close(hc_read(hc, 'gains', 2560), fg)
                hc.write(ticket, us[8+t]); streams = fr.write(streams, xp.asarray(fs[t]), fg)
                close(hc_read(hc, 'streams', 2560), streams)
                hc.collapse(final); close(hc_read(hc, 'hidden', 2560), cr.collapse(streams))
                assert bytes(e.stats) == stable
                hc.reset()
            branch.reset()
            if with_ple: ple.reset()
        if with_ple: pr.close()


@pytest.mark.parametrize('failure', ['pending', 'position', 'capacity', 'cross', 'width', 'kind', 'stale', 'null', 'numeric'])
def test_mix_failure_contract(tmp_path, failure):
    r = Runtime()
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=2**20))
        w, _, tensors = make_weights(stack, tmp_path, r, e)
        spec = Qwen4ExpMixerSpec(7, 2, 6, 5, 4, 1e-6)
        arrays = fixture(spec)
        if failure == 'numeric': arrays[7][0] = 1  # converted A must be negative
        ts, _ = upload(stack, tmp_path, r, e, arrays)
        branch = stack.enter_context(create(e, spec, ts, capacity=1))
        hc = stack.enter_context(e.hc_state(7, 5))
        us, _ = upload(stack, tmp_path, r, e, [np.ones(7, np.float32)*.2])
        hc.initialize(us[0])
        position, kind, handle = 0, 1, branch._handle
        if failure == 'pending': hc.read(w)
        if failure == 'position': position = 1
        if failure == 'capacity':
            hc.mix(branch, w, 0); hc.reset(); hc.initialize(us[0]); position = 1
        if failure in ('cross', 'width'):
            other = stack.enter_context(r.engine(budget_bytes=2**20)) if failure == 'cross' else e
            sp = Qwen4ExpMixerSpec(3, 1, 1, 2, 1, 1e-6) if failure == 'width' else spec
            more, _ = upload(stack, tmp_path, r, other, fixture(sp))
            handle = stack.enter_context(create(other, sp, more))._handle
        if failure == 'kind': kind = 2
        if failure == 'stale': branch.close()
        done = C.c_uint64(99)
        with pytest.raises(NativeError):
            r._call('ml_hc_mix', hc._handle, w._handle, handle, kind, position,
                    None if failure == 'null' else C.byref(done))
        if failure != 'null': assert done.value == 0
        with pytest.raises(NativeError, match='invalid'): hc_read(hc, 'streams', 7)
        if failure == 'numeric':
            with pytest.raises(NativeError, match='invalid'): _ = branch.position
        elif failure != 'stale': assert branch.position == (1 if failure == 'capacity' else 0)
        hc.reset(); hc.initialize(us[0])
        if failure not in ('stale', 'numeric'):
            branch.reset()
            # Closed public sources and engine must not break native owner lifetimes.
            for u in tensors+ts: u.close()
            e.close()
            assert hc.mix(branch, w, 0) == 1
            with pytest.raises(NativeError, match='unavailable'): hc_read(hc, 'gains', 7)


def checkpoint_upload(stack, e, store, name):
    import cupy as cp
    from mini_llm.ops.quantized_linear import DevicePackedGGUFWeight, quantized_linear
    lease = stack.enter_context(store.lease(name)); d = lease.descriptor
    tensor = stack.enter_context(e.upload(lease)); raw = lease.read(0, d.nbytes)
    if d.type_id == 30:
        bits = cp.asarray(np.frombuffer(raw, '<u2'))
        value = (bits.astype(cp.uint32)<<cp.uint32(16)).view(cp.float32).reshape(d.shape)
    elif d.type_id == 0:
        value = cp.asarray(np.frombuffer(raw, '<f4')).reshape(d.shape)
    else:
        packed = DevicePackedGGUFWeight(cp.asarray(np.frombuffer(raw, np.uint8)), d.shape, KINDS[d.type_id])
        value = lambda x, w=packed: quantized_linear(x[None], w)[0] if x.ndim == 1 else quantized_linear(x, w)
    return tensor, value


@pytest.mark.parametrize('layer_index', [0, 3])
def test_selected_checkpoint_chain(tmp_path, layer_index):
    if xp.__name__ != 'cupy':
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-s',
                        str(Path(__file__).resolve())+f'::test_selected_checkpoint_chain[{layer_index}]', '-x'],
                       env=dict(os.environ, MINI_LLM_BACKEND='cupy', NVIDIA_TF32_OVERRIDE='0'),
                       check=True, timeout=120)
        return
    import cupy as cp
    assert xp is cp
    r = Runtime()
    inventory = json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    with ExitStack() as stack:
        e = stack.enter_context(r.engine(budget_bytes=256*2**20))
        store = stack.enter_context(r.gguf(inventory['shards'][0]['path']))
        eps = store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']
        weights, refs = [], []
        for stem in (f'blk.{layer_index}.hc_attn_', f'blk.{layer_index}.hc_ffn_', 'output_hc_'):
            suffixes = ['norm.weight', 'down.weight', 'up.weight']
            if stem != 'output_hc_': suffixes += ['inject.weight']
            pairs = [checkpoint_upload(stack, e, store, stem+s) for s in suffixes]
            weights.append(stack.enter_context(e.hc_weights(*(p[0] for p in pairs), epsilon=eps,
                                                           arithmetic='reference_fp32')))
            refs.append(Qwen4ExpHyperConnection(*(p[1] for p in pairs), epsilon=eps))
        recurrent = layer_index == 0
        stems = REC_STEMS if recurrent else STEMS+['attn_output.weight']+NORMS
        pairs = [checkpoint_upload(stack, e, store, f'blk.{layer_index}.'+s) for s in stems]
        tensors, values = [p[0] for p in pairs], [p[1] for p in pairs]
        if recurrent:
            spec = Qwen4ExpMixerSpec.from_metadata(store.metadata)
            ref = reference(spec, values)
            branch = stack.enter_context(create(e, spec, tensors, capacity=8))
        else:
            ref = Qwen4ExpAttention(QSAConfig(2560, epsilon=eps), **dict(zip(NAMES, values[:6])),
                                   **dict(zip(['gamma_q','gamma_k','gamma_index_q','gamma_index_k'], values[6:])))
            branch = stack.enter_context(e.qsa_layer(4096, *tensors, policy='ggml_cpu', epsilon=eps))
        hc = stack.enter_context(e.hc_state(2560, 320))
        xs = [(np.sin(np.arange(2560, dtype=np.float32)*.03125+t*.125)*.2).astype(np.float32) for t in range(8)]
        fs = [(x*.1).astype(np.float32) for x in xs]
        us, _ = upload(stack, tmp_path, r, e, xs+fs)
        stable = bytes(e.stats); maximum = 0.
        for replay in range(2):
            state = ref.create_state(1 if recurrent else 4096)
            for t in range(8):
                hc.initialize(us[t]); streams = refs[0].initialize(cp.asarray(xs[t]))
                hidden, gains = refs[0].read(streams)
                assert hc.mix(branch, weights[0], t) == t+1
                mixed = check_branch(branch, ref, state, hidden, recurrent)
                streams = refs[0].write(streams, mixed, gains)
                close(hc_read(hc, 'streams', 2560), streams)
                _, gains = refs[1].read(streams)
                ticket = hc.read(weights[1]); hc.write(ticket, us[8+t])
                streams = refs[1].write(streams, cp.asarray(fs[t]), gains)
                close(hc_read(hc, 'streams', 2560), streams)
                hc.collapse(weights[2]); want = refs[2].collapse(streams)
                got = hc_read(hc, 'hidden', 2560); close(got, want)
                maximum = max(maximum, float(np.max(np.abs(got-asnumpy(want)))))
                hc.reset()
            branch.reset()
        assert bytes(e.stats) == stable and e.stats.peak_bytes < 256*2**20
        print('WP4_CHAIN_CHECKPOINT='+json.dumps(dict(layer=layer_index, steps=8, replays=2,
              max_final_abs=maximum, device_bytes=e.stats.device_bytes, pinned_bytes=e.stats.pinned_bytes)))
