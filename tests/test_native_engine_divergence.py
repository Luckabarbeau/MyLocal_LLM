"""Read-only MoE snapshots and retained identical-input WP7 evidence gates."""
from contextlib import ExitStack
import ctypes as C
import json
from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import NativeError
from test_native_engine_moe import libraries, selected_library, fixture, native_setup
from test_native_engine_decoder import small_setup
from native_decoder_divergence import snapshot, metric


@pytest.fixture(scope='module', autouse=True)
def ggml_dso_keepalive(selected_library):
    # Sequential native-only owners otherwise dlclose/reload the backend DSO.
    # Keep the oracle library selected for this test module loaded throughout.
    return C.CDLL(selected_library)


def check_decomposition(diag):
    assert diag['ids'].dtype == np.int32
    assert np.all(np.diff(diag['ids']) > 0)
    weighted = diag['cpu_rows']*diag['weights'][:,None]
    np.testing.assert_array_equal(diag['terms'],weighted)
    routed = np.zeros_like(diag['routed'])
    for term in weighted:
        routed += term
    np.testing.assert_array_equal(diag['routed'],routed)
    np.testing.assert_array_equal(diag['output'],diag['routed']+diag['shared'])


def test_owner_diagnostic_readonly_and_gating(libraries,tmp_path):
    with ExitStack() as stack:
        path=tmp_path/'owner.gguf'; data=fixture(path)
        r,e,store,leases,uploads,owner,_=native_setup(stack,path,data,libraries,())
        with pytest.raises(NativeError,match='unavailable'):
            snapshot(owner,256,256)
        with owner.submit(data['x']) as job:
            with pytest.raises(NativeError,match='unavailable'):
                snapshot(owner,256,256)
            job.wait(); output,ids,weights=job.readback_for_test()
            accounting=bytes(e.stats); diag=snapshot(owner,256,256)
            check_decomposition(diag)
            np.testing.assert_array_equal(diag['input'],data['x'])
            np.testing.assert_array_equal(diag['ids'],ids)
            np.testing.assert_array_equal(diag['weights'],weights)
            np.testing.assert_array_equal(diag['output'],output)
            sentinel=np.full(10,123,np.float32)
            for kind,count in ((1,9),(99,10)):
                with pytest.raises(NativeError):
                    r._call('ml_moe_diagnostic_for_test',owner._handle,kind,sentinel.ctypes.data,count)
                np.testing.assert_array_equal(sentinel,np.full(10,123,np.float32))
            with pytest.raises(NativeError):
                r._call('ml_moe_diagnostic_for_test',owner._handle,1,None,10)
            assert bytes(e.stats)==accounting
        owner.reset()
        with pytest.raises(NativeError,match='unavailable'):
            snapshot(owner,256,256)


def test_decoder_diagnostic_retained_epoch(libraries,tmp_path):
    with ExitStack() as stack:
        decoder,(r,e,_),_=small_setup(stack,tmp_path,libraries)
        with pytest.raises(NativeError,match='unavailable'):
            snapshot(decoder,256,256,15)
        decoder.step(1); accounting=bytes(e.stats)
        diag=snapshot(decoder,256,256,15)
        check_decomposition(diag)
        np.testing.assert_array_equal(diag['input'],decoder.trace_for_test(15,'hc_ffn_read'))
        assert bytes(e.stats)==accounting
        with pytest.raises(NativeError,match='unavailable'):
            snapshot(decoder,256,256,48)
        decoder.reset()
        with pytest.raises(NativeError,match='unavailable'):
            snapshot(decoder,256,256,15)


@pytest.mark.parametrize('kind',('actual','synthetic'))
def test_retained_identical_input_evidence(kind):
    """No recalculation of trajectories; verify retained independent experiments.

    Opt-in evidence files are local activation snapshots, never production inputs.
    Capturing fresh evidence uses native_decoder_divergence.py's bounded phases.
    """
    root=Path('logs/wp7_divergence')/kind
    if not (root/'report.json').is_file():
        pytest.skip('capture WP7 divergence evidence explicitly first')
    report=json.loads((root/'report.json').read_text())
    with np.load(root/'python.npz') as python,np.load(root/'replay.npz') as native:
        for lineage in ('native','python'):
            np.testing.assert_array_equal(native[f'{lineage}_ids'],python[f'{lineage}_ids'])
            np.testing.assert_array_equal(native[f'{lineage}_cpu_rows'],python[f'{lineage}_cpu_rows'])
            for field in ('logits','weights','terms','routed','shared','output','hc_write'):
                np.testing.assert_allclose(native[f'{lineage}_{field}'],python[f'{lineage}_{field}'],
                    rtol=2e-4,atol=2e-4,err_msg=f'{kind} {lineage} {field}')
            assert report['same_input'][lineage]['cpu_rows']['exact']
        # Cross-lineage differences must not be mislabeled as same-input failures.
        assert not np.array_equal(native['native_input'],native['python_input'])
        assert not np.array_equal(native['native_cpu_rows'],native['python_cpu_rows'])
        assert report['lineage']['native']['ids_equal']
        assert report['captured_native_hc_write']['exact']
        assert report['captured_python_hc_write']['exact']


@pytest.mark.parametrize('sample',('layer6_token0','layer6_token1','layer7_token1'))
def test_full48_conditioned_moe_diagnostic_only(sample):
    """Pointwise operator contract, deliberately NOT full-trajectory acceptance."""
    root=Path('logs/wp7_divergence/actual_full')
    folder=root if sample=='layer6_token0' else root/('token1' if sample=='layer6_token1' else 'layer7_token1')
    if not (folder/'report.json').is_file():
        pytest.skip('capture full-model four-way evidence explicitly first')
    report=json.loads((folder/'report.json').read_text())
    with np.load(folder/'native.npz',allow_pickle=False) as capture, np.load(folder/'python.npz',allow_pickle=False) as oracle, np.load(folder/'replay.npz',allow_pickle=False) as replay:
        assert int(capture['full_model_layers'][0])==48
        assert int(oracle['full_model_layers'][0])==48
        assert all(f'{i}_{stage}' in capture and f'{i}_{stage}' in oracle for i in range(48)
                   for stage in ('hc_attention_read','post_mixer','hc_ffn_read','post_moe'))
        for lineage in ('native','python'):
            np.testing.assert_array_equal(replay[f'{lineage}_input'],oracle[f'{lineage}_input'])
            np.testing.assert_array_equal(replay[f'{lineage}_ids'],oracle[f'{lineage}_ids'])
            np.testing.assert_array_equal(replay[f'{lineage}_cpu_rows'],oracle[f'{lineage}_cpu_rows'])
            for field in ('logits','weights','terms','routed','shared','output','hc_write'):
                np.testing.assert_allclose(replay[f'{lineage}_{field}'],oracle[f'{lineage}_{field}'],
                    rtol=2e-4,atol=2e-4,err_msg=f'{sample} identical-input {lineage} {field}')
            check_decomposition({k:replay[f'{lineage}_{k}'] for k in
                                 ('ids','weights','cpu_rows','terms','routed','shared','output')})
        assert report['captured_native_hc_write']['exact']
        assert report['captured_python_hc_write']['exact']
        if sample.startswith('layer6'):
            assert report['original_owner_buffers_retained']
            assert all(v['exact'] for v in report['capture_replay'].values())
            np.testing.assert_array_equal(replay['first_hc_hidden'],capture['0_hc_attention_read'])
            np.testing.assert_array_equal(oracle['first_hc_hidden'],oracle['0_hc_attention_read'])


def test_full48_trajectory_status_is_separate_from_conditioned_gate():
    root=Path('logs/wp7_divergence/actual_full')
    if not (root/'native_replay.json').is_file():
        pytest.skip('capture full-model two-token/reset evidence explicitly first')
    replay=json.loads((root/'native_replay.json').read_text())
    assert replay==dict(tokens=[1,7],layers=48,exact_logits_after_reset=True,final_position=0)
    for phase in ('native','python'):
        preflight=json.loads((root/f'{phase}_preflight.json').read_text())
        assert preflight['gpu_mask']=='GPU-e961516f-d77d-813c-98e9-2f5403f71631'
        assert preflight['admitted']
        assert preflight['free_bytes']>=preflight['budget_bytes']+preflight['headroom_bytes']
    for folder in (root,root/'token1'):
        report=json.loads((folder/'report.json').read_text())
        with np.load(folder/'native.npz') as native,np.load(folder/'python.npz') as oracle:
            n,p=native['full_logits'],oracle['full_logits']
            assert n.shape==p.shape==(1,248320)
            assert np.isfinite(n).all() and np.isfinite(p).all()
            logits=metric(n.reshape(-1),p.reshape(-1))
            assert logits['violations']==report['full_logits']['violations']
            expected=logits['violations']==0 and all(v['violations']==0 for v in report['boundaries'].values())
            assert report['strict_trajectory_gate_passed']==expected
            # Passing a pointwise test must never turn logit diagnostics into a
            # quality/performance acceptance flag, regardless of future drift.
            assert report['short_token_distribution_diagnostics']['accepted_quality_gate'] is False


def test_full48_gdn8_same_input_history_diagnostic_only():
    root=Path('logs/wp7_divergence/actual_full/gdn8')
    if not (root/'states.npz').is_file():
        pytest.skip('capture retained two-step GDN8 evidence explicitly first')
    report=json.loads((root/'report.json').read_text())
    with np.load(root/'states.npz',allow_pickle=False) as saved:
        for lineage in ('native','python'):
            for t in (0,1):
                for field in ('output','history','matrix'):
                    np.testing.assert_allclose(saved[f'{lineage}_{t}_native_{field}'],
                        saved[f'{lineage}_{t}_python_{field}'],rtol=2e-4,atol=2e-4)
                assert report['captured_hc_write'][lineage][t]['violations']==0
        assert all(v['exact'] for v in report['captured_hc_write']['native'])
        control=metric(saved['native_history_common_current_output'],saved['python_history_common_current_output'])
        assert control==report['same_current_input_different_prior_history']
