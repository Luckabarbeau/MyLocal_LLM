"""0077D: automatic mixture-phase changes preserve optimizer-state resumes."""
import hashlib
import json

import pytest

from train_model import plan_data_mixture_resume
from mini_llm.data.streaming_sources import StreamSpec, _canonical


def spec(*, weight=0.15, revision='a' * 40, name='finemath'):
    return StreamSpec(name=name, dataset='demo/finemath', revision=revision,
                      weight=weight, text_field='text', license='test')


def snapshot(specs, *, segment_tokens=524288, reuse_batches=64):
    return {
        'registry': [s.__dict__ for s in specs],
        'registry_fingerprint': hashlib.sha256(
            _canonical([s.__dict__ for s in specs]).encode()
        ).hexdigest(),
        'segment_tokens': segment_tokens,
        'reuse_batches': reuse_batches,
    }


def plan(prior, specs, *, manifest='aaa', seg=524288, reuse=64):
    return plan_data_mixture_resume(
        resume_from='checkpoint', stored_provenance=prior, stream_specs=specs,
        segment_tokens=seg, reuse_batches=reuse, mixture_sha=manifest,
    )


def test_initial_new_mixture_automatically_starts_new_phase():
    reasons, reuse = plan({'local_mixture_manifest_sha256': 'aaa'}, [spec()])
    assert reasons == ['remote sources enabled/disabled']
    assert reuse is False


def test_exact_stream_registry_resume_keeps_cursor():
    s = spec()
    reasons, reuse = plan({'streaming': snapshot([s]),
                           'local_mixture_manifest_sha256': 'aaa'}, [s])
    assert reasons == []
    assert reuse is True


@pytest.mark.parametrize('changes', [
    {'weight': 0.20}, {'revision': 'b' * 40},
])
def test_changed_source_or_weight_starts_phase(changes):
    old = spec()
    reasons, reuse = plan({'streaming': snapshot([old]),
                           'local_mixture_manifest_sha256': 'aaa'}, [spec(**changes)])
    assert 'remote dataset registry/revision/weights changed' in reasons
    assert not reuse


def test_local_manifest_change_starts_phase_even_if_remote_unchanged():
    s = spec()
    reasons, reuse = plan({'streaming': snapshot([s]),
                           'local_mixture_manifest_sha256': 'previous'}, [s])
    assert 'local shard mixture manifest changed' in reasons
    assert not reuse


def test_segment_policy_change_starts_phase():
    s = spec()
    reasons, reuse = plan({'streaming': snapshot([s])}, [s], reuse=32)
    assert 'remote segment/reuse policy changed' in reasons
    assert not reuse


def test_remove_remote_sources_is_automatic():
    s = spec()
    reasons, reuse = plan({'streaming': snapshot([s])}, [])
    assert 'remote sources enabled/disabled' in reasons
    assert not reuse


def test_no_checkpoint_means_initial_run():
    reasons, reuse = plan_data_mixture_resume(
        resume_from=None, stored_provenance=None, stream_specs=[spec()],
        segment_tokens=524288, reuse_batches=64, mixture_sha=None)
    assert reasons == [] and not reuse


def test_phase_history_saved_to_checkpoint(tmp_path):
    import numpy as np
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer
    from test_streaming_sources import spec as fixture_spec, TinyTokenizer, ResumableRows
    import mini_llm.data.streaming_sources as streaming
    original = streaming._default_factory
    streaming._default_factory = lambda _: ResumableRows()
    shard = tmp_path / 'local.bin'
    np.random.default_rng(42).integers(3, 40, size=4096, dtype=np.uint16).tofile(shard)
    try:
        trainer = ExtendedTrainer(
            model=DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=1),
            train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
            train_source_shards={'local': [str(shard)]},
            val_source_shards={'local': [str(shard)]},
            source_weights={'local': 1.0},
            batch_size=1, seq_length=16, grad_accum_steps=1,
            warmup_steps=1, total_steps=5,
            checkpoint_dir=str(tmp_path/'checkpoint'),
            stream_specs=[fixture_spec()], stream_tokenizer=TinyTokenizer(),
            stream_segment_tokens=1024, stream_reuse_batches=3,
            stream_cache_dir=str(tmp_path),
        )
        try:
            history = [
                {'kind': 'data_mixture_changed', 'start_step': 40500,
                 'previous': {'remote_registry': []},
                 'new': {'remote_registry': [{'name': 'fake'}]}}
            ]
            trainer.data_fork = history[0]
            trainer.mixture_history = history
            trainer.mixture_phase_start_source_counts = {'local': 14000}
            trainer.mixture_phase_start_batch_counts = {'local': 700}
            trainer.step = 40500
            trainer.save()
            saved = json.loads((tmp_path/'checkpoint'/'training_state.json').read_text())
            assert saved['data_provenance']['mixture_history'] == history
            assert saved['data_provenance']['mixture_phase_start_source_counts'] == {'local': 14000}
            assert saved['data_provenance']['mixture_phase_start_batch_counts'] == {'local': 700}
            assert saved['step'] == 40500
            assert saved['data_provenance']['streaming']['registry'][0]['name'] == 'fake'
        finally:
            trainer.streaming_cache.close()
    finally:
        streaming._default_factory = original


def test_weight_change_keeps_unchanged_remote_stream_cursor():
    from train_model import restore_compatible_stream_cursors

    class FakeSource:
        state = None
        def load_state_dict(self, state):
            self.state = state

    class FakeCache:
        segment_tokens = 524288
        reuse_batches = 64
        specs = {'finemath': spec(weight=0.2),
                 'new_source': spec(name='new_source')}
        sources = {'finemath': FakeSource(), 'new_source': FakeSource()}

    old = spec(weight=0.15)
    prior = snapshot([old])
    prior['sources'] = {'finemath': {'generation': 12, 'next_cursor': {'pos': 1234}}}
    restored = restore_compatible_stream_cursors(FakeCache(), prior)
    assert restored == ['finemath']
    assert FakeCache.sources['finemath'].state['generation'] == 12
    assert FakeCache.sources['new_source'].state is None


def test_revision_change_resets_remote_stream_cursor():
    from train_model import restore_compatible_stream_cursors

    class FakeSource:
        state = None
        def load_state_dict(self, state):
            self.state = state

    class FakeCache:
        segment_tokens = 524288
        reuse_batches = 64
        specs = {'finemath': spec(revision='b' * 40)}
        sources = {'finemath': FakeSource()}

    old = spec(revision='a' * 40)
    prior = snapshot([old])
    prior['sources'] = {'finemath': {'generation': 3}}
    assert restore_compatible_stream_cursors(FakeCache(), prior) == []
    assert FakeCache.sources['finemath'].state is None
