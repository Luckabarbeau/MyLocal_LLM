"""0107B: profile is opt-in and measures the real composed training path."""
import pytest

from mini_llm.specialists.training_run import SpecialistTrainingRun
from test_specialist_training_run_0105e import prepare


def test_cpu_stage_profile_wraps_real_frozen_composition(tmp_path):
    _, comp, sampler, cfg = prepare(tmp_path)
    run = SpecialistTrainingRun(comp, sampler, config=cfg)
    report = run.train_step(profile=True)
    profile = report['profile']
    for key in ('foundation_forward_s', 'specialist_forward_s', 'head_forward_s',
                'head_backward_s', 'specialist_backward_s', 'optimizer_and_refresh_s',
                'clip_s', 'sample_s', 'total_step_s', 'supervised_tokens_per_s'):
        assert profile[key] > 0, key
    assert profile['total_step_s'] >= profile['specialist_forward_s']
    assert run.step == 1
    assert 'profile' not in run.train_step()


def test_profile_steps_is_explicitly_bounded(tmp_path):
    _, comp, sampler, cfg = prepare(tmp_path)
    run = SpecialistTrainingRun(comp, sampler, config=cfg)
    found = run.train_steps(2, profile_steps=1)
    assert 'profile' in found[0]
    assert 'profile' not in found[1]
    with pytest.raises(ValueError, match='profile_steps'):
        run.train_steps(0, profile_steps=-1)
