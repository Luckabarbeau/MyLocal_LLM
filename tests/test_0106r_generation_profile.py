"""CPU-only HostStageProfiler contracts using fake components and a fake clock."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mini_llm/diagnostics/generation_profile.py'


def _load_profiler():
    spec = importlib.util.spec_from_file_location('_generation_profile_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self):
        self.now = 0
        self.scale = 1

    def __call__(self):
        return self.now

    def advance(self, nanoseconds):
        self.now += nanoseconds * self.scale


class FakeStage:
    def __init__(self, clock, duration_ns, increment=1):
        self.clock = clock
        self.duration_ns = duration_ns
        self.increment = increment
        self.calls = []

    def forward(self, value, *, token):
        self.calls.append((value, token))
        self.clock.advance(self.duration_ns)
        return value + self.increment


class FakeMoe:
    def __init__(self, clock):
        self.clock = clock
        self.router = FakeStage(clock, 3_000_000)
        self.experts = FakeStage(clock, 7_000_000)

    def forward(self, value, *, token):
        self.clock.advance(2_000_000)
        value = self.router.forward(value, token=token)
        self.clock.advance(5_000_000)
        value = self.experts.forward(value, token=token)
        self.clock.advance(11_000_000)
        return value


class FakeBlock:
    def __init__(self, clock):
        self.norm1 = FakeStage(clock, 2_000_000)
        self.mixer = FakeStage(clock, 5_000_000)
        self.norm2 = FakeStage(clock, 3_000_000)
        self.moe = FakeMoe(clock)

    def forward(self, value, *, token):
        value = self.norm1.forward(value, token=token)
        value = self.mixer.forward(value, token=token)
        value = self.norm2.forward(value, token=token)
        return self.moe.forward(value, token=token)


class FakeModel:
    def __init__(self, clock, blocks=2):
        self.blocks = [FakeBlock(clock) for _ in range(blocks)]

    def forward(self, value, *, token):
        for block in self.blocks:
            value = block.forward(value, token=token)
        return value


@pytest.fixture
def profiler_module(monkeypatch):
    module = _load_profiler()
    clock = FakeClock()
    monkeypatch.setattr(module, 'perf_counter_ns', clock)
    return module, clock


def _assert_stats(actual, *, calls, total_ms, max_ms):
    assert set(actual) == {'calls', 'total_ms', 'mean_ms', 'max_ms'}
    assert actual['calls'] == calls
    # Integer-nanosecond inputs leave only float division roundoff, not clock jitter.
    assert actual['total_ms'] == pytest.approx(total_ms, rel=1e-12, abs=1e-12)
    assert actual['mean_ms'] == pytest.approx(total_ms / calls, rel=1e-12, abs=1e-12)
    assert actual['max_ms'] == pytest.approx(max_ms, rel=1e-12, abs=1e-12)


def _remember(component):
    return dict(vars(component)), component.forward


def _assert_restored(component, original):
    attributes, forward = original
    assert vars(component).keys() == attributes.keys()
    for key, value in attributes.items():
        assert vars(component)[key] is value
    if 'forward' in attributes:
        assert component.forward is forward
    else:
        assert component.forward.__self__ is forward.__self__
        assert component.forward.__func__ is forward.__func__


def test_counts_phases_inclusive_totals_and_precise_restoration(profiler_module):
    module, clock = profiler_module
    model = FakeModel(clock)
    components = [component for block in model.blocks for component in (
        block.norm1, block.mixer, block.norm2, block.moe,
        block.moe.router, block.moe.experts)]
    originals = [(component, _remember(component)) for component in components]
    profiler = module.HostStageProfiler(model)
    token = object()

    with profiler as entered:
        assert entered is profiler
        profiler.phase = 'prefill'
        assert model.forward(0, token=token) == 10
        clock.scale = 2
        assert model.forward(10, token=token) == 20
        profiler.phase = 'decode'
        clock.scale = 3
        assert model.forward(20, token=token) == 30
        snapshot = profiler.snapshot()

    assert set(snapshot['phases']) == {'prefill', 'decode'}
    durations_ms = {'norm1': 2, 'mixer': 5, 'norm2': 3,
                    'moe': 28, 'router': 3, 'experts': 7}
    for phase, calls, maximum_scale in [('prefill', 4, 2), ('decode', 2, 3)]:
        assert set(snapshot['phases'][phase]) == set(durations_ms)
        for stage, duration in durations_ms.items():
            _assert_stats(snapshot['phases'][phase][stage], calls=calls,
                          total_ms=duration * 6, max_ms=duration * maximum_scale)
    # MoE includes its 18 ms of own work and both nested calls; it is not exclusive.
    prefill = snapshot['phases']['prefill']
    assert prefill['moe']['total_ms'] == 108 + sum(
        prefill[stage]['total_ms'] for stage in ('router', 'experts'))
    assert 'Inclusive host wall time' in snapshot['timing_semantics']
    assert 'not additive' in snapshot['timing_semantics']
    assert 'not isolated GPU kernel time' in snapshot['timing_semantics']
    assert json.loads(json.dumps(snapshot)) == snapshot
    for component, original in originals:
        _assert_restored(component, original)
    for block in model.blocks:
        assert [value for value, _ in block.norm1.calls] == (
            [0, 10, 20] if block is model.blocks[0] else [5, 15, 25])
        assert all(received_token is token for _, received_token in block.moe.experts.calls)


def test_clear_retains_phase_and_wrappers_and_snapshots_are_detached(profiler_module):
    module, clock = profiler_module
    stage = FakeStage(clock, 250_001)
    profiler = module.HostStageProfiler(SimpleNamespace(blocks=[SimpleNamespace(norm1=stage)]))
    with profiler:
        wrapper = stage.forward
        stage.forward(0, token=None)
        saved = profiler.snapshot()
        assert set(saved['phases']) == {'unassigned'}
        saved['phases']['unassigned']['norm1']['calls'] = 999
        assert profiler.snapshot()['phases']['unassigned']['norm1']['calls'] == 1
        before_clear = profiler.snapshot()
        profiler.phase = 'decode'
        profiler.clear()
        assert profiler.phase == 'decode'
        assert stage.forward is wrapper
        assert profiler.snapshot()['phases'] == {}
        assert before_clear['phases']['unassigned']['norm1']['calls'] == 1
        stage.forward(2, token=None)
        _assert_stats(profiler.snapshot()['phases']['decode']['norm1'], calls=1,
                      total_ms=0.250001, max_ms=0.250001)
        assert set(before_clear['phases']) == {'unassigned'}
    profiler.clear()
    assert profiler.phase == 'decode'
    assert profiler.snapshot()['phases'] == {}


def test_context_reuse_preserves_counters_until_cleared(profiler_module):
    module, clock = profiler_module
    stage = FakeStage(clock, 1_000_000)
    original = _remember(stage)
    profiler = module.HostStageProfiler(SimpleNamespace(blocks=[SimpleNamespace(norm1=stage)]))
    for _ in range(2):
        with profiler:
            stage.forward(0, token=None)
        _assert_restored(stage, original)
    _assert_stats(profiler.snapshot()['phases']['unassigned']['norm1'], calls=2,
                  total_ms=2, max_ms=1)


def test_phase_is_stringified_and_captured_before_forward(profiler_module):
    module, clock = profiler_module
    stage = SimpleNamespace()
    profiler = module.HostStageProfiler(SimpleNamespace(blocks=[SimpleNamespace(mixer=stage)]))
    result = object()

    def forward():
        profiler.phase = 'changed_inside_forward'
        clock.advance(123)
        return result

    stage.forward = forward
    profiler.phase = 42
    with profiler:
        assert stage.forward() is result
    assert set(profiler.snapshot()['phases']) == {'42'}
    _assert_stats(profiler.snapshot()['phases']['42']['mixer'], calls=1,
                  total_ms=0.000123, max_ms=0.000123)


def test_alias_is_wrapped_once_and_first_encountered_stage_owns_calls(profiler_module):
    module, clock = profiler_module
    shared = FakeStage(clock, 4_000_000)
    shared.router = shared
    shared.experts = shared
    original = _remember(shared)
    block = SimpleNamespace(norm1=shared, mixer=shared, norm2=shared, moe=shared)
    model = SimpleNamespace(blocks=[block, block])
    profiler = module.HostStageProfiler(model)
    with profiler:
        for alias in (block.norm1, block.mixer, block.norm2, block.moe,
                      block.moe.router, block.moe.experts, model.blocks[1].mixer):
            assert alias.forward(0, token=None) == 1
        assert set(profiler.snapshot()['phases']['unassigned']) == {'norm1'}
        _assert_stats(profiler.snapshot()['phases']['unassigned']['norm1'], calls=7,
                      total_ms=28, max_ms=4)
    _assert_restored(shared, original)


class ForwardFailure(RuntimeError):
    pass


class CallableForward:
    def __init__(self, clock, error):
        self.clock = clock
        self.error = error

    def __call__(self, value, *, token):
        self.clock.advance(1_000_000)
        if self.error is not None:
            raise self.error
        return value, token


class RestorableStage:
    def __init__(self, clock, error):
        self.implementation = CallableForward(clock, error)

    def forward(self, value, *, token):
        return self.implementation(value, token=token)


@pytest.mark.parametrize('attribute_kind', ['inherited', 'instance_bound', 'function', 'callable'])
@pytest.mark.parametrize('exit_kind', ['normal', 'body_exception', 'forward_exception'])
def test_exact_forward_attribute_restoration(profiler_module, attribute_kind, exit_kind):
    module, clock = profiler_module
    error = ForwardFailure('sentinel failure')
    stage = RestorableStage(clock, error if exit_kind == 'forward_exception' else None)
    if attribute_kind == 'instance_bound':
        stage.forward = stage.forward
    elif attribute_kind == 'function':
        stage.forward = lambda value, *, token: stage.implementation(value, token=token)
    elif attribute_kind == 'callable':
        stage.forward = stage.implementation
    original = _remember(stage)
    class_forward = RestorableStage.forward
    profiler = module.HostStageProfiler(SimpleNamespace(blocks=[SimpleNamespace(norm1=stage)]))
    token = object()

    def exercise():
        with profiler:
            assert vars(stage)['forward'] is not original[1]
            assert RestorableStage.forward is class_forward
            assert stage.forward(7, token=token) == (7, token)
            if exit_kind == 'body_exception':
                raise error

    if exit_kind == 'normal':
        exercise()
    else:
        with pytest.raises(ForwardFailure) as caught:
            exercise()
        assert caught.value is error
    _assert_restored(stage, original)
    assert RestorableStage.forward is class_forward
    _assert_stats(profiler.snapshot()['phases']['unassigned']['norm1'], calls=1,
                  total_ms=1, max_ms=1)
    # Restoration must leave the same profiler usable, even after exceptional exit.
    with profiler:
        pass
    _assert_restored(stage, original)


def test_nested_failure_records_inclusive_parent_and_child_and_restores(profiler_module):
    module, clock = profiler_module
    model = FakeModel(clock, blocks=1)
    moe = model.blocks[0].moe
    error = ForwardFailure('expert failure')
    moe.experts.forward = CallableForward(clock, error)
    components = (model.blocks[0].norm1, model.blocks[0].mixer, model.blocks[0].norm2,
                  moe, moe.router, moe.experts)
    originals = [(component, _remember(component)) for component in components]
    profiler = module.HostStageProfiler(model)
    profiler.phase = 'decode'
    with pytest.raises(ForwardFailure) as caught:
        with profiler:
            model.forward(0, token=None)
    assert caught.value is error
    stats = profiler.snapshot()['phases']['decode']
    _assert_stats(stats['router'], calls=1, total_ms=3, max_ms=3)
    _assert_stats(stats['experts'], calls=1, total_ms=1, max_ms=1)
    _assert_stats(stats['moe'], calls=1, total_ms=11, max_ms=11)
    for component, original in originals:
        _assert_restored(component, original)


class SlottedForward:
    __slots__ = ()

    def forward(self, value):
        return value


class ReadOnlyForward:
    @property
    def forward(self):
        return self._forward

    def _forward(self, value):
        return value


class TypeErrorForward:
    def __setattr__(self, name, value):
        if name == 'forward':
            raise TypeError('forward is read-only')
        super().__setattr__(name, value)

    def forward(self, value):
        return value


@pytest.mark.parametrize('kind', ['absent', 'noncallable', 'slots', 'property', 'typeerror'])
def test_unwritable_absent_and_noncallable_forwards_are_skipped(profiler_module, kind):
    module, clock = profiler_module
    skipped = {
        'absent': SimpleNamespace(),
        'noncallable': SimpleNamespace(forward=None),
        'slots': SlottedForward(),
        'property': ReadOnlyForward(),
        'typeerror': TypeErrorForward(),
    }[kind]
    attributes = dict(vars(skipped)) if hasattr(skipped, '__dict__') else None
    writable = FakeStage(clock, 2_000_000)
    block = SimpleNamespace(norm1=skipped, mixer=writable,
                            moe=SimpleNamespace(router=skipped, experts=writable))
    profiler = module.HostStageProfiler(SimpleNamespace(blocks=[block]))
    with profiler:
        if callable(getattr(skipped, 'forward', None)):
            assert skipped.forward(9) == 9
        assert writable.forward(0, token=None) == 1
        assert set(profiler.snapshot()['phases']['unassigned']) == {'mixer'}
    if attributes is not None:
        assert vars(skipped) == attributes
    assert 'forward' not in vars(writable)


@pytest.mark.parametrize('model', [SimpleNamespace(), SimpleNamespace(blocks=None),
                                  SimpleNamespace(blocks=[]),
                                  SimpleNamespace(blocks=[None, SimpleNamespace()])])
def test_missing_or_empty_blocks_are_harmless(profiler_module, model):
    module, _ = profiler_module
    profiler = module.HostStageProfiler(model)
    with profiler:
        assert profiler.snapshot()['phases'] == {}
    assert profiler.snapshot()['phases'] == {}


def test_partial_enter_failure_restores_already_instrumented_components(profiler_module):
    module, clock = profiler_module
    stage = FakeStage(clock, 1_000_000)
    original = _remember(stage)
    error = ForwardFailure('block iteration failed')

    def blocks():
        yield SimpleNamespace(norm1=stage)
        raise error

    model = SimpleNamespace(blocks=blocks())
    profiler = module.HostStageProfiler(model)
    with pytest.raises(ForwardFailure) as caught:
        with profiler:
            pytest.fail('entry must fail before executing the body')
    assert caught.value is error
    _assert_restored(stage, original)
    model.blocks = [SimpleNamespace(norm1=stage)]
    with profiler:
        stage.forward(0, token=None)
    _assert_restored(stage, original)
    assert profiler.snapshot()['phases']['unassigned']['norm1']['calls'] == 1


def test_double_enter_raises_without_replacing_or_removing_active_wrapper(profiler_module):
    module, clock = profiler_module
    stage = FakeStage(clock, 1_000_000)
    original = _remember(stage)
    profiler = module.HostStageProfiler(SimpleNamespace(blocks=[SimpleNamespace(norm1=stage)]))
    with profiler:
        wrapper = stage.forward
        with pytest.raises(RuntimeError, match='serial, exclusive use'):
            profiler.__enter__()
        assert stage.forward is wrapper
        stage.forward(0, token=None)
    _assert_restored(stage, original)
    _assert_stats(profiler.snapshot()['phases']['unassigned']['norm1'], calls=1,
                  total_ms=1, max_ms=1)


def test_direct_importlib_load_does_not_import_package_or_initialize_backend():
    script = textwrap.dedent('''
        import importlib.abc
        import importlib.util
        import sys

        forbidden = ('mini_llm', 'numpy', 'cupy', 'torch', 'triton')

        class NoBackendImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in forbidden:
                    raise AssertionError('unexpected backend/package import: ' + fullname)
                return None

        assert not any(name.split('.')[0] in forbidden for name in sys.modules)
        sys.meta_path.insert(0, NoBackendImports())
        spec = importlib.util.spec_from_file_location('_standalone_generation_profile', sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        profiler = module.HostStageProfiler(object())
        with profiler:
            assert profiler.snapshot()['phases'] == {}
        assert not any(name.split('.')[0] in forbidden for name in sys.modules)
    ''')
    result = subprocess.run(
        [sys.executable, '-I', '-B', '-c', script, str(SOURCE)],
        cwd=ROOT, env=dict(os.environ, MINI_LLM_BACKEND='numpy'),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
