"""Bounded, generation-only Qwen3-Next workload/profile; no reference replay.

Runs serial raw-completion requests through existing GenerationSession. The
first request includes lazy-upload/JIT/cache warmup; subsequent requests reuse
weights/cache but receive fresh model state. Host memory guards are soft checks,
not OS limits. No generated code is executed and no chat template is guessed.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MiB, GiB = 1024**2, 1024**3


def load_observer(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latency_summary(values_ms):
    values = sorted(float(v) for v in values_ms)
    if not values:
        return dict(count=0, mean_ms=None, p50_ms=None, p95_ms=None, max_ms=None, calls_per_s=None)
    def percentile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    total = sum(values)
    return dict(count=len(values), mean_ms=total / len(values), p50_ms=percentile(.5),
                p95_ms=percentile(.95), max_ms=values[-1],
                calls_per_s=len(values) * 1000 / total if total else None)


class ObservedAdapter:
    """Observe existing adapter calls without changing sampling or expert dispatch."""

    def __init__(self, serving, executor, trace_factory, guard, stages, pool=None,
                 prefill_chunk_tokens=32):
        self.serving, self.executor = serving, executor
        self.trace_factory, self.guard, self.stages, self.pool = trace_factory, guard, stages, pool
        self.context_length, self.stop_ids = serving.context_length, serving.stop_ids
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.phase = None
        self.traces, self.latencies = {}, {'prefill': [], 'decode': []}
        self.observations = []
        self.memory_check_wall_ms = 0.0
        self.request_start = 0.0
        self.first_sample_ms = None
        self.sample_ms = []

    def observe_memory(self, label):
        start = time.perf_counter()
        try:
            snapshot = self.guard.check()
        finally:
            self.memory_check_wall_ms += (time.perf_counter() - start) * 1000
        value = dict(label=label, **snapshot)
        if self.pool is not None:
            value.update(pool_used_bytes=int(self.pool.used_bytes()),
                         pool_reserved_bytes=int(self.pool.total_bytes()))
        self.observations.append(value)
        return value

    def encode(self, prompt):
        return self.serving.encode(prompt)

    def decode(self, ids):
        return self.serving.decode(ids)

    def create_state(self, total_tokens):
        self.observe_memory('before_state')
        result = self.serving.create_state(total_tokens)
        self.observe_memory('after_state')
        return result

    def _forward(self, phase, function):
        from mini_llm.backend import BACKEND_NAME, xp
        if phase != self.phase:
            self.phase = phase
            self.traces[phase] = self.trace_factory(phase)
            self.executor.routing_profiler = self.traces[phase]
        self.stages.phase = phase
        self.observe_memory('before_' + phase)
        if BACKEND_NAME == 'cupy':
            xp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        result = function()
        if BACKEND_NAME == 'cupy':
            xp.cuda.get_current_stream().synchronize()
        self.latencies[phase].append((time.perf_counter() - start) * 1000)
        self.observe_memory('after_' + phase)
        return result

    def prefill(self, ids, state):
        result = None
        for begin in range(0, len(ids), self.prefill_chunk_tokens):
            chunk = ids[begin:begin + self.prefill_chunk_tokens]
            result = self._forward('prefill', lambda: self.serving.prefill(chunk, state))
        return result

    def decode_one(self, token_id, state):
        return self._forward('decode', lambda: self.serving.decode_one(token_id, state))

    def sampler(self, logits, **kwargs):
        from mini_llm.generation import sample_token
        start = time.perf_counter()
        token = sample_token(logits, **kwargs)
        self.sample_ms.append((time.perf_counter() - start) * 1000)
        if self.first_sample_ms is None:
            self.first_sample_ms = (time.perf_counter() - self.request_start) * 1000
        return token


def run_request(model, tokenizer, executor, args, request_id, guard, pool, stages):
    from mini_llm.diagnostics.routing_trace import RoutingTrace
    from mini_llm.generation import GenerationSession
    from mini_llm.serving_models import ServingModel, _external_stops
    from mini_llm.backend import BACKEND_NAME, xp

    serving = ServingModel(model, tokenizer, args.context, stop_ids=_external_stops(tokenizer))
    def trace_factory(phase):
        return RoutingTrace(model_id=executor.model_id, quantization='GGUF-mixed',
            expert_mapping='identity', workload=f'0106r/request-{request_id}/{phase}',
            mode=args.profile, max_records=args.trace_records)
    adapter = ObservedAdapter(serving, executor, trace_factory, guard, stages, pool,
                              args.prefill_chunk_tokens)
    session = GenerationSession(adapter, sampler=adapter.sampler)
    before = executor.statistics()
    stages.clear()
    adapter.request_start = time.perf_counter()
    first_text_ms, pieces, done = None, [], None
    for event in session.stream(args.prompt, max_tokens=args.max_new_tokens,
        temperature=0, stop=args.stop, emit_text=True, log_speed=False):
        if event['kind'] == 'content':
            if first_text_ms is None:
                first_text_ms = (time.perf_counter() - adapter.request_start) * 1000
            pieces.append(event['text'])
            if args.stream:
                print(event['text'], end='', flush=True)
        elif event['kind'] == 'done':
            done = event
    wall_ms = (time.perf_counter() - adapter.request_start) * 1000
    if BACKEND_NAME == 'cupy':
        xp.cuda.get_current_stream().synchronize()
    if done is None:
        raise AssertionError('GenerationSession did not finish its request')
    if not all(layer.length == session.state.length for layer in session.state.layer_states):
        raise AssertionError('Hybrid layer state positions disagree')
    expected_length = session.prompt_tokens + len(adapter.latencies['decode'])
    if session.state.length != expected_length:
        raise AssertionError('Generation traces/state length do not match actual forwarded tokens')
    traces = {phase: trace.snapshot() for phase, trace in adapter.traces.items()}
    phase_tokens = {'prefill': session.prompt_tokens, 'decode': len(adapter.latencies['decode'])}
    for phase, trace in traces.items():
        expected = {str(i): phase_tokens[phase] for i in range(model.config.num_hidden_layers)}
        if trace['tokens_by_layer'] != expected:
            raise AssertionError(f'{phase}: trace includes missing or replayed model inputs')
    prefill_ms = sum(adapter.latencies['prefill'])
    result = dict(request_id=request_id, label='first_use' if request_id == 0 else 'warm',
        prompt_tokens=session.prompt_tokens, completion_tokens=len(session.generated_ids),
        generated_token_ids=list(session.generated_ids), text=''.join(pieces), finish_reason=done['reason'],
        forwarded_decode_tokens=len(adapter.latencies['decode']), state_length=session.state.length,
        timing=dict(request_wall_ms=wall_ms, first_sampled_token_ms=adapter.first_sample_ms,
            first_visible_text_ms=first_text_ms, prefill_forward_ms=prefill_ms,
            prefill_tokens_per_s=session.prompt_tokens * 1000 / prefill_ms if prefill_ms else None,
            decode=latency_summary(adapter.latencies['decode']), decode_forward_ms=adapter.latencies['decode'],
            prefill_chunk_ms=adapter.latencies['prefill'], sample_ms=adapter.sample_ms,
            completion_tokens_per_s_end_to_end=len(session.generated_ids) * 1000 / wall_ms if wall_ms else None,
            memory_observation_ms=adapter.memory_check_wall_ms),
        stages=stages.snapshot(), routing=traces, executor_before=before,
        executor_after=executor.statistics(), memory=adapter.observations)
    session.state = None
    gc.collect()  # Outside request timing: release KV/recurrent arrays, retain weights/expert cache.
    adapter.observe_memory('after_state_release')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--prompt')
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--requests', type=int, default=2)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--prefill-chunk-tokens', type=int, default=32)
    parser.add_argument('--cpu-workers', type=int, default=2)
    parser.add_argument('--expert-cache-mib', type=int, default=256)
    parser.add_argument('--promote-after', type=int, default=1)
    parser.add_argument('--device-budget-mib', type=int, default=4096)
    parser.add_argument('--device-reserve-mib', type=int, default=2048)
    parser.add_argument('--ram-reserve-mib', type=int, default=16384)
    parser.add_argument('--max-rss-mib', type=int, default=12288)
    parser.add_argument('--profile', choices=('summary', 'detailed'), default='summary')
    parser.add_argument('--trace-records', type=int, default=4096)
    parser.add_argument('--stop', action='append', default=[])
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not 1 <= args.max_new_tokens <= 256 or not 1 <= args.requests <= 4
            or not 2 <= args.context <= 512 or not 1 <= args.prefill_chunk_tokens <= 128
            or not 0 <= args.cpu_workers <= 4 or not 1 <= args.promote_after
            or args.trace_records < 1 or args.expert_cache_mib < 0 or args.ram_reserve_mib < 0
            or args.max_rss_mib < 1 or args.device_budget_mib < 1 or args.device_reserve_mib < 0):
        parser.error('Invalid bounded generation/resource options')
    if args.prompt is not None and args.prompt_file:
        parser.error('Choose --prompt or --prompt-file, not both')
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Refusing to overwrite existing report')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(patch='0106R', status='starting', requests=[], failed=False,
        scope='Serial raw generation only; no reference process, validation replay, chat framing or speculative decoding',
        configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        timing_semantics='Forward host wall time includes current-stream completion; stage times are inclusive, nested and not additive; first_use is not disk-cold')
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    host = load_observer('host_memory_0106r', 'mini_llm/diagnostics/host_memory.py')
    guard = host.HostMemoryGuard(args.ram_reserve_mib * MiB, args.max_rss_mib * MiB)
    model = executor = pool = None
    stage = 'host_preflight'
    try:
        report['host_preflight'] = guard.check(loading=True)
        if args.prompt_file:
            if args.prompt_file.stat().st_size > 65536:
                raise ValueError('Prompt file exceeds 64 KiB limit')
            args.prompt = args.prompt_file.read_text(encoding='utf-8')
        if args.prompt is None:
            args.prompt = 'ab a\nab' if args.synthetic else 'def add(a, b):\n    return'
        if len(args.prompt.encode('utf-8')) > 65536:
            raise ValueError('Prompt exceeds 64 KiB limit')
        stage = 'index_and_tokenizer'
        from mini_llm.backend import BACKEND_NAME, xp
        from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
        from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel
        from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
        from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
        if args.synthetic:
            from benchmarks.fixture_0106q_qwen_reference import write_fixture
            fixture = write_fixture(args.output.with_suffix('') / 'fixture')
            args.model = fixture['model_path']
        index = Qwen3NextGGUFIndex.open(args.model)
        vocab = int(index.tensors['token_embd.weight'].shape[1])
        tokenizer = load_gguf_tokenizer(index.reader, vocab_size=vocab)
        ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
        if not ids or len(ids) + args.max_new_tokens > args.context:
            raise ValueError('Prompt plus output token budget must fit explicit context')
        report.update(backend=BACKEND_NAME, model_path=str(args.model.resolve()),
            model_stat=dict(bytes=args.model.stat().st_size, mtime_ns=args.model.stat().st_mtime_ns),
            prompt=args.prompt, prompt_token_ids=ids, prompt_sha256=hashlib.sha256(args.prompt.encode()).hexdigest(),
            host_after_index=guard.check(), reference_or_full_model_hash_run=False)
        report['configuration'].update(prompt=args.prompt, model=str(args.model))
        report['implementation_revision'] = subprocess.check_output(
            ['git', '--no-pager', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if BACKEND_NAME == 'cupy':
            pool = xp.cuda.MemoryPool()
            pool.set_limit(size=args.device_budget_mib * MiB)
            allocator = xp.cuda.using_allocator(pool.malloc)
        else:
            allocator = nullcontext()
        stage = 'model_load'
        with allocator:
            executor = HeterogeneousExpertExecutor(index.provider, model_id=str(args.model.resolve()),
                d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
                device_budget_bytes=args.expert_cache_mib * MiB, cpu_workers=args.cpu_workers,
                promote_after=args.promote_after, cpu_packed_backend='native')
            begin = time.perf_counter()
            model = Qwen3NextGGUFModel.from_index(index, expert_executor=executor,
                device_memory_budget_bytes=args.device_budget_mib * MiB,
                device_reserve_bytes=args.device_reserve_mib * MiB)
            if BACKEND_NAME == 'cupy':
                xp.cuda.get_current_stream().synchronize()
            report['load_wall_ms'] = (time.perf_counter() - begin) * 1000
            report['memory_plan'] = model.memory_plan(1, args.context)
            report['host_after_load'] = guard.check()
            observer = load_observer('generation_profile_0106r', 'mini_llm/diagnostics/generation_profile.py')
            with observer.HostStageProfiler(model) as stages:
                for request_id in range(args.requests):
                    stage = f'generation_request_{request_id}'
                    result = run_request(model, tokenizer, executor, args, request_id, guard, pool, stages)
                    report['requests'].append(result)
                    save()
                    print(json.dumps(dict(request_id=request_id, completion_tokens=result['completion_tokens'],
                        finish_reason=result['finish_reason'], decode=result['timing']['decode']), allow_nan=False),
                        flush=True)
            report['status'] = 'generation_completed'
            report['warm_request_ids_equal_first'] = all(
                value['generated_token_ids'] == report['requests'][0]['generated_token_ids']
                for value in report['requests'][1:])
            report['host_final'] = guard.check()
            if pool is not None:
                report.update(pool_reserved_bytes=int(pool.total_bytes()), pool_used_bytes=int(pool.used_bytes()),
                              pool_limit_bytes=int(pool.get_limit()))
                graphs = [dict(layer_id=block.layer_id, **block.mixer._decode_graph.statistics())
                          for block in model.blocks if getattr(block.mixer, '_decode_graph', None) is not None]
                report['recurrent_graphs'] = graphs
                report['graph_pool_reserved_bytes'] = sum(g['pool_reserved_bytes'] for g in graphs)
                report['combined_private_pools_reserved_bytes'] = (
                    report['pool_reserved_bytes'] + report['graph_pool_reserved_bytes'])
    except Exception as exc:
        report.update(status='host_memory_guard_stopped' if isinstance(exc, host.MemoryBudgetError) else 'failed',
                      failed=True, failure=dict(stage=stage, type=type(exc).__name__, message=str(exc)),
                      last_host_observation=guard.last_snapshot)
        save()
        raise
    finally:
        if model is not None:
            model.close()
        if executor is not None:
            executor.close()
    save()
    return report


if __name__ == '__main__':
    main()
