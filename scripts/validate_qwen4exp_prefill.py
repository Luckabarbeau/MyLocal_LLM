#!/usr/bin/env python3
"""Bounded real-checkpoint prefill correctness validator, NOT a benchmark.

Defaults: production BEST_ENV/options, context 262144, host QSA history,
32 prompt tokens, chunk sizes 16,64,256,1024, and three fixed decode tokens.
One model and at most ONE live generation state. CPU artifacts retain only
populated QSA rows/completed pools, recurrent/PLE state, logits and routes.

Route adapters enqueue observation-only GPU copies, including during captured
replay. Warmup/capture may overwrite the same token slot; the real replay writes
it last. Step positions are device inputs, not constants baked into a graph.
Chunk routers are uncaptured and use per-layer token offsets. Coverage is
mandatory: missed graph observations cannot silently count as matching IDs.
Instrumentation changes timing, not the router's inputs/outputs or arithmetic.

Float checks: abs(actual-reference) <= atol + rtol*abs(reference), default
atol=rtol=1e-4, FP32 storage. Report actual errors and worst tolerance ratio;
NEVER widen tolerances on failure. Integer counters and route IDs are exact.
This is a short populated-prefix test at large reserved capacity, NOT evidence
of long-context correctness or 1024-row execution when prompt length is 32.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Importing the launcher does not load the backend, checkpoint or HTTP server.
from scripts.serve_qwen4exp import BEST_ENV, MODEL, GiB, byte_budget


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--qsa-state-storage', choices=('host',), default='host')
    p.add_argument('--host-state-budget-gib', type=float, default=24)
    p.add_argument('--cache-gib', default='6')
    p.add_argument('--cpu-workers', type=int, default=19)
    p.add_argument('--prompt-length', type=int, choices=(16, 32, 128, 1024), default=32)
    p.add_argument('--chunk-sizes', default='16,64,256,1024')
    p.add_argument('--decode-tokens', default='1,2,3', help='Fixed IDs, not argmax sampling; 1 to 8 tokens')
    p.add_argument('--sequential-repeats', type=int, default=1, help='Repeat the sequential control before chunk runs (0 to 2)')
    p.add_argument('--trace-layers', type=int, default=0, help='Observe forward dependencies in the first N layers (0 to 4)')
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--output', type=Path, required=True, help='JSON report; sibling *_arrays directory holds NPZ snapshots')
    return p


def integer_list(text):
    return [int(v.strip()) for v in text.split(',')]


def production_options(args, chunks):
    # Same options as serve_qwen4exp.main, with the largest requested chunk's
    # activation allowance. Host page prefetch is omitted: it changes I/O only.
    cache = byte_budget(args.cache_gib, '--cache-gib')
    return dict(max_length=args.context,
        activation_bytes=max(64 * 1024**2, max(chunks) * 2560 * 4 * 64),
        cpu_workers=args.cpu_workers, static_device_budget_bytes=6 * GiB,
        device_budget_bytes=14 * GiB, cache_bytes=cache,
        expert_promote_after=1, group_after_admission=True,
        expert_cache_slots=cache != 0, qsa_fast_decode=True, hc_fast_decode=True,
        router_fast_decode=True, gdn_fast_decode=True, hc_graph_decode=False,
        hc_native_read=True, elementwise_fast_decode=True, expert_victim_policy='heap',
        qsa_state_storage=args.qsa_state_storage,
        host_budget_bytes=int(args.host_state_budget_gib * GiB), host_expert_bytes=0,
        expert_cpu_backend='ggml_pool', group_mixed_decode=False, overlap_shared=True,
        gdn_graph_decode=False, device_expert_plan=True, pre_moe_graph=True,
        shared_graph_decode=True, defer_shared=True)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def placement(executor):
    entries = []
    for key, (expert, size) in executor.cache.resident.items():
        pointers = [int(w.data.data.ptr) for w in (expert.gate, expert.up, expert.down)]
        entries.append([key.layer_id, key.expert_id, size, pointers])
    return dict(admission_limit=executor.cache._admission_limit,
                promotions=int(executor.cache.counters['promotions']),
                evictions=int(executor.cache.counters['evictions']),
                residents=sorted(entries))


def retire(model, xp):
    xp.cuda.get_current_stream().synchronize()
    gc.collect()
    if model._states:
        raise RuntimeError('State still live: refusing a second aggregate allocation')
    model._retire_orphaned_state_graphs()
    # Also retire unbound request captures: they may hold trace buffer pointers.
    # Keep wrapper/plan identities; subsequent calls recapture normally.
    owners = list(getattr(model, '_pre_moe_segments', {}).values())
    owners += list(getattr(model, '_qsa_block_segments', {}).values())
    visited = set()
    while owners:
        owner = owners.pop()
        if id(owner) in visited:
            continue
        visited.add(id(owner))
        owners.extend(getattr(owner, 'members', ()))
        graph = getattr(owner, 'graph', None)
        if graph is not None:
            graph.close()
            owner.graph = None
    token_graph = getattr(model, '_token_graph', None)
    if token_graph is not None:
        token_graph.close()
        model._token_graph = None
    xp.cuda.get_current_stream().synchronize()
    xp.get_default_memory_pool().free_all_blocks()


class RouteTrace:
    def __init__(self, model, total, xp, asnumpy, trace_layers=0):
        self.model, self.total, self.xp, self.asnumpy = model, total, xp, asnumpy
        self.position = xp.zeros(1, dtype=xp.int64)
        self.ids, self.weights, self.seen, self.offsets = {}, {}, {}, {}
        self.mode = 'step'
        self.trace_layers = trace_layers
        self.forward_arrays, self.forward_seen, self.forward_offsets = {}, {}, {}
        for i, block in enumerate(model.blocks):
            k = block.moe.router.k
            self.ids[i] = xp.full((total, k), -1, dtype=xp.int64)
            self.weights[i] = xp.zeros((total, k), dtype=xp.float32)
            self.seen[i] = xp.zeros(total, dtype=xp.int32)
            self.offsets[i] = 0

    def step(self, position):
        self.mode = 'step'
        self.position.fill(position)

    def chunk(self):
        self.mode = 'chunk'

    def observe(self, name, value, tail):
        rows = value.reshape((-1,) + value.shape[-tail:])
        count = rows.shape[0]
        if name not in self.forward_arrays:
            if self.xp.cuda.get_current_stream().is_capturing():
                raise RuntimeError(f'Forward trace {name} was not initialized in capture warmup')
            self.forward_arrays[name] = self.xp.zeros((self.total,) + rows.shape[1:], dtype=value.dtype)
            self.forward_seen[name] = self.xp.zeros(self.total, dtype=self.xp.int32)
            self.forward_offsets[name] = 0
        if self.mode == 'step':
            if count != 1:
                raise RuntimeError(f'Unexpected trace row count in step: {name}')
            indices = self.position
        else:
            start = self.forward_offsets[name]
            if start + count > self.total:
                raise RuntimeError(f'Forward observation overflow: {name}')
            indices = self.xp.arange(start, start + count, dtype=self.xp.int64)
            self.forward_offsets[name] += count
        self.forward_arrays[name][indices] = rows
        self.forward_seen[name][indices] = 1

    @contextmanager
    def forward_installed(self):
        from mini_llm.external import qwen4exp_prefill as prefill
        saved = []

        def replace(owner, name, function):
            own = owner.__dict__.get(name)
            saved.append((owner, name, own))
            setattr(owner, name, function)

        def read_observation(layer, kind, streams, result):
            name = f'forward.{layer}.{kind}'
            self.observe(name + '.input', streams, 2)
            self.observe(name + '.hidden', result[0], 1)
            if result[1] is not None:
                self.observe(name + '.gains', result[1], 1)

        connections = {}
        try:
            for i, block in enumerate(self.model.blocks[:self.trace_layers]):
                for kind, hc in (('attention', block.attention_hc), ('ffn', block.ffn_hc)):
                    connections[id(hc)] = i, kind
                    original_read, original_write = hc.read, hc.write

                    def read(streams, original=original_read, layer=i, kind=kind):
                        result = original(streams)
                        read_observation(layer, kind, streams, result)
                        return result

                    def write(streams, output, gains, original=original_write, layer=i, kind=kind):
                        self.observe(f'forward.{layer}.{kind}.block_output', output, 1)
                        result = original(streams, output, gains)
                        self.observe(f'forward.{layer}.{kind}.residual', result, 2)
                        return result

                    replace(hc, 'read', read)
                    replace(hc, 'write', write)
                if block.ple is not None:
                    original_ple = block.ple.step

                    def ple(token, streams, state, original=original_ple, layer=i):
                        self.observe(f'forward.{layer}.ple.input', streams, 2)
                        result = original(token, streams, state)
                        self.observe(f'forward.{layer}.ple.output', result[0], 2)
                        return result

                    replace(block.ple, 'step', ple)
            original_chunk_read = prefill._hc_read

            def chunk_read(hc, streams):
                result = original_chunk_read(hc, streams)
                if (id(hc) in connections and getattr(hc, 'packed_bf16', False)
                        and self.xp.__name__ != 'cupy'):
                    layer, kind = connections[id(hc)]
                    read_observation(layer, kind, streams, result)
                return result

            replace(prefill, '_hc_read', chunk_read)
            yield
        finally:
            for owner, name, own in reversed(saved):
                if own is None:
                    delattr(owner, name)
                else:
                    setattr(owner, name, own)

    @contextmanager
    def installed(self):
        saved = []
        try:
            for i, block in enumerate(self.model.blocks):
                router = block.moe.router
                own = router.__dict__.get('forward')
                original = router.forward

                def traced(hidden, original=original, layer=i):
                    weights, ids = original(hidden)
                    rows = int(ids.size // ids.shape[-1])
                    if self.mode == 'step':
                        if rows != 1:
                            raise RuntimeError('Unexpected batched router in scalar step trace')
                        indices = self.position
                    else:
                        start = self.offsets[layer]
                        if start + rows > self.total:
                            raise RuntimeError('Chunk route trace exceeded bounded observation buffer')
                        indices = self.xp.arange(start, start + rows, dtype=self.xp.int64)
                        self.offsets[layer] += rows
                    if layer < self.trace_layers:
                        self.observe(f'forward.{layer}.router.input', hidden, 1)
                    self.ids[layer][indices] = ids.reshape(rows, -1)
                    self.weights[layer][indices] = weights.reshape(rows, -1)
                    self.seen[layer][indices] = 1
                    return weights, ids

                saved.append((router, own))
                router.forward = traced
            yield self
        finally:
            for router, own in reversed(saved):
                if own is None:
                    del router.forward
                else:
                    router.forward = own

    def snapshot(self):
        result = {}
        for i in self.ids:
            seen = self.asnumpy(self.seen[i]).copy()
            if not bool((seen == 1).all()):
                missing = [j for j, value in enumerate(seen) if value != 1]
                raise RuntimeError(f'Incomplete route coverage in layer {i}: positions {missing}')
            result[f'route.{i}.ids'] = self.asnumpy(self.ids[i]).copy()
            result[f'route.{i}.weights'] = self.asnumpy(self.weights[i]).copy()
        for name, value in self.forward_arrays.items():
            seen = self.asnumpy(self.forward_seen[name])
            if not bool((seen == 1).all()):
                raise RuntimeError(f'Incomplete forward observation: {name}')
            result[name] = self.asnumpy(value).copy()
        return result


def state_snapshot(model, state, prefix, asnumpy):
    import numpy as np
    result = {f'{prefix}.length': np.asarray(state.length, dtype=np.int64),
              f'{prefix}.capacity': np.asarray(state.max_length, dtype=np.int64)}

    def put(name, value):
        result[f'{prefix}.{name}'] = asnumpy(value).copy()

    for i, s in state.recurrent.items():
        put(f'recurrent.{i}.conv', s.conv.history)
        put(f'recurrent.{i}.memory', s.delta.memory)
        put(f'recurrent.{i}.lengths', np.asarray([s.conv.length, s.delta.length], dtype=np.int64))
    for i, s in state.qsa.items():
        completed = s.length // model.blocks[i].mixer.config.block_size
        for name in ('keys', 'values', 'raw_index'):
            put(f'qsa.{i}.{name}', getattr(s, name)[:s.length])
        put(f'qsa.{i}.pooled', s.pooled[:completed])
        put(f'qsa.{i}.length', np.asarray(s.length, dtype=np.int64))
    for i, s in state.ple.items():
        put(f'ple.{i}.predecessors', np.asarray(s.predecessors, dtype=np.int64))
        put(f'ple.{i}.history', s.normalized_history)
    return result


def run_case(model, tokens, decode, size, context, xp, asnumpy, trace_layers=0):
    import numpy as np
    from mini_llm.external.qwen4exp_prefill import prefill_chunked
    if model._states:
        raise RuntimeError('Previous request still live before validation run')
    trace = RouteTrace(model, len(tokens) + len(decode), xp, asnumpy, trace_layers)
    arrays = {'tokens': np.asarray(tokens, dtype=np.int64),
              'decode_tokens': np.asarray(decode, dtype=np.int64)}
    state = None
    with trace.installed(), trace.forward_installed():
        try:
            state = model.create_generation_state(max_length=context)
            if size == 0:
                for t, token in enumerate(tokens):
                    trace.step(t)
                    logits = model._step(int(token), state, project_logits=t == len(tokens) - 1)
            else:
                trace.chunk()
                logits = prefill_chunked(model, tokens, state, size)
            arrays['prefill.logits'] = asnumpy(logits).copy()
            arrays.update(state_snapshot(model, state, 'prefill', asnumpy))
            for j, token in enumerate(decode):
                trace.step(len(tokens) + j)
                logits = model.decode_one([token], state)
                arrays[f'decode.{j}.logits'] = asnumpy(logits).copy()
            model._validate(state)
            model.expert_executor.check_native_service()
            arrays.update(state_snapshot(model, state, 'decoded', asnumpy))
            arrays.update(trace.snapshot())
        finally:
            # Drop strong request ownership BEFORE admitting another full context.
            state = None
            retire(model, xp)
    return arrays


def compare_array(reference, actual, atol, rtol):
    import numpy as np
    result = dict(shape=list(reference.shape), dtype=str(reference.dtype))
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return dict(result, passed=False, reason='shape/dtype mismatch')
    if reference.dtype.kind in 'biu':
        mismatch = reference != actual
        return dict(result, passed=not bool(mismatch.any()), exact=True,
                    mismatches=int(np.count_nonzero(mismatch)))
    finite = np.isfinite(reference) & np.isfinite(actual)
    if not bool(finite.all()):
        return dict(result, passed=False, reason='nonfinite values',
                    nonfinite=int(np.count_nonzero(~finite)))
    if reference.size == 0:
        return dict(result, passed=True, max_abs=0.0, max_tolerance_ratio=0.0)
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = np.abs(a - b)
    tolerance = atol + rtol * np.abs(b)
    ratio = np.divide(error, tolerance, out=np.full_like(error, np.inf), where=tolerance != 0)
    ratio[(error == 0) & (tolerance == 0)] = 0
    largest = np.unravel_index(int(error.argmax()), error.shape)
    worst = np.unravel_index(int(ratio.argmax()), error.shape)
    return dict(result, passed=bool(np.all(error <= tolerance)), max_abs=float(error[largest]),
                relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(b), 1e-30)),
                max_tolerance_ratio=float(ratio[worst]),
                largest_error_index=[int(i) for i in largest],
                reference_at_largest=float(b[largest]), actual_at_largest=float(a[largest]),
                worst_tolerance_index=[int(i) for i in worst],
                reference_at_worst=float(b[worst]), actual_at_worst=float(a[worst]))


def compare(reference, actual, atol, rtol):
    if reference.keys() != actual.keys():
        return dict(passed=False, reason='Snapshot keys differ',
                    missing=sorted(reference.keys() - actual.keys()),
                    unexpected=sorted(actual.keys() - reference.keys()))
    metrics = {name: compare_array(value, actual[name], atol, rtol)
               for name, value in reference.items()}
    numeric = [(name, m) for name, m in metrics.items() if 'max_abs' in m]
    largest = max(numeric, key=lambda pair: pair[1]['max_abs'], default=None)
    worst = max(numeric, key=lambda pair: pair[1]['max_tolerance_ratio'], default=None)
    return dict(passed=all(m['passed'] for m in metrics.values()), arrays=metrics,
                largest_error=None if largest is None else dict(name=largest[0], **largest[1]),
                worst_tolerance=None if worst is None else dict(name=worst[0], **worst[1]),
                failed_arrays=[name for name, m in metrics.items() if not m['passed']])


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        chunks, decode = integer_list(args.chunk_sizes), integer_list(args.decode_tokens)
        if not 0 <= args.sequential_repeats <= 2 or not 0 <= args.trace_layers <= 4:
            raise ValueError('sequential-repeats must be 0..2 and trace-layers 0..4')
        if not chunks or len(chunks) > 8 or len(set(chunks)) != len(chunks) or any(not 1 <= n <= 1024 for n in chunks):
            raise ValueError('chunk sizes must be 1 to 8 distinct integers in [1,1024]')
        if not 1 <= len(decode) <= 8 or any(t < 0 for t in decode):
            raise ValueError('decode-tokens requires 1 to 8 nonnegative IDs')
        if not args.prompt_length + len(decode) <= args.context <= 262144:
            raise ValueError('context must fit prompt+decode and be <=262144')
        if args.cpu_workers < 1 or not math.isfinite(args.host_state_budget_gib) or args.host_state_budget_gib <= 0:
            raise ValueError('workers and finite host-state budget must be positive')
        if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)) or args.atol + args.rtol == 0:
            raise ValueError('tolerances must be finite, nonnegative, and not both zero')
        options = production_options(args, chunks)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        p.error(str(exc))
    for key, value in BEST_ENV.items():
        os.environ.setdefault(key, value)
    report = dict(status='running', model=args.model, options=options,
            prefill_experts_mode=os.environ.get('MINI_LLM_PREFILL_EXPERTS', 'decode'),
        environment={key: os.environ[key] for key in BEST_ENV},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        atol=args.atol, rtol=args.rtol, prompt_length=args.prompt_length,
        chunk_sizes=chunks, decode_tokens=decode, trace_layers=args.trace_layers,
        sequential_repeats=args.sequential_repeats,
        limitations=['Short populated prefix only; full host capacity is reserved.',
                      'Chunks larger than the prompt do not exercise their advertised row count.',
                      'Scoped captured route-copy instrumentation; timings are not benchmarks.',
                      'No expert page prefetch; production arithmetic/options otherwise retained.'], cases=[])
    artifacts = args.output.parent / (args.output.stem + '_arrays')
    write_report(args.output, report)
    try:
        import numpy as np
        from mini_llm.backend import BACKEND_NAME, xp, asnumpy
        from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
        from mini_llm.external.qwen4exp_chat import ChatSession
        from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('Real-checkpoint validation requires MINI_LLM_BACKEND=cupy')
        if os.environ.get('MINI_LLM_TOKEN_GRAPH') != '0':
            raise RuntimeError('Use production MINI_LLM_TOKEN_GRAPH=0 for complete scoped route tracing')
        artifacts.mkdir(parents=True, exist_ok=True)
        with Qwen4ExpGGUFModel.open(args.model, **options) as model:
            report['expert_arena'] = model.expert_executor.reserve_device_slots()
            tokenizer = load_gguf_tokenizer(model.weights.store.metadata, vocab_size=model.config.vocab_size)
            if any(t >= model.config.vocab_size for t in decode):
                raise ValueError('Fixed decode token exceeds checkpoint vocabulary')
            text = ('Numerical validation of causal chunked prefill preserves recurrent memory, '
                    'attention history, routing, and subsequent decoding. ' * 128)
            tokens = np.asarray(tokenizer.encode(text).ids, dtype=np.int64)[:args.prompt_length]
            if len(tokens) != args.prompt_length:
                raise ValueError('Tokenizer produced fewer than the requested prompt tokens')
            report['tokens'] = tokens.tolist()
            print('Warming production decode graphs and expert placement...', flush=True)
            session = ChatSession(model, tokenizer, args.context, prefill_chunk_size=0)
            for _ in session.generate('Reply with the single word: ready', 16):
                pass
            report['warmup'] = dict(session.last_stats)
            session.reset()
            del session
            retire(model, xp)
            model.expert_executor.cache.set_admission_budget(0)
            frozen = placement(model.expert_executor)
            report['frozen_placement'] = frozen
            write_report(args.output, report)
            reference = None
            for ordinal, size in enumerate([0] * (1 + args.sequential_repeats) + chunks):
                name = ('sequential' if ordinal == 0 else f'sequential_repeat_{ordinal}') if size == 0 else f'chunk_{size}'
                if placement(model.expert_executor) != frozen:
                    raise RuntimeError(f'Frozen residency changed before {name}')
                print(f'Validating {name}: context={args.context}, prompt={len(tokens)}...', flush=True)
                started = time.monotonic()
                # Streaming packed misses uses GPU rather than CPU activation
                # quantizers. Its reference must use that same explicit policy,
                # one token at a time, not claim mixed-CPU arithmetic equivalence.
                run_size = 1 if size == 0 and report['prefill_experts_mode'] in ('stream', 'strata') else size
                arrays = run_case(model, tokens, decode, run_size, args.context, xp, asnumpy, args.trace_layers)
                if placement(model.expert_executor) != frozen:
                    raise RuntimeError(f'Frozen residency changed during {name}')
                file = artifacts / f'{name}.npz'
                np.savez(file, **arrays)
                case = dict(name=name, chunk_size=size, max_executed_chunk_rows=min(size or 1, len(tokens)),
                            snapshot=str(file), observed_seconds=time.monotonic() - started)
                if reference is None:
                    reference = arrays
                    case['passed'] = True
                    case['role'] = ('singleton explicit GPU prefill policy reference, not mixed-CPU arithmetic equivalence'
                                                        if report['prefill_experts_mode'] in ('stream', 'strata') else
                                                        'sequential reference, not an independent checkpoint oracle')
                else:
                    case.update(compare(reference, arrays, args.atol, args.rtol))
                    del arrays
                report['cases'].append(case)
                write_report(args.output, report)
                if size == 0 and ordinal > 0 and not case['passed']:
                    raise RuntimeError('Sequential control is not repeatable; chunk comparison is not a valid next action')
                print(json.dumps({key: case.get(key) for key in ('name', 'passed', 'largest_error', 'failed_arrays')}), flush=True)
            report['status'] = 'passed' if all(c['passed'] for c in report['cases']) else 'failed'
            report['largest_error'] = max((dict(case=c['name'], **c['largest_error']) for c in report['cases']
                                           if c.get('largest_error') is not None),
                                          key=lambda m: m['max_abs'], default=None)
    except Exception as exc:
        report['status'] = 'error'
        report['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print(report['error']['traceback'], file=sys.stderr)
    write_report(args.output, report)
    print(json.dumps(dict(status=report['status'], output=str(args.output),
                          largest_error=report.get('largest_error'))), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
