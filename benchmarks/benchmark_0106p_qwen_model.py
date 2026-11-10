"""0106P bounded full Qwen3-Next completion and CPU/CUDA consistency smoke.

No external inference runtime, model download, speculative decoding or automatic
VRAM sizing. CUDA validation exports matched CPU references in a subprocess.
Synthetic mode requires the repository's pytest development fixtures.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Field:
    def __init__(self, value):
        self.value = value
    def contents(self, index):
        return self.value


def synthetic_index(manifest, arrays):
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    reader = SimpleNamespace(fields={key: Field(value) for key, value in manifest['fields'].items()},
        tensors=[SimpleNamespace(name=t['name'], shape=t['shape'],
            tensor_type=SimpleNamespace(name=t['kind']), data=arrays[t['payload']])
            for t in manifest['tensors']])
    return Qwen3NextGGUFIndex(reader)


def metrics(actual, expected, *, synthetic):
    from benchmarks.benchmark_0106n_recurrent_gpu import numerical_metrics
    from mini_llm.backend import asnumpy
    # Forty-eight FP32 layers include distinct CPU/GPU reduction orders. The
    # tighter synthetic oracle tolerance is retained from 0106O's chain check.
    return numerical_metrics(asnumpy(actual), expected, rtol=4e-4,
                             atol=8e-6 if synthetic else 1e-3, path='full_model_logits')


def prepare(args):
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    if args.model:
        index = Qwen3NextGGUFIndex.open(args.model)
        vocab = int(index.tensors['token_embd.weight'].shape[1])
        tokenizer = load_gguf_tokenizer(index.reader, vocab_size=vocab)
        return index, tokenizer, None
    fixture = runpy.run_path(str(ROOT / 'tests/test_0106p_qwen3next_model.py'))
    index, weights, globals_, _ = fixture['fixture']()
    return index, fixture['TinyTokenizer'](), (fixture, weights, globals_)


def export_reference(args):
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'numpy':
        raise RuntimeError('Reference export must run in an isolated NumPy process')
    from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel
    from mini_llm.generation import GenerationSession
    from mini_llm.serving_models import ServingModel, _external_stops
    index, tokenizer, fixture = prepare(args)
    ids = np.asarray([tokenizer.encode(args.prompt, add_special_tokens=False).ids], np.int32)
    if ids.shape[1] < 1 or ids.shape[1] + args.max_new_tokens > args.context:
        raise ValueError('Prompt plus requested completion must fit explicit --context')
    arrays = {'prompt_ids': ids}
    manifest = dict(synthetic=not bool(args.model), model=str(args.model) if args.model else None,
                    prompt=args.prompt, context=args.context)
    if fixture is not None:
        manifest['fields'] = {name: field.contents(0) for name, field in index.reader.fields.items()}
        manifest['tensors'] = []
        for n, tensor in enumerate(index.tensors.values()):
            key = f'tensor_{n}'
            arrays[key] = tensor.data
            manifest['tensors'].append(dict(name=tensor.name, shape=list(tensor.shape),
                                            kind=tensor.tensor_type.name, payload=key))
    with Qwen3NextGGUFModel.from_index(index, cpu_workers=args.cpu_workers) as model:
        adapter = ServingModel(model, tokenizer, args.context, stop_ids=_external_stops(tokenizer))
        session = GenerationSession(adapter)
        outcome = session.generate(args.prompt, max_tokens=args.max_new_tokens, temperature=0,
                                   include_stop_token=True, log_speed=True)
        generated = list(outcome.token_ids)
        # Also validate EOS logits even when generation stopped at its first token.
        matched = np.concatenate((ids, np.asarray([generated[:-1]], np.int32)), axis=1)
        begin = time.perf_counter()
        full = model.forward(matched)
        full_ms = (time.perf_counter() - begin) * 1000
        arrays['matched_ids'], arrays['cpu_full_logits'] = matched, full
        comparisons = {}
        state = model.create_generation_state(1, args.context)
        arrays['cpu_prefill_logits'] = model.prefill(ids, state)
        comparisons['prefill_vs_full'] = metrics(arrays['cpu_prefill_logits'], full[:, ids.shape[1]-1],
                                                synthetic=fixture is not None)
        for step, token in enumerate(generated[:-1]):
            logits = model.decode_one([token], state)
            arrays[f'cpu_decode_{step}'] = logits
            comparisons[f'decode_{step}_vs_full'] = metrics(logits, full[:, ids.shape[1]+step],
                                                           synthetic=fixture is not None)
        if fixture is not None:
            functions, weights, globals_ = fixture
            arrays['oracle_logits'] = functions['reference'](matched, model, weights, globals_)
            comparisons['independent_fp64'] = metrics(full, arrays['oracle_logits'], synthetic=True)
        report = dict(patch='0106P', backend='numpy', passed=all(m['passed'] for m in comparisons.values()),
            scope='Whole-model raw completion; no chat-template or independent real-model reference claim',
            synthetic=fixture is not None, model=str(args.model) if args.model else 'synthetic',
            prompt_token_ids=ids[0].tolist(), generated_token_ids=generated, generated_text=outcome.text,
            finish_reason=outcome.reason, comparisons=comparisons,
            memory_plan=model.memory_plan(1, args.context), embedding=model.embedding.statistics(),
            expert_execution=model.expert_executor.statistics(),
            timing=dict(prefill_ms=session.prefill_time_s * 1000,
                        decode_ms=[value * 1000 for value in session.forward_times[1:]],
                        full_forward_ms=full_ms),
            timing_warning='Single correctness smoke; cold mmap/compiler/cache effects, no fair Strata comparison')
        manifest['cpu_report'] = report
        arrays['manifest'] = np.asarray(json.dumps(manifest))
        np.savez(args.export_reference, **arrays)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        if not report['passed']:
            raise AssertionError(f'CPU model consistency failed; see {args.output}')
        return report


def validate_cuda(args, manifest, arrays):
    from mini_llm.backend import BACKEND_NAME, xp, asnumpy
    if BACKEND_NAME != 'cupy':
        raise RuntimeError('CUDA validation requires MINI_LLM_BACKEND=cupy')
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel
    from mini_llm.diagnostics.routing_trace import RoutingTrace
    index = Qwen3NextGGUFIndex.open(args.model) if args.model else synthetic_index(manifest, arrays)
    model_id = index.source_path or '0106p-synthetic'
    # A supplied executor retains this explicit trace identity across all layers.
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    trace = RoutingTrace(model_id=model_id, quantization='GGUF-mixed', expert_mapping='identity',
                         workload='0106p-raw-completion', mode='summary')
    pool = xp.cuda.MemoryPool()
    pool.set_limit(size=args.device_budget_mib * 1024**2)
    report = dict(patch='0106P', backend='cupy', passed=False, comparisons={},
        cpu_reference=manifest['cpu_report'], timing={}, pool_limit_bytes=pool.get_limit())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def check(name, actual, expected):
        metric = metrics(actual, expected, synthetic=manifest['synthetic'])
        report['comparisons'][name] = metric
        if not metric['passed']:
            save()
            raise AssertionError(f'{name}: {metric}')
    run = None
    try:
        with xp.cuda.using_allocator(pool.malloc):
            run = HeterogeneousExpertExecutor(index.provider, model_id=model_id,
                d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
                device_budget_bytes=args.expert_cache_mib * 1024**2,
                cpu_workers=args.cpu_workers, cpu_packed_backend='native',
                promote_after=args.promote_after, routing_profiler=trace)
            with Qwen3NextGGUFModel.from_index(index, expert_executor=run,
                device_memory_budget_bytes=args.device_budget_mib * 1024**2,
                device_reserve_bytes=args.device_reserve_mib * 1024**2) as model:
                ready = xp.cuda.Event(disable_timing=True)
                ready.record(xp.cuda.get_current_stream())
                stream = xp.cuda.Stream(non_blocking=True)
                stream.wait_event(ready)  # Constructor weight uploads precede this stream's reads.
                report['non_default_stream'] = bool(stream.ptr)
                def timed(name, function):
                    stream.synchronize()
                    begin, end = xp.cuda.Event(), xp.cuda.Event()
                    start = time.perf_counter()
                    begin.record(stream)
                    result = function()
                    end.record(stream)
                    end.synchronize()
                    report['timing'][name] = dict(cuda_event_ms=float(xp.cuda.get_elapsed_time(begin, end)),
                                                 wall_ms=(time.perf_counter() - start)*1000)
                    return result
                with stream:
                    state = model.create_generation_state(1, args.context)
                    ids = xp.asarray(arrays['prompt_ids'])
                    logits = timed('prefill', lambda: model.prefill(ids, state))
                    check('prefill_cpu', logits, arrays['cpu_prefill_logits'])
                    generated = manifest['cpu_report']['generated_token_ids']
                    actual_ids = [int(xp.argmax(logits[0]).item())]
                    for step, token in enumerate(generated[:-1]):
                        logits = timed(f'decode_{step}', lambda: model.decode_one([token], state))
                        check(f'decode_{step}_cpu', logits, arrays[f'cpu_decode_{step}'])
                        actual_ids.append(int(xp.argmax(logits[0]).item()))
                    report['matched_input_greedy_ids'] = actual_ids
                    if actual_ids != generated:
                        raise AssertionError('CPU/CUDA greedy IDs differ; inspect saved logits metrics')
                    full = timed('full_forward', lambda: model.forward(xp.asarray(arrays['matched_ids'])))
                    check('full_cpu', full, arrays['cpu_full_logits'])
                    if 'oracle_logits' in arrays:
                        check('full_independent_fp64', full, arrays['oracle_logits'])
                    if not all(layer.length == state.length for layer in state.layer_states):
                        raise AssertionError('Hybrid state lengths disagree')
                    report['state_length'] = state.length
                    stream.synchronize()
                    report['memory_plan'] = model.memory_plan(1, args.context)
                    report['embedding'] = model.embedding.statistics()
                    report['expert_execution'] = run.statistics()
                    report['routing_trace'] = trace.snapshot()
                    report['pool_reserved_bytes'] = int(pool.total_bytes())
                    report['pool_used_bytes'] = int(pool.used_bytes())
                    report['device'] = xp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                    report['passed'] = True
    except Exception as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    finally:
        if run is not None:
            run.close()
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--prompt', default='def add')
    parser.add_argument('--max-new-tokens', type=int, default=2)
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--cpu-workers', type=int, default=2)
    parser.add_argument('--device-budget-mib', type=int, default=4096)
    parser.add_argument('--device-reserve-mib', type=int, default=2048)
    parser.add_argument('--expert-cache-mib', type=int, default=0)
    parser.add_argument('--promote-after', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-reference', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 8 or not 2 <= args.context <= 128 or not 0 <= args.cpu_workers <= 4:
        parser.error('Smoke requires 1..8 new tokens, 2..128 context and 0..4 CPU workers')
    if args.export_reference is not None:
        report = export_reference(args)
    else:
        with tempfile.TemporaryDirectory(prefix='0106p-model-') as temp:
            reference_path = Path(temp) / 'reference.npz'
            if os.environ.get('MINI_LLM_BACKEND', 'numpy').lower() == 'numpy':
                args.export_reference = reference_path
                report = export_reference(args)
            else:
                command = [sys.executable, str(Path(__file__).resolve()), '--export-reference', str(reference_path),
                    '--output', str(args.output.with_suffix('.cpu.json')), '--prompt', args.prompt,
                    '--max-new-tokens', str(args.max_new_tokens), '--context', str(args.context),
                    '--cpu-workers', str(args.cpu_workers)]
                if args.model:
                    command += ['--model', str(args.model)]
                completed = subprocess.run(command, cwd=ROOT,
                    env=dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1'),
                    capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    raise RuntimeError(f'CPU reference export failed:\n{completed.stdout}\n{completed.stderr}')
                with np.load(reference_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files if key != 'manifest'}
                    manifest = json.loads(str(archive['manifest'].item()))
                report = validate_cuda(args, manifest, arrays)
    print(json.dumps(dict(patch=report['patch'], backend=report['backend'], passed=report['passed'],
        output=str(args.output), cpu_text=report.get('generated_text',
            report.get('cpu_reference', {}).get('generated_text'))), indent=2))
    return report


if __name__ == '__main__':
    main()
