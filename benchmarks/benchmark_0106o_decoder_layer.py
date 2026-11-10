"""CPU-only complete-layer correctness/timing smoke; never requests CUDA.

Synthetic mode runs a four-layer hybrid fixture and independent FP64 oracles
(pytest dev extra required). --model checks one real GGUF layer's sequence /
chunk / step parity, not independent full-model logits or generation speed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import runpy
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, *, rtol=4e-4, atol=8e-6):
    from benchmarks.benchmark_0106n_recurrent_gpu import numerical_metrics
    return numerical_metrics(actual, expected, rtol=rtol, atol=atol, path='decoder_layer')


def buffers(state):
    native = state.mixer_state
    if hasattr(native, 'k_cache'):
        return {'k_cache': native.k_cache, 'v_cache': native.v_cache}
    return {'history': native.conv.history, 'memory': native.delta.memory}


def memory_preflight(index, layer_ids, tokens, prefix_tokens, batch, workers, limit):
    """Header-based conservative working-set estimate, not an OS RAM guarantee."""
    from mini_llm.external.qwen3next_gguf import _shape
    from mini_llm.external.qwen3next_moe import Qwen3NextMoESpec
    from mini_llm.external.qwen3next_mixer import Qwen3NextMixerSpec
    from mini_llm.external.qwen3next_attention import Qwen3NextAttentionSpec
    non_expert_expanded = selected_packed = state_bytes = 0
    largest_dense_matrix = 0
    for layer_id in layer_ids:
        spec = Qwen3NextMoESpec.from_index(index, layer_id)
        stem = f'blk.{layer_id}.'
        for name, tensor in index.tensors.items():
            if name.startswith(stem) and '_exps.weight' not in name:
                elements = int(np.prod(_shape(tensor)))
                non_expert_expanded += elements * 4
                largest_dense_matrix = max(largest_dense_matrix, elements * 4)
        selected = min(spec.n_experts, spec.top_k * (prefix_tokens + 3 * tokens) * batch)
        selected_packed += selected * sum(w.nbytes for w in index.expert_weights(layer_id, 0))
        interval = index.reader.fields[f'{index.architecture}.full_attention_interval'].contents(0)
        if (layer_id + 1) % interval == 0:
            attn = Qwen3NextAttentionSpec.from_index(index)
            state_bytes += batch * attn.n_kv_heads * (prefix_tokens + tokens + 2) * (
                attn.key_dim + attn.value_dim) * 4
        else:
            mixer = Qwen3NextMixerSpec.from_index(index)
            state_bytes += batch * ((mixer.conv_kernel_size - 1) * mixer.qkv_channels
                + mixer.n_value_heads * mixer.key_dim * mixer.value_dim) * 4
    # Includes potential reference dequant scratch and caller state forks.
    worker_scratch = max(1, workers) * 12 * index.d_model * index.d_ff
    estimate = non_expert_expanded + selected_packed + 5 * state_bytes + worker_scratch + largest_dense_matrix
    if estimate > limit:
        raise MemoryError(f'Header working-set estimate {estimate} exceeds explicit budget {limit}')
    return dict(limit_bytes=limit, estimate_bytes=estimate,
                expanded_non_expert_upper_bytes=non_expert_expanded,
                selected_packed_upper_bytes=selected_packed, single_state_bytes=state_bytes,
                note='Estimate excludes interpreter/compiler/library overhead and OS page-cache accounting')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--tokens', type=int, default=4)
    parser.add_argument('--cpu-backend', choices=('reference', 'native'), default='native')
    parser.add_argument('--cpu-workers', type=int, default=0)
    parser.add_argument('--profile', choices=('summary', 'detailed'), default='summary')
    parser.add_argument('--memory-budget-mib', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'numpy':
        parser.error('This smoke is CPU-only; use MINI_LLM_BACKEND=numpy')
    if not 1 <= args.tokens <= 32 or not 0 <= args.cpu_workers <= 4 or args.memory_budget_mib < 1:
        parser.error('Require 1..32 tokens, 0..4 CPU workers and a positive memory budget')
    from mini_llm.diagnostics.routing_trace import RoutingTrace
    from mini_llm.external.qwen3next_decoder import Qwen3NextDecoderLayerInference
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor

    functions = weights = None
    if args.model:
        index = Qwen3NextGGUFIndex.open(args.model)
        layer_ids, batch = [args.layer], 1
    else:
        functions = runpy.run_path(str(ROOT / 'tests/test_0106o_qwen3next_decoder.py'))
        index, weights, _ = functions['fixture'](packed_shared=True)
        layer_ids, batch = list(range(4)), 2
    prefix_tokens, capacity = 2, 2 + args.tokens + 2
    budget = memory_preflight(index, layer_ids, args.tokens, prefix_tokens, batch,
                              args.cpu_workers, args.memory_budget_mib * 1024**2)
    model_id = str(args.model.resolve()) if args.model else '0106o-synthetic'

    def trace(mode):
        return RoutingTrace(model_id=model_id, quantization='GGUF-mixed',
            expert_mapping='identity', workload='decoder-smoke/' + mode,
            mode=args.profile, max_records=128)

    run = HeterogeneousExpertExecutor(index.provider, model_id=model_id,
        d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
        device_budget_bytes=0, cpu_packed_backend=args.cpu_backend,
        cpu_workers=args.cpu_workers, routing_profiler=trace('prefix'))
    try:
        layers = [Qwen3NextDecoderLayerInference.from_index(index, layer_id,
            expert_executor=run, cpu_packed_backend=args.cpu_backend) for layer_id in layer_ids]
        rng = np.random.default_rng(10615)
        prefix = rng.normal(0, .2, (batch, prefix_tokens, index.d_model)).astype(np.float32)
        hidden = rng.normal(0, .2, (batch, args.tokens, index.d_model)).astype(np.float32)
        original = hidden.copy()
        initial = [layer.create_state(batch, capacity) for layer in layers]
        x = prefix
        for layer, state in zip(layers, initial):
            x = layer.forward(x, state)
        reference = ref_states = None
        if functions is not None:
            reference = np.concatenate((prefix, hidden), axis=1)
            ref_states = {}
            for layer in layers:
                reference, a, b = functions['reference_layer'](reference, weights[layer.layer_id], layer)
                ref_states[layer.layer_id] = (a, b)
            reference = reference[:, prefix_tokens:]
        reports, baseline, baseline_states = {}, None, None
        for mode in ('full', 'chunks', 'steps'):
            run.routing_profiler = trace(mode)
            states = [layer.copy_state(state) for layer, state in zip(layers, initial)]
            pointers = [{name: value.__array_interface__['data'][0]
                         for name, value in buffers(state).items()} for state in states]
            n = args.tokens
            edges = ((0, n) if mode == 'full' else
                     (0, min(2, n), min(2, n), n) if mode == 'chunks' else tuple(range(n + 1)))
            outputs, per_layer = [], {str(i): 0.0 for i in layer_ids}
            start = time.perf_counter()
            for a, b in zip(edges, edges[1:]):
                x = hidden[:, a:b]
                for layer, state in zip(layers, states):
                    begin = time.perf_counter()
                    x = layer.forward(x, state)
                    per_layer[str(layer.layer_id)] += (time.perf_counter() - begin) * 1000
                outputs.append(x)
            output = np.concatenate(outputs, axis=1)
            elapsed = (time.perf_counter() - start) * 1000
            if baseline is None:
                baseline = output.copy()
                baseline_states = [{name: value.copy() for name, value in buffers(s).items()} for s in states]
            comparisons = {'sequence_output': compare(output, baseline)}
            if reference is not None:
                comparisons['independent_fp64_output'] = compare(output, reference)
            for i, (layer, state) in enumerate(zip(layers, states)):
                for j, (name, value) in enumerate(buffers(state).items()):
                    comparisons[f'{layer.layer_id}/{name}'] = compare(value, baseline_states[i][name])
                    if ref_states is not None:
                        actual = value[:, :, :state.length] if layer.is_attention else value
                        comparisons[f'{layer.layer_id}/fp64/{name}'] = compare(actual, ref_states[layer.layer_id][j])
                    if value.__array_interface__['data'][0] != pointers[i][name]:
                        raise AssertionError('Layer replaced native state storage')
            passed = all(value['passed'] for value in comparisons.values()) and all(
                state.length == prefix_tokens + n for state in states)
            reports[mode] = dict(passed=passed, comparisons=comparisons,
                observed_cpu_wall_ms=elapsed, per_layer_wall_ms=per_layer,
                routing_trace=run.routing_profiler.snapshot(), lengths=[s.length for s in states])
        if not np.array_equal(hidden, original):
            raise AssertionError('Decoder mutated input hidden states')
        result = dict(patch='0106O', backend='numpy', model=str(args.model) if args.model else 'synthetic',
            scope='Complete decoder layer(s), no embeddings/logits/token generation', layer_ids=layer_ids,
            batch=batch, prefix_tokens=prefix_tokens, continuation_tokens=args.tokens,
            dtype='float32', seed=10615, memory_preflight=budget,
            peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            moe_specs={str(layer.layer_id): asdict(layer.moe.spec) for layer in layers},
            executor=run.statistics(), runs=reports,
            passed=all(report['passed'] for report in reports.values()),
            independent_fp64_reference=functions is not None,
            gpu_validation='pending; existing Strata process left undisturbed',
            timing_warning='CPU wall correctness smoke under concurrent workload; not controlled throughput',
            full_model_decode_available=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({key: result[key] for key in ('patch', 'layer_ids', 'passed', 'peak_process_rss_bytes')}, indent=2))
        if not result['passed']:
            raise AssertionError(f'Decoder parity failed; see {args.output}')
        return result
    finally:
        run.close()


if __name__ == '__main__':
    main()
