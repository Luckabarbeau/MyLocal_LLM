"""Diagnose sparse QSA on identical real-checkpoint inputs; not a benchmark.

Run via the normal validator CLI/environment. Stops deliberately after the
second QSA layer's sparse chunk; writes a separate *_probe.json artifact.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import validate_qwen4exp_prefill as validator


class ProbeComplete(BaseException):
    pass


def main():
    args = validator.parser().parse_args()
    for key, value in validator.BEST_ENV.items():
        os.environ.setdefault(key, value)
    from mini_llm.backend import xp, asnumpy
    from mini_llm.ops import qsa_sparse_prefill_gpu as sparse
    from mini_llm.external.qwen4exp_prefill import _project
    original = sparse.try_sparse_qsa_chunk
    report = []
    path = args.output.with_name(args.output.stem + '_probe.json')

    def metric(a, b):
        a, b = asnumpy(a), asnumpy(b)
        delta = abs(a.astype('float64') - b.astype('float64'))
        index = tuple(int(i) for i in __import__('numpy').unravel_index(delta.argmax(), delta.shape))
        return dict(unequal=int((a != b).sum()), max_abs=float(delta.max()),
                    index=index, reference=float(a[index]), actual=float(b[index]))

    def probe(model, layer, attention, hidden, state):
        if (layer not in (3,7) or state.length < 2051
                or hidden.shape[0] < 2):
            return original(model, layer, attention, hidden, state)
        print('Isolating QSA', layer, state.length, hidden.shape, flush=True)
        item = dict(layer=layer, start=state.length, rows=len(hidden), projections={})
        names = dict(query_gate='attn_q', key='attn_k', value='attn_v',
                     index_key='indexer.k_proj', index_query='indexer.q_proj')
        for name, tensor in names.items():
            reference = xp.stack([attention._project(name, row) for row in hidden])
            actual = _project(model, attention.projections[name][0],
                              f'blk.{layer}.{tensor}.weight', hidden)
            item['projections'][name] = metric(reference, actual)
        end = state.length + len(hidden)
        a, b = attention.create_state(end+1), attention.create_state(end+1)
        for clone in (a,b):
            for name in ('keys','values','raw_index'):
                getattr(clone,name)[:state.length] = getattr(state,name)[:state.length]
            clone.pooled[:state.length//attention.config.block_size] = state.pooled[:state.length//attention.config.block_size]
            clone.length = state.length
        gated = []
        project_original = attention._project
        def observe(name, value, **kwargs):
            if name == 'output':
                gated.append(value.copy())
            return project_original(name, value, **kwargs)
        attention._project = observe
        try:
            expected = attention.forward(hidden, a, start_position=a.length)
        finally:
            attention._project = project_original
        g = xp.stack(gated)
        batched_output = _project(model, attention.projections['output'][0],
                                 f'blk.{layer}.attn_output.weight', g)
        item['common_input_output_projection'] = metric(expected, batched_output)
        os.environ['MINI_LLM_QSA_SPARSE_PREFILL'] = '1'
        try:
            actual = original(model, layer, attention, hidden, b)
        finally:
            os.environ['MINI_LLM_QSA_SPARSE_PREFILL'] = '0'
        item['result'] = metric(expected, actual)
        item['state'] = {name: metric(getattr(a,name)[:end], getattr(b,name)[:end])
                         for name in ('keys','values','raw_index')}
        report.append(item)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(item), flush=True)
        if layer == 7:
            raise ProbeComplete()
        return expected

    sparse.try_sparse_qsa_chunk = probe
    try:
        validator.main()
    except ProbeComplete:
        driver = json.loads(args.output.read_text())
        driver['status'] = 'diagnostic_stopped_deliberately'
        driver['diagnostic'] = str(path)
        args.output.write_text(json.dumps(driver, indent=2) + '\n')
        print('Diagnostic saved:', path, flush=True)
    finally:
        sparse.try_sparse_qsa_chunk = original


if __name__ == '__main__':
    main()
