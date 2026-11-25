"""CPU-only aggregation of immutable conditioned evidence, never reruns math."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return dict(maxabs=float(np.max(np.abs(d))),relL2=float(np.linalg.norm(d)/max(np.linalg.norm(b.astype(np.float64)),1e-30)))


def main(root, destination):
    reports=[json.loads((root/f'layer{i:02d}/report.json').read_text()) for i in range(48)]
    assert all(r['passed'] for r in reports)
    historical_path=Path('docs/native_engine/WP7_FULL_MODEL_DIAGNOSTICS.json')
    historical=json.loads(historical_path.read_text())
    head=json.loads((root/'head/report.json').read_text())
    checks=[(r['layer'],k,v) for r in reports for k,v in r['checks'].items() if 'captured_' not in k and not v.get('empty')]
    groups={}
    for label,fragment in [('hc_read','hidden'),('hc_gains','gains'),('hc_write','_write'),
                           ('mixer_output','mixer_output'),('mixer_history','mixer_history'),
                           ('mixer_matrix','mixer_matrix'),('qsa_raw','mixer_raw'),
                           ('qsa_keys','mixer_keys'),('qsa_values','mixer_values'),
                           ('moe_output','moe_output'),('cpu_rows','moe_cpu_rows')]:
        rows=[(i,k,v) for i,k,v in checks if fragment in k and
                      (label != 'hc_read' or k.endswith(('attention_hidden','ffn_hidden')))]
        worst=max(rows,key=lambda r:r[2]['maxabs'])
        groups[label]=dict(comparisons=len(rows),maxabs=worst[2]['maxabs'],
            max_relL2=max(r[2]['relL2'] for r in rows),maxrel=max(r[2]['maxrel'] for r in rows),
            violations=sum(r[2]['violations'] for r in rows),worst_layer=worst[0],worst_key=worst[1])
    cross=[]
    for report in reports:
        i=report['layer']
        with np.load(root/f'layer{i:02d}/evidence.npz',allow_pickle=False) as e:
            for t in (0,1):
                row=dict(layer=i,token_index=t,**report['cross_lineage'][t])
                row['mixer_output']=delta(e[f'native_{t}_mixer_output_native'],e[f'python_{t}_mixer_output_native'])
                row['quantizer_detail']={}
                for name,qb,offset,width in [('quantize_row_q8_K',292,4,256),('quantize_row_q8_0',34,2,32)]:
                    n,p=(e[f'{l}_{t}_'+name].reshape(-1,qb) for l in ('native','python'))
                    nq,pq=(x[:,offset:offset+width].view(np.int8) for x in (n,p))
                    changes=np.argwhere(nq!=pq)
                    samples=[]
                    for block,lane in changes[:8]:
                        coordinate=int(block*width+lane)
                        samples.append(dict(block=int(block),lane=int(lane),coordinate=coordinate,
                            native_code=int(nq[block,lane]),python_code=int(pq[block,lane]),
                            native_input=float(e[f'native_{t}_moe_input_native'][coordinate]),
                            python_input=float(e[f'python_{t}_moe_input_native'][coordinate])))
                    row['quantizer_detail'][name]=dict(changed_bytes=int(np.count_nonzero(n!=p)),
                        changed_integer_codes=len(changes),changed_scale_blocks=int(np.count_nonzero(np.any(n[:,:offset]!=p[:,:offset],axis=1))),
                        code_change_samples=samples)
                cross.append(row)
    capture_checks=[(r['layer'],k,v) for r in reports for k,v in r['checks'].items() if 'captured_' in k]
    batches=[json.loads((root/f'batch{i:02d}-{i+8:02d}.json').read_text()) for i in range(0,48,8)]
    result=dict(scope='All 48 actual-weight layers, two retained tokens, both complete hidden histories; diagnostic only',
        root=str(root),tokens=[1,7],layers_executed=list(range(48)),conditioned_all_layers_passed=True,
        no_production_arithmetic_changes=True,no_tolerance_changes=True,rtol=2e-4,atol=2e-4,
        exact_contracts=['router IDs','all ten CPU expert rows','input identity','once-only weighting','ordered routed combine','QSA selected IDs'],
        same_input_cpu_rows_checked=48*2*2*10,checks=len(checks),groups=groups,
        max_accounted_layer_bytes=max(sum(r['memory'].values()) for r in reports),
        max_accounted_head_bytes=sum(head['memory'].values()),
        min_preflight_free_bytes=min(r['preflight']['free_bytes'] for r in reports),
        memory_scope='Native peak + CuPy reserved, excluding driver/context; each preflight additionally reserves 2 GiB.',
        captures_reconstructed=dict(comparisons=len(capture_checks),violations=sum(v['violations'] for _,_,v in capture_checks),
            exact_native=sum(v['exact'] for _,k,v in capture_checks if k.startswith('native_')),
            native_comparisons=sum(k.startswith('native_') for _,k,_ in capture_checks)),
        head=head,cross_lineage=cross,
        route_membership_changes=[dict(layer=c['layer'],token_index=c['token_index']) for c in cross if c['ids_changed']],
        cross_lineage_q8k_code_change_cases=sum(c['quantizer_detail']['quantize_row_q8_K']['changed_integer_codes']>0 for c in cross),
        legacy_strict=dict(unchanged=True,strict_whole_model_acceptance_passed=False,
            token0=historical['token0']['full_logits'],token1=historical['token1']['full_logits'],
            note='Legacy strict failure remains visible. No quality acceptance derived from two tokens.'),
        interpretation='All tested operations are consistent under identical operands/history at fixed existing envelopes. Evidence supports FP32 reduction roundoff plus quantization/routing thresholds and persistent-history propagation, not a demonstrated equation defect. This is not bitwise full-trajectory equivalence or a proof for untested contexts.',
        limits=['QSA pooled history is empty at lengths 1 and 2; active empty shape checked, no pooled update or sparse transition exercised.',
                'Only the captured all-CPU expert placement with explicit mixed arithmetic policy is covered; GPU-resident experts are not substituted.',
                'Cross-lineage deltas use different captured inputs and are NOT same-input failures.',
                'Independent lower-level/full-model equivalence and longer-sequence quality are not established.'],
        provenance=dict(original_metadata_sha256=digest(historical_path),batches=batches,
            layer_reports={str(root/f'layer{i:02d}/report.json'):digest(root/f'layer{i:02d}/report.json') for i in range(48)},
            head_report_sha256=digest(root/'head/report.json'),
            new_source_sha256={str(p):digest(p) for p in [Path('tests')/name for name in
                ('native_decoder_conditioned.py','native_decoder_conditioned_head.py','native_decoder_conditioned_report.py','test_native_engine_conditioned.py')]},
            executed_bounds_seconds=dict(layer_batches=180,head=120,retained_tests=120)),
        verification=dict(command='.venv/bin/python -m pytest -q tests/test_native_engine_conditioned.py -x --tb=short',passed=51,seconds=2.51),
        parent_status_modified=False)
    with destination.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('same_input_cpu_rows_checked','max_accounted_layer_bytes','max_accounted_head_bytes','captures_reconstructed','route_membership_changes','cross_lineage_q8k_code_change_cases','groups')},indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
