"""Consolidate retained0281-0288 artifacts; no model construction or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/qwen4exp'


def parity(a,b):
    a,b=DOC/a,DOC/b
    ra,rb=json.loads((a/'result.json').read_text()),json.loads((b/'result.json').read_text())
    rows=[]
    for p in sorted(a.glob('logits_*.npy')):
        x,y=np.load(p),np.load(b/p.name)
        rows.append(dict(file=p.name,bit_exact=bool(np.array_equal(x,y)),
            max_abs=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))))
    states=ra['requests']==rb['requests']
    if len(rows)!=10 or not states or not all(r['bit_exact'] for r in rows):
        raise AssertionError('Same-arithmetic parity failed; preserve artifacts')
    return dict(candidate=str(a.relative_to(ROOT)),control=str(b.relative_to(ROOT)),
        token_ids=ra['token_ids'],requests=2,all_state_hashes_exact=states,logits=rows,
        scope='same-arithmetic optimization parity; not new independent llama.cpp equivalence')


def junit(name):
    root=ET.parse(DOC/name).getroot()
    suites=list(root.iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    names=['diagnostics_0281_expert_q8_v1','diagnostics_0282_all_q8_v2',
        'diagnostics_0282_q8_workers19_spin50_v1','diagnostics_0283_fused_hidden_v1',
        'diagnostics_0284_bf16_hc_v1','diagnostics_0284_bf16_hc_cache7_v1',
        'diagnostics_0284_q8_graphs_cache7_v1','diagnostics_0285_warm_profile_v1',
        'diagnostics_0285_q8_workers15_v1','diagnostics_0287_native_service_v1',
        'diagnostics_0288_sized_slots_v1','diagnostics_0288_forced_control_v1',
        'diagnostics_0288_forced_sized_v1','diagnostics_0288_reproduce_v1',
                'diagnostics_0287_native_service_v2','diagnostics_0287_native_service_v3']
    runs=[]
    for name in names:
        p=DOC/name/'result.json'; r=json.loads(p.read_text()); case=r['cases'][-1]
        h=case['cache_after']['native_handoffs']
        runs.append(dict(artifact=str(p.relative_to(ROOT)),tokens=case['output_token_ids'],
            forced_trajectory=case.get('forced_trajectory',False),timing=case['timing'],
            retired_timeout_path=name=='diagnostics_0287_native_service_v1',
            cpu_ms_per_forward_including_prefill=sum(x['cpu_seconds'] for x in h)*1000/h[0]['calls'],
            gpu_experts_per_forward_including_prefill=sum(x['gpu_experts'] for x in h)/h[0]['calls'],
            sampled_highwater_bytes=r['sampled_highwater_bytes'],
            gpu_pool_used_bytes=r['gpu_pool_used_bytes'],gpu_pool_reserved_bytes=r['gpu_pool_reserved_bytes'],
            arithmetic=r['packed_gpu_arithmetic'],cpu_configuration=r['cpu_configuration'],
            slots=case['cache_after']['slots']))
    sources=['mini_llm/ops/mixed_q8_gpu.py','mini_llm/ops/quantized_linear.py',
        'mini_llm/ops/device_expert_plan.py','mini_llm/ops/native_expert_handoff.py',
        'mini_llm/ops/ggml_packed_cpu.cpp','mini_llm/ops/ggml_packed_cpu.py',
        'mini_llm/ops/native_expert_service.h','mini_llm/ops/native_expert_service.py',
        'mini_llm/ops/expert_residency.py','mini_llm/ops/expert_slot_layout.py',
        'mini_llm/ops/packed_expert_slots.py','mini_llm/ops/mapped_packed_experts.py',
        'mini_llm/ops/hyperconnection_read_gpu.py','mini_llm/external/qwen4exp_model.py',
        'mini_llm/external/qwen4exp_weights.py','mini_llm/external/qwen4exp_hyperconnection.py',
                'scripts/probe_0254_strata_workload.py','scripts/probe_0266_critical_path.py']
    report=dict(target_decode_forwards_per_second=50,target_reached=False,
        git_head=subprocess.check_output(['git','--no-pager','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        runs=runs,resource_failures=[dict(artifact=f'docs/qwen4exp/{n}/child.log',
            failure=(DOC/n/'child.log').read_text().strip().splitlines()[-1],
            result_inferred=False) for n in ('diagnostics_0288_cache8_v1','diagnostics_0288_cache8_v2')],
        parity=[parity('diagnostics_0284_q8_bf16_parity_graph_v1','diagnostics_0284_q8_bf16_parity_control_v1'),
            parity('diagnostics_0287_service_parity_v1','diagnostics_0284_q8_bf16_parity_graph_v1'),
                        parity('diagnostics_0287_service_parity_v3','diagnostics_0284_q8_bf16_parity_graph_v1')],
        regressions={p:junit(p) for p in ('0288_cpu_regression_v2.xml','0288_gpu_regression_v2.xml')},
        limitations=['One sequential run/configuration, not confidence intervals',
            'No speedup ratio across different sampled token IDs',
            'Full logits/head/sampler retained in forced-trajectory tests',
            'GPU pool snapshots exclude private graph/driver/device-wide peak allocations',
            'Strata suffix drafting, eligible tail omission, INT8 KV and context differ'])
    with args.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(regressions=report['regressions'],parity_exact=True,
        best_safe_configuration_observation=max(r['timing']['decode_forward_rate_N_minus_1_per_decode_second']
                              for r in runs if not r['forced_trajectory'] and not r['retired_timeout_path'])),indent=2))


if __name__=='__main__': main()
