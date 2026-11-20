"""Summarize retained 0293..0307 artifacts; never initialize a model or CUDA."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/qwen4exp'


def main():
    names=('0293_service_control_v1','0294_readonly_control_v1','0296_bound_v1',
        '0297_sized725_v1','0297_deferred_v1','0298_phases_v1','0299_workers23_v1',
        '0300_workers15_v1','0302_qsa_v1','0304_factor8_v1','0305_fused_v1',
        '0305_control_v1','0306_publish_v1','0306_combined_v1','0307_control_v2',
        '0307_long_v1','0307_long_v2','0307_combined_v1','0307_natural_v1','0307_natural_v2')
    results=[]
    for name in names:
        path=DOC/('diagnostics_'+name)/'result.json'
        d=json.loads(path.read_text()); case=d['cases'][-1]
        ids=case['output_token_ids']
        def delta(field):
            return case['cache_after']['execution'].get(field,0)-case['cache_before']['execution'].get(field,0)
        results.append(dict(artifact=str(path.relative_to(ROOT)),forced=case['forced_trajectory'],
            timing=case['timing'],output_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            cpu_workers=d['cpu_configuration']['workers'],cache_bytes=d['options']['cache_bytes'],
            cpu_elapsed_ns=delta('native_handoff_cpu_ns'),cpu_experts=delta('ggml_cpu_experts'),
            response_model_steps=len(case['prompt']['token_ids'])+case['timing']['decode_forwards'],
            response_interval_disk_read_bytes=case['memory']['io']['read_bytes']-d['cases'][-2]['memory']['io']['read_bytes'],
            response_interval_major_faults=case['memory']['major_faults']-d['cases'][-2]['memory']['major_faults'],
            gpu_experts=delta('gpu_experts'),sampled_highwater_bytes=d.get('sampled_highwater_bytes'),
            gpu_pool_used_bytes=d['gpu_pool_used_bytes'],gpu_pool_reserved_bytes=d['gpu_pool_reserved_bytes'],
            memory=case['memory']))
    guards=[]
    for path in sorted(DOC.glob('diagnostics_03*_guard_v*/guard.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        samples=[r for r in rows if 'memory_peak' in r]
        if not samples: continue
        guards.append(dict(artifact=str(path.relative_to(ROOT)),
            peak_charged_bytes=max(r['memory_peak'] for r in samples),
            minimum_host_available_bytes=min(r['host_available'] for r in samples),
            max_psi_some_avg10=max(r['psi_some_avg10'] for r in samples),
            max_psi_full_avg10=max(r['psi_full_avg10'] for r in samples),
            oom_kill=max(r['memory_events'].get('oom_kill',0) for r in samples),
            last_event=rows[-1]))
    sources={}
    tracked=subprocess.check_output(['git','--no-pager','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','--no-pager','ls-files','--others','--exclude-standard','mini_llm','scripts','tests'],cwd=ROOT,text=True).splitlines()
    for name in sorted(set(tracked+untracked)):
        path=ROOT/name
        if path.is_file() and path.suffix in ('.py','.cpp','.h'):
            sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    tests=[]
    for name in ('0307_cpu_regression_v1.xml','0307_gpu_regression_v1.xml'):
        path=DOC/name
        suites=list(ET.parse(path).getroot().iter('testsuite'))
        totals={key:sum(int(s.attrib.get(key,'0')) for s in suites)
                for key in ('tests','failures','errors','skipped')}
        assert not totals['failures'] and not totals['errors'],(name,totals)
        tests.append(dict(artifact=str(path.relative_to(ROOT)),**totals))
    operators=json.loads((DOC/'diagnostics_0307_operator_v2/result.json').read_text())
    result=dict(target_decode_forwards_per_second=50,target_met=False,
        reference_revisions=dict(strata='712fec6b4297c9cfd32ab43daaa2979a9aa01edf',
                                ggml='171e8846b4af9766c354064cb776cb34a50f053f'),
        comparison_limit='Full top10, no suffix drafting, context1024 FP32 KV; historical Strata emitted-token rates are not equivalent.',
        results=results,guards=guards,tests=tests,source_sha256=sources,
        latest_gpu_operator_speedup_range=[min(r['speedup'] for r in operators['rows']),max(r['speedup'] for r in operators['rows'])],
        local_parity=json.loads((DOC/'0307_parity_comparison_v1.json').read_text()))
    output=DOC/'0295_0307_optimization_results_v2.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(output=str(output.relative_to(ROOT)),tests=tests,
        operator_speedup_range=result['latest_gpu_operator_speedup_range'],
        latest_rates=[(r['artifact'],r['timing']['decode_forward_rate_N_minus_1_per_decode_second']) for r in results[-5:]]),indent=2))


if __name__=='__main__': main()
