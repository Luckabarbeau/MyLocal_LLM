#!/usr/bin/env python3
"""Reproduce retained 0318..0323 results with stdlib only; never run a model.

The sole write is exclusive creation of 0318_0323_execution_results_v1.json.
--check recomputes the complete snapshot read-only, including report/input/source
hashes. Historical source hashes remain historical; no old snapshot is refreshed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

# Import only the existing stdlib analyzer, without creating bytecode artifacts.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_0310_0317_results import (  # noqa: E402
    DOC, ROOT, RATE, Evidence, compare_parity, guard_result, junit,
    local_result, relative, sha256,
)

OUTPUT = DOC / '0318_0323_execution_results_v1.json'
REPORT = DOC / '0318_0323_execution_report.md'
PATCHES = ('0318', '0319', '0320', '0321', '0322', '0323')
PROFILE = 'python_critical_path_pid1177617_instance001_repeat001'
SOURCE_SUFFIXES = {'.py', '.pyi', '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp',
                   '.cu', '.cuh', '.sh', '.rs', '.js', '.ts', '.cmake'}
ASSET_DIRS = {'docs', 'data', 'checkpoints', 'tokenizers', 'logs', 'assets',
              'weights', 'oldweights', 'old_weights', '__pycache__', '.git'}

# Roles are a logical patch ledger, not invented per-patch Git commits: the
# retained repository commit groups several numbered experiments together.
LEDGER = {
    '0318': dict(
        objective='Test existing kernel publication with complete recurrent/QSA graphs',
        status='configuration_only_no_new_production_code',
        files=['scripts/probe_0302_decode_configuration.py',
               'mini_llm/ops/native_expert_handoff.py',
               'mini_llm/external/qwen4exp_recurrent_block_graph.py',
               'mini_llm/external/qwen4exp_recurrent_chain_graph.py',
               'mini_llm/external/qwen4exp_qsa_block_graph.py'],
        interface='Existing --kernel-publish plus --recurrent-block-graph or --recurrent-chain-graph and --qsa-block-graph; all files reused',
        tests='Two saved two-request parity fixtures versus 0317; no newly attributed test suite',
        performance='Separate publish and combined long response pairs; no robust small-gain claim',
        numerical='Ten logits and all recorded states exact versus 0317',
        memory='Existing publication fences, ownership and guards unchanged',
        limitations='Publication gaps are not an exclusive recoverable critical-path budget',
        next_dependency='Measure remaining useful work rather than assuming launch reduction gives model gain'),
    '0319': dict(
        objective='Coalesce key reads without changing serial FP32 arithmetic',
        status='standalone_bit_exact_not_integrated_performance_rejected',
        files=['mini_llm/ops/qsa_staged_gpu.py', 'scripts/probe_0319_qsa_staged_operator.py',
               'tests/test_0319_qsa_staged.py'],
        interface='qsa_staged_gpu and fail-closed staged_source adapter; no model dispatch',
        tests='0319_gpu_tests_v1.xml: 78 passed, 1 skipped',
        performance='Serial/staged ratios below one at n=141/435/1024; 1.6-1.8x slower',
        numerical='FP32 d=0..255 order, original softmax/value/sigmoid, --fmad=false, bitwise equality',
        memory='256*33 FP32 shared tile (33 KiB), no context-sized scratch or cache writes',
        limitations='Synthetic standalone operator only; correctness acceptance is not performance acceptance',
        next_dependency='Expose parallelism while retaining each output accumulation order'),
    '0320': dict(
        objective='Exact warp-per-dot plus localized value-dimension parallelism',
        status='operator_performance_optimized_model_opt_in_scope_validated',
        files=['mini_llm/ops/qsa_warp_dots_gpu.py', 'mini_llm/ops/qsa_decode_gpu.py',
               'mini_llm/ops/qsa_graph_gpu.py', 'scripts/probe_0320_qsa_warp_dots_operator.py',
               'scripts/probe_0302_decode_configuration.py',
               'tests/test_0320_qsa_warp_dots.py', 'tests/test_0320_qsa_graph_dispatch.py'],
        interface='MINI_LLM_QSA_WARP_DOTS=1 / --qsa-warp-dots; normal and dynamic-position graph dispatch; integrated value_tile=32',
        tests='92 initial, 168 tiled and 231 integration tests passed (overlapping suites, not additive coverage)',
        performance='Untiled neutral/slower; tile32 kernel 2.383/2.051/1.849x, API 2.002/1.970/1.884x; long run partial',
        numerical='Ordered products/additions and unchanged per-dimension FP32 softmax/V/gate order; ten logits/all states exact vs 0318 combined',
        memory='At most 24*8192 FP32 logits (786432 bytes); runtime n bounded by fixed capacity; no input/cache mutation',
        limitations='0320 parity v1 refused RAM before launch; actual parity v2 succeeded; second long request rejected by VRAM admission',
        next_dependency='Retire orphaned request graphs before next allocation/admission, not relax guards'),
    '0321': dict(
        objective='Release graphs retaining dead-request arrays before next admission',
        status='execution_resource_fix_scope_validated',
        files=['mini_llm/external/qwen4exp_model.py', 'tests/test_0321_orphaned_request_graphs.py'],
        interface='_retire_orphaned_state_graphs before create_generation_state memory_plan; only if weak live-state set is empty',
        tests='8 focused tests; selected CPU suite 67 passed/20 skipped',
        performance='Two-response long run completed; no robust speedup claim',
        numerical='No arithmetic change; ten logits and all state records exact versus 0318 combined',
        memory='Nonblocking call lock, live-state recheck, graph close/fence, unused allocator blocks released; guard/cache unchanged',
        limitations='Not permission to retire live states or skip device-capacity admission',
        next_dependency='Assess whole-token scheduling with immutable PLE and frozen complete expert plans'),
    '0322': dict(
        objective='One full 48-layer token graph with runtime positions and owned outputs',
        status='implemented_scope_validated_not_model_performance_optimized_off',
        files=['mini_llm/external/qwen4exp_token_graph.py', 'mini_llm/external/qwen4exp_model.py',
               'mini_llm/external/qwen4exp_ple.py', 'scripts/probe_0302_decode_configuration.py',
               'tests/test_0322_token_graph.py'],
        reused_components=['mini_llm/external/qwen4exp_qsa_block_graph.py',
                           'mini_llm/external/qwen4exp_recurrent_block_graph.py',
                           'mini_llm/ops/qsa_graph_gpu.py', 'mini_llm/ops/inference_graph.py',
                           'mini_llm/external/qwen4exp_weights.py',
                           'scripts/probe_0265_gdn_graph_parity.py'],
        interface='MINI_LLM_TOKEN_GRAPH / --token-graph; MINI_LLM_TOKEN_GRAPH_PREP / --token-graph-baseline for same prepared math without replay',
        tests='0322_gpu_tests_v1.xml 56 passed; later final-source suites include tiny real-math and lifecycle coverage',
        performance='39.32466/42.22064 forwards/s: no gain, stays off',
        numerical='Baseline prepared FP32 PLE vs candidate v3 exact; old cuBLAS PLE vs prepared differs, NOT quality accepted',
        memory='256 MiB private pool; old subgraph pools retired first; request-bound recapture; owned history/logits survive retirement',
        limitations='48 layers/12 QSA/PLE layer1, bounded all-cells context, frozen complete CPU/GPU expert plans; no CPU row I/O during capture; original PLEState immutable',
        next_dependency='Keep prepared PLE quality gate separate from graph parity and useful-work ranking'),
    '0323': dict(
        objective='Profile exact first-sampler-return to last-sampler-return decode boundary',
        status='diagnostic_only_no_terminal_50_claim',
        files=['scripts/probe_0323_python_critical_path.py', 'tests/test_0323_python_critical_path.py'],
        interface='0290-isolated wrapper forwards 0302 flags, one forced RESPONSE; exclusive JSON/text/pstats',
        tests='35 profiler tests in final CPU JUnit; final selected CPU 732/116 skip, GPU 586/24 skip, one deselection each',
        performance='Profiled submitting-thread wall time, not throughput, CPU utilization or native-worker timing',
        numerical='No new model arithmetic; final source identity with last model parity is user execution-record attestation',
        memory='Unchanged strict isolation, reserve and guards; no new source/weight residency',
        limitations='Cumulative costs overlap; tiny status reads wait for queued work; deleting checks relocates waits',
        next_dependency='Rank useful packed-GPU/native-CPU time against matched hot Strata with same expert placement/counts'),
}


def git(*arguments):
    return subprocess.check_output(
        ['git', '--no-pager', '--no-optional-locks', *arguments], cwd=ROOT)


def git_names(*arguments):
    return {name.decode('utf-8', 'surrogateescape')
            for name in git(*arguments).split(b'\0') if name}


def is_source(name):
    p = Path(name)
    return (not any(part in ASSET_DIRS or part.startswith(('.venv', 'token_shards'))
                    for part in p.parts) and
            (p.suffix in SOURCE_SUFFIXES or p.name in ('CMakeLists.txt', 'Makefile')))


def source_snapshot():
    tracked = git_names('ls-files', '-z')
    untracked = git_names('ls-files', '--others', '--exclude-standard', '-z')
    unstaged = git_names('diff', '--name-only', '-z')
    staged = git_names('diff', '--cached', '--name-only', '-z')
    names = sorted(name for name in tracked | untracked if is_source(name))
    dirty = sorted(name for name in unstaged | staged | untracked if is_source(name))
    hashes = {name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in names}
    return dict(head=git('rev-parse', 'HEAD').decode().strip(), source_sha256=hashes,
                modified_or_untracked_source_paths=dirty,
                source_path_status={name: dict(staged=name in staged, unstaged=name in unstaged,
                                               untracked=name in untracked,
                                               exists=(ROOT / name).is_file()) for name in dirty},
                selection=dict(suffixes=sorted(SOURCE_SUFFIXES), excluded_directories=sorted(ASSET_DIRS),
                               excluded_prefixes=['.venv', 'token_shards']),
                scope='Current working tree, all Git-listed source paths, including modified/staged/deleted/untracked; excludes old weights/assets. Not historical run-source proof.')


def inventory(e):
    for patch in PATCHES:
        for directory in sorted(DOC.glob('diagnostics_' + patch + '*')):
            if directory.is_dir():
                for path in sorted(directory.rglob('*')):
                    if path.is_file():
                        e.record(path)
                        if path.suffix in ('.json', '.jsonl'):
                            e.load(path)
        for path in sorted(DOC.glob(patch + '*')):
            if path.is_file() and path not in (OUTPUT, REPORT) and path.suffix in ('.json', '.xml'):
                e.record(path)
                if path.suffix == '.json':
                    e.load(path)
    e.record(REPORT)


def tests(e, path):
    result = junit(e, path)
    result['skips'] = [dict(classname=case.get('classname', ''), name=case.get('name'),
                            message=case.find('skipped').get('message'),
                            detail=case.find('skipped').text)
                       for case in ET.parse(path).iter('testcase') if case.find('skipped') is not None]
    if path.name.startswith('0323_final_'):
        result.update(deselected=1, deselected_source='User final execution record; absent from JUnit')
    return result


def operator(e, filename):
    data = e.load(DOC / filename)
    provenance = data['provenance']
    embedded = {}
    for group in ('files', 'kernels'):
        for name, item in provenance.get(group, {}).items():
            if isinstance(item, dict) and 'sha256' in item and 'source' in item:
                embedded[group + ':' + name] = dict(
                    historical_sha256=item['sha256'],
                    embedded_source_hash_verified=hashlib.sha256(item['source'].encode()).hexdigest() == item['sha256'],
                    current_source_sha256=(sha256(ROOT / name) if group == 'files' and (ROOT / name).is_file() else None),
                    meaning='Retained execution-source snapshot; never refreshed to current code')
    return dict(artifact=relative(DOC / filename), scope=data['scope'], precision=data['precision'],
                accepted=data['accepted'], value_tile=data.get('value_tile', 0),
                cases=data['cases'], view_checks=data.get('view_checks'),
                remainder_byte_identical=data.get('remainder_byte_identical'),
                historical_sources=embedded,
                timing=data.get('timing'), provenance={k: v for k, v in provenance.items()
                                                     if k not in ('files', 'kernels', 'git_diff', 'git_status')})


def trace(e, patch):
    path = DOC / (patch + '_critical_path_v1.json')
    data = e.load(path)
    count = sum(row['name'].startswith('token/') for row in data['nvtx_wall'])
    launches = sum(row['count'] for row in data['runtime_api'] if row['name'].startswith('cudaGraphLaunch'))
    driver = sum(row['count'] for row in data['runtime_api'] if row['name'] == 'cuLaunchKernel')
    publication = sum(row['total_ms'] for row in data['gaps_by_next_operation'] if row['name'] == 'kernel:publish')
    return dict(artifact=relative(path), tokens=count, gpu_span_ms=data['gpu_span_ms'],
                useful_nonwait_union_ms=data['non_wait_gpu_operation_union_ms'],
                summed_wait_kernel_ms=data['summed_wait_kernel_ms'], graph_launches=launches,
                graph_launches_per_token=launches/count, host_driver_kernel_launches=driver,
                gaps_preceding_publication_ms=publication, additive_exclusive_critical_budget=False,
                caveat=data['caveat'])


def analyze():
    e = Evidence()
    inventory(e)
    snapshot = source_snapshot()
    historical = {}
    for filename in ('0310_0317_results_v1.json', '0310_0317_results_v2.json'):
        data = e.load(DOC / filename)
        old = data['git']['source_sha256']
        historical[filename] = dict(sha256=e.manifest[relative(DOC / filename)]['sha256'],
            retained_source_sha256=old,
            current_comparison={name: snapshot['source_sha256'].get(name) == digest for name, digest in old.items()},
            status='historical_unchanged_not_recertified_or_refreshed')
    for filename in ('0310_0317_strata_comparison.md', 'implementation_status.md',
                     'performance_history.md', 'reference_contract.md', 'tensor_inventory.json'):
        e.record(DOC / filename)
    local = {}
    for patch in PATCHES:
        for path in sorted(DOC.glob('diagnostics_' + patch + '*/result.json')):
            if 'cases' not in e.load(path):
                continue
            item = local_result(e, path)
            raw = e.load(path)
            item.update(status='completed_diagnostic' if patch == '0323' else 'completed',
                        token_graph_statistics=raw.get('token_graph_statistics'),
                        reproduction=e.load(path.parent / 'reproduce.json'),
                        initial_memory=raw['initial_memory'], final_memory=raw['final_memory'])
            for row, case in zip(item['cases'], raw['cases']):
                timing = row['timing']
                wall = timing['decode_wall_seconds_after_first_token']
                if abs(timing[RATE] - timing['decode_forwards']/wall) > 1e-10:
                    raise ValueError('Inconsistent retained rate denominator: ' + str(path))
                row['output_token_ids'] = case['output_token_ids']
            local[path.parent.name.removeprefix('diagnostics_')] = item
    partial_dir = DOC / 'diagnostics_0320_long_v1'
    partial_cases = e.load(partial_dir / 'cases.jsonl')
    partial_log = (partial_dir / 'child.log').read_text()
    if 'need 1861791744, free 1820459008' not in partial_log or (partial_dir / 'result.json').exists():
        raise ValueError('0320 partial failure evidence changed')
    partial = dict(status='partial_not_completed_benchmark', cases=partial_cases,
                   reproduction=e.load(partial_dir / 'reproduce.json'), failure_log=partial_log,
                   classification='second_request_device_capacity_admission_not_arithmetic',
                   needed_bytes=1861791744, free_bytes=1820459008)
    parity = [compare_parity(e, baseline, candidate, scope) for baseline, candidate, scope in (
        ('0317_parity_v1', '0318_publish_parity_v1', 'Existing publication with recurrent/QSA graphs'),
        ('0317_parity_v1', '0318_combined_parity_v1', 'Existing publication with recurrent chain/QSA graphs'),
        ('0318_combined_parity_v1', '0320_parity_v2', 'Integrated exact tile32 QSA'),
        ('0318_combined_parity_v1', '0321_parity_v1', 'Orphaned-request graph resource fix'),
        ('0322_baseline_parity_v1', '0322_parity_v3', 'Same prepared FP32 PLE math, whole-token replay off/on'))]
    if not all(p['exact_fixture_parity'] for p in parity):
        raise ValueError('Expected within-path saved parity no longer holds')
    ple_difference = compare_parity(e, '0318_combined_parity_v1', '0322_baseline_parity_v1',
        'OLD PLE cuBLAS versus prepared FP32 Triton projection; NOT quality accepted; not a graph defect or FP16 change')
    profile_path = DOC / 'diagnostics_0323_profile_v1' / (PROFILE + '.json')
    profile = e.load(profile_path)
    profile = dict(profile, artifact=relative(profile_path),
                   total_calls=sum(row['total_calls'] for row in profile['functions']),
                   primitive_calls=sum(row['primitive_calls'] for row in profile['functions']),
                   summed_self_seconds=sum(row['self_seconds'] for row in profile['functions']),
                   interpretation='Tiny GPU status transfers wait for queued work, not nine seconds of physical PCIe; removing a check moves the wait. Native workers unprofiled; no CPU-utilization inference. Cumulative costs overlap.')
    guards = []
    test_rows = []
    for patch in PATCHES:
        for path in sorted(DOC.glob('diagnostics_' + patch + '*_guard_v*/guard.jsonl')):
            rows = e.load(path)
            guards.append(dict(guard_result(e, path), command=rows[0]['command']))
        test_rows.extend(tests(e, path) for path in sorted(DOC.glob(patch + '*.xml')))
    guard_summary = dict(count=len(guards), reserve_bytes=12*1024**3,
        all_reserves_preserved=all(g['reserve_preserved'] for g in guards),
        minimum_host_available_bytes=min(g['minimum_host_available_bytes'] for g in guards),
        maximum_charged_peak_bytes=max(g['peak_charged_bytes'] for g in guards),
        max_psi_some_avg10=max(g['max_psi_some_avg10'] for g in guards),
        max_psi_full_avg10=max(g['max_psi_full_avg10'] for g in guards),
        memory_events_max={key: max(g['memory_events_max'].get(key, 0) for g in guards)
                           for key in ('high', 'max', 'oom', 'oom_kill')},
        swap_max_bytes=0, swap_evidence='Unchanged 0290 isolator verified before exec; user sequential execution record',
        caps_note='Model caps tightened from high/max 32/36 to 30/34 then 28/32 GiB as editor RSS grew; reserve/VRAM/cache guards not weakened')
    rates = {name: [c['timing'][RATE] for c in item['cases'] if c['request_id'] != 'warmup']
             for name, item in local.items() if item['status'] == 'completed'}
    missing = []
    for guard_name, run_name in (('0320_parity_guard_v1', '0320_parity_v1'),
                                 ('0322_parity_guard_v2', '0322_parity_v2')):
        paths = [DOC / ('diagnostics_' + n) for n in (guard_name, run_name)]
        missing.append(dict(requested_guard=guard_name, requested_run=run_name,
                            any_artifact_directory_exists=any(p.exists() for p in paths),
                            results=None, evidence='User execution record: RAM preflight refused before launch/directory creation; absence is not an invented result'))
    result = dict(schema_version=1, analysis_version=1,
        scope='Retained artifact analysis ONLY; stdlib, no model/backend/GPU/network calls',
        status='implemented_scope_validated_not_model_performance_optimized',
        target_decode_forwards_per_second=50, target_met=False,
        best_new_controlled_warm_forward_rate=max(max(r) for r in rates.values()),
        rate_contract='359 forced output IDs, 358 post-prefill forwards; rate=358/decode wall seconds after first sampler; warmup 128 IDs. Full output head/sampling; not natural throughput or standard-v1 aggregate.',
        comparison_limits=['Cold/warm I/O and cgroup file-page charge differ; small gains not robust',
            'Strata/local KV precision, placement and cache capacities differ; see historical 0310 report',
            'Parity fixture has no GPU admissions, complete CPU/Q8 staged expert split; not independent full-reference quality',
            'Full training suite not run; no terminal 50/s claim'],
        patch_ledger=LEDGER, local=local, partial_0320_long=partial,
        matched_response_rates=rates, parity=parity, old_ple_vs_prepared=ple_difference,
        prepared_ple_quality_accepted_vs_old_path=False,
        token_graph_parity_statistics=e.run('0322_parity_v3')['token_graph_statistics'],
        operators={name: operator(e, name) for name in ('0319_qsa_operator_v1.json',
            '0320_qsa_operator_v1.json', '0320_qsa_tile32_operator_v2.json')},
        traces={patch: trace(e, patch) for patch in ('0313', '0317')},
        profile_0323=profile, junit=test_rows, guards=guards, guard_summary=guard_summary,
        missing_preflight_runs=missing,
        failed_0322_parity_v1=dict(status='failed_capture_no_result',
            result_exists=(DOC/'diagnostics_0322_parity_v1/result.json').exists(),
            evidence='Guard exit 1; user execution record identifies CuPy cuBLAS PLE-key projection capture refusal',
            localized_fix='MINI_LLM_TOKEN_GRAPH_PREP uses existing capturable FP32 Triton matmul for PLE projections; baseline v1 and candidate v3 share it'),
        output_head=dict(storage_format='Q6_K', activation_compute='Q8',
            evidence='tensor_inventory.json tensor_contract.families.output.weight; token-graph eligibility accepts supported resident packed formats, not Q8_0-only'),
        source_identity=dict(current=snapshot,
            final_source_same_as_last_model_parity=True,
            evidence='User final execution record after all production changes; current hashes are a new snapshot, not reconstructed historical proof',
            historical_analyses=historical),
        recommended=dict(whole_token_graph=False, token_graph_prep=False,
            experimental_switch_defaults_changed=False, qsa_staged_integrated=False,
            cpu_spin_us=50, task_factor=3, workers=19, expert_cache_gib=7,
            next_experiment='Rank remaining useful packed-GPU/native-CPU time with matched hot Strata, identical expert placement and CPU counts; retain fences/validation, no speculative new knob'),
        parent_handoff=dict(status_documents_unchanged=['docs/qwen4exp/implementation_status.md',
            'docs/qwen4exp/performance_history.md'],
            note='Parent owns maintained status-doc updates. This task creates ONLY analyzer, JSON and report; historical reports/source-hash verifications remain unchanged.'))
    for patch in LEDGER.values():
        for name in patch['files'] + patch.get('reused_components', []):
            if name not in snapshot['source_sha256']:
                raise ValueError('Missing ledger source hash: ' + name)
    result['artifact_manifest'] = dict(sorted(e.manifest.items()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute existing snapshot read-only')
    args = parser.parse_args()
    if not args.check and OUTPUT.exists():
        raise FileExistsError(f'Refusing overwrite: {OUTPUT}; use --check')
    result = analyze()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.check:
        if json.loads(OUTPUT.read_text()) != json.loads(serialized):
            raise ValueError('Snapshot differs: evidence, report, Git state or source hashes changed; no file written')
        print('PASS: read-only --check matches retained JSON, report, artifacts and source snapshot')
    else:
        with OUTPUT.open('x') as stream:
            stream.write(serialized)
        print('Created ' + relative(OUTPUT))
    print(json.dumps(dict(artifacts=len(result['artifact_manifest']),
        source_files=len(result['source_identity']['current']['source_sha256']),
        modified_or_untracked_sources=result['source_identity']['current']['modified_or_untracked_source_paths'],
        exact_parity_pairs=len(result['parity']), guards=len(result['guards']),
        target_met=result['target_met'], rates=result['matched_response_rates']), indent=2))


if __name__ == '__main__':
    main()
