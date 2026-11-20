#!/usr/bin/env python3
"""Analyze retained 0310..0317 evidence using only the standard library.

No model/backend imports, subprocess benchmarks, CUDA probes, or network access.
Default output is exclusive; --check recomputes and compares without writing.
Source hashes describe the current working tree, not the historical run sources.
"""
from __future__ import annotations

import argparse
from array import array
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/qwen4exp'
OUTPUT = DOC / '0310_0317_results_v2.json'
REVISION = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
POOL_URL = ('https://raw.githubusercontent.com/Niko1221/Strata/' + REVISION +
            '/include/strata/kernels/cpu/pool.hpp')
RATE = 'decode_forward_rate_N_minus_1_per_decode_second'
PREFIXES = (1, 2, 4, 8, 16)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


class Evidence:
    def __init__(self):
        self.manifest = {}
        self.parsed = {}

    def record(self, path):
        key = relative(path)
        if key not in self.manifest:
            self.manifest[key] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        return key

    def load(self, path):
        key = self.record(path)
        if key not in self.parsed:
            if path.suffix == '.jsonl' or path.name == 'ple_timing.json':
                value = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            else:
                value = json.loads(path.read_text())
            self.parsed[key] = value
        return self.parsed[key]

    def run(self, name):
        return self.load(DOC / ('diagnostics_' + name) / 'result.json')

    def inventory(self):
        # Parse all retained structured run artifacts, including failed/partial runs.
        # Large profiler databases/reports are referenced, not replayed or rewritten.
        for directory in sorted(DOC.glob('diagnostics_031[0-7]*')):
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob('*')):
                if not path.is_file():
                    continue
                self.record(path)
                if path.suffix in ('.json', '.jsonl'):
                    self.load(path)
        for path in sorted(DOC.glob('031[0-7]*')):
            if path.is_file() and path != OUTPUT and path.suffix in ('.json', '.xml'):
                self.record(path)
                if path.suffix == '.json':
                    self.load(path)


def junit(evidence, path):
    evidence.record(path)
    root = ET.parse(path).getroot()
    # Count testcases rather than summing nested suite totals twice.
    counts = Counter(tests=0, passed=0, failures=0, errors=0, skipped=0)
    groups = {}
    for case in root.iter('testcase'):
        status = next((name for tag, name in (('error', 'errors'), ('failure', 'failures'),
                      ('skipped', 'skipped')) if case.find(tag) is not None), 'passed')
        counts['tests'] += 1
        counts[status] += 1
        group = groups.setdefault(case.get('classname', ''), Counter())
        group[status] += 1
    return dict(artifact=relative(path), **counts, by_class=groups,
                status='failed_retained' if counts['failures'] or counts['errors'] else 'passed_with_skips' if counts['skipped'] else 'passed')


def npy_fp32(evidence, path):
    """Read numeric FP32 NPY payload only; never unpickle or import NumPy."""
    evidence.record(path)
    with path.open('rb') as stream:
        magic = stream.read(8)
        if magic[:6] != b'\x93NUMPY' or magic[6:] not in (b'\x01\x00', b'\x02\x00'):
            raise ValueError(f'Unsupported NPY version: {path}')
        fmt = '<H' if magic[6] == 1 else '<I'
        length = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]
        if length > 65536:
            raise ValueError(f'Oversized NPY header: {path}')
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        if header['descr'] != '<f4' or header['fortran_order']:
            raise ValueError(f'Expected little-endian C-order FP32: {path}')
        payload = stream.read()
    if len(payload) != 4 * math.prod(header['shape']):
        raise ValueError(f'Invalid NPY payload length: {path}')
    values = array('f')
    values.frombytes(payload)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f'Nonfinite retained logits: {path}')
    return header['shape'], payload, values


def compare_parity(evidence, baseline, candidate, scope):
    a, b = evidence.run(baseline), evidence.run(candidate)
    if a['token_ids'] != b['token_ids'] or len(a['requests']) != 2 or len(b['requests']) != 2:
        raise ValueError('Parity fixture IDs/request count changed')
    rows = []
    for repeat in range(2):
        for prefix in PREFIXES:
            filename = f'logits_{repeat}_{prefix}.npy'
            ap = DOC / ('diagnostics_' + baseline) / filename
            bp = DOC / ('diagnostics_' + candidate) / filename
            ashape, abytes, av = npy_fp32(evidence, ap)
            bshape, bbytes, bv = npy_fp32(evidence, bp)
            if ashape != bshape:
                raise ValueError(f'Logit shape mismatch: {filename}')
            same = abytes == bbytes
            norm2 = math.fsum(x*x for x in av)
            error2 = 0.0 if same else math.fsum((x-y)**2 for x, y in zip(av, bv))
            rows.append(dict(repeat=repeat, prefix=prefix, dtype='float32', shape=ashape,
                bit_exact=same, max_abs=0.0 if same else max(abs(x-y) for x, y in zip(av, bv)),
                relative_l2=math.sqrt(error2/norm2) if norm2 else (0.0 if not error2 else None)))
    states_equal = a['requests'] == b['requests']
    return dict(baseline=baseline, candidate=candidate, scope=scope, logits=rows,
        exact_logit_arrays=sum(r['bit_exact'] for r in rows),
        all_saved_state_hashes_equal=states_equal,
        baseline_state_records=a['requests'], candidate_state_records=b['requests'],
        baseline_routing=a['routing'], candidate_routing=b['routing'],
        exact_fixture_parity=all(r['bit_exact'] for r in rows) and states_equal)


def local_result(evidence, path):
    d = evidence.load(path)
    rows = []
    for ordinal, case in enumerate(d['cases']):
        timing = case['timing']
        wall = timing['decode_wall_seconds_after_first_token']
        cpu = case.get('decode_process_cpu_seconds')
        before, after = case['cache_before'], case['cache_after']
        delta = {key: after['execution'].get(key, 0)-before['execution'].get(key, 0)
                 for key in ('ggml_cpu_experts', 'gpu_experts', 'native_handoff_cpu_ns')}
        rows.append(dict(ordinal=ordinal, request_id=case['request_id'], timing=timing,
            forced_trajectory=case['forced_trajectory'],
            prompt_tokens=len(case['prompt']['token_ids']),
            output_ids_sha256=hashlib.sha256(json.dumps(case['output_token_ids'], separators=(',', ':')).encode()).hexdigest(),
            decode_process_cpu_seconds=cpu,
            active_core_equivalents=None if cpu is None or not wall else cpu/wall,
            response_interval_execution_delta=delta,
            response_interval_scope='cache delta includes prefill, not decode-only',
            resident_experts=after['device'].get('resident_experts'),
            slots_capacity=after.get('slots', {}).get('capacity'), memory=case['memory']))
    graph_keys = ('pre_graph_statistics', 'shared_graph_statistics', 'qsa_block_graph_statistics')
    return dict(artifact=relative(path), cases=rows, options=d['options'],
        cpu_configuration=d['cpu_configuration'], cpu_pool_spin_us=d['cpu_pool_spin_us'],
        packed_gpu_arithmetic=d['packed_gpu_arithmetic'],
        cpu_backend_provenance=d['cpu_backend_provenance'], forced_tokens_source=d.get('forced_tokens_source'),
        historical_comparison_metadata_not_current=d['comparison'],
        ple_row_io=d.get('ple_row_io'),
        graph_statistics={key: d.get(key) for key in graph_keys},
        gpu_memory=dict(final_cupy_pool_used_bytes=d['gpu_pool_used_bytes'],
            final_cupy_pool_reserved_bytes=d['gpu_pool_reserved_bytes'],
            capture_peak_live_gpu_bytes=None,
            limitation='Capture-time live peak was not sampled in retained artifacts; final CuPy pool is not NVML or a peak.',
            planned_device_reserved_bytes=d['plan']['device_reserved_bytes'],
            planned_graph_workspace_bytes=d['plan']['graph_workspace_bytes']),
        sampled_host_highwater_bytes=d.get('sampled_highwater_bytes'))


def guard_result(evidence, path):
    rows = evidence.load(path)
    samples = [row for row in rows if 'memory_peak' in row]
    first = rows[0]
    event_names = sorted({key for row in samples for key in row['memory_events']})
    return dict(artifact=relative(path), samples=len(samples),
        peak_charged_bytes=max(row['memory_peak'] for row in samples),
        minimum_host_available_bytes=min(row['host_available'] for row in samples),
        max_psi_some_avg10=max(row['psi_some_avg10'] for row in samples),
        max_psi_full_avg10=max(row['psi_full_avg10'] for row in samples),
        memory_events_max={key: max(row['memory_events'].get(key, 0) for row in samples) for key in event_names},
        high_bytes=first.get('high'), maximum_bytes=first.get('maximum'), reserve_bytes=first.get('reserve'),
        reserve_preserved=all(row['host_available'] >= first['reserve'] for row in samples),
        last_event=rows[-1])


def git_sources(evidence):
    def git(*args):
        return subprocess.check_output(['git', '--no-pager', *args], cwd=ROOT, text=True).splitlines()
    names = set(git('diff', '--name-only') + git('diff', '--cached', '--name-only') +
                git('ls-files', '--others', '--exclude-standard', 'mini_llm', 'scripts', 'tests'))
    names.update(('scripts/analyze_0310_0317_results.py', 'strata_launch_bench_capture.sh'))
    sources = {name: sha256(ROOT/name) for name in sorted(names)
               if (ROOT/name).is_file() and Path(name).suffix in ('.py', '.cpp', '.h', '.hpp', '.sh')}
    return dict(head=git('rev-parse', 'HEAD')[0], source_sha256=sources,
        caveat='Current working-tree hashes only; not proof of source identity at historical execution. No giant untracked artifact status included.')


def analyze():
    e = Evidence()
    e.inventory()
    locals_ = {path.parent.name.removeprefix('diagnostics_'): local_result(e, path)
               for path in sorted(DOC.glob('diagnostics_031[1-7]*/result.json'))
               if 'cases' in e.load(path)}
    strata = {}
    for name in ('0310_strata_v1', '0310_strata_v2', '0310_strata_profile_v3', '0310_strata_hot_v4'):
        d = e.run(name)
        provenance = e.load(DOC / ('diagnostics_' + name) / 'provenance.json')
        rows = [{key: row.get(key) for key in ('request_id', 'validated', 'engine', 'rates', 'ttft_observed_seconds')}
                for row in d['requests']]
        # Do not promote the stored aggregate into a performance claim.
        strata[name] = dict(ok=d['ok'], error=d.get('error'), cleanup=d['cleanup'],
            requests=rows, provenance=provenance,
            aggregate_rate_claim=False, timing_note=d['timing_note'])
    parity = [compare_parity(e, base, candidate, scope) for base, candidate, scope in (
        ('0296_parity_uncaptured_v1', '0313_parity_v1', 'Full checkpoint, frozen empty GPU expert cache, complete CPU-expert fixture'),
        ('0313_parity_v1', '0314_parity_v1', 'Changed cooperative FP32 reduction order; NOT quality accepted'),
        ('0313_parity_v1', '0315_parity_v1', 'Same arithmetic recurrent-chain fixture'),
        ('0317_baseline_parity_v1', '0317_parity_v1', 'SAME capture-ready projection math; replay off versus on, not legacy 0313 QSA math'))]
    tests = [junit(e, path) for path in sorted(DOC.glob('031[0-7]*.xml'))]
    final_cpu = next(test for test in tests if Path(test['artifact']).name == '0317_final_cpu_v1.xml')
    for test in tests:
        if Path(test['artifact']).name in ('0317_final_cpu_v1.xml', '0317_final_gpu_v1.xml'):
            test.update(deselected=1,
                deselected_source='User final-test execution record; JUnit does not record deselections')
    guards = [guard_result(e, path) for path in sorted(DOC.glob('diagnostics_031[0-7]*_guard_v*/guard.jsonl'))]
    ple = {name: e.load(DOC / ('diagnostics_' + name) / 'ple_timing.json')
           for name in ('0311_local_v2', '0312_profile_v1')}
    trace = e.load(DOC/'0313_critical_path_v1.json')
    token_count = len([row for row in trace['nvtx_wall'] if row['name'].startswith('token/')])
    graph_launches = sum(row['count'] for row in trace['runtime_api'] if row['name'].startswith('cudaGraphLaunch'))
    trace_summary = dict(artifact='docs/qwen4exp/0313_critical_path_v1.json', tokens=token_count,
        gpu_span_ms=trace['gpu_span_ms'], useful_nonwait_union_ms=trace['non_wait_gpu_operation_union_ms'],
        summed_wait_kernel_ms=trace['summed_wait_kernel_ms'],
        gpu_span_ms_per_token=trace['gpu_span_ms']/token_count,
        useful_ms_per_token=trace['non_wait_gpu_operation_union_ms']/token_count,
        wait_ms_per_token=trace['summed_wait_kernel_ms']/token_count,
        graph_launches=graph_launches, graph_launches_per_token=graph_launches/token_count,
        caveat=trace['caveat'], additive_exclusive_breakdown=False,
        producer_hypothesis='Host/producer gaps remain a hypothesis, not proved recoverable; chain and QSA capture showed no hot gain.')
    matched = {}
    for name in ('0312_control_v1', '0312_candidate_v1', '0313_fullblock_v1', '0314_long_v1', '0315_chain_v1', '0317_long_v1'):
        matched[name] = [case['timing'][RATE] for case in locals_[name]['cases'] if case['request_id'] != 'warmup']
    result = dict(schema_version=1, analysis_version=2,
        previous_analysis=dict(artifact='docs/qwen4exp/0310_0317_results_v1.json',
            sha256=sha256(DOC/'0310_0317_results_v1.json'), status='retained_historical',
            source_hash_note='V1 remains unchanged; updating this analyzer changes its current source hash, not historical evidence.'),
        scope='Retained artifact analysis only; no new model/GPU runs',
        target_decode_forwards_per_second=50, target_met=False, full_configuration_equivalence=False,
        comparison_contract=dict(routing='complete top10, no drafting, context1024 for fresh harness/local subset',
            kv='Strata INT8 versus local FP32', cpu_workers=19,
            strata_effective_cache_mib=7407, strata_effective_slots=3786,
            local_cache_bytes=7516192768, local_slots=3673,
            limitations=['Different placement and dense/projection arithmetic', 'Subset, forced trajectory, not standard-v1 aggregate or sampled-output quality',
                        'Historical ~50 emitted tokens/s was not fully equivalent', 'Local comparison metadata is historical; use effective startup/options instead']),
        strata=strata, local=locals_, matched_response_rates=matched, junit=tests,
        focused_counts_confirmed_by_final_cpu_junit=dict(source=final_cpu['artifact'],
            strata_0310=final_cpu['by_class']['tests.test_0310_strata_comparison']['passed'],
            ple_diagnostic_0311=final_cpu['by_class']['tests.test_0311_local_ple']['passed']),
        parity=parity, ple_diagnostics=ple, qsa_operator=e.load(DOC/'0314_qsa_operator_v1.json'),
        cooperative_model_quality_accepted=False, trace_0313=trace_summary, guards=guards,
        reference=dict(strata_revision=REVISION, pool_url=POOL_URL,
            pool_source_fetch_confirmed=True, fetch_provenance='Read-only fetch confirmed during report preparation; analyzer performs no network request',
            strata_spin_us=20000, local_default_spin_us=50,
            scheduler_defaults_changed=False, launch_reference='strata_launch_bench_capture.sh',
            launch_reference_executed_unchanged=False,
            launch_reference_limit='Leaves server running, historical ~43 GiB pinned-memory risk, default new drafter may require missing download'),
        user_diagnostic_notes=dict(source='User-supplied live observation, not reconstructed thread traces',
            cpu_utilization='Strata 100% versus local 30-50%; utilization alone is not useful work',
            affinity_v1='Both host threads inherit CPU5', affinity_v2='Initialize service thread before host pin; diagnostic only'),
        git=git_sources(e))
    result['artifact_manifest'] = dict(sorted(e.manifest.items()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute and compare existing JSON without writing')
    args = parser.parse_args()
    if not args.check and OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUTPUT}; use --check for read-only validation')
    result = analyze()
    # Round-trip normalizes tuple-shaped NPY metadata before equality checks.
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.check:
        if json.loads(OUTPUT.read_text()) != json.loads(serialized):
            raise ValueError('Analysis differs: inputs or current source hashes changed; retained output was not modified')
        print('PASS: read-only reproduction matches retained analysis')
    else:
        with OUTPUT.open('x') as stream:
            stream.write(serialized)
        print(f'Created {relative(OUTPUT)}')
    print(json.dumps(dict(target_met=result['target_met'], rates=result['matched_response_rates'],
        parity=[dict(candidate=p['candidate'], exact_logits=p['exact_logit_arrays'],
                     exact_states=p['all_saved_state_hashes_equal']) for p in result['parity']],
        artifacts=len(result['artifact_manifest']), junit_files=len(result['junit'])), indent=2))


if __name__ == '__main__':
    main()
