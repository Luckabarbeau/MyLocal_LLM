"""Bounded provenance/QSA evidence only; never load a checkpoint or launch Strata."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/native_engine'
os.chdir(ROOT)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def identity(path):
    p = Path(path)
    if not p.exists():
        return dict(path=str(p), exists=False)
    s = p.stat()
    return dict(path=str(p), exists=True, resolved=str(p.resolve()), size=s.st_size,
                mtime_ns=s.st_mtime_ns, sha256=sha(p))

def query(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return dict(argv=argv, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(argv=argv, error=str(e))

inventory = json.loads((OUT / 'WP2_GGUF_INVENTORY.json').read_text())
manifest = dict(schema_version=1, measured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    wp0_status='NOT COMPLETE', specification_origin='User-supplied WP0 specification; not missing requirements',
    wp0_requirements=['reproducible Strata config/version', 'exact checkpoint hashes',
                      'frozen local execution config', 'numerical fixtures', 'throughput/resource baseline'],
    scope='Current identities and bounded synthetic CPU QSA evidence; no full-model execution or performance baseline',
    git_head=query(['git', '--no-pager', 'rev-parse', 'HEAD']),
    git_status_before_capture=query(['git', '--no-optional-locks', 'status', '--short']),
    numerical_policy=dict(name='reference_fp32', implementation='current local Python',
        authority='docs/qwen4exp/reference_contract.md', backend='numpy', activation_dtype='float32',
        persistent_qsa_dtype='float32', stored_quantization='unchanged; no checkpoint decoded by this capture',
        qsa_fixture_tie_policy='strata: lower-index ties',
        production_default_tie_policy='ggml_cpu; fixture explicitly selects strata, not a global policy change',
        independent_oracle_dtype='float64',
        distinction='Not Strata native INT8 KV / BF16 HC prompt arithmetic; no independent full-model parity claim'),
    source_files={}, build={}, checkpoint_shards=[])
paths = set(inventory['source_sha256'])
paths.update(str(p.relative_to(ROOT)) for p in (ROOT / 'mini_llm/external').glob('qwen4exp*.py'))
paths.update(['mini_llm/backend.py', 'tests/test_0208_qwen4exp_attention.py',
              'scripts/probe_0310_strata_comparison.py', 'scripts/run_0290_isolated_benchmark.py',
              'strata_launch_bench_capture.sh', 'docs/qwen4exp/reference_contract.md',
              'docs/native_engine/WP2_GGUF_INVENTORY.json', 'docs/native_engine/capture_wp0_reference.py'])
for path in sorted(paths):
    manifest['source_files'][path] = identity(path)
build = Path('/tmp/mini-llm-wp1-build')
manifest['build']['files'] = [identity(build / p) for p in ['libmini_llm_native.so', 'CMakeCache.txt', 'build.ninja', 'CMakeFiles/rules.ninja']]
manifest['build']['meaning'] = 'Existing build measured, not rebuilt; current hashes do not retroactively identify historical runs'
for filename in ['CMakeCache.txt', 'build.ninja', 'CMakeFiles/rules.ninja']:
    p = build / filename
    if p.exists():
        (OUT / ('WP0_' + filename.replace('/', '_') + '.txt')).write_text(p.read_text())
manifest['toolchains'] = [dict(identity=identity(p), version=query([p, '--version'])) for p in
                          ['/usr/bin/c++', '/usr/bin/cmake', '/usr/local/cuda/bin/nvcc']]
manifest['python'] = dict(executable=identity(sys.executable), version=sys.version,
    packages={p: importlib.metadata.version(p) for p in ['numpy', 'pytest', 'gguf']})
import numpy as np
import io
import contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    np.show_config()
manifest['python']['numpy_build_config'] = buf.getvalue()
engine = '/home/lucka/.local/share/strata/components/engine/0.1.59-linux-x86_64-cuda-cuda13/strata'
config = Path('/home/lucka/.local/share/strata/install/strata-iq3_s.json')
manifest['strata'] = dict(engine=identity(engine), installed_config=identity(config),
    declared_source_commit='712fec6b4297c9cfd32ab43daaa2979a9aa01edf',
    version_label='0.1.59 (installation path; engine not executed)',
    compiler_identity='unknown for shipped engine; local nvcc is not shipped compiler evidence')
if config.exists():
    cfg = json.loads(config.read_text())
    manifest['strata']['installed_config_contents'] = cfg
    args = cfg.get('args', [])
    manifest['strata']['assets'] = []
    for flag in ['--pack', '--native', '--ple-gguf', '--expert-profile']:
        if flag in args:
            p = Path(args[args.index(flag)+1])
            manifest['strata']['assets'].append(dict(flag=flag, path=str(p), exists=p.exists(),
                size=p.stat().st_size if p.is_file() else None, payload_sha256=None))
for shard in inventory['shards']:
    p = Path(shard['path'])
    row = dict(path=str(p), exists=p.exists(), full_payload_sha256=None,
               immutable_revision=None, limitation='No 83.6 GB payload scan in this bounded capture')
    if p.exists():
        s = p.stat()
        row.update(size=s.st_size, device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
        with p.open('rb') as f:
            row['header_sha256'] = hashlib.sha256(f.read(shard['data_offset'])).hexdigest()
        row['header_bytes_hashed'] = shard['data_offset']
        row['matches_inventory_header'] = row['header_sha256'] == shard['header_sha256']
    manifest['checkpoint_shards'].append(row)
manifest['host_observation'] = query(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used', '--format=csv'])
# Explicit environment, no inherited experimental switches, no GPU access.
env = {k:v for k,v in os.environ.items() if not k.startswith(('MINI_LLM_', 'STRATA_'))}
env.update(MINI_LLM_BACKEND='numpy', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
argv = [str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_0208_qwen4exp_attention.py', '-k',
        'graph_fp64_scores_selection_and_boundaries or ties_finite_tail_and_partial_token_budget',
        '--junitxml=docs/native_engine/WP0_QSA_reference.xml']
def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
start = time.monotonic()
try:
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60, preexec_fn=limits)
    record = dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
except subprocess.TimeoutExpired as e:
    record = dict(argv=argv, timed_out=True, error=str(e))
record.update(elapsed_seconds=time.monotonic()-start, environment_overrides={k:env[k] for k in
    ['MINI_LLM_BACKEND','CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD']},
    limits=dict(address_space_bytes=2*1024**3, cpu_seconds=30, wall_seconds=60),
    scope='Synthetic component numerical test, seed208, 41 tokens, D7; no checkpoint/model allocation',
    tolerances=dict(output_rtol=3e-5, output_atol=1e-5, scores_rtol=8e-6, scores_atol=2e-5, selected_ids='exact'),
    throughput_baseline=False)
(OUT / 'WP0_QSA_reference_run.json').write_text(json.dumps(record, indent=2)+'\n')
manifest['numerical_test'] = record
manifest['throughput_resource_baseline'] = dict(status='not measured', prefill=None, ttft=None,
    decode_forwards_per_second=None, peak_model_ram_bytes=None, peak_model_vram_bytes=None,
    reason='No full model executed; test elapsed time is not model throughput')
manifest['external_blockers'] = [
    'Immutable upstream checkpoint/model-definition revision is not embedded/established; requires authoritative upstream provenance.',
    'Shipped Strata compiler/build flags unavailable in inspected evidence; local compiler identity cannot substitute.',
    'Historical raw production_recovery_20261121 and diagnostics_0335_cpu_ab_v2/result.json artifacts absent; restoration requires their holder, or regenerate new evidence.']
manifest['remaining_implementation_work'] = [
    'Stream-hash both available checkpoint payloads in a separately budgeted IO run; absence of hashes is work, not missing model availability.',
    'Freeze a complete local whole-model execution configuration (all effective switches, placement, worker affinity, cache/KV policy, exact prompt and continuation IDs). This capture freezes only component execution.',
    'Bind exact Strata asset identities and effective launch configuration to a safe isolated workload; existing launcher is not a frozen accepted baseline.',
    'Persist matched-policy numerical fixtures and first-divergence evidence; historical independent full-model parity remains unclosed.',
    'Run repeated interleaved throughput/resource measurements with the existing isolation harness and full-top10/no drafting; do not substitute unlike historical timings.',
    'Finish native graph/composition validation; descriptor/runtime tests are not integrated model acceptance.']
(OUT / 'native_reference_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(record, indent=2))
