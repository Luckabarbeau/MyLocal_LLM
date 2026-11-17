"""Bounded independent linked llama.cpp comparison; production files are read-only.

Normal mode requires --model and --llama-source. Artifacts live in a temporary
folder and are removed. No downloads, source builds, hashing, or text-parity claims.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIN = "171e8846b4af9766c354064cb776cb34a50f053f"
STAGES = ("hc_attention_read", "post_mixer", "hc_ffn_read", "post_moe")


def metrics(actual, reference):
    """Descriptive FP64 reductions, not an asserted numerical tolerance."""
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or not a.size:
        raise ValueError(f"comparison shape mismatch: {a.shape} vs {r.shape}")
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite comparison input")
    delta = a - r
    norm = float(np.linalg.norm(r))
    return dict(max_absolute_error=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / max(norm, 1e-30)),
                reference_l2=norm, count=int(a.size))


def logits_metrics(actual, reference):
    result = metrics(actual, reference)
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    def logprob(x):
        shifted = x - x.max()
        return shifted - np.log(np.exp(shifted).sum())
    la, lr = logprob(a), logprob(r)
    ai, ri = np.argsort(-a, kind="stable")[:10], np.argsort(-r, kind="stable")[:10]
    result.update(actual_argmax=int(ai[0]), reference_argmax=int(ri[0]),
                  actual_top10=ai.tolist(), reference_top10=ri.tolist(),
                  top10_overlap=len(set(ai) & set(ri)),
                  logprob_max_absolute_error=float(np.max(np.abs(la - lr))),
                  reference_top10_logprob_differences=[
                      dict(id=int(i), actual=float(la[i]), reference=float(lr[i]),
                           difference=float(la[i] - lr[i])) for i in ri])
    return result


def child_model(model_path, output, greedy_count):
    sys.path.insert(0, str(ROOT))
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    position = 0
    def diagnostic(layer, stage, value):
        if stage != "logits":
            np.save(output / f"p{position}_l{layer}_{stage}.npy", np.asarray(value, dtype=np.float32))
    start = time.monotonic()
    with Qwen4ExpGGUFModel.open(model_path, max_length=32, cpu_workers=1,
                               diagnostic_callback=diagnostic) as model:
        state = model.create_generation_state(1, 32)
        greedy = []
        token = 1
        for position in range(2 + max(0, greedy_count - 1)):
            if position == 1:
                token = 2
            logits = np.asarray(model.step(token, state)[0], dtype=np.float32)
            np.save(output / f"logits_{position}.npy", logits)
            token = int(np.argmax(logits))
            if position >= 1 and greedy_count:
                greedy.append(token)
        stats = dict(greedy_ids=greedy, memory_plan=model.plan,
                     elapsed_seconds=time.monotonic() - start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        (output / "stats.json").write_text(json.dumps(stats, indent=2))


def limit_child():
    # Virtual allowance includes mmap files, shared libraries and allocator arenas;
    # it is not a resident-memory reservation. Parent separately samples RSS.
    resource.setrlimit(resource.RLIMIT_AS, (112 * 1024**3, 112 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def bounded(command, deadline, directory, label, env, rss_limit=12 * 1024**3):
    log = directory / f"{label}.log"
    start = time.monotonic()
    peak = 0
    reason = None
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, preexec_fn=limit_child)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    reason = "overall 120-second execution deadline reached"
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    rss = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                                if line.startswith("VmRSS:")), 0)
                    peak = max(peak, rss)
                    if rss > rss_limit:
                        reason = f"sampled RSS exceeded {rss_limit} bytes"
                except FileNotFoundError:
                    pass
                if reason:
                    process.kill()
                    break
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    text = log.read_text(errors="replace")
    result = dict(command=[str(x) for x in command], returncode=process.returncode,
                  seconds=time.monotonic() - start, sampled_peak_rss_bytes=peak,
                  stop_reason=reason, log_tail=text[-10000:])
    if reason or process.returncode:
        raise RunFailure(label, result)
    return result


class RunFailure(RuntimeError):
    def __init__(self, label, result):
        super().__init__(f"{label} failed")
        self.result = result


def compare(ours, reference, greedy_count):
    results = []
    positions = 2 + max(0, greedy_count - 1)
    for p in range(positions):
        a = np.load(ours / f"logits_{p}.npy", allow_pickle=False)
        r = np.fromfile(reference / f"logits_{p}.f32", dtype=np.float32)
        results.append(dict(position=p, same_prefix=p < 2, **logits_metrics(a, r)))
    boundaries, missing = [], []
    for path in sorted(ours.glob("p*_l*_*.npy")):
        target = reference / (path.stem + ".f32")
        if not target.exists():
            missing.append(path.stem)
            continue
        a = np.load(path, allow_pickle=False).reshape(-1)
        r = np.fromfile(target, dtype=np.float32)
        boundaries.append(dict(boundary=path.stem, **metrics(a, r)))
    def order(record):
        pos, layer, stage = record["boundary"].split("_", 2)
        il = int(layer[1:])
        return (int(pos[1:]), il if il >= 0 else 10**6,
                STAGES.index(stage) if stage in STAGES else 4)
    boundaries.sort(key=order)
    first = next((r for r in boundaries if r["max_absolute_error"] > 0), None)
    return dict(logits=results, boundaries=boundaries, missing_reference_boundaries=missing,
                first_observed_nonzero_boundary_difference=first,
                first_difference_is_not_a_defect_claim=True,
                actual=json.loads((ours / "stats.json").read_text()),
                reference=json.loads((reference / "stats.json").read_text()))


def self_test():
    a = np.array([1, 2, 4], dtype=np.float32)
    assert metrics(a, a)["max_absolute_error"] == 0
    assert metrics(a + 1, a)["max_absolute_error"] == 1
    assert np.isclose(metrics(a + 1, a)["relative_l2"], np.sqrt(3 / 21))
    assert logits_metrics(a + 2, a)["logprob_max_absolute_error"] < 1e-14
    with tempfile.TemporaryDirectory(prefix="qwen4exp-metrics-") as folder:
        base = Path(folder)
        ours, ref = base / "ours", base / "ref"
        ours.mkdir()
        ref.mkdir()
        for p in range(2):
            np.save(ours / f"logits_{p}.npy", a)
            a.tofile(ref / f"logits_{p}.f32")
            np.save(ours / f"p{p}_l0_hc_attention_read.npy", a)
            a.tofile(ref / f"p{p}_l0_hc_attention_read.f32")
        for path in (ours, ref):
            (path / "stats.json").write_text('{"greedy_ids": []}')
        result = compare(ours, ref, 0)
        assert len(result["logits"]) == 2 and len(result["boundaries"]) == 2
        assert result["first_observed_nonzero_boundary_difference"] is None
    print("PASS: descriptive metrics and synthetic artifact comparison (not model parity)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--llama-source", type=Path)
    parser.add_argument("--llama-build", type=Path,
                        help="separate CPU no-prefetch build; original --llama-source stays clean")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--greedy-count", type=int, choices=range(4), default=0,
                        help="optional 1..3 greedy IDs after fixed [1,2]; default fixed inputs only")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.model or not args.model.is_file():
        parser.error("--model must explicitly identify the existing first GGUF shard")
    if args.child_output:
        child_model(args.model, args.child_output, args.greedy_count)
        return 0
    if not args.llama_source:
        parser.error("--llama-source is required")
    source = args.llama_source.resolve()
    model = args.model.resolve()
    shards = sorted(model.parent.glob(model.name.replace("00001-of-00002", "*-of-00002")))
    before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in shards}
    env = dict(os.environ, MINI_LLM_BACKEND="numpy", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    report = dict(scope="independent linked llama.cpp full-model diagnostic attempt",
                  model=str(model), expected_source_sha=PIN, fixed_token_ids=[1, 2],
                  context=32, sequences=1, dtype="float32", kv_dtype="float32",
                  greedy_count=args.greedy_count, actual_model_ran=False,
                  reference_model_ran=False, numerical_comparison_completed=False,
                  numerical_parity_claim=False, text_correctness_claim=False,
                  sequential_separate_processes=True, full_expert_dequantization=False,
                  all_expert_hashing=False, production_files_modified=False,
                  wall_limit_seconds=120, sampled_child_rss_limit_bytes=12 * 1024**3,
                  virtual_child_limit_bytes=112 * 1024**3,
                  arithmetic_caveat="Native experts use FP32 activations; GGML CPU quantized matmuls may use Q8_K/Q8_0 activation quantization. No equality or broad acceptance tolerance is assumed.")
    start = time.monotonic()
    deadline = start + 120
    with tempfile.TemporaryDirectory(prefix="qwen4exp-reference-") as folder:
        work = Path(folder)
        try:
            report["source_revision"] = bounded(
                ["git", "--no-pager", "-C", str(source), "rev-parse", "HEAD"],
                deadline, work, "revision", env)
            sha = report["source_revision"]["log_tail"].strip()
            if sha != PIN:
                raise ValueError(f"source SHA mismatch: {sha}")
            report["source_status"] = bounded(
                ["git", "--no-optional-locks", "-C", str(source), "status", "--porcelain"],
                deadline, work, "source-status", env)
            if report["source_status"]["log_tail"].strip():
                raise ValueError("pinned reference source is not clean")
            verified_no_prefetch = False
            driver = ROOT / "tests/fixtures/qwen4exp_reference.cpp"
            if args.llama_build:
                build = args.llama_build.resolve()
                copied = build.parent
                old = 'ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);'
                new = old.replace('(true,', '(false,')
                base_model = (source / 'src/llama-model.cpp').read_text()
                if base_model.count(old) != 1:
                    raise ValueError('unexpected pinned loader call')
                expected_model = base_model.replace(old, new)
                def source_files(directory):
                    files = {}
                    for current, dirs, names in os.walk(directory):
                        dirs[:] = [d for d in dirs if d != '.git' and
                                   not d.startswith('build') and d != '__pycache__']
                        for name in names:
                            path = Path(current) / name
                            files[str(path.relative_to(directory))] = path
                    return files
                original_files, copied_files = source_files(source), source_files(copied)
                if original_files.keys() != copied_files.keys():
                    raise ValueError('reference copy source file inventory mismatch')
                differences = []
                for relative, original in original_files.items():
                    if original.read_bytes() != copied_files[relative].read_bytes():
                        differences.append(relative)
                if differences != ['src/llama-model.cpp'] or (
                        copied / 'src/llama-model.cpp').read_text() != expected_model:
                    raise ValueError(f'reference must contain exactly the loader argument patch: {differences}')
                patch = ''.join(difflib.unified_diff(
                    base_model.splitlines(True), expected_model.splitlines(True),
                    fromfile='a/src/llama-model.cpp', tofile='b/src/llama-model.cpp'))
                artifact = ROOT / 'docs/qwen4exp/0210_noprefetch_reference.patch'
                if artifact.read_text() != patch:
                    raise ValueError('saved unified patch differs from verified source patch')
                loader = (copied / 'src/llama-model-loader.cpp').read_text()
                mmap = (copied / 'src/llama-mmap.cpp').read_text()
                if ('prefetch && use_mmap ? -1 : 0' not in loader or
                        'if (prefetch && lazy_ranges.empty()) { flags |= MAP_POPULATE; }' not in mmap or
                        'if (prefetch > 0)' not in mmap):
                    raise ValueError('cannot positively verify pinned zero-prefetch mmap flow')
                cache = (build / 'CMakeCache.txt').read_text()
                required = ['BUILD_SHARED_LIBS:BOOL=ON', 'CMAKE_BUILD_TYPE:STRING=Release']
                required += [f'{key}:BOOL=OFF' for key in (
                    'GGML_CUDA', 'GGML_HIP', 'GGML_VULKAN', 'GGML_METAL', 'GGML_SYCL',
                    'GGML_OPENCL', 'GGML_RPC', 'GGML_BLAS', 'LLAMA_BUILD_COMMON',
                    'LLAMA_BUILD_TESTS', 'LLAMA_BUILD_TOOLS', 'LLAMA_BUILD_EXAMPLES',
                    'LLAMA_BUILD_SERVER', 'LLAMA_BUILD_APP', 'LLAMA_BUILD_MTMD')]
                if any(line not in cache.splitlines() for line in required):
                    raise ValueError('CPU-only shared build configuration not verified')
                lib = build / 'bin'
                if (lib / 'libllama.so').stat().st_mtime_ns < (
                        copied / 'src/llama-model.cpp').stat().st_mtime_ns:
                    raise ValueError('reference library predates patched loader')
                verified_no_prefetch = True
                report['no_prefetch_verification'] = dict(
                    base_sha=sha, source_copy=str(copied), build=str(build),
                    exact_unified_diff=patch, changed_source_files=differences,
                    source_files_compared=len(original_files), license_preserved=True,
                    cpu_only_cache_verified=True, zero_prefetch_control_flow_verified=True)
                # Keep the repository harness frozen. Only its temporary driver's
                # obsolete NUMA-only guard changes after the source/build checks.
                driver_text = driver.read_text()
                guard = 'if (!demand_paging) {'
                if driver_text.count(guard) != 1:
                    raise ValueError('unexpected harness guard')
                adapted = driver_text.replace(guard, 'if (!demand_paging && !true /* verified loader prefetch=false */) {')
                report['temporary_driver_diff'] = ''.join(difflib.unified_diff(
                    driver_text.splitlines(True), adapted.splitlines(True),
                    fromfile='repository-harness', tofile='temporary-verified-no-prefetch-driver'))
                driver = work / 'verified_reference.cpp'
                driver.write_text(adapted)
            else:
                lib = source / "build-cuda/bin"
                report["build_version"] = bounded([str(lib / "llama-cli"), "--version"],
                                                  deadline, work, "version", env)
                if PIN[:9] not in report["build_version"]["log_tail"]:
                    raise ValueError("existing build revision does not match source")
            harness = work / "reference"
            report["compile"] = bounded(
                ["c++", "-std=c++17", "-O2", str(driver),
                 "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
                 "-L" + str(lib), "-Wl,-rpath," + str(lib),
                 "-lllama", "-lggml", "-lggml-base", "-lggml-cpu", "-o", str(harness)],
                deadline, work, "compile", env)
            if verified_no_prefetch:
                report['linked_libraries'] = bounded(['ldd', str(harness)],
                                                      deadline, work, 'linked-libraries', env)
                links = report['linked_libraries']['log_tail']
                for name in ('libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so'):
                    line = next((line for line in links.splitlines() if name in line), '')
                    if str(lib) not in line:
                        raise ValueError(f'wrong independent reference library resolution: {name}')
                if 'cuda' in links.lower() or 'libhip' in links.lower():
                    raise ValueError('unexpected GPU library dependency')
            report["capabilities_run"] = bounded([str(harness), "--capabilities"],
                                                 deadline, work, "capabilities", env)
            lines = report["capabilities_run"]["log_tail"].splitlines()
            caps = next(json.loads(line) for line in lines if line.startswith('{"q2_0_type"'))
            report["capabilities"] = caps
            if caps["q2_0_type"] != 42:
                raise ValueError("Q2_0 type mismatch")
            if not caps["numa_disables_prefetch"] and not verified_no_prefetch:
                report["status"] = "blocked_memory_safety_before_model_load"
                report["blocker"] = (
                    "Pinned llama-model.cpp calls init_mappings(true). On this single-NUMA-node "
                    "host ggml_is_numa() is false, so llama-mmap.cpp requests MAP_POPULATE on "
                    "shards without lazy ranges and MADV_WILLNEED on all non-lazy ranges. "
                    "Packed expert banks are non-lazy. No public no-prefetch parameter exists "
                    "in this pinned llama_model_params. Refusing to fault in the expert banks.")
                report["next_experiment"] = (
                    "With separate approval, use a separately built reference exposing a "
                    "no-prefetch loader option; first verify CPU mmap/lazy PLE and bounded RSS, "
                    "then compare the fixed [1,2] logits. Do not change the frozen production model.")
            else:
                ref, ours = work / "ref", work / "ours"
                ref.mkdir()
                ours.mkdir()
                report["reference_run"] = bounded(
                    [str(harness), str(model), str(ref), "1", str(args.greedy_count)],
                    deadline, work, "reference", env)
                report["reference_model_ran"] = True
                report["actual_run"] = bounded(
                    [sys.executable, str(Path(__file__).resolve()), "--model", str(model),
                     "--child-output", str(ours), "--greedy-count", str(args.greedy_count)],
                    deadline, work, "actual", env)
                report["actual_model_ran"] = True
                report["comparison"] = compare(ours, ref, args.greedy_count)
                report["numerical_comparison_completed"] = True
                report["status"] = "diagnostics_completed_no_acceptance_threshold"
        except RunFailure as error:
            report.update(status="blocked_child_failure", blocker=str(error), failed_run=error.result)
        except (OSError, ValueError, StopIteration) as error:
            report.update(status="blocked_preflight_or_artifacts", blocker=str(error))
    report["checkpoint_size_mtime_unchanged"] = before == {
        str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in shards}
    report["checkpoint_file_stats"] = before
    report["elapsed_seconds"] = time.monotonic() - start
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.result:
        permitted = {ROOT / 'docs/qwen4exp/0210_reference_validation.json',
                     ROOT / 'docs/qwen4exp/0210_noprefetch_reference_validation.json'}
        if args.result.resolve() not in permitted:
            parser.error('--result must be one of the two scoped 0210 reference reports')
        if args.result.exists():
            parser.error("refusing to overwrite an existing result; preserve the previous run")
        args.result.write_text(text)
    print(text, end="")
    return 0 if report["numerical_comparison_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
