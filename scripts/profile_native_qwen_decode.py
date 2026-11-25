#!/usr/bin/env python3
"""Bounded WP9 actual-checkpoint diagnostic. No server or native source changes."""
import argparse
from contextlib import ExitStack, nullcontext
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-e961516f-d77d-813c-98e9-2f5403f71631'
BASE = ROOT / 'benchmarks/native_http_gpu1_new'
REQUEST = BASE / 'http/0000-128-cold_prompt_first_submission/request.json'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity():
    old = json.loads((BASE / 'provenance.json').read_text())['files']
    files = []
    for item in old:
        p = Path(item['path'])
        stat = p.stat()
        large = p.suffix == '.gguf'
        files.append(dict(path=str(p), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            sha256=item['sha256'] if large else sha(p),
            hash_basis='prior full SHA256; current size/mtime verified' if large else 'current full SHA256',
            matches_baseline=(stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns']
                              and (large or sha(p) == item['sha256']))))
    return files


def worker(args):
    dump(Path(args.output) / (args.worker + '_pid.json'), dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21]))
    def progress(stage):
        print(json.dumps(dict(stage=stage, monotonic=time.monotonic())), flush=True)
    progress('worker_start')
    sys.path.insert(0, str(ROOT))
    from mini_llm.native import Runtime
    from mini_llm.native.decoder import NativeDecoder
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    from mini_llm.external.qwen4exp_chat import render_chat
    provenance = json.loads((BASE / 'provenance.json').read_text())['files']
    model, _, lib, bridge, ggml = [x['path'] for x in provenance[:5]]
    out = Path(args.output)
    result = dict(mode=args.worker, complete_layers=48, capacity=256,
                  steps=args.steps, sampling='native GPU greedy, temperature=0, seed=0',
                  logits='full 248320 vocabulary, device resident', trace_for_test=False)
    with ExitStack() as stack:
        class ProbeRuntime(Runtime):
            def engine(self, **kw):
                self.engine_owner = stack.enter_context(super().engine(**kw))
                return nullcontext(self.engine_owner)
        started = time.perf_counter()
        runtime = ProbeRuntime(args.diagnostic_library if args.worker == 'diagnostic' else lib)
        decoder = stack.enter_context(NativeDecoder.open(runtime, model,
            bridge_library=bridge, ggml_library=ggml, capacity=256, threads=4))
        sampler = stack.enter_context(runtime.engine_owner.sampler(decoder.vocab, seed=0))
        assert decoder.layers == 48
        with MultiShardTensorStore.open(model) as store:
            tokenizer = load_gguf_tokenizer(store.metadata, vocab_size=decoder.vocab)
        request = json.loads(REQUEST.read_text())
        prompt = render_chat(tokenizer, request['messages'], thinking=False)
        ids = list(tokenizer.encode(prompt).ids)
        assert len(ids) == 128
        full_ids = ids
        ids = ids[:args.prompt_tokens]
        result['full_http_prompt_ids'] = full_ids
        result['active_prompt_tokens'] = len(ids)
        progress('setup_complete')
        result.update(setup_seconds=time.perf_counter()-started, memory_plan=decoder.memory_plan,
                      prompt_ids=ids, prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      prompt_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                      request_file_sha256=sha(REQUEST),
                      http_request_sha256=json.loads(REQUEST.with_name('metadata.json').read_text())['request_sha256'])
        def prefix():
            start = time.perf_counter()
            decoder.reset()
            sampler.reset(seed=0)
            assert decoder.consume(ids[:-1]) == len(ids)-1
            return time.perf_counter()-start
        result['warm_prefill_seconds'] = prefix()
        progress('warm_prefill_complete')
        token = ids[-1]
        warm_inputs, warm_outputs = [], []
        for _ in range(args.steps):
            warm_inputs.append(token)
            decoder.step_device(token)
            token = sampler.sample_decoder(decoder, temperature=0.)
            warm_outputs.append(token)
        if args.worker in ('profile', 'diagnostic'):
            baseline = json.loads((out / 'unprofiled.json').read_text())
            assert ids == baseline['prompt_ids']
            assert warm_inputs == baseline['input_ids'], 'fixed trajectory changed'
            inputs = baseline['input_ids']
        else:
            inputs = warm_inputs
        result['reset_replay_seconds'] = prefix()
        progress('reset_replay_complete')
        result['active_history_before'] = decoder.state[0]
        # Only explicit diagnostic mode loads profiler/NVTX. No production overhead.
        cudart = nvtx = None
        if args.worker == 'profile':
            cudart = C.CDLL('/usr/local/cuda/lib64/libcudart.so')
            nvtx_path = Path('/usr/local/cuda/lib64/libnvToolsExt.so')
            if nvtx_path.exists():
                nvtx = C.CDLL(str(nvtx_path))
                nvtx.nvtxRangePushA.argtypes = [C.c_char_p]
            result['nvtx_ranges_available'] = nvtx is not None
            for name in ('cudaProfilerStart', 'cudaProfilerStop'):
                getattr(cudart, name).restype = C.c_int
            progress('before_cudaProfilerStart')
            assert cudart.cudaProfilerStart() == 0
            progress('capture_started')
        if args.worker == 'diagnostic':
            runtime._lib.ml_wp9_diag_start()
            runtime._lib.ml_wp9_diag_stop.restype = C.c_char_p
        walls, samples = [], []
        try:
            for i, token in enumerate(inputs):
                if nvtx:
                    nvtx.nvtxRangePushA(f'WP9_complete_step_{i}_history_{len(ids)-1+i}'.encode())
                start = time.perf_counter_ns()
                decoder.step_device(token)
                sampled = sampler.sample_decoder(decoder, temperature=0.)
                walls.append((time.perf_counter_ns()-start)/1e9)
                if nvtx:
                    nvtx.nvtxRangePop()
                samples.append(sampled)
        finally:
            if args.worker == 'diagnostic':
                result['diagnostic_stage_timers'] = json.loads(runtime._lib.ml_wp9_diag_stop())
            if cudart:
                dump(out / 'profile_steps_pending_stop.json', dict(step_wall_seconds=walls, sampled_ids=samples,
                     input_ids=inputs, active_history_after=decoder.state[0], capture_finalized=False))
                progress('before_cudaProfilerStop')
                assert cudart.cudaProfilerStop() == 0
                progress('capture_stopped')
        assert samples == warm_outputs, 'same-input replay greedy sample mismatch'
        result.update(input_ids=inputs, sampled_ids=samples, step_wall_seconds=walls,
                      median_step_seconds=statistics.median(walls), total_step_seconds=sum(walls),
                      active_history_after=decoder.state[0],
                      measured_scope='step_device + sample_decoder; no HTTP, queue, detokenization or logits D2H',
                      reset_policy=f'literal decoder/sampler reset + {len(ids)-1}-token prefix replay after warmup')
        dump(out / (args.worker + '.json'), result)


def run_owned(command, output, deadline):
    record = dict(command=command, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    start = time.monotonic()
    with output.with_suffix('.log').open('w') as log:
        p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        record['owned_pid'] = p.pid
        try:
            p.wait(timeout=max(.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            record['timeout'] = True
        finally:
            # Nsight may setsid() its target: its own PID/start-time receipt is
            # required before cleaning that separately owned session.
            receipt = output.parent / (output.stem.replace('_launch', '') + '_pid.json')
            if receipt.exists():
                owner = json.loads(receipt.read_text())
                proc = Path('/proc') / str(owner['pid'])
                try:
                    if proc.joinpath('stat').read_text().split()[21] == owner['start_ticks']:
                        os.kill(owner['pid'], signal.SIGKILL)
                        record['detached_worker_cleanup'] = owner
                except ProcessLookupError:
                    pass
                except FileNotFoundError:
                    pass
            # Signal only the session we created, including any profiler child.
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=3)
            # Reap/kill descendants even if the group leader exited first.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    record.update(returncode=p.returncode, seconds=time.monotonic()-start)
    dump(output, record)
    if p.returncode or record.get('timeout'):
        raise RuntimeError(f'Owned command failed: {output}; see log')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, choices=range(4, 9), default=4)
    p.add_argument('--deadline', type=float, default=172)
    p.add_argument('--prompt-tokens', type=int, choices=(16, 128), default=128)
    p.add_argument('--worker', choices=('unprofiled', 'profile', 'diagnostic'))
    p.add_argument('--diagnostic-library', help='Separate timer-only library; never a speed baseline')
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        p.error('Requires physical GPU1 full UUID mask only')
    if not 0 < args.deadline <= 172:
        p.error('deadline must be in (0,172] for outer 180-second bound')
    if args.worker:
        return worker(args)
    deadline = time.monotonic()+args.deadline
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    query = ['nvidia-smi', '-i', GPU]
    gpu = subprocess.check_output(query + ['--query-gpu=index,uuid,name,driver_version,memory.used,memory.free', '--format=csv'], text=True, timeout=5)
    apps = subprocess.check_output(query + ['--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'], text=True, timeout=5)
    dump(out / 'preflight.json', dict(gpu=gpu, compute_apps=apps))
    if apps.strip() or int(gpu.splitlines()[1].split(',')[-2].split()[0]) > 64:
        raise RuntimeError('GPU1 externally occupied; refusing without controlling any existing job')
    before = identity()
    dump(out / 'provenance_before.json', before)
    if not all(x['matches_baseline'] for x in before):
        raise RuntimeError('Baseline source/library/model identity changed; inspect provenance')
    probe = [sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--steps', str(args.steps),
             '--prompt-tokens', str(args.prompt_tokens)]
    try:
        run_owned(probe + ['--worker', 'unprofiled'], out / 'unprofiled_launch.json', deadline)
        if args.diagnostic_library:
            run_owned(probe + ['--worker', 'diagnostic', '--diagnostic-library', str(Path(args.diagnostic_library).resolve())],
                      out / 'diagnostic_launch.json', deadline)
        else:
            nsys = ['/usr/local/cuda/bin/nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                    '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--export=sqlite',
                    '--output=' + str(out / 'decode')]
            run_owned(nsys + probe + ['--worker', 'profile'], out / 'profile_launch.json', deadline)
    finally:
        after = identity()
        dump(out / 'provenance_after.json', after)
        dump(out / 'integrity.json', dict(all_baseline_files_unchanged=before == after,
             parent_status_modified=False, native_core_modified=False))
        dump(out / 'postflight.json', dict(gpu=subprocess.check_output(query + [
            '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv'], text=True, timeout=5),
            compute_apps=subprocess.check_output(query + ['--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                                         '--format=csv,noheader'], text=True, timeout=5)))


if __name__ == '__main__':
    main()
