"""Bounded, serial Strata short-workload replay; no HTTP or configuration parity.

CLI execution requires the separate memory-limited 0290 systemd launcher.
The inner parent also watches time, RssAnon and MemAvailable, NOT total RSS:
these sampled guards alone do not protect the desktop from memory pressure.
No model is loaded by importing this module.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
BENCHMARK = ROOT / 'docs/qwen4exp/diagnostics_0243_off_v1/benchmark'
STRATA_TOKENIZER = Path('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/packs/iq3_s/tokenizer/tokenizer.json')
CASES = ('warmup', 'performance-short-r1', 'performance-short-r2', 'performance-short-r3')
GiB = 1024**3
MODEL_OPTIONS = dict(max_length=1024, cpu_workers=4,
    static_device_budget_bytes=6*GiB, device_budget_bytes=14*GiB,
    cache_bytes=6*GiB, expert_promote_after=1, group_after_admission=True,
    expert_cache_slots=True, qsa_fast_decode=True, hc_fast_decode=True,
    router_fast_decode=True, gdn_fast_decode=True, hc_graph_decode=False,
    hc_native_read=True, elementwise_fast_decode=True)
COMPARISON = dict(full_configuration_equivalence=False,
    capacity='MyLocal 1024 vs Strata 262144; no new KV quantizer',
    kv='MyLocal FP32 vs Strata INT8',
    routing='MyLocal full top10 vs Strata eligible tail miss skip at rank >=7',
    drafting='MyLocal suffix off vs Strata suffix active (model drafter off)',
    host='MyLocal reclaimable mmap vs Strata 43.12 GiB pinned plus file fallback',
    expert_cache='MyLocal explicit 6 GiB vs Strata captured 5.31 GiB / 2768 slots',
    sampling='Same settings, different PRNG/backend; NOT identical sampled sequences',
    timing='Local nonstreaming, no HTTP; wall times include sampler/text/guard/journal overhead',
    gpu_memory='CuPy allocator used/reserved and explicit budgets, NOT NVML/device-wide usage')


def select_records(path, cases):
    if not cases or len(set(cases)) != len(cases) or any(c not in CASES for c in cases):
        raise ValueError('Select unique captured warmup/short request IDs only')
    selected = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            name = row['request_id']
            if name in cases:
                if name in selected:
                    raise ValueError(f'Duplicate captured request: {name}')
                selected[name] = row
    if set(selected) != set(cases):
        raise ValueError(f'Missing captured requests: {set(cases)-set(selected)}')
    rows = [selected[c] for c in cases]
    for row in rows:
        sampling_settings(row)
    return rows


def sampling_settings(row):
    name, request = row['request_id'], row['request']
    seed = 1 if name == 'warmup' else int(name.rsplit('r', 1)[1])
    expected = dict(max_tokens=128 if name == 'warmup' else 512, seed=seed,
        temperature=1.0, top_p=.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=1.0, reasoning_effort='xhigh',
        reasoning_budget_tokens=0,
        chat_template_kwargs=dict(enable_thinking=True, preserve_thinking=True))
    if any(request.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Captured settings changed: {name}')
    if request.get('frequency_penalty', 0.0) != 0 or row.get('tools') is not None:
        raise ValueError('Unsupported penalties/tools; refusing to guess')
    return dict(temperature=request['temperature'], top_p=request['top_p'],
        top_k=request['top_k'], repetition_penalty=request['repetition_penalty'],
        presence_penalty=request['presence_penalty'], frequency_penalty=0.0)


def discover_tokenizer(path=STRATA_TOKENIZER):
    path = Path(path)
    if path.is_file():
        return path
    # Limit discovery to the installed iq3_s pack, never pick an arbitrary model.
    pack = STRATA_TOKENIZER.parent.parent
    candidates = sorted(p for p in pack.rglob('tokenizer.json') if p.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one installed iq3_s tokenizer: {candidates}')
    return candidates[0]


def load_strata_tokenizer(path):
    """Use installed Strata assets, not GGUF's vocabulary or regex constants."""
    from tokenizers import Tokenizer, AddedToken, Regex, models, pre_tokenizers, decoders
    manifest = json.loads(path.read_text())
    if isinstance(manifest.get('model'), dict):
        return Tokenizer.from_file(str(path)), [source_info(path)]
    if (manifest.get('model'), manifest.get('pre'), manifest.get('add_bos_token')) != ('gpt2', 'qwen35', False):
        raise ValueError('Unsupported installed Strata tokenizer manifest; refusing to guess')
    vocab_path, merges_path, types_path = (path.parent/name for name in
        ('vocab.json', 'merges.txt', 'token_type.json'))
    vocab = json.loads(vocab_path.read_text())
    types = json.loads(types_path.read_text())
    merges = [tuple(line.split(' ', 1)) for line in merges_path.read_text().splitlines()]
    if len(vocab) != manifest['vocab_size'] or len(types) != len(vocab) or len(merges) != manifest['n_merges']:
        raise ValueError('Installed Strata tokenizer asset sizes disagree with manifest')
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(manifest['pre_pattern']), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    for special in (True, False):
        tokens = [AddedToken(text, single_word=False, lstrip=False, rstrip=False,
                            normalized=False, special=special)
                  for text, index in vocab.items() if types[index] == (3 if special else 4)]
        if special:
            tokenizer.add_special_tokens(tokens)
        else:
            tokenizer.add_tokens(tokens)
    for key, index in manifest['special_ids'].items():
        if not 0 <= index < len(types):
            raise ValueError(f'Invalid installed special token ID: {key}={index}')
    return tokenizer, [source_info(p) for p in (path, vocab_path, merges_path, types_path)]


def checked_prompt(row, tokenizer, strata_tokenizer, template):
    from mini_llm.external.qwen4exp_chat import render_chat
    request = row['request']
    text = render_chat(SimpleNamespace(chat_template=template), row['messages'],
        thinking=request['chat_template_kwargs']['enable_thinking'],
        reasoning_effort=request['reasoning_effort'])
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    reference = list(strata_tokenizer.encode(text, add_special_tokens=False).ids)
    expected = 75 if row['request_id'] == 'warmup' else 77
    captured = row['usage']['prompt_tokens']
    if len(ids) != expected or len(reference) != expected or captured != expected or ids != reference:
        raise ValueError(f'STOP prompt mismatch {row["request_id"]}: expected={expected}, '
            f'captured={captured}, GGUF={len(ids)}, Strata={len(reference)}, '
            f'exact_ids={ids == reference}; no framing/count guesses allowed')
    return dict(request_id=row['request_id'], text=text, token_ids=ids,
                expected_prompt_tokens=expected, strata_token_ids=reference)


def source_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def preflight(rows):
    """Read headers/tokenizers only, before allocating a model or GPU state."""
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    from probe_0218_hc_identity import MODEL
    path = discover_tokenizer()
    template_path = BENCHMARK / 'provenance/chat_template.jinja'
    template = template_path.read_text()
    strata, tokenizer_assets = load_strata_tokenizer(path)
    with MultiShardTensorStore.open(MODEL) as store:
        tokenizer = load_gguf_tokenizer(store.metadata)
        prompts = [checked_prompt(row, tokenizer, strata, template) for row in rows]
        sources = [source_info(p) for p in store.paths]
    provenance = dict(checkpoint_sources=sources, tokenizer=source_info(path),
        tokenizer_assets=tokenizer_assets,
        template=source_info(template_path), template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        capture=source_info(BENCHMARK / 'outputs.jsonl'), exact_prompt_ids_verified=True)
    return tokenizer, prompts, provenance


def memory_snapshot(pid='self', *, include_io=True):
    status = Path(f'/proc/{pid}/status').read_text()
    if any(line.startswith('State:') and line.split()[1] == 'Z' for line in status.splitlines()):
        raise ProcessLookupError(f'Process {pid} exited (zombie status has no RSS fields)')
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in status.splitlines()
              if line.startswith(('RssAnon:', 'RssFile:', 'VmRSS:', 'VmHWM:', 'VmSwap:'))}
    memory['MemAvailable'] = next(int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    # Fields after comm: state is field 3; minflt=10, majflt=12.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    memory.update(minor_faults=int(fields[7]), major_faults=int(fields[9]))
    # Cross-process /proc/io can lose ptrace access before waitpid observes
    # exit. The supervisor needs memory guards, not these counters; the live
    # child records its own I/O. Do not race that optional file in the parent.
    if include_io:
        memory['io'] = {k: int(v) for k, v in (line.split(':', 1)
                       for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
    return memory


def enforce_limits(memory, floor, anon_limit, deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError('180-second parent/child time budget reached')
    if floor and memory['MemAvailable'] < floor:
        raise RuntimeError('MemAvailable floor reached')
    if anon_limit and memory['RssAnon'] > anon_limit:
        raise RuntimeError('RssAnon private allocation limit reached (not total RSS)')


def append_json(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def timing_summary(n, prefill_seconds, ttft, decode_seconds, total_seconds, sampling_seconds):
    if n < 1 or min(prefill_seconds, ttft, decode_seconds, total_seconds, sampling_seconds) < 0:
        raise ValueError('Invalid timing/count')
    return dict(emitted_tokens_including_eos=n, first_token_from_prefill=1,
        decode_forwards=n-1, prefill_seconds=prefill_seconds, ttft_seconds=ttft,
        sampling_call_wall_seconds=sampling_seconds,
        decode_wall_seconds_after_first_token=decode_seconds, total_wall_seconds=total_seconds,
        decode_forward_rate_N_minus_1_per_decode_second=(n-1)/decode_seconds if decode_seconds else None,
        emitted_N_per_total_second=n/total_seconds if total_seconds else None,
        emitted_N_per_decode_second=n/decode_seconds if decode_seconds else None)


def replay_case(model, tokenizer, row, prompt, xp, sample_token, guard, journal):
    """Sampler host ID completes queued kernels; no forced per-token GPU sync.

    CUDA events measure prefill completion without a prefill synchronization.
    Sampler call wall time therefore also contains queued model completion wait.
    """
    settings = sampling_settings(row)
    forced=row.get('_forced_output_token_ids')
    if forced is not None and any(t>=model.config.vocab_size for t in forced):
        raise ValueError('Forced token ID exceeds model vocabulary')
    xp.random.seed(row['request']['seed'])
    guard()
    state = model.create_generation_state(1, 1024)
    before = model.expert_executor.statistics()
    cpu_start = time.process_time()
    decode_cpu_start = cpu_start
    start = time.perf_counter()
    begin_event, end_event = xp.cuda.Event(), xp.cuda.Event()
    begin_event.record()
    logits = model.prefill(prompt['token_ids'], state)
    prefill_submission = time.perf_counter() - start
    end_event.record()
    ids, windows, arrivals = [], [], []
    sampling, text_seconds = 0.0, 0.0
    text = ''
    limit=len(forced) if forced is not None else row['request']['max_tokens']
    for index in range(limit):
        guard()
        if index:
            logits = model.step(ids[-1], state)
        sample_start = time.perf_counter()
        token = sample_token(logits, **settings)
        if forced is not None:
            token=forced[index] # Retain full head/sampler cost; only trajectory is controlled.
        sampling += time.perf_counter() - sample_start
        ids.append(token)  # Count EOS just like an emitted completion token.
        text_start = time.perf_counter()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        text_seconds += time.perf_counter() - text_start
        arrivals.append(time.perf_counter()-start)
        if index == 0:
            decode_cpu_start = time.process_time()
        stopped = token in model.stop_token_ids
        if len(ids) % 32 == 0 or stopped or len(ids) == limit:
            journal(dict(request_id=row['request_id'], emitted_tokens=len(ids),
                elapsed_seconds=arrivals[-1], memory=guard(),
                cache=model.expert_executor.statistics()))
        if len(ids) % 64 == 0 or stopped or len(ids) == limit:
            lo = (len(ids)-1)//64*64
            windows.append(dict(start_token_1based=lo+1, end_token_1based=len(ids),
                token_ids=ids[lo:], text=tokenizer.decode(ids[lo:], skip_special_tokens=False),
                start_seconds=arrivals[lo-1] if lo else 0.0,
                end_seconds=time.perf_counter()-start))
        if stopped:
            break
    xp.cuda.get_current_stream().synchronize()  # Final timing boundary only.
    total = time.perf_counter()-start
    prefill_seconds = xp.cuda.get_elapsed_time(begin_event, end_event)/1000
    report = dict(request_id=row['request_id'], captured_request=row['request'],
        messages=row['messages'], captured_usage=row['usage'], prompt=prompt,
        forced_trajectory=forced is not None,
        output_token_ids=ids, output_text=text, token_ready_seconds=arrivals,
        windows_64_tokens=windows, stop_reason=('eos' if stopped else
                    'forced_prefix' if forced is not None and limit<row['request']['max_tokens'] else 'max_tokens'),
        eos_included_in_usage=True, prefill_submission_wall_seconds=prefill_submission,
        process_cpu_seconds=time.process_time()-cpu_start,
        decode_process_cpu_seconds=time.process_time()-decode_cpu_start,
        prefill_completion_cuda_event_seconds=prefill_seconds, text_decode_wall_seconds=text_seconds,
        timing=timing_summary(len(ids), prefill_seconds, arrivals[0], total-arrivals[0], total, sampling),
        cache_before=before, cache_after=model.expert_executor.statistics(), memory=guard())
    del state, logits
    gc.collect()  # Release request state, not the shared expert executor/cache.
    # The next request's planner credits live model allocations, not unused pool
    # blocks. Retire unused scratch at this synchronized, untimed request seam.
    if hasattr(xp, 'get_default_memory_pool'):
        xp.get_default_memory_pool().free_all_blocks()
    return report


def child(args):
    os.environ.update(MINI_LLM_BACKEND='cupy', MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global',
                      MINI_LLM_IQ_GPU_GROUPED_KERNEL='warp')
    deadline = time.monotonic()+180
    rows = select_records(BENCHMARK / 'outputs.jsonl', args.cases)
    forced_path=getattr(args,'force_tokens_from',None)
    if forced_path is not None:
        forced_report=json.loads(Path(forced_path).read_text())
        forced_cases={r['request_id']:r['output_token_ids'] for r in forced_report['cases']}
        for row in rows:
            tokens=forced_cases[row['request_id']]
            if (not isinstance(tokens,list) or not tokens or len(tokens)>row['request']['max_tokens'] or
                    any(isinstance(t,bool) or not isinstance(t,int) or t<0 for t in tokens)):
                raise ValueError('Invalid fixed-token workload')
            row['_forced_output_token_ids']=tokens
    options = dict(MODEL_OPTIONS, expert_victim_policy=args.victim_policy,
                   cache_bytes=int(getattr(args,'expert_cache_gib',6)*GiB),
                   host_budget_bytes=int(getattr(args,'host_budget_gib',8)*GiB),
                   host_expert_bytes=int(getattr(args,'resident_host_gib',0)*GiB),
                   expert_cpu_backend=args.cpu_backend, cpu_workers=args.cpu_workers,
                   group_mixed_decode=args.group_mixed_decode,
                   overlap_shared=args.overlap_shared,
                   gdn_graph_decode=getattr(args, 'gdn_graph_decode', False),
                   device_expert_plan=getattr(args, 'device_expert_plan', False),
                   pre_moe_graph=getattr(args, 'pre_moe_graph', False),
                   shared_graph_decode=getattr(args,'shared_graph_decode',False),
                   defer_shared=getattr(args,'defer_shared',False))
    tokenizer, prompts, provenance = preflight(rows)  # Fail closed before model open.
    (args.output/'preflight.json').write_text(json.dumps(dict(provenance=provenance,
        prompts=prompts, comparison=COMPARISON, model_options=options), indent=2))
    from mini_llm.backend import xp
    from mini_llm.generation import sample_token
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from probe_0218_hc_identity import MODEL
    baseline = memory_snapshot()
    highwater = dict(RssAnon=baseline['RssAnon'], RssFile=baseline['RssFile'])
    def guard():
        memory = memory_snapshot()
        enforce_limits(memory, int(args.mem_available_floor_gib*GiB),
                       int(args.rss_anon_limit_gib*GiB), deadline)
        for key in highwater:
            highwater[key] = max(highwater[key], memory[key])
        return memory
    guard()
    results = []
    with Qwen4ExpGGUFModel.open(MODEL, **options) as model:
        # Initialize CUDA/native workers first, then pin only the submitting host.
        # A later-created service would inherit the submitter's one-core mask.
        if (os.environ.get('MINI_LLM_HOST_CPU') and
                os.environ.get('MINI_LLM_NATIVE_MOE_SERVICE', '0') == '1'):
            row_pool = model.expert_executor._ggml_row_pool
            if row_pool is not None and row_pool._native_service is None:
                from mini_llm.ops.native_expert_service import NativeExpertService
                row_pool._native_service = NativeExpertService(row_pool)
        if os.environ.get('MINI_LLM_HOST_CPU'):
            host_cpu = int(os.environ['MINI_LLM_HOST_CPU'])
            if host_cpu not in os.sched_getaffinity(0):
                raise ValueError('Requested host CPU is outside the allowed affinity mask')
            os.sched_setaffinity(0, {host_cpu})
        # Preserve actual prefill API while checking every prompt/decode forward.
        original_step = model._step
        forward_count = 0
        def guarded_step(*values, **kwargs):
            nonlocal forward_count
            memory = guard()
            if forward_count % 32 == 0:
                append_json(args.output/'memory.jsonl', dict(forwards=forward_count,
                    memory=memory, anon_growth_bytes=memory['RssAnon']-baseline['RssAnon']))
            forward_count += 1
            return original_step(*values, **kwargs)
        model._step = guarded_step
        warming = dict(requested_budget_bytes=0, touched_bytes=0)
        if args.host_warm_gib:
            from mini_llm.ops.expert_host_prefetch import warm_expert_pages
            warming = warm_expert_pages(model.weights.store,
                budget_bytes=int(args.host_warm_gib*GiB),
                reserve_bytes=int(args.mem_available_floor_gib*GiB),
                progress=lambda touched, target: guard())
        # Constructor conversions leave unused pool blocks too. Retire them at
        # the same untimed setup seam, before the first physical capacity check.
        xp.cuda.get_current_stream().synchronize()
        xp.get_default_memory_pool().free_all_blocks()
        placement_profile=None
        resident_host_setup=None
        repeated = [(row, prompt) for row, prompt in zip(rows, prompts)
                    for _ in range(1 if row['request_id'] == 'warmup' else getattr(args, 'response_repeats', 1))]
        for row, prompt in repeated:
            if (row['request_id']!='warmup' and placement_profile is None and
                    getattr(args,'warmup_frequency_placement',False)):
                placement_profile=model.expert_executor.install_observed_profile()
                (args.output/'placement_profile.json').write_text(json.dumps(placement_profile,indent=2))
            if (row['request_id']!='warmup' and resident_host_setup is None and
                    getattr(args,'resident_host_gib',0)>0):
                resident_host_setup=model.prepare_host_experts(pin=getattr(args,'resident_host_pin',False),
                    reserve_bytes=int(args.mem_available_floor_gib*GiB),progress=lambda *unused:guard())
                (args.output/'resident_host_setup.json').write_text(json.dumps(resident_host_setup,indent=2))
            if row['request_id'] != 'warmup' and args.promotion_limit is not None:
                model.expert_executor.cache.set_admission_budget(
                    args.promotion_limit, window_requests=480)
            result = replay_case(model, tokenizer, row, prompt, xp, sample_token, guard,
                lambda record: append_json(args.output/'progress.jsonl', record))
            results.append(result)
            append_json(args.output/'cases.jsonl', result)
        pool = xp.get_default_memory_pool()
        report = dict(cases=results, provenance=provenance, comparison=COMPARISON,
            options=options, placement=dict(after_warmup_promotions_per_480_requests=args.promotion_limit),
            host_warming=warming, initial_memory=baseline,placement_profile=placement_profile,
                        resident_host_setup=resident_host_setup,
            forced_tokens_source=None if forced_path is None else str(Path(forced_path).resolve()),
            cpu_backend_provenance=getattr(model.expert_executor._ggml_cpu, 'provenance', None),
            gdn_graph_statistics=[m.graph_statistics() for m in getattr(model, '_gdn_graph_owners', ())],
            pre_graph_statistics=[m.graph_statistics() for m in getattr(model, '_pre_moe_segments', {}).values()],
            shared_graph_statistics=[m.graph_statistics() for m in getattr(model,'_shared_graph_owners',())],
            token_graph_statistics=(model._token_graph.graph_statistics()
                if getattr(model, '_token_graph', None) is not None else None),
            qsa_block_graph_statistics={i:g.graph_statistics() for i,g in
                getattr(model, '_qsa_block_segments', {}).items()},
            qsa_frontend_graph_statistics=[b.mixer._frontend_graph.statistics() for b in model.blocks
                if not b.recurrent and b.mixer._frontend_graph is not None],
            native_timelines=[p.native_handoff.timeline_snapshot() for p in
                model.expert_executor._device_expert_plans.values() if p.native_handoff is not None and
                p.native_handoff.timeline is not None],
            cpu_pool_spin_us=getattr(model.expert_executor._ggml_row_pool, 'spin_us', None),
            cpu_configuration=dict(workers=args.cpu_workers,
                host_help=getattr(model.expert_executor._ggml_row_pool, 'host_help', False),
                scheduler=getattr(model.expert_executor._ggml_row_pool,'scheduler','barrier'),
                fused_hidden=getattr(model.expert_executor._ggml_row_pool,'fused_hidden',False),
                                native_moe_handoff=os.environ.get('MINI_LLM_NATIVE_MOE_HANDOFF','0')=='1',
                                                native_moe_service=os.environ.get('MINI_LLM_NATIVE_MOE_SERVICE','0')=='1',
                                                native_stage_limit=int(os.environ.get('MINI_LLM_NATIVE_MOE_STAGE_LIMIT','0')),
                                                native_stage_source=os.environ.get('MINI_LLM_NATIVE_MOE_STAGE_SOURCE','bounce'),
                                                native_timeline=os.environ.get('MINI_LLM_NATIVE_MOE_TIMELINE','0')=='1',
                                                bound_recurrent_state=os.environ.get('MINI_LLM_BOUND_RECURRENT_STATE','0')=='1',
                                                graph_device_copy=os.environ.get('MINI_LLM_GRAPH_DEVICE_COPY','0')=='1',
                                                                sized_expert_slots=os.environ.get('MINI_LLM_SIZED_EXPERT_SLOTS','0')=='1',
                                                native_moe_graph=os.environ.get('MINI_LLM_NATIVE_MOE_GRAPH','0')=='1',
                                                                device_plan_specialized=os.environ.get('MINI_LLM_DEVICE_PLAN_SPECIALIZED','0')=='1',
                                                                                qsa_frontend_graph=os.environ.get('MINI_LLM_QSA_FRONTEND_GRAPH','0')=='1',
                                                                                                graph_readonly_inputs=os.environ.get('MINI_LLM_GRAPH_READONLY_INPUTS','0')=='1',
                                                                                                                qsa_defer_validation=os.environ.get('MINI_LLM_QSA_DEFER_VALIDATION','0')=='1',
                registered_experts=len(getattr(model.expert_executor._ggml_row_pool,'_registered',{})),
                worker_cpus=getattr(model.expert_executor._ggml_row_pool, 'cpu_ids', None),
                host_affinity=sorted(os.sched_getaffinity(0))),
            sampled_highwater_bytes=highwater, final_memory=guard(),
            ple_row_io=[b.ple._row_reader.stats() for b in model.blocks
                        if b.ple is not None and b.ple._row_reader is not None],
            packed_gpu_arithmetic=dict(experts_q8=os.environ.get('MINI_LLM_EXPERT_GPU_Q8','0')=='1',
                linear_q8=os.environ.get('MINI_LLM_LINEAR_GPU_Q8','0')=='1',
                hc_bf16_storage=os.environ.get('MINI_LLM_HC_BF16_WEIGHTS','0')=='1'),
            gpu_pool_used_bytes=pool.used_bytes(), gpu_pool_reserved_bytes=pool.total_bytes(),
            gpu_pool_limit_bytes=pool.get_limit(), plan=model.plan,
            time_budget_seconds=180, journal_interval_forwards=32)
        (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--output', str(args.output.resolve()),
        '--cases', *args.cases, '--host-warm-gib', str(args.host_warm_gib),
        '--mem-available-floor-gib', str(args.mem_available_floor_gib),
        '--rss-anon-limit-gib', str(args.rss_anon_limit_gib),
        '--victim-policy', args.victim_policy, '--cpu-backend', args.cpu_backend,
        '--cpu-workers', str(args.cpu_workers), '--response-repeats', str(getattr(args, 'response_repeats', 1)),
        '--expert-cache-gib',str(getattr(args,'expert_cache_gib',6)),
        '--host-budget-gib',str(getattr(args,'host_budget_gib',8)),
        '--resident-host-gib',str(getattr(args,'resident_host_gib',0))]
    if getattr(args,'resident_host_pin',False):
        command.append('--resident-host-pin')
    if getattr(args,'force_tokens_from',None) is not None:
        command.extend(['--force-tokens-from',str(args.force_tokens_from.resolve())])
    if getattr(args,'warmup_frequency_placement',False):
        command.append('--warmup-frequency-placement')
    if args.group_mixed_decode:
        command.append('--group-mixed-decode')
    if getattr(args,'shared_graph_decode',False):
        command.append('--shared-graph-decode')
    if getattr(args,'defer_shared',False):
        command.append('--defer-shared')
    if getattr(args, 'pre_moe_graph', False):
        command.append('--pre-moe-graph')
    if getattr(args, 'device_expert_plan', False):
        command.append('--device-expert-plan')
    if getattr(args, 'gdn_graph_decode', False):
        command.append('--gdn-graph-decode')
    if getattr(args, 'overlap_shared', False):
        command.append('--overlap-shared')
    if getattr(args, 'promotion_limit', None) is not None:
        command.extend(['--promotion-limit', str(args.promotion_limit)])
    started, deadline = time.monotonic(), time.monotonic()+180
    reason = None
    with (args.output/'child.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError('180-second parent timeout')
                try:
                    memory = memory_snapshot(process.pid, include_io=False)
                except (FileNotFoundError, ProcessLookupError):
                    time.sleep(.05)
                    continue
                except PermissionError:
                    # Linux may deny /proc/io as the child exits. Do not hide
                    # a real permission failure while inference is still live.
                    if process.poll() is None:
                        raise
                    break
                append_json(args.output/'parent_memory.jsonl', dict(elapsed_seconds=time.monotonic()-started,
                    memory=memory))
                enforce_limits(memory, int(args.mem_available_floor_gib*GiB),
                               int(args.rss_anon_limit_gib*GiB), deadline)
                time.sleep(.25)
        except BaseException as error:
            reason = str(error)
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            (args.output/'parent.json').write_text(json.dumps(dict(command=command,
                returncode=process.returncode, stop_reason=reason,
                seconds=time.monotonic()-started, time_budget_seconds=180), indent=2))
    if process.returncode:
        raise RuntimeError(f'Child failed ({process.returncode}); inspect {args.output}/child.log; no results inferred')
    print(f'Completed: {args.output}/result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Exclusive new directory')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES[:2]))
    parser.add_argument('--host-warm-gib', type=float, default=0)
    parser.add_argument('--expert-cache-gib',type=float,default=6)
    parser.add_argument('--host-budget-gib',type=float,default=8)
    parser.add_argument('--resident-host-gib',type=float,default=0)
    parser.add_argument('--resident-host-pin',action='store_true')
    parser.add_argument('--warmup-frequency-placement',action='store_true')
    parser.add_argument('--force-tokens-from',type=Path,help='Controlled trajectory from retained result.json; full logits/sampler still execute')
    parser.add_argument('--victim-policy', choices=('sort', 'heap'), default='sort')
    parser.add_argument('--cpu-backend', choices=('native', 'ggml', 'ggml_pool'), default='native')
    parser.add_argument('--cpu-workers', type=int, default=15)
    parser.add_argument('--response-repeats', type=int, choices=range(1,5), default=1)
    parser.add_argument('--group-mixed-decode', action='store_true')
    parser.add_argument('--overlap-shared', action='store_true')
    parser.add_argument('--gdn-graph-decode', action='store_true')
    parser.add_argument('--device-expert-plan', action='store_true')
    parser.add_argument('--pre-moe-graph', action='store_true')
    parser.add_argument('--shared-graph-decode',action='store_true')
    parser.add_argument('--defer-shared',action='store_true')
    parser.add_argument('--promotion-limit', type=int, default=None,
        help='After warmup: max promotion attempts per 480 requests; 0 freezes placement')
    parser.add_argument('--mem-available-floor-gib', type=float, default=8, help='0 disables this guard')
    parser.add_argument('--rss-anon-limit-gib', type=float, default=8, help='Absolute RssAnon cap; 0 disables')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in
           (args.host_warm_gib, args.mem_available_floor_gib, args.rss_anon_limit_gib,args.expert_cache_gib,
                       args.host_budget_gib,args.resident_host_gib)):
        parser.error('Memory budgets must be finite and nonnegative')
    if args.cpu_workers < 1:
        parser.error('CPU workers must be positive')
    if args.promotion_limit is not None and (args.promotion_limit < 0 or args.cases[0] != 'warmup'):
        parser.error('Promotion limit must be nonnegative and requires a leading warmup case')
    if len(set(args.cases)) != len(args.cases):
        parser.error('Cases must be unique')
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()  # Before headers, CUDA initialization, warmup or payloads.
    if args.child:
        child(args)
    else:
        parent(args)


if __name__ == '__main__':
    main()
