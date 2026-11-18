"""Bounded two-turn real-text cache-pressure diagnostic; no reference/model overlap."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from probe_0218_hc_identity import MODEL


class ForwardLimitReached(RuntimeError):
    pass


def child(output):
    os.environ.update(MINI_LLM_BACKEND='cupy', MINI_LLM_IQ_GPU_CODEBOOK_STORAGE='global',
                      MINI_LLM_IQ_GPU_GROUPED_KERNEL='warp')
    from mini_llm.backend import xp
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from mini_llm.external.qwen4exp_chat import ChatSession
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    with Qwen4ExpGGUFModel.open(MODEL, max_length=256, cpu_workers=4,
            static_device_budget_bytes=6*1024**3, device_budget_bytes=14*1024**3,
            cache_bytes=6*1024**3, expert_promote_after=1, group_after_admission=True,
            expert_cache_slots=True, qsa_fast_decode=True, hc_fast_decode=True,
            router_fast_decode=True, gdn_fast_decode=True) as model:
        tokenizer = load_gguf_tokenizer(model.weights.store.metadata, vocab_size=model.config.vocab_size)
        session = ChatSession(model, tokenizer, 256, thinking=False)
        records = []
        step = model.step
        def measured(token, state):
            if len(records) >= 24:
                raise ForwardLimitReached('24-forward diagnostic cap')
            before = dict(model.expert_executor.counters)
            start = time.perf_counter()
            logits = step(token, state)
            xp.cuda.get_current_stream().synchronize()
            memory = {line.split(':',1)[0]:int(line.split()[1])*1024
                      for line in Path('/proc/self/status').read_text().splitlines()
                      if line.startswith(('RssAnon:', 'RssFile:', 'VmRSS:'))}
            record = dict(position=state.length-1, token=token, seconds=time.perf_counter()-start,
                experts={key:model.expert_executor.counters[key]-before.get(key, 0)
                    for key in ('gpu_experts','native_cpu_experts','cpu_submitted')}, memory=memory)
            records.append(record)
            with (output/'forwards.jsonl').open('a') as journal:
                journal.write(json.dumps(record, allow_nan=False)+'\n')
            return logits
        model.step = measured
        turns = []
        for prompt in ('Hi', 'Explain GPU memory in one short sentence.'):
            before = len(records)
            pieces, complete = [], True
            try:
                for piece in session.generate(prompt, 16):
                    pieces.append(piece)
            except ForwardLimitReached:
                complete = False
            turns.append(dict(prompt=prompt, answer=''.join(pieces), complete=complete,
                              statistics=session.last_stats if complete else {},
                              forward_records=records[before:]))
            if not complete:
                break
        report = dict(status='completed_bounded_measurement', max_forwards=24, turns=turns,
                      expert_statistics=model.expert_executor.statistics(), plan=model.plan,
                      default_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
                      note='Actual text, greedy, thinking disabled,16 output-token limit per turn; full top10 FP32. Per-step sync instrumentation; no throughput equivalence claim to Strata.')
    with (output/'conversation.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    docs = ROOT/'docs/qwen4exp'
    output = docs/f'diagnostics_0240_{args.version}'
    result = docs/f'0240_conversation_{args.version}.json'
    if output.exists() or result.exists():
        raise FileExistsError('Preserve previous diagnostics')
    output.mkdir()
    from probe_0227_short_sequences_and_profile import bounded_gpu
    # Text routing touches more unique mapped experts than IDs1..16. Keep an
    # explicit20GiB RSS/time guard, not an unbounded long-conversation run.
    import probe_0227_short_sequences_and_profile as bounds
    bounds.RSS_LIMIT = 20*1024**3
    run = bounded_gpu([sys.executable, str(Path(__file__).resolve()), '--child', str(output)], output, 'conversation')
    report = json.loads((output/'conversation.json').read_text())
    report['run'] = run
    with result.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(run=run, turns=[dict(prompt=t['prompt'], answer=t['answer'],
        statistics=t['statistics']) for t in report['turns']], expert_statistics=report['expert_statistics']), indent=2))


if __name__ == '__main__':
    main()
