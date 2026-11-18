#!/usr/bin/env python3
"""Interactive local Qwen4Exp text chat using the original GGUF tokenizer/template."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Original GGUF shard or shard directory')
    parser.add_argument('--prompt', help='First user message; shell-quote your own prompt')
    parser.add_argument('--interactive', action='store_true', help='Continue the same conversation after --prompt')
    parser.add_argument('--system', default='')
    parser.add_argument('--context', type=int, default=2048, help='Bounded allocated capacity; not metadata maximum')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--cache-gib', default='auto', help='GPU expert cache GiB, or auto within the14GiB device plan')
    parser.add_argument('--host-moe-gib', default='auto', help='Startup packed expert RAM warmup GiB, auto, or0 to disable')
    parser.add_argument('--ram-reserve-gib', type=float, default=8, help='Leave this much system RAM available (minimum2GiB)')
    parser.add_argument('--startup-report', type=Path, help='Exclusive-created JSON memory/warmup report')
    parser.add_argument('--no-thinking', action='store_true')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'xhigh'))
    parser.add_argument('--dry-run', action='store_true', help='Render/tokenize first prompt without loading weights or CUDA')
    args = parser.parse_args()
    if args.context < 1 or args.max_new_tokens < 1 or args.cpu_workers < 1:
        parser.error('context, max-new-tokens and cpu-workers must be positive')
    if args.dry_run and args.prompt is None:
        parser.error('--dry-run requires --prompt')
    def byte_budget(value, name):
        if value == 'auto':
            return value
        try:
            number = float(value)
        except ValueError:
            parser.error(f'{name} must be auto or nonnegative GiB')
        if not math.isfinite(number) or number < 0:
            parser.error(f'{name} must be auto or nonnegative GiB')
        return int(number*1024**3)
    cache_bytes = byte_budget(args.cache_gib, '--cache-gib')
    host_bytes = byte_budget(args.host_moe_gib, '--host-moe-gib')
    if not math.isfinite(args.ram_reserve_gib) or args.ram_reserve_gib < 2:
        parser.error('--ram-reserve-gib must be finite and at least2GiB')
    if args.startup_report is not None and args.startup_report.exists():
        parser.error('--startup-report already exists; preserve previous results')

    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    from mini_llm.external.qwen4exp_chat import ChatSession, render_chat
    if args.dry_run:
        from mini_llm.external.gguf_store import MultiShardTensorStore
        with MultiShardTensorStore.open(args.model) as store:
            tokenizer = load_gguf_tokenizer(store.metadata)
            messages = ([dict(role='system', content=args.system)] if args.system else [])
            messages.append(dict(role='user', content=args.prompt))
            rendered = render_chat(tokenizer, messages, thinking=not args.no_thinking,
                                   reasoning_effort=args.reasoning_effort)
            print(rendered)
            print(json.dumps(dict(prompt_tokens=len(tokenizer.encode(rendered).ids),
                                  tokenizer=tokenizer.spec.pre, weights_loaded=False)))
        return

    os.environ.setdefault('MINI_LLM_BACKEND', 'cupy')
    os.environ.setdefault('MINI_LLM_IQ_GPU_CODEBOOK_STORAGE', 'global')
    os.environ.setdefault('MINI_LLM_IQ_GPU_GROUPED_KERNEL', 'warp')
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    options = dict(cpu_workers=args.cpu_workers)
    if BACKEND_NAME == 'cupy':
        options.update(static_device_budget_bytes=6*1024**3, device_budget_bytes=14*1024**3,
            cache_bytes=cache_bytes, expert_promote_after=1, group_after_admission=True,
            expert_cache_slots=cache_bytes != 0, qsa_fast_decode=True, hc_fast_decode=True,
            router_fast_decode=True, gdn_fast_decode=True, hc_graph_decode=False)
    print(f'Loading original model ({BACKEND_NAME}, context {args.context}, full top-10, FP32)...',
          file=sys.stderr, flush=True)
    with Qwen4ExpGGUFModel.open(args.model, max_length=args.context, **options) as model:
        arena = model.expert_executor.reserve_device_slots()
        if arena is not None:
            print(f"GPU expert arena reserved: {arena['device_reserved_bytes']/1024**3:.2f}GiB, "
                  f"{arena['capacity']} slots; expert contents fill by routing, not arbitrary IDs.", file=sys.stderr, flush=True)
        tokenizer = load_gguf_tokenizer(model.weights.store.metadata, vocab_size=model.config.vocab_size)
        warmup = None
        if host_bytes != 0:
            from mini_llm.ops.expert_host_prefetch import warm_expert_pages
            last = [0.0]
            def progress(done, target):
                now = time.monotonic()
                if now-last[0] >= 1 or done == target:
                    print(f'\rWarming packed MoE pages in RAM: {done/1024**3:.2f}/{target/1024**3:.2f}GiB',
                          end='', file=sys.stderr, flush=True)
                    last[0] = now
            warmup = warm_expert_pages(model.weights.store, budget_bytes=host_bytes,
                reserve_bytes=int(args.ram_reserve_gib*1024**3), progress=progress)
            print(f"\nHost MoE warmup: {warmup['touched_bytes']/1024**3:.2f}GiB touched, "
                  f"~{warmup['resident_bytes_estimate']/1024**3:.2f}GiB resident; "
                  f"{warmup['available_after_bytes']/1024**3:.2f}GiB system-available. "
                  'No full-weight copy or pinning.', file=sys.stderr, flush=True)
        if args.startup_report is not None:
            with args.startup_report.open('x') as report:
                json.dump(dict(plan=model.plan, gpu_arena=arena, host_moe_warmup=warmup), report, indent=2)
        session = ChatSession(model, tokenizer, args.context, system=args.system,
                              thinking=not args.no_thinking, reasoning_effort=args.reasoning_effort)
        print('Ready. /reset starts a fresh conversation; /quit exits. Ctrl-C resets an interrupted response.',
              file=sys.stderr, flush=True)
        prompt = args.prompt
        interactive = args.interactive or prompt is None
        while True:
            if prompt is None:
                try:
                    prompt = input('\nYou> ')
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            if prompt.strip() in ('/quit', '/exit'):
                break
            if prompt.strip() == '/reset':
                session.reset()
                print('Conversation reset; model/expert cache remains loaded.', file=sys.stderr)
                prompt = None
                continue
            if not prompt.strip():
                prompt = None
                continue
            try:
                print('Assistant> ', end='', flush=True)
                for piece in session.generate(prompt, args.max_new_tokens):
                    print(piece, end='', flush=True)
                print()
                stats = session.last_stats
                cache = model.expert_executor.statistics()['device']
                print(f"[prefill {stats['new_prompt_tokens']} new/{stats['reused_tokens']} reused tokens: "
                      f"{stats['prefill_seconds']:.2f}s; decode {stats['decode_forward_tokens']} forwards: "
                      f"{stats['decode_forwards_per_second']:.2f} tokens/s; "
                      f"state {stats['state_tokens']}/{args.context}; cache lifetime hit {cache['hit_rate']:.1%}]",
                      file=sys.stderr, flush=True)
                if stats['rebuilt_prefix']:
                    print('[Template changed an earlier prefix; state was recomputed, cache kept warm.]', file=sys.stderr)
                if not stats['stopped']:
                    print('[Reached --max-new-tokens; response may be incomplete.]', file=sys.stderr)
            except KeyboardInterrupt:
                session.reset()
                print('\n[Interrupted; conversation reset, model/cache kept loaded.]', file=sys.stderr)
            except Exception as error:
                session.reset()
                if not interactive:
                    raise
                print(f'\n[Error: {error}; conversation reset.]', file=sys.stderr)
            if not interactive:
                break
            prompt = None


if __name__ == '__main__':
    main()
