#!/usr/bin/env python3
"""OpenAI-compatible chat server for the native Qwen4Exp IQ3_S engine.

This is the served version of scripts/chat_qwen4exp.py: one resident model, one
conversation, one request at a time, with conversation-prefix state reuse kept
across turns. The configuration baked in is the best measured decode setup from
the 0281-0333 campaign: the 0321 control (native ggml_pool CPU experts with 19
workers, GPU Q8 expert/linear compute, sized expert slots, native MoE service +
handoff + graph + kernel publication, pre-MoE/shared/recurrent-chain/QSA-block
graph captures, QSA warp dots, 7GiB sized expert cache, BF16 HC weights,
bounded PLE rows) plus the bit-exact 0327/0329 warp router. With this family of
settings the retained controlled runs measured 41.99/43.70 and 44.64-47.20
decode forwards/s (forced replay) and 34.68/45.94 forwards/s (natural two
responses); cold responses are much slower, so keep the server warm. Every
switch that measurement work rejected (whole-token graph, cooperative QSA,
Strata affinity, longer spin, IQ3_XXS/IQ4-VNNI gathers, Q2 rows, staged
execution) stays off. Each MINI_LLM_* name below is applied with setdefault, so
an explicit value in the parent environment still wins.
"""
import argparse
import cProfile
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ('/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/'
         'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
GGML_CPU = ('/home/lucka/Documents/my_local_LLM/qwen4exp-references/'
            'llama-no-prefetch/build-cpu/bin/libggml-cpu.so')
GiB = 1024**3

# 0321-control settings from scripts/probe_0302_decode_configuration.py plus
# the 0327/0329 exact warp-router flags. Losing/quality-rejected arms stay off.
BEST_ENV = {
    'MINI_LLM_BACKEND': 'cupy',
    'MINI_LLM_IQ_GPU_CODEBOOK_STORAGE': 'global',
    'MINI_LLM_IQ_GPU_GROUPED_KERNEL': 'warp',
    'MINI_LLM_GGML_CPU_LIBRARY': GGML_CPU,
    'MINI_LLM_GGML_POOL_TASK_FACTOR': '3',
    'MINI_LLM_GGML_POOL_SCHEDULER': 'tasks',
    'MINI_LLM_GGML_POOL_HOST_HELP': '1',
    'MINI_LLM_GGML_POOL_SPIN_US': '50',
    'MINI_LLM_GGML_POOL_FUSED_HIDDEN': '0',
    'MINI_LLM_GGML_IQ3_GATHER': '1',
    'MINI_LLM_GGML_IQ3XXS_GATHER': '0',
    'MINI_LLM_GGML_IQ3XXS_PAIR': '0',
    'MINI_LLM_GGML_IQ4_VNNI': '0',
    'MINI_LLM_GGML_Q2_SIMD': '1',
    'MINI_LLM_GGML_Q2_ROWS': '0',
    'MINI_LLM_GGML_Q2_ROWS_EXACT': '0',
    'MINI_LLM_ROUTER_WARP_SELECT': '1',
    'MINI_LLM_ROUTER_WARP_REDUCE': '1',
    'MINI_LLM_EXPERT_GPU_Q8': '1',
    'MINI_LLM_LINEAR_GPU_Q8': '1',
    'MINI_LLM_HC_BF16_WEIGHTS': '1',
    'MINI_LLM_SIZED_EXPERT_SLOTS': '1',
    'MINI_LLM_DEVICE_PLAN_SPECIALIZED': '0',
    'MINI_LLM_NATIVE_MOE_HANDOFF': '1',
    'MINI_LLM_NATIVE_MOE_GRAPH': '1',
    'MINI_LLM_NATIVE_MOE_SERVICE': '1',
    'MINI_LLM_NATIVE_MOE_KERNEL_PUBLISH': '1',
    'MINI_LLM_NATIVE_MOE_FUSED_SHARED': '0',
    'MINI_LLM_NATIVE_MOE_STAGE_LIMIT': '0',
    'MINI_LLM_NATIVE_MOE_ROUTE_PROFILE': '0',
    'MINI_LLM_NATIVE_MOE_TIMELINE': '0',
    'MINI_LLM_BOUND_RECURRENT_STATE': '1',
    'MINI_LLM_GRAPH_DEVICE_COPY': '1',
    'MINI_LLM_GRAPH_READONLY_INPUTS': '1',
    'MINI_LLM_QSA_DEFER_VALIDATION': '1',
    'MINI_LLM_QSA_FRONTEND_GRAPH': '1',
    'MINI_LLM_QSA_BLOCK_GRAPH': '1',
    'MINI_LLM_QSA_BLOCK_REPLAY': '1',
    'MINI_LLM_RECURRENT_BLOCK_GRAPH': '1',
    'MINI_LLM_RECURRENT_CHAIN_GRAPH': '1',
    'MINI_LLM_QSA_WARP_DOTS': '1',
    'MINI_LLM_QSA_NATIVE_SELECT': '1',
    'MINI_LLM_Q8_SHARED_GRID': '1',
    'MINI_LLM_TOKEN_GRAPH': '0',
    'MINI_LLM_TOKEN_GRAPH_PREP': '0',
    'MINI_LLM_QSA_COOPERATIVE': '0',
    'MINI_LLM_HC_PARALLEL_READ': '0',
    'MINI_LLM_Q8_RECURRENT_PAIR': '0',
    'MINI_LLM_PLE_BATCHED_ROWS': '1',
    'MINI_LLM_MEMORY_CENSUS': '0',
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
}


# Small offline WebUI (no external assets), llama.cpp/Strata style, talking to
# the same /v1/chat/completions endpoint with streaming on.
WEBUI_HTML = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen4Exp chat</title>
<style>
:root{color-scheme:dark}
body{font:15px/1.45 system-ui,sans-serif;background:#14161b;color:#e8eaf0;margin:0}
header{display:flex;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid #2a2d35;flex-wrap:wrap}
.dot{width:10px;height:10px;border-radius:50%;background:#777}
.dot.ok{background:#3fb950}.dot.bad{background:#f0504f}
#model{font-weight:600}#ctx{color:#8b93a3;font-size:13px}
#chat{max-width:880px;margin:0 auto;padding:16px}
.msg{padding:10px 12px;border-radius:8px;white-space:pre-wrap;word-break:break-word;margin:10px 0}
.user{background:#1d2634;border:1px solid #2c3a4f}
.asst{background:#191c22;border:1px solid #262a33}
.meta{color:#8b93a3;font-size:12px;margin:4px 2px 10px}
.err{color:#f0504f}.cur{opacity:.6}
#input{max-width:880px;margin:0 auto;padding:0 16px 18px}
textarea{width:100%;min-height:110px;box-sizing:border-box;background:#101216;color:inherit;border:1px solid #2a2d35;border-radius:8px;padding:10px;font:inherit}
.row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
label{font-size:13px;color:#8b93a3}
input[type=number]{width:90px}
button{background:#1d2634;color:#e8eaf0;border:1px solid #2c3a4f;border-radius:6px;padding:7px 14px;cursor:pointer}
button:hover:not(:disabled){background:#283242}button:disabled{opacity:.45;cursor:default}
input#key{flex:1;min-width:150px}
</style>
</head>
<body>
<header>
 <span id="dot" class="dot"></span>
 <span id="model">connecting...</span>
 <span id="ctx"></span>
 <span style="flex:1"></span>
 <button id="newchat">New chat</button>
</header>
<div id="chat"><div class="meta">Single conversation, one request at a time. Speed numbers appear under each reply and in the server log.</div></div>
<div id="input">
 <textarea id="prompt" placeholder="Type a message... (Ctrl+Enter sends)"></textarea>
 <div class="row">
  <label>max tokens <input type="number" id="maxtok" value="512" min="1"></label>
  <label>temp <input type="number" id="temp" value="0" step="0.1" min="0" max="2"></label>
  <input id="key" type="password" placeholder="API key (only if the server was started with --api-key)">
  <button id="stop" disabled>Stop</button>
  <button id="send">Send</button>
 </div>
</div>
<script>
"use strict";
const $=(s)=>document.querySelector(s);
let history=[],busy=false,aborter=null;
function esc(s){return s.replace(/[&<>]/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function headers(extra){const h=extra||{};const k=$('#key').value.trim();if(k)h['Authorization']='Bearer '+k;return h;}
function bubble(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;$('#chat').appendChild(d);return d;}
function meta(html){const d=document.createElement('div');d.className='meta';d.innerHTML=html;$('#chat').appendChild(d);return d;}
async function boot(){try{const r=await fetch('/health');if(!r.ok)throw new Error(String(r.status));const h=await r.json();
 $('#dot').className='dot ok';$('#model').textContent=h.model;$('#ctx').textContent=h.max_context+' context';}
catch(e){$('#dot').className='dot bad';$('#model').textContent='server unreachable';}}
async function send(){
 if(busy)return;
 const text=$('#prompt').value.trim();if(!text)return;
 $('#prompt').value='';
 history.push({role:'user',content:text});bubble('user',text);
 busy=true;$('#send').disabled=true;$('#stop').disabled=false;
 const el=bubble('asst','');let full='';
 const body={model:$('#model').textContent,messages:history,
  max_tokens:parseInt($('#maxtok').value,10)||512,temperature:parseFloat($('#temp').value)||0,stream:true};
 const t0=performance.now();let tFirst=null,tLast=t0,usage=null,err=null;
 aborter=new AbortController();
 try{
  const res=await fetch('/v1/chat/completions',{method:'POST',headers:headers({'Content-Type':'application/json'}),body:JSON.stringify(body),signal:aborter.signal});
  if(!res.ok){err='HTTP '+res.status+': '+esc(await res.text());}
  else{
   const rd=res.body.getReader();const dec=new TextDecoder();let buf='';
   outer:while(true){
    const r=await rd.read();if(r.done)break;
    buf+=dec.decode(r.value,{stream:true});
    let i;
    while((i=buf.indexOf('\\n\\n'))>=0){const block=buf.slice(0,i);buf=buf.slice(i+2);
     for(const line of block.split('\\n')){
      if(!line.startsWith('data:'))continue;
      const payload=line.slice(5).trim();
      if(payload==='[DONE]')break outer;
      let c;try{c=JSON.parse(payload);}catch(e){continue;}
      if(c.error){err=esc(c.error.message||JSON.stringify(c.error));break outer;}
      if(c.usage)usage=c.usage;
      const d=c.choices&&c.choices[0]&&c.choices[0].delta;
      if(d&&d.content){if(tFirst===null)tFirst=performance.now();tLast=performance.now();
       full+=d.content;el.innerHTML=esc(full)+'<span class="cur">\\u258d</span>';}
     }}}
  }
 }catch(e){if(e.name==='AbortError')err='stopped by client (the server finishes the current turn; New chat keeps things aligned)';else err='network: '+esc(e.message||String(e));}
 $('#stop').disabled=true;
 el.innerHTML=esc(full);
 if(full&&!err)history.push({role:'assistant',content:full});
 const parts=[];
 if(tFirst!==null)parts.push('first token '+((tFirst-t0)/1000).toFixed(2)+'s (prefill+first forward)');
 if(usage&&usage.completion_tokens&&tLast>tFirst)parts.push(usage.completion_tokens+' tok \\u00b7 '+Math.round(usage.completion_tokens/((tLast-tFirst)/1000))+' tok/s decode');
 if(usage)parts.push('prompt '+usage.prompt_tokens+' tok');
 if(full||err){if(err)parts.push('<span class="err">'+err+'</span>');meta(parts.join(' \\u00b7 '));}
 busy=false;$('#send').disabled=false;
}
function newChat(){history=[];$('#chat').innerHTML='';meta('New conversation: the next first reply re-prefills the whole prompt (see the [ttft]/[turn] lines in the server log).');}
$('#send').addEventListener('click',send);
$('#newchat').addEventListener('click',newChat);
$('#stop').addEventListener('click',()=>{if(aborter)aborter.abort();});
$('#prompt').addEventListener('keydown',(e)=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')send();});
boot();
</script>
</body>
</html>'''


# A self-destructing service worker. A previous Strata/llama.cpp server on this
# port may have left a cached PWA service worker installed for the origin; it
# would hijack the page load and hammer llama-style endpoints like /props.
# Serving this at /sw.js replaces the stale worker, clears its caches and
# reloads the page from the real server.
SW_JS = ("self.addEventListener('install',(event)=>{self.skipWaiting();});"
         "self.addEventListener('activate',(event)=>{event.waitUntil((async()=>{"
         "const keys=await caches.keys();for(const key of keys){await caches.delete(key);}"
         "await self.clients.claim();"
         "const all=await self.clients.matchAll();for(const client of all){client.reload();}"
         "})());});")


class QuietServer(ThreadingHTTPServer):
    """Threads exit with the process; client disconnects never print tracebacks."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def byte_budget(value, name):
    if value == 'auto':
        return value
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{name} must be auto or nonnegative GiB')
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f'{name} must be auto or nonnegative GiB')
    return int(number * GiB)


class Service:
    """Serial serving state; importing this helper never loads a model."""
    def __init__(self, session, config=None):
        self.session = session
        self.config = config or {}
        self.lock = threading.Lock()
        self.last_request = {}

    def prepare(self, messages):
        self.session.prepare(messages)

    def record(self, request_id, started):
        stats = dict(self.session.last_stats)
        stats.update(request_id=request_id, total_seconds=time.perf_counter() - started)
        stats['cache'] = dict(total_tokens=stats['prompt_tokens'],
                              new_tokens=stats['new_prompt_tokens'], reused_tokens=stats['reused_tokens'],
                              prefix_match=stats['prefix_match'], state_reset=stats['state_reset'],
                              reset_reason=stats['reset_reason'], state_tokens=stats['state_tokens'])
        self.last_request = stats
        return stats

    def metrics(self):
        # Only sampled on an explicit metrics request, never in the token loop.
        with self.lock:
            return {'config': self.config, 'mylocal_stats': dict(self.last_request),
                    'experts': self.session.model.expert_executor.statistics()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL, help='Original GGUF shard or shard directory')
    parser.add_argument('--model-id', default='qwen3.8-flash-next-iq3_s')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--api-key', default=os.environ.get('MYLOCAL_LLM_API_KEY'))
    parser.add_argument('--context', type=int, default=16384,
                        help='Conversation capacity; use --qsa-state-storage host for 262144 '
                             'on a 16GiB GPU (full FP32 history is retained in system RAM)')
    parser.add_argument('--qsa-state-storage', choices=('device', 'host'), default='device')
    parser.add_argument('--host-state-budget-gib', type=float, default=24,
                        help='Aggregate host memory budget in host-state mode (GiB)')
    parser.add_argument('--prefill-experts', choices=('decode', 'grouped', 'stream', 'strata'), default='decode',
                        help='grouped retains mixed CPU/GPU arithmetic; stream uploads packed misses '
                             'for GPU Q8 execution (a distinct prefill arithmetic policy)')
    parser.add_argument('--strata-prefill-library', help='Explicit fused-kernel library from build_strata_prefill.py')
    parser.add_argument('--prefill-chunk-size', type=int, default=0,
                        help='Reference prefill when 0; otherwise forward chunk size to model')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--cpu-workers', type=int, default=19)
    parser.add_argument('--cache-gib', default='6',
                        help='GPU expert cache GiB, or auto. 7 is the measured best but '
                             'only coexists with a modest context in the 14GiB device plan')
    parser.add_argument('--host-moe-gib', default='auto',
                        help='Startup packed-expert page warmup GiB, auto, or 0 to disable')
    parser.add_argument('--ram-reserve-gib', type=float, default=8,
                        help='Leave this much system RAM available during warmup (minimum 2GiB)')
    parser.add_argument('--profile-directory', type=Path,
                        help='Dump one cProfile .pstats file per POST from its handler thread')
    parser.add_argument('--no-thinking', action='store_true')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'xhigh'))
    parser.add_argument('--expert-placement', choices=('adaptive', 'warm-frozen'),
                        default='warm-frozen',
                        help='Freeze complete expert placement after warmup to activate native device plans; '
                             'adaptive retains synchronous admissions and disables those plans')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip the bounded decode warmup turn (cold first responses are slow)')
    args = parser.parse_args()
    if args.context < 2 or args.max_new_tokens < 1 or args.cpu_workers < 1:
        parser.error('context, max-new-tokens and cpu-workers must be positive')
    if not math.isfinite(args.host_state_budget_gib) or args.host_state_budget_gib < 0:
        parser.error('--host-state-budget-gib must be finite and nonnegative')
    if args.prefill_chunk_size < 0:
        parser.error('--prefill-chunk-size must be nonnegative')
    if not math.isfinite(args.ram_reserve_gib) or args.ram_reserve_gib < 2:
        parser.error('--ram-reserve-gib must be finite and at least 2GiB')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not args.api_key:
        parser.error('Binding beyond localhost requires --api-key')
    cache_bytes = byte_budget(args.cache_gib, '--cache-gib')
    host_bytes = byte_budget(args.host_moe_gib, '--host-moe-gib')
    profile_directory = None
    if args.profile_directory is not None:
        try:
            profile_directory = args.profile_directory.expanduser().resolve()
            profile_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f'Cannot create --profile-directory: {exc}')

    if args.prefill_experts != 'decode' and not args.prefill_chunk_size:
        parser.error('Grouped/streamed experts require --prefill-chunk-size > 0')
    if args.prefill_experts == 'strata':
        if not args.strata_prefill_library or not Path(args.strata_prefill_library).is_file():
            parser.error('strata prefill requires --strata-prefill-library pointing to a built kernel library')
        os.environ['MINI_LLM_STRATA_PREFILL_LIBRARY'] = str(Path(args.strata_prefill_library).resolve())
    os.environ['MINI_LLM_PREFILL_EXPERTS'] = args.prefill_experts
    for key, value in BEST_ENV.items():
        os.environ.setdefault(key, value)

    from mini_llm.backend import BACKEND_NAME
    from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
    from mini_llm.external.qwen4exp_chat import ChatSession
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from mini_llm.generation import sample_token

    print(f'Loading original model natively ({BACKEND_NAME}, context {args.context}, '
          'full top-10, FP32, best measured decode config)...', file=sys.stderr, flush=True)
    options = dict(
        max_length=args.context,
        activation_bytes=max(64 * 1024**2, args.prefill_chunk_size * 2560 * 4 * 64),
        cpu_workers=args.cpu_workers,
        static_device_budget_bytes=6 * GiB,
        device_budget_bytes=14 * GiB,
        cache_bytes=cache_bytes,
        expert_promote_after=1,
        group_after_admission=True,
        expert_cache_slots=cache_bytes != 0,
        qsa_fast_decode=True,
        hc_fast_decode=True,
        router_fast_decode=True,
        gdn_fast_decode=True,
        hc_graph_decode=False,
        hc_native_read=True,
        elementwise_fast_decode=True,
        expert_victim_policy='heap',
        qsa_state_storage=args.qsa_state_storage,
        host_budget_bytes=(int(args.host_state_budget_gib * GiB)
                           if args.qsa_state_storage == 'host' else 8 * GiB),
        host_expert_bytes=0,
        expert_cpu_backend='ggml_pool',
        group_mixed_decode=False,
        overlap_shared=True,
        gdn_graph_decode=False,
        device_expert_plan=True,
        pre_moe_graph=True,
        shared_graph_decode=True,
        defer_shared=True,
    )
    with Qwen4ExpGGUFModel.open(args.model, **options) as model:
        arena = model.expert_executor.reserve_device_slots()
        if arena is not None:
            print(f"GPU expert arena reserved: {arena['device_reserved_bytes'] / GiB:.2f}GiB, "
                  f"{arena['capacity']} slots", file=sys.stderr, flush=True)
        tokenizer = load_gguf_tokenizer(model.weights.store.metadata,
                                        vocab_size=model.config.vocab_size)
        if host_bytes != 0:
            from mini_llm.ops.expert_host_prefetch import warm_expert_pages
            last = [0.0]

            def progress(done, target):
                now = time.monotonic()
                if now - last[0] >= 1 or done == target:
                    print(f'\rWarming packed MoE pages in RAM: {done / GiB:.2f}/{target / GiB:.2f}GiB',
                          end='', file=sys.stderr, flush=True)
                    last[0] = now
            warmup = warm_expert_pages(model.weights.store, budget_bytes=host_bytes,
                                       reserve_bytes=int(args.ram_reserve_gib * GiB) + model.plan.get('host_state_bytes', 0),
                                       progress=progress)
            print(f"\nHost MoE warmup: {warmup['touched_bytes'] / GiB:.2f}GiB touched, "
                  f"~{warmup['resident_bytes_estimate'] / GiB:.2f}GiB resident; "
                  f"{warmup['available_after_bytes'] / GiB:.2f}GiB system-available.",
                  file=sys.stderr, flush=True)

        session = ChatSession(model, tokenizer, args.context, thinking=not args.no_thinking,
                              reasoning_effort=args.reasoning_effort,
                              prefill_chunk_size=args.prefill_chunk_size)
        if not args.no_warmup:
            print('Warming decode graphs and expert placement...', file=sys.stderr, flush=True)
            warm_start = time.perf_counter()
            for piece in session.generate('Reply with the single word: ready', 16):
                pass
            warm_stats = dict(session.last_stats)
            print(f"[warmup] prompt {warm_stats['prompt_tokens']} tok prefilled in "
                  f"{warm_stats['prefill_seconds']:.2f}s = "
                  f"{warm_stats['new_prompt_tokens'] / max(warm_stats['prefill_seconds'], 1e-30):.1f} "
                  f"prefill tok/s; decode {warm_stats['decode_forward_tokens']} forwards in "
                  f"{warm_stats['decode_seconds']:.2f}s = "
                  f"{warm_stats['decode_forwards_per_second']:.2f} model-forwards/s "
                  f"(warm state is now ready; total {time.perf_counter() - warm_start:.1f}s)",
                  file=sys.stderr, flush=True)
            session.reset()

        if args.expert_placement == 'warm-frozen':
            model.expert_executor.cache.set_admission_budget(0)
            print('Expert placement frozen: native device plans enabled; all CPU misses still execute.',
                  file=sys.stderr, flush=True)

        effective_config = dict(options, prefill_chunk_size=args.prefill_chunk_size,
                                        prefill_experts=args.prefill_experts,
                                                                        strata_prefill_library=os.environ.get('MINI_LLM_STRATA_PREFILL_LIBRARY') if args.prefill_experts == 'strata' else None,
                                        expert_placement=args.expert_placement,
                                profile_directory=str(profile_directory) if profile_directory is not None else None,
                                thinking=not args.no_thinking, reasoning_effort=args.reasoning_effort,
                                environment={key: os.environ[key] for key in BEST_ENV})
        # The model disables QSA block graphs for host state regardless of the env request.
        effective_config['qsa_block_graph'] = (args.qsa_state_storage == 'device' and
                                               os.environ.get('MINI_LLM_QSA_BLOCK_GRAPH') == '1')
        service = Service(session, effective_config)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            seen_noisy = set()

            def log_message(self, format, *values):
                line = format % values
                if any(needle in line for needle in ('/props', 'sw.js', 'workbox-',
                                                     '/models/sse', 'favicon')):
                    parts = line.split(' ')
                    key = parts[1].split('?')[0] if len(parts) > 1 else line
                    if key in self.seen_noisy:
                        return
                    self.seen_noisy.add(key)
                    sys.stderr.write(
                        f'[{time.strftime("%H:%M:%S")}] {line}  <- stale PWA/service-worker '
                        'request from an older server on this port; a self-destruct worker '
                        'is served now, repeats are suppressed\n')
                    return
                sys.stderr.write(f'[{time.strftime("%H:%M:%S")}] {format % values}\n')

            def _write(self, body):
                try:
                    self.wfile.write(body)
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                    return False

            def _json(self, payload, status=200):
                body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                try:
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self._write(body)
                except OSError:
                    self.close_connection = True

            def _html(self, page):
                body = page.encode('utf-8')
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self._write(body)
                except OSError:
                    self.close_connection = True

            def _sw(self):
                # Browsers fetch /sw.js directly (not through the stale worker),
                # so this reliably replaces it on the next navigation.
                body = SW_JS.encode('utf-8')
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/javascript; charset=utf-8')
                    self.send_header('Service-Worker-Allowed', '/')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self._write(body)
                except OSError:
                    self.close_connection = True

            def _error(self, message, status=400, code='invalid_request_error'):
                self._json({'error': {'message': message, 'type': code}}, status)

            def _authenticated(self):
                if not service_api_key:
                    return True
                header = self.headers.get('Authorization', '')
                return header == f'Bearer {service_api_key}'

            def do_GET(self):
                route = self.path.split('?', 1)[0]
                if route == '/sw.js':
                    self._sw()
                    return
                if route in ('/', '/ui', '/webui'):
                    # The page itself is public; its API calls still carry the key.
                    self._html(WEBUI_HTML)
                    return
                if not self._authenticated():
                    self._error('Invalid API key', 401, 'authentication_error')
                    return
                if self.path == '/health':
                    self._json({'status': 'ok', 'model': model_id, 'loaded': True,
                                'max_context': args.context, 'service': 'mylocal_llm_qwen4exp',
                                'config': service.config})
                elif self.path == '/metrics':
                    self._json(service.metrics())
                elif self.path in ('/v1/models', '/v1/models/'):
                    self._json({'object': 'list', 'data': [{
                        'id': model_id, 'object': 'model', 'owned_by': 'mylocal_llm',
                        'status': {'value': 'loaded'}, 'meta': {'n_ctx': args.context}}]})
                else:
                    self._error('Unknown endpoint', 404, 'not_found_error')

            def do_POST(self):
                if profile_directory is None:
                    return self._do_POST()
                profiler = cProfile.Profile()
                try:
                    return profiler.runcall(self._do_POST)
                finally:
                    profiler.dump_stats(str(profile_directory / f'request-{uuid.uuid4().hex}.pstats'))

            def _do_POST(self):
                if not self._authenticated():
                    self._error('Invalid API key', 401, 'authentication_error')
                    return
                if self.path not in ('/v1/chat/completions', '/v1/chat/completions/'):
                    self._error('Unknown endpoint', 404, 'not_found_error')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if length < 1 or length > 32 * 1024 * 1024:
                        raise ValueError('Request body must be 1B-32MiB')
                    request = json.loads(self.rfile.read(length))
                except (ValueError, json.JSONDecodeError) as exc:
                    self._error(f'Invalid request body: {exc}')
                    return
                messages = request.get('messages')
                if not isinstance(messages, list) or not messages:
                    self._error('messages must be a nonempty array')
                    return
                roles = {'system', 'user', 'assistant'}
                for message in messages:
                    if (not isinstance(message, dict) or message.get('role') not in roles
                            or not isinstance(message.get('content'), str)
                            or ('reasoning_content' in message and
                                not isinstance(message['reasoning_content'], str))):
                        self._error('Only string-content system/user/assistant messages are supported')
                        return
                if messages[-1]['role'] != 'user':
                    self._error('The last message must have role user')
                    return
                value = request.get('max_completion_tokens',
                                     request.get('max_tokens', args.max_new_tokens))
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    self._error('max_tokens / max_completion_tokens must be a positive integer')
                    return
                requested = request.get('model')
                if isinstance(requested, str) and requested and requested != model_id:
                    print(f'[note] requested model {requested!r} is served as {model_id!r}',
                          file=sys.stderr, flush=True)
                temperature = request.get('temperature', 0.0)
                top_p = request.get('top_p', 1.0)
                top_k = request.get('top_k')
                try:
                    temperature = float(temperature)
                    top_p = float(top_p)
                    top_k = None if top_k is None else int(top_k)
                except (TypeError, ValueError):
                    self._error('Invalid sampling parameters')
                    return
                if (not math.isfinite(temperature) or not math.isfinite(top_p) or
                        temperature < 0 or not 0 < top_p <= 1 or (top_k is not None and top_k < 1)):
                    self._error('Invalid sampling parameters')
                    return

                def sampler(logits_row):
                    return sample_token(logits_row, temperature=temperature, top_p=top_p,
                                        top_k=top_k)

                created = int(time.time())
                completion = f'chatcmpl-{uuid.uuid4().hex[:24]}'
                t0 = time.perf_counter()
                print(f'[request] accepted: {len(messages)} messages, max_tokens {value}, '
                      f'temperature {temperature} (prefill starts now; slow prompts show '
                      f'their speed at the [ttft] and [turn] lines)', file=sys.stderr, flush=True)

                def turn():
                    service.prepare(messages)
                    first = True
                    for piece in service.session.generate(messages[-1]['content'], value,
                                                          sampler=sampler,
                                                          raw_response=request.get('stream') is True):
                        if first:
                            first = False
                            waited = time.perf_counter() - t0
                            print(f'[ttft] first token after {waited:.2f}s '
                                  f'(prompt prefill + first forward)', file=sys.stderr, flush=True)
                        yield piece

                if request.get('stream') is not True:
                    try:
                        with service.lock:
                            for _ in turn():
                                pass
                            stats = service.record(completion, t0)
                            assistant = (service.session.messages[-1]
                                         if service.session.messages
                                         and service.session.messages[-1]['role'] == 'assistant'
                                         else {})
                    except ValueError as exc:
                        self._error(str(exc))
                        return
                    except Exception as exc:
                        with service.lock:
                            service.session.reset(reason='generation_error')
                        self._error(f'Generation failed: {exc}', 500, 'server_error')
                        return
                    answer = assistant.get('content', '')
                    reasoning = assistant.get('reasoning_content', '')
                    message = {'role': 'assistant', 'content': answer}
                    if reasoning:
                        message['reasoning_content'] = reasoning
                    self._json({
                        'id': completion, 'object': 'chat.completion', 'created': created,
                        'model': model_id, 'request_id': completion, 'mylocal_stats': stats,
                        'choices': [{'index': 0, 'message': message,
                                     'finish_reason': 'stop' if stats.get('stopped') else 'length'}],
                        'usage': {'prompt_tokens': stats.get('prompt_tokens', 0),
                                  'completion_tokens': stats.get('generated_tokens', 0),
                                  'total_tokens': stats.get('prompt_tokens', 0)
                                                  + stats.get('generated_tokens', 0)},
                    })
                    self._report(stats)
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True

                def chunk(delta, finish=None):
                    payload = {'id': completion, 'object': 'chat.completion.chunk',
                               'created': created, 'model': model_id,
                               'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                    if not self._write(b'data: ' + json.dumps(payload, ensure_ascii=False).encode()
                                       + b'\n\n'):
                        raise BrokenPipeError('client disconnected')
                    try:
                        self.wfile.flush()
                    except OSError:
                        raise BrokenPipeError('client disconnected')

                stats = {}
                try:
                    with service.lock:
                        chunk({'role': 'assistant', 'content': ''})
                        broken = False
                        for piece in turn():
                            if not broken:
                                try:
                                    chunk({'content': piece})
                                except (BrokenPipeError, ConnectionResetError):
                                    broken = True
                        stats = service.record(completion, t0)
                        if not broken:
                            chunk({}, 'stop' if stats.get('stopped') else 'length')
                            self._write(b'data: ' + json.dumps({
                                'id': completion, 'object': 'chat.completion.chunk',
                                'created': created, 'model': model_id, 'choices': [],
                                'request_id': completion, 'mylocal_stats': stats,
                                'usage': {'prompt_tokens': stats.get('prompt_tokens', 0),
                                          'completion_tokens': stats.get('generated_tokens', 0),
                                          'total_tokens': stats.get('prompt_tokens', 0)
                                                          + stats.get('generated_tokens', 0)},
                            }).encode() + b'\n\n')
                            self._write(b'data: [DONE]\n\n')
                            try:
                                self.wfile.flush()
                            except OSError:
                                pass
                except ValueError as exc:
                    if not stats:
                        self._write(b'data: ' + json.dumps(
                            {'error': {'message': str(exc), 'type': 'invalid_request_error'}}
                        ).encode() + b'\n\n')
                except (BrokenPipeError, ConnectionResetError):
                    broken = True
                except Exception as exc:
                    with service.lock:
                        service.session.reset(reason='generation_error')
                    if not stats:
                        self._write(b'data: ' + json.dumps(
                            {'error': {'message': f'Generation failed: {exc}',
                                       'type': 'server_error'}}
                        ).encode() + b'\n\n')
                self._report(stats)

            def _report(self, stats):
                if not stats:
                    return
                cache = model.expert_executor.statistics()['device']
                prefill_tokens = stats.get('new_prompt_tokens', 0)
                prefill_seconds = max(stats.get('prefill_seconds', 0.0), 1e-30)
                generated = stats.get('generated_tokens', 0)
                decode_seconds = max(stats.get('decode_seconds', 0.0), 1e-30)
                print(f"[turn] prompt {stats.get('prompt_tokens')} tokens "
                      f"(prefilled {prefill_tokens}, reused {stats.get('reused_tokens')}); "
                      f"PREFILL {prefill_tokens} tok in {prefill_seconds:.2f}s "
                      f"= {prefill_tokens / prefill_seconds:.1f} tok/s; "
                      f"DECODE {generated} tok in {decode_seconds:.2f}s "
                      f"= {generated / decode_seconds:.2f} tok/s "
                      f"({stats.get('decode_forwards_per_second', 0):.2f} model-forwards/s, "
                      f"{stats.get('decode_forward_tokens', 0)} forwards); "
                      f"cache hit {cache['hit_rate']:.1%}", file=sys.stderr, flush=True)

        model_id = args.model_id
        service_api_key = args.api_key
        server = QuietServer((args.host, args.port), Handler)
        print(f'MyLocal_LLM native Qwen4Exp OpenAI-style API at '
              f'http://{args.host}:{args.port}/v1 (model {model_id}, context {args.context}, '
              f'cache {args.cache_gib}GiB, {args.cpu_workers} CPU workers).',
              file=sys.stderr, flush=True)
        print(f'WebUI for quick chat tests at http://{args.host}:{args.port}/',
              file=sys.stderr, flush=True)
        print('Ctrl-C stops the server and releases GPU/host resources.', file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
