#!/usr/bin/env python3
"""Build a separate WP9 timer-only host runtime against unchanged CUDA objects.

Never edits/rebuilds the baseline library or original sources. Timers are not a
speed baseline. Source expansion is retained in the requested artifact directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mini_llm/native_cpp'
BUILD = Path('/tmp/mini-llm-wp1-build')

HEADER = r'''
// WP9 standalone diagnostic ONLY. No added CUDA synchronization or changed math.
namespace wp9 {
using Clock=std::chrono::steady_clock;
struct Entry { unsigned long long count=0; double seconds=0; };
thread_local bool enabled=false;
thread_local Entry entries[10];
const char* names[]={"hc_read", "moe_submit", "moe_finish", "recurrent",
    "qsa", "hc_write", "ple", "cpu_expert_exposed_wait", "final_head_tail", "cudaStreamSynchronize"};
struct Timer {
    int slot; bool on; Clock::time_point start;
    explicit Timer(int s):slot(s),on(enabled) { if(on) start=Clock::now(); }
    ~Timer() { if(on) { entries[slot].count++; entries[slot].seconds+=std::chrono::duration<double>(Clock::now()-start).count(); } }
};
cudaError_t sync(cudaStream_t s) { Timer t(9); return ::cudaStreamSynchronize(s); }
}
extern "C" void ml_wp9_diag_start() { for(auto& e:wp9::entries) e={}; wp9::enabled=true; }
extern "C" const char* ml_wp9_diag_stop() {
    wp9::enabled=false; static thread_local char out[4096]; size_t n=0;
    n+=std::snprintf(out+n,sizeof(out)-n,"{");
    for(int i=0;i<10;++i) n+=std::snprintf(out+n,sizeof(out)-n,
        "%s\"%s\":{\"count\":%llu,\"seconds\":%.12g}",i?",":"",wp9::names[i],wp9::entries[i].count,wp9::entries[i].seconds);
    std::snprintf(out+n,sizeof(out)-n,"}"); return out;
}
#define cudaStreamSynchronize wp9::sync
'''


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = (SOURCE / 'runtime.cpp').read_text()
    # Expand only private implementation includes; headers remain authoritative originals.
    for inc in SOURCE.glob('*.inc'):
        tag = '#include "' + inc.name + '"'
        if tag not in runtime:
            continue
        text = inc.read_text()
        if inc.name == 'decoder_runtime.inc':
            text = replace_once(text, 'hc_read_device(w,ffn?b.ffn:b.attention,&ticket);',
                '{ wp9::Timer t(0); hc_read_device(w,ffn?b.ffn:b.attention,&ticket); }')
            text = replace_once(text, 'auto job=moe_submit_input(b.moe,w->hidden,scale,true); job->finish(false);',
                'std::shared_ptr<MoeJob> job;\n            { wp9::Timer t(1); job=moe_submit_input(b.moe,w->hidden,scale,true); }\n            { wp9::Timer t(2); job->finish(false); }')
            text = replace_once(text,
                '} else if(b.recurrent) recurrent_step_device(b.recurrent,w->hidden,position,&done);\n        else qsa_layer_step_device(b.qsa,w->hidden,position,&done);',
                '} else if(b.recurrent) { wp9::Timer t(3); recurrent_step_device(b.recurrent,w->hidden,position,&done); }\n        else { wp9::Timer t(4); qsa_layer_step_device(b.qsa,w->hidden,position,&done); }')
            text = replace_once(text, 'hc_write_device(w,ffn?b.moe->output:b.recurrent?b.recurrent->output:b.qsa->output);',
                '{ wp9::Timer t(5); hc_write_device(w,ffn?b.moe->output:b.recurrent?b.recurrent->output:b.qsa->output); }')
            text = replace_once(text, 'if(b.ple) ple_step_device(b.ple,0,token,p->position,1,w);',
                'if(b.ple) { wp9::Timer t(6); ple_step_device(b.ple,0,token,p->position,1,w); }')
            text = replace_once(text, 'w->begin(); w->ready(); auto v=p->final;',
                'wp9::Timer head_tail(8);\n            w->begin(); w->ready(); auto v=p->final;')
        elif inc.name == 'moe_runtime.inc':
            text = replace_once(text, '                cpu->finish();',
                '                { wp9::Timer t(7); cpu->finish(); }')
        runtime = runtime.replace(tag, text)
    runtime = replace_once(runtime, 'namespace {', HEADER + '\nnamespace {')
    target = out / 'runtime_diagnostic.cpp'
    target.write_text(runtime)
    compile_cmd = ['/usr/bin/mpicxx', '-Dmini_llm_native_EXPORTS', '-isystem',
        '/usr/local/cuda/targets/x86_64-linux/include', '-I', str(SOURCE), '-O3', '-DNDEBUG',
        '-std=c++20', '-fPIC', '-c', str(target), '-o', str(out / 'runtime_diagnostic.o')]
    objects = sorted((BUILD / 'CMakeFiles/mini_llm_native.dir').glob('*.o'))
    assert len(objects) == 16
    objects = [p for p in objects if p.name != 'runtime.cpp.o']
    link_cmd = ['/usr/bin/mpicxx', '-shared', '-fPIC', '-o', str(out / 'libmini_llm_native_diagnostic.so'),
        str(out / 'runtime_diagnostic.o'), *map(str, objects), '-L/usr/local/cuda/lib64',
        '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-ldl', '-lpthread']
    for name, cmd in [('compile', compile_cmd), ('link', link_cmd)]:
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(cmd, check=True, timeout=90, stdout=log, stderr=subprocess.STDOUT)
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'build.json').write_text(json.dumps(dict(compile=compile_cmd, link=link_cmd,
        source_sha256=digest(target), library_sha256=digest(out / 'libmini_llm_native_diagnostic.so'),
        unchanged_reused_objects=[dict(path=str(p), sha256=digest(p)) for p in objects]), indent=2)+'\n')


if __name__ == '__main__':
    main()
