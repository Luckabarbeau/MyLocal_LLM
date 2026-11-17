"""Profile the optimized FP32 decode path, separately from throughput timing."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from probe_0218_hc_identity import MODEL
from probe_0227_short_sequences_and_profile import Profile, bounded_gpu


def child(output):
    from mini_llm.backend import xp
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    with Qwen4ExpGGUFModel.open(MODEL, max_length=16, cpu_workers=4,
            static_device_budget_bytes=6*1024**3, device_budget_bytes=14*1024**3,
            cache_bytes=6*1024**3, expert_promote_after=1, qsa_fast_decode=True,
            hc_fast_decode=True, router_fast_decode=True, gdn_fast_decode=True) as model:
        profile = Profile(xp)
        profile.install(model)
        state = model.create_generation_state(1, 16)
        times = []
        for position in range(16):
            profile.position = position
            before = time.perf_counter()
            model.step(position+1, state)
            xp.cuda.get_current_stream().synchronize()
            times.append(time.perf_counter()-before)
        report = dict(profile=profile.summary(), token_seconds=times,
                      expert_statistics=model.expert_executor.statistics(), plan=model.plan)
    with (output/'profile.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--version', default='v2')
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    output = ROOT/f'docs/qwen4exp/diagnostics_0231_profile_{args.version}'
    result = ROOT/f'docs/qwen4exp/0231_fast_profile_{args.version}.json'
    if output.exists() or result.exists():
        raise FileExistsError('Preserve previous artifacts')
    output.mkdir()
    run = bounded_gpu([sys.executable, str(Path(__file__).resolve()), '--child', str(output)], output, 'profile')
    report = json.loads((output/'profile.json').read_text())
    report['run'] = run
    with result.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report['profile']['stages'], indent=2))


if __name__ == '__main__':
    main()
