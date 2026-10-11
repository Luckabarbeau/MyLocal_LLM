"""0076A: reference microbenchmark for AdamW versus hybrid Muon step overhead.

Not a convergence benchmark: use equal model parameters and gradient tensors,
measure per-step wall time and report relative optimizer-state sizes. GPU timing
includes a stream synchronization. Intended for quick hardware sanity checks.
"""
import argparse
import time

from mini_llm.backend import xp, BACKEND_NAME, synchronize
from mini_llm.optim.adamw import AdamW
from mini_llm.optim.hybrid_muon import HybridMuonAdamW
from mini_llm.parameter import Parameter


def run(optimizer, repetitions, *, warmup=2):
    samples = []
    for step in range(warmup + repetitions):
        for p in optimizer.parameters:
            p.grad[...] = 0.01
        synchronize()
        start = time.perf_counter()
        optimizer.step(lr=1.5e-4)
        synchronize()
        if step >= warmup:
            samples.append(time.perf_counter() - start)
    return sum(samples) / len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--matrices', type=int, default=6)
    args = parser.parse_args()
    print('backend:', BACKEND_NAME)
    for kind in ('adamw', 'hybrid-muon'):
        params = [Parameter(xp.ones((args.width, args.width), dtype='float32'),
                            name=f'blocks.0.attention.Wq' if i == 0 else
                                 f'blocks.{i}.attention.Wq')
                  for i in range(args.matrices)]
        params.append(Parameter(xp.ones((args.width,), dtype='float32'), name='final_norm.gamma'))
        common = dict(lr=1.5e-4, beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)
        opt = (HybridMuonAdamW(params, **common) if kind == 'hybrid-muon'
               else AdamW(params, **common))
        elapsed = run(opt, args.steps)
        if isinstance(opt, HybridMuonAdamW):
            state_bytes = (sum(x.nbytes for x in opt.muon.m) +
                           sum(x.nbytes for x in opt.adam.m) +
                           sum(x.nbytes for x in opt.adam.v) +
                           sum(x.nbytes for x in opt.adam.master_weights) +
                           sum(x.nbytes for x in opt.muon.master_weights))
        else:
            state_bytes = sum(x.nbytes for arr in (opt.m, opt.v, opt.master_weights)
                              for x in arr)
        print(f'{kind}: mean_step_ms={elapsed*1000:.2f}  state_MiB={state_bytes / 2**20:.1f}')


if __name__ == '__main__':
    main()
