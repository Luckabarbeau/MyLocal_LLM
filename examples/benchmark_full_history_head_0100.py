"""Isolated BF16/CuPy RTX benchmark of one independent full-history head.

Does not load or modify checkpoints. Forward and backward benchmark includes
projection of raw token embeddings and native Triton attention, not optimizer updates.

Example:
 MINI_LLM_BACKEND=cupy .venv/bin/python -m examples.benchmark_full_history_head_0100 \\
   --queries 4096 --history 61440 --batch-size 1 --repeats 3
"""

import argparse
import time
from types import SimpleNamespace

from mini_llm.backend import BACKEND_NAME, xp, RandomStream, synchronize
from mini_llm.parameter import Parameter
from mini_llm.ops.full_history_head import FullHistoryHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-head", type=int, default=64)
    args = parser.parse_args()
    if BACKEND_NAME != "cupy":
        parser.error("set MINI_LLM_BACKEND=cupy to benchmark real RTX CUDA execution")
    try:
        import triton
    except ImportError:
        parser.error("install the triton-attention optional dependency")
    if args.d_head != 64:
        parser.error("native full-history Triton attention currently requires --d-head 64")
    print(f"Triton version: {triton.__version__}")
    dtype = "bfloat16"
    rng = RandomStream(1000)
    params = Parameter(
        rng.normal((32768, args.d_model), std=.02, dtype=dtype),
        "embedding.W",
    )
    head = FullHistoryHead(
        d_model=args.d_model, d_head=args.d_head, embedding=params,
        rng=rng, init_std=.02, rope_base=10000,
        name="test.full_history", dtype=dtype,
    )
    head.o.data[...] = rng.normal(head.o.data.shape, std=.01, dtype=dtype)
    history = xp.random.randint(0,32768,size=(args.batch_size,args.history),dtype=xp.int32)
    context = rng.normal((args.batch_size,args.queries,args.d_model),std=.1,dtype=dtype)
    state = SimpleNamespace(
        valid=xp.ones((args.batch_size,args.history),dtype=bool),
        ids=history,positions=xp.arange(args.history,dtype=xp.int64),
    )
    positions = xp.broadcast_to(
        xp.arange(args.history,args.history+args.queries,dtype=xp.int64)[None,:],
        (args.batch_size,args.queries),
    )
    print("Full history head GPU microbenchmark:",vars(args))
    for rep in range(args.repeats):
        head.o.grad[...] = 0
        synchronize()
        free_before, total = xp.cuda.runtime.memGetInfo()
        before = time.perf_counter()
        y,cache = head.forward(context,state,positions,return_cache=True)
        synchronize()
        fwd = time.perf_counter()-before
        free_after_forward, _ = xp.cuda.runtime.memGetInfo()
        dy = xp.ones(y.shape,dtype=y.dtype)
        before=time.perf_counter()
        head.backward(dy,cache)
        synchronize()
        bwd=time.perf_counter()-before
        free_after_backward, _ = xp.cuda.runtime.memGetInfo()
        peak_delta = max(free_before - free_after_forward,
                         free_before - free_after_backward, 0)
        print(f"rep={rep} fwd_ms={fwd*1e3:.2f} "
              f"bwd_ms={bwd*1e3:.2f} "
              f"run_device_delta_MiB={peak_delta/2**20:.1f} "
              f"device_used_MiB={(total-free_after_backward)/2**20:.1f}")


if __name__ == "__main__":
    main()
