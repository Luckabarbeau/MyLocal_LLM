#!/usr/bin/env python3
"""Matched B1/T4096 comparison: 0100B rectangular vs 0088A full head 0.

The direct baseline launches the existing 0088A private kernels over only head 0,
so it excludes the public wrapper's 16Q/4KV output and backward-grid overhead.
"""

import argparse
import statistics


def _time(call, *, warmup, repeats):
    import cupy as cp

    for _ in range(warmup):
        call()
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record(stream)
        result = call()
        end.record(stream)
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        del result
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100_021)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("invalid sequence length, warmup, or repeat count")

    import cupy as cp
    import numpy as np
    import triton
    from mini_llm.ops.triton_full_history_attention import (
        full_history_attention_backward,
        full_history_attention_forward,
    )
    import mini_llm.ops.triton_mixed_attention as square_ops

    def strides(array):
        return tuple(int(value // array.dtype.itemsize) for value in array.strides)

    def direct_square_forward(query, key, value):
        """Launch the existing 0088A full-head kernel with one-head storage."""
        batch, sequence = map(int, query.shape[:2])
        output = cp.empty(query.shape, dtype=cp.float32)
        lse = cp.empty(query.shape[:3], dtype=cp.float32)
        head_indices = cp.empty((0,), dtype=cp.int32)
        q_s, k_s, v_s, o_s, l_s = map(strides, (query, key, value, output, lse))
        grid_m = triton.cdiv(sequence, 64)
        square_ops._fwd_kernel[(grid_m, batch)](
            square_ops._ptr(query), square_ops._ptr(key), square_ops._ptr(value),
            square_ops._ptr(output), square_ops._ptr(lse),
            square_ops._ptr(head_indices),
            q_s[0], q_s[1], q_s[2], k_s[0], k_s[1], k_s[2],
            v_s[0], v_s[1], v_s[2], o_s[0], o_s[1], o_s[2],
            l_s[0], l_s[1], l_s[2],
            n_ctx=sequence, scale_log2=0.125 * square_ops._LOG2E,
            active_mask=1, head_begin=0, head_count=1,
            use_head_indices=False, window=sequence,
            block_m=64, block_n=64, num_warps=4, num_stages=3,
        )
        return output, (query, key, value, output, lse, head_indices)

    def direct_square_backward(upstream, cache):
        """Launch existing 0088A backward kernels for only Q/KV head zero."""
        query, key, value, output, lse, head_indices = cache
        batch, sequence = map(int, query.shape[:2])
        d_query = cp.empty(query.shape, dtype=cp.float32)
        d_key = cp.empty(key.shape, dtype=cp.float32)
        d_value = cp.empty(value.shape, dtype=cp.float32)
        delta = cp.empty(query.shape[:3], dtype=cp.float32)
        q_s, k_s, v_s = map(strides, (query, key, value))
        o_s, do_s, l_s, delta_s = map(strides, (output, upstream, lse, delta))
        dq_s, dk_s, dv_s = map(strides, (d_query, d_key, d_value))
        grid_m = triton.cdiv(sequence, 64)
        square_ops._delta_kernel[(grid_m, batch)](
            square_ops._ptr(output), square_ops._ptr(upstream), square_ops._ptr(delta),
            o_s[0], o_s[1], o_s[2], do_s[0], do_s[1], do_s[2],
            delta_s[0], delta_s[1], delta_s[2],
            n_ctx=sequence, active_mask=1, block_m=64, num_warps=4,
        )
        square_ops._dq_kernel[(grid_m, batch)](
            square_ops._ptr(query), square_ops._ptr(key), square_ops._ptr(value),
            square_ops._ptr(upstream), square_ops._ptr(lse),
            square_ops._ptr(delta), square_ops._ptr(d_query),
            square_ops._ptr(head_indices),
            q_s[0], q_s[1], q_s[2], k_s[0], k_s[1], k_s[2],
            v_s[0], v_s[1], v_s[2], do_s[0], do_s[1], do_s[2],
            l_s[0], l_s[1], l_s[2], delta_s[0], delta_s[1], delta_s[2],
            dq_s[0], dq_s[1], dq_s[2],
            n_ctx=sequence, scale=0.125,
            scale_log2=0.125 * square_ops._LOG2E,
            active_mask=1, head_begin=0, head_count=1,
            use_head_indices=False, window=sequence,
            block_m=64, block_n=64, num_warps=4, num_stages=3,
        )
        square_ops._dkdv_kernel[(grid_m, batch)](
            square_ops._ptr(query), square_ops._ptr(key), square_ops._ptr(value),
            square_ops._ptr(upstream), square_ops._ptr(lse),
            square_ops._ptr(delta), square_ops._ptr(d_key), square_ops._ptr(d_value),
            q_s[0], q_s[1], q_s[2], k_s[0], k_s[1], k_s[2],
            v_s[0], v_s[1], v_s[2], do_s[0], do_s[1], do_s[2],
            l_s[0], l_s[1], l_s[2], delta_s[0], delta_s[1], delta_s[2],
            dk_s[0], dk_s[1], dk_s[2], dv_s[0], dv_s[1], dv_s[2],
            n_ctx=sequence, scale=0.125,
            scale_log2=0.125 * square_ops._LOG2E,
            active_mask=1, local_window=square_ops.LOCAL_WINDOW,
            block_m=64, block_n=64, num_warps=4, num_stages=3,
        )
        return d_query, d_key, d_value

    rng = np.random.default_rng(args.seed)
    length = args.sequence_length
    q_host = rng.normal(0, .2, (1, length, 16, 64)).astype(np.float32)
    k_host = rng.normal(0, .2, (1, length, 4, 64)).astype(np.float32)
    v_host = rng.normal(0, .2, (1, length, 4, 64)).astype(np.float32)
    do_host = np.zeros(q_host.shape, dtype=np.float32)
    do_host[:, :, 0] = rng.normal(0, .2, (1, length, 64)).astype(np.float32)
    q = cp.asarray(q_host).astype(cp.dtype("bfloat16"))
    k = cp.asarray(k_host).astype(cp.dtype("bfloat16"))
    v = cp.asarray(v_host).astype(cp.dtype("bfloat16"))
    d_out = cp.asarray(do_host)
    q0, k0, v0, do0 = q[0, :, 0], k[0, :, 0], v[0, :, 0], d_out[0, :, 0]
    q_direct, k_direct, v_direct = q[:, :, :1], k[:, :, :1], v[:, :, :1]
    do_direct = d_out[:, :, :1]

    # Compile both complete paths before timing.
    rectangular_output, rectangular_cache = full_history_attention_forward(
        q0, k0, v0, return_cache=True
    )
    rectangular_gradients = full_history_attention_backward(do0, rectangular_cache)
    direct_output, direct_cache = direct_square_forward(q_direct, k_direct, v_direct)
    direct_gradients = direct_square_backward(do_direct, direct_cache)
    square_output, square_cache = square_ops.mixed_attention_forward(q, k, v, ())
    square_gradients = square_ops.mixed_attention_backward(d_out, square_cache)
    cp.cuda.get_current_stream().synchronize()

    rectangular_values = (
        cp.asnumpy(rectangular_output),
        *(cp.asnumpy(value) for value in rectangular_gradients),
    )
    direct_values = (
        cp.asnumpy(direct_output[0, :, 0]),
        *(cp.asnumpy(value[0, :, 0]) for value in direct_gradients),
    )
    wrapper_values = (
        cp.asnumpy(square_output[0, :, 0]),
        cp.asnumpy(square_gradients[0][0, :, 0]),
        cp.asnumpy(square_gradients[1][0, :, 0]),
        cp.asnumpy(square_gradients[2][0, :, 0]),
    )
    names = ("output", "dQ", "dK", "dV")
    parity = {}
    for name, rectangular, direct, wrapper in zip(
        names, rectangular_values, direct_values, wrapper_values
    ):
        parity[name] = {}
        for baseline_name, baseline in (("direct", direct), ("wrapper", wrapper)):
            difference = rectangular - baseline
            parity[name][baseline_name] = {
                "max_abs": float(np.max(np.abs(difference))),
                "relative_l2": float(
                    np.linalg.norm(difference.ravel())
                    / max(np.linalg.norm(baseline.ravel()), 1e-30)
                ),
            }

    def rectangular_forward():
        return full_history_attention_forward(q0, k0, v0, return_cache=True)

    def rectangular_backward():
        return full_history_attention_backward(do0, rectangular_cache)

    def rectangular_combined():
        output, cache = rectangular_forward()
        return output, full_history_attention_backward(do0, cache)

    def direct_forward():
        return direct_square_forward(q_direct, k_direct, v_direct)

    def direct_backward():
        return direct_square_backward(do_direct, direct_cache)

    def direct_combined():
        output, cache = direct_forward()
        return output, direct_square_backward(do_direct, cache)

    def square_forward():
        return square_ops.mixed_attention_forward(q, k, v, ())

    def square_backward():
        return square_ops.mixed_attention_backward(d_out, square_cache)

    def square_combined():
        output, cache = square_forward()
        return output, square_ops.mixed_attention_backward(d_out, cache)

    timings = {}
    # Alternate implementations by phase to reduce temperature/order bias.
    for phase, rectangular_call, direct_call, square_call in (
        ("forward", rectangular_forward, direct_forward, square_forward),
        ("backward", rectangular_backward, direct_backward, square_backward),
        ("combined", rectangular_combined, direct_combined, square_combined),
    ):
        timings[phase] = {
            "rectangular": _time(rectangular_call, warmup=args.warmup, repeats=args.repeats),
            "existing_square_direct": _time(
                direct_call, warmup=args.warmup, repeats=args.repeats
            ),
            "existing_square_wrapper": _time(
                square_call, warmup=args.warmup, repeats=args.repeats
            ),
        }
        rectangular_ms = timings[phase]["rectangular"]["median_ms"]
        direct_ms = timings[phase]["existing_square_direct"]["median_ms"]
        wrapper_ms = timings[phase]["existing_square_wrapper"]["median_ms"]
        timings[phase]["rectangular_over_direct"] = rectangular_ms / direct_ms
        timings[phase]["rectangular_over_wrapper"] = rectangular_ms / wrapper_ms

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    print({
        "device": name,
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
        "triton": triton.__version__,
        "shape": {"batch": 1, "queries": length, "keys": length, "head_dim": 64},
        "precision": "BF16 Q/K/V, FP32 output/upstream/gradients",
        "parity": parity,
        "timings": timings,
        "baseline_notes": {
            "direct": (
                "0088A private kernels launched for only head 0 with one-head "
                "output/LSE/delta/gradient storage"
            ),
            "wrapper": (
                "0088A public storage remains 16Q/4KV heads, but only full head 0 "
                "is active; inactive local-head compute is skipped"
            ),
        },
    })


if __name__ == "__main__":
    main()
