"""Probe: Triton fp16 tl.dot lowering on this device (products exact, fp32 acc)."""
import cupy as cp
import triton
import triton.language as tl


class P:
    def __init__(self, a):
        self.a = a
        self.dtype = tl.int8 if str(a.dtype) == 'int8' else (
            tl.float16 if str(a.dtype) == 'float16' else tl.float32)
    def data_ptr(self):
        return int(self.a.data.ptr)


@triton.jit
def k(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros((M, N), tl.float32)
    for t in range(K // BK):
        rk = t * BK + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        b = tl.load(B + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def main():
    M, N, K = 64, 128, 256
    a = cp.random.randint(-100, 100, (M, K), dtype=cp.int8).astype(cp.float16)
    b = cp.random.randint(-100, 100, (K, N), dtype=cp.int8).astype(cp.float16)
    c = cp.zeros((M, N), cp.float32)
    for BK in (16, 32, 64):
        c[:] = 0.0
        args = (P(a), P(b), P(c))
        k[(1,)](*args, M, N, K, BK, num_warps=4)
        ref = (a.astype(cp.float32) @ b.astype(cp.float32))
        exact = cp.array_equal(c, ref)
        err = float(cp.max(cp.abs(c - ref)))
        print('BK', BK, 'exact' if exact else f'max_abs_err {err}')


if __name__ == '__main__':
    main()
