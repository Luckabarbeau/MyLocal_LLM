"""Probe: Triton int8 tl.dot tile shapes supported on this device."""
import cupy as cp
import triton
import triton.language as tl


import triton.language as _tl
class P:
    def __init__(self, a):
        self.a = a
        self.dtype = {str(a.dtype): _tl.int8 if str(a.dtype)=='int8' else _tl.float32}[str(a.dtype)]
    def data_ptr(self):
        return int(self.a.data.ptr)


@triton.jit
def k(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros((M, N), tl.int32)
    for t in range(K // BK):
        rk = t * BK + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        b = tl.load(B + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.float32))


def main():
    import sys
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    BK = int(sys.argv[4])
    a = cp.random.randint(-100, 100, (M, K), dtype=cp.int8)
    b = cp.random.randint(-100, 100, (K, N), dtype=cp.int8)
    c = cp.zeros((M, N), cp.float32)
    args = (P(a), P(b), P(c))
    compiled = k.warmup(*args, M, N, K, BK, grid=(1,), num_warps=4)
    compiled[(1,)](*args, M, N, K, BK, stream=int(cp.cuda.get_current_stream().ptr))
    ref = (a.astype(cp.int32) @ b.astype(cp.int32)).astype(cp.float32)
    print('M', M, 'N', N, 'BK', BK, 'exact' if cp.array_equal(c, ref) else 'MISMATCH')


if __name__ == '__main__':
    main()
