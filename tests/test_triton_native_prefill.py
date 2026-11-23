"""Independent block decoding and scalar CUDA arithmetic oracles; no Strata import.

GPU runs must explicitly select GPU0 by UUID. CPU collection remains usable
without CuPy, Triton, or a CUDA driver. All numerical comparisons are bitwise.
"""
import os

import numpy as np
import pytest

from mini_llm.ops import gguf_codebooks as tables


FORMATS = {'IQ2_S': (256, 82), 'IQ3_S': (256, 110),
           'IQ3_XXS': (256, 98), 'IQ4_NL': (32, 18),
           'IQ4_XS': (256, 136), 'Q2_0': (64, 18)}
UUID = 'GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d'


def decode_numpy(raw, kind, n, k):
    """Scalar GGUF decoder, returning signed codes and per16 FP32 scales."""
    qk, qb = FORMATS[kind]
    raw = raw.reshape(n, k//qk, qb)
    codes = np.empty((n, k), np.int8)
    scales = np.empty((n, k//16), np.float32)
    for row in range(n):
        for col in range(k):
            b = raw[row, col//qk]
            d = b[:2].copy().view('<f2')[0].astype(np.float32)
            group, within = (col % 256)//32, col % 32
            lane, sub = within//8, within % 8
            if kind == 'Q2_0':
                j = col % 64
                c, s = ((int(b[2+j//4]) >> (2*(j % 4))) & 3)-1, d
            elif kind in ('IQ4_NL', 'IQ4_XS'):
                if kind == 'IQ4_NL':
                    index, s = 2+within % 16, d
                else:
                    high = int.from_bytes(b[2:4].tobytes(), 'little')
                    factor = ((int(b[4+group//2]) >> (4*(group % 2))) & 15) | (((high >> (2*group)) & 3) << 4)
                    index, s = 8+16*group+within % 16, np.float32(d*(factor-32))
                c = int(tables.kvalues_iq4nl[(int(b[index]) >> (4*(within//16))) & 15])
            else:
                if kind == 'IQ2_S':
                    index = int(b[2+4*group+lane]) | (((int(b[66+group]) >> (2*lane)) & 3) << 8)
                    c = int(tables.iq2s_grid.reshape(-1, 8)[index, sub])
                    signs = int(b[34+4*group+lane])
                    factor = (int(b[74+group]) >> (4*(lane//2))) & 15
                    s = np.float32(np.float32(d*(2*factor+1))*np.float32(.125))
                elif kind == 'IQ3_S':
                    index = int(b[2+8*group+2*lane+sub//4]) | (((int(b[66+group]) >> (2*lane+sub//4)) & 1) << 8)
                    c = int(tables.iq3s_grid.reshape(-1, 4)[index, sub % 4])
                    signs = int(b[74+4*group+lane])
                    factor = (int(b[106+group//2]) >> (4*(group % 2))) & 15
                    s = np.float32(d*(1+2*factor))
                else:
                    index = int(b[2+8*group+2*lane+sub//4])
                    c = int(tables.iq3xxs_grid.reshape(-1, 4)[index, sub % 4])
                    aux = int.from_bytes(b[66+4*group:70+4*group].tobytes(), 'little')
                    signs = int(tables.ksigns_iq2xs[(aux >> (7*lane)) & 127])
                    s = np.float32(np.float32(d*(2*(aux >> 28)+1))*np.float32(.25))
                if (signs >> sub) & 1:
                    c = -c
            codes[row, col] = c
            scales[row, col//16] = s
    return codes, scales


def make_weights(kind, n, k, seed):
    qk, qb = FORMATS[kind]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n, k//qk, qb), dtype=np.uint8)
    # Finite, signed, non-power-of-two half scales exercise scale rounding.
    ds = rng.uniform(-.05, .05, (n, k//qk)).astype('<f2')
    raw[:, :, :2] = ds.view(np.uint8).reshape(n, k//qk, 2)
    return raw.reshape(-1)


@pytest.mark.parametrize('kind', FORMATS)
def test_numpy_block_reference(kind):
    raw = make_weights(kind, 3, 512, 901)
    c, s = decode_numpy(raw, kind, 3, 512)
    assert c.shape == (3, 512) and s.shape == (3, 32)
    assert c.dtype == np.int8 and s.dtype == np.float32
    assert np.isfinite(s).all()
    if kind == 'Q2_0':
        assert set(np.unique(c)) == {-1, 0, 1, 2}
    if kind.startswith('IQ4'):
        assert set(np.unique(c)).issubset(set(tables.kvalues_iq4nl))


@pytest.fixture(scope='module')
def gpu():
    cp = pytest.importorskip('cupy')
    pytest.importorskip('triton')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        pytest.skip('GPU tests require CUDA_VISIBLE_DEVICES='+UUID)
    try:
        if cp.cuda.runtime.getDeviceCount() != 1:
            pytest.skip('exactly one selected CUDA device required')
        cp.cuda.Device(0).use()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    from mini_llm.ops import triton_native_prefill as native
    return cp, native


CUDA = r'''
extern "C" __global__ void quant(const float* x, const float* u,
    signed char* q, float* scales, unsigned char* packed, int k, int swiglu) {
    int row=blockIdx.x, block=blockIdx.y;
    float v[32], am=0.f;
    for(int j=0;j<32;++j) {
        float g=x[row*k+block*32+j];
        v[j]=swiglu ? g/(1.f+__expf(-g))*u[row*k+block*32+j] : g;
        am=fmaxf(am,fabsf(v[j]));
    }
    float inv=am>0.f ? 127.f/am : 0.f;
    int sum=0;
    unsigned char* p=packed+(row*(k/64)+block/2)*80;
    for(int j=0;j<32;++j) {
        int c=__float2int_rn(v[j]*inv);
        q[row*k+block*32+j]=(signed char)c;
        p[(block%2)*32+j]=(unsigned char)c;
        sum+=c;
    }
    scales[row*(k/32)+block]=am/127.f;
    ((float*)(p+64))[block%2]=am/127.f;
    ((float*)(p+64))[2+block%2]=(float)sum;
}
extern "C" __global__ void linear(const signed char* a, const float* dx,
    const signed char* w, const float* sw, float* y, int r, int n, int k, int per16) {
    int row=blockIdx.x, col=blockIdx.y;
    float acc=0.f;
    for(int b=0;b<k/32;++b) {
        int d0=0,d1=0;
        for(int j=0;j<16;++j) {
            d0+=(int)a[row*k+b*32+j]*(int)w[col*k+b*32+j];
            d1+=(int)a[row*k+b*32+16+j]*(int)w[col*k+b*32+16+j];
        }
        if(per16) {
            float v=fmaf(sw[col*(k/16)+b*2+1],(float)d1,
                         sw[col*(k/16)+b*2]*(float)d0);
            acc=fmaf(dx[row*(k/32)+b],v,acc);
        } else {
            float p=sw[col*(k/16)+b*2]*dx[row*(k/32)+b];
            acc=fmaf(p,(float)(d0+d1),acc);
        }
    }
    y[row*n+col]=acc;
}
'''


def oracle(cp, name):
    return cp.RawKernel(CUDA, name, options=('--std=c++17', '--prec-div=true', '--ftz=false'))


def exact(cp, actual, expected):
    a, b = cp.asnumpy(actual), cp.asnumpy(expected)
    if a.dtype == np.float32:
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    else:
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('swiglu', [False, True])
def test_quantization_scalar_cuda(gpu, swiglu):
    cp, native = gpu
    rng = np.random.default_rng(762)
    x = rng.normal(0, 4, (7, 512)).astype(np.float32)
    x[0] = 0
    # Ties, signed zero, large exp arguments, and very small normal blocks.
    x[1, :32] = np.resize(np.array([127, -127, .5, 1.5, 2.5, -.5, -1.5, -2.5], np.float32), 32)
    x[2, :32] = np.linspace(-100, 100, 32, dtype=np.float32)
    x[3, :32] *= np.float32(1e-30)
    u = cp.asarray(rng.normal(size=x.shape).astype(np.float32))
    x = cp.asarray(x)
    result = native.native_swiglu_quantize(x, u) if swiglu else native.quantize_native(x)
    q, s, p = cp.empty_like(result.codes), cp.empty_like(result.scales), cp.empty_like(result.packed)
    oracle(cp, 'quant')((7, 16), (1,), (x, u, q, s, p, np.int32(512), np.int32(swiglu)))
    exact(cp, result.codes, q)
    exact(cp, result.scales, s)
    exact(cp, result.packed, p)
    if not swiglu:
        host = cp.asnumpy(x).reshape(7, 16, 32)
        am = np.max(np.abs(host), axis=2)
        inv = np.divide(np.float32(127), am, out=np.zeros_like(am), where=am > 0)
        np.testing.assert_array_equal(cp.asnumpy(q), np.rint(host*inv[:, :, None]).astype(np.int8).reshape(7, 512))
        # CUDA lowers this constant denominator to an FP32 reciprocal multiply.
        np.testing.assert_array_equal(cp.asnumpy(s), am*np.float32(1/127))


@pytest.mark.parametrize('kind', FORMATS)
def test_grouped_linear_scalar_cuda(gpu, kind):
    cp, native = gpu
    n, k = 37, 512
    counts = np.array([3, 0, 19], np.int32)
    offsets = np.array([0, 3, 3, 22], np.int32)
    rng = np.random.default_rng(43)
    x = cp.asarray(rng.normal(size=(22, k)).astype(np.float32))
    aq = native.quantize_native(x)
    blobs = [make_weights(kind, n, k, 71+e) for e in range(3)]
    stride = blobs[0].size+17  # unaligned stage padding must not enter decode
    raw = np.full(3*stride, 0xED, np.uint8)
    for e, blob in enumerate(blobs):
        raw[e*stride:e*stride+blob.size] = blob
    out = native.native_linear(aq, cp.asarray(raw), cp.asarray(counts), cp.asarray(offsets),
                               N=n, K=k, kind=kind, weight_stride=stride, max_rows=19)
    expected = cp.empty((22, n), cp.float32)
    for e in (0, 2):
        c, s = decode_numpy(blobs[e], kind, n, k)
        start, count = int(offsets[e]), int(counts[e])
        oracle(cp, 'linear')((count, n), (1,), (
            aq.codes[start:start+count], aq.scales[start:start+count],
            cp.asarray(c), cp.asarray(s), expected[start:start+count],
            np.int32(count), np.int32(n), np.int32(k), np.int32(kind == 'IQ2_S')))
    exact(cp, out, expected)


@pytest.mark.parametrize('kind', FORMATS)
def test_weight_codes_and_scales(gpu, kind):
    cp, native = gpu
    import triton
    import triton.language as tl

    @triton.jit
    def unpack(W, C, S, G2, G3, GX, SG, KV, K: tl.constexpr,
               QK: tl.constexpr, QB: tl.constexpr, FMT: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, 512)
        code, scale = native._decode(W, G2, G3, GX, SG, KV, row, col,
                                     col < K, K, QK, QB, FMT)
        tl.store(C+row*K+col, code)
        tl.store(S+row*K+col, scale)

    raw = make_weights(kind, 5, 512, 109)
    expected_c, expected_s = decode_numpy(raw, kind, 5, 512)
    c, s = cp.empty((5, 512), cp.int8), cp.empty((5, 512), cp.float32)
    fmt, qk, qb = native._FORMATS[kind]
    unpack[(5,)](native._ptr(cp.asarray(raw)), native._ptr(c), native._ptr(s),
                 *native._codebooks(cp), K=512, QK=qk, QB=qb, FMT=fmt,
                 enable_fp_fusion=False)
    exact(cp, c, cp.asarray(expected_c))
    exact(cp, s, cp.asarray(np.repeat(expected_s, 16, axis=1)))


def test_empty_and_contracts(gpu):
    cp, native = gpu
    q = native.quantize_native(cp.empty((0, 256), cp.float32))
    assert q.packed.shape == (0, 4, 80)
    out = native.native_linear(q, cp.empty(0, cp.uint8), cp.empty(0, cp.int32),
        cp.zeros(1, cp.int32), N=17, K=256, kind='IQ2_S', weight_stride=1394, max_rows=0)
    assert out.shape == (0, 17)
    with pytest.raises(ValueError, match='64'):
        native.quantize_native(cp.empty((1, 32), cp.float32))
    with pytest.raises(TypeError, match='float32'):
        native.quantize_native(cp.empty((1, 64), cp.float16))
    with pytest.raises(ValueError, match='agree'):
        native.native_swiglu_quantize(cp.empty((1, 64), cp.float32), cp.empty((2, 64), cp.float32))
