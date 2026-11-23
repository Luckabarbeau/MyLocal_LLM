"""Native tiled packed-weight prompt products; no full FP16 expert expansion.

Decode GGUF blocks directly in the tl.dot operand tiles. The arithmetic policy
is the existing native Triton path's FP16 rounded weights/activations and FP32
accumulation, not the mixed CPU/GPU decode activation policy.
"""
import triton
import triton.language as tl


@triton.jit
def _weight_tile(W, G2S, G3S, G3X, KSG, KV, row, col,
                 K: tl.constexpr, QK: tl.constexpr, QB: tl.constexpr, FMT: tl.constexpr):
    base = W + row.to(tl.int64)*(K//QK)*QB + (col//QK)*QB
    scale = (tl.load(base).to(tl.uint16) |
             (tl.load(base+1).to(tl.uint16)<<8)).to(tl.float16,bitcast=True).to(tl.float32)
    if FMT == 5:
        within=col%64
        code=(tl.load(base+2+within//4)>>(2*(within%4)))&3
        value=scale*(code.to(tl.float32)-1.0)
    elif FMT == 3:
        within=col%32
        code=(tl.load(base+2+within%16)>>(4*(within//16)))&15
        value=scale*tl.load(KV+code).to(tl.float32)
    elif FMT == 4:
        group=(col%256)//32
        within=col%32
        high=tl.load(base+2).to(tl.uint32)|(tl.load(base+3).to(tl.uint32)<<8)
        factor=((tl.load(base+4+group//2)>>(4*(group%2)))&15)|(((high>>(2*group))&3)<<4)
        code=(tl.load(base+8+16*group+within%16)>>(4*(within//16)))&15
        value=scale*(factor.to(tl.float32)-32.0)*tl.load(KV+code).to(tl.float32)
    else:
        group=(col%256)//32
        within=col%32
        lane=within//8
        sub=within%8
        if FMT == 0:
            index=tl.load(base+2+4*group+lane).to(tl.uint32)|(((tl.load(base+66+group).to(tl.uint32)>>(2*lane))&3)<<8)
            code=tl.load(G2S+index*8+sub).to(tl.float32)
            factor=(tl.load(base+74+group)>>(4*(lane//2)))&15
            value=scale*(0.5+factor.to(tl.float32))*0.25*code
            signs=tl.load(base+34+4*group+lane)
        elif FMT == 1:
            index=tl.load(base+2+8*group+2*lane+sub//4).to(tl.uint32)|(((tl.load(base+66+group).to(tl.uint32)>>(2*lane+sub//4))&1)<<8)
            code=tl.load(G3S+index*4+sub%4).to(tl.float32)
            signs=tl.load(base+74+4*group+lane)
            factor=(tl.load(base+106+group//2)>>(4*(group%2)))&15
            value=scale*(1.0+2.0*factor.to(tl.float32))*code
        else:
            index=tl.load(base+2+8*group+2*lane+sub//4).to(tl.uint32)
            code=tl.load(G3X+index*4+sub%4).to(tl.float32)
            aux=(tl.load(base+66+4*group).to(tl.uint32)|
                 (tl.load(base+67+4*group).to(tl.uint32)<<8)|
                 (tl.load(base+68+4*group).to(tl.uint32)<<16)|
                 (tl.load(base+69+4*group).to(tl.uint32)<<24))
            signs=tl.load(KSG+((aux>>(7*lane.to(tl.uint32)))&127))
            value=scale*(0.5+(aux>>28).to(tl.float32))*0.5*code
        value=tl.where(((signs.to(tl.uint32)>>sub.to(tl.uint32))&1)!=0,-value,value)
    return value.to(tl.float16)


@triton.jit
def gateup_packed(A16, G, U, H16, TOKROW, TSTART, TCNT, TSLOT,
                  G2S, G3S, G3X, KSG, KV, gstride, ustride,
                  N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr,
                  GQK: tl.constexpr, GQB: tl.constexpr, GFMT: tl.constexpr,
                  UQK: tl.constexpr, UQB: tl.constexpr, UFMT: tl.constexpr):
    tile=tl.program_id(0)
    slot=tl.load(TSLOT+tile)
    start=tl.load(TSTART+tile)
    count=tl.load(TCNT+tile)
    row=tl.arange(0,BM)
    feature=tl.program_id(1)*BN+tl.arange(0,BN)
    token=tl.load(TOKROW+start+row,mask=row<count,other=0)
    ag=tl.zeros((BM,BN),tl.float32)
    au=tl.zeros((BM,BN),tl.float32)
    for block in range(K//BK):
        col=block*BK+tl.arange(0,BK)
        a=tl.load(A16+token[:,None].to(tl.int64)*K+col[None,:],mask=row[:,None]<count,other=0.)
        g=_weight_tile(G+slot.to(tl.int64)*gstride,G2S,G3S,G3X,KSG,KV,
            feature[None,:],col[:,None],K,GQK,GQB,GFMT)
        u=_weight_tile(U+slot.to(tl.int64)*ustride,G2S,G3S,G3X,KSG,KV,
            feature[None,:],col[:,None],K,UQK,UQB,UFMT)
        ag=tl.dot(a,g,ag)
        au=tl.dot(a,u,au)
    h=ag*(1./(1.+tl.exp(-ag)))*au
    tl.store(H16+(start+row)[:,None].to(tl.int64)*N+feature[None,:],h,mask=row[:,None]<count)


@triton.jit
def down_packed(H16,D,VALUES,TSTART,TCNT,TSLOT,G2S,G3S,G3X,KSG,KV,dstride,
                 N: tl.constexpr,K: tl.constexpr,BM: tl.constexpr,BN: tl.constexpr,
                 BK: tl.constexpr,DQK: tl.constexpr,DQB: tl.constexpr,DFMT: tl.constexpr):
    tile=tl.program_id(0)
    slot=tl.load(TSLOT+tile)
    start=tl.load(TSTART+tile)
    count=tl.load(TCNT+tile)
    row=tl.arange(0,BM)
    feature=tl.program_id(1)*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(K//BK):
        col=block*BK+tl.arange(0,BK)
        a=tl.load(H16+(start+row)[:,None].to(tl.int64)*K+col[None,:],mask=row[:,None]<count,other=0.)
        d=_weight_tile(D+slot.to(tl.int64)*dstride,G2S,G3S,G3X,KSG,KV,
            feature[None,:],col[:,None],K,DQK,DQB,DFMT)
        acc=tl.dot(a,d,acc)
    tl.store(VALUES+(start+row)[:,None].to(tl.int64)*N+feature[None,:],acc,mask=row[:,None]<count)
