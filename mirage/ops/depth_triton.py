"""Fused AttnRes: FP32 RMS score, depth softmax and weighted history read; no history stack."""
import torch
import os
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['K','R','CACHED'])
def _scores(Values,Q,W,Logits,Inv,D:tl.constexpr,K,BD:tl.constexpr,EPS:tl.constexpr,DT:tl.constexpr,
            Cached,R,CACHED):
    r=tl.program_id(0);d=tl.arange(0,BD)
    qw=tl.load(Q+d,d<D,0).to(tl.float32)*tl.load(W+d,d<D,0).to(tl.float32)
    for k in range(K):
        x=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r*D+d,d<D,0).to(tl.float32)
        if k<CACHED:inv=tl.load(Cached+k*R+r)
        else:inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
        score=tl.sum((x*inv)*qw,0)
        tl.store(Logits+r*K+k,score);tl.store(Inv+r*K+k,inv)


@triton.jit(do_not_specialize=['K','R','CACHED','N'])
def _online(Values,Q,DW,Weights,Inv,Y,D:tl.constexpr,K,BD:tl.constexpr,EPS:tl.constexpr,DT:tl.constexpr,
            Cached,R,CACHED,W,S,T,N,SS:tl.constexpr,TS:tl.constexpr,MODE:tl.constexpr,Read=0,SAVE_READ:tl.constexpr=False,
            SAVE_STATS:tl.constexpr=True):
    r=tl.program_id(0);d=tl.arange(0,BD)
    qw=tl.load(Q+d,d<D,0).to(tl.float32)*tl.load(DW+d,d<D,0).to(tl.float32)
    maximum=tl.full((),float('-inf'),tl.float32);denom=tl.full((),0.,tl.float32)
    total=tl.full((BD,),0.,tl.float32)
    for k in range(K):
        x=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r*D+d,d<D,0).to(tl.float32)
        if k<CACHED:inv=tl.load(Cached+k*R+r)
        else:inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
        score=tl.sum((x*inv)*qw,0)
        new=tl.maximum(maximum,score);rescale=tl.exp(maximum-new);weight=tl.exp(score-new)
        total=total*rescale+weight*x;denom=denom*rescale+weight;maximum=new
        if SAVE_STATS:
            tl.store(Weights+r*K+k,score);tl.store(Inv+r*K+k,inv)
    if SAVE_STATS:
        for k in range(K):
            score=tl.load(Weights+r*K+k)
            tl.store(Weights+r*K+k,tl.exp(score-maximum)/denom)
    value=total/denom
    if SAVE_READ:tl.store(Read+r*D+d,value,d<D)
    if MODE>0:
        x=value.to(Y.dtype.element_ty).to(tl.float32)
        inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
        w=tl.load(W+d,d<D,0).to(tl.float32)
        value=((x*inv)*w).to(Y.dtype.element_ty).to(tl.float32)
        if MODE==1:
            scale=tl.load(S+(r//N)*SS+d,d<D,0).to(tl.float32)
            shift=tl.load(T+(r//N)*TS+d,d<D,0).to(tl.float32)
            value=value*(1+scale)+shift
    tl.store(Y+r*D+d,value,d<D)


@triton.jit(do_not_specialize=['K'])
def _combine(Values,Logits,Weights,Out,D:tl.constexpr,K,BD:tl.constexpr,BK:tl.constexpr,DT:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD);ks=tl.arange(0,BK)
    scores=tl.load(Logits+r*K+ks,ks<K,float('-inf'))
    ex=tl.exp(scores-tl.max(scores,0));weights=ex/tl.sum(ex,0)
    tl.store(Weights+r*K+ks,weights,ks<K)
    total=tl.full((BD,),0.,tl.float32)
    for k in range(K):
        # Reload the small score vector to obtain a compile-time-indexed scalar weight.
        score=tl.load(Logits+r*K+k)
        weight=tl.exp(score-tl.max(scores,0))/tl.sum(ex,0)
        x=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r*D+d,d<D,0).to(tl.float32)
        total+=weight*x
    tl.store(Out+r*D+d,total,d<D)


@triton.jit(do_not_specialize=['K','R'])
def _grad_values(Values,Q,W,Weights,Inv,Out,GO,DValues,Common,
                 R,D:tl.constexpr,K,BD:tl.constexpr,BR:tl.constexpr,DT:tl.constexpr):
    block=tl.program_id(0);r=block*BR+tl.arange(0,BR);d=tl.arange(0,BD);valid=(r[:,None]<R)&(d[None,:]<D)
    q=tl.load(Q+d,d<D,0).to(tl.float32);w=tl.load(W+d,d<D,0).to(tl.float32);qw=q*w
    go=tl.load(GO+r[:,None]*D+d[None,:],valid,0).to(tl.float32)
    out=tl.load(Out+r[:,None]*D+d[None,:],valid,0)
    common=tl.full((BR,BD),0.,tl.float32)
    for k in range(K):
        x=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r[:,None]*D+d[None,:],valid,0).to(tl.float32)
        a=tl.load(Weights+r*K+k,r<R,0);inv=tl.load(Inv+r*K+k,r<R,0)
        dl=a*tl.sum(go*(x-out),1)
        dot=tl.sum(x*qw[None,:],1)
        dx=a[:,None]*go+dl[:,None]*(inv[:,None]*qw[None,:]-x*(inv[:,None]*inv[:,None]*inv[:,None]/D)*dot[:,None])
        tl.store(tl.load(DValues+k).to(tl.pointer_type(DT))+r[:,None]*D+d[None,:],dx,valid)
        common+=(dl*inv)[:,None]*x
    tl.store(Common+block*D+d,tl.sum(common,0),d<D)


@triton.jit(do_not_specialize=['R'])
def _grad_params(Common,Q,W,DQ,DW,R,D:tl.constexpr,BC:tl.constexpr,BR:tl.constexpr):
    d=tl.program_id(0)*BC+tl.arange(0,BC);rows=tl.arange(0,BR)
    total=tl.full((BC,),0.,tl.float32)
    for start in range(tl.cdiv(R,BR)):
        r=start*BR+rows
        x=tl.load(Common+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
        total+=tl.sum(x,0)
    q=tl.load(Q+d,d<D,0).to(tl.float32);w=tl.load(W+d,d<D,0).to(tl.float32)
    tl.store(DQ+d,total*w,d<D);tl.store(DW+d,total*q,d<D)


@triton.jit
def _pointer_table(Values,Table,K:tl.constexpr):
    for k in tl.static_range(K):
        tl.store(Table+k,Values[k].to(tl.uint64))

def make_table(values):
    table=torch.empty((len(values),),device=values[0].device,dtype=torch.uint64)
    _pointer_table[(1,)](tuple(values),table,len(values),num_warps=1)
    return table


def _shape(values):
    d=values[0].shape[-1];r=values[0].numel()//d
    return r,dict(D=d,K=len(values),BD=triton.next_power_of_2(d),
                  DT={torch.bfloat16:tl.bfloat16,torch.float32:tl.float32,torch.float16:tl.float16}[values[0].dtype])


@triton.jit(do_not_specialize=['R'])
def _history_rms(X,Inv,R,D:tl.constexpr,BD:tl.constexpr,EPS:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD)
    x=tl.load(X+r*D+d,d<D,0).to(tl.float32)
    tl.store(Inv+r,tl.rsqrt(tl.sum(x*x,0)/D+EPS))


@torch.library.custom_op('v31rf::history_rms',mutates_args=())
def _cached_rms(x:torch.Tensor,eps:float)->torch.Tensor:
    r=x.numel()//x.shape[-1];d=x.shape[-1]
    inv=torch.empty((r,),device=x.device,dtype=torch.float32)
    _history_rms[(r,)](x,inv,r,d,triton.next_power_of_2(d),eps,num_warps=4)
    return inv


@_cached_rms.register_fake
def _fake_cached_rms(x,eps):return torch.empty((x.numel()//x.shape[-1],),device=x.device,dtype=torch.float32)


def history_rms(x,eps):
    # Scratch statistics, NOT a replacement gradient path: depth backward below
    # explicitly includes d(rsqrt(mean(x*x)+eps))/dx for every cached feature.
    with torch.no_grad():return _cached_rms(x,eps)


@torch.library.custom_op('v31rf::depth_forward',mutates_args=())
def _fwd(values:list[torch.Tensor],q:torch.Tensor,w:torch.Tensor,eps:float,cached:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    r,args=_shape(values)
    out=torch.empty_like(values[0],dtype=torch.float32)
    inv=torch.empty((r,len(values)),device=q.device,dtype=torch.float32);weights=torch.empty_like(inv)
    ptrs=make_table(values)
    _online[(r,)](ptrs,q,w,weights,inv,out,**args,EPS=eps,Cached=cached,R=r,CACHED=cached.shape[0],
                    W=w,S=w,T=w,N=1,SS=0,TS=0,MODE=0,num_warps=4,enable_fp_fusion=False)
    return out,weights,inv


@_fwd.register_fake
def _fake(values,q,w,eps,cached):
    r=values[0].numel()//values[0].shape[-1]
    return (torch.empty_like(values[0],dtype=torch.float32),
            torch.empty((r,len(values)),device=q.device,dtype=torch.float32),
            torch.empty((r,len(values)),device=q.device,dtype=torch.float32))


@torch.library.custom_op('v31rf::depth_backward',mutates_args=())
def _bwd(values:list[torch.Tensor],q:torch.Tensor,w:torch.Tensor,out:torch.Tensor,
         weights:torch.Tensor,inv:torch.Tensor,go:torch.Tensor)->tuple[list[torch.Tensor],torch.Tensor,torch.Tensor]:
    r,args=_shape(values);dv=[torch.empty_like(x) for x in values]
    br=4;tiles=triton.cdiv(r,br);common=torch.empty((tiles,q.numel()),device=q.device,dtype=torch.float32)
    dq,dw=torch.empty_like(q),torch.empty_like(w)
    _grad_values[(tiles,)](make_table(values),q,w,weights,inv,out,go.contiguous(),make_table(dv),common,r,**args,BR=br,num_warps=4)
    _grad_params[(triton.cdiv(q.numel(),128),)](common,q,w,dq,dw,tiles,q.numel(),BC=128,BR=32,num_warps=4)
    return dv,dq,dw


@_bwd.register_fake
def _fake_bwd(values,q,w,out,weights,inv,go):
    return [torch.empty_like(x) for x in values],torch.empty_like(q),torch.empty_like(w)


def _setup(ctx,inputs,output):
    values,q,w,eps,cached=inputs
    ctx.count=len(values);ctx.save_for_backward(*values,q,w,*output)
    ctx.mark_non_differentiable(output[1],output[2])


def _autograd(ctx,go,unused_weights,unused_inv):
    saved=ctx.saved_tensors;k=ctx.count
    values=saved[:k];q,w,out,weights,inv=saved[k:]
    dv,dq,dw=_bwd(list(values),q,w,out,weights,inv,go)
    return dv,dq,dw,None,None


_fwd.register_autograd(_autograd,setup_context=_setup)


def depth_read(values,q,w,eps,*,cached=None):
    if inference_kernels():
        return inference_read(values,q,w,eps,cached=cached)
    if cached is None:cached=torch.empty((0,values[0].numel()//values[0].shape[-1]),device=q.device,dtype=torch.float32)
    return _fwd(list(values),q,w,eps,cached)[0].to(values[0].dtype)


def inference_kernels():
    return torch.is_inference_mode_enabled() and os.environ.get('MIRAGE_INFERENCE_KERNELS','1')=='1'


def inference_read(values,q,dw,eps,*,cached=None,norm_weight=None,scale=None,shift=None):
    """Same online read and BF16 boundaries, without backward-only outputs."""
    r,args=_shape(values)
    if cached is None:cached=torch.empty((0,r),device=q.device,dtype=torch.float32)
    y=torch.empty_like(values[0])
    mode=0 if norm_weight is None else (2 if scale is None else 1)
    w=dw if norm_weight is None else norm_weight
    s=w if scale is None else scale
    t=w if shift is None else shift
    warps=int(os.environ.get('MIRAGE_DEPTH_WARPS','4'))
    if warps not in (2,4,8):raise ValueError('MIRAGE_DEPTH_WARPS must be 2, 4 or 8')
    _online[(r,)](make_table(values),q,dw,y,y,y,**args,EPS=eps,Cached=cached,R=r,CACHED=cached.shape[0],
                  W=w,S=s,T=t,N=values[0].shape[-2],SS=0 if scale is None else scale.stride(0),
                  TS=0 if shift is None else shift.stride(0),MODE=mode,SAVE_STATS=False,
                  num_warps=warps,enable_fp_fusion=False)
    return y
