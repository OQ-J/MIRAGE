"""AttnRes weighted read + AdaRMS; preserve BF16 cast boundaries, recompute read in backward."""
import torch,os
import triton
import triton.language as tl
from mirage.ops.depth_triton import _online,_grad_params,_shape,make_table
from mirage.ops.norm_triton import _reduce_w,_reduce_pair



def cache_shape(values):
    if os.environ.get('MIRAGE_ATTNRES_CACHE','1')!='1':return (0,)
    policy=os.environ.get('MIRAGE_ATTNRES_POLICY','adaptive')
    if policy not in ('adaptive','all','none'):raise ValueError('MIRAGE_ATTNRES_POLICY: adaptive/all/none')
    if policy=='none':return (0,)
    if policy=='all':return values[0].shape
    minimum=int(os.environ.get('MIRAGE_ATTNRES_MIN_HISTORY','8'))
    rows=int(os.environ.get('MIRAGE_ATTNRES_MIN_ROWS','1024'))
    if minimum<1 or rows<1:raise ValueError('AttnRes thresholds must be positive')
    return values[0].shape if len(values)>=minimum and values[0].numel()//values[0].shape[-1]>=rows else (0,)

@triton.jit(do_not_specialize=['N','K'])
def _combine(Values,Logits,Weights,W,S,T,Y,D:tl.constexpr,K,BD:tl.constexpr,
             BK:tl.constexpr,N,SS:tl.constexpr,TS:tl.constexpr,EPS:tl.constexpr,DT:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD);ks=tl.arange(0,BK)
    scores=tl.load(Logits+r*K+ks,ks<K,float('-inf'))
    ex=tl.exp(scores-tl.max(scores,0));denom=tl.sum(ex,0)
    tl.store(Weights+r*K+ks,ex/denom,ks<K)
    total=tl.full((BD,),0.,tl.float32)
    for k in range(K):
        a=tl.exp(tl.load(Logits+r*K+k)-tl.max(scores,0))/denom
        v=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r*D+d,d<D,0).to(tl.float32)
        total+=a*v
    # Original path: FP32 depth read -> BF16 -> RMS with FP32 stats -> BF16 -> modulation.
    x=total.to(Y.dtype.element_ty).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    w=tl.load(W+d,d<D,0).to(tl.float32)
    normalized=((x*inv)*w).to(Y.dtype.element_ty).to(tl.float32)
    s=tl.load(S+(r//N)*SS+d,d<D,0).to(tl.float32)
    t=tl.load(T+(r//N)*TS+d,d<D,0).to(tl.float32)
    tl.store(Y+r*D+d,normalized*(1+s)+t,d<D)


@triton.jit(do_not_specialize=['N','K','MERGE_K'])
def _backward(Values,Q,DW,W,S,Weights,Inv,GO,DValues,Common,WRows,SRows,TRows,
              D:tl.constexpr,K,BD:tl.constexpr,N,SS:tl.constexpr,EPS:tl.constexpr,DT:tl.constexpr,BR:tl.constexpr,MERGE:tl.constexpr=False,EXISTING:tl.constexpr=False,MERGE_K=0,GROUP:tl.constexpr=1,DP=0,HAS_PARTIAL:tl.constexpr=False,Read=0,USE_READ:tl.constexpr=False):
    tile=tl.program_id(0);b=tl.program_id(1);d=tl.arange(0,BD)
    cs_total=tl.full((BD,),0.,tl.float32);ws_total=tl.full((BD,),0.,tl.float32)
    ss_total=tl.full((BD,),0.,tl.float32);ts_total=tl.full((BD,),0.,tl.float32)
    for sub in range(GROUP):
        local=(tile*GROUP+sub)*BR+tl.arange(0,BR);r=b*N+local
        valid=(local[:,None]<N)&(d[None,:]<D)
        total=tl.full((BR,BD),0.,tl.float32)
        if USE_READ:
            total=tl.load(Read+r[:,None]*D+d[None,:],valid,0)
        else:
            for k in range(K):
                v=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r[:,None]*D+d[None,:],valid,0).to(tl.float32)
                total+=tl.load(Weights+r*K+k,local<N,0)[:,None]*v
        x=total.to(GO.dtype.element_ty).to(tl.float32)
        inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS)
        w=tl.load(W+d,d<D,0).to(tl.float32)
        s=tl.load(S+b*SS+d,d<D,0).to(tl.float32)
        go=tl.load(GO+r[:,None]*D+d[None,:],valid,0).to(tl.float32)
        gn=(go*(1+s[None,:])).to(GO.dtype.element_ty).to(tl.float32)
        normalized=((x*inv[:,None])*w[None,:]).to(GO.dtype.element_ty).to(tl.float32)
        ss_total+=tl.sum(go*normalized,0)
        ts_total+=tl.sum(go,0)
        ws_total+=tl.sum((gn*x)*inv[:,None],0)
        weighted=gn*w[None,:];dot=tl.sum(weighted*x,1)
        gx=(inv[:,None]*(weighted-x*(inv[:,None]*inv[:,None]/D)*dot[:,None])).to(GO.dtype.element_ty).to(tl.float32)
        q=tl.load(Q+d,d<D,0).to(tl.float32)
        dw=tl.load(DW+d,d<D,0).to(tl.float32);qw=q*dw
        common=tl.full((BR,BD),0.,tl.float32)
        for k in range(K):
            v=tl.load(tl.load(Values+k).to(tl.pointer_type(DT))+r[:,None]*D+d[None,:],valid,0).to(tl.float32)
            a=tl.load(Weights+r*K+k,local<N,0);vi=tl.load(Inv+r*K+k,local<N,0)
            dl=a*tl.sum(gx*(v-total),1)
            vd=tl.sum(v*qw[None,:],1)
            dx=a[:,None]*gx+dl[:,None]*(vi[:,None]*qw[None,:]-v*(vi[:,None]*vi[:,None]*vi[:,None]/D)*vd[:,None])
            if MERGE and HAS_PARTIAL:
                if k<MERGE_K:destination=tl.load(DValues+k).to(tl.pointer_type(DT))+r[:,None]*D+d[None,:]
                else:destination=DP+r[:,None]*D+d[None,:]
            else:destination=tl.load(DValues+k).to(tl.pointer_type(DT))+r[:,None]*D+d[None,:]
            if MERGE:
                if k<MERGE_K:
                    dx=dx.to(DT).to(tl.float32)
                    if EXISTING:dx=tl.load(destination,valid,0).to(tl.float32)+dx
            tl.store(destination,dx,valid)
            common+=(dl*vi)[:,None]*v
        cs_total+=tl.sum(common,0)
    partial=(b*tl.cdiv(N,BR*GROUP)+tile)*D+d
    tl.store(Common+partial,cs_total,d<D);tl.store(WRows+partial,ws_total,d<D)
    tl.store(SRows+partial,ss_total,d<D);tl.store(TRows+partial,ts_total,d<D)


@torch.library.custom_op('v31rf::depth_ada_forward',mutates_args=())
def _fwd(values:list[torch.Tensor],q:torch.Tensor,dw:torch.Tensor,w:torch.Tensor,s:torch.Tensor,t:torch.Tensor,
         eps:float,cached:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    r,args=_shape(values);y=torch.empty_like(values[0]);ptrs=make_table(values)
    inv=torch.empty((r,len(values)),device=q.device,dtype=torch.float32);weights=torch.empty_like(inv)
    read=torch.empty(cache_shape(values),device=q.device,dtype=torch.float32)
    _online[(r,)](ptrs,q,dw,weights,inv,y,**args,EPS=eps,Cached=cached,R=r,CACHED=cached.shape[0],
                   W=w,S=s,T=t,N=values[0].shape[-2],SS=s.stride(0),TS=t.stride(0),MODE=1,Read=read,SAVE_READ=read.numel()>0,
                   num_warps=4,enable_fp_fusion=False)
    return y,weights,inv,ptrs,read


@_fwd.register_fake
def _fake(values,q,dw,w,s,t,eps,cached):
    r=values[0].numel()//values[0].shape[-1]
    return (torch.empty_like(values[0]),torch.empty((r,len(values)),device=q.device,dtype=torch.float32),
            torch.empty((r,len(values)),device=q.device,dtype=torch.float32),torch.empty((len(values),),device=q.device,dtype=torch.uint64),torch.empty(cache_shape(values),device=q.device,dtype=torch.float32))


@torch.library.custom_op('v31rf::depth_ada_backward',mutates_args=())
def _bwd(values:list[torch.Tensor],q:torch.Tensor,dw:torch.Tensor,w:torch.Tensor,s:torch.Tensor,t:torch.Tensor,
         weights:torch.Tensor,inv:torch.Tensor,go:torch.Tensor,eps:float,ptrs:torch.Tensor|None=None,read:torch.Tensor|None=None)->tuple[list[torch.Tensor],torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    r,args=_shape(values);dv=[torch.empty_like(x) for x in values]
    br=4;group=4;n=values[0].shape[-2];b=values[0].shape[0];tiles=triton.cdiv(n,br*group)
    common=torch.empty((b,tiles,q.numel()),device=q.device,dtype=torch.float32)
    wr=torch.empty_like(common);sr=torch.empty_like(common);tr=torch.empty_like(common)
    dq,ddw,denw,ds,dt=map(torch.empty_like,(q,dw,w,s,t));go=go.contiguous()
    _backward[(tiles,b)](make_table(values) if ptrs is None else ptrs,q,dw,w,s,weights,inv,go,make_table(dv),common,wr,sr,tr,**args,
                    N=n,SS=s.stride(0),EPS=eps,BR=br,GROUP=group,Read=read if read is not None else go,USE_READ=read is not None and read.numel()>0,num_warps=4,enable_fp_fusion=False)
    _grad_params[(triton.cdiv(q.numel(),128),)](common,q,dw,dq,ddw,b*tiles,q.numel(),BC=128,BR=32,num_warps=4)
    _reduce_w[(triton.cdiv(w.numel(),128),1)](wr,denw,b*tiles,w.numel(),1,BC=128,BR=32,num_warps=4)
    _reduce_pair[(triton.cdiv(w.numel(),128),b)](sr,tr,ds,dt,tiles,w.numel(),BC=128,BR=32,num_warps=4)
    return dv,dq,ddw,denw,ds,dt


@_bwd.register_fake
def _fake_bwd(values,q,dw,w,s,t,weights,inv,go,eps,ptrs=None,read=None):
    return [torch.empty_like(v) for v in values],*(torch.empty_like(v) for v in (q,dw,w,s,t))


def _setup(ctx,inputs,output):
    values,q,dw,w,s,t,eps,cached=inputs
    ctx.count=len(values);ctx.eps=eps;ctx.save_for_backward(*values,q,dw,w,s,t,*output[1:])
    ctx.mark_non_differentiable(*output[1:])


def _autograd(ctx,go,unused_weights,unused_inv,unused_ptrs,unused_read):
    saved=ctx.saved_tensors;k=ctx.count
    return (*_bwd(list(saved[:k]),*saved[k:-2],go,ctx.eps,saved[-2],saved[-1]),None,None)


_fwd.register_autograd(_autograd,setup_context=_setup)


def depth_ada(values,q,dw,w,s,t,eps,*,cached=None):
    from mirage.ops.depth_triton import inference_kernels,inference_read
    if inference_kernels():
        return inference_read(values,q,dw,eps,cached=cached,norm_weight=w,scale=s,shift=t)
    if cached is None:cached=torch.empty((0,values[0].numel()//values[0].shape[-1]),device=q.device,dtype=torch.float32)
    return _fwd(list(values),q,dw,w,s,t,eps,cached)[0]


def depth_final(values,q,dw,w,eps,*,cached=None):
    from mirage.ops.depth_triton import inference_kernels,inference_read
    if inference_kernels():
        return inference_read(values,q,dw,eps,cached=cached,norm_weight=w)
    # Zero modulation turns the fused read+AdaRMS into read+RMSNorm. Backward
    # still includes all history RMS derivatives and the final norm gain.
    b=values[0].shape[0]
    zero=torch.zeros((b,w.numel()),device=w.device,dtype=torch.float32)
    return depth_ada(values,q,dw,w,zero,zero,eps,cached=cached)
