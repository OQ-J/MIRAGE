"""FP32-stat RMSNorm, token-major head RMSNorm and BF16-boundary-preserving AdaRMS."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _forward(X,W,S,T,Y,Inv,D:tl.constexpr,G:tl.constexpr,STRIDE:tl.constexpr,N,
             SS:tl.constexpr,TS:tl.constexpr,BD:tl.constexpr,EPS:tl.constexpr,ADA:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD);group=r%G
    x=tl.load(X+(r//G)*STRIDE+group*D+d,d<D,0).to(tl.float32)
    w=tl.load(W+group*D+d,d<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS);y=(x*inv)*w
    if ADA:
        b=r//N;s=tl.load(S+b*SS+d,d<D,0).to(tl.float32)
        t=tl.load(T+b*TS+d,d<D,0).to(tl.float32)
        y=y.to(X.dtype.element_ty).to(tl.float32)*(1+s)+t
    tl.store(Y+r*D+d,y,d<D);tl.store(Inv+r,inv)


@triton.jit(do_not_specialize=['N'])
def _backward(X,W,S,Inv,GO,DX,DWRows,DSRows,D:tl.constexpr,G:tl.constexpr,STRIDE:tl.constexpr,
              DTRows,N,SS:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr,ADA:tl.constexpr):
    tile=tl.program_id(0);index=tl.program_id(1);rows=tile*BR+tl.arange(0,BR);d=tl.arange(0,BD)
    if ADA:b=index;group=0
    else:b=0;group=index
    r=(b*N+rows)*G+group;valid=(rows[:,None]<N)&(d[None,:]<D)
    x=tl.load(X+(b*N+rows[:,None])*STRIDE+group*D+d[None,:],valid,0).to(tl.float32)
    w=tl.load(W+group*D+d,d<D,0).to(tl.float32);inv=tl.load(Inv+r,rows<N,0)
    go=tl.load(GO+r[:,None]*D+d[None,:],valid,0).to(tl.float32);gn=go
    if ADA:
        s=tl.load(S+b*SS+d,d<D,0).to(tl.float32)
        gn=(go*(1+s[None,:])).to(X.dtype.element_ty).to(tl.float32)
        normalized=((x*inv[:,None])*w[None,:]).to(X.dtype.element_ty).to(tl.float32)
        partial=(b*tl.cdiv(N,BR)+tile)*D+d
        tl.store(DSRows+partial,tl.sum(go*normalized,0),d<D)
        tl.store(DTRows+partial,tl.sum(go,0),d<D)
    weighted=gn*w[None,:];dot=tl.sum(weighted*x,1)
    dx=inv[:,None]*(weighted-x*(inv[:,None]*inv[:,None]/D)*dot[:,None])
    tl.store(DX+r[:,None]*D+d[None,:],dx,valid)
    partial=((b*tl.cdiv(N,BR)+tile)*G+group)*D+d
    tl.store(DWRows+partial,tl.sum((gn*x)*inv[:,None],0),d<D)


@triton.jit(do_not_specialize=['R'])
def _reduce_w(Rows,DW,R,D:tl.constexpr,G:tl.constexpr,BC:tl.constexpr,BR:tl.constexpr):
    d=tl.program_id(0)*BC+tl.arange(0,BC);g=tl.program_id(1);rows=tl.arange(0,BR)
    total=tl.full((BC,),0.,tl.float32)
    for start in range(tl.cdiv(R//G,BR)):
        t=start*BR+rows;r=t*G+g
        v=tl.load(Rows+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
        total+=tl.sum(v,0)
    tl.store(DW+g*D+d,total,d<D)


@triton.jit(do_not_specialize=['N'])
def _reduce_ada(Rows,GO,DS,DT,N,D:tl.constexpr,BC:tl.constexpr,BR:tl.constexpr):
    d=tl.program_id(0)*BC+tl.arange(0,BC);b=tl.program_id(1);rows=tl.arange(0,BR)
    s=tl.full((BC,),0.,tl.float32);t=tl.full((BC,),0.,tl.float32)
    for start in range(tl.cdiv(N,BR)):
        n=start*BR+rows;offset=(b*N+n[:,None])*D+d[None,:];valid=(n[:,None]<N)&(d[None,:]<D)
        s+=tl.sum(tl.load(Rows+offset,valid,0),0)
        t+=tl.sum(tl.load(GO+offset,valid,0).to(tl.float32),0)
    tl.store(DS+b*D+d,s,d<D);tl.store(DT+b*D+d,t,d<D)


@triton.jit(do_not_specialize=['N'])
def _reduce_pair(SRows,TRows,DS,DT,N,D:tl.constexpr,BC:tl.constexpr,BR:tl.constexpr):
    d=tl.program_id(0)*BC+tl.arange(0,BC);b=tl.program_id(1);rows=tl.arange(0,BR)
    s=tl.full((BC,),0.,tl.float32);t=tl.full((BC,),0.,tl.float32)
    for start in range(tl.cdiv(N,BR)):
        r=start*BR+rows;off=(b*N+r[:,None])*D+d[None,:];valid=(r[:,None]<N)&(d[None,:]<D)
        s+=tl.sum(tl.load(SRows+off,valid,0),0);t+=tl.sum(tl.load(TRows+off,valid,0),0)
    tl.store(DS+b*D+d,s,d<D);tl.store(DT+b*D+d,t,d<D)


def _args(x,w,s,t,head,ada):
    d=x.shape[-1];g=w.shape[0] if head else 1
    stride=x.stride(-3) if head else x.stride(-2)
    return dict(D=d,G=g,STRIDE=stride,N=x.shape[-2] if ada else 1,
                SS=s.stride(0) if ada else 0,TS=t.stride(0) if ada else 0,
                BD=triton.next_power_of_2(d),ADA=ada)


@torch.library.custom_op('v31rf::norm_forward',mutates_args=())
def _fwd(x:torch.Tensor,w:torch.Tensor,s:torch.Tensor,t:torch.Tensor,eps:float,
         head:bool,fp32:bool,ada:bool)->tuple[torch.Tensor,torch.Tensor]:
    r=x.numel()//x.shape[-1];args=_args(x,w,s,t,head,ada)
    y=torch.empty(x.shape,device=x.device,dtype=torch.float32 if fp32 else x.dtype)
    inv=torch.empty((r,),device=x.device,dtype=torch.float32)
    _forward[(r,)](x,w,s,t,y,inv,**args,EPS=eps,num_warps=4,enable_fp_fusion=False)
    return y,inv


@_fwd.register_fake
def _fake(x,w,s,t,eps,head,fp32,ada):
    return (torch.empty(x.shape,device=x.device,dtype=torch.float32 if fp32 else x.dtype),
            torch.empty((x.numel()//x.shape[-1],),device=x.device,dtype=torch.float32))


@torch.library.custom_op('v31rf::norm_backward',mutates_args=())
def _bwd(x:torch.Tensor,w:torch.Tensor,s:torch.Tensor,t:torch.Tensor,inv:torch.Tensor,go:torch.Tensor,
         head:bool,ada:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    r=x.numel()//x.shape[-1];args=_args(x,w,s,t,head,ada);d=args['D'];g=args['G']
    dx=torch.empty(x.shape,device=x.device,dtype=x.dtype);dw=torch.empty_like(w)
    ds=torch.empty_like(s);dt=torch.empty_like(t);go=go.contiguous()
    br=4;length=args['N'] if ada else r//g;b=s.shape[0] if ada else 1;tiles=triton.cdiv(length,br)
    wr=torch.empty((b*tiles,g,d),device=x.device,dtype=torch.float32)
    sr=torch.empty((b,tiles,d),device=x.device,dtype=torch.float32) if ada else torch.empty(0,device=x.device)
    tr=torch.empty_like(sr)
    backward={k:v for k,v in args.items() if k!='TS'}
    backward['N']=length
    _backward[(tiles,b if ada else g)](x,w,s,inv,go,dx,wr,sr,DTRows=tr,**backward,BR=br,num_warps=4,enable_fp_fusion=False)
    _reduce_w[(triton.cdiv(d,128),g)](wr,dw,b*tiles*g,d,g,BC=128,BR=32,num_warps=4)
    if ada:_reduce_pair[(triton.cdiv(d,128),b)](sr,tr,ds,dt,tiles,d,BC=128,BR=32,num_warps=4)
    return dx,dw,ds,dt


@_bwd.register_fake
def _fake_bwd(x,w,s,t,inv,go,head,ada):
    return torch.empty(x.shape,device=x.device,dtype=x.dtype),torch.empty_like(w),torch.empty_like(s),torch.empty_like(t)


def _setup(ctx,inputs,output):
    x,w,s,t,eps,head,fp32,ada=inputs
    ctx.head=head;ctx.ada=ada;ctx.save_for_backward(x,w,s,t,output[1]);ctx.mark_non_differentiable(output[1])


def _autograd(ctx,go,unused):
    x,w,s,t,inv=ctx.saved_tensors
    dx,dw,ds,dt=_bwd(x,w,s,t,inv,go,ctx.head,ctx.ada)
    return dx,dw,ds,dt,None,None,None,None


_fwd.register_autograd(_autograd,setup_context=_setup)


def rms(x,w,eps,*,fp32=False):
    empty=x.new_empty(0)
    return _fwd(x,w,empty,empty,eps,w.ndim==2,fp32,False)[0]


def ada_rms(x,w,s,t,eps):return _fwd(x,w,s,t,eps,False,False,True)[0]
