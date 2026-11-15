"""Joint Q/K RMSNorm+RoPE and V repack; combined input-gradient and partial gain reductions."""
import torch,os
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['N','T'])
def _forward(X,WQ,WK,C,S,Q,K,V,IQ,IK,N,T,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
             WIDTH:tl.constexpr,BD:tl.constexpr,EPS:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR);h=tl.program_id(1);d=tl.arange(0,BD)
    p=r%N;b=r//N;valid=(r[:,None]<T)&(d[None,:]<D)
    c=tl.load(C+p[:,None]*(D//2)+d[None,:]//2,valid,0)
    s=tl.load(S+p[:,None]*(D//2)+d[None,:]//2,valid,0)
    idx=tl.broadcast_to((d[None,:]^1),(BR,BD))
    if h<H:
        x=tl.load(X+r[:,None]*WIDTH+h*D+d[None,:],valid,0).to(tl.float32)
        w=tl.load(WQ+h*D+d,d<D,0).to(tl.float32)
        inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS);a=(x*inv[:,None])*w[None,:]
        partner=tl.gather(a,idx,1)
        y=a*c+tl.where(d[None,:]%2==0,-partner,partner)*s
        tl.store(Q+((b[:,None]*H+h)*N+p[:,None])*D+d[None,:],y,valid)
        tl.store(IQ+r*H+h,inv,r<T)
    if h<HK:
        x=tl.load(X+r[:,None]*WIDTH+H*D+h*D+d[None,:],valid,0).to(tl.float32)
        w=tl.load(WK+h*D+d,d<D,0).to(tl.float32)
        inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS);a=(x*inv[:,None])*w[None,:]
        partner=tl.gather(a,idx,1)
        y=a*c+tl.where(d[None,:]%2==0,-partner,partner)*s
        offset=((b[:,None]*HK+h)*N+p[:,None])*D+d[None,:]
        tl.store(K+offset,y,valid);tl.store(IK+r*HK+h,inv,r<T)
        v=tl.load(X+r[:,None]*WIDTH+(H+HK)*D+h*D+d[None,:],valid,0)
        tl.store(V+offset,v,valid)

@triton.jit(do_not_specialize=['N','T'])
def _backward(X,WQ,WK,C,S,IQ,IK,GQ,GK,GV,DX,PQ,PK,N,T,H:tl.constexpr,HK:tl.constexpr,
              D:tl.constexpr,WIDTH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    block=tl.program_id(0);h=tl.program_id(1);r=block*BR+tl.arange(0,BR)
    d=tl.arange(0,BD);valid=(r[:,None]<T)&(d[None,:]<D);p=r%N;b=r//N
    c=tl.load(C+p[:,None]*(D//2)+d[None,:]//2,valid,0)
    s=tl.load(S+p[:,None]*(D//2)+d[None,:]//2,valid,0)
    if h<H:
        x=tl.load(X+r[:,None]*WIDTH+h*D+d[None,:],valid,0).to(tl.float32)
        w=tl.load(WQ+h*D+d,d<D,0).to(tl.float32)
        inv=tl.load(IQ+r*H+h,r<T,0)
        go=tl.load(GQ+((b[:,None]*H+h)*N+p[:,None])*D+d[None,:],valid,0).to(tl.float32)
        partner=tl.gather(go,tl.broadcast_to((d^1)[None,:],(BR,BD)),1)
        gn=go*c+tl.where(d[None,:]%2==0,partner,-partner)*s
        weighted=gn*w[None,:];dot=tl.sum(weighted*x,1)
        dx=inv[:,None]*(weighted-x*(inv[:,None]*inv[:,None]/D)*dot[:,None])
        tl.store(DX+r[:,None]*WIDTH+h*D+d[None,:],dx,valid)
        tl.store(PQ+(block*H+h)*D+d,tl.sum((gn*x)*inv[:,None],0),d<D)
    if h<HK:
        x=tl.load(X+r[:,None]*WIDTH+H*D+h*D+d[None,:],valid,0).to(tl.float32)
        w=tl.load(WK+h*D+d,d<D,0).to(tl.float32)
        inv=tl.load(IK+r*HK+h,r<T,0)
        go=tl.load(GK+((b[:,None]*HK+h)*N+p[:,None])*D+d[None,:],valid,0).to(tl.float32)
        partner=tl.gather(go,tl.broadcast_to((d^1)[None,:],(BR,BD)),1)
        gn=go*c+tl.where(d[None,:]%2==0,partner,-partner)*s
        weighted=gn*w[None,:];dot=tl.sum(weighted*x,1)
        dx=inv[:,None]*(weighted-x*(inv[:,None]*inv[:,None]/D)*dot[:,None])
        tl.store(DX+r[:,None]*WIDTH+H*D+h*D+d[None,:],dx,valid)
        tl.store(PK+(block*HK+h)*D+d,tl.sum((gn*x)*inv[:,None],0),d<D)
        gv=tl.load(GV+((b[:,None]*HK+h)*N+p[:,None])*D+d[None,:],valid,0)
        tl.store(DX+r[:,None]*WIDTH+(H+HK)*D+h*D+d[None,:],gv,valid)

@triton.jit(do_not_specialize=['R'])
def _reduce(PQ,PK,WQ,WK,R,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BC:tl.constexpr,BR:tl.constexpr):
    head=tl.program_id(1);d=tl.program_id(0)*BC+tl.arange(0,BC);rows=tl.arange(0,BR)
    total=tl.full((BC,),0.,tl.float32)
    for start in range(tl.cdiv(R,BR)):
        r=start*BR+rows;valid=(r[:,None]<R)&(d[None,:]<D)
        if head<H:v=tl.load(PQ+(r[:,None]*H+head)*D+d[None,:],valid,0)
        else:v=tl.load(PK+(r[:,None]*HK+head-H)*D+d[None,:],valid,0)
        total+=tl.sum(v,0)
    if head<H:tl.store(WQ+head*D+d,total,d<D)
    else:tl.store(WK+(head-H)*D+d,total,d<D)

def _args(x,wq,wk):
    return dict(N=x.shape[1],T=x.shape[0]*x.shape[1],H=wq.shape[0],HK=wk.shape[0],D=wq.shape[1],
                WIDTH=x.shape[-1],BD=triton.next_power_of_2(wq.shape[1]))

@torch.library.custom_op('v31rf::qkv_norm_rope_forward',mutates_args=())
def _fwd(x:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,c:torch.Tensor,s:torch.Tensor,eps:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    a=_args(x,wq,wk);b,n=x.shape[:2];h,hk,d=a['H'],a['HK'],a['D']
    q=torch.empty((b,h,n,d),device=x.device,dtype=x.dtype)
    k=torch.empty((b,hk,n,d),device=x.device,dtype=x.dtype);v=torch.empty_like(k)
    iq=torch.empty((b*n,h),device=x.device,dtype=torch.float32);ik=torch.empty((b*n,hk),device=x.device,dtype=torch.float32)
    br=int(os.environ.get('MIRAGE_NORM_FWD_ROWS','4'));warps=int(os.environ.get('MIRAGE_NORM_WARPS','4'))
    if br not in (1,2,4,8) or warps not in (1,2,4,8):raise ValueError('Invalid V32 Norm launch configuration')
    _forward[(triton.cdiv(b*n,br),max(h,hk))](x,wq,wk,c,s,q,k,v,iq,ik,**a,EPS=eps,BR=br,num_warps=warps,enable_fp_fusion=False)
    return q,k,v,iq,ik

@_fwd.register_fake
def _fake(x,wq,wk,c,s,eps):
    b,n=x.shape[:2];h,d=wq.shape;hk=wk.shape[0]
    return (torch.empty((b,h,n,d),device=x.device,dtype=x.dtype),torch.empty((b,hk,n,d),device=x.device,dtype=x.dtype),
            torch.empty((b,hk,n,d),device=x.device,dtype=x.dtype),torch.empty((b*n,h),device=x.device,dtype=torch.float32),
            torch.empty((b*n,hk),device=x.device,dtype=torch.float32))

@torch.library.custom_op('v31rf::qkv_norm_rope_backward',mutates_args=())
def _bwd(x:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,c:torch.Tensor,s:torch.Tensor,iq:torch.Tensor,ik:torch.Tensor,
         gq:torch.Tensor,gk:torch.Tensor,gv:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    a=_args(x,wq,wk);br=4;tiles=triton.cdiv(a['T'],br)
    dx=torch.empty_like(x);dq=torch.empty_like(wq);dk=torch.empty_like(wk)
    pq=torch.empty((tiles,a['H'],a['D']),device=x.device,dtype=torch.float32)
    pk=torch.empty((tiles,a['HK'],a['D']),device=x.device,dtype=torch.float32)
    _backward[(tiles,max(a['H'],a['HK']))](x,wq,wk,c,s,iq,ik,gq.contiguous(),gk.contiguous(),gv.contiguous(),dx,pq,pk,**a,BR=br,num_warps=4,enable_fp_fusion=False)
    _reduce[(triton.cdiv(a['D'],128),a['H']+a['HK'])](pq,pk,dq,dk,tiles,a['H'],a['HK'],a['D'],BC=128,BR=32,num_warps=4)
    return dx,dq,dk

@_bwd.register_fake
def _fake_bwd(x,wq,wk,c,s,iq,ik,gq,gk,gv):return torch.empty_like(x),torch.empty_like(wq),torch.empty_like(wk)

def _setup(ctx,inputs,output):
    x,wq,wk,c,s,eps=inputs;ctx.save_for_backward(x,wq,wk,c,s,output[3],output[4]);ctx.mark_non_differentiable(*output[3:])

def _autograd(ctx,gq,gk,gv,unused_iq,unused_ik):
    return (*_bwd(*ctx.saved_tensors,gq,gk,gv),None,None,None)

_fwd.register_autograd(_autograd,setup_context=_setup)

def qkv_norm_rope(x,wq,wk,c,s,eps):return _fwd(x,wq,wk,c,s,eps)[:3]
