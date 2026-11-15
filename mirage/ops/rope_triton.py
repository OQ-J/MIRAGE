"""Token-major Q/K RMSNorm + fixed 2D rotary transform, FP32 math, BF16 output."""
import torch
import triton
import triton.language as tl
from mirage.ops.norm_triton import _reduce_w


@triton.jit(do_not_specialize=['N'])
def _forward(X,W,C,S,Y,Inv,D:tl.constexpr,H:tl.constexpr,N,STRIDE:tl.constexpr,
             BD:tl.constexpr,EPS:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD);h=r%H;p=(r//H)%N
    off=(r//H)*STRIDE+h*D
    x=tl.load(X+off+d,d<D,0).to(tl.float32)
    xp=tl.load(X+off+(d^1),d<D,0).to(tl.float32)
    w=tl.load(W+h*D+d,d<D,0).to(tl.float32)
    wp=tl.load(W+h*D+(d^1),d<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    c=tl.load(C+p*(D//2)+d//2,d<D,0)
    s=tl.load(S+p*(D//2)+d//2,d<D,0)
    a=(x*inv)*w;b=(xp*inv)*wp
    y=a*c+tl.where(d%2==0,-b,b)*s
    out=((r//(H*N))*H+h)*N+p
    tl.store(Y+out*D+d,y,d<D);tl.store(Inv+r,inv)


@triton.jit(do_not_specialize=['N'])
def _backward(X,W,C,S,Inv,GO,DX,WR,D:tl.constexpr,H:tl.constexpr,N,
              STRIDE:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD);h=r%H;p=(r//H)%N
    x=tl.load(X+(r//H)*STRIDE+h*D+d,d<D,0).to(tl.float32)
    w=tl.load(W+h*D+d,d<D,0).to(tl.float32);inv=tl.load(Inv+r)
    out=((r//(H*N))*H+h)*N+p
    go=tl.load(GO+out*D+d,d<D,0).to(tl.float32)
    gp=tl.load(GO+out*D+(d^1),d<D,0).to(tl.float32)
    c=tl.load(C+p*(D//2)+d//2,d<D,0)
    s=tl.load(S+p*(D//2)+d//2,d<D,0)
    gn=go*c+tl.where(d%2==0,gp,-gp)*s
    weighted=gn*w;dot=tl.sum(weighted*x,0)
    dx=inv*(weighted-x*(inv*inv/D)*dot)
    tl.store(DX+r*D+d,dx,d<D)
    tl.store(WR+r*D+d,(gn*x)*inv,d<D)


def _args(x):
    return dict(D=x.shape[-1],H=x.shape[-2],N=x.shape[-3],
                STRIDE=x.stride(-3),BD=triton.next_power_of_2(x.shape[-1]))


@torch.library.custom_op('v31rf::rope_norm_forward',mutates_args=())
def _fwd(x:torch.Tensor,w:torch.Tensor,c:torch.Tensor,s:torch.Tensor,eps:float)->tuple[torch.Tensor,torch.Tensor]:
    r=x.numel()//x.shape[-1]
    b,n,h,d=x.shape
    y=torch.empty((b,h,n,d),device=x.device,dtype=x.dtype)
    inv=torch.empty(r,device=x.device,dtype=torch.float32)
    _forward[(r,)](x,w,c,s,y,inv,**_args(x),EPS=eps,num_warps=4,enable_fp_fusion=False)
    return y.transpose(1,2),inv


@_fwd.register_fake
def _fake(x,w,c,s,eps):
    b,n,h,d=x.shape
    return torch.empty((b,h,n,d),device=x.device,dtype=x.dtype).transpose(1,2),torch.empty(x.numel()//x.shape[-1],device=x.device,dtype=torch.float32)


@torch.library.custom_op('v31rf::rope_norm_backward',mutates_args=())
def _bwd(x:torch.Tensor,w:torch.Tensor,c:torch.Tensor,s:torch.Tensor,inv:torch.Tensor,go:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    args=_args(x);r=x.numel()//x.shape[-1]
    dx=torch.empty(x.shape,device=x.device,dtype=x.dtype);dw=torch.empty_like(w)
    rows=torch.empty(x.shape,device=x.device,dtype=torch.float32)
    _backward[(r,)](x,w,c,s,inv,go.transpose(1,2).contiguous(),dx,rows,**args,num_warps=4,enable_fp_fusion=False)
    _reduce_w[(triton.cdiv(args['D'],128),args['H'])](rows,dw,r,args['D'],args['H'],BC=128,BR=32,num_warps=4)
    return dx,dw


@_bwd.register_fake
def _fake_bwd(x,w,c,s,inv,go):
    return torch.empty(x.shape,device=x.device,dtype=x.dtype),torch.empty_like(w)


def _setup(ctx,inputs,output):
    x,w,c,s,eps=inputs
    ctx.save_for_backward(x,w,c,s,output[1]);ctx.mark_non_differentiable(output[1])


def _autograd(ctx,go,unused):
    x,w,c,s,inv=ctx.saved_tensors
    dx,dw=_bwd(x,w,c,s,inv,go)
    return dx,dw,None,None,None


_fwd.register_autograd(_autograd,setup_context=_setup)


def head_norm_rope(x,w,c,s,eps):return _fwd(x,w,c,s,eps)[0]
