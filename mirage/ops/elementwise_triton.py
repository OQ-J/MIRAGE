"""SwiGLU on a contiguous combined projection [*,2*FFN]; save only that projection."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _forward(X,Y,N,D:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);offset=(i//D)*(2*D)+i%D
    a=tl.load(X+offset,i<N,0).to(tl.float32);b=tl.load(X+offset+D,i<N,0).to(tl.float32)
    sigmoid=1/(1+tl.exp(-a))
    silu=(a*sigmoid).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,silu*b,i<N)


@triton.jit(do_not_specialize=['N'])
def _backward(X,GO,DX,N,D:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);offset=(i//D)*(2*D)+i%D
    a=tl.load(X+offset,i<N,0).to(tl.float32);b=tl.load(X+offset+D,i<N,0).to(tl.float32)
    go=tl.load(GO+i,i<N,0).to(tl.float32);sigmoid=1/(1+tl.exp(-a))
    silu=(a*sigmoid).to(X.dtype.element_ty).to(tl.float32)
    gs=(go*b).to(X.dtype.element_ty).to(tl.float32)
    tl.store(DX+offset,gs*sigmoid*(1+a*(1-sigmoid)),i<N)
    tl.store(DX+offset+D,go*silu,i<N)


@torch.library.custom_op('v31rf::swiglu_forward',mutates_args=())
def _fwd(x:torch.Tensor)->torch.Tensor:
    d=x.shape[-1]//2;y=torch.empty((*x.shape[:-1],d),device=x.device,dtype=x.dtype)
    _forward[(triton.cdiv(y.numel(),512),)](x,y,y.numel(),d,BLOCK=512,num_warps=4,enable_fp_fusion=False)
    return y


@_fwd.register_fake
def _fake(x):return torch.empty((*x.shape[:-1],x.shape[-1]//2),device=x.device,dtype=x.dtype)


@torch.library.custom_op('v31rf::swiglu_backward',mutates_args=())
def _bwd(x:torch.Tensor,go:torch.Tensor)->torch.Tensor:
    dx=torch.empty_like(x);d=x.shape[-1]//2
    _backward[(triton.cdiv(go.numel(),512),)](x,go.contiguous(),dx,go.numel(),d,BLOCK=512,num_warps=4,enable_fp_fusion=False)
    return dx


@_bwd.register_fake
def _fake_bwd(x,go):return torch.empty_like(x)


def _setup(ctx,inputs,output):ctx.save_for_backward(inputs[0])
def _autograd(ctx,go):return _bwd(ctx.saved_tensors[0],go)
_fwd.register_autograd(_autograd,setup_context=_setup)


def swiglu(x):return _fwd(x.contiguous())
