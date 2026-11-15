"""FP32 direct-velocity RF interpolation and loss; RNG stays in PyTorch."""
import torch
import triton
import triton.language as tl

@triton.jit
def _inputs(Z,T,E,X,V,N:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    z=tl.load(Z+i,valid,0).to(tl.float32);e=tl.load(E+i,valid,0).to(tl.float32)
    t=tl.load(T+i//S,valid,0).to(tl.float32)
    tl.store(X+i,(1-t)*z+t*e,valid);tl.store(V+i,e-z,valid)

def inputs(z,t,noise):
    x=torch.empty_like(z,dtype=torch.float32);v=torch.empty_like(x)
    _inputs[(triton.cdiv(z.numel(),1024),)](z,t,noise,x,v,z.numel(),z[0].numel(),1024,enable_fp_fusion=False)
    return x,t,v

@triton.jit
def _loss(P,T,Parts,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    diff=tl.load(P+i,valid,0).to(tl.float32)-tl.load(T+i,valid,0).to(tl.float32)
    tl.store(Parts+tl.program_id(0),tl.sum(diff*diff,0)/N)
@triton.jit
def _loss_sum(Parts,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK);tl.store(Y,tl.sum(tl.load(Parts+i,i<N,0),0))
@triton.jit
def _loss_grad(P,T,G,DP,DT,N:tl.constexpr,BLOCK:tl.constexpr,HAS_DT:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    diff=tl.load(P+i,valid,0).to(tl.float32)-tl.load(T+i,valid,0).to(tl.float32)
    v=(diff*(tl.load(G)/N))*2
    tl.store(DP+i,v,valid)
    if HAS_DT:tl.store(DT+i,-v,valid)

@torch.library.custom_op('v31rf::velocity_mse',mutates_args=())
def mse(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    n=pred.numel();tiles=triton.cdiv(n,4096)
    parts=torch.empty((tiles,),device=pred.device,dtype=torch.float32);y=torch.empty((),device=pred.device,dtype=torch.float32)
    _loss[(tiles,)](pred,target,parts,n,4096,enable_fp_fusion=False)
    _loss_sum[(1,)](parts,y,tiles,triton.next_power_of_2(tiles))
    return y
@mse.register_fake
def _fake(pred,target):return torch.empty((),device=pred.device,dtype=torch.float32)
def _setup(ctx,inputs,output):
    ctx.save_for_backward(*inputs);ctx.target_grad=inputs[1].requires_grad
@torch.library.custom_op('v31rf::velocity_mse_backward',mutates_args=())
def _mse_grad(p:torch.Tensor,t:torch.Tensor,go:torch.Tensor,target_grad:bool)->tuple[torch.Tensor,torch.Tensor]:
    dp=torch.empty_like(p);dt=torch.empty_like(t) if target_grad else torch.empty((0,),device=t.device,dtype=t.dtype)
    _loss_grad[(triton.cdiv(p.numel(),1024),)](p,t,go,dp,dt,p.numel(),1024,target_grad,enable_fp_fusion=False)
    return dp,dt
@_mse_grad.register_fake
def _fake_grad(p,t,go,target_grad):return torch.empty_like(p),torch.empty_like(t) if target_grad else torch.empty((0,),device=t.device,dtype=t.dtype)
def _back(ctx,go):
    dp,dt=_mse_grad(*ctx.saved_tensors,go,ctx.target_grad)
    return dp,dt if ctx.target_grad else None
mse.register_autograd(_back,setup_context=_setup)
