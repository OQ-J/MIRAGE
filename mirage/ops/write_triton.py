import torch
import triton
import triton.language as tl


@triton.jit
def _residual(P,D,M,Y,N:tl.constexpr,C:tl.constexpr,HAS_P:tl.constexpr,HAS_M:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    d=tl.load(D+i,valid,0)
    if HAS_M:d=tl.where(tl.load(M+i//C,valid,0),d,0.)
    if HAS_P:d=tl.load(P+i,valid,0).to(tl.float32)+d.to(tl.float32)
    tl.store(Y+i,d,valid)


@torch.library.custom_op('v31rf::residual_write',mutates_args=())
def residual(p:torch.Tensor|None,d:torch.Tensor,m:torch.Tensor|None)->torch.Tensor:
    y=torch.empty_like(d)
    _residual[(triton.cdiv(d.numel(),1024),)](p if p is not None else d,d,m if m is not None else d,y,
        d.numel(),d.shape[-1],p is not None,m is not None,1024,num_warps=4)
    return y

@residual.register_fake
def _fake(p,d,m):return torch.empty_like(d)
def _setup(ctx,inputs,output):
    p,d,m=inputs;ctx.has_p=p is not None;ctx.save_for_backward(m)
def _back(ctx,gy):
    m,=ctx.saved_tensors
    return gy if ctx.has_p else None,gy if m is None else gy.masked_fill(~m[...,None],0),None
residual.register_autograd(_back,setup_context=_setup)


@triton.jit(do_not_specialize=['NT'])
def _registers(T,R,G,Y,NT,C:tl.constexpr,NR:tl.constexpr,B:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//C;d=i%C;valid=i<NT*C
    source=tl.load(G+row,valid,0)
    reg=source<B*NR
    a=tl.load(R+(source%NR)*C+d,valid&reg,0)
    b=tl.load(T+(source-B*NR)*C+d,valid&~reg,0)
    tl.store(Y+i,tl.where(reg,a,b),valid)

@triton.jit(do_not_specialize=['NT'])
def _registers_back(GY,G,DT,DR,NT,C:tl.constexpr,NR:tl.constexpr,B:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//C;d=i%C;valid=i<NT*C
    source=tl.load(G+row,valid,0);reg=source<B*NR
    value=tl.load(GY+i,valid,0)
    tl.store(DR+source*C+d,value,valid&reg)
    tl.store(DT+(source-B*NR)*C+d,value,valid&~reg)

@torch.library.custom_op('v31rf::insert_registers',mutates_args=())
def registers(text:torch.Tensor,reg:torch.Tensor,gather:torch.Tensor,b:int)->torch.Tensor:
    y=torch.empty((gather.numel(),text.shape[-1]),device=text.device,dtype=text.dtype)
    _registers[(triton.cdiv(y.numel(),1024),)](text,reg,gather,y,gather.numel(),text.shape[-1],reg.shape[0],b,1024)
    return y
@registers.register_fake
def _fake_registers(text,reg,gather,b):return torch.empty((gather.numel(),text.shape[-1]),device=text.device,dtype=text.dtype)
def _reg_setup(ctx,inputs,output):
    text,reg,gather,b=inputs;ctx.shape=text.shape;ctx.reg_shape=reg.shape;ctx.batch=b;ctx.save_for_backward(gather)

@torch.library.custom_op('v31rf::register_backward',mutates_args=())
def _register_grad(gy:torch.Tensor,gather:torch.Tensor,b:int,nr:int,text_rows:int)->tuple[torch.Tensor,torch.Tensor]:
    c=gy.shape[-1]
    dt=torch.empty((text_rows,c),device=gy.device,dtype=gy.dtype)
    dr=torch.empty((b,nr,c),device=gy.device,dtype=gy.dtype)
    _registers_back[(triton.cdiv(gy.numel(),1024),)](gy.contiguous(),gather,dt,dr,gather.numel(),c,nr,b,1024)
    return dt,dr.sum(0)
@_register_grad.register_fake
def _fake_register_grad(gy,gather,b,nr,text_rows):
    return torch.empty((text_rows,gy.shape[-1]),device=gy.device,dtype=gy.dtype),torch.empty((nr,gy.shape[-1]),device=gy.device,dtype=gy.dtype)
def _reg_back(ctx,gy):
    gather,=ctx.saved_tensors;nr,c=ctx.reg_shape
    return (*_register_grad(gy,gather,ctx.batch,nr,ctx.shape[0]),None,None)
registers.register_autograd(_reg_back,setup_context=_reg_setup)
