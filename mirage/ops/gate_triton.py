"""Fused output head gate with the original BF16 forward/backward rounding boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Y,L,O,D:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD)
    y=tl.load(Y+r*D+d,d<D,0).to(tl.float32)
    logit=tl.load(L+r).to(tl.float32)
    gate=tl.sigmoid(logit).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(O+r*D+d,y*gate,d<D)


@triton.jit
def _backward(Y,L,GO,DY,DL,D:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0);d=tl.arange(0,BD)
    y=tl.load(Y+r*D+d,d<D,0).to(tl.float32)
    go=tl.load(GO+r*D+d,d<D,0).to(tl.float32)
    gate=tl.sigmoid(tl.load(L+r).to(tl.float32))
    rounded=gate.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(DY+r*D+d,go*rounded,d<D)
    # Mul backward rounds per-element products, then broadcast-reduction rounds once.
    contributions=(go*y).to(Y.dtype.element_ty).to(tl.float32)
    dg=tl.sum(contributions,0).to(Y.dtype.element_ty).to(tl.float32)
    dl=(dg*(1-gate))*gate
    tl.store(DL+r,dl)


@torch.library.custom_op('v31rf::head_gate_forward',mutates_args=())
def _fwd(y:torch.Tensor,logits:torch.Tensor)->torch.Tensor:
    out=torch.empty_like(y)
    _forward[(logits.numel(),)](y,logits,out,D=y.shape[-1],BD=triton.next_power_of_2(y.shape[-1]),num_warps=4,enable_fp_fusion=False)
    return out


@_fwd.register_fake
def _fake(y,logits):return torch.empty_like(y)


@torch.library.custom_op('v31rf::head_gate_backward',mutates_args=())
def _bwd(y:torch.Tensor,logits:torch.Tensor,go:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    dy=torch.empty_like(y);dl=torch.empty_like(logits)
    _backward[(logits.numel(),)](y,logits,go.contiguous(),dy,dl,D=y.shape[-1],BD=triton.next_power_of_2(y.shape[-1]),num_warps=4,enable_fp_fusion=False)
    return dy,dl


@_bwd.register_fake
def _fake_bwd(y,logits,go):return torch.empty_like(y),torch.empty_like(logits)


def _setup(ctx,inputs,output):ctx.save_for_backward(*inputs)


def _autograd(ctx,go):
    y,logits=ctx.saved_tensors
    return _bwd(y,logits,go)


_fwd.register_autograd(_autograd,setup_context=_setup)


def head_gate(y,logits):return _fwd(y.contiguous(),logits.contiguous())
