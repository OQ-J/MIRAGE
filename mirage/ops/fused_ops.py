"""CPU formulas and CUDA fused SwiGLU/RMS/AdaRMS, preserving the BF16 rounding boundaries."""
from importlib.util import find_spec
import os
import torch
from torch.nn import functional as F


TRITON_AVAILABLE = find_spec('triton') is not None


def use_triton(device):
    """Use portable PyTorch formulas when the Triton package is absent."""
    return device.type == 'cuda' and TRITON_AVAILABLE and os.environ.get('MIRAGE_TORCH_ONLY', '0') != '1'


def swiglu(combined):
    if use_triton(combined.device):
        from mirage.ops.elementwise_triton import swiglu as fused
        return fused(combined)
    a,b=combined.chunk(2,-1)
    return F.silu(a)*b


def rms(x,weight,eps,*,fp32=False):
    if use_triton(x.device):
        from mirage.ops.norm_triton import rms as fused
        return fused(x,weight,eps,fp32=fp32)
    y=x.float();y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)
    y=y*weight.float()
    return y if fp32 else y.to(x.dtype)


def ada_rms(x,weight,scale,shift,eps):
    if use_triton(x.device):
        from mirage.ops.norm_triton import ada_rms as fused
        return fused(x,weight,scale,shift,eps)
    normalized=rms(x,weight,eps).float()
    return (normalized*(1+scale[:,None])+shift[:,None]).to(x.dtype)


def head_norm_rope(x,weight,cos,sin,eps):
    if use_triton(x.device):
        from mirage.ops.rope_triton import head_norm_rope as fused
        return fused(x,weight,cos,sin,eps)
    y=rms(x,weight,eps,fp32=True)
    even,odd=y[...,0::2],y[...,1::2]
    c,s=cos[None,:,None],sin[None,:,None]
    return torch.stack((even*c-odd*s,even*s+odd*c),-1).flatten(-2).to(x.dtype)


def qkv_norm_rope(x,wq,wk,cos,sin,eps):
    if use_triton(x.device):
        from mirage.ops.qkv_triton import qkv_norm_rope as fused
        return fused(x,wq,wk,cos,sin,eps)
    b,n,_=x.shape;h,d=wq.shape;hk=wk.shape[0]
    q,k,v=x.split((h*d,hk*d,hk*d),-1)
    q=head_norm_rope(q.reshape(b,n,h,d),wq,cos,sin,eps).transpose(1,2).contiguous()
    k=head_norm_rope(k.reshape(b,n,hk,d),wk,cos,sin,eps).transpose(1,2).contiguous()
    v=v.reshape(b,n,hk,d).transpose(1,2).contiguous()
    return q,k,v


def text_qkv(q,kv,wq,wk,eps):
    """Flat token-major projections; the row stride may include other QKV slices."""
    if use_triton(q.device):
        from mirage.ops.text_qkv_triton import text_qkv as fused
        return fused(q,kv,wq,wk,eps)
    k,v=kv.chunk(2,-1)
    return rms(q,wq,eps),rms(k.reshape(-1,*wk.shape),wk,eps),v.reshape(-1,*wk.shape).contiguous()
