"""Ordinary masked GQA attention and selectable inference cores."""
import math, os
import torch
from torch.nn import functional as F
from mirage.ops.fused_ops import use_triton

def sdpa_attention(q,k,v,mask=None,backend='auto'):
    if backend=='fa4':
        if mask is not None:raise ValueError('FA4 image path requires no image padding; packed text has valid-length slices')
        from pathlib import Path
        import sys
        optional=Path(__file__).resolve().parents[2]/'_fa4deps'
        env_site=Path(__file__).resolve().parents[2]/'.fa4env'/f'lib/python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
        if env_site.is_dir():optional=env_site
        if optional.is_dir() and str(optional) not in sys.path:
            import site
            site.addsitedir(str(optional))
            sys.path.insert(0,str(optional))
        from flash_attn.cute import flash_attn_func
        result=flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=False)
        if isinstance(result,tuple):result=result[0]
        return result.transpose(1,2)
    args=dict(attn_mask=mask,enable_gqa=q.shape[1]!=k.shape[1])
    if backend in ('auto','sdpa'):return F.scaled_dot_product_attention(q,k,v,**args)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    choices={'flash':SDPBackend.FLASH_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION}
    if backend not in choices:raise ValueError(f'Unknown SDPA backend: {backend}')
    with sdpa_kernel(choices[backend]):return F.scaled_dot_product_attention(q,k,v,**args)

def gated_text_attention(q, k, v, mask):
    """Select the attention core independently of fused normalization/FFN kernels."""
    backend=os.environ.get('MIRAGE_IMAGE_ATTN','auto')
    if backend=='auto':backend='triton' if use_triton(q.device) else 'sdpa'
    if backend!='triton':
        return sdpa_attention(q,k,v,None if mask is None else mask[:,None,None,:],backend)
    if not use_triton(q.device):raise RuntimeError('Triton attention requires --backend triton')
    from mirage.attention.image_attention_triton import image_attention
    return image_attention(q,k,v,mask)
