"""Packed text metadata prepared outside the compiled body; no padded CUDA text activations."""
from typing import NamedTuple
import os
import torch
from torch.nn import functional as F
from mirage.ops.fused_ops import rms, use_triton
from mirage.attention.attention import sdpa_attention


class TextLayout(NamedTuple):
    cu: torch.Tensor
    gather: torch.Tensor
    lengths: tuple
    offsets: tuple
    maximum: int


def pack_sequences(sequences,registers,*,dtype=torch.bfloat16,pin_memory=False,out=None):
    """CPU rows -> one compact buffer; never allocate [B,Lmax,D]."""
    lengths=tuple(len(row) for row in sequences)
    flat=out['text'] if out is not None else torch.empty((sum(lengths),sequences[0].shape[-1]),dtype=dtype,pin_memory=pin_memory)
    offsets=[0];gather=[];cursor=0;b=len(lengths)
    for i,(row,n) in enumerate(zip(sequences,lengths)):
        flat[cursor:cursor+n].copy_(row.float())
        gather.extend(range(i*registers,(i+1)*registers))
        gather.extend(range(b*registers+cursor,b*registers+cursor+n))
        cursor+=n;offsets.append(offsets[-1]+n+registers)
    full_lengths=tuple(n+registers for n in lengths)
    cu=torch.tensor(offsets,dtype=torch.int32);indices=torch.tensor(gather,dtype=torch.int64)
    if out is not None:out['cu'].copy_(cu);out['gather'].copy_(indices);cu,indices=out['cu'],out['gather']
    elif pin_memory:cu,indices=cu.pin_memory(),indices.pin_memory()
    layout=TextLayout(cu,indices,full_lengths,tuple(offsets),max(full_lengths))
    return flat,layout


def pack_text(text, mask, registers, *, device=None, dtype=None):
    lengths=tuple(int(x) for x in mask.detach().cpu().sum(-1).tolist())
    lengths_with_registers=tuple(n+registers for n in lengths)
    offsets=[0];gather=[];cursor=0;b=len(lengths)
    for i,n in enumerate(lengths):
        gather.extend(range(i*registers,(i+1)*registers))
        gather.extend(range(b*registers+cursor,b*registers+cursor+n))
        cursor+=n;offsets.append(offsets[-1]+n+registers)
    flat=text[mask].to(device=device,dtype=dtype)
    layout=TextLayout(torch.tensor(offsets,device=flat.device,dtype=torch.int32),
        torch.tensor(gather,device=flat.device,dtype=torch.int64),lengths_with_registers,
        tuple(offsets),max(lengths_with_registers))
    return flat,layout


def head_norm_packed(norm,x,*,fp32=True):
    return rms(x,norm.weight,norm.eps,fp32=fp32)


def attention(q,k,v,cu_q,cu_k,max_q,max_k,q_lengths,k_lengths):
    """Token-major [total,heads,dim]; CPU oracle processes each sequence independently."""
    backend=os.environ.get('MIRAGE_TEXT_ATTN','auto')
    if use_triton(q.device) and backend not in ('flash','sdpa','fa4'):
        from mirage.attention.varlen_triton import varlen_attention
        return varlen_attention(q,k,v,cu_q,cu_k,max_q,max_k,q_lengths,k_lengths)
    outputs=[];qi=ki=0
    for nq,nk in zip(q_lengths,k_lengths):
        qs=q[qi:qi+nq].transpose(0,1)[None]
        ks=k[ki:ki+nk].transpose(0,1)[None]
        vs=v[ki:ki+nk].transpose(0,1)[None]
        outputs.append(sdpa_attention(qs,ks,vs,backend=backend if backend in ('flash','sdpa','fa4') else 'auto')[0].transpose(0,1))
        qi+=nq;ki+=nk
    return torch.cat(outputs)
