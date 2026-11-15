"""Fused writes with CPU references and the original BF16 rounding boundaries."""
import torch
from mirage.ops.fused_ops import use_triton


def residual_write(partial,delta,mask):
    if partial is None and mask is None:return delta
    if use_triton(delta.device):
        from mirage.ops.write_triton import residual
        return residual(partial,delta,mask)
    if mask is not None:delta=delta.masked_fill(~mask[...,None],0)
    return delta if partial is None else partial+delta


def insert_registers(text,registers,gather,batch):
    if registers.shape[0]==0:return text.index_select(0,gather)
    if use_triton(text.device):
        from mirage.ops.write_triton import registers as fused
        return fused(text,registers,gather,batch)
    prefix=registers[None].expand(batch,-1,-1).reshape(-1,registers.shape[-1]).to(text.dtype)
    return torch.cat((prefix,text),0).index_select(0,gather)
