"""Explicit per-inference cache; never registered in state_dict or reused in training."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TextCache:
    owner:int
    signature:tuple
    tokens:torch.Tensor
    layout:object
    layers:tuple


def signature(model):
    parameters=list(model.text_adapter.parameters())
    for branch in model.text_attentions():parameters.extend(branch.kv.parameters());parameters.extend(branch.k_norm.parameters())
    return tuple((id(p),p._version,str(p.device),p.dtype) for p in parameters)


def prepare(model,text,mask):
    if isinstance(text,TextCache):return text,mask
    if not model.training and getattr(model,'text_cache_enabled',True) and hasattr(model,'build_text_cache'):return model.build_text_cache(text,mask),mask
    return text,mask
