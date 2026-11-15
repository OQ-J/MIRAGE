"""Per-token/head output gate; CUDA fuses sigmoid, dtype conversion and multiplication."""
from mirage.ops.fused_ops import use_triton
def head_gate(y,logits):
    if use_triton(y.device):
        from mirage.ops.gate_triton import head_gate as fused
        return fused(y,logits)
    return y*logits.float().sigmoid()[...,None].to(y.dtype)
