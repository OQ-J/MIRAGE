"""Convert gradients once; norm statistics are collected during the CUDA write."""
import torch


def cast_and_norm(sources,destinations):
    if not sources:raise ValueError('Empty gradient conversion chunk')
    sources=[x.contiguous() for x in sources]
    if sources[0].device.type=='cuda' and all(x.is_contiguous() for x in destinations):
        from mirage.optimizers.grad_norm_triton import cast_norm
        return cast_norm(sources,destinations)
    torch._foreach_copy_(destinations,sources)
    return torch.stack(torch._foreach_norm(destinations,2))
