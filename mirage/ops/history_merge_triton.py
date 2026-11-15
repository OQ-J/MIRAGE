"""Analytical AdaRMS+depth backward; mutate only private per-backward gradient buffers."""
import torch
import triton
from mirage.ops.depth_triton import _shape,make_table,_grad_params
from mirage.ops.depth_ada_triton import _backward
from mirage.ops.norm_triton import _reduce_w,_reduce_pair


@torch.library.custom_op('v31rf::merged_depth_backward',mutates_args={'accum'})
def merged_backward(values:list[torch.Tensor],q:torch.Tensor,dw:torch.Tensor,w:torch.Tensor,s:torch.Tensor,t:torch.Tensor,
                    weights:torch.Tensor,inv:torch.Tensor,go:torch.Tensor,eps:float,
                    accum:list[torch.Tensor],initialized:bool,ptrs:torch.Tensor|None=None,self_table:torch.Tensor|None=None,read:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    r,args=_shape(values);has_partial=len(values)>len(accum)
    dp=torch.empty_like(values[-1]) if has_partial else torch.empty((0,),device=q.device,dtype=values[0].dtype)
    br=4;group=4;n=values[0].shape[-2];b=values[0].shape[0];tiles=triton.cdiv(n,br*group)
    common=torch.empty((b,tiles,q.numel()),device=q.device,dtype=torch.float32)
    wr=torch.empty_like(common);sr=torch.empty_like(common);tr=torch.empty_like(common)
    dq,ddw,denw,ds,dt=map(torch.empty_like,(q,dw,w,s,t))
    _backward[(tiles,b)](make_table(values) if ptrs is None else ptrs,q,dw,w,s,weights,inv,go.contiguous(),make_table(accum) if self_table is None else self_table,common,wr,sr,tr,**args,
        N=n,SS=s.stride(0),EPS=eps,BR=br,GROUP=group,DP=dp,HAS_PARTIAL=has_partial,MERGE=True,EXISTING=initialized,MERGE_K=len(accum),Read=read if read is not None else go,USE_READ=read is not None and read.numel()>0,num_warps=4,enable_fp_fusion=False)
    _grad_params[(triton.cdiv(q.numel(),128),)](common,q,dw,dq,ddw,b*tiles,q.numel(),BC=128,BR=32,num_warps=4)
    _reduce_w[(triton.cdiv(w.numel(),128),1)](wr,denw,b*tiles,w.numel(),1,BC=128,BR=32,num_warps=4)
    _reduce_pair[(triton.cdiv(w.numel(),128),b)](sr,tr,ds,dt,tiles,w.numel(),BC=128,BR=32,num_warps=4)
    return dp,dq,ddw,denw,ds,dt

@merged_backward.register_fake
def _fake(values,q,dw,w,s,t,weights,inv,go,eps,accum,initialized,ptrs=None,self_table=None,read=None):
    dp=torch.empty_like(values[-1]) if len(values)>len(accum) else torch.empty((0,),device=q.device,dtype=values[0].dtype)
    return dp,*(torch.empty_like(v) for v in (q,dw,w,s,t))
