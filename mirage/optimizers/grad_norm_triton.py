"""Ragged multi-tensor cast+local squared norms; metadata never comes back to CPU."""
import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(Sources,Destinations,Sizes:tl.constexpr,Ends:tl.constexpr,Src,Dst,N,End,K:tl.constexpr):
    for i in tl.static_range(K):
        tl.store(Src+i,Sources[i].to(tl.uint64));tl.store(Dst+i,Destinations[i].to(tl.uint64))
        tl.store(N+i,Sizes[i]);tl.store(End+i,Ends[i])


@triton.jit(do_not_specialize=['K'])
def _cast(Src,Dst,N,End,Parts,K,BK:tl.constexpr,BLOCK:tl.constexpr,DT:tl.constexpr):
    tile=tl.program_id(0);ks=tl.arange(0,BK)
    ends=tl.load(End+ks,ks<K,0)
    which=tl.sum(((tile>=ends)&(ks<K)).to(tl.int32),0)
    previous=tl.load(End+which-1,which>0,0);n=tl.load(N+which)
    i=(tile-previous)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(tl.load(Src+which).to(tl.pointer_type(DT))+i,i<n,0).to(tl.float32)
    tl.store(tl.load(Dst+which).to(tl.pointer_type(tl.float32))+i,x,i<n)
    tl.store(Parts+tile,tl.sum(x*x,0))


@triton.jit
def _parameter_norm(Parts,End,Norms,BLOCK:tl.constexpr):
    k=tl.program_id(0);start=tl.load(End+k-1,k>0,0);stop=tl.load(End+k)
    r=tl.arange(0,BLOCK);total=tl.full((BLOCK,),0.,tl.float32)
    for block in range(tl.cdiv(stop-start,BLOCK)):
        i=start+block*BLOCK+r;total+=tl.load(Parts+i,i<stop,0)
    tl.store(Norms+k,tl.sqrt(tl.sum(total,0)))


@torch.library.custom_op('v31rf::cast_grad_norm',mutates_args={'destinations'})
def cast_norm(sources:list[torch.Tensor],destinations:list[torch.Tensor])->torch.Tensor:
    if any(x.dtype!=sources[0].dtype or x.device!=sources[0].device for x in sources):raise ValueError('Gradient chunks must share dtype and device')
    block=8192;sizes=tuple(x.numel() for x in sources);ends=[];count=0
    for n in sizes:count+=triton.cdiv(n,block);ends.append(count)
    k=len(sources);device=sources[0].device
    src=torch.empty((k,),device=device,dtype=torch.uint64);dst=torch.empty_like(src)
    n=torch.empty((k,),device=device,dtype=torch.int64);end=torch.empty_like(n)
    parts=torch.empty((count,),device=device,dtype=torch.float32);norms=torch.empty((k,),device=device,dtype=torch.float32)
    _metadata[(1,)](tuple(sources),tuple(destinations),sizes,tuple(ends),src,dst,n,end,k,num_warps=1)
    dtype={torch.float32:tl.float32,torch.bfloat16:tl.bfloat16,torch.float16:tl.float16}[sources[0].dtype]
    if count:_cast[(count,)](src,dst,n,end,parts,k,triton.next_power_of_2(k),block,dtype,num_warps=4,enable_fp_fusion=False)
    _parameter_norm[(k,)](parts,end,norms,256,num_warps=4,enable_fp_fusion=False)
    return norms

@cast_norm.register_fake
def _fake(sources,destinations):return torch.empty((len(sources),),device=sources[0].device,dtype=torch.float32)
