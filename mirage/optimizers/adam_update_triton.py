"""FP32 AdamW state update + clipping + BF16/FP32 model writeback in one data pass."""
import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(P,G,M,V,Y,Steps,Sizes:tl.constexpr,Ends:tl.constexpr,PT,GT,MT,VT,YT,NT,ET,B1C,B2C,
              BETA1:tl.constexpr,BETA2:tl.constexpr,K:tl.constexpr):
    for i in tl.static_range(K):
        tl.store(PT+i,P[i].to(tl.uint64));tl.store(GT+i,G[i].to(tl.uint64))
        tl.store(MT+i,M[i].to(tl.uint64));tl.store(VT+i,V[i].to(tl.uint64));tl.store(YT+i,Y[i].to(tl.uint64))
        tl.store(NT+i,Sizes[i]);tl.store(ET+i,Ends[i])
        step=tl.load(Steps[i])+1.;tl.store(Steps[i],step)
        # Same FP32 bias correction definition as the existing fused AdamW.
        b1=tl.full((),BETA1,tl.float32);b2=tl.full((),BETA2,tl.float32)
        tl.store(B1C+i,1-tl.exp(tl.log(b1)*step))
        tl.store(B2C+i,tl.sqrt(1-tl.exp(tl.log(b2)*step)))


@triton.jit(do_not_specialize=['K','LR','CLIP','WD','EPS'])
def _update(PT,GT,MT,VT,YT,NT,ET,B1C,B2C,Norm,K,LR,CLIP,WD,EPS,
            BETA1:tl.constexpr,BETA2:tl.constexpr,BK:tl.constexpr,BLOCK:tl.constexpr,YDT:tl.constexpr):
    tile=tl.program_id(0);ks=tl.arange(0,BK)
    ends=tl.load(ET+ks,ks<K,0);which=tl.sum(((tile>=ends)&(ks<K)).to(tl.int32),0)
    start=tl.load(ET+which-1,which>0,0);n=tl.load(NT+which)
    i=(tile-start)*BLOCK+tl.arange(0,BLOCK);valid=i<n
    pp=tl.load(PT+which).to(tl.pointer_type(tl.float32))+i
    gp=tl.load(GT+which).to(tl.pointer_type(tl.float32))+i
    mp=tl.load(MT+which).to(tl.pointer_type(tl.float32))+i
    vp=tl.load(VT+which).to(tl.pointer_type(tl.float32))+i
    yp=tl.load(YT+which).to(tl.pointer_type(YDT))+i
    p=tl.load(pp,valid,0);g=tl.load(gp,valid,0)
    m=tl.load(mp,valid,0);v=tl.load(vp,valid,0)
    if CLIP>0:g*=tl.minimum(1.,CLIP/(tl.load(Norm)+1.e-6))
    p=p-(LR*WD)*p
    # Explicit FMA follows PyTorch's FP32 fused Adam moment update.
    b1=tl.full((),BETA1,tl.float32);b2=tl.full((),BETA2,tl.float32)
    m=tl.fma(b1,m,tl.fma(-b1,g,g));gg=g*g
    v=tl.fma(b2,v,tl.fma(-b2,gg,gg))
    step_size=LR/tl.load(B1C+which)
    denominator=tl.sqrt(v)/tl.load(B2C+which)+EPS
    p=p-(step_size*m)/denominator
    tl.store(mp,m,valid);tl.store(vp,v,valid);tl.store(pp,p,valid);tl.store(yp,p,valid)


@torch.library.custom_op('v31rf::adam_clip_writeback',mutates_args={'masters','moments','variances','models','steps'})
def update(masters:list[torch.Tensor],grads:list[torch.Tensor],moments:list[torch.Tensor],variances:list[torch.Tensor],
           models:list[torch.Tensor],steps:list[torch.Tensor],norm:torch.Tensor,lr:float,clip:float,
           beta1:float,beta2:float,weight_decay:float,eps:float)->None:
    block=4096;sizes=tuple(x.numel() for x in masters);ends=[];total=0
    for n in sizes:total+=triton.cdiv(n,block);ends.append(total)
    k=len(sizes);device=norm.device
    ptrs=[torch.empty((k,),device=device,dtype=torch.uint64) for _ in range(5)]
    ns=torch.empty((k,),device=device,dtype=torch.int64);es=torch.empty_like(ns)
    c1=torch.empty((k,),device=device,dtype=torch.float32);c2=torch.empty_like(c1)
    _metadata[(1,)](tuple(masters),tuple(grads),tuple(moments),tuple(variances),tuple(models),tuple(steps),
        sizes,tuple(ends),*ptrs,ns,es,c1,c2,beta1,beta2,k,num_warps=1,enable_fp_fusion=False)
    dtype={torch.bfloat16:tl.bfloat16,torch.float32:tl.float32}[models[0].dtype]
    if total:_update[(total,)](*ptrs,ns,es,c1,c2,norm,k,lr,clip,weight_decay,eps,
        beta1,beta2,triton.next_power_of_2(k),block,dtype,num_warps=4,enable_fp_fusion=False)


@update.register_fake
def _fake(masters,grads,moments,variances,models,steps,norm,lr,clip,beta1,beta2,weight_decay,eps):return None
