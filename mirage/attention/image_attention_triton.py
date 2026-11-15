"""Ordinary masked image Softmax, tuned forward/dQ/dKV; no Null computation."""
import math,os
import torch
import triton
import triton.language as tl
from mirage.attention.attention_delta_triton import row_delta


@triton.jit
def _dot(a,b,FAST:tl.constexpr):
    if FAST:
        return tl.dot(a.to(tl.bfloat16),b.to(tl.bfloat16),out_dtype=tl.float32)
    return tl.dot(a,b,input_precision='tf32x3')


@triton.jit(do_not_specialize=['N','L','N_BUCKET','L_BUCKET','B_BUCKET'])
def _forward(Q,K,V,Mask,O,LSE,N,L,N_BUCKET,L_BUCKET,B_BUCKET,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
             BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr,ALL_VALID:tl.constexpr,SAVE_STATS:tl.constexpr=True):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    b,h=tl.program_id(1)//H,tl.program_id(1)%H
    kh=h//(H//HK)
    d=tl.arange(0,BD)
    q=tl.load(Q+((b*H+h)*N+r[:,None])*D+d[None,:],(r[:,None]<N)&(d[None,:]<D),0).to(tl.float32)
    m=tl.full((BM,),float('-inf'),tl.float32)
    denom=tl.full((BM,),0.,tl.float32)
    acc=tl.full((BM,BD),0,tl.float32)
    for start in range(tl.cdiv(L,BK)):
        c=start*BK+tl.arange(0,BK)
        k=tl.load(K+((b*HK+kh)*L+c[None,:])*D+d[:,None],(c[None,:]<L)&(d[:,None]<D),0).to(tl.float32)
        if ALL_VALID:valid=tl.full((BK,),True,tl.int1)
        else:valid=tl.load(Mask+b*L+c,c<L,0)
        s=_dot(q,k,FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
        s=tl.where((c[None,:]<L)&valid[None,:],s,float('-inf'))
        mnew=tl.maximum(m,tl.max(s,1))
        safe_m=tl.where(mnew==float('-inf'),0.,mnew)
        alpha=tl.exp(m-safe_m)
        p=tl.where(s==float('-inf'),0.,tl.exp(s-safe_m[:,None]))
        v=tl.load(V+((b*HK+kh)*L+c[:,None])*D+d[None,:],(c[:,None]<L)&(d[None,:]<D),0).to(tl.float32)
        acc=acc*alpha[:,None]+_dot(p,v,FAST=Q.dtype.element_ty==tl.bfloat16)
        denom=denom*alpha+tl.sum(p,1)
        m=mnew
    safe_denom=tl.where(denom>0,denom,1.)
    result=acc/safe_denom[:,None]
    tl.store(O+((b*N+r[:,None])*H+h)*D+d[None,:],result,(r[:,None]<N)&(d[None,:]<D))
    if SAVE_STATS:tl.store(LSE+(b*H+h)*N+r,m+tl.log(safe_denom),r<N)


@triton.jit(do_not_specialize=['N','L','N_BUCKET','L_BUCKET','B_BUCKET'])
def _dq(Q,K,V,Mask,Delta,LSE,GO,DQ,N,L,N_BUCKET,L_BUCKET,B_BUCKET,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
        BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr,ALL_VALID:tl.constexpr):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    b,h=tl.program_id(1)//H,tl.program_id(1)%H
    kh=h//(H//HK);d=tl.arange(0,BD)
    offset=((b*H+h)*N+r[:,None])*D+d[None,:]
    active=(r[:,None]<N)&(d[None,:]<D)
    q=tl.load(Q+offset,active,0).to(tl.float32)
    go=tl.load(GO+((b*N+r[:,None])*H+h)*D+d[None,:],active,0).to(tl.float32)
    delta=tl.load(Delta+(b*N+r)*H+h,r<N,0)
    lse=tl.load(LSE+(b*H+h)*N+r,r<N,0)
    lse=tl.where(lse==float('-inf'),0.,lse)
    dq=tl.full((BM,BD),0,tl.float32)
    for start in range(tl.cdiv(L,BK)):
        c=start*BK+tl.arange(0,BK)
        k=tl.load(K+((b*HK+kh)*L+c[None,:])*D+d[:,None],(c[None,:]<L)&(d[:,None]<D),0).to(tl.float32)
        v=tl.load(V+((b*HK+kh)*L+c[None,:])*D+d[:,None],(c[None,:]<L)&(d[:,None]<D),0).to(tl.float32)
        if ALL_VALID:valid=tl.full((BK,),True,tl.int1)
        else:valid=tl.load(Mask+b*L+c,c<L,0)
        s=_dot(q,k,FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
        p=tl.where((c[None,:]<L)&valid[None,:],tl.exp(s-lse[:,None]),0.)
        ds=p*(_dot(go,v,FAST=Q.dtype.element_ty==tl.bfloat16)-delta[:,None])
        dq+=_dot(ds,tl.trans(k),FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
    tl.store(DQ+offset,dq,active)


@triton.jit(do_not_specialize=['N','L','N_BUCKET','L_BUCKET','B_BUCKET'])
def _dkv(Q,K,V,Mask,Delta,LSE,GO,DK,DV,N,L,N_BUCKET,L_BUCKET,B_BUCKET,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
         BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr,ALL_VALID:tl.constexpr):
    c=tl.program_id(0)*BK+tl.arange(0,BK)
    b,kh=tl.program_id(1)//HK,tl.program_id(1)%HK
    d=tl.arange(0,BD)
    kv_offset=((b*HK+kh)*L+c[:,None])*D+d[None,:]
    kv_active=(c[:,None]<L)&(d[None,:]<D)
    k=tl.load(K+kv_offset,kv_active,0).to(tl.float32)
    v=tl.load(V+kv_offset,kv_active,0).to(tl.float32)
    if ALL_VALID:valid=tl.full((BK,),True,tl.int1)
    else:valid=tl.load(Mask+b*L+c,c<L,0)
    dk=tl.full((BK,BD),0,tl.float32);dv=tl.full((BK,BD),0,tl.float32)
    for group in range(H//HK):
        h=kh*(H//HK)+group
        for start in range(tl.cdiv(N,BM)):
            r=start*BM+tl.arange(0,BM)
            off=((b*H+h)*N+r[:,None])*D+d[None,:]
            active=(r[:,None]<N)&(d[None,:]<D)
            q=tl.load(Q+off,active,0).to(tl.float32)
            go=tl.load(GO+((b*N+r[:,None])*H+h)*D+d[None,:],active,0).to(tl.float32)
            delta=tl.load(Delta+(b*N+r)*H+h,r<N,0)
            lse=tl.load(LSE+(b*H+h)*N+r,r<N,0)
            lse=tl.where(lse==float('-inf'),0.,lse)
            s=_dot(q,tl.trans(k),FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
            p=tl.where((r[:,None]<N)&(c[None,:]<L)&valid[None,:],tl.exp(s-lse[:,None]),0.)
            ds=p*(_dot(go,tl.trans(v),FAST=Q.dtype.element_ty==tl.bfloat16)-delta[:,None])
            dk+=_dot(tl.trans(ds),q,FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
            dv+=_dot(tl.trans(p),go,FAST=Q.dtype.element_ty==tl.bfloat16)
    tl.store(DK+kv_offset,dk,kv_active)
    tl.store(DV+kv_offset,dv,kv_active)


TILES=((16,32,4,1),(32,64,4,1),(64,64,4,1),(32,32,4,2),(64,32,4,2),(64,64,4,2),(64,128,4,2),(128,64,8,2))
_tuned={}

def _launch(kernel,grid,inputs,args):
    if kernel.__name__=='_forward':args.setdefault('SAVE_STATS',True)
    if os.environ.get('TRITON_INTERPRET')=='1':
        m,n,w=map(int,os.environ.get('MIRAGE_ATTN_TEST_TILE','16,32,4').split(','))
        kernel[grid(dict(BM=m,BK=n))](*inputs,**args,BM=m,BK=n,num_warps=w,num_stages=1)
    else:
        if kernel.__name__ not in _tuned:
            configs=[triton.Config(dict(BM=m,BK=n),num_warps=w,num_stages=st) for m,n,w,st in TILES]
            keys=['D','H','HK','N_BUCKET','L_BUCKET','B_BUCKET','ALL_VALID']
            if kernel.__name__=='_forward':keys.append('SAVE_STATS')
            _tuned[kernel.__name__]=triton.autotune(configs,key=keys,cache_results=True)(kernel)
        _tuned[kernel.__name__][grid](*inputs,**args)

def _args(q,k):
    return dict(N=q.shape[2],L=k.shape[2],H=q.shape[1],HK=k.shape[1],D=q.shape[-1],
                BD=max(32,triton.next_power_of_2(q.shape[-1])),SCALE=1/math.sqrt(q.shape[-1]),
                N_BUCKET=triton.next_power_of_2(q.shape[2]),L_BUCKET=triton.next_power_of_2(k.shape[2]),
                B_BUCKET=triton.next_power_of_2(q.shape[0]))

@torch.library.custom_op('v31rf::image_attention_forward',mutates_args=())
def _fwd(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor|None)->tuple[torch.Tensor,torch.Tensor]:
    out=torch.empty((q.shape[0],q.shape[2],q.shape[1],q.shape[3]),device=q.device,dtype=torch.float32).transpose(1,2)
    lse=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
    _launch(_forward,lambda meta:(triton.cdiv(q.shape[2],meta['BM']),q.shape[0]*q.shape[1]),
            (q,k,v,mask if mask is not None else q,out,lse),dict(_args(q,k),ALL_VALID=mask is None))
    return out,lse

@_fwd.register_fake
def _fake(q,k,v,mask):
    return (torch.empty((q.shape[0],q.shape[2],q.shape[1],q.shape[3]),device=q.device,dtype=torch.float32).transpose(1,2),
            torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32))

@torch.library.custom_op('v31rf::image_attention_backward',mutates_args=())
def _bwd(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor|None,
         out:torch.Tensor,lse:torch.Tensor,go:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    args=dict(_args(q,k),ALL_VALID=mask is None);go=go.transpose(1,2).contiguous().transpose(1,2);delta=row_delta(out,go)
    _launch(_dq,lambda meta:(triton.cdiv(q.shape[2],meta['BM']),q.shape[0]*q.shape[1]),
            (q,k,v,mask if mask is not None else q,delta,lse,go,dq),args)
    _launch(_dkv,lambda meta:(triton.cdiv(k.shape[2],meta['BK']),q.shape[0]*k.shape[1]),
            (q,k,v,mask if mask is not None else q,delta,lse,go,dk,dv),args)
    return dq,dk,dv

@_bwd.register_fake
def _fake_backward(q,k,v,mask,out,lse,go):
    return torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)

def _setup(ctx,inputs,output):
    ctx.save_for_backward(*inputs,*output);ctx.mark_non_differentiable(output[1])

def _autograd(ctx,go,unused):
    q,k,v,mask,out,lse=ctx.saved_tensors
    return (*_bwd(q,k,v,mask,out,lse,go),None)

_fwd.register_autograd(_autograd,setup_context=_setup)

def image_attention(q,k,v,mask):
    if torch.is_inference_mode_enabled() and os.environ.get('MIRAGE_INFERENCE_KERNELS','1')=='1':
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        out=torch.empty((q.shape[0],q.shape[2],q.shape[1],q.shape[3]),device=q.device,dtype=v.dtype).transpose(1,2)
        _launch(_forward,lambda meta:(triton.cdiv(q.shape[2],meta['BM']),q.shape[0]*q.shape[1]),
                (q,k,v,mask if mask is not None else q,out,out),dict(_args(q,k),ALL_VALID=mask is None,SAVE_STATS=False))
        return out
    return _fwd(q.contiguous(),k.contiguous(),v.contiguous(),None if mask is None else mask.contiguous())[0].to(v.dtype)
