"""Token-major Q/K RMSNorm plus packed KV repack, including strided joint QKV views."""
import torch,os
import triton
import triton.language as tl
from mirage.ops.norm_triton import _reduce_w


@triton.jit(do_not_specialize=['RQ','RK'])
def _forward(Q,KV,WQ,WK,OQ,OK,OV,IQ,IK,RQ,RK,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
             SQ:tl.constexpr,SK:tl.constexpr,BD:tl.constexpr,EPS:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR);h=tl.program_id(1);d=tl.arange(0,BD)
    if h<H:
        valid=(r[:,None]<RQ)&(d[None,:]<D)
        x=tl.load(Q+r[:,None]*SQ+h*D+d[None,:],valid,0).to(tl.float32)
        inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS);w=tl.load(WQ+h*D+d,d<D,0).to(tl.float32)
        tl.store(OQ+(r[:,None]*H+h)*D+d[None,:],(x*inv[:,None])*w[None,:],valid)
        tl.store(IQ+r*H+h,inv,r<RQ)
    if h<HK:
        valid=(r[:,None]<RK)&(d[None,:]<D)
        x=tl.load(KV+r[:,None]*SK+h*D+d[None,:],valid,0).to(tl.float32)
        inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS);w=tl.load(WK+h*D+d,d<D,0).to(tl.float32)
        tl.store(OK+(r[:,None]*HK+h)*D+d[None,:],(x*inv[:,None])*w[None,:],valid)
        tl.store(IK+r*HK+h,inv,r<RK)
        v=tl.load(KV+r[:,None]*SK+HK*D+h*D+d[None,:],valid,0)
        tl.store(OV+(r[:,None]*HK+h)*D+d[None,:],v,valid)

@triton.jit(do_not_specialize=['RQ','RK'])
def _backward(Q,KV,WQ,WK,IQ,IK,GQ,GK,GV,DQ,DKV,PQ,PK,RQ,RK,H:tl.constexpr,HK:tl.constexpr,
              D:tl.constexpr,SQ:tl.constexpr,SK:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1);r=tile*BR+tl.arange(0,BR);d=tl.arange(0,BD)
    if h<H and tile*BR<RQ:
        valid=(r[:,None]<RQ)&(d[None,:]<D)
        x=tl.load(Q+r[:,None]*SQ+h*D+d[None,:],valid,0).to(tl.float32)
        inv=tl.load(IQ+r*H+h,r<RQ,0);w=tl.load(WQ+h*D+d,d<D,0).to(tl.float32)
        g=tl.load(GQ+(r[:,None]*H+h)*D+d[None,:],valid,0).to(tl.float32)
        gw=g*w[None,:];dot=tl.sum(gw*x,1)
        dx=inv[:,None]*(gw-x*(inv[:,None]*inv[:,None]/D)*dot[:,None])
        tl.store(DQ+(r[:,None]*H+h)*D+d[None,:],dx,valid)
        tl.store(PQ+(tile*H+h)*D+d,tl.sum((g*x)*inv[:,None],0),d<D)
    if h<HK and tile*BR<RK:
        valid=(r[:,None]<RK)&(d[None,:]<D)
        x=tl.load(KV+r[:,None]*SK+h*D+d[None,:],valid,0).to(tl.float32)
        inv=tl.load(IK+r*HK+h,r<RK,0);w=tl.load(WK+h*D+d,d<D,0).to(tl.float32)
        g=tl.load(GK+(r[:,None]*HK+h)*D+d[None,:],valid,0).to(tl.float32)
        gw=g*w[None,:];dot=tl.sum(gw*x,1)
        dx=inv[:,None]*(gw-x*(inv[:,None]*inv[:,None]/D)*dot[:,None])
        tl.store(DKV+r[:,None]*(2*HK*D)+h*D+d[None,:],dx,valid)
        gv=tl.load(GV+(r[:,None]*HK+h)*D+d[None,:],valid,0)
        tl.store(DKV+r[:,None]*(2*HK*D)+HK*D+h*D+d[None,:],gv,valid)
        tl.store(PK+(tile*HK+h)*D+d,tl.sum((g*x)*inv[:,None],0),d<D)


@torch.library.custom_op('v31rf::text_qkv_forward',mutates_args=())
def _fwd(q:torch.Tensor,kv:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,eps:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    rq,h,d=q.shape;rk=kv.shape[0];hk=wk.shape[0]
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    ok=torch.empty((rk,hk,d),device=q.device,dtype=q.dtype);ov=torch.empty_like(ok)
    iq=torch.empty((rq,h),device=q.device,dtype=torch.float32);ik=torch.empty((rk,hk),device=q.device,dtype=torch.float32)
    br=int(os.environ.get('MIRAGE_NORM_FWD_ROWS','4'));warps=int(os.environ.get('MIRAGE_NORM_WARPS','4'))
    if br not in (1,2,4,8) or warps not in (1,2,4,8):raise ValueError('Invalid V32 Norm launch configuration')
    _forward[(triton.cdiv(max(rq,rk),br),max(h,hk))](q,kv,wq,wk,oq,ok,ov,iq,ik,rq,rk,h,hk,d,q.stride(0),kv.stride(0),
        triton.next_power_of_2(d),eps,BR=br,num_warps=warps,enable_fp_fusion=False)
    return oq,ok,ov,iq,ik


@_fwd.register_fake
def _fake(q,kv,wq,wk,eps):
    rq,h,d=q.shape;rk=kv.shape[0];hk=wk.shape[0]
    return (torch.empty(q.shape,device=q.device,dtype=q.dtype),
        torch.empty((rk,hk,d),device=q.device,dtype=q.dtype),torch.empty((rk,hk,d),device=q.device,dtype=q.dtype),
        torch.empty((rq,h),device=q.device,dtype=torch.float32),torch.empty((rk,hk),device=q.device,dtype=torch.float32))


@torch.library.custom_op('v31rf::text_qkv_backward',mutates_args=())
def _bwd(q:torch.Tensor,kv:torch.Tensor,wq:torch.Tensor,wk:torch.Tensor,iq:torch.Tensor,ik:torch.Tensor,
         gq:torch.Tensor,gk:torch.Tensor,gv:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    rq,h,d=q.shape;rk=kv.shape[0];hk=wk.shape[0];br=int(os.environ.get('MIRAGE_TEXT_NORM_ROWS','16'));assert br in (4,8,16);tq=triton.cdiv(rq,br);tk=triton.cdiv(rk,br)
    dq=torch.empty(q.shape,device=q.device,dtype=q.dtype);dkv=torch.empty(kv.shape,device=kv.device,dtype=kv.dtype)
    dwq=torch.empty_like(wq);dwk=torch.empty_like(wk)
    pq=torch.empty((tq,h,d),device=q.device,dtype=torch.float32);pk=torch.empty((tk,hk,d),device=q.device,dtype=torch.float32)
    _backward[(max(tq,tk),max(h,hk))](q,kv,wq,wk,iq,ik,gq.contiguous(),gk.contiguous(),gv.contiguous(),dq,dkv,pq,pk,
        rq,rk,h,hk,d,q.stride(0),kv.stride(0),triton.next_power_of_2(d),br,num_warps=4,enable_fp_fusion=False)
    _reduce_w[(triton.cdiv(d,128),h)](pq,dwq,tq*h,d,h,BC=128,BR=32,num_warps=4)
    _reduce_w[(triton.cdiv(d,128),hk)](pk,dwk,tk*hk,d,hk,BC=128,BR=32,num_warps=4)
    return dq,dkv,dwq,dwk


@_bwd.register_fake
def _fake_bwd(q,kv,wq,wk,iq,ik,gq,gk,gv):
    return torch.empty(q.shape,device=q.device,dtype=q.dtype),torch.empty(kv.shape,device=kv.device,dtype=kv.dtype),torch.empty_like(wq),torch.empty_like(wk)


def _setup(ctx,inputs,output):
    ctx.save_for_backward(*inputs[:4],*output[3:]);ctx.mark_non_differentiable(*output[3:])
def _autograd(ctx,gq,gk,gv,unused_iq,unused_ik):
    return (*_bwd(*ctx.saved_tensors,gq,gk,gv),None)
_fwd.register_autograd(_autograd,setup_context=_setup)


def text_qkv(q,kv,wq,wk,eps):return _fwd(q,kv,wq,wk,eps)[:3]
