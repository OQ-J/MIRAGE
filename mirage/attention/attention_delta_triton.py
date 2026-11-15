"""One FP32 O*dO reduction per attention row, reused by dQ and all dK/dV tiles."""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['R'])
def _delta(O,GO,Delta,R,D:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR);d=tl.arange(0,BD)
    out=tl.load(O+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0).to(tl.float32)
    go=tl.load(GO+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0).to(tl.float32)
    tl.store(Delta+r,tl.sum(out*go,1),r<R)

def row_delta(out,go):
    d=out.shape[-1];rows=out.numel()//d
    delta=torch.empty((rows,),device=out.device,dtype=torch.float32)
    _delta[(triton.cdiv(rows,4),)](out,go,delta,rows,d,triton.next_power_of_2(d),BR=4,num_warps=4,enable_fp_fusion=False)
    return delta
