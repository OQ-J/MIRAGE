"""Ordinary variable-length Softmax attention, token-major storage and tiled backward."""
import math,os
import hashlib
from pathlib import Path
from collections import OrderedDict
import torch
import triton
import triton.language as tl
from mirage.attention.attention_delta_triton import row_delta


@triton.jit
def _dot(a,b,FAST:tl.constexpr):
    if FAST:
        return tl.dot(a.to(tl.bfloat16),b.to(tl.bfloat16),out_dtype=tl.float32)
    return tl.dot(a,b,input_precision='tf32x3')


@triton.jit(do_not_specialize=['N_BUCKET','L_BUCKET','B_BUCKET'])
def _forward(Q,K,V,CQ,CK,O,LSE,Tasks,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
             BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr,
             N_BUCKET=0,L_BUCKET=0,B_BUCKET=0,G=0,Y=0,GATED:tl.constexpr=False,SAVE_STATS:tl.constexpr=True):
    b=tl.load(Tasks+tl.program_id(0)*2);block=tl.load(Tasks+tl.program_id(0)*2+1);h=tl.program_id(1)
    q0,q1=tl.load(CQ+b),tl.load(CQ+b+1)
    k0,k1=tl.load(CK+b),tl.load(CK+b+1)
    nq,nk=q1-q0,k1-k0;kh=h//(H//HK)
    r=block*BM+tl.arange(0,BM);d=tl.arange(0,BD)
    if block*BM<nq:
        q=tl.load(Q+((q0+r[:,None])*H+h)*D+d[None,:],(r[:,None]<nq)&(d[None,:]<D),0).to(tl.float32)
        m=tl.full((BM,),float('-inf'),tl.float32);denom=tl.full((BM,),0.,tl.float32)
        acc=tl.full((BM,BD),0.,tl.float32)
        for start in range(tl.cdiv(nk,BK)):
            c=start*BK+tl.arange(0,BK)
            k=tl.load(K+((k0+c[None,:])*HK+kh)*D+d[:,None],(c[None,:]<nk)&(d[:,None]<D),0).to(tl.float32)
            s=_dot(q,k,FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
            s=tl.where(c[None,:]<nk,s,float('-inf'))
            new=tl.maximum(m,tl.max(s,1));alpha=tl.exp(m-new)
            p=tl.exp(s-new[:,None])
            v=tl.load(V+((k0+c[:,None])*HK+kh)*D+d[None,:],(c[:,None]<nk)&(d[None,:]<D),0).to(tl.float32)
            acc=acc*alpha[:,None]+_dot(p,v,FAST=Q.dtype.element_ty==tl.bfloat16)
            denom=denom*alpha+tl.sum(p,1);m=new
        result=acc/denom[:,None]
        if SAVE_STATS or not GATED:
            tl.store(O+((q0+r[:,None])*H+h)*D+d[None,:],result,(r[:,None]<nq)&(d[None,:]<D))
        if SAVE_STATS:tl.store(LSE+(q0+r)*H+h,m+tl.log(denom),r<nq)
        if GATED:
            logit=tl.load(G+(q0+r)*H+h,r<nq,0).to(tl.float32)
            gate=tl.sigmoid(logit).to(V.dtype.element_ty).to(tl.float32)
            rounded=result.to(V.dtype.element_ty).to(tl.float32)
            tl.store(Y+((q0+r[:,None])*H+h)*D+d[None,:],rounded*gate[:,None],(r[:,None]<nq)&(d[None,:]<D))


@triton.jit(do_not_specialize=['N_BUCKET','L_BUCKET','B_BUCKET'])
def _dq(Q,K,V,CQ,CK,Delta,LSE,GO,DQ,Tasks,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
        BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr,
             N_BUCKET=0,L_BUCKET=0,B_BUCKET=0):
    b=tl.load(Tasks+tl.program_id(0)*2);block=tl.load(Tasks+tl.program_id(0)*2+1);h=tl.program_id(1)
    q0,q1=tl.load(CQ+b),tl.load(CQ+b+1);k0,k1=tl.load(CK+b),tl.load(CK+b+1)
    nq,nk=q1-q0,k1-k0;kh=h//(H//HK)
    r=block*BM+tl.arange(0,BM);d=tl.arange(0,BD)
    if block*BM<nq:
        off=((q0+r[:,None])*H+h)*D+d[None,:];active=(r[:,None]<nq)&(d[None,:]<D)
        q=tl.load(Q+off,active,0).to(tl.float32);go=tl.load(GO+off,active,0).to(tl.float32)
        delta=tl.load(Delta+(q0+r)*H+h,r<nq,0)
        lse=tl.load(LSE+(q0+r)*H+h,r<nq,0);dq=tl.full((BM,BD),0.,tl.float32)
        for start in range(tl.cdiv(nk,BK)):
            c=start*BK+tl.arange(0,BK);activek=(c[None,:]<nk)&(d[:,None]<D)
            offk=((k0+c[None,:])*HK+kh)*D+d[:,None]
            k=tl.load(K+offk,activek,0).to(tl.float32);v=tl.load(V+offk,activek,0).to(tl.float32)
            s=_dot(q,k,FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
            p=tl.where(c[None,:]<nk,tl.exp(s-lse[:,None]),0.)
            ds=p*(_dot(go,v,FAST=Q.dtype.element_ty==tl.bfloat16)-delta[:,None])
            dq+=_dot(ds,tl.trans(k),FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
        tl.store(DQ+off,dq,active)


@triton.jit(do_not_specialize=['N_BUCKET','L_BUCKET','B_BUCKET','TK'])
def _dkv(Q,K,V,CQ,CK,Delta,LSE,GO,DK,DV,Tasks,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
         BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr,
             N_BUCKET=0,L_BUCKET=0,B_BUCKET=0,SPLITS:tl.constexpr=1,TK=0):
    split=tl.program_id(2) if SPLITS>1 else 0
    b=tl.load(Tasks+tl.program_id(0)*2);block=tl.load(Tasks+tl.program_id(0)*2+1);kh=tl.program_id(1)
    q0,q1=tl.load(CQ+b),tl.load(CQ+b+1);k0,k1=tl.load(CK+b),tl.load(CK+b+1)
    nq,nk=q1-q0,k1-k0
    c=block*BK+tl.arange(0,BK);d=tl.arange(0,BD)
    if block*BK<nk:
        off=((k0+c[:,None])*HK+kh)*D+d[None,:];active=(c[:,None]<nk)&(d[None,:]<D)
        k=tl.load(K+off,active,0).to(tl.float32);v=tl.load(V+off,active,0).to(tl.float32)
        dk=tl.full((BK,BD),0.,tl.float32);dv=tl.full((BK,BD),0.,tl.float32)
        for group in range(H//HK):
            h=kh*(H//HK)+group
            tiles=tl.cdiv(nq,BM)
            for start in range(tiles*split//SPLITS,tiles*(split+1)//SPLITS):
                r=start*BM+tl.arange(0,BM);qoff=((q0+r[:,None])*H+h)*D+d[None,:]
                qa=(r[:,None]<nq)&(d[None,:]<D)
                q=tl.load(Q+qoff,qa,0).to(tl.float32);go=tl.load(GO+qoff,qa,0).to(tl.float32)
                delta=tl.load(Delta+(q0+r)*H+h,r<nq,0)
                lse=tl.load(LSE+(q0+r)*H+h,r<nq,0)
                s=_dot(q,tl.trans(k),FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
                p=tl.where((r[:,None]<nq)&(c[None,:]<nk),tl.exp(s-lse[:,None]),0.)
                ds=p*(_dot(go,tl.trans(v),FAST=Q.dtype.element_ty==tl.bfloat16)-delta[:,None])
                dk+=_dot(tl.trans(ds),q,FAST=Q.dtype.element_ty==tl.bfloat16)*SCALE
                dv+=_dot(tl.trans(p),go,FAST=Q.dtype.element_ty==tl.bfloat16)
        outoff=split*TK*HK*D+off
        tl.store(DK+outoff,dk,active);tl.store(DV+outoff,dv,active)


TILES=((16,32,4,1),(32,64,4,1),(64,64,4,1),(32,32,4,2),
       (64,32,4,2),(64,64,4,2),(64,128,4,2),(128,64,8,2))
if os.environ.get('MIRAGE_TUNE_EXTENDED','1')=='1':
    TILES=TILES+((16,32,2,1),(32,32,2,2),(32,64,4,3),(64,64,4,3))
_selected={}
_code_id=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
def _selection_file():
    from pathlib import Path
    return Path(os.environ.get('MIRAGE_TUNE_CACHE',str(Path.home()/'.cache/mirage')))/'varlen_mirage_v1.json'
def _save_selection(key,value):
    import json
    path=_selection_file();path.parent.mkdir(parents=True,exist_ok=True)
    try:data=json.loads(path.read_text())
    except (OSError,ValueError):data={}
    data[repr(key)]=value;tmp=path.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(data));os.replace(tmp,path)
def _load_selection(key):
    import json
    try:return json.loads(_selection_file().read_text()).get(repr(key))
    except (OSError,ValueError):return None
_forward_tuned=_dq_tuned=_dkv_tuned=None

@triton.jit(do_not_specialize=['TOTAL'])
def _build_tasks(Cu,Tasks,TOTAL,B:tl.constexpr,TILE:tl.constexpr,BB:tl.constexpr,BT:tl.constexpr):
    seq=tl.arange(0,BB)
    left=tl.load(Cu+seq,seq<B,0);right=tl.load(Cu+seq+1,seq<B,0)
    counts=tl.where(seq<B,tl.cdiv(right-left,TILE),0)
    ends=tl.cumsum(counts,0)
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    batch=tl.sum(((t[:,None]>=ends[None,:])&(seq[None,:]<B)).to(tl.int32),1)
    previous=tl.gather(ends,tl.minimum(tl.maximum(batch-1,0),BB-1),0)
    start=tl.where(batch==0,0,previous)
    tl.store(Tasks+t*2,batch,t<TOTAL);tl.store(Tasks+t*2+1,t-start,t<TOTAL)


_task_cache=OrderedDict()
def _tasks(lengths,tile,cu):
    key=(lengths,tile,str(cu.device));cached=_task_cache.get(key)
    stream=torch.cuda.current_stream(cu.device) if cu.device.type=='cuda' else None
    if cached is not None:
        _task_cache.move_to_end(key);table,event,owner=cached
        if stream is not None and stream.cuda_stream!=owner:
            stream.wait_event(event);table.record_stream(stream)
        return table
    count=sum(triton.cdiv(n,tile) for n in lengths)
    table=torch.empty((count,2),device=cu.device,dtype=torch.int32)
    _build_tasks[(triton.cdiv(count,128),)](cu,table,count,len(lengths),tile,triton.next_power_of_2(len(lengths)),128,num_warps=4)
    event=None;owner=None
    if stream is not None:event=torch.cuda.Event();event.record(stream);owner=stream.cuda_stream
    _task_cache[key]=(table,event,owner)
    if len(_task_cache)>256:_task_cache.popitem(last=False)
    return table

def _args(q,k,max_q,max_k,batch):
    return dict(H=q.shape[1],HK=k.shape[1],D=q.shape[2],
                BD=max(32,triton.next_power_of_2(q.shape[2])),SCALE=1/math.sqrt(q.shape[2]),
                N_BUCKET=triton.next_power_of_2(max_q),L_BUCKET=triton.next_power_of_2(max_k),
                B_BUCKET=triton.next_power_of_2(batch))

def _launch(kernel,tuned,grid,args,kwargs,lengths):
    kind='BK' if kernel.__name__=='_dkv' else 'BM'
    heads=kwargs['HK'] if kind=='BK' else kwargs['H']
    if os.environ.get('TRITON_INTERPRET')=='1':
        m,n,w=map(int,os.environ.get('MIRAGE_ATTN_TEST_TILE','16,32,4').split(','))
        tasks=_tasks(tuple(lengths),n if kind=='BK' else m,args[4] if kind=='BK' else args[3])
        kernel[(len(tasks),heads,kwargs.get('SPLITS',1))](*args,Tasks=tasks,**kwargs,BM=m,BK=n,num_warps=w,num_stages=1)
    else:
        gated=bool(kwargs.get('GATED',False));key=(kernel.__name__,gated,kwargs.get('SPLITS',1),kwargs.get('SAVE_STATS',True))
        # Each candidate dispatches its own exact compact task list. Cache only
        # scalar tile/warp/stage choices; no device addresses go into disk state.
        selection=(key,kwargs['D'],kwargs['H'],kwargs['HK'],kwargs['N_BUCKET'],kwargs['L_BUCKET'],kwargs['B_BUCKET'],str(args[0].dtype),torch.cuda.get_device_capability(args[0].device),torch.__version__,triton.__version__)
        props=torch.cuda.get_device_properties(args[0].device)
        shape_profile=(tuple(args[0].stride()),tuple(args[1].stride()),sum(lengths)//max(1,len(lengths))//32)
        selection=selection+(_code_id,TILES,props.name,props.multi_processor_count,props.total_memory,torch.version.cuda,shape_profile)
        config=_selected.get(selection) or _load_selection(selection)
        if config is None:
            from triton.testing import do_bench
            choices=[]
            for m,n,w,st in TILES:
                tasks=_tasks(tuple(lengths),n if kind=='BK' else m,args[4] if kind=='BK' else args[3])
                launch=lambda:kernel[(len(tasks),heads,kwargs.get('SPLITS',1))](*args,Tasks=tasks,**kwargs,BM=m,BK=n,num_warps=w,num_stages=st)
                try:
                    launch();ms=do_bench(launch,warmup=5,rep=15)
                    choices.append((ms,(m,n,w,st)))
                except triton.OutOfResources:pass
            if not choices:raise RuntimeError('No valid attention tile')
            config=min(choices)[1];_selected[selection]=config
            _save_selection(selection,config)
        _selected[selection]=config
        m,n,w,st=config;tasks=_tasks(tuple(lengths),n if kind=='BK' else m,args[4] if kind=='BK' else args[3])
        kernel[(len(tasks),heads,kwargs.get('SPLITS',1))](*args,Tasks=tasks,**kwargs,BM=m,BK=n,num_warps=w,num_stages=st)


@triton.jit(do_not_specialize=['N'])
def _reduce_split(K,V,DK,DV,N,S:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.full((BLOCK,),0.,tl.float32);b=tl.full((BLOCK,),0.,tl.float32)
    for j in range(S):
        a+=tl.load(K+j*N+i,i<N,0);b+=tl.load(V+j*N+i,i<N,0)
    tl.store(DK+i,a,i<N);tl.store(DV+i,b,i<N)


def split_count(max_q,max_k,elements):
    choice=os.environ.get('MIRAGE_DKV_SPLITS','auto')
    if choice=='auto':count=4 if max_q>=1024 and max_q>=4*max_k else 1
    else:
        count=int(choice)
        if count not in (1,2,4,8):raise ValueError('MIRAGE_DKV_SPLITS: auto/1/2/4/8')
    budget=int(os.environ.get('MIRAGE_DKV_WORKSPACE_MB','64'))*1024**2
    if budget<1:raise ValueError('MIRAGE_DKV_WORKSPACE_MB must be positive')
    while count>1 and (2*count*elements*4>budget or max_q<32*count):count//=2
    return count


def _dispatch_dkv(q,k,v,cq,ck,delta,lse,go,dk,dv,args,kl,max_q,max_k):
    splits=split_count(max_q,max_k,k.numel())
    if splits==1:
        pk,pv=dk,dv
    else:
        pk=torch.empty((splits,*k.shape),device=k.device,dtype=torch.float32)
        pv=torch.empty_like(pk)
    options=dict(args,SPLITS=splits,TK=k.shape[0])
    _launch(_dkv,None,None,(q,k,v,cq,ck,delta,lse,go,pk,pv),options,kl)
    if splits>1:
        _reduce_split[(triton.cdiv(k.numel(),256),)](pk,pv,dk,dv,k.numel(),splits,256,num_warps=4)



@torch.library.custom_op('v31rf::varlen_forward',mutates_args=())
def _fwd(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,cq:torch.Tensor,ck:torch.Tensor,
         max_q:int,max_k:int,ql:list[int],kl:list[int])->tuple[torch.Tensor,torch.Tensor]:
    out=torch.empty_like(q,dtype=torch.float32);lse=torch.empty(q.shape[:2],device=q.device,dtype=torch.float32)
    _launch(_forward,_forward_tuned,lambda meta:(triton.cdiv(max_q,meta['BM']),(cq.numel()-1)*q.shape[1]),
            (q,k,v,cq,ck,out,lse),_args(q,k,max_q,max_k,cq.numel()-1),ql)
    return out,lse


@_fwd.register_fake
def _fake(q,k,v,cq,ck,max_q,max_k,ql,kl):
    return torch.empty_like(q,dtype=torch.float32),torch.empty(q.shape[:2],device=q.device,dtype=torch.float32)


@torch.library.custom_op('v31rf::varlen_backward',mutates_args=())
def _bwd(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,cq:torch.Tensor,ck:torch.Tensor,
         out:torch.Tensor,lse:torch.Tensor,go:torch.Tensor,max_q:int,max_k:int,ql:list[int],kl:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    args=_args(q,k,max_q,max_k,cq.numel()-1);go=go.contiguous();b=cq.numel()-1;delta=row_delta(out,go)
    _launch(_dq,_dq_tuned,lambda meta:(triton.cdiv(max_q,meta['BM']),b*q.shape[1]),
            (q,k,v,cq,ck,delta,lse,go,dq),args,ql)
    _dispatch_dkv(q,k,v,cq,ck,delta,lse,go,dk,dv,args,kl,max_q,max_k)
    return dq,dk,dv


@_bwd.register_fake
def _fake_bwd(q,k,v,cq,ck,out,lse,go,max_q,max_k,ql,kl):
    return torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)


def _setup(ctx,inputs,output):
    q,k,v,cq,ck,max_q,max_k,ql,kl=inputs
    ctx.save_for_backward(q,k,v,cq,ck,*output);ctx.maximum=(max_q,max_k,ql,kl)
    ctx.mark_non_differentiable(output[1])


def _autograd(ctx,go,unused):
    q,k,v,cq,ck,out,lse=ctx.saved_tensors
    dq,dk,dv=_bwd(q,k,v,cq,ck,out,lse,go,*ctx.maximum)
    return dq,dk,dv,None,None,None,None,None,None


_fwd.register_autograd(_autograd,setup_context=_setup)


def varlen_attention(q,k,v,cq,ck,max_q,max_k,ql,kl):
    if torch.is_inference_mode_enabled() and os.environ.get('MIRAGE_INFERENCE_KERNELS','1')=='1':
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        out=torch.empty_like(q)
        _launch(_forward,None,None,(q,k,v,cq,ck,out,out),dict(_args(q,k,max_q,max_k,len(ql)),SAVE_STATS=False),ql)
        return out
    return _fwd(q.contiguous(),k.contiguous(),v.contiguous(),cq,ck,max_q,max_k,list(ql),list(kl))[0].to(v.dtype)


@triton.jit(do_not_specialize=['R'])
def _gated_preprocess(O,G,GY,GO,DG,Delta,R,D:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR);d=tl.arange(0,BD)
    original=tl.load(O+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0).to(tl.float32)
    y=original.to(GY.dtype.element_ty).to(tl.float32)
    gy=tl.load(GY+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0).to(tl.float32)
    gate=tl.sigmoid(tl.load(G+r,r<R,0).to(tl.float32))
    rounded=gate.to(GY.dtype.element_ty).to(tl.float32)
    go=(gy*rounded[:,None]).to(GY.dtype.element_ty).to(tl.float32)
    tl.store(GO+r[:,None]*D+d[None,:],go,(r[:,None]<R)&(d[None,:]<D))
    contributions=(gy*y).to(GY.dtype.element_ty).to(tl.float32)
    dg=tl.sum(contributions,1).to(GY.dtype.element_ty).to(tl.float32)
    tl.store(DG+r,(dg*(1-gate))*gate,r<R)
    tl.store(Delta+r,tl.sum(original*go,1),r<R)


@torch.library.custom_op('v31rf::gated_varlen_forward',mutates_args=())
def _gated_fwd(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,cq:torch.Tensor,ck:torch.Tensor,
               max_q:int,max_k:int,g:torch.Tensor,ql:list[int],kl:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    out=torch.empty_like(q,dtype=torch.float32);lse=torch.empty(q.shape[:2],device=q.device,dtype=torch.float32)
    y=torch.empty_like(q)
    kwargs=dict(_args(q,k,max_q,max_k,cq.numel()-1),G=g,Y=y,GATED=True)
    _launch(_forward,None,lambda meta:(triton.cdiv(max_q,meta['BM']),(cq.numel()-1)*q.shape[1]),
            (q,k,v,cq,ck,out,lse),kwargs,ql)
    return y,out,lse


@_gated_fwd.register_fake
def _fake_gated(q,k,v,cq,ck,max_q,max_k,g,ql,kl):
    return torch.empty_like(q),torch.empty_like(q,dtype=torch.float32),torch.empty(q.shape[:2],device=q.device,dtype=torch.float32)


@torch.library.custom_op('v31rf::gated_varlen_backward',mutates_args=())
def _gated_bwd(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,cq:torch.Tensor,ck:torch.Tensor,
               out:torch.Tensor,lse:torch.Tensor,g:torch.Tensor,gy:torch.Tensor,
               max_q:int,max_k:int,ql:list[int],kl:list[int])->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    go=torch.empty_like(q);dg=torch.empty_like(g);delta=torch.empty(q.shape[:2],device=q.device,dtype=torch.float32)
    r=q.numel()//q.shape[-1];d=q.shape[-1];b=cq.numel()-1
    _gated_preprocess[(triton.cdiv(r,4),)](out,g,gy.contiguous(),go,dg,delta,r,d,triton.next_power_of_2(d),BR=4,num_warps=4,enable_fp_fusion=False)
    args=_args(q,k,max_q,max_k,b)
    _launch(_dq,None,lambda meta:(triton.cdiv(max_q,meta['BM']),b*q.shape[1]),
            (q,k,v,cq,ck,delta,lse,go,dq),args,ql)
    _dispatch_dkv(q,k,v,cq,ck,delta,lse,go,dk,dv,args,kl,max_q,max_k)
    return dq,dk,dv,dg


@_gated_bwd.register_fake
def _fake_gated_bwd(q,k,v,cq,ck,out,lse,g,gy,max_q,max_k,ql,kl):
    return torch.empty_like(q),torch.empty_like(k),torch.empty_like(v),torch.empty_like(g)


def _gated_setup(ctx,inputs,output):
    q,k,v,cq,ck,mq,mk,g,ql,kl=inputs
    ctx.save_for_backward(q,k,v,cq,ck,output[1],output[2],g);ctx.maximum=(mq,mk,ql,kl)
    ctx.mark_non_differentiable(output[1],output[2])
def _gated_autograd(ctx,gy,unused_o,unused_lse):
    dq,dk,dv,dg=_gated_bwd(*ctx.saved_tensors,gy,*ctx.maximum)
    return dq,dk,dv,None,None,None,None,dg,None,None
_gated_fwd.register_autograd(_gated_autograd,setup_context=_gated_setup)


def gated_varlen_attention(q,k,v,cq,ck,max_q,max_k,g,ql,kl):
    if torch.is_inference_mode_enabled() and os.environ.get('MIRAGE_INFERENCE_KERNELS','1')=='1':
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        y=torch.empty_like(q)
        kwargs=dict(_args(q,k,max_q,max_k,len(ql)),G=g.contiguous(),Y=y,GATED=True,SAVE_STATS=False)
        _launch(_forward,None,None,(q,k,v,cq,ck,y,y),kwargs,ql)
        return y
    return _gated_fwd(q.contiguous(),k.contiguous(),v.contiguous(),cq,ck,max_q,max_k,g.contiguous(),list(ql),list(kl))[0]
