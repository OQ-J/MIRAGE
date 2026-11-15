"""Per-block history gradients: one buffer per source, three consumers, no detach loss.

Each consumer sends a scalar dependency to HistoryToken. The token backward runs
after all consumers and returns the combined gradients to the ORIGINAL features.
Only fixed history is merged; each changing partial has its ordinary gradient.
"""
import threading
import torch
from mirage.ops.fused_ops import rms,ada_rms


def reference(values,q,dw,w,s,t,eps):
    logits=torch.stack([(rms(v.float(),dw,eps)*q.float()).sum(-1) for v in values])
    weights=logits.softmax(0)
    read=sum(a[...,None]*v.float() for a,v in zip(weights.unbind(0),values)).to(values[0].dtype)
    return ada_rms(read,w,s,t,eps)


class HistoryGroup:
    def __init__(self,values):
        self.original=tuple(values)
        self.values=tuple(v.detach() for v in values)
        self.states={};self.lock=threading.RLock()
        self.stream=torch.cuda.current_stream(values[0].device).cuda_stream if values[0].device.type=='cuda' else None

    def state(self):
        task=torch._C._current_graph_task_id()
        if task not in self.states:
            self.states[task]=dict(buffers=[torch.empty_like(v) for v in self.values],initialized=False,table=None)
            # Also release buffers when autograd.grad prunes the history inputs.
            torch.autograd.Variable._execution_engine.queue_callback(lambda:self.states.pop(task,None))
        return self.states[task]


class _HistoryToken(torch.autograd.Function):
    @staticmethod
    def forward(ctx,*args):
        *values,group=args;ctx.group=group
        return torch.zeros((),device=values[0].device,dtype=torch.float32)

    @staticmethod
    def backward(ctx,unused):
        group=ctx.group;task=torch._C._current_graph_task_id()
        with group.lock:state=group.states.pop(task,None)
        if state is None:raise RuntimeError('History gradients were not produced before their dependency token')
        return (*state['buffers'],None)


def begin(history):
    group=HistoryGroup(history)
    return group,_HistoryToken.apply(*history,group)


class _MergedAda(torch.autograd.Function):
    @staticmethod
    def forward(ctx,token,partial,q,dw,w,s,t,eps,cached,group):
        values=group.values if partial is None else group.values+(partial,)
        ctx.group=group;ctx.partial=partial is not None;ctx.count=len(values);ctx.eps=eps
        if values[0].device.type=='cuda':
            from mirage.ops.depth_ada_triton import _fwd
            if cached is None:cached=torch.empty((0,values[0].numel()//values[0].shape[-1]),device=q.device,dtype=torch.float32)
            y,weights,inv,ptrs,read=_fwd(list(values),q,dw,w,s,t,eps,cached)
        else:
            y=reference(values,q,dw,w,s,t,eps)
            weights=inv=torch.empty((0,),device=q.device);ptrs=torch.empty((0,),device=q.device,dtype=torch.uint64)
        if values[0].device.type!='cuda':read=torch.empty((0,),device=q.device)
        ctx.save_for_backward(*values,q,dw,w,s,t,weights,inv,ptrs,read)
        return y

    @staticmethod
    def backward(ctx,go):
        saved=ctx.saved_tensors;k=ctx.count;values=saved[:k];q,dw,w,s,t,weights,inv,ptrs,read=saved[k:]
        group=ctx.group
        if go.device.type=='cuda':
            from mirage.ops.history_merge_triton import merged_backward
            with group.lock:
                state=group.state()
                if state['table'] is None:
                    from mirage.ops.depth_triton import make_table
                    state['table']=make_table(state['buffers'])
                dp,dq,ddw,denw,ds,dt=merged_backward(list(values),q,dw,w,s,t,weights,inv,go,ctx.eps,state['buffers'],state['initialized'],ptrs,self_table=state['table'],read=read)
                state['initialized']=True
            grads=(dq,ddw,denw,ds,dt)
            dp=dp if ctx.partial else None
        else:
            with torch.enable_grad():
                inputs=[x.detach().requires_grad_(True) for x in (*values,q,dw,w,s,t)]
                y=reference(inputs[:k],*inputs[k:],ctx.eps)
                derivative=torch.autograd.grad(y,inputs,go,create_graph=False)
            with group.lock:
                state=group.state()
                for destination,gradient in zip(state['buffers'],derivative[:len(group.values)]):
                    if state['initialized']:destination.add_(gradient)
                    else:destination.copy_(gradient)
                state['initialized']=True
            dp=derivative[k-1] if ctx.partial else None;grads=derivative[k:]
        signal=torch.zeros((),device=go.device,dtype=torch.float32)
        return signal,dp,*grads,None,None,None


def read(handle,partial,q,dw,w,s,t,eps,cached=None):
    group,token=handle
    if group.stream is not None and torch.cuda.current_stream(q.device).cuda_stream!=group.stream:
        from mirage.ops.depth_ada_triton import depth_ada
        values=group.original if partial is None else group.original+(partial,)
        return depth_ada(values,q,dw,w,s,t,eps,cached=cached)
    return _MergedAda.apply(token,partial,q,dw,w,s,t,eps,cached,group)
