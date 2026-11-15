"""Bounded batched Muon workspace; optimizer-only fixed-shape compilation."""
import torch

def ns(x,steps):
    x=(x.float()/(torch.linalg.vector_norm(x.float(),dim=(-2,-1),keepdim=True)+1e-7)).to(torch.bfloat16)
    for _ in range(steps):
        a=x@x.mT
        b=-4.7750*a+2.0315*(a@a)
        x=3.4445*x+b@x
    return x

_compiled=None
def orthogonalize_batch(x,steps):
    global _compiled
    if x.device.type=='cuda':
        if _compiled is None:
            _compiled=torch.compile(ns,fullgraph=True,dynamic=False,
                options={'triton.cudagraphs':False,'emulate_precision_casts':True})
        with torch._dynamo.config.patch(recompile_limit=64):
            return _compiled(x,steps)
    return ns(x,steps)

def prepare(g,buffer,mu,nesterov):
    buffer.mul_(mu).add_(g)
    return g.add(buffer,alpha=mu) if nesterov else buffer

# Compiled elementwise functions fuse GPU reads/writes while retaining FP32 state.
_prepare=torch.compile(prepare,fullgraph=True,dynamic=True)
def direction(master,buffer,mu,nesterov):
    return _prepare(master.grad,buffer,mu,nesterov) if master.is_cuda else prepare(master.grad,buffer,mu,nesterov)

def finish(master,model,update,decay,rate):
    master.mul_(decay)
    master.add_(update*rate)
    model.copy_(master)
_finish=torch.compile(finish,fullgraph=True,dynamic=True,
    options={'emulate_precision_casts':True})
def write(master,model,update,decay,rate):
    with torch._dynamo.config.patch(recompile_limit=64):
        return _finish(master,model,update,decay,rate) if master.is_cuda else finish(master,model,update,decay,rate)

class Workspace:
    def __init__(self):self.buffers={}
    def get(self,shape,device):
        # At most four logical matrices per batch, <=128 MiB input staging.
        if shape not in self.buffers:self.buffers[shape]=torch.empty(shape,device=device,dtype=torch.bfloat16)
        return self.buffers[shape]
