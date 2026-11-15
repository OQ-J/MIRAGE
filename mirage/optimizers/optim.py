"""FP32-master AdamW update wrapper used by the optional hybrid optimizer."""
import math
import torch

class MasterAdamW:
    grad_cast_chunk_bytes=64*1024*1024
    def __init__(self,model,lr,betas=(.9,.99),weight_decay=.01):
        self.fused_grad_norm=True
        self.fused_body_update=True
        self.parameter_names=[n for n,p in model.named_parameters() if p.requires_grad]
        self.model_params=[p for p in model.parameters() if p.requires_grad]
        self.masters=[torch.nn.Parameter(p.detach().float().clone()) for p in self.model_params]
        self.optimizer=torch.optim.AdamW(self.masters,lr=lr,betas=betas,weight_decay=weight_decay,
                                         fused=self.masters[0].device.type=='cuda')

    @torch.no_grad()
    def step(self,lr,clip=1.,*,loss=None):
        active=[];pending=[];pending_bytes=0;norm_parts=[]
        def convert():
            from mirage.optimizers.grad_norm import cast_and_norm
            sources=[p.grad.detach() for p,_ in pending]
            destinations=[torch.empty_like(master) for _,master in pending]
            if self.fused_grad_norm:norm_parts.append(cast_and_norm(sources,destinations))
            else:
                torch._foreach_copy_(destinations,sources)
                norm_parts.append(torch.stack(torch._foreach_norm(destinations,2)))
            for (p,master),gradient in zip(pending,destinations):
                master.grad=gradient;p.grad=None
        for p,master in zip(self.model_params,self.masters):
            if p.grad is not None:
                size=master.numel()*master.element_size()
                if pending and (pending_bytes+size>self.grad_cast_chunk_bytes or p.grad.dtype!=pending[0][0].grad.dtype):
                    convert();pending=[];pending_bytes=0
                pending.append((p,master));active.append((p,master));pending_bytes+=size
        if pending:convert()
        del pending
        norm=torch.linalg.vector_norm(torch.cat(norm_parts)) if norm_parts else torch.zeros((),device=self.masters[0].device)
        parameters=[m for _,m in active]
        if loss is not None:
            # The mandatory nonfinite check and loss reporting share ONE device->host wait.
            host_loss,host_norm=torch.stack((loss.double(),norm.double())).cpu().tolist()
            if not math.isfinite(host_norm):
                raise RuntimeError('The total gradient norm is non-finite; optimizer update was not performed.')
        else:
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('The total gradient norm is non-finite; optimizer update was not performed.')
        self._update(active,norm,clip,lr)
        for _,master in active:master.grad=None
        self.optimizer.zero_grad(set_to_none=True)
        return norm.detach() if loss is None else (host_loss,host_norm)

    def _update(self,active,norm,clip,lr):
        parameters=[m for _,m in active]
        for group in self.optimizer.param_groups:group['lr']=lr
        from mirage.optimizers.adam_update import try_update
        fused=self.fused_body_update and try_update(self.optimizer,active,norm,clip,lr,self.grad_cast_chunk_bytes)
        if not fused:
            if clip>0:torch.nn.utils.clip_grads_with_norm_(parameters,clip,norm)
            self.optimizer.step()
            if active:torch._foreach_copy_([p for p,_ in active],[m for _,m in active])

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for master in self.masters:master.grad=None
        for p in self.model_params:p.grad=None

    def state_dict(self):
        return dict(masters=[m.detach() for m in self.masters],adamw=self.optimizer.state_dict(),parameter_names=self.parameter_names)

    def load_state_dict(self,state):
        with torch.no_grad():
            for master,value in zip(self.masters,state['masters']):master.copy_(value)
        self.optimizer.load_state_dict(state['adamw'])
        # torch optimizer loading follows the FP32 master parameter dtype.
        assert all(v.dtype==torch.float32 for s in self.optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq'))
