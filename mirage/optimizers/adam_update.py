"""Use the original optimizer state format, with a fused CUDA data pass when supported."""
import torch


def try_update(optimizer,active,norm,clip,lr,chunk_bytes):
    if not active or norm.device.type!='cuda':return False
    lookup={id(master):model for model,master in active};batches=[]
    # Validate every group before mutating any parameter or step counter.
    for group in optimizer.param_groups:
        selected=[p for p in group['params'] if id(p) in lookup]
        if not selected:continue
        if any(group.get(k,False) for k in ('amsgrad','maximize','differentiable','capturable')):return False
        if hasattr(optimizer,'grad_scale') or hasattr(optimizer,'found_inf'):return False
        if not all(isinstance(group[k],(int,float)) for k in ('lr','eps','weight_decay')):return False
        if not all(isinstance(beta,(int,float)) and 0<beta<1 for beta in group['betas']):return False
        for p in selected:
            y=lookup[id(p)];state=optimizer.state[p]
            if p.dtype!=torch.float32 or y.dtype not in (torch.bfloat16,torch.float32):return False
            if p.device!=norm.device or y.device!=norm.device or p.grad.device!=norm.device or p.grad.dtype!=torch.float32:return False
            if p.shape!=y.shape or p.grad.shape!=p.shape or not p.is_contiguous() or not y.is_contiguous() or not p.grad.is_contiguous():return False
            if state:
                if set(state)!= {'step','exp_avg','exp_avg_sq'}:return False
                if state['step'].device!=norm.device or state['step'].dtype!=torch.float32 or state['step'].numel()!=1:return False
                if any(state[k].dtype!=torch.float32 or state[k].device!=p.device or state[k].shape!=p.shape or not state[k].is_contiguous() for k in ('exp_avg','exp_avg_sq')):return False
        batches.append((group,selected))
    from mirage.optimizers.adam_update_triton import update
    for group,selected in batches:
        chunks=[];chunk=[];size=0;dtype=None
        for p in selected:
            y=lookup[id(p)]
            if chunk and (size+p.numel()*4>chunk_bytes or y.dtype!=dtype):chunks.append(chunk);chunk=[];size=0
            state=optimizer.state[p]
            if not state:
                state.update(step=torch.zeros((),device=p.device,dtype=torch.float32),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
            chunk.append((p,y,state));size+=p.numel()*4;dtype=y.dtype
        if chunk:chunks.append(chunk)
        for chunk in chunks:
            update([p for p,_,_ in chunk],[p.grad for p,_,_ in chunk],
                [s['exp_avg'] for _,_,s in chunk],[s['exp_avg_sq'] for _,_,s in chunk],
                [y for _,y,_ in chunk],[s['step'] for _,_,s in chunk],norm,float(lr),float(clip),
                *map(float,group['betas']),float(group['weight_decay']),float(group['eps']))
    return True
