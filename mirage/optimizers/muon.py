"""Single-device Muon/AdamW with FP32 masters and RMS-matched updates.
Fused QKV/KV/SwiGLU projections use separate logical matrices.
This is the project integration, not an official upstream implementation.
See ACKNOWLEDGEMENTS.md for algorithm references.
"""
import copy,math
import torch
from torch import nn
from mirage.optimizers.optim import MasterAdamW
from mirage.architecture.model import HeadNorm,TextAdapterBlock,ImageAttention,TextCrossAttention,SwiGLU,TransformerBlock

FORMAT='v27-moss-style-muon-adamw-v1'


def parameter_rules(model):
    modules=dict(model.named_modules())
    final=next((name for name,module in reversed(list(modules.items())) if isinstance(module,TransformerBlock)),None)
    rules=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        parent,_,leaf=name.rpartition('.');module=modules.get(parent)
        # Control/time projections are explicitly excluded in this architecture.
        excluded=name in ('input.weight','output.weight') or name.startswith('time_embedding.') or parent.endswith(('.time','.gate'))
        muon=isinstance(module,nn.Linear) and leaf=='weight' and p.ndim==2 and not excluded
        splits=(p.shape[0],) if p.ndim==2 else ()
        if muon:
            owner=modules.get(parent.rpartition('.')[0])
            if isinstance(owner,TextAdapterBlock) and parent.endswith('.qkv'):splits=(owner.heads*owner.dim,)*3
            elif isinstance(owner,ImageAttention) and parent.endswith('.qkv'):splits=(owner.q_heads*owner.dim,owner.kv_heads*owner.dim,owner.kv_heads*owner.dim)
            elif isinstance(owner,TextCrossAttention) and parent.endswith('.kv'):splits=(owner.kv_heads*owner.dim,)*2
            elif isinstance(owner,TextAdapterBlock) and parent.endswith('.ffn_in'):splits=(p.shape[0]//2,)*2
            elif isinstance(owner,SwiGLU) and parent.endswith('.input'):splits=(p.shape[0]//2,)*2
            if sum(splits)!=p.shape[0]:raise ValueError(f'Invalid Muon split for {name}')
        normalization=isinstance(module,HeadNorm) or module.__class__.__name__=='RMSNorm'
        embedding=name=='input.weight' or name=='text_adapter.register_tokens' or name.startswith('time_embedding.')
        no_decay=p.ndim<2 or leaf=='bias' or normalization or embedding
        decay_to_one=bool(final and parent.startswith(final+'.layers.') and isinstance(module,HeadNorm))
        rules.append(dict(name=name,shape=list(p.shape),algorithm='muon' if muon else 'adamw',
                          splits=list(splits) if muon else [],decay=not no_decay,decay_to_one=decay_to_one))
    return rules


def orthogonalize(matrix,steps=5):
    """Quintic Newton-Schulz, BF16 matrix multiplies and FP32 norm statistic.

    The shorter dimension is the Gram axis; only one logical matrix is live here.
    FP32 master/momentum are never replaced by BF16 optimizer state.
    """
    x=matrix.to(torch.bfloat16)
    transposed=x.shape[0]>x.shape[1]
    if transposed:x=x.mT
    x=(x.float()/(x.float().norm()+1e-7)).to(torch.bfloat16)
    for _ in range(steps):
        a=x@x.mT
        b=-4.7750*a+2.0315*(a@a)
        x=3.4445*x+b@x
    return x.mT if transposed else x


class MasterMuonAdamW(MasterAdamW):
    def __init__(self,model,lr=5e-5,betas=(.9,.95),weight_decay=.01,eps=1e-8,
                 momentum=.95,ns_steps=5,nesterov=True,rms_scale=.2,decay_to_one=.01):
        values=(lr,weight_decay,eps,momentum,rms_scale,decay_to_one,*betas)
        if not all(math.isfinite(v) for v in values) or lr<0 or weight_decay<0 or eps<=0 or not 0<=momentum<1 or ns_steps<1 or rms_scale<=0 or decay_to_one<0 or not all(0<=v<1 for v in betas):
            raise ValueError('Invalid Muon/AdamW hyperparameters')
        super().__init__(model,lr,betas,weight_decay)
        self.rules=parameter_rules(model)
        self.settings=dict(momentum=momentum,ns_steps=ns_steps,nesterov=nesterov,rms_scale=rms_scale,decay_to_one=decay_to_one)
        groups=[]
        for decay in (True,False):
            parameters=[m for r,m in zip(self.rules,self.masters) if r['algorithm']=='adamw' and r['decay']==decay]
            if parameters:groups.append(dict(params=parameters,weight_decay=weight_decay if decay else 0.))
        self.optimizer=torch.optim.AdamW(groups,lr=lr,betas=betas,eps=eps,fused=self.masters[0].device.type=='cuda')
        self.weight_decay=weight_decay
        self.muon_state={}
        self.updates=0

    @torch.no_grad()
    def _update(self,active,norm,clip,lr):
        # Global norm is checked by the inherited step BEFORE either optimizer moves.
        parameters=[m for _,m in active]
        if clip>0:torch.nn.utils.clip_grads_with_norm_(parameters,clip,norm)
        mu=self.settings['momentum'];active_ids={id(m) for _,m in active}
        from mirage.optimizers.muon_fast import Workspace,direction,orthogonalize_batch,write
        if not hasattr(self,'workspace'):self.workspace=Workspace()
        grouped={};directions=[];written=set()
        model_by_master={id(m):p for p,m in active}
        for rule,master in zip(self.rules,self.masters):
            if id(master) not in active_ids or rule['algorithm']!='muon':continue
            state=self.muon_state.get(rule['name'])
            if state is None:
                state=dict(step=0,momentum_buffer=torch.zeros_like(master));self.muon_state[rule['name']]=state
            state['step']+=1
            d=direction(master,state['momentum_buffer'],mu,self.settings['nesterov'])
            directions.append(d);offset=0
            for rows in rule['splits']:
                view=master.narrow(0,offset,rows);dv=d.narrow(0,offset,rows)
                model_view=model_by_master[id(master)].narrow(0,offset,rows)
                transposed=rows>view.shape[1]
                shape=(min(view.shape),max(view.shape))
                grouped.setdefault(shape,[]).append((view,model_view,dv,transposed,rule['decay']))
                offset+=rows
            written.add(id(master))
        # Tensor scalars avoid recompiling on each learning-rate warmup step.
        scalar_key=(lr,self.weight_decay,self.settings['rms_scale'],str(self.masters[0].device))
        if getattr(self,'_scalar_key',None)!=scalar_key:
            self._scalar_key=scalar_key;self._scalar_cache={}
        scalars=self._scalar_cache
        for shape,entries in grouped.items():
            for decay in (True,False):
                if (shape,decay) not in scalars:
                    scalars[shape,decay]=(torch.tensor(1-lr*self.weight_decay if decay else 1.,device=self.masters[0].device),torch.tensor(-lr*self.settings['rms_scale']*math.sqrt(max(shape)),device=self.masters[0].device))
            limit=max(1,min(4,(128*1024*1024)//(2*shape[0]*shape[1])))
            for start in range(0,len(entries),limit):
                chunk=entries[start:start+limit]
                buf=self.workspace.get((len(chunk),*shape),self.masters[0].device)
                for j,(_,_,dv,transpose,_) in enumerate(chunk):buf[j].copy_(dv.mT if transpose else dv)
                updates=orthogonalize_batch(buf,self.settings['ns_steps'])
                for j,(view,model_view,_,transpose,decay) in enumerate(chunk):
                    u=updates[j].mT if transpose else updates[j]
                    write(view,model_view,u,*scalars[shape,decay])
        del grouped,directions
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.step()
        # Final block self/cross HeadNorms have zero ordinary weight decay.
        # Pull toward one after the Adam update, only when that parameter had grad.
        for rule,master in zip(self.rules,self.masters):
            if rule['decay_to_one'] and id(master) in active_ids:
                master.add_(1-master,alpha=lr*self.settings['decay_to_one'])
        remaining=[(p,m) for p,m in active if id(m) not in written]
        if remaining:torch._foreach_copy_([p for p,_ in remaining],[m for _,m in remaining])
        self.updates+=1

    def summary(self):
        return dict(optimizer='muon_adamw',updates=self.updates,settings=self.settings,
                    **{kind+'_parameters':sum(m.numel() for r,m in zip(self.rules,self.masters) if r['algorithm']==kind) for kind in ('muon','adamw')},
                    logical_muon_matrices=sum(len(r['splits']) for r in self.rules if r['algorithm']=='muon'))

    def state_dict(self):
        return dict(format=FORMAT,masters=[m.detach() for m in self.masters],adamw=self.optimizer.state_dict(),
                    parameter_names=self.parameter_names,rules=copy.deepcopy(self.rules),settings=dict(self.settings),
                    weight_decay=self.weight_decay,muon=self.muon_state,updates=self.updates)

    def validate_state(self,state):
        if state.get('format')!=FORMAT:raise ValueError('AdamW-only checkpoint cannot resume Muon state. Use --reset-optimizer to explicitly switch optimizers while preserving weights/global step/data/RNG.')
        if state.get('parameter_names')!=self.parameter_names or state.get('rules')!=self.rules:
            raise ValueError('Optimizer parameter names/shapes/split rules do not match this model')
        if len(state.get('masters',()))!=len(self.masters):raise ValueError('Incomplete FP32 master weights')
        for master,value in zip(self.masters,state['masters']):
            if value.shape!=master.shape or value.dtype!=torch.float32:raise ValueError('Invalid FP32 master shape/dtype')
        by_name=dict(zip(self.parameter_names,self.masters))
        eligible={r['name'] for r in self.rules if r['algorithm']=='muon'}
        for name,slot in state['muon'].items():
            value=slot['momentum_buffer']
            if name not in eligible or value.shape!=by_name[name].shape or value.dtype!=torch.float32 or not isinstance(slot['step'],int) or slot['step']<1:
                raise ValueError(f'Invalid Muon momentum for {name}')

    def load_state_dict(self,state):
        self.validate_state(state)
        with torch.no_grad():
            for master,value in zip(self.masters,state['masters']):master.copy_(value)
        requested=[{k:v for k,v in group.items() if k!='params'} for group in self.optimizer.param_groups]
        self.optimizer.load_state_dict(state['adamw'])
        for group,settings in zip(self.optimizer.param_groups,requested):group.update(settings)
        # Resume uses current requested hyperparameters; moments and step remain saved.
        self.muon_state={name:dict(step=slot['step'],momentum_buffer=slot['momentum_buffer'].to(device=self.masters[0].device,dtype=torch.float32,copy=True)) for name,slot in state['muon'].items()}
        self.updates=state['updates']
        for slot in self.optimizer.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if key in slot and slot[key].dtype!=torch.float32:raise ValueError('AdamW moments must remain FP32')

    @torch.no_grad()
    def reset_from_state(self,state):
        # An explicit optimizer switch preserves the unrounded FP32 masters if available.
        names=state.get('parameter_names');values=state.get('masters')
        if names!=self.parameter_names or values is None or len(values)!=len(self.masters):
            raise ValueError('Cannot map legacy FP32 masters by exact parameter names')
        for master,value in zip(self.masters,values):
            if value.shape!=master.shape or value.dtype!=torch.float32:raise ValueError('Legacy FP32 master mismatch')
        self.optimizer.state.clear();self.muon_state.clear();self.updates=0
        for p,master,value in zip(self.model_params,self.masters,values):
            master.copy_(value);p.copy_(master)


def build_optimizer(model,config):
    return MasterMuonAdamW(model,config['lr'],tuple(config['betas']),config['weight_decay'],
        config['adam_eps'],config['muon_momentum'],config['muon_ns_steps'],config['muon_nesterov'],
        config['muon_rms_scale'],config['final_qk_decay_to_one'])
