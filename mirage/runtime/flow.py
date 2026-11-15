"""Direct RF velocity: t=0 clean data, t=1 noise; FP32 velocity MSE."""
import torch


def training_inputs(z,t=None,noise=None,*,timestep_sampling="logit_normal"):
    z=z.float()
    if t is None:
        if timestep_sampling=='uniform':
            t=torch.rand(z.shape[0],device=z.device,dtype=torch.float32)
        elif timestep_sampling=='logit_normal':
            t=torch.randn(z.shape[0],device=z.device,dtype=torch.float32).sigmoid().clamp(1e-4,1-1e-4)
        else:
            raise ValueError(f'Unknown timestep sampling: {timestep_sampling}')
    if noise is None:noise=torch.randn_like(z)
    t=t.float()
    if z.device.type=='cuda':
        from mirage.runtime.flow_triton import inputs
        return inputs(z,t,noise)
    xt=(1-t[:,None,None,None])*z+t[:,None,None,None]*noise.float()
    return xt,t,noise.float()-z


def velocity(model,xt,t,text,mask,*,is_null=None,image_mask=None):
    dtype=next(model.parameters()).dtype
    options={}
    if is_null is not None:options['is_null']=is_null
    if image_mask is not None:options['image_mask']=image_mask
    from mirage.runtime.text_cache import TextCache
    return model(xt.to(dtype),t,text if isinstance(text,TextCache) else text.to(dtype),mask,**options).float()


@torch.inference_mode()
def midpoint(model,noise,text,mask,*,steps=40,guidance=1.,null_text=None,null_mask=None,callback=None):
    from mirage.runtime.text_cache import prepare
    text,mask=prepare(model,text,mask)
    if guidance!=1:null_text,null_mask=prepare(model,null_text,null_mask)
    state=noise.float().clone()
    def field(x,time):
        cond=velocity(model,x,time,text,mask)
        if guidance==1:return cond
        uncond=velocity(model,x,time,null_text,null_mask,is_null=torch.ones(x.shape[0],device=x.device,dtype=torch.bool))
        return uncond+guidance*(cond-uncond)
    dt=-1./steps
    for index in range(steps):
        t=torch.full((state.shape[0],),1-index/steps,device=state.device,dtype=torch.float32)
        first=field(state,t)
        middle=state+(.5*dt)*first.float()
        state=state+dt*field(middle,t+.5*dt).float()
        if callback is not None:callback(index+1,steps)
    return state


def velocity_mse(prediction,target):
    if prediction.device.type=='cuda':
        from mirage.runtime.flow_triton import mse
        return mse(prediction.contiguous(),target.contiguous())
    return (prediction.float()-target).square().mean()


@torch.inference_mode()
def euler(model,noise,text,mask,*,steps=40,guidance=1.,null_text=None,null_mask=None,callback=None):
    from mirage.runtime.text_cache import prepare
    text,mask=prepare(model,text,mask)
    if guidance!=1:null_text,null_mask=prepare(model,null_text,null_mask)
    state=noise.float().clone();dt=-1./steps
    for index in range(steps):
        t=torch.full((state.shape[0],),1-index/steps,device=state.device,dtype=torch.float32)
        cond=velocity(model,state,t,text,mask)
        if guidance!=1:
            uncond=velocity(model,state,t,null_text,null_mask,is_null=torch.ones(state.shape[0],device=state.device,dtype=torch.bool))
            cond=uncond+guidance*(cond-uncond)
        state=state+dt*cond.float()
        if callback is not None:callback(index+1,steps)
    return state
