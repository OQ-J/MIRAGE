"""2048-wide /16 image Self-Attention + text Cross-Attention, per-block AttnRes."""
from dataclasses import asdict, dataclass
import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from mirage.attention.attention import gated_text_attention
from mirage.ops.fused_ops import swiglu,rms,ada_rms,head_norm_rope,qkv_norm_rope,text_qkv,use_triton
from mirage.ops.gate import head_gate
from mirage.architecture.packed_text import TextLayout,pack_text,head_norm_packed,attention as packed_attention

@dataclass
class ModelConfig:
    width: int = 2048
    ffn: int = 8192
    text_dim: int = 1024
    head_dim: int = 128
    text_q_heads: int = 16
    text_kv_heads: int = 16
    self_q_heads: int = 16
    self_kv_heads: int = 16
    num_blocks: int = 20
    time_dim: int = 256
    time_hidden: int = 1024
    fourier_dim: int = 256
    time_scale: float = 1000.
    eps: float = 1e-6
    rope_theta: float = 10000.
    attnres_enabled: bool = True
    text_output_std: float = .001
    text_gate_init: float = .5
    text_adapter_blocks: int = 2
    text_adapter_heads: int = 16
    text_adapter_ffn: int = 4096
    text_register_tokens: int = 8

    def __post_init__(self):
        if isinstance(self.num_blocks, bool) or not isinstance(self.num_blocks, int) or self.num_blocks < 1:
            raise ValueError('num_blocks must be a positive integer')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        return cls(**data)


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return rms(x,self.weight,self.eps)


class HeadNorm(nn.Module):
    def __init__(self, heads, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()[None, :, None]


def rope2d(q, k, height, width, theta):
    half = q.shape[-1] // 2
    inv = theta ** (-torch.arange(0, half, 2, device=q.device, dtype=torch.float32) / half)
    y, x = torch.meshgrid(torch.arange(height, device=q.device), torch.arange(width, device=q.device), indexing='ij')
    angles = torch.cat((y.flatten()[:, None] * inv, x.flatten()[:, None] * inv), -1)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    def apply(a):
        even, odd = a.float()[..., 0::2], a.float()[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
    return apply(q), apply(k)


def _make_rope_tables(height,width,dim,theta,device):
    half=dim//2
    inv=theta**(-torch.arange(0,half,2,device=device,dtype=torch.float32)/half)
    y,x=torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing='ij')
    angles=torch.cat((y.flatten()[:,None]*inv,x.flatten()[:,None]*inv),-1)
    return angles.cos(),angles.sin()


def rope_tables(height,width,dim,theta,device):
    from mirage.runtime.constant_cache import get
    return get(('rope',height,width,dim,theta,str(device)),lambda:_make_rope_tables(height,width,dim,theta,device),device)


def time_frequency(dim,device):
    from mirage.runtime.constant_cache import get
    return get(('time',dim,str(device)),lambda:torch.exp(-math.log(10000.)*torch.arange(dim//2,device=device).float()/(dim//2)),device)


class TimeEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dim, self.scale = c.fourier_dim, c.time_scale
        self.mlp = nn.Sequential(nn.Linear(c.fourier_dim, c.time_hidden), nn.SiLU(), nn.Linear(c.time_hidden, c.time_dim))

    def forward(self, t):
        frequency = time_frequency(self.dim,t.device)
        angles = t.float()[:, None] * self.scale * frequency[None]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), -1).to(self.mlp[0].weight.dtype))


class TextAdapterBlock(nn.Module):
    """Bidirectional text-only Pre-RMSNorm Transformer, always at text width."""
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = c.text_adapter_heads, c.text_dim // c.text_adapter_heads
        self.attn_norm = RMSNorm(c.text_dim, c.eps)
        self.qkv = nn.Linear(c.text_dim, 3*c.text_dim, bias=False)
        self.q_norm = HeadNorm(self.heads, self.dim, c.eps)
        self.k_norm = HeadNorm(self.heads, self.dim, c.eps)
        self.attn_out = nn.Linear(c.text_dim, c.text_dim, bias=False)
        self.ffn_norm = RMSNorm(c.text_dim, c.eps)
        self.ffn_in = nn.Linear(c.text_dim, 2*c.text_adapter_ffn, bias=False)
        self.ffn_out = nn.Linear(c.text_adapter_ffn, c.text_dim, bias=False)
        nn.init.normal_(self.attn_out.weight, std=c.text_output_std)
        nn.init.normal_(self.ffn_out.weight, std=c.text_output_std)

    def forward(self, text, mask):
        total,width=text.shape
        q,kv=self.qkv(self.attn_norm(text)).split((width,2*width),-1)
        q,k,v=text_qkv(q.reshape(total,self.heads,self.dim),kv,self.q_norm.weight,self.k_norm.weight,self.q_norm.eps)
        y=packed_attention(q,k,v,mask.cu,mask.cu,mask.maximum,mask.maximum,mask.lengths,mask.lengths)
        text=text+self.attn_out(y.flatten(-2))
        return text+self.ffn_out(swiglu(self.ffn_in(self.ffn_norm(text))))


class TextAdapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.ModuleList(TextAdapterBlock(c) for _ in range(c.text_adapter_blocks))
        self.register_tokens = nn.Parameter(torch.empty(c.text_register_tokens,c.text_dim))
        nn.init.normal_(self.register_tokens,std=.02)

    def forward(self, text, mask, checkpointing=False):
        for block in self.blocks:
            text = checkpoint(block,text,mask,use_reentrant=False) if checkpointing else block(text,mask)
        return text


class TextCrossAttention(nn.Module):
    def _load_from_state_dict(self,state_dict,prefix,*args,**kwargs):
        if prefix+'k.weight' in state_dict:
            state_dict[prefix+'kv.weight']=torch.cat((state_dict.pop(prefix+'k.weight'),state_dict.pop(prefix+'v.weight')),0)
        super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)

    def __init__(self, c):
        super().__init__()
        self.q_heads, self.kv_heads, self.dim = c.text_q_heads, c.text_kv_heads, c.head_dim
        self.register_count = c.text_register_tokens
        self.q = nn.Linear(c.width, c.text_q_heads * c.head_dim, bias=False)
        self.kv = nn.Linear(c.text_dim, 2*c.text_kv_heads*c.head_dim, bias=False)
        self.out = nn.Linear(c.text_q_heads * c.head_dim, c.width, bias=False)
        self.q_norm = HeadNorm(c.text_q_heads, c.head_dim, c.eps)
        self.k_norm = HeadNorm(c.text_kv_heads, c.head_dim, c.eps)
        self.gate = nn.Linear(c.width, c.text_q_heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(c.text_gate_init/(1-c.text_gate_init)))
        self.capture = False
        self.statistics = None
        self.punctuation_mask = None
        nn.init.normal_(self.out.weight, std=c.text_output_std)

    def encode_kv(self,text):
        k,v=self.kv(text).chunk(2,-1)
        return rms(k.reshape(-1,self.kv_heads,self.dim),self.k_norm.weight,self.k_norm.eps),v.reshape(-1,self.kv_heads,self.dim).contiguous()

    def forward(self, x, text, mask,kv_cache=None):
        b,n,_=x.shape
        if kv_cache is None:
            query,k,v=text_qkv(self.q(x).reshape(b*n,self.q_heads,self.dim),self.kv(text),
                              self.q_norm.weight,self.k_norm.weight,self.q_norm.eps)
        else:
            k,v=kv_cache
            query=rms(self.q(x).reshape(b*n,self.q_heads,self.dim),self.q_norm.weight,self.q_norm.eps)
        cuq=torch.arange(b+1,device=x.device,dtype=torch.int32)*n
        gate_logits=self.gate(x)
        if use_triton(x.device) and os.environ.get('MIRAGE_TEXT_ATTN','auto') not in ('flash','sdpa','fa4'):
            from mirage.attention.varlen_triton import gated_varlen_attention
            y=gated_varlen_attention(query,k,v,cuq,mask.cu,n,mask.maximum,gate_logits.reshape(b*n,self.q_heads),(n,)*b,mask.lengths)
        else:
            y=packed_attention(query,k,v,cuq,mask.cu,n,mask.maximum,(n,)*b,mask.lengths)
            y=head_gate(y,gate_logits.reshape(b*n,self.q_heads))
        output=self.out(y.reshape(b,n,self.q_heads*self.dim))
        if self.capture:
            with torch.no_grad():
                q=query.reshape(b,n,self.q_heads,self.dim).transpose(1,2)
                gate=gate_logits.float().sigmoid()
                register_sum=punct_sum=count=0.
                for i in range(b):
                    keys=k[mask.offsets[i]:mask.offsets[i+1]].transpose(0,1).repeat_interleave(self.q_heads//self.kv_heads,0)
                    for chunk in q[i].split(32,dim=1):
                        scores=(chunk.float()@keys.float().transpose(-1,-2))/math.sqrt(self.dim)
                        probs=scores.softmax(-1);count+=probs.shape[0]*probs.shape[1]
                        register_sum+=float(probs[...,:self.register_count].sum())
                        if self.punctuation_mask is not None:
                            punct_sum+=float((probs*self.punctuation_mask[i,:keys.shape[1]][None,None]).sum())
                self.statistics=dict(output_rms=float(output.float().square().mean().sqrt()),
                    gate_mean=float(gate.mean()),gate_min=float(gate.min()),gate_max=float(gate.max()),
                    gate_per_head=gate.mean((0,1)).tolist(),register_mass=register_sum/count,
                    content_mass=1-register_sum/count,
                    punctuation_mass=punct_sum/count if self.punctuation_mask is not None else None)
        return output


class ImageAttention(nn.Module):
    def _load_from_state_dict(self,state_dict,prefix,*args,**kwargs):
        if prefix+'q.weight' in state_dict:
            state_dict[prefix+'qkv.weight']=torch.cat((state_dict.pop(prefix+'q.weight'),state_dict.pop(prefix+'kv.weight')),0)
        super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)

    def __init__(self, c):
        super().__init__()
        self.q_heads, self.kv_heads, self.dim = c.self_q_heads, c.self_kv_heads, c.head_dim
        self.theta = c.rope_theta
        self.qkv = nn.Linear(c.width,(c.self_q_heads+2*c.self_kv_heads)*c.head_dim,bias=False)
        self.out = nn.Linear(c.self_q_heads*c.head_dim, c.width, bias=False)
        self.q_norm = HeadNorm(c.self_q_heads, c.head_dim, c.eps)
        self.k_norm = HeadNorm(c.self_kv_heads, c.head_dim, c.eps)

    def forward(self, x, height, width, image_mask,rope=None):
        b, n, _ = x.shape
        if rope is None:rope=rope_tables(height,width,self.dim,self.theta,x.device)
        q,k,v=qkv_norm_rope(self.qkv(x),self.q_norm.weight,self.k_norm.weight,*rope,self.q_norm.eps)
        y = gated_text_attention(q.to(x.dtype),k.to(x.dtype),v,image_mask)
        return self.out(y.transpose(1,2).flatten(-2))


class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.input = nn.Linear(c.width,2*c.ffn,bias=False)
        self.output = nn.Linear(c.ffn,c.width,bias=False)

    def forward(self,x):
        return self.output(swiglu(self.input(x)))


class DepthRead(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.query=nn.Parameter(torch.zeros(c.width))
        self.norm=RMSNorm(c.width,c.eps)

    def forward(self,history,partial,cached=None,final_weight=None):
        values=history if partial is None else history+(partial,)
        if use_triton(values[0].device):
            if final_weight is not None:
                from mirage.ops.depth_ada_triton import depth_final
                return depth_final(values,self.query,self.norm.weight,final_weight,self.norm.eps,cached=cached)
            from mirage.ops.depth_triton import depth_read
            return depth_read(values,self.query,self.norm.weight,self.norm.eps,cached=cached)
        logits=torch.stack([(self.norm(value.float())*self.query.float()).sum(-1) for value in values])
        weights=logits.softmax(0)
        result=sum(weight[...,None]*value.float() for weight,value in zip(weights.unbind(0),values))
        result=result.to(values[0].dtype)
        return rms(result,final_weight,self.norm.eps) if final_weight is not None else result


class TransformerBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.attnres=c.attnres_enabled
        self.history_grad_merge=True
        self.time=nn.Linear(c.time_dim,2*c.width)
        nn.init.zeros_(self.time.weight);nn.init.zeros_(self.time.bias)
        self.layers=nn.ModuleList([ImageAttention(c),TextCrossAttention(c),SwiGLU(c)])
        self.norms=nn.ModuleList(RMSNorm(c.width,c.eps) for _ in self.layers)
        self.reads=nn.ModuleList(DepthRead(c) for _ in self.layers) if self.attnres else nn.ModuleList()

    def forward(self,history,partial,time,text,text_mask,image_mask,height,width,rope=None,checkpoint_ffn=False,history_inv=None,text_kv=None):
        scale,shift=self.time(time).float().chunk(2,-1)
        merged=None
        if self.attnres and self.training and torch.is_grad_enabled() and self.history_grad_merge and not torch.compiler.is_compiling():
            from mirage.ops.history_merge import begin
            merged=begin(history)
        for index,(layer,norm) in enumerate(zip(self.layers,self.norms)):
            if merged is not None:
                from mirage.ops.history_merge import read as merged_read
                read=self.reads[index]
                h=merged_read(merged,partial,read.query,read.norm.weight,norm.weight,scale,shift,norm.eps,history_inv)
            elif self.attnres and use_triton(history[0].device):
                from mirage.ops.depth_ada_triton import depth_ada
                read=self.reads[index]
                values=history if partial is None else history+(partial,)
                h=depth_ada(values,read.query,read.norm.weight,norm.weight,scale,shift,norm.eps,cached=history_inv)
            else:
                h=self.reads[index](history,partial) if self.attnres else partial
                h=ada_rms(h,norm.weight,scale,shift,norm.eps)
            call=getattr(layer,'_segment_forward',None) if self.training else None
            if call is None:call=layer
            if isinstance(layer,ImageAttention):delta=call(h,height,width,image_mask,rope)
            elif isinstance(layer,TextCrossAttention):delta=layer(h,text,text_mask,kv_cache=text_kv)
            else:delta=checkpoint(call,h,use_reentrant=False) if checkpoint_ffn else call(h)
            from mirage.ops.write_ops import residual_write
            partial=residual_write(partial,delta,image_mask)
        return partial


class MIRAGEModel(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        c=self.config=config or ModelConfig()
        self.input=nn.Linear(128,c.width,bias=False)
        self.output=nn.Linear(c.width,128,bias=False)
        self.time_embedding=TimeEmbedding(c)
        self.text_adapter=TextAdapter(c)
        self.blocks=nn.ModuleList(TransformerBlock(c) for _ in range(c.num_blocks))
        self.final_read=DepthRead(c) if c.attnres_enabled else None
        self.final_norm=RMSNorm(c.width,c.eps)
        self.gradient_checkpointing=True
        self.checkpoint_blocks=len(self.blocks)
        self.checkpoint_mode='block'
        self.text_adapter_checkpointing=False
        self.final_read_checkpointing=False
        nn.init.normal_(self.output.weight,std=.01)

    def _encode_text(self,text,mask,b,device,dtype):
        if not isinstance(mask,TextLayout):
            text,mask=pack_text(text,mask,self.config.text_register_tokens,device=device,dtype=dtype)
        else:
            text=text.to(device=device,dtype=dtype)
            mask=mask._replace(cu=mask.cu.to(device),gather=mask.gather.to(device))
        from mirage.ops.write_ops import insert_registers
        text=insert_registers(text,self.text_adapter.register_tokens,mask.gather,b)
        return self.text_adapter(text,mask,checkpointing=self.training and self.gradient_checkpointing and self.text_adapter_checkpointing),mask

    @torch.no_grad()
    def build_text_cache(self,text,mask):
        from mirage.runtime.text_cache import TextCache,signature
        if self.training:raise ValueError('Text caches are inference-only; call model.eval() first')
        if isinstance(text,TextCache):
            if text.owner!=id(self) or text.signature!=signature(self):raise ValueError('Text cache belongs to another model or stale weights')
            return text
        b=len(mask.lengths) if isinstance(mask,TextLayout) else text.shape[0]
        tokens,layout=self._encode_text(text,mask,b,self.input.weight.device,self.input.weight.dtype)
        layers=tuple(branch.encode_kv(tokens) for branch in self.text_attentions())
        return TextCache(id(self),signature(self),tokens,layout,layers)

    def forward(self,latent,t,text,mask,is_null=None,image_mask=None):
        packed=latent
        b,_,height,width=packed.shape
        x=self.input(packed.flatten(2).transpose(1,2))
        from mirage.runtime.text_cache import TextCache,signature
        text_kvs=None
        if isinstance(text,TextCache):
            if self.training or torch.is_grad_enabled():raise ValueError('Text caches cannot be used in training or gradient-enabled forwards')
            if text.owner!=id(self) or text.signature!=signature(self):raise ValueError('Text cache belongs to another model or stale weights')
            if b!=len(text.layout.lengths):raise ValueError('Text cache batch size does not match the image batch')
            if text.tokens.device!=x.device or text.tokens.dtype!=x.dtype:raise ValueError('Text cache device/dtype does not match the model')
            text_kvs=text.layers;mask=text.layout;text=text.tokens
        else:
            # Text remains fully differentiable on every training forward.
            text,mask=self._encode_text(text,mask,b,x.device,x.dtype)
        if image_mask is not None:x=x.masked_fill(~image_mask[...,None],0)
        time=self.time_embedding(t)
        rope=rope_tables(height,width,self.config.head_dim,self.config.rope_theta,x.device)
        history=(x,);partial=None if self.config.attnres_enabled else x
        history_inv=None
        if use_triton(x.device) and self.config.attnres_enabled:
            from mirage.ops.depth_triton import history_rms
            history_inv=history_rms(x,self.config.eps)[None]
        checkpoint_start=len(self.blocks)-self.checkpoint_blocks
        for block_index,block in enumerate(self.blocks):
            if block_index and self.config.attnres_enabled:
                history=history+(partial,)
                if history_inv is not None:
                    history_inv=torch.cat((history_inv,history_rms(partial,self.config.eps)[None]),0)
                partial=None
            use_checkpoint=self.training and self.gradient_checkpointing and block_index>=checkpoint_start
            args=(history,partial,time,text,mask,image_mask,height,width,rope,use_checkpoint and self.checkpoint_mode=='ffn',history_inv,
                  None if text_kvs is None else text_kvs[block_index])
            partial=checkpoint(block,*args,use_reentrant=False) if use_checkpoint and self.checkpoint_mode=='block' else block(*args)
        if self.final_read is not None:
            x=checkpoint(self.final_read,history,partial,history_inv,self.final_norm.weight,use_reentrant=False) if self.training and self.gradient_checkpointing and self.final_read_checkpointing else self.final_read(history,partial,history_inv,self.final_norm.weight)
        else:x=self.final_norm(partial)
        y=self.output(x).transpose(1,2).reshape(b,128,height,width)
        return y

    def text_attentions(self):
        return [m for m in self.modules() if isinstance(m,TextCrossAttention)]
