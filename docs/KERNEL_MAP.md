# Kernel and wrapper map

| Area | Kernel | Python integration |
|---|---|---|
| Image self attention | attention/image_attention_triton.py | attention/attention.py / architecture/model.py |
| Packed text/cross attention and gate | attention/varlen_triton.py | architecture/packed_text.py / architecture/model.py |
| Attention backward delta | attention/attention_delta_triton.py | image/varlen kernels |
| RMSNorm and AdaRMS | ops/norm_triton.py | ops/fused_ops.py |
| SwiGLU | ops/elementwise_triton.py | ops/fused_ops.py |
| Head norm and RoPE | ops/rope_triton.py | ops/fused_ops.py |
| QKV normalization/layout/RoPE | ops/qkv_triton.py | ops/fused_ops.py |
| Text Q/K normalization/layout | ops/text_qkv_triton.py | ops/fused_ops.py |
| AttnRes weighted history read | ops/depth_triton.py | architecture/model.py |
| AttnRes plus modulation/final norm | ops/depth_ada_triton.py | architecture/model.py |
| Shared history gradient accumulation | ops/history_merge_triton.py | ops/history_merge.py |
| Output head gate | ops/gate_triton.py | ops/gate.py |
| Residual/register write | ops/write_triton.py | ops/write_ops.py |
| RF interpolation and FP32 MSE | runtime/flow_triton.py | runtime/flow.py |
| Gradient cast/norm | optimizers/grad_norm_triton.py | optimizers/grad_norm.py / optimizers/optim.py |
| AdamW/master writeback | optimizers/adam_update_triton.py | optimizers/adam_update.py / optimizers/optim.py |

Training kernels keep their registered backward functions.
Inference-only shortcuts are guarded by inference mode and preserve required BF16 cast boundaries.

