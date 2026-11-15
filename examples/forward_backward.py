"""Tiny CPU usage example, not a trained generator or a CUDA kernel test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mirage import MIRAGEModel, ModelConfig
from mirage.runtime.flow import training_inputs, velocity_mse

torch.manual_seed(42)
config = ModelConfig(width=64, ffn=128, text_dim=32, head_dim=16,
    text_q_heads=4, text_kv_heads=4, self_q_heads=4, self_kv_heads=4,
    num_blocks=2, time_dim=16, time_hidden=32, fourier_dim=16,
    text_adapter_heads=4, text_adapter_ffn=64, text_register_tokens=2)
model = MIRAGEModel(config).train()
model.gradient_checkpointing = False
clean = torch.randn(2, 128, 2, 3)
tokens = torch.randn(2, 7, 32, requires_grad=True)
mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
noisy, timestep, target = training_inputs(clean)
prediction = model(noisy, timestep, tokens, mask)
loss = velocity_mse(prediction, target)
loss.backward()
assert prediction.shape == clean.shape
assert torch.isfinite(loss) and tokens.grad is not None
assert torch.isfinite(tokens.grad).all()
assert torch.count_nonzero(tokens.grad[~mask]) == 0
assert model.input.weight.grad is not None and torch.isfinite(model.input.weight.grad).all()
print(f'Tiny CPU forward/backward OK: shape={tuple(prediction.shape)}, loss={loss.item():.6f}')

# Inference reuses encoded text; random weights do not produce meaningful images.
model.eval()
with torch.inference_mode():
    cache = model.build_text_cache(tokens.detach(), mask)
    cached = model(noisy, timestep, cache, mask)
    uncached = model(noisy, timestep, tokens.detach(), mask)
assert torch.allclose(cached, uncached, atol=1e-6, rtol=1e-5)
print('Tiny CPU text-cache parity OK')
