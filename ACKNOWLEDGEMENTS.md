# Algorithm and dependency references

The model combines existing Rectified Flow and Transformer ideas. The project-specific
work is the architecture integration and fused execution; it does not claim invention
of RMSNorm, RoPE, SwiGLU, softmax attention, Muon, or external model encoders/decoders.

- Muon: https://github.com/KellerJordan/Muon
- RMS-matched/scalable Muon reference: https://github.com/MoonshotAI/Moonlight
- FlashAttention and optional FA4: https://github.com/Dao-AILab/flash-attention
- PyTorch: https://github.com/pytorch/pytorch
- Triton: https://github.com/triton-lang/triton
- Qwen3-Embedding: https://github.com/QwenLM/Qwen3-Embedding
- FLUX.2 VAE integration uses the Diffusers AutoencoderKLFlux2 implementation:
  https://github.com/huggingface/diffusers

Dependencies, external weights, and their repositories are not vendored here.
The optional Muon integration is project-specific, not an official MOSS/Moonshot release.
