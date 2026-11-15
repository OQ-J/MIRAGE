# Code Directory and Muon File Responsibilities

```text
mirage/
  __init__.py       # Unified model entry point
  architecture/    # Model architecture and packed layout of full text tokens
  attention/       # Self/Cross attention dispatch, variable-length kernels, and backward
  ops/             # Fused operations such as RMSNorm, RoPE, QKV, Gate, and AttnRes
  optimizers/      # Muon/AdamW, gradient norms, and weight updates
  runtime/         # RF, sampling, text/constant caches, and weight loading
```

Wrappers and their corresponding Triton implementations are kept in the same functional directory so that the PyTorch code and GPU kernels can be compared.
Training and inference share many kernels, so they are not forcibly duplicated into two sets of files.
