# MIRAGE

[English](README.md) | [简体中文](README.zh-CN.md)

[Model weights on Hugging Face](https://huggingface.co/OQ-J/MIRAGE)

Recommended inference settings: Euler sampling, 40 steps, CFG=4.

**The model currently available on Hugging Face produces unsatisfactory results. See the [technical report](docs/TECHNICAL_DESIGN.en.md) for details.**

MIRAGE-1.65B — Tag-conditioned Rectified Flow image generation with Triton training and inference kernels.

[Technical design and engineering notes](docs/TECHNICAL_DESIGN.en.md)


A code-only snapshot of a text-conditioned Rectified Flow image model and its
Triton training/inference implementation. 

## Model

- 20 image Self-Attention + text Cross-Attention + image SwiGLU blocks.
- Image width 2048, FFN width 8192; 16 attention heads with head dimension 128.
- Full 1024-dimensional text token states, two TextAdapter blocks, eight prefix registers.
- Per-branch AttnRes; /16 packed latents with 128 channels.
- Direct velocity Rectified Flow: `t=0` is data, `t=1` is noise.

## Included implementation

`mirage/architecture/model.py` defines the architecture. Matching Python wrappers and
Triton kernels are kept together: attention and gradients, RMS/AdaRMS, RoPE/QKV,
SwiGLU, gated packed attention, AttnRes/history-gradient merge, residual writes,
RF interpolation/MSE, and gradient/update kernels.

Inference additionally skips backward-only buffers and provides independent
image/text backend choices. The optional Muon/AdamW integration is in
`muon.py`, `muon_fast.py`, and `optim.py`. 

## Install and inspect

First install the CUDA-enabled PyTorch build appropriate for the machine, with
its matching Triton. GPU inference was tested on Linux RTX 3090 / SM86,
PyTorch 2.8.0+cu128 and Triton 3.4.0. The original training optimizer used a
newer training environment; its compiled GPU update is not newly validated here.

```bash
python -m pip install -e ".[inference]"
python infer.py --help
python examples/forward_backward.py
```

The example uses a tiny random CPU model to exercise tensor shapes and gradients;
it does not generate a trained image or benchmark production kernels.

## Inference with your own compatible weights

```bash
python infer.py --checkpoint models/model.pt \
  --qwen-model models/Qwen3-Embedding-0.6B \
  --vae-path models/FLUX.2-klein-4B \
  --prompt "sfw, 1girl, blue_hair, full_body" \
  --width 512 --height 512 --steps 40 --cfg 4 --num-images 1
```

The VAE directory contains its `vae/` subdirectory; the FLUX denoising model is
not needed. Qwen, VAE, and trained model weights must be supplied separately.
The inference reader uses `weights_only=True`; tensor-and-metadata exports are
supported, not arbitrary Python-pickled training objects.

Default GPU combination: Triton fusions + PyTorch Flash image attention + custom
Triton text attention/Gate. `--image-attention` and `--text-attention` also accept
`fa4` with the optional FA4 dependency installed. CPU/PyTorch fallback:

```bash
python infer.py --backend torch --image-attention sdpa --text-attention sdpa ...
```

## Measurements and scope

Historical 3090 Euler40/CFG4 sampling: 512², 18.34s → 4.86s; 768×1024,
51.70s → 15.29s, from the PyTorch path to the mixed fused path. The original
Triton path already took 4.98s and 15.69s: most of the gain comes from enabling
existing fusions, not the small new inference-only changes.

[Detailed measurements](docs/TECHNICAL_DESIGN.en.md) separate attention cores from complete
sampling, include the memory metric, and preserve the limits of the comparison.



See [kernel map](docs/KERNEL_MAP.md), and
[algorithm/dependency references](ACKNOWLEDGEMENTS.md).

## License

The optional `.[optim]` extra requires Torch>=2.12 and may upgrade an existing
environment. It is not required for inference; keep Torch, CUDA and Triton compatible.

This repository is licensed under the MIT License.
External dependencies and external model weights retain their own terms.


[Directory](docs/DIRECTORY_GUIDE.en.md)


[Training and inference benchmarks](docs/TECHNICAL_DESIGN.en.md)
