# MIRAGE Image Generation Model Technical Overview

This document introduces the model architecture, training and inference design, Triton execution optimizations, and current issues.
It is intended for readers who want to understand the code or continue development. Parameters, formulas, and execution behavior follow the implementation in this repository.

This is an anime image generation model with approximately **1.651B parameters**, conditioned on tags. It uses pretrained Qwen text hidden states and
FLUX.2 VAE representations to predict Rectified Flow velocities in latent space.

## 1 Usage Requirements at a Glance

| Item | Requirement and current status |
|---|---|
| Prompts | Use comma-separated tags |
| First tag | Explicitly include `sfw` or `nsfw`; do not omit it |
| Rating | The project's data-rating rules classify NSFW content relatively strictly |
| Natural language | No systematic evaluation has been conducted |
| Registers | The current trained weights require **8 learnable text-side registers**, automatically inserted at the beginning of each sequence |
| Resolution | Up to 768×1024 / 1024×768, or dimensions within the same pixel budget with both sides divisible by 32 |
| Sampling | Euler, 40 steps, and CFG 4 by default |
| Current limitations | Arms, legs, fingers, and full-body poses are prone to errors; complex compositions are also unstable |

Recommended input format:

```text
sfw, 1girl, blue_hair, full_body, standing, white_shirt
```

`sfw/nsfw` is a content-rating tag included in text encoding, not a moderation switch executed by the program.
The current entry point does not automatically add or validate this prefix. Users should provide it explicitly; the default example prompt does not replace this requirement.
The training data uses relatively strict NSFW ratings. This does not mean that the model guarantees filtering of particular outputs, nor should inference-time tags be treated as a reliable classifier.

## 2 Model Data Flow

```text
Comma-separated tags
  → Frozen Qwen3-Embedding-0.6B
  → All valid token hidden states [B,L,1024]
  → Compact packing + 8 learnable registers prepended to each sample
  → 2 bidirectional TextAdapter layers, width 1024
  → Text K/V inputs for the 20 backbone blocks

Training images
  → Frozen FLUX.2 VAE
  → Official latent packing and normalization
  → [B,128,H/16,W/16]
  → RF noise addition
  → Flatten into image tokens + Linear(128,2048)
  → 20 HybridBlocks with AttnRes history reads
  → Final AttnRes read + RMSNorm + Linear(2048,128)
  → Predicted latent velocity with the same shape
```

`[B,L,1024]` in the diagram is a logical representation for readability. The backbone executes with a compact layout of `[total_valid_tokens,1024]`.
Cumulative offsets record each sample's boundaries so that text from different samples does not attend to one another.
Qwen encoding temporarily pads samples for batching and excludes padding through a validity mask.

### 2.1 Main Configuration

| Configuration | Default |
|---|---:|
| Trainable parameters | 1,650,703,936 |
| Backbone depth `num_blocks` | 20 |
| Image width | 2048 |
| Backbone FFN hidden width | 8192, SwiGLU |
| Image self-attention | 16 Q heads, 16 KV heads, head dimension 128 |
| Image–text cross-attention | 16 Q heads, 16 KV heads, head dimension 128 |
| Qwen input feature width | 1024 |
| TextAdapter | 2 layers, 16 heads, head dimension 64, FFN 4096 |
| Text registers | 8, each with width 1024 |
| Latents | 128 channels, effective spatial stride 16 |

### 2.2 Supported Resolutions

The maximum supported image area is `768 × 1024 = 786432` pixels.
Other aspect ratios can be used within the same pixel budget,
but both width and height must be positive multiples of 32:

```text
W % 32 == 0
H % 32 == 0
W * H <= 786432
```

There are too few training samples at extreme resolutions, and their results are poor.

With an effective spatial stride of 16, the image-token count is `(W/16) × (H/16)`. The maximum area corresponds to 3072 image tokens.
Images with the same area have the same token count, but different 2D RoPE coordinates and composition distributions.


## 3 Image Self-Attention Plus Text Cross-Attention

The design references ANIMA's implementation.
Each backbone block contains, in order:

1. Image self-attention: exchanges composition, geometry, and visual information among image positions.
2. Text cross-attention: uses image tokens as queries to read text-token keys and values.
3. Image SwiGLU FFN: applies nonlinear transformations to image-channel features.

Each branch has its own normalization and AttnRes read. Image self-attention uses 2D RoPE.
Cross-attention applies a sigmoid gate independently for each image position and head before the output projection.

Here, “full self-attention” means concatenating image and text tokens for joint self-attention, referencing MOSS's implementation.
Let N be the number of image tokens and L the number of text tokens.

| Approach | Pairwise attention interactions per backbone layer |
|---|---|
| Joint self-attention after concatenation | `(N+L)^2 = N^2 + 2NL + L^2` |
| Image self-attention + image queries reading text | `N^2 + NL` |

The latter does not compute text queries reading all image tokens at every backbone layer, nor repeatedly compute text self-attention within the backbone.
The backbone FFN also processes only image tokens. Bidirectional text interaction is concentrated in the first two, narrower TextAdapter layers.
This makes long-text training costs more manageable and allows the complete text states and each layer's K/V to be cached across inference steps.

The project chose this design to balance the training budget; full self-attention is more expensive.

## 4 Text Features from Qwen3-Embedding-0.6B

The project reads all valid tokens from Qwen3-Embedding-0.6B's `last_hidden_state`, referencing ANIMA.

TextAdapter uses bidirectional pre-RMSNorm attention and SwiGLU. Its output projections are initialized with a small standard deviation,
so that newly learned text processing starts with relatively modest changes.

### 4.1 Tag Format and Natural-Language Support

Data preparation uses tag lists, removes quotation marks, shuffles comma-separated tags, and then re-encodes them. Words belonging to the same tag still need underscores.

Use the same tag format for inference, with `sfw` or `nsfw` as the first tag.
**Long natural-language sentences have not been systematically evaluated.**

## 5 Why Gate and Register Tokens Are Used

During training, text attention concentrates excessively on the first token and on positions such as commas and quotation marks that have little relevance to the tags. Gates and register tokens are therefore used.

### 5.1 Gates Control the Amount Written

For image position j and attention head h, text reading is approximately:

```text
a[j,h] = Attention(normalize(Q[j,h]), normalize(K), V)
g[j,h] = sigmoid(W_gate * X[j] + b_gate)
y[j,h] = g[j,h] * a[j,h]
output[j] = W_out * concatenate_heads(y[j])
```

Gates allow different image positions and heads to control the strength of information written by the text branch.


### 5.2 Registers Provide Additional Learnable Positions

Standard softmax weights sum to 1. Even when a specific word is not needed, the model must assign attention to some positions.
Adding 8 learnable registers provides additional working and information-aggregation positions, reducing the need for semantic tokens to take on these roles.

Registers are placed **before TextAdapter, at the beginning of each text sequence**. After two bidirectional updates, they become backbone K/V together with the full text.

### 5.3 All 8 Registers Must Be Retained

The current weights were trained with 8 registers. They are stored in the model parameter `text_adapter.register_tokens`
and inserted automatically by the model.

## 6 AttnRes

An attention-based residual mechanism is used to improve residual connections.

## 7 RF Target and Precision

The repository currently uses direct velocity prediction:

```text
t = sample_uniform_or_logit_normal_per_image()
noise = randn_like(z)
x_t = (1-t) * z + t * noise
target = noise - z
loss = mean((prediction.float() - target.float())**2)
```

**Here, t=0 is the clean latent and t=1 is pure noise.** Inference integrates from 1 to 0, so Euler's dt is negative.
Training uses uniform timestep sampling.

FLUX.2 latents are normalized using the official running mean and variance. Decoding must restore them through the corresponding codec.
The backbone is primarily BF16. Loss, integration states, FP32 master weights, and other precision boundaries are retained; the VAE interface uses FP32.


### 7.1 Resolution Curriculum and Steps per Stage

This table describes the training run associated with the released weights, through **540000 optimizer updates**.
`step` means an actual parameter update.


| Stage | global_step range | Updates in this stage | Effective batch | Usual micro-batch | Usual accumulation steps | Learning rate |
|---|---|---:|---:|---:|---:|---:|
| 256-area stage | 1–106000 | 106000 | 256 | 32 | 8 | 1e-4 |
| 512-area stage | 106001–212000 | 106000 | 64 | 16 | 4 | 1e-4 |
| Native resolution, up to the area of 768×1024 | 212001–540000 | 328000 | 16 | 4 | 4 | 5e-5 |
| Total | 1–540000 | **540000** | Varies by stage | — | — | — |

The “256/512-area stages” preserve the original aspect ratio,
prepare latents at an area close to 256² or 512², and align the actual encoded width and height to 32.

### 7.2 Historical Optimizer and Timestep-Sampling Changes

This training run did not use Muon from step 0:

| Update range | Optimizer | Timestep sampling |
|---|---|---|
| 1–126000 | Muon + AdamW | LogitNormal |
| 126001–540000 | Muon + AdamW | Uniform |

LogitNormal was used early to accelerate convergence. Uniform sampling was used later to strengthen inference at pure noise and near-noiseless timesteps.

## 8 Using Muon Plus AdamW

The implementation references MOSS.

| Parameter type | Treatment in this implementation |
|---|---|
| Eligible two-dimensional Linear weights | Muon |
| Fused QKV, KV, and SwiGLU weights | Split into logical matrices, then updated separately with Muon |
| Biases, norms, registers, and excluded control/input/output weights | AdamW, with weight decay assigned by the parameter rules |
| Model copy used for training | Primarily BF16 |
| Master weights, momentum, and Adam states | Retained in FP32 |

## 9 Why Triton Was Written

The reason is practical training cost: tag lengths differ across samples, padding wastes memory and computation, and some fixed-shape compilation paths recompile repeatedly when packed row counts change.
In addition, small operations such as norms, RoPE, gates, history reads, and precision conversions create many intermediate tensors.


| Optimization | Implementation | Cost reduced |
|---|---|---|
| Variable-length text | Compact valid-token storage with cumulative sequence boundaries | Padding activations and invalid computation |
| Compact task tables | Dispatch only blocks that actually exist in each sequence | Empty task blocks for short text |
| Runtime lengths | Do not specialize the main attention kernel for every exact length | Unnecessary length-specific recompilation |
| QKV/Norm/RoPE fusion | Perform adjacent layout and normalization operations in one kernel | Intermediate tensors and launches |
| Cross + Gate fusion | Apply gating when writing attention output, with corresponding backward logic | Additional feature reads and writes |
| AttnRes fusion | Connect history scoring/reading, normalization, and modulation; reuse history RMS | Repeated history reads and reductions |
| History-gradient merging | Accumulate history gradients cooperatively across reader branches | Repeated allocations and additions |
| RF and gradient processing | Fuse interpolation/MSE and gradient conversion/norm reduction | Temporary FP32 tensors and scheduling |
| Inference text cache | Compute TextAdapter and each layer's text K/V once | Repeated text computation across 40 steps |


## 10 Comparison with FA4 and PyTorch Flash


### 10.1 PRO6000 Training Measurements

RTX PRO 6000 Blackwell Server Edition, SM120; Torch 2.12.1+cu130.
The same checkpoint, native 1024×768, effective B16=micro4×4, 0 backbone checkpoints, fixed 256 valid text tokens plus 8 registers.
Timing includes packing/H2D/full forward and backward/Muon + AdamW updates.

| Approach | Seconds/update, lower is faster | Peak allocated GiB |
|---|---:|---:|
| MIRAGE default training execution | 2.7412 | 53.774 |
| FA4 for all attention | 2.8397 | 51.889 |
| Default attention + GEMM autotuning | 2.7350 | 53.773 |
| FA4 + GEMM autotuning | 2.8281 | 51.891 |

All-FA4 was approximately 3.59% slower in this test, but saved approximately 1.89 GiB of allocated memory.
The candidate backend's fixed-shape GEMM path compiled new shapes as variable-length text changed the row count, producing slow steps lasting tens of seconds.

### 10.2 RTX3090 Inference Measurements

RTX3090 24GB / SM86, Torch2.8.0+cu128, Triton3.4.0, FA4 4.0.0b38.
Euler40, CFG4, B1, 768×1024; only backbone sampling after warmup is timed:

| Approach | Seconds, lower is faster |
|---|---:|
| PyTorch automatic path with built-in Flash enabled | 51.697 |
| Forward pass using the original training Triton kernels | 15.691 |
| Mixed inference path | 15.288 |
| Triton fusions + FA4 core | 15.391 |

Most of the improvement comes from enabling the full set of fused operations.
In single-layer tests, PyTorch Flash was faster for image self-attention, while the custom fused kernel was faster for short-text Cross + Gate.

### 10.3 Single-Layer Attention Forward Pass

Times are in milliseconds, with 16 heads, head dimension 128, and BF16; projections and normalization are excluded.
Cross + Gate includes gating: the custom path fuses the gate, while the other paths use the same Triton gate afterward.

| Case | Custom Triton | PyTorch Flash | FA4 |
|---|---:|---:|---:|
| Self 1024 | 0.21065 | 0.18614 | 0.18944 |
| Self 3072 | 1.51082 | 1.14361 | 1.16575 |
| Cross 3072/64 | 0.04899 | 0.04725 | 0.06085 |
| Cross + Gate 3072/64 | 0.05258 | 0.08634 | 0.09919 |
| Cross 3072/256 | 0.12763 | 0.11555 | 0.12801 |
| Cross + Gate 3072/256 | 0.13806 | 0.15444 | 0.16815 |
| Cross 3072/1024 | 0.42754 | 0.39112 | 0.40540 |
| Cross + Gate 3072/1024 | 0.44611 | 0.42824 | 0.44480 |


### 10.4 Measurement Scope and Detailed Records

MIRAGE's default training execution on PRO6000 uses cuDNN for images, Triton for variable-length text, and other fused operations.
Each training configuration runs 12 updates, excludes the first 2, and reports the median of the remaining 10; shard disk reads, saving, previews, and diagnostics are excluded.
Memory figures are PyTorch peak allocated memory, not total process memory reported by nvidia-smi.

3090 inference timing includes only backbone sampling after warmup, excluding loading, text encoding, first compilation, VAE decoding, and image writing.
Single-layer attention uses the same Q/K/V, 16 heads, head dimension 128, and BF16.

Additional 512×512 sampling results: PyTorch automatic 18.337 seconds, original training Triton forward 4.976 seconds,
MIRAGE mixed inference 4.857 seconds, and Triton fusions + FA4 core 4.916 seconds; their peak allocated memory is approximately 3.248, 3.233, 3.233, and 3.233 GiB, respectively.
For 768×1024, peak allocated memory is approximately 3.561, 3.515, 3.515, and 3.515 GiB, respectively.


## 11 Current Image-Quality Issues and Further Training

Current generation quality is still unsatisfactory. Common issues include unstable arms, legs, fingers, joint connections, and complex full-body poses, as well as broken limbs in multi-person images. Single-style generation and multi-style blending are reasonably good.
Good main subjects or local textures do not mean that anatomical relationships have been learned. More sampling steps or higher CFG cannot reliably compensate for missing training knowledge.

**The current main hypothesis is that insufficient full-body images and pose coverage, together with insufficient total image exposure, may be important causes of the limb problems.**

Further training and optimization will continue by adding high-quality full-body, varied-pose, anatomically sound, and multi-person-interaction images to the dataset.

### 11.1 Style Instability Across Seeds with the Same Prompt

There is another clear usability issue: **with the same prompt and no additional style guidance, changing the random seed may produce noticeably different styles rather than a consistently similar visual style, possibly because the effective batch is too small.**

This is a current limitation in style controllability and output consistency.

## 12 Inference and Compatibility Requirements

Run from the repository root, with the model and external encoders placed in the models directory as described in the README:

```bash
python infer.py --prompt "sfw, 1girl, blue_hair, full_body, standing, white_shirt" --width 512 --height 512 --steps 40 --sampler euler --cfg 4
```

Without a seed, each run uses a random seed and prints its actual value. Set `--seed 42` explicitly for reproducible comparisons.
The default negative prompt is empty, and the CFG reference branch uses encoded empty text.

Make sure to:

- Use the matching Qwen3-Embedding-0.6B and FLUX.2 VAE.
- Retain all 8 text registers and their weights, and load the model configuration strictly.
- Use a tag list with sfw or nsfw as its first item.


## 13 Code Navigation and Release Scope

| Component | Location |
|---|---|
| Backbone and TextAdapter/Gate/Register | `mirage/architecture/model.py` |
| Packed text and offsets | `mirage/architecture/packed_text.py` |
| Attention dispatch and variable-length kernels | `mirage/attention/` |
| Norm, RoPE, AttnRes, and residual fusions | `mirage/ops/` |
| Muon/AdamW and gradient processing | `mirage/optimizers/` |
| RF, caches, and weight loading | `mirage/runtime/` |
| Qwen and VAE interfaces | `integrations/` |
| Inference entry point | `infer.py` |

The public directory provides the reusable model, training/inference kernels, and inference entry point.
