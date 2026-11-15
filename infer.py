"""MIRAGE inference. Run python infer.py --help for configurable options."""
import argparse
import gc
import math
import os
import secrets
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from mirage.runtime.checkpoint import read
from mirage.runtime.flow import euler, midpoint
from mirage.architecture.model import MIRAGEModel, ModelConfig


def trim(tokens,max_tokens=1024):
    return torch.cat((tokens[:max_tokens-1],tokens[-1:])) if len(tokens)>max_tokens else tokens

def parse_args():
    # Edit these defaults, then run: python infer.py
    default_prompt = 'sfw, 1girl, blue_hair, full_body'
    default_negative_prompt = ''
    default_steps = 40
    default_cfg = 4
    default_width = 512
    default_height = 512
    default_seed = None  # None generates a random seed each run.
    default_sampler = 'euler'
    default_num_images = 1
    default_output = Path(__file__).resolve().parent / 'output' / 'image.png'
    default_backend = 'triton'
    default_image_attention = 'flash'
    default_text_attention = 'triton'

    parser = argparse.ArgumentParser(description=__doc__)
    models_dir = Path(__file__).resolve().parent / 'models'
    parser.add_argument('--checkpoint', type=Path, default=models_dir / 'model.pt',
                        help='MIRAGE-compatible full or weights-only checkpoint, or checkpoint directory')
    parser.add_argument('--vae-path', type=Path, default=models_dir / 'FLUX.2-klein-4B',
                        help='FLUX.2 model directory containing the vae subdirectory')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--prompt', '--text', dest='prompt')
    source.add_argument('--text-embedding', type=Path,
                        help='Tensor [L,1024], or dictionary containing embedding and optional mask')
    parser.add_argument('--qwen-model', type=Path, default=models_dir / 'Qwen3-Embedding-0.6B',
                        help='Local Qwen3-Embedding-0.6B directory, required for --prompt')
    parser.add_argument('--negative-prompt', default=default_negative_prompt,
                        help='CFG reference prompt. Default uses empty text, as during training')
    parser.add_argument('--null-text-embedding', type=Path,
                        help='CFG reference embedding when using --text-embedding')
    parser.add_argument('--output', type=Path, default=default_output)
    parser.add_argument('--width', type=int, default=default_width)
    parser.add_argument('--height', type=int, default=default_height)
    parser.add_argument('--steps', type=int, default=default_steps)
    parser.add_argument('--cfg', '--guidance', dest='cfg', type=float, default=default_cfg)
    parser.add_argument('--sampler', choices=('euler', 'midpoint'), default=default_sampler)
    parser.add_argument('--seed', type=int, default=default_seed,
                        help='Omit for a random seed. The actual seed is printed')
    parser.add_argument('--num-images', type=int, default=default_num_images,
                        help='Sequential images with seed, seed+1, ...')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16',
                        help='Body precision. VAE and latent integration use FP32')
    parser.add_argument('--text-dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--text-cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--backend', choices=('torch', 'triton'), default=default_backend)
    parser.add_argument('--image-attention', choices=('auto','triton','flash','fa4','cudnn','sdpa'), default=default_image_attention)
    parser.add_argument('--text-attention', choices=('auto','triton','flash','fa4','sdpa'), default=default_text_attention)
    parser.add_argument('--inference-kernels', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--depth-warps', type=int, choices=(2,4,8), default=4)
    parser.add_argument('--latent-output', type=Path,
                        help='Optionally save generated normalized latents before VAE decode')
    args = parser.parse_args()
    if args.prompt is None and args.text_embedding is None:
        args.prompt = default_prompt
    if min(args.width, args.height) <= 0 or args.width % 32 or args.height % 32:
        parser.error('--width and --height must be positive multiples of 32')
    if args.steps <= 0 or args.num_images <= 0:
        parser.error('--steps and --num-images must be positive')
    if not math.isfinite(args.cfg) or args.cfg < 0:
        parser.error('--cfg must be finite and nonnegative')
    if args.prompt is not None and args.qwen_model is None:
        parser.error('--prompt requires --qwen-model')
    if args.text_embedding is not None:
        if args.negative_prompt:
            parser.error('With --text-embedding, supply --null-text-embedding instead of --negative-prompt')
        if args.cfg != 1 and args.null_text_embedding is None:
            parser.error('CFG with --text-embedding requires --null-text-embedding')
    elif args.null_text_embedding is not None:
        parser.error('--null-text-embedding is used only with --text-embedding')
    return args


def load_embedding(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    tokens = saved['embedding'] if isinstance(saved, dict) else saved
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.ndim != 2 or tokens.shape[1] != 1024 or len(tokens) == 0:
        raise ValueError(f'{path}: expected a nonempty [L,1024] embedding')
    if isinstance(saved, dict) and 'mask' in saved:
        mask = saved['mask'].reshape(-1).bool()
        if mask.numel() != len(tokens):
            raise ValueError(f'{path}: mask length does not match embedding')
        tokens = tokens[mask]
    if len(tokens) == 0 or not torch.isfinite(tokens).all():
        raise ValueError(f'{path}: embedding is empty or contains nonfinite values')
    return tokens.float()


def release_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def image_path(path, index, seed):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    unique = secrets.token_hex(6)
    suffix = path.suffix or '.png'
    return path.with_name(f'{path.stem}_{timestamp}_seed{seed}_{index + 1:03d}_{unique}{suffix}')


def main():
    args = parse_args()
    os.environ['MIRAGE_TORCH_ONLY'] = '1' if args.backend == 'torch' else '0'
    os.environ['MIRAGE_IMAGE_ATTN'] = args.image_attention
    os.environ['MIRAGE_TEXT_ATTN'] = args.text_attention
    os.environ['MIRAGE_INFERENCE_KERNELS'] = '1' if args.inference_kernels else '0'
    os.environ['MIRAGE_DEPTH_WARPS'] = str(args.depth_warps)
    print(f'Inference backend: {args.backend}', flush=True)
    print(f'Attention cores: image={args.image_attention}, text={args.text_attention}; '
          f'inference kernels={args.inference_kernels}, depth warps={args.depth_warps}', flush=True)
    if args.seed is None:
        args.seed = secrets.randbits(63)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from mirage.ops.fused_ops import TRITON_AVAILABLE
        if not TRITON_AVAILABLE:
            print('Triton is unavailable. Using PyTorch CUDA operators and SDPA.', flush=True)

    print('Preparing text...', flush=True)
    reference = None
    if args.prompt is not None:
        from integrations.qwen import TextEncoder
        encoder = TextEncoder(args.qwen_model, str(device), compute_dtype=args.text_dtype)
        prompts = [args.prompt] if args.cfg == 1 else [args.prompt, args.negative_prompt]
        embeddings = encoder.encode_ids(encoder.tokenize(prompts))
        tokens = embeddings[0]
        if args.cfg != 1:
            reference = embeddings[1]
        del encoder, embeddings
        release_memory(device)
    else:
        tokens = load_embedding(args.text_embedding)
        if args.cfg != 1:
            reference = load_embedding(args.null_text_embedding)
    if len(tokens) > 1024 or (reference is not None and len(reference) > 1024):
        print('Text exceeds 1024 tokens. Keeping the first 1023 and final token, matching training.', flush=True)
    text = trim(tokens)[None].to(device)
    mask = torch.ones(text.shape[:2], device=device, dtype=torch.bool)
    null_text = trim(reference)[None].to(device) if reference is not None else None
    null_mask = torch.ones(null_text.shape[:2], device=device, dtype=torch.bool) if null_text is not None else None

    print('Loading Body checkpoint...', flush=True)
    saved = read(args.checkpoint)
    config = ModelConfig.from_dict(saved['model_config'])
    with torch.device('meta'):
        model = MIRAGEModel(config)
    model.load_state_dict(saved['model'], strict=True, assign=True)
    step = saved['global_step']
    del saved
    model = model.to(device=device, dtype=getattr(torch, args.dtype)).eval().requires_grad_(False)
    model.text_cache_enabled = args.text_cache
    release_memory(device)
    from mirage.runtime.text_cache import prepare
    with torch.inference_mode():
        text, mask = prepare(model, text, mask)
        if null_text is not None:
            null_text, null_mask = prepare(model, null_text, null_mask)
    sampler = euler if args.sampler == 'euler' else midpoint
    latents = []
    for index in range(args.num_images):
        seed = args.seed + index
        print(f'Sampling {index + 1}/{args.num_images}, checkpoint={step}, seed={seed}, '
              f'{args.width}x{args.height}, {args.sampler} steps={args.steps}, CFG={args.cfg}', flush=True)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(1, 128, args.height // 16, args.width // 16,
                            device=device, dtype=torch.float32, generator=generator)
        with tqdm(total=args.steps, desc='Denoising', unit='step') as progress:
            def report_step(completed, total):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                progress.update(completed - progress.n)
            latent = sampler(model, noise, text, mask, steps=args.steps, guidance=args.cfg,
                             null_text=null_text, null_mask=null_mask, callback=report_step)
        latents.append(latent.cpu())
        del noise, latent
    del model, text, mask, null_text, null_mask
    release_memory(device)

    if args.latent_output:
        args.latent_output.parent.mkdir(parents=True, exist_ok=True)
        latent_destination = image_path(args.latent_output, 0, args.seed)
        with latent_destination.open('xb') as stream:
            torch.save(dict(latent=torch.cat(latents), image_hw=[args.height, args.width],
                            seeds=[args.seed + i for i in range(args.num_images)]), stream)
        print(f'Saved {latent_destination.resolve()}', flush=True)
    print('Loading FLUX.2 VAE and decoding...', flush=True)
    from integrations.flux2_vae import Flux2Codec
    codec = Flux2Codec(args.vae_path, str(device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, latent in enumerate(latents):
        rgb = codec.decode(latent)[0].clamp(0, 1).permute(1, 2, 0).cpu()
        destination = image_path(args.output, index, args.seed + index)
        image_format = Image.registered_extensions()[destination.suffix.lower()]
        with destination.open('xb') as stream:
            Image.fromarray((rgb * 255).round().byte().numpy()).save(stream, format=image_format)
        print(f'Saved {destination.resolve()}', flush=True)


if __name__ == '__main__':
    main()
