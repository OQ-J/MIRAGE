"""Official FLUX.2 VAE posterior mode, FP32 compute, normalized 128-channel /16 latents."""
from pathlib import Path
import hashlib
import torch


class Flux2Codec:
    def __init__(self, path, device='cuda'):
        from diffusers import AutoencoderKLFlux2
        self.path = Path(path)
        self.device = torch.device(device)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            str(path), subfolder='vae', torch_dtype=torch.float32, local_files_only=True
        ).to(self.device).eval().requires_grad_(False)
        self.vae.disable_slicing()
        self.vae.disable_tiling()

    @torch.inference_mode()
    def encode(self, rgb):
        with torch.autocast(device_type=self.device.type, enabled=False):
            z = self.vae.encode(rgb.to(self.device, dtype=torch.float32) * 2 - 1).latent_dist.mode()
            b, c, h, w = z.shape
            z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(b, 4 * c, h // 2, w // 2)
            mean = self.vae.bn.running_mean.float()[None, :, None, None]
            std = (self.vae.bn.running_var.float() + self.vae.config.batch_norm_eps).sqrt()[None, :, None, None]
            return (z.float() - mean) / std

    @torch.inference_mode()
    def decode(self, latent):
        with torch.autocast(device_type=self.device.type, enabled=False):
            mean = self.vae.bn.running_mean.float()[None, :, None, None]
            std = (self.vae.bn.running_var.float() + self.vae.config.batch_norm_eps).sqrt()[None, :, None, None]
            z = latent.to(self.device, dtype=torch.float32) * std + mean
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(b, c // 4, 2 * h, 2 * w)
            return (self.vae.decode(z, return_dict=False)[0].float() + 1) / 2

    def metadata(self):
        weights = self.path / 'vae/diffusion_pytorch_model.safetensors'
        with weights.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(model='AutoencoderKLFlux2', weights_sha256=sha,
                    config=dict(self.vae.config), compute_dtype='float32', posterior='mode',
                    normalization='official_bn_running_mean_and_sqrt_running_var_plus_eps',
                    latent_channels=128, downsample_factor=16, packing='2x2_spatial_into_channels')
