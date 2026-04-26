"""
SPADE-style conditional GAN for live preview.

Architecture:
- SPADE Generator with gated/partial convolutions
- PatchGAN Multi-scale Discriminator with spectral normalization
- Training: L1 + VGG perceptual + Gram-matrix style + mask-weighted adversarial
- TTUR: G lr=2e-4, D lr=1e-4

Until trained weights exist, the preview falls back to a compositing pipeline:
color-matched product warped into the mask region with feathered blending.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from app.core.config import get_config
from app.core.postprocess import (
    composite_product_into_mask,
    color_transfer_reinhard,
    feather_mask,
)


# ═══════════════════════════════════════════════════
#  SPADE Architecture (ready for training)
# ═══════════════════════════════════════════════════

class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization block."""
    
    def __init__(self, norm_nc: int, label_nc: int):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class GatedConv2d(nn.Module):
    """Gated convolution for handling masked regions without seam artifacts."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv_feature = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv_feature(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.norm(feature * gate)


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization."""
    
    def __init__(self, in_ch: int, out_ch: int, label_nc: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        
        self.spade_1 = SPADE(in_ch, label_nc)
        self.conv_1 = GatedConv2d(in_ch, mid_ch)
        self.spade_2 = SPADE(mid_ch, label_nc)
        self.conv_2 = GatedConv2d(mid_ch, out_ch)
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.spade_1(x, segmap)
        out = F.leaky_relu(out, 0.2)
        out = self.conv_1(out)
        out = self.spade_2(out, segmap)
        out = F.leaky_relu(out, 0.2)
        out = self.conv_2(out)
        return out + residual


class SPADEGenerator(nn.Module):
    """
    SPADE Generator for interior inpainting.
    
    Inputs: (masked_image, semantic_map, mask) as spatial conditioning
    + CLIP/IP-Adapter embedding of inventory item via AdaIN at bottleneck
    """
    
    def __init__(self, input_nc: int = 7, output_nc: int = 3, 
                 ngf: int = 64, label_nc: int = 7, embed_dim: int = 512):
        """
        Args:
            input_nc: masked_image(3) + semantic_map(3) + mask(1) = 7
            output_nc: RGB output channels
            ngf: Base number of generator filters  
            label_nc: Same as input_nc for SPADE conditioning
            embed_dim: CLIP embedding dimension for product conditioning
        """
        super().__init__()
        
        # Encoder with gated convolutions
        self.enc1 = GatedConv2d(input_nc, ngf, 3, 2, 1)         # /2
        self.enc2 = GatedConv2d(ngf, ngf * 2, 3, 2, 1)          # /4
        self.enc3 = GatedConv2d(ngf * 2, ngf * 4, 3, 2, 1)      # /8
        self.enc4 = GatedConv2d(ngf * 4, ngf * 8, 3, 2, 1)      # /16
        
        # AdaIN for product embedding injection at bottleneck
        self.embed_fc = nn.Linear(embed_dim, ngf * 8 * 2)  # gamma + beta
        
        # Decoder with SPADE blocks
        self.dec4 = SPADEResBlock(ngf * 8, ngf * 4, label_nc)
        self.dec3 = SPADEResBlock(ngf * 4, ngf * 2, label_nc)
        self.dec2 = SPADEResBlock(ngf * 2, ngf, label_nc)
        self.dec1 = SPADEResBlock(ngf, ngf, label_nc)
        
        self.out_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, output_nc, 3, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, masked_image: torch.Tensor, semantic_map: torch.Tensor,
                mask: torch.Tensor, product_embedding: torch.Tensor) -> torch.Tensor:
        # Concatenate inputs
        x = torch.cat([masked_image, semantic_map, mask], dim=1)
        segmap = x  # Use as SPADE conditioning map
        
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(F.leaky_relu(e1, 0.2))
        e3 = self.enc3(F.leaky_relu(e2, 0.2))
        e4 = self.enc4(F.leaky_relu(e3, 0.2))
        
        # Inject product embedding via AdaIN
        style = self.embed_fc(product_embedding)  # (B, ngf*8*2)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.view(e4.shape[0], -1, 1, 1)
        beta = beta.view(e4.shape[0], -1, 1, 1)
        e4 = e4 * (1 + gamma) + beta
        
        # Decode with SPADE
        d4 = self.dec4(F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), segmap)
        d3 = self.dec3(F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), segmap)
        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), segmap)
        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), segmap)
        
        return self.out_conv(d1)


class PatchGANDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator with spectral normalization.
    """
    
    def __init__(self, input_nc: int = 10, ndf: int = 64, n_layers: int = 4):
        """
        Args:
            input_nc: image(3) + semantic_map(3) + mask(1) + generated/real(3) = 10
        """
        super().__init__()
        
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        
        # Final patch output
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)),
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ═══════════════════════════════════════════════════
#  Live Preview Engine
# ═══════════════════════════════════════════════════

class LivePreviewEngine:
    """
    Manages live preview generation.
    Uses trained SPADE-GAN when available, otherwise falls back to
    compositing (color-matched warp + feathered blend).
    """
    
    def __init__(self):
        self.generator: SPADEGenerator | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../models/gan/spade_generator.pth"
        )
    
    def _try_load_gan(self) -> bool:
        """Attempt to load trained GAN weights. Returns True if successful."""
        if not os.path.exists(self._weights_path):
            return False
        
        try:
            print("[GAN Preview] Loading trained SPADE generator...")
            self.generator = SPADEGenerator().to(self.device)
            state_dict = torch.load(self._weights_path, map_location=self.device)
            self.generator.load_state_dict(state_dict)
            self.generator.eval()
            print("[GAN Preview] Generator loaded successfully.")
            return True
        except Exception as e:
            print(f"[GAN Preview] Failed to load GAN weights: {e}")
            self.generator = None
            return False
    
    def generate_preview(
        self,
        scene_image: Image.Image,
        product_image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """
        Generate a live preview of the product placed into the masked region.
        
        Args:
            scene_image: Original room photo (PIL RGB)
            product_image: Inventory product photo (PIL RGB)
            mask: Binary mask (H, W) bool
        
        Returns:
            Composited preview image (PIL RGB)
        """
        config = get_config()
        
        # Try GAN path if enabled
        if config.gan_enabled and self.generator is None:
            self._try_load_gan()
        
        if config.gan_enabled and self.generator is not None:
            return self._gan_preview(scene_image, product_image, mask)
        else:
            return self._composite_preview(scene_image, product_image, mask)
    
    def _composite_preview(
        self,
        scene_image: Image.Image,
        product_image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """Fallback compositing preview with color matching."""
        print("[GAN Preview] Using composite fallback (GAN not trained yet)...")
        start = time.time()
        
        config = get_config()
        scene_bgr = cv2.cvtColor(np.array(scene_image), cv2.COLOR_RGB2BGR)
        product_bgr = cv2.cvtColor(np.array(product_image), cv2.COLOR_RGB2BGR)
        
        result_bgr = composite_product_into_mask(
            scene_bgr=scene_bgr,
            product_bgr=product_bgr,
            mask=mask,
            feather_sigma=config.feather_sigma,
            do_color_transfer=config.color_transfer,
        )
        
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        print(f"[GAN Preview] Composite generated in {time.time() - start:.2f}s")
        return Image.fromarray(result_rgb)
    
    def _gan_preview(
        self,
        scene_image: Image.Image,
        product_image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """SPADE-GAN inference path (when trained weights exist)."""
        print("[GAN Preview] Running SPADE-GAN inference...")
        start = time.time()
        
        # Prepare tensors (simplified — full pipeline would include CLIP embedding)
        scene_np = np.array(scene_image.resize((512, 512)))
        mask_resized = cv2.resize(mask.astype(np.uint8), (512, 512)) > 0
        
        # Masked image: zero out the masked region
        masked = scene_np.copy()
        masked[mask_resized] = 0
        
        # Convert to tensors
        masked_t = torch.from_numpy(masked).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        sem_t = masked_t.clone()  # Simplified: use masked image as semantic map
        mask_t = torch.from_numpy(mask_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        # Dummy product embedding (replaced with real CLIP when fully integrated)
        embed_t = torch.randn(1, 512)
        
        # Move to device
        masked_t = masked_t.to(self.device)
        sem_t = sem_t.to(self.device)
        mask_t = mask_t.to(self.device)
        embed_t = embed_t.to(self.device)
        
        with torch.no_grad():
            output = self.generator(masked_t, sem_t, mask_t, embed_t)
        
        # Convert back to image
        output_np = ((output[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        output_pil = Image.fromarray(output_np).resize(scene_image.size)
        
        # Blend only the masked region back into the original scene
        scene_np_full = np.array(scene_image)
        output_np_full = np.array(output_pil)
        feathered = feather_mask(mask, sigma=2.0)
        alpha = np.stack([feathered] * 3, axis=-1)
        
        result = (output_np_full * alpha + scene_np_full * (1 - alpha)).astype(np.uint8)
        
        print(f"[GAN Preview] Generated in {time.time() - start:.2f}s")
        return Image.fromarray(result)


# ── Global singleton ──
_preview_engine: LivePreviewEngine | None = None


def get_preview_engine() -> LivePreviewEngine:
    global _preview_engine
    if _preview_engine is None:
        _preview_engine = LivePreviewEngine()
    return _preview_engine
