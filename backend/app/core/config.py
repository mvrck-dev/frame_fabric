"""
Centralized pipeline configuration for VisionPhase.
All tunable parameters are exposed here and can be modified
via the /api/config endpoint from the frontend settings panel.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class PipelineConfig:
    # ── SAM Segmentation ──
    sam_model_type: str = "vit_b"

    # ── Mask Refinement ──
    dilation_px: int = 5                    # 3–8 px morphological dilation
    feather_sigma: float = 2.0              # 1.0–5.0 Gaussian edge softening

    # ── CLIP Classifier ──
    clip_model: str = "openai/clip-vit-base-patch32"

    # ── GAN Live Preview ──
    gan_enabled: bool = False               # Flip True once SPADE weights exist

    # ── SDXL Export ──
    sdxl_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    sdxl_steps: int = 20                    # 4–30
    sdxl_guidance: float = 5.0              # 4.0–7.0
    sdxl_denoise: float = 0.5               # 0.4–0.6
    sdxl_scheduler: Literal["euler", "lcm"] = "euler"
    lora_scale: float = 0.7                 # 0.6–0.9
    controlnet_scale: float = 0.6           # 0.3–1.0
    controlnet_model: str = "diffusers/controlnet-depth-sdxl-1.0-small"
    ip_adapter_model: str = "h94/IP-Adapter"

    # ── Post-Processing ──
    poisson_blend: bool = True
    color_transfer: bool = True
    esrgan_enabled: bool = False            # Optional, heavy on VRAM
    esrgan_scale: int = 4                   # 2 or 4

    # ── Class Labels for CLIP ──
    class_labels: list[str] = field(default_factory=lambda: [
        "curtain", "pillow", "pillowcase", "mattress", "towel",
        "bedsheet", "rug", "carpet", "sofa", "chair",
        "wall", "floor", "window", "table", "lamp",
        "blanket", "cushion", "throw", "duvet", "tapestry",
    ])

    def to_dict(self) -> dict:
        return asdict(self)

    def update_from_dict(self, data: dict) -> None:
        """Safely update config fields from a partial dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                expected_type = type(getattr(self, key))
                try:
                    setattr(self, key, expected_type(value))
                except (ValueError, TypeError):
                    pass  # Skip invalid values silently

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Global singleton ──
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config
