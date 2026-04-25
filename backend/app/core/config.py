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
    sdxl_denoise: float = 0.85              # 0.7-1.0 for complete object replacement
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
        "ceiling surface", "plain ceiling", "bookcase", "cabinet", "chest of drawers", "desk", 
        "shelf", "wardrobe", "door", "indoor plant", "mirror", "tv", "fireplace", "bed", 
        "curtain", "pillow", "pillowcase", "mattress", "towel", "bedsheet", "rug", "carpet", 
        "sofa", "chair", "wall surface", "plain wall", "floor surface", "plain floor", 
        "window", "table", "ceiling lamp", "pendant light", "chandelier", "floor lamp", 
        "blanket", "cushion", "throw", "duvet", "painting", "picture frame",
    ])

    def to_dict(self) -> dict:
        return asdict(self)

    def update_from_dict(self, data: dict) -> None:
        """Safely update config fields from a partial dictionary."""
        for key, value in data.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            # Skip list fields from the settings panel (not user-editable via UI)
            if isinstance(current, list):
                continue
            try:
                # Accept int where float is expected (slider sends int for whole numbers)
                if isinstance(current, float) and isinstance(value, int):
                    setattr(self, key, float(value))
                elif isinstance(current, bool):
                    # JSON booleans are already bool; guard against "true"/"false" strings
                    if isinstance(value, bool):
                        setattr(self, key, value)
                    elif isinstance(value, str):
                        setattr(self, key, value.lower() in ("true", "1", "yes"))
                else:
                    setattr(self, key, type(current)(value))
            except (ValueError, TypeError):
                pass  # Skip genuinely invalid values

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Global singleton ──
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config
