import time
import gc
import torch
import numpy as np
from PIL import Image
import cv2

from app.core.config import get_config


class FastPreviewPipeline:
    """
    Lightweight, fast Stable Diffusion 1.5 Inpainting for Live Previews.
    Designed to run significantly faster than the full SDXL pipeline and uses minimal VRAM.
    """
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _ensure_loaded(self):
        if self.pipe is not None:
            return
            
        print("[Fast Preview] Loading lightweight SD 1.5 inpainting model...")
        start = time.time()
        
        try:
            from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
            
            # Using standard v1.5 inpainting as the fast preview backbone
            model_id = "runwayml/stable-diffusion-inpainting"
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(self.device)
            
            # Use Euler for fast standard diffusion steps
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            
            # Optimizations to prevent VRAM competition with SDXL
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
                
            print(f"[Fast Preview] Ready in {time.time() - start:.2f}s")
            
        except Exception as e:
            print(f"[Fast Preview] Failed to load dummy inpaint pipeline: {e}")
            self.pipe = None
            raise

    def generate_preview(self, scene_image: Image.Image, product_image: Image.Image, mask: np.ndarray, product_name: str = "") -> Image.Image:
        """
        Generate a rapid preview by completely regenerating the masked area.
        """
        self._ensure_loaded()
        if self.pipe is None:
            # Absolute fallback if network/loading fails
            return scene_image
            
        print("[Fast Preview] Generating fast structural preview...")
        start = time.time()
        
        # Prepare mask
        mask_pil = Image.fromarray((mask.astype(np.uint8) * 255))
        original_size = scene_image.size
        
        # Downscale for extreme speed in preview
        preview_size = (512, 512)
        scene_small = scene_image.resize(preview_size, Image.Resampling.LANCZOS)
        mask_small = mask_pil.resize(preview_size, Image.Resampling.NEAREST)
        
        # Run incredibly fast generation for live feedback
        prompt = f"a seamlessly integrated high quality {product_name}, modern interior design, proper placement, sharp lighting, 4k"
        
        try:
            result_small = self.pipe(
                prompt=prompt,
                image=scene_small,
                mask_image=mask_small,
                num_inference_steps=12,
                guidance_scale=6.5, # Standard guidance
                strength=1.0, # Complete structural recreation!
            ).images[0]
            
            # Upscale back to original size
            result_full = result_small.resize(original_size, Image.Resampling.LANCZOS)
            
            # Feathered blending back into the un-downscaled scene for crisp edges
            from app.core.postprocess import feather_mask
            scene_np = np.array(scene_image)
            result_np = np.array(result_full)
            
            alpha = feather_mask(mask, sigma=2.0)
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            
            final_np = (result_np.astype(np.float32) * alpha_3ch + 
                        scene_np.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)
                        
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"[Fast Preview] Done in {time.time() - start:.2f}s")
            return Image.fromarray(final_np)
            
        except Exception as e:
            print(f"[Fast Preview] Error: {e}")
            import traceback
            traceback.print_exc()
            return scene_image

# Global singleton
_fast_preview_engine = None

def get_fast_preview_engine() -> FastPreviewPipeline:
    global _fast_preview_engine
    if _fast_preview_engine is None:
        _fast_preview_engine = FastPreviewPipeline()
    return _fast_preview_engine
