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
            )
            
            # Use Euler for fast standard diffusion steps
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            
            # Optimizations to prevent VRAM competition with SDXL
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
                
            try:
                self.pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin"
                )
                self.pipe.set_ip_adapter_scale(0.7)
                print("[Fast Preview] IP-Adapter loaded successfully")
            except Exception as e:
                print(f"[Fast Preview] Continuing without IP-adapter (Missing): {e}")

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
            ip_image = product_image.resize((256, 256))
        except Exception:
            ip_image = None
            
        gen_kwargs = {
            "prompt": prompt,
            "image": scene_small,
            "mask_image": mask_small,
            "num_inference_steps": 12,
            "guidance_scale": 6.5,
            "strength": 1.0,
        }
        
        if ip_image is not None and getattr(self.pipe, "set_ip_adapter_scale", None):
            gen_kwargs["ip_adapter_image"] = ip_image
        
        try:
            result_small = self.pipe(**gen_kwargs).images[0]
            
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

    def generate_t2i(self, prompt: str, steps: int = 15, seed: int = 42, num_samples: int = 1) -> list[Image.Image]:
        """
        Generate rapid fabric swatches using SD 1.5. 
        Much faster than SDXL for simple texture previews.
        """
        self._ensure_loaded()
        if self.pipe is None:
            return [Image.new("RGB", (512, 512), (128, 128, 128))] * num_samples

        # SD 1.5 T2I simulation via inpaint (black image + full mask)
        dummy_image = Image.new("RGB", (512, 512), (0, 0, 0))
        dummy_mask = Image.new("L", (512, 512), 255)
        
        # Disable IP-Adapter for pure T2I fabric generation
        _has_ip_adapter = (
            hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None
        )
        if _has_ip_adapter and getattr(self.pipe, "set_ip_adapter_scale", None):
            self.pipe.set_ip_adapter_scale(0.0)

        gen_kwargs = {
            "prompt": prompt,
            "image": dummy_image,
            "mask_image": dummy_mask,
            "num_inference_steps": steps,
            "guidance_scale": 7.5,
            "strength": 1.0,
            "num_images_per_prompt": num_samples,
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }

        # CRITICAL FIX: If IP-Adapter is loaded, diffusers REQUIRES an image even if scale is 0.
        # Failing to provide it causes added_cond_kwargs to be None, triggering a crash.
        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            gen_kwargs["ip_adapter_image"] = dummy_image

        print(f"[Fast Preview] Generating rapid swatch: {prompt[:40]}...")
        result = self.pipe(**gen_kwargs).images
        
        # Re-enable IP-Adapter for future preview calls
        if _has_ip_adapter and getattr(self.pipe, "set_ip_adapter_scale", None):
            self.pipe.set_ip_adapter_scale(0.7)
        
        return result

# Global singleton
_fast_preview_engine = None

def get_fast_preview_engine() -> FastPreviewPipeline:
    global _fast_preview_engine
    if _fast_preview_engine is None:
        _fast_preview_engine = FastPreviewPipeline()
    return _fast_preview_engine
