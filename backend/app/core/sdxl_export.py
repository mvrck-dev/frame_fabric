"""
SDXL Inpainting pipeline for high-quality export.

Integrates:
- Stable Diffusion XL Inpainting (fp16, CPU offload, VAE slicing)
- ControlNet (depth) for geometric locking
- IP-Adapter for exact product visual conditioning
- Category LoRA for consistency
- Post-processing: edge-aware compositing, color transfer, Poisson blending, ESRGAN
"""
import os
import time
import gc
import torch
import numpy as np
from PIL import Image
import cv2

from app.core.config import get_config
from app.core.postprocess import (
    feather_mask,
    color_transfer_reinhard,
    poisson_blend,
    histogram_match,
)


class SDXLExportPipeline:
    """
    Lazy-loaded SDXL inpainting pipeline for final high-quality exports.
    
    Memory strategy for 8 GB VRAM:
    - fp16 throughout
    - Sequential CPU offloading (model components move to GPU only when active)
    - VAE slicing for large images
    - Explicit gc + torch.cuda.empty_cache() after inference
    """
    
    def __init__(self):
        self.pipe = None
        self.controlnet = None
        self.depth_estimator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../models"
        )
    
    def _ensure_loaded(self):
        """Lazy-load the full SDXL pipeline on first export."""
        if self.pipe is not None:
            return
        
        config = get_config()
        print("[SDXL Export] Loading pipeline (this may take a while on first run)...")
        start = time.time()
        
        try:
            from diffusers import (
                StableDiffusionXLInpaintPipeline,
                StableDiffusionXLControlNetInpaintPipeline,
                ControlNetModel,
                EulerDiscreteScheduler,
                LCMScheduler,
            )
            
            # Load ControlNet for depth conditioning
            print("[SDXL Export] Loading ControlNet (depth)...")
            try:
                self.controlnet = ControlNetModel.from_pretrained(
                    config.controlnet_model,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
            except Exception as e:
                print(f"[SDXL Export] ControlNet load failed, proceeding without: {e}")
                self.controlnet = None
            
            # Load SDXL Inpainting pipeline
            print("[SDXL Export] Loading SDXL Inpainting model...")
            pipe_kwargs = {
                "torch_dtype": torch.float16,
                "variant": "fp16",
                "use_safetensors": True,
            }
            
            try:
                if self.controlnet is not None:
                    pipe_kwargs["controlnet"] = self.controlnet
                    self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                        config.sdxl_model,
                        **pipe_kwargs,
                    )
                else:
                    self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                        config.sdxl_model,
                        **pipe_kwargs,
                    )
            except Exception as e:
                # Fallback: try without controlnet integration
                print(f"[SDXL Export] Pipeline load failed ({e}), falling back without ControlNet...")
                pipe_kwargs.pop("controlnet", None)
                self.controlnet = None
                self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    config.sdxl_model,
                    **pipe_kwargs,
                )
            
            # Memory optimizations for 8GB VRAM
            self.pipe.enable_model_cpu_offload() # Better memory/speed tradeoff than sequential offload
            self.pipe.enable_vae_slicing()
            try:
                self.pipe.enable_vae_tiling()
            except Exception:
                pass
            
            # Try xformers for memory-efficient attention
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[SDXL Export] xformers memory-efficient attention enabled")
            except Exception:
                print("[SDXL Export] xformers not available, using default attention")
            
            # Set scheduler
            if config.sdxl_scheduler == "lcm":
                self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            else:
                self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            
            # Try loading IP-Adapter
            try:
                self.pipe.load_ip_adapter(
                    config.ip_adapter_model,
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                )
                print("[SDXL Export] IP-Adapter loaded successfully")
            except Exception as e:
                print(f"[SDXL Export] IP-Adapter load failed, proceeding without: {e}")
            
            print(f"[SDXL Export] Pipeline ready in {time.time() - start:.2f}s")
            
        except Exception as e:
            print(f"[SDXL Export] CRITICAL: Failed to load pipeline: {e}")
            self.pipe = None
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"SDXL pipeline initialization failed: {e}")
    
    def _load_lora(self, category: str) -> bool:
        """Attempt to load a category-specific LoRA. Returns True if successful."""
        if self.pipe is None:
            return False
        
        lora_path = os.path.join(self._models_dir, "lora", f"{category}.safetensors")
        if not os.path.exists(lora_path):
            print(f"[SDXL Export] No LoRA found for category '{category}', skipping")
            return False
        
        try:
            config = get_config()
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=config.lora_scale)
            print(f"[SDXL Export] LoRA loaded for '{category}' (scale={config.lora_scale})")
            return True
        except Exception as e:
            print(f"[SDXL Export] Failed to load LoRA: {e}")
            return False
    
    def _estimate_depth(self, image: Image.Image) -> Image.Image:
        """Generate depth map using MiDaS for ControlNet conditioning."""
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            
            if self.depth_estimator is None:
                print("[SDXL Export] Loading MiDaS depth estimator...")
                self.depth_estimator = {
                    "processor": DPTImageProcessor.from_pretrained("Intel/dpt-small"),
                    "model": DPTForDepthEstimation.from_pretrained(
                        "Intel/dpt-small", torch_dtype=torch.float16
                    ),
                }
            
            processor = self.depth_estimator["processor"]
            model = self.depth_estimator["model"].to(self.device)
            
            inputs = processor(images=image, return_tensors="pt").to(self.device, torch.float16)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Normalize to 0–1 range
            depth = predicted_depth.squeeze().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = (depth * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth).resize(image.size)
            
            # Move model back to CPU to free VRAM
            model.to("cpu")
            torch.cuda.empty_cache()
            
            return depth_image
            
        except Exception as e:
            print(f"[SDXL Export] Depth estimation failed, using blank: {e}")
            return Image.new("L", image.size, 128)
    
    def export(
        self,
        scene_image: Image.Image,
        product_image: Image.Image,
        mask: np.ndarray,
        category: str = "",
        prompt: str = "",
    ) -> Image.Image:
        """
        Run the full SDXL inpainting export pipeline.
        """
        self._ensure_loaded()
        if self.pipe is None:
            raise RuntimeError("SDXL pipeline not loaded")
        
        config = get_config()
        print(f"[SDXL Export] Starting export (steps={config.sdxl_steps}, "
              f"guidance={config.sdxl_guidance}, denoise={config.sdxl_denoise})...")
        start = time.time()
        
        # Try loading category LoRA
        if category:
            self._load_lora(category)
        
        # Prepare mask image (PIL, white = inpaint area)
        mask_pil = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_pil = mask_pil.resize(scene_image.size)
        
        # Generate depth map for ControlNet
        depth_image = self._estimate_depth(scene_image)
        
        # Build prompt if not provided
        if not prompt:
            prompt = f"interior design photo, {category} replacement, photorealistic, professional lighting, high quality"
        negative_prompt = "blurry, distorted, artifacts, low quality, cartoon, painting"
        
        # Set IP-Adapter image if available
        try:
            self.pipe.set_ip_adapter_scale(0.6)
            ip_adapter_image = product_image.resize((512, 512))
        except Exception:
            ip_adapter_image = None
        
        # Run SDXL inpainting with full latent regeneration (no fake sticker overlay)
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": scene_image,        # Provide the raw scene
            "mask_image": mask_pil,      # Only the masked area will be regenerated
            "num_inference_steps": config.sdxl_steps,
            "guidance_scale": config.sdxl_guidance,
            "strength": config.sdxl_denoise, # Standard deep denoise (1.0 or high) for complete generation
            "generator": torch.Generator(device="cpu").manual_seed(42),
        }
        
        # Add IP-Adapter image if loaded
        if ip_adapter_image is not None:
            gen_kwargs["ip_adapter_image"] = ip_adapter_image
        
        # Add ControlNet image if available  
        if self.controlnet is not None:
            gen_kwargs["control_image"] = depth_image
            gen_kwargs["controlnet_conditioning_scale"] = config.controlnet_scale
        
        print("[SDXL Export] Executing generation. Please monitor memory...")
        result = self.pipe(**gen_kwargs).images[0]
        
        # ── Post-Processing ──
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        scene_np = cv2.cvtColor(np.array(scene_image), cv2.COLOR_RGB2BGR)
        
        # Color transfer to match scene illumination
        if config.color_transfer:
            print("[SDXL Export] Applying color transfer...")
            result_np = color_transfer_reinhard(result_np, scene_np, mask)
        
        # Poisson blending for lighting continuity
        if config.poisson_blend:
            print("[SDXL Export] Applying Poisson blending...")
            result_np = poisson_blend(result_np, scene_np, mask)
        
        # Edge-aware compositing with feathered mask
        feathered = feather_mask(mask, sigma=config.feather_sigma)
        alpha = np.stack([feathered] * 3, axis=-1)
        final_np = (result_np.astype(np.float32) * alpha + 
                    scene_np.astype(np.float32) * (1 - alpha))
        final_np = final_np.astype(np.uint8)
        
        # Optional ESRGAN super-resolution on edited patch
        if config.esrgan_enabled:
            final_np = self._apply_esrgan(final_np, mask)
        
        final_rgb = cv2.cvtColor(final_np, cv2.COLOR_BGR2RGB)
        
        # Cleanup VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        elapsed = time.time() - start
        print(f"[SDXL Export] Export completed in {elapsed:.1f}s")
        
        return Image.fromarray(final_rgb)
    
    def _apply_esrgan(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply Real-ESRGAN super-resolution to the edited region only."""
        try:
            # Crop the edited region
            from app.core.postprocess import crop_to_mask
            
            crop, (y1, y2, x1, x2) = crop_to_mask(image_bgr, mask, padding=10)
            
            # Try to use Real-ESRGAN
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                config = get_config()
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=config.esrgan_scale)
                
                upsampler = RealESRGANer(
                    scale=config.esrgan_scale,
                    model_path=os.path.join(self._models_dir, "esrgan", "RealESRGAN_x4plus.pth"),
                    model=model,
                    half=True,
                    device=self.device,
                )
                
                sr_crop, _ = upsampler.enhance(crop, outscale=config.esrgan_scale)
                # Downscale back to original crop size
                sr_crop = cv2.resize(sr_crop, (crop.shape[1], crop.shape[0]),
                                     interpolation=cv2.INTER_LANCZOS4)
                
                result = image_bgr.copy()
                result[y1:y2, x1:x2] = sr_crop
                print("[SDXL Export] ESRGAN applied to edited region")
                return result
                
            except ImportError:
                print("[SDXL Export] Real-ESRGAN not installed, skipping super-res")
                return image_bgr
                
        except Exception as e:
            print(f"[SDXL Export] ESRGAN failed: {e}")
            return image_bgr


# ── Global singleton ──
_export_pipeline: SDXLExportPipeline | None = None


def get_export_pipeline() -> SDXLExportPipeline:
    global _export_pipeline
    if _export_pipeline is None:
        _export_pipeline = SDXLExportPipeline()
    return _export_pipeline
