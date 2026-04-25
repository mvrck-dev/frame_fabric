"""
SDXL Inpainting pipeline for high-quality export.

Integrates:
- Stable Diffusion XL Inpainting (fp16)
- ControlNet (depth) for geometric locking
- IP-Adapter for exact product visual conditioning
- Category LoRA for consistency
- Post-processing: color transfer → feathered composite → Poisson seamless clone
"""
import os
import time
import gc
import threading
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

# Module-level cancel event so the HTTP layer can signal the inference thread
_cancel_event = threading.Event()


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
        self._lock = threading.Lock()  # Prevent concurrent pipeline loads
    
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
            
            # RTX 5070 has 12 GB+ VRAM — do NOT use CPU offload, it cripples speed.
            # Move everything to GPU directly and use VAE tiling instead of slicing.
            self.pipe.to(self.device)
            try:
                self.pipe.vae.enable_tiling()  # Handles large images without OOM
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
                # Manually move the image_encoder to GPU to prevent "meta device" errors
                if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
                    self.pipe.image_encoder.to(self.device, dtype=torch.float16)
                print("[SDXL Export] IP-Adapter loaded and synced to GPU")
            except Exception as e:
                print(f"[SDXL Export] IP-Adapter load failed (continuing without): {e}")
            
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
            # Clear previous LoRAs
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
                
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=config.lora_scale)
            print(f"[SDXL Export] LoRA loaded for '{category}' (scale={config.lora_scale})")
            return True
        except Exception as e:
            print(f"[SDXL Export] Failed to load LoRA: {e}")
            return False
            
    def generate_t2i(self, prompt: str, negative_prompt: str = "", steps: int = 25, seed: int = 42) -> Image.Image:
        """Generate a Text-to-Image result using the inpaint pipeline (simulated)."""
        self._ensure_loaded()
        
        # Inpainting as T2I: black image + full white mask
        dummy_image = Image.new("RGB", (1024, 1024), (128, 128, 128))
        dummy_mask = Image.new("L", (1024, 1024), 255)
        
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": dummy_image,
            "mask_image": dummy_mask,
            "num_inference_steps": steps,
            "guidance_scale": 7.5,
            "strength": 1.0,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }
        
        # Add dummy control image if ControlNet is active
        if self.controlnet is not None:
            gen_kwargs["control_image"] = dummy_image.convert("L")
            gen_kwargs["controlnet_conditioning_scale"] = 0.0 # Disable its effect
        
        # IP-Adapter requires an image in gen_kwargs if loaded, even at 0.0 scale
        try:
            self.pipe.set_ip_adapter_scale(0.0)
            gen_kwargs["ip_adapter_image"] = dummy_image.resize((512, 512))
        except Exception:
            pass
            
        print(f"[SDXL Export] Generating T2I swatch: {prompt[:50]}...")
        result = self.pipe(**gen_kwargs).images[0]
        
        # Reset IPAdapter scale
        try:
            self.pipe.set_ip_adapter_scale(0.6)
        except Exception:
            pass
            
        return result
    
    def _estimate_depth(self, image: Image.Image) -> Image.Image:
        """Generate depth map using MiDaS for ControlNet conditioning."""
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            
            if self.depth_estimator is None:
                print("[SDXL Export] Loading MiDaS depth estimator...")
                self.depth_estimator = {
                    "processor": DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas"),
                    "model": DPTForDepthEstimation.from_pretrained(
                        "Intel/dpt-hybrid-midas", torch_dtype=torch.float16
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
        cancel_event: threading.Event | None = None,
        fabric_mode: bool = False,
    ) -> Image.Image:
        """
        Run the full SDXL inpainting export pipeline.
        
        fabric_mode=True: skips color transfer and Poisson blend so new
        fabric colours/textures are preserved rather than mapped back to
        the original scene palette.
        """
        with self._lock:
            return self._export_inner(
                scene_image, product_image, mask,
                category, prompt, cancel_event, fabric_mode
            )
    
    def _export_inner(
        self,
        scene_image: Image.Image,
        product_image: Image.Image,
        mask: np.ndarray,
        category: str = "",
        prompt: str = "",
        cancel_event: threading.Event | None = None,
        fabric_mode: bool = False,
    ) -> Image.Image:
        def _check_cancel():
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Generation cancelled by client")
        
        _check_cancel()
        self._ensure_loaded()
        if self.pipe is None:
            raise RuntimeError("SDXL pipeline not loaded")
        
        config = get_config()
        print(f"[SDXL Export] Starting export (steps={config.sdxl_steps}, "
              f"guidance={config.sdxl_guidance}, denoise={config.sdxl_denoise}, "
              f"fabric_mode={fabric_mode})...")
        start = time.time()
        
        # Validate mask — log its coverage so we can catch empty/tiny masks
        mask = mask.astype(bool)
        mask_coverage = mask.sum() / mask.size
        print(f"[SDXL Export] Mask coverage: {mask.sum()} px ({mask_coverage:.2%} of image)")
        if not mask.any():
            raise ValueError("Empty mask — nothing to inpaint")
        
        # Try loading category LoRA
        _check_cancel()
        if category:
            self._load_lora(category)
        
        # Resize scene to a clean multiple of 8 for SDXL (required by VAE)
        orig_size = scene_image.size  # (W, H)
        target_w = (orig_size[0] // 8) * 8
        target_h = (orig_size[1] // 8) * 8
        if (target_w, target_h) != orig_size:
            scene_image = scene_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            # Resize mask to match
            mask_resized = Image.fromarray((mask.astype(np.uint8) * 255))
            mask_resized = mask_resized.resize((target_w, target_h), Image.Resampling.NEAREST)
            mask = np.array(mask_resized) > 127
        
        # Prepare mask image (PIL, white = inpaint area)
        # Dilate mask slightly to avoid hard boundary artifacts at edges
        from app.core.postprocess import dilate_mask
        mask_dilated = dilate_mask(mask, kernel_size=3)
        mask_pil = Image.fromarray((mask_dilated.astype(np.uint8) * 255))
        
        _check_cancel()
        
        # Generate depth map for ControlNet
        depth_image = self._estimate_depth(scene_image)
        # Zero out depth inside the mask so ControlNet doesn't constrain the generation
        depth_np = np.array(depth_image)
        if depth_np.ndim == 2:
            depth_np[mask_dilated] = 0
        depth_image = Image.fromarray(depth_np)
        
        # Build prompt if not provided — be very specific to prevent hallucination
        if not prompt:
            cat_str = category if category else "furniture"
            prompt = (
                f"photorealistic interior design photo, a {cat_str} seamlessly "
                f"integrated into the room, matching existing lighting and perspective, "
                f"professional architectural photography, 8k, sharp focus"
            )
        negative_prompt = (
            "blurry, distorted, artifacts, low quality, cartoon, painting, "
            "floating, detached, wrong perspective, extra limbs, duplicate, "
            "ugly, bad anatomy, watermark, text, signature"
        )
        
        _check_cancel()
        
        # Set IP-Adapter — higher scale in fabric mode since the swatch IS the reference
        ip_adapter_image = None
        ip_scale = 0.75 if fabric_mode else 0.5
        try:
            self.pipe.set_ip_adapter_scale(ip_scale)
            ip_adapter_image = product_image.resize((512, 512))
            print(f"[SDXL Export] IP-Adapter scale set to {ip_scale}")
        except Exception as e:
            print(f"[SDXL Export] IP-Adapter not available: {e}")
        
        # In fabric mode use full strength so the mask region is completely regenerated
        denoise_strength = 1.0 if fabric_mode else config.sdxl_denoise
        
        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": scene_image,
            "mask_image": mask_pil,
            "num_inference_steps": config.sdxl_steps,
            "guidance_scale": config.sdxl_guidance,
            "strength": denoise_strength,
            "generator": torch.Generator(device=self.device).manual_seed(42),
        }
        print(f"[SDXL Export] denoise_strength={denoise_strength}, steps={config.sdxl_steps}")
        
        if ip_adapter_image is not None:
            gen_kwargs["ip_adapter_image"] = ip_adapter_image
        
        if self.controlnet is not None:
            gen_kwargs["control_image"] = depth_image
            gen_kwargs["controlnet_conditioning_scale"] = config.controlnet_scale
        
        print("[SDXL Export] Executing generation...")
        _check_cancel()
        result = self.pipe(**gen_kwargs).images[0]
        
        # ── Post-Processing ──
        # SDXL inpainting returns the FULL composited scene (not just the patch).
        # We composite again with a feathered mask to get soft edges, then
        # optionally apply colour correction and Poisson blending.
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        scene_np  = cv2.cvtColor(np.array(scene_image), cv2.COLOR_RGB2BGR)
        
        print(f"[SDXL Export] result shape={result_np.shape}, scene shape={scene_np.shape}, "
              f"mask shape={mask.shape}, mask sum={mask.sum()}")
        
        # 1. Feathered alpha composite — primary compositing step.
        #    Use the ORIGINAL (non-dilated) mask so we don't over-extend into scene.
        feathered = feather_mask(mask, sigma=config.feather_sigma)
        alpha     = np.stack([feathered] * 3, axis=-1)
        
        # Extract only the inpainted patch from the SDXL result and blend it in.
        # We deliberately do NOT use result_np outside the mask — SDXL already
        # filled the rest with the scene; blending again could double-shift colours.
        composited_np = scene_np.copy()
        composited_np = (
            result_np.astype(np.float32) * alpha +
            scene_np.astype(np.float32)  * (1.0 - alpha)
        ).astype(np.uint8)
        
        if fabric_mode:
            # ── Fabric overlay: preserve new texture colours ──
            # No colour transfer (it maps new fabric back to original palette).
            # Light edge feather only — no Poisson (MIXED_CLONE restores original texture).
            print("[SDXL Export] Fabric mode: skipping colour transfer and Poisson blend")
            final_np = composited_np
        else:
            # ── Furniture replacement: match scene illumination ──
            # 2a. Colour transfer on MASKED region only — leaves rest of scene untouched.
            if config.color_transfer:
                print("[SDXL Export] Applying colour transfer to masked region...")
                # Extract masked patch, transfer, paste back
                result_patch = result_np.copy()
                transferred  = color_transfer_reinhard(result_np, scene_np, mask)
                # Only replace pixels inside the mask
                result_patch[mask] = transferred[mask]
                # Re-composite with the corrected patch
                composited_np = (
                    result_patch.astype(np.float32) * alpha +
                    scene_np.astype(np.float32)      * (1.0 - alpha)
                ).astype(np.uint8)
            
            # 2b. Poisson blending for lighting continuity at seam edges
            if config.poisson_blend:
                print("[SDXL Export] Applying Poisson blending...")
                composited_np = poisson_blend(composited_np, scene_np, mask)
            
            final_np = composited_np
        
        if config.esrgan_enabled:
            final_np = self._apply_esrgan(final_np, mask)
        
        final_rgb = cv2.cvtColor(final_np, cv2.COLOR_BGR2RGB)
        
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


def get_cancel_event() -> threading.Event:
    """Return the module-level cancel event for the current inference job."""
    return _cancel_event


def reset_cancel_event() -> None:
    """Clear the cancel event before starting a new inference job."""
    _cancel_event.clear()


def signal_cancel() -> None:
    """Signal the running inference thread to abort as soon as possible."""
    _cancel_event.set()
    print("[SDXL Export] Cancellation signalled.")
