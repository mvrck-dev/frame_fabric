import time
import gc
import torch
from PIL import Image
from app.core.fast_preview import get_fast_preview_engine

class FabricEngine:
    """
    Manages fabric generation. Now uses the lightweight FastPreview (SD 1.5)
    for rapid swatch generation to avoid the 20-minute SDXL wait times.
    """
    def __init__(self):
        pass

    def generate_swatches(self, prompt: str = "luxury velvet", num_samples: int = 4) -> list[Image.Image]:
        """Generate a set of fabric swatches using the fast preview pipeline."""
        engine = get_fast_preview_engine()
        
        full_prompt = f"seamless high quality {prompt} fabric texture, textile pattern, macro photography, 8k, professional lighting"
        
        print(f"[Fabric Engine] Generating {num_samples} swatches for: '{prompt}' using TURBO mode (SD 1.5)")
        start = time.time()
        
        try:
            # Batch inference: single GPU pass
            swatches = engine.generate_t2i(
                prompt=full_prompt,
                steps=15, 
                seed=42,
                num_samples=num_samples
            )
            
            print(f"[Fabric Engine] Generation complete in {time.time() - start:.2f}s")
            return swatches
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Singleton
_fabric_engine = None

def get_fabric_engine() -> FabricEngine:
    global _fabric_engine
    if _fabric_engine is None:
        _fabric_engine = FabricEngine()
    return _fabric_engine
