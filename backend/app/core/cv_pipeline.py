"""
SAM-based segmentation engine for VisionPhase.
Supports multi-point selection with Shift (add) / Ctrl (subtract) modifiers,
accumulated mask management, and edge refinement (dilation + feathering).
"""
import os
import time
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

from app.core.config import get_config
from app.core.postprocess import dilate_mask, feather_mask


class SAMInferenceEngine:
    """
    A persistent engine that loads the SAM model once and keeps it in memory
    for fast interactive API endpoints. Supports multi-click mask accumulation.
    """
    
    def __init__(self, model_type="vit_b"):
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"[SAM Engine] Initializing. Hardware acceleration: {self.device}")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_type == "vit_h":
            chkpt = "sam_vit_h_4b8939.pth"
        elif model_type == "vit_b":
            chkpt = "sam_vit_b_01ec64.pth"
        else:
            raise ValueError(f"Unknown SAM model type: {model_type}")
        
        # script_dir is backend/app/core. We need to go up 3 levels to models/
        checkpoint_path = os.path.join(script_dir, "../../../models/segmentation", chkpt)
        
        print(f"[SAM Engine] Loading {model_type} weights from {checkpoint_path}...")
        start = time.time()
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        print(f"[SAM Engine] Loaded in {time.time() - start:.2f}s")
        
        # State
        self.current_image_shape: tuple[int, int] | None = None
        self.current_image: Image.Image | None = None  # Keep reference for classifier
        self.accumulated_mask: np.ndarray | None = None  # (H, W) bool
    
    def set_image(self, image: Image.Image):
        """Processes and embeds the image into the SAM predictor."""
        print("[SAM Engine] Encoding new image...")
        start = time.time()
        
        # Convert PIL to numpy (RGB)
        image_np = np.array(image.convert("RGB"))
        self.current_image_shape = image_np.shape[:2]
        self.current_image = image.copy()
        
        # Reset accumulated mask on new image
        self.accumulated_mask = np.zeros(self.current_image_shape, dtype=bool)
        
        self.predictor.set_image(image_np)
        
        print(f"[SAM Engine] Image encoded in {time.time() - start:.2f}s")
    
    def predict_mask(self, x: int, y: int) -> np.ndarray:
        """Generates a single segmentation mask based on a click coordinate."""
        if self.current_image_shape is None:
            raise ValueError("No image has been set. Call set_image() first.")
        
        print(f"[SAM Engine] Predicting mask for click ({x}, {y})...")
        start = time.time()
        
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # Foreground point
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Take the best scoring mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        print(f"[SAM Engine] Mask generated in {time.time() - start:.2f}s. Score: {scores[best_idx]:.2f}")
        return best_mask
    
    def predict_and_accumulate(self, x: int, y: int, 
                                mode: str = "add") -> np.ndarray:
        """
        Predict a mask and merge it with the accumulated mask.
        
        Args:
            x, y: Click coordinates in original image space
            mode: "add" (Shift+click, boolean OR) | "subtract" (Ctrl+click, boolean AND NOT)
                  | "single" (plain click, replace accumulated mask entirely)
        
        Returns:
            The updated accumulated mask (H, W) bool
        """
        new_mask = self.predict_mask(x, y)
        config = get_config()
        
        # Apply dilation to the raw mask
        new_mask = dilate_mask(new_mask, kernel_size=config.dilation_px)
        
        if mode == "add":
            # Shift+click: union
            self.accumulated_mask = self.accumulated_mask | new_mask
            print(f"[SAM Engine] Mask ADDED (union). Total mask pixels: {self.accumulated_mask.sum()}")
        elif mode == "subtract":
            # Ctrl+click: subtraction
            self.accumulated_mask = self.accumulated_mask & ~new_mask
            print(f"[SAM Engine] Mask SUBTRACTED. Total mask pixels: {self.accumulated_mask.sum()}")
        else:
            # Plain click: fresh single mask
            self.accumulated_mask = new_mask
            print(f"[SAM Engine] Mask REPLACED (single). Total mask pixels: {self.accumulated_mask.sum()}")
        
        return self.accumulated_mask
    
    def get_accumulated_mask(self) -> np.ndarray | None:
        """Return the current accumulated mask."""
        return self.accumulated_mask
    
    def get_feathered_mask(self) -> np.ndarray | None:
        """Return the accumulated mask with feathered edges (float32 alpha)."""
        if self.accumulated_mask is None:
            return None
        config = get_config()
        return feather_mask(self.accumulated_mask, sigma=config.feather_sigma)
    
    def clear_masks(self):
        """Reset accumulated mask to empty."""
        if self.current_image_shape is not None:
            self.accumulated_mask = np.zeros(self.current_image_shape, dtype=bool)
        print("[SAM Engine] Masks cleared.")


# ── Global singleton ──
engine: SAMInferenceEngine | None = None


def get_sam_engine() -> SAMInferenceEngine:
    global engine
    if engine is None:
        config = get_config()
        engine = SAMInferenceEngine(model_type=config.sam_model_type)
    return engine
