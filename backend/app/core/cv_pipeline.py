import os
import time
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import gc

class SAMInferenceEngine:
    """
    A persistent engine that loads the SAM model once and keeps it in memory
    for fast interactive API endpoints.
    """
    def __init__(self, model_type="vit_b"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[SAM Engine] Initializing. Hardware acceleration: {self.device}")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_type == "vit_h":
            chkpt = "sam_vit_h_4b8939.pth"
        elif model_type == "vit_b":
            chkpt = "sam_vit_b_01ec64.pth"
        else:
            raise ValueError(f"Unknown SAM model type: {model_type}")
            
        # script_dir is visionphase/backend/app/core. We need to go up 3 levels to visionphase/models
        checkpoint_path = os.path.join(script_dir, "../../../models/segmentation", chkpt)
        
        print(f"[SAM Engine] Loading {model_type} weights from {checkpoint_path}...")
        start = time.time()
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        print(f"[SAM Engine] Loaded in {time.time() - start:.2f}s")
        
        self.current_image_shape = None

    def set_image(self, image: Image.Image):
        """Processes and embeds the image into the SAM predictor."""
        print("[SAM Engine] Encoding new image...")
        start = time.time()
        
        # Convert PIL to numpy (RGB)
        image_np = np.array(image.convert("RGB"))
        self.current_image_shape = image_np.shape[:2]
        
        self.predictor.set_image(image_np)
        
        print(f"[SAM Engine] Image encoded in {time.time() - start:.2f}s")
        

    def predict_mask(self, x: int, y: int):
        """Generates a segmentation mask based on a single click coordinate."""
        if self.current_image_shape is None:
            raise ValueError("No image has been set. Call set_image() first.")
            
        print(f"[SAM Engine] Predicting mask for click ({x}, {y})...")
        start = time.time()
        
        input_point = np.array([[x, y]])
        input_label = np.array([1]) # Foreground point
        
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

engine = None

def get_sam_engine():
    global engine
    if engine is None:
        # Defaulting to vit_b for interactivity
        engine = SAMInferenceEngine(model_type="vit_b")
    return engine
