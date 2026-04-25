"""
CLIP-based zero-shot classifier for segmented regions.
Given a binary mask + original image, crops the masked area and
classifies it against a configurable label set.
"""
import time
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from app.core.config import get_config


class CLIPClassifier:
    """
    Lazy-loaded CLIP model for zero-shot classification of masked regions.
    Not loaded at startup to conserve VRAM alongside SAM.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
    
    def _ensure_loaded(self):
        """Load CLIP weights on first use."""
        if self.model is not None:
            return
        
        config = get_config()
        model_name = config.clip_model
        
        print(f"[CLIP Classifier] Loading {model_name}...")
        start = time.time()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        print(f"[CLIP Classifier] Loaded in {time.time() - start:.2f}s on {self.device}")
    
    def classify(self, image: Image.Image, mask: np.ndarray, 
                 top_k: int = 3) -> list[dict]:
        """
        Classify the masked region of an image using zero-shot CLIP.
        
        Args:
            image: Full PIL Image (RGB)
            mask: Binary mask (H, W) bool — defines region of interest
            top_k: Number of top predictions to return
        
        Returns:
            List of {"label": str, "confidence": float} sorted by confidence
        """
        self._ensure_loaded()
        config = get_config()
        labels = config.class_labels
        
        # Crop the masked region for better classification
        cropped = self._crop_masked_region(image, mask)
        
        # Prepare text prompts (Ensemble)
        templates = [
            "a photo of a {label} in an interior room",
            "the surface of a {label}",
            "a {label} in a home",
            "a close up photo of a {label}",
            "a clean view of a {label}",
        ]
        
        all_probs = []
        
        # Run CLIP inference for each template and average results
        with torch.no_grad():
            for template in templates:
                text_prompts = [template.format(label=label) for label in labels]
                inputs = self.processor(
                    text=text_prompts,
                    images=cropped,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]  # (num_labels,)
                probs = logits.softmax(dim=-1).cpu().numpy()
                all_probs.append(probs)
        
        # Average probabilities across templates
        avg_probs = np.mean(all_probs, axis=0)
        
        # Build sorted results
        results = [
            {"label": labels[i], "confidence": round(float(avg_probs[i]), 4)}
            for i in range(len(labels))
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        top = results[:top_k]
        print(f"[CLIP Classifier] Top prediction: {top[0]['label']} ({top[0]['confidence']:.1%})")
        
        return top
    
    def _crop_masked_region(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Extract a tight crop of the masked region from the image.
        Uses bounding box with some context padding.
        """
        img_np = np.array(image)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any():
            return image  # Fallback: use full image
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        # Dynamic padding: scale context inversely with mask size
        # Larger regions (ceilings/walls) get less padding to avoid distractors
        h, w = img_np.shape[:2]
        mask_area_frac = mask.sum() / (h * w)
        
        # Scale padding from 5% (large regions) to 30% (small objects)
        pad_scale = 0.30 - (min(0.25, mask_area_frac) * 1.0) 
        pad_scale = max(0.05, pad_scale)
        
        pad_y = int((y2 - y1) * pad_scale)
        pad_x = int((x2 - x1) * pad_scale)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y + 1)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x + 1)
        
        cropped = img_np[y1:y2, x1:x2]
        return Image.fromarray(cropped)


# ── Global singleton ──
_classifier: CLIPClassifier | None = None


def get_classifier() -> CLIPClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CLIPClassifier()
    return _classifier
