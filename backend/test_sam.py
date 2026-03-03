import time
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os

def test_sam():
    print("="*50)
    print("VISIONPHASE SAM INFERENCE BENCHMARK")
    print("="*50)
    
    print("1. Checking Hardware Acceleration...")
    # Check for Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon GPU (MPS) detected! Using hardware acceleration.")
    else:
        device = "cpu"
        print("⚠️ MPS not available. Falling back to CPU. This will be slow.")
        
    print("\n2. Loading SAM model (vit_h)...")
    start_load = time.time()
    
    # Ensure relative path works
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sam_checkpoint = os.path.join(script_dir, "../models/segmentation/sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    load_time = time.time() - start_load
    print(f"✅ Model loaded into {device.upper()} memory in {load_time:.2f} seconds.")
    
    print("\n3. Generating dummy image (1024x1024)...")
    dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    print("\n4. Encoding Image Data (Heaviest operation)...")
    start_set = time.time()
    predictor.set_image(dummy_image)
    set_time = time.time() - start_set
    print(f"✅ Image embedded & encoded in {set_time:.2f} seconds.")
    
    print("\n5. Running point-based mask prediction...")
    # Dummy point prompt
    input_point = np.array([[500, 500]])
    input_label = np.array([1])
    
    start_predict = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    predict_time = time.time() - start_predict
    print(f"✅ Mask prediction completed in {predict_time:.2f} seconds.")
    print(f"✅ Generated {masks.shape[0]} candidate masks. Best score: {scores[0]:.2f}")
    
    print("="*50)
    print(f"FAST INTERACTION TIME (Prediction only): {predict_time:.2f} seconds")
    print(f"TOTAL IMAGE PROCESSING TIME (Encode + Predict): {set_time + predict_time:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    test_sam()
