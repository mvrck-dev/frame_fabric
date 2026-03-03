from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import base64
import numpy as np

from app.core.cv_pipeline import get_sam_engine

router = APIRouter()

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Receives an image, loads it into the global SAM engine to prepare for fast clicks.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Load image into SAM engine (This is the slow part, ~2 seconds)
        engine = get_sam_engine()
        engine.set_image(pil_image)
        
        return JSONResponse(content={
            "status": "success", 
            "message": "Image encoded successfully.",
            "width": pil_image.width,
            "height": pil_image.height
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment")
async def segment_point(x: float = Form(...), y: float = Form(...), width: float = Form(...), height: float = Form(...)):
    """
    Receives click coordinates, scales them to the original image dimensions, 
    and returns a base64 encoded mask image.
    """
    engine = get_sam_engine()
    
    if engine.current_image_shape is None:
        raise HTTPException(status_code=400, detail="No image currently loaded in engine.")
        
    # Scale coordinates if the frontend sent relative/scaled coords
    # Note: Frontend must send the displayed width/height of the image element
    orig_h, orig_w = engine.current_image_shape
    
    scale_x = orig_w / float(width)
    scale_y = orig_h / float(height)
    
    actual_x = int(float(x) * scale_x)
    actual_y = int(float(y) * scale_y)
    
    try:
        # Generates a boolean mask (H, W)
        mask = engine.predict_mask(actual_x, actual_y)
        
        # Convert boolean mask to an RGBA image for frontend rendering
        # Making foreground pure white with 60% opacity, background transparent
        rgba_mask = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        rgba_mask[mask, :] = [168, 85, 247, 150] # Purple accent color with alpha
        
        mask_image = Image.fromarray(rgba_mask, mode="RGBA")
        
        # Encode to base64
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse(content={
            "status": "success",
            "mask_base64": f"data:image/png;base64,{img_str}"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
