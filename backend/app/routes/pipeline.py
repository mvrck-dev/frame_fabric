"""
VisionPhase API Routes — Full ML Pipeline.

Endpoints:
- POST /api/upload_image      — Upload + encode image in SAM
- POST /api/segment            — Multi-click segmentation (add/subtract/single) + CLIP class label
- POST /api/clear_masks        — Reset accumulated mask
- POST /api/preview            — Live GAN/composite preview
- POST /api/export             — SDXL inpainting export
- GET  /api/config             — Read pipeline config
- POST /api/config             — Update pipeline config
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import io
import base64
import anyio
import numpy as np
from PIL import Image

from app.core.cv_pipeline import get_sam_engine
from app.core.config import get_config

router = APIRouter()


def _mask_to_base64_png(mask: np.ndarray, color: tuple = (168, 85, 247, 150)) -> str:
    """Convert a boolean mask to a base64 RGBA PNG for frontend overlay."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, :] = list(color)
    
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _image_to_base64_png(image: Image.Image) -> str:
    """Convert a PIL image to a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Upload ──

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Receives an image, loads it into the global SAM engine to prepare for fast clicks."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        engine = get_sam_engine()
        engine.set_image(pil_image)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Image encoded successfully.",
            "width": pil_image.width,
            "height": pil_image.height,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Segment (Multi-click) ──

@router.post("/segment")
async def segment_point(
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    mode: str = Form("single"),  # "single" | "add" | "subtract"
):
    """
    Multi-click segmentation with Shift (add) / Ctrl (subtract) support.
    Returns the accumulated mask overlay + CLIP class label.
    """
    engine = get_sam_engine()
    
    if engine.current_image_shape is None:
        raise HTTPException(status_code=400, detail="No image currently loaded in engine.")
    
    # Scale coordinates from frontend display size to original image dimensions
    orig_h, orig_w = engine.current_image_shape
    scale_x = orig_w / float(width)
    scale_y = orig_h / float(height)
    actual_x = int(float(x) * scale_x)
    actual_y = int(float(y) * scale_y)
    
    try:
        # Predict and accumulate mask
        accumulated = engine.predict_and_accumulate(actual_x, actual_y, mode=mode)
        
        # Convert accumulated mask to overlay
        mask_base64 = _mask_to_base64_png(accumulated)
        
        # Run CLIP classification on the accumulated mask
        class_label = ""
        class_confidence = 0.0
        class_top3 = []
        
        if accumulated.any() and engine.current_image is not None:
            try:
                from app.core.classifier import get_classifier
                classifier = get_classifier()
                results = classifier.classify(engine.current_image, accumulated)
                if results:
                    class_label = results[0]["label"]
                    class_confidence = results[0]["confidence"]
                    class_top3 = results
            except Exception as e:
                print(f"[Segment] CLIP classification failed: {e}")
        
        return JSONResponse(content={
            "status": "success",
            "mask_base64": mask_base64,
            "class_label": class_label,
            "class_confidence": class_confidence,
            "class_top3": class_top3,
            "mask_pixel_count": int(accumulated.sum()),
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Clear Masks ──

@router.post("/clear_masks")
async def clear_masks():
    """Reset the accumulated mask to empty."""
    engine = get_sam_engine()
    engine.clear_masks()
    return JSONResponse(content={"status": "success", "message": "Masks cleared."})


# ── Live Preview ──

@router.post("/preview")
async def generate_preview(
    product_image: UploadFile = File(...),
    product_name: str = Form("")
):
    """
    Generate a live preview with the selected inventory product composited
    into the currently accumulated mask region.
    """
    engine = get_sam_engine()
    
    if engine.current_image is None or engine.accumulated_mask is None:
        raise HTTPException(status_code=400, detail="No image or mask available.")
    
    if not engine.accumulated_mask.any():
        raise HTTPException(status_code=400, detail="No mask region selected.")
    
    try:
        contents = await product_image.read()
        product_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        from app.core.fast_preview import get_fast_preview_engine
        preview_engine = get_fast_preview_engine()
        
        result = preview_engine.generate_preview(
            scene_image=engine.current_image,
            product_image=product_pil,
            mask=engine.accumulated_mask,
            product_name=product_name,
        )
        
        return JSONResponse(content={
            "status": "success",
            "preview_base64": _image_to_base64_png(result),
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── SDXL Export ──

@router.post("/export")
async def export_design(
    product_image: UploadFile = File(...),
    product_name: str = Form(""),
    category: str = Form(""),
    prompt: str = Form(""),
):
    """
    Run the full SDXL inpainting pipeline for high-quality export.
    """
    engine = get_sam_engine()
    
    if engine.current_image is None or engine.accumulated_mask is None:
        raise HTTPException(status_code=400, detail="No image or mask available.")
    
    if not engine.accumulated_mask.any():
        raise HTTPException(status_code=400, detail="No mask region selected.")
    
    try:
        contents = await product_image.read()
        product_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        from app.core.sdxl_export import get_export_pipeline, reset_cancel_event, get_cancel_event
        export_pipeline = get_export_pipeline()
        
        # Build a strong prompt with whatever info is available
        if not prompt:
            parts = []
            if product_name:
                parts.append(product_name)
            if category:
                parts.append(category)
            if parts:
                item_desc = ", ".join(parts)
                prompt = (
                    f"photorealistic interior design photo, {item_desc} standing on the floor, "
                    f"seamlessly integrated, realistic shadows, professional architectural lighting, "
                    f"sharp focus, 8k"
                )
            else:
                prompt = (
                    "photorealistic interior design photo, furniture seamlessly integrated into "
                    "the room, professional lighting, sharp focus, 8k"
                )
            
        # Arm cancel event so the thread can be interrupted if client disconnects
        reset_cancel_event()
        cancel_event = get_cancel_event()

        def _run_export():
            return export_pipeline.export(
                scene_image=engine.current_image,
                product_image=product_pil,
                mask=engine.accumulated_mask,
                category=category,
                prompt=prompt,
                cancel_event=cancel_event,
            )
            
        result = await anyio.to_thread.run_sync(_run_export)
        
        return JSONResponse(content={
            "status": "success",
            "export_base64": _image_to_base64_png(result),
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Config ──

@router.get("/config")
async def get_pipeline_config():
    """Return current pipeline configuration."""
    config = get_config()
    return JSONResponse(content=config.to_dict())


@router.post("/config")
async def update_pipeline_config(data: dict):
    """Update pipeline configuration from frontend settings panel."""
    config = get_config()
    config.update_from_dict(data)
    return JSONResponse(content={
        "status": "success",
        "config": config.to_dict(),
    })
