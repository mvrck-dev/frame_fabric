"""
Fabric API Routes — Generation and Application.

Key design decisions:
- generate_fabrics: uses lightweight SD 1.5, returns quickly
- apply_fabric:    SDXL can take 3-10 minutes on first run (model download).
                   We fire it in a background thread and expose a poll endpoint
                   so the frontend never needs to guess a timeout.
- cancel_fabric:   signals the running thread to abort cleanly.
- fabric_status:   the frontend polls this every few seconds to get progress.
"""
import os
import io
import uuid
import base64
import hashlib
import threading
from urllib.parse import urlparse
from typing import Optional

import anyio
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from app.core.fabric_engine import get_fabric_engine
from app.core.cv_pipeline import get_sam_engine
from app.core.sdxl_export import get_export_pipeline, reset_cancel_event, signal_cancel, get_cancel_event

router = APIRouter()

# ── Directory for generated fabric swatches ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FABRIC_DIR = os.path.normpath(
    os.path.join(_BASE_DIR, "../../../public/assets/fabrics")
)
os.makedirs(_FABRIC_DIR, exist_ok=True)


# ── Per-job state (one active fabric job at a time) ──
_job_lock = threading.Lock()
_active_job: Optional[dict] = None  # {"id", "thread", "result_queue", "status", "error"}


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Generate Swatches ──

@router.post("/generate_fabrics")
async def generate_fabrics(
    prompt: str = Form("luxury velvet"),
    count: int = Form(4)
):
    """
    Generate fabric texture swatches using the fast SD 1.5 pipeline.
    Returns quickly (10-30 seconds). Does NOT use SDXL.
    """
    engine = get_fabric_engine()
    try:
        swatches = await anyio.to_thread.run_sync(
            lambda: engine.generate_swatches(prompt, count)
        )

        results = []
        for i, img in enumerate(swatches):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            md5_hash = hashlib.md5(img_bytes).hexdigest()
            filename = f"{md5_hash}.png"
            filepath = os.path.join(_FABRIC_DIR, filename)

            if not os.path.exists(filepath):
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

            url = f"http://localhost:8000/assets/fabrics/{filename}"
            results.append({
                "id": str(i),
                "name": f"{prompt} {i+1}",
                "thumbnail_url": url,
                "full_url": url,
                "base64": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
            })

        return JSONResponse(content={"status": "success", "fabrics": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Apply Fabric (async background job) ──

@router.post("/apply_fabric")
async def apply_fabric(
    fabric_url: str = Form(...),
    prompt: str = Form("")
):
    """
    Start an SDXL fabric application job in the background.
    Returns a job_id immediately so the frontend can poll /api/fabric_status/{job_id}.
    
    This avoids the timeout problem: SDXL can take 3-10 min on first run (model download).
    The frontend should poll rather than holding an open HTTP connection.
    """
    global _active_job

    sam_engine = get_sam_engine()

    # Guard: image state must exist
    if sam_engine.current_image is None or sam_engine.accumulated_mask is None:
        raise HTTPException(
            status_code=400,
            detail="Session state lost. Please re-upload your image and re-click the furniture."
        )

    if not sam_engine.accumulated_mask.any():
        raise HTTPException(
            status_code=400,
            detail="No furniture selected. Click on furniture in the canvas first."
        )

    # Cancel any currently running job before starting a new one
    with _job_lock:
        if _active_job is not None and _active_job.get("status") == "running":
            signal_cancel()
            _active_job["thread"].join(timeout=5)

    # Resolve the fabric image from disk (avoid self-deadlock via HTTP)
    parsed_url = urlparse(fabric_url)
    filename = os.path.basename(parsed_url.path)
    filepath = os.path.join(_FABRIC_DIR, filename)

    if os.path.exists(filepath):
        fabric_pil = Image.open(filepath).convert("RGB")
    else:
        # Fallback for external URLs (should be rare)
        try:
            import requests as _requests
            print(f"[Fabric API] Fetching external fabric: {fabric_url}")
            resp = _requests.get(fabric_url, timeout=10)
            resp.raise_for_status()
            fabric_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not load fabric image: {e}")

    # Snapshot the SAM state so the background thread has stable references
    scene_snapshot = sam_engine.current_image.copy()
    mask_snapshot = sam_engine.accumulated_mask.copy()

    # Build the effective prompt — be explicit about texture change
    fabric_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
    effective_prompt = prompt or (
        "close-up photorealistic furniture covered in new fabric textile, "
        "detailed fabric texture visible on surface, high quality material, "
        "studio lighting, 8k sharp"
    )
    
    # Sanity-check mask size before firing SDXL
    mask_pct = mask_snapshot.sum() / mask_snapshot.size
    if mask_pct < 0.003:
        raise HTTPException(
            status_code=400,
            detail=f"Selected region is too small ({mask_snapshot.sum()} px, {mask_pct:.2%}). "
                   "Please click directly on the furniture to select a larger area."
        )

    job_id = uuid.uuid4().hex

    # Reset cancel event for the new job
    reset_cancel_event()
    cancel_event = get_cancel_event()

    def _run():
        try:
            export_pipeline = get_export_pipeline()
            print(f"[Fabric API] Mask coverage: {mask_snapshot.sum()} px "
                  f"({mask_snapshot.sum() / mask_snapshot.size:.2%} of image)")
            raw_result = export_pipeline.export(
                scene_image=scene_snapshot,
                product_image=fabric_pil,
                mask=mask_snapshot,
                category="furniture",
                prompt=effective_prompt,
                cancel_event=cancel_event,
                fabric_mode=True,   # preserve new fabric colours
            )
            # Encode base64 here in the thread so the poll endpoint is instant
            b64 = _image_to_base64_png(raw_result)
            with _job_lock:
                if _active_job and _active_job["id"] == job_id:
                    _active_job["status"] = "done"
                    _active_job["result_base64"] = b64
            print(f"[Fabric API] Job {job_id} completed, result encoded ({len(b64)//1024} KB)")
        except RuntimeError as e:
            msg = str(e)
            with _job_lock:
                if _active_job and _active_job["id"] == job_id:
                    if "cancelled" in msg.lower():
                        _active_job["status"] = "cancelled"
                    else:
                        _active_job["status"] = "error"
                        _active_job["error"] = msg
            print(f"[Fabric API] Job {job_id} runtime error: {msg}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            with _job_lock:
                if _active_job and _active_job["id"] == job_id:
                    _active_job["status"] = "error"
                    _active_job["error"] = str(e)
            print(f"[Fabric API] Job {job_id} failed: {e}")

    thread = threading.Thread(target=_run, daemon=True)

    with _job_lock:
        _active_job = {
            "id": job_id,
            "thread": thread,
            "status": "running",
            "result_base64": None,
            "error": None,
        }

    thread.start()
    print(f"[Fabric API] Job {job_id} started in background thread.")

    return JSONResponse(content={
        "status": "started",
        "job_id": job_id,
        "message": "SDXL generation started. Poll /api/fabric_status/{job_id} for completion."
    })


@router.get("/fabric_status/{job_id}")
async def fabric_status(job_id: str):
    """
    Poll endpoint. The worker thread writes its result directly into _active_job,
    so this is always a fast, non-blocking read.
    """
    global _active_job

    with _job_lock:
        job = _active_job
        if job is None or job["id"] != job_id:
            return JSONResponse(content={"status": "not_found"})
        status = job["status"]
        result_b64 = job.get("result_base64")
        error = job.get("error")

    if status == "running":
        return JSONResponse(content={"status": "running"})
    elif status == "done":
        return JSONResponse(content={"status": "done", "result_base64": result_b64})
    elif status == "cancelled":
        return JSONResponse(content={"status": "cancelled"})
    else:
        return JSONResponse(content={"status": "error", "detail": error or "Unknown error"})


@router.post("/cancel_fabric")
async def cancel_fabric():
    """
    Signal the currently running fabric application job to stop.
    """
    global _active_job

    with _job_lock:
        job = _active_job

    if job is None or job.get("status") != "running":
        return JSONResponse(content={"status": "no_active_job"})

    signal_cancel()
    print(f"[Fabric API] Cancel requested for job {job['id']}")
    return JSONResponse(content={"status": "cancelling", "job_id": job["id"]})
