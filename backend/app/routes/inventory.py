"""
Inventory API routes with IKEA Dataset mapping.
Serves product images organized by category from public/assets/ikea_dataset.
"""
import os
import json
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()

# Path to ikea_dataset directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IKEA_DIR = os.path.normpath(
    os.path.join(_BASE_DIR, "../../../public/assets/ikea_dataset")
)
_PRODUCTS_JSON = os.path.join(_IKEA_DIR, "products.json")

# Map of CLIP labels to the closest IKEA category
_CLASS_TO_CATEGORY = {
    "sofa": "sofas",
    "chair": "armchairs",
    "table": "dining_tables",
    "bed": "beds",
    "bedsheet": "beds",
    "mattress": "beds",
    "wardrobe": "wardrobes",
    "cabinet": "cabinets",
    "bookcase": "bookcases",
    "dresser": "dressers",
    "chest of drawers": "dressers",
    "ceiling lamp": "ceiling_lamps",
    "pendant light": "ceiling_lamps",
    "chandelier": "ceiling_lamps",
    "floor lamp": "floor_lamps",
    "tv": "tv_units",
    "desk": "desks",
    "shelf": "wall_shelves",
}


def _get_products() -> list:
    """Load the IKEA products.json."""
    if os.path.exists(_PRODUCTS_JSON):
        with open(_PRODUCTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@router.get("/inventory/{class_label}")
async def get_inventory_for_class(class_label: str, request: Request):
    """
    Return inventory items matching a CLIP class label.
    """
    class_key = class_label.lower()
    
    # Simple logic to handle refined labels (e.g., "ceiling surface" -> "ceiling")
    # or just use direct mapping.
    category = _CLASS_TO_CATEGORY.get(class_key)
    
    if not category:
        # Check if base word matches (e.g. "sofa surface" -> "sofa")
        for k in _CLASS_TO_CATEGORY.keys():
            if k in class_key:
                category = _CLASS_TO_CATEGORY[k]
                break
                
    if not category:
        # Return empty list instead of random "sofas"
        return JSONResponse(content={
            "status": "success",
            "class_label": class_label,
            "category": None,
            "items": [],
            "total": 0,
        })
        
    products = _get_products()
    items = []
    
    # Base URL for the FastAPI server
    base_url = f"{request.url.scheme}://{request.client.host}:{request.url.port}"
    if not request.url.port:
        port = 8000 # fallback
        base_url = f"http://localhost:{port}"

    for p in products:
        if p.get("category") == category:
            images = p.get("image_paths", [])
            if images:
                first_img = images[0]
                url = f"http://localhost:8000/assets/ikea_dataset/{first_img}"
                
                short_desc = p.get("short_desc", "")
                price = p.get("price", "")
                name = p.get("name", "")
                if "$" in name: 
                    name = name.split("$")[0].strip()
                
                items.append({
                    "name": name,
                    "filename": first_img,
                    "thumbnail_url": url,
                    "full_url": url,
                    "product_id": p.get("product_id"),
                    "price": price,
                    "short_desc": short_desc
                })
                
    return JSONResponse(content={
        "status": "success",
        "class_label": class_label,
        "category": category,
        "items": items,
        "total": len(items),
    })


@router.get("/inventory")
async def get_all_inventory():
    """Return all categories found in the IKEA dataset."""
    products = _get_products()
    categories = set(p.get("category") for p in products if p.get("category"))
    
    return JSONResponse(content={
        "status": "success",
        "categories": list(categories),
        "category_map": _CLASS_TO_CATEGORY,
    })


@router.post("/inventory/custom")
async def upload_custom_product(file: UploadFile = File(...)):
    """
    Upload a custom product image for use in preview/export.
    Returns a URL pointing to the temporary uploaded file.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    try:
        import uuid
        from PIL import Image
        import io
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save to a temp directory under our assets
        custom_dir = os.path.join(_IKEA_DIR, "custom")
        os.makedirs(custom_dir, exist_ok=True)
        
        filename = f"custom_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(custom_dir, filename)
        img.save(filepath, format="PNG")
        
        url = f"http://localhost:8000/assets/ikea_dataset/custom/{filename}"
        
        return JSONResponse(content={
            "status": "success",
            "name": file.filename or "Custom Upload",
            "thumbnail_url": url,
            "full_url": url,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

