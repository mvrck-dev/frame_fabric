"""
Post-processing utilities for VisionPhase pipeline.
Handles mask refinement, color transfer, Poisson blending,
and histogram matching for seamless compositing.
"""
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological dilation to expand mask edges.
    
    Args:
        mask: Binary mask (H, W) dtype bool or uint8
        kernel_size: Dilation kernel radius in pixels (3–8)
    
    Returns:
        Dilated binary mask (H, W) dtype bool
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1)
    )
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    return dilated > 127


def feather_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Gaussian feathering on mask edges for soft alpha transitions.
    
    Args:
        mask: Binary mask (H, W) dtype bool or uint8
        sigma: Gaussian blur sigma (1.0–5.0)
    
    Returns:
        Feathered alpha mask (H, W) dtype float32 in [0, 1]
    """
    mask_float = mask.astype(np.float32)
    if sigma > 0:
        feathered = gaussian_filter(mask_float, sigma=sigma)
    else:
        feathered = mask_float
    return np.clip(feathered, 0.0, 1.0)


def color_transfer_reinhard(source: np.ndarray, target: np.ndarray, 
                             mask: np.ndarray | None = None) -> np.ndarray:
    """
    Reinhard color transfer in LAB space.
    Transfers the color statistics of the target image onto the source,
    optionally restricted to the masked region.
    
    Args:
        source: Source image (H, W, 3) BGR uint8 — the product/replacement
        target: Target image (H, W, 3) BGR uint8 — the scene
        mask: Optional mask (H, W) indicating the region of interest in target
    
    Returns:
        Color-transferred source image (H, W, 3) BGR uint8
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if mask is not None:
        mask_bool = mask.astype(bool)
        target_pixels = target_lab[mask_bool]
    else:
        target_pixels = target_lab.reshape(-1, 3)
    
    source_pixels = source_lab.reshape(-1, 3)
    
    # Compute mean and std for each channel
    src_mean = source_pixels.mean(axis=0)
    src_std = source_pixels.std(axis=0) + 1e-6
    tgt_mean = target_pixels.mean(axis=0)
    tgt_std = target_pixels.std(axis=0) + 1e-6
    
    # Transfer: normalize source, apply target statistics
    result_lab = source_lab.copy()
    for i in range(3):
        result_lab[:, :, i] = (
            (result_lab[:, :, i] - src_mean[i]) * (tgt_std[i] / src_std[i]) + tgt_mean[i]
        )
    
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def poisson_blend(source: np.ndarray, target: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
    """
    Poisson / seamless clone blending for lighting continuity.
    
    Args:
        source: Source patch (H, W, 3) BGR uint8
        target: Target scene (H, W, 3) BGR uint8
        mask: Binary mask (H, W) uint8 (255 = foreground)
    
    Returns:
        Blended result (H, W, 3) BGR uint8
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    
    # Find center of the mask for seamlessClone
    moments = cv2.moments(mask_uint8)
    if moments["m00"] == 0:
        return target  # Empty mask, return target unchanged
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    center = (cx, cy)
    
    # Ensure source and target same size
    if source.shape != target.shape:
        source = cv2.resize(source, (target.shape[1], target.shape[0]))
    if mask_uint8.shape[:2] != target.shape[:2]:
        mask_uint8 = cv2.resize(mask_uint8, (target.shape[1], target.shape[0]))
    
    try:
        result = cv2.seamlessClone(
            source, target, mask_uint8, center, cv2.MIXED_CLONE
        )
        return result
    except cv2.error:
        # Fallback: simple alpha blending if seamlessClone fails
        alpha = feather_mask(mask, sigma=3.0)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        blended = (source.astype(np.float32) * alpha_3ch + 
                    target.astype(np.float32) * (1 - alpha_3ch))
        return blended.astype(np.uint8)


def histogram_match(source: np.ndarray, reference: np.ndarray,
                     mask: np.ndarray | None = None) -> np.ndarray:
    """
    Per-channel histogram matching of source to reference.
    
    Args:
        source: Source image (H, W, 3) uint8
        reference: Reference image (H, W, 3) uint8
        mask: Optional mask for reference region
    
    Returns:
        Histogram-matched source (H, W, 3) uint8
    """
    from skimage.exposure import match_histograms
    
    if mask is not None:
        # Only match against masked region of reference
        mask_bool = mask.astype(bool)
        ref_region = reference[mask_bool]
        # match_histograms needs 2D+ arrays, reshape
        # Use channel-by-channel matching
        result = source.copy()
        for c in range(3):
            src_ch = source[:, :, c].ravel()
            ref_ch = ref_region[:, c]
            matched = _match_channel_hist(src_ch, ref_ch)
            result[:, :, c] = matched.reshape(source.shape[:2])
        return result
    else:
        return match_histograms(source, reference, channel_axis=-1).astype(np.uint8)


def _match_channel_hist(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match histogram of a single channel."""
    src_values, src_unique_idx, src_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(reference, return_counts=True)
    
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]
    
    interp_values = np.interp(src_cdf, ref_cdf, ref_values)
    return interp_values[src_unique_idx].astype(np.uint8)


def crop_to_mask(image: np.ndarray, mask: np.ndarray, 
                  padding: int = 20) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Extract a tight bounding-box crop around the masked region.
    
    Args:
        image: Full image (H, W, 3)
        mask: Binary mask (H, W)
        padding: Extra pixels around the bounding box
    
    Returns:
        (cropped_image, (y1, y2, x1, x2)) bounding box coordinates
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any():
        return image, (0, image.shape[0], 0, image.shape[1])
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Add padding
    h, w = image.shape[:2]
    y1 = max(0, y1 - padding)
    y2 = min(h, y2 + padding + 1)
    x1 = max(0, x1 - padding)
    x2 = min(w, x2 + padding + 1)
    
    return image[y1:y2, x1:x2], (y1, y2, x1, x2)


def composite_product_into_mask(
    scene_bgr: np.ndarray,
    product_bgr: np.ndarray,
    mask: np.ndarray,
    feather_sigma: float = 2.0,
    do_color_transfer: bool = True,
) -> np.ndarray:
    """
    Advanced composite fallback for live preview / SDXL initializing.
    Preserves aspect ratio, removes white backgrounds from catalog products,
    centers the product in the mask bounding box, and blends seamlessly.
    """
    h, w = scene_bgr.shape[:2]
    
    # 1. Bounding box of the scene mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return scene_bgr
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    
    # 2. Extract product, isolate from generic white background
    gray = cv2.cvtColor(product_bgr, cv2.COLOR_BGR2GRAY)
    _, prod_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    p_rows = np.any(prod_mask, axis=1)
    p_cols = np.any(prod_mask, axis=0)
    if not p_rows.any():
        # Fallback if product is entirely white or threshold failed
        return scene_bgr
        
    py1, py2 = np.where(p_rows)[0][[0, -1]]
    px1, px2 = np.where(p_cols)[0][[0, -1]]
    
    product_cropped = product_bgr[py1:py2+1, px1:px2+1]
    prod_mask_cropped = prod_mask[py1:py2+1, px1:px2+1]
    
    # 3. Resize cropped product to fit into the scene bounding box while keeping aspect ratio
    p_h, p_w = product_cropped.shape[:2]
    scale = min(box_w / max(1, p_w), box_h / max(1, p_h))
    new_w = int(p_w * scale)
    new_h = int(p_h * scale)
    
    if new_w < 1 or new_h < 1:
        return scene_bgr
        
    product_resized = cv2.resize(product_cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    prod_mask_resized = cv2.resize(prod_mask_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 4. Center it inside the bounding box array
    pad_y = (box_h - new_h) // 2
    pad_x = (box_w - new_w) // 2
    
    product_padded = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    product_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = product_resized
    
    prod_mask_padded = np.zeros((box_h, box_w), dtype=np.uint8)
    prod_mask_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = prod_mask_resized
    
    # Optional color transfer to match scene illumination
    if do_color_transfer:
        # Transfer using only the foreground non-transparent pixels
        valid_mask = prod_mask_padded > 127
        product_padded = color_transfer_reinhard(product_padded, scene_bgr, mask)
    
    # 5. Place it correctly into the full frame context
    product_full = np.zeros_like(scene_bgr)
    # Be careful with +1 slicing
    product_full[y1:y1+box_h, x1:x1+box_w] = product_padded
    
    prod_mask_full = np.zeros(scene_bgr.shape[:2], dtype=np.float32)
    prod_mask_full[y1:y1+box_h, x1:x1+box_w] = prod_mask_padded.astype(np.float32) / 255.0
    
    # Feather the scene mask for soft edges
    scene_alpha = feather_mask(mask, sigma=feather_sigma)
    
    # Combine product alpha (background removed) and scene mask limit
    final_alpha = np.minimum(scene_alpha, prod_mask_full)
    alpha_3ch = np.stack([final_alpha] * 3, axis=-1)
    
    # Final blend
    result = (product_full.astype(np.float32) * alpha_3ch + 
              scene_bgr.astype(np.float32) * (1 - alpha_3ch))
    
    return result.astype(np.uint8)
