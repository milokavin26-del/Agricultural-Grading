"""
Measurable Feature Extraction Module for Produce Quality Grading (Tomato MVP)
-----------------------------------------------------------------------------
Extracts quantitative, explainable visual attributes:
1. Colour (RGB, HSV, red/green ratio, uniformity)
2. Size (area, bounding box, equivalent diameter)
3. Shape (circularity, aspect ratio, solidity)
4. Visible damage (necrotic rot, lesions, growth cracks)
5. Surface defects (scabbing, russeting, rough skin)
"""

import cv2
import numpy as np
import math
from typing import Dict, Any

def extract_measurable_features(preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts numerical and calibrated 0-100 scores from preprocessed produce data.
    """
    img = preprocessed_data["image"]
    mask = preprocessed_data["mask"]
    cnt = preprocessed_data["contour"]
    bbox = preprocessed_data["bbox"]
    x, y, bw, bh = bbox
    
    produce_pixels = mask > 0
    total_produce_area = float(np.sum(produce_pixels))
    if total_produce_area == 0:
        total_produce_area = 1.0

    # =========================================================================
    # 1. COLOUR ATTRIBUTES
    # =========================================================================
    b_vals = img[:, :, 0][produce_pixels].astype(np.float32)
    g_vals = img[:, :, 1][produce_pixels].astype(np.float32)
    r_vals = img[:, :, 2][produce_pixels].astype(np.float32)
    
    mean_b = float(np.mean(b_vals))
    mean_g = float(np.mean(g_vals))
    mean_r = float(np.mean(r_vals))
    
    # Red-to-Green ratio (primary indicator of lycopene ripeness)
    rg_ratio = float(mean_r / (mean_g + 1e-5))
    
    # HSV color statistics
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0][produce_pixels]
    s_vals = hsv[:, :, 1][produce_pixels]
    v_vals = hsv[:, :, 2][produce_pixels]
    
    mean_hue = float(np.mean(h_vals))
    std_hue = float(np.std(h_vals))      # Lower std = higher color uniformity
    mean_sat = float(np.mean(s_vals))
    mean_val = float(np.mean(v_vals))
    
    # Calibrated Colour Score (0 - 100):
    # Grade A tomato: mean_rg > 3.0, mean_r > 115, std_hue < 6.0 -> Score 85-98
    # Grade B tomato: mean_rg 1.8-3.0, turning color -> Score 60-79
    # Grade C tomato: mean_rg < 1.8, blotchy / green / mottled -> Score 15-58
    rg_component = np.clip((rg_ratio - 1.2) / 3.0 * 55.0, 0.0, 55.0)
    red_component = np.clip((mean_r - 70.0) / 60.0 * 25.0, 0.0, 25.0)
    uniformity_component = np.clip((12.0 - std_hue) / 12.0 * 20.0, 0.0, 20.0)
    colour_score = float(np.clip(rg_component + red_component + uniformity_component, 10.0, 99.0))

    # =========================================================================
    # 2. SIZE ATTRIBUTES
    # =========================================================================
    image_total_area = float(img.shape[0] * img.shape[1])
    area_ratio = float(total_produce_area / image_total_area)
    equiv_diameter = float(2.0 * math.sqrt(total_produce_area / math.pi))
    
    # Calibrated Size Score (0 - 100):
    # Standard table tomato occupies ~12% to 20% of standard 640x640 frame
    ideal_area_ratio = 0.14
    size_deviation = abs(area_ratio - ideal_area_ratio) / ideal_area_ratio
    raw_size_score = 95.0 - (size_deviation * 70.0)
    size_score = float(np.clip(raw_size_score, 30.0, 98.0))

    # =========================================================================
    # 3. SHAPE ATTRIBUTES
    # =========================================================================
    aspect_ratio = float(bw / max(bh, 1))
    
    if cnt is not None:
        perimeter = float(cv2.arcLength(cnt, True))
        contour_area = float(cv2.contourArea(cnt))
        if perimeter > 0:
            circularity = float(4.0 * math.pi * contour_area / (perimeter ** 2))
        else:
            circularity = 0.5
            
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(contour_area / max(hull_area, 1e-5))
    else:
        circularity = 0.8
        solidity = 0.85
        
    # Calibrated Shape Score (0 - 100):
    # Circularity + aspect ratio symmetry
    circ_comp = np.clip((circularity - 0.70) / 0.25 * 50.0, 0.0, 50.0)
    ar_comp = np.clip(50.0 - (abs(1.0 - aspect_ratio) * 180.0), 0.0, 50.0)
    shape_score = float(np.clip(circ_comp + ar_comp, 20.0, 98.0))

    # =========================================================================
    # 4. VISIBLE DAMAGE ATTRIBUTES (Blossom-end rot, necrosis, dark lesions)
    # =========================================================================
    # Isolate tomato body (excluding top stem/calyx area)
    cy_mid = y + int(bh * 0.28)
    body_mask = mask.copy()
    body_mask[:cy_mid, :] = 0
    
    b_body = img[:, :, 0][body_mask > 0].astype(np.float32)
    g_body = img[:, :, 1][body_mask > 0].astype(np.float32)
    r_body = img[:, :, 2][body_mask > 0].astype(np.float32)
    rg_body = r_body / (g_body + 1e-5)
    
    # Pathological lesions: low redness and low red-to-green ratio
    lesion = (rg_body < 1.4) & (r_body < 80.0)
    # Scars / corky tissue: moderate discoloration
    scar = (rg_body < 1.6) & (r_body < 120.0) & ~lesion
    
    total_body = float(max(len(r_body), 1))
    lesion_pct = float(np.sum(lesion) / total_body * 100.0)
    scar_pct = float(np.sum(scar) / total_body * 100.0)
    damage_ratio = float((lesion_pct + scar_pct) / 100.0)
    
    # Calibrated Damage Score (0 - 100):
    # Lesions penalize heavily (7.5x); minor scars penalize moderately (1.8x)
    raw_damage_score = 98.0 - (lesion_pct * 7.5 + scar_pct * 1.8)
    damage_score = float(np.clip(raw_damage_score, 10.0, 99.0))

    # =========================================================================
    # 5. SURFACE DEFECTS ATTRIBUTES (Scabbing, russeting, rough skin)
    # =========================================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    
    skin_gradients = grad_mag[body_mask > 0] if np.any(body_mask > 0) else grad_mag[produce_pixels]
    grad_std = float(np.std(skin_gradients)) if len(skin_gradients) > 0 else 5.0
    grad_mean = float(np.mean(skin_gradients)) if len(skin_gradients) > 0 else 8.0
    
    # Surface defect score (0 - 100):
    # Grade A: smooth skin, zero scars -> 94-98
    # Grade B: small scars, moderate texture -> 65-79
    # Grade C: heavy scarring / lesions -> 15-55
    raw_defect_score = 96.0 - (scar_pct * 2.8 + lesion_pct * 4.0)
    surface_defect_score = float(np.clip(raw_defect_score, 15.0, 98.0))

    # =========================================================================
    # BUNDLE FEATURES
    # =========================================================================
    raw_features = {
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "rg_ratio": round(rg_ratio, 3),
        "mean_hue": round(mean_hue, 2),
        "std_hue": round(std_hue, 2),
        "mean_sat": round(mean_sat, 2),
        "mean_val": round(mean_val, 2),
        "area_pixels": int(total_produce_area),
        "area_ratio": round(area_ratio, 4),
        "equiv_diameter": round(equiv_diameter, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "damage_ratio": round(damage_ratio, 4),
        "lesion_pct": round(lesion_pct, 2),
        "scar_pct": round(scar_pct, 2),
        "grad_std": round(grad_std, 2),
        "grad_mean": round(grad_mean, 2)
    }
    
    calibrated_attributes = {
        "colour": round(colour_score, 1),
        "damage": round(damage_score, 1),
        "shape": round(shape_score, 1),
        "size": round(size_score, 1),
        "surface_defects": round(surface_defect_score, 1)
    }
    
    return {
        "raw_features": raw_features,
        "attributes": calibrated_attributes
    }
