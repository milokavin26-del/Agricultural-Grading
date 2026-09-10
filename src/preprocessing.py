"""
Preprocessing and Image Validation Module for Produce Quality Grading (Tomato MVP)
----------------------------------------------------------------------------------
Handles image input validation (edge case detection: blur, poor lighting, multiple objects)
and produce segmentation into foreground masks and standardized color spaces.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Quality thresholds for field edge-case detection
MIN_LUMINANCE = 30.0         # Below this is considered poor/underexposed lighting
MAX_LUMINANCE = 250.0        # Above this is overexposed/blown out
MIN_LAPLACIAN_VAR = 5.0      # Below this is heavily blurred (defocused/motion blur)
MIN_OBJECT_AREA_RATIO = 0.04 # Produce must occupy at least 4% of the frame
MAX_SECONDARY_OBJ_RATIO = 0.25 # If a 2nd object is >25% the size of main object -> Multiple objects

class ValidationError(Exception):
    """Raised when an image fails field validation checks."""
    def __init__(self, message: str, failure_code: str):
        super().__init__(message)
        self.message = message
        self.failure_code = failure_code

def load_image(image_input) -> np.ndarray:
    """
    Loads an image from file path or byte buffer into BGR numpy array.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {image_input}")
        return img
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

def validate_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Performs field quality validation checks:
    1. Poor lighting (too dark or extreme glare)
    2. Blurriness (Laplacian variance)
    3. Multiple objects detection
    4. Produce presence
    
    Returns validation info dict or raises ValidationError.
    """
    if image is None or image.size == 0:
        raise ValidationError("Empty image provided.", "EMPTY_IMAGE")
        
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Lighting Check (Mean Luminance)
    mean_luminance = float(np.mean(gray))
    if mean_luminance < MIN_LUMINANCE:
        raise ValidationError(
            f"Low image quality / poor lighting (mean luminance: {mean_luminance:.1f} < {MIN_LUMINANCE}). Human review recommended.",
            "POOR_LIGHTING"
        )
    if mean_luminance > MAX_LUMINANCE:
        raise ValidationError(
            f"Image overexposed / severe glare (mean luminance: {mean_luminance:.1f} > {MAX_LUMINANCE}). Human review recommended.",
            "OVEREXPOSED"
        )
        
    # 2. Blurriness Check (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < MIN_LAPLACIAN_VAR:
        raise ValidationError(
            f"Image quality insufficient for reliable grading (blurry image, focus variance: {laplacian_var:.1f} < {MIN_LAPLACIAN_VAR}).",
            "BLURRY_IMAGE"
        )
        
    # 3. Produce Detection & Multiple Objects Check
    # Segment potential produce using color saturation & Otsu thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    
    # Tomatoes have strong saturation compared to neutral staging trays/burlap/wood
    _, thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up with morphological opening & closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(h * w)
    
    significant_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area / image_area >= MIN_OBJECT_AREA_RATIO:
            significant_contours.append((area, c))
            
    if not significant_contours:
        raise ValidationError(
            "No produce detected in image. Please ensure the produce is centered on the grading surface.",
            "NO_PRODUCE_DETECTED"
        )
        
    significant_contours.sort(key=lambda x: x[0], reverse=True)
    
    # Check for multiple produce items
    if len(significant_contours) > 1:
        main_area = significant_contours[0][0]
        second_area = significant_contours[1][0]
        if (second_area / main_area) >= MAX_SECONDARY_OBJ_RATIO:
            raise ValidationError(
                f"Multiple objects detected ({len(significant_contours)} items). Please provide one produce item.",
                "MULTIPLE_OBJECTS"
            )
            
    return {
        "valid": True,
        "mean_luminance": mean_luminance,
        "laplacian_var": laplacian_var,
        "main_contour": significant_contours[0][1],
        "produce_area": significant_contours[0][0],
        "image_area": image_area
    }

def preprocess_and_segment(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
    """
    Standardizes image size and extracts segmented foreground mask and ROI.
    """
    # 1. Validate
    val_info = validate_image(image)
    
    # 2. Resize to standard resolution
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    h, w = target_size
    
    # 3. Create precise foreground mask
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Threshold based on color saturation and brightness
    _, mask_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_clean = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest connected component (main produce item)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        largest_cnt = None
    else:
        largest_cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        
    # Masked foreground image
    fg = cv2.bitwise_and(resized, resized, mask=mask)
    
    # Bounding box
    if largest_cnt is not None:
        x, y, bw, bh = cv2.boundingRect(largest_cnt)
    else:
        x, y, bw, bh = 0, 0, w, h
        
    return {
        "image": resized,
        "mask": mask,
        "foreground": fg,
        "contour": largest_cnt,
        "bbox": (x, y, bw, bh),
        "validation_info": val_info
    }
