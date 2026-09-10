"""
Baseline Rule-Based Grading Module for Agricultural Produce (Tomato MVP)
------------------------------------------------------------------------
Implements the transparent, deterministic baseline produce quality grading algorithm.
Formula:
    Quality Score = (Colour * 0.25) + (Damage * 0.30) + (Shape * 0.15) + (Size * 0.15) + (Defects * 0.15)

Grade Assignment:
    80 - 100 : Grade A (Premium / Export Quality)
    60 - 79  : Grade B (Good / Standard Market Quality)
    0  - 59  : Grade C (Substandard / Processing / Rejected)
"""

from typing import Dict, Any

# Scientifically justified weighting scheme for tomato produce grading
BASELINE_WEIGHTS = {
    "damage": 0.30,          # 30% - Pathological rot / necrotic lesions destroy marketability immediately
    "colour": 0.25,          # 25% - Ripeness, carotenoid/lycopene uniformity, consumer appeal
    "surface_defects": 0.15, # 15% - Russeting, scabs, and micro-cracks reduce shelf life
    "shape": 0.15,           # 15% - Symmetry and circularity for crate packaging efficiency
    "size": 0.15             # 15% - Standardization for retail market classification
}

GRADE_THRESHOLDS = {
    "A": (80.0, 100.0),
    "B": (60.0, 79.99),
    "C": (0.0, 59.99)
}

def calculate_baseline_grade(attributes: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes deterministic baseline quality score and assigned grade.
    
    Args:
        attributes: Dictionary containing 'colour', 'damage', 'shape', 'size', 'surface_defects' (0-100).
    
    Returns:
        dict with:
            - grade: 'A', 'B', or 'C'
            - quality_score: float (0 - 100)
            - confidence: float (proxy confidence based on margin from decision thresholds)
            - attributes: input attributes dict
            - explanation: detailed rule-based justification
    """
    col = float(attributes.get("colour", 50.0))
    dmg = float(attributes.get("damage", 50.0))
    shp = float(attributes.get("shape", 50.0))
    siz = float(attributes.get("size", 50.0))
    defects = float(attributes.get("surface_defects", 50.0))
    
    # Weighted linear combination
    quality_score = (
        col * BASELINE_WEIGHTS["colour"] +
        dmg * BASELINE_WEIGHTS["damage"] +
        shp * BASELINE_WEIGHTS["shape"] +
        siz * BASELINE_WEIGHTS["size"] +
        defects * BASELINE_WEIGHTS["surface_defects"]
    )
    quality_score = round(max(0.0, min(100.0, quality_score)), 1)
    
    # Grade assignment
    if quality_score >= 80.0:
        grade = "A"
    elif quality_score >= 60.0:
        grade = "B"
    else:
        grade = "C"
        
    # Baseline distance-to-boundary pseudo confidence
    # If the score is far from the 60 and 80 boundaries, confidence is higher
    dist_to_80 = abs(quality_score - 80.0)
    dist_to_60 = abs(quality_score - 60.0)
    min_boundary_dist = min(dist_to_80, dist_to_60)
    
    # Range 0.60 to 0.95
    baseline_confidence = round(min(0.95, max(0.60, 0.65 + (min_boundary_dist / 20.0) * 0.30)), 2)

    # Explanation generation
    low_factors = []
    if dmg < 75.0:
        low_factors.append(f"visible damage (score: {dmg:.0f})")
    if col < 75.0:
        low_factors.append(f"color uniformity (score: {col:.0f})")
    if defects < 75.0:
        low_factors.append(f"surface defects (score: {defects:.0f})")
    if shp < 70.0:
        low_factors.append(f"shape irregularity (score: {shp:.0f})")
    if siz < 70.0:
        low_factors.append(f"size deviation (score: {siz:.0f})")

    if grade == "A":
        explanation = (
            f"Grade A was assigned because the produce achieved a high overall quality score of {quality_score}/100 "
            f"with excellent color consistency ({col:.0f}) and negligible damage ({dmg:.0f})."
        )
    elif grade == "B":
        if low_factors:
            explanation = (
                f"Grade B was assigned (quality score: {quality_score}/100). "
                f"The primary factors moderating the grade were: {', '.join(low_factors)}."
            )
        else:
            explanation = f"Grade B was assigned with a standard commercial score of {quality_score}/100."
    else:
        if low_factors:
            explanation = (
                f"Grade C was assigned (quality score: {quality_score}/100) due to severe deductions in: "
                f"{', '.join(low_factors)}."
            )
        else:
            explanation = f"Grade C was assigned due to overall low quality metrics ({quality_score}/100)."

    return {
        "grade": grade,
        "quality_score": quality_score,
        "confidence": baseline_confidence,
        "attributes": {
            "colour": col,
            "damage": dmg,
            "shape": shp,
            "size": siz,
            "surface_defects": defects
        },
        "explanation": explanation,
        "weights": BASELINE_WEIGHTS
    }
