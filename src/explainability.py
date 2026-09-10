"""
Explainability Module for Agricultural Produce Grading (Tomato MVP)
-------------------------------------------------------------------
Translates raw model features and attribute scores into clear, human-readable explanations
tailored for field agricultural extension officers and farmers.
Provides:
1. Attribute breakdown summaries
2. Root-cause defect explanations (primary negative/positive drivers)
3. Model feature importance mapping
4. Human review trigger warnings
"""

from typing import Dict, Any, List

DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # Configurable: below 75% triggers human review

def generate_prediction_explanation(
    grade: str,
    confidence: float,
    attributes: Dict[str, float],
    feature_importances: Dict[str, float] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Generates structured, human-readable explanations for produce grading.
    
    Args:
        grade: Predicted grade ('A', 'B', or 'C')
        confidence: Model prediction confidence (0.0 - 1.0)
        attributes: Dictionary of 5 measured produce attributes (0 - 100)
        feature_importances: Global feature importance percentages
        confidence_threshold: Threshold below which human review is mandatory
    
    Returns:
        Dictionary containing summary text, bullet factors, feature importance, and review flag.
    """
    human_review_required = confidence < confidence_threshold
    
    col = attributes.get("colour", 50.0)
    dmg = attributes.get("damage", 50.0)
    shp = attributes.get("shape", 50.0)
    siz = attributes.get("size", 50.0)
    defects = attributes.get("surface_defects", 50.0)
    
    # Identify positive and negative drivers
    negative_drivers = []
    positive_drivers = []
    
    if dmg < 65.0:
        negative_drivers.append("visible damage or necrotic lesions")
    elif dmg >= 85.0:
        positive_drivers.append("low visible damage")
        
    if col < 65.0:
        negative_drivers.append("uneven color / incomplete ripening")
    elif col >= 85.0:
        positive_drivers.append("good color consistency and ripeness")
        
    if defects < 65.0:
        negative_drivers.append("surface scabbing or blemishes")
    elif defects >= 85.0:
        positive_drivers.append("smooth, unblemished skin")
        
    if shp < 65.0:
        negative_drivers.append("irregular shape / asymmetry")
    elif shp >= 85.0:
        positive_drivers.append("well-formed symmetrical shape")
        
    if siz < 65.0:
        negative_drivers.append("non-standard fruit size")
    elif siz >= 85.0:
        positive_drivers.append("standard market size")

    # Formulate main narrative explanation
    if grade == "A":
        pos_text = " and ".join(positive_drivers[:2]) if positive_drivers else "high overall visual uniformity"
        summary = f"Grade A was assigned because the produce has {pos_text}."
    elif grade == "B":
        if negative_drivers:
            neg_text = " and ".join(negative_drivers[:2])
            summary = f"Grade B was assigned. The main factors reducing quality from Grade A were {neg_text}."
        else:
            summary = "Grade B was assigned reflecting standard commercial table grade with minor variations."
    else: # Grade C
        if negative_drivers:
            neg_text = ", ".join(negative_drivers[:3])
            summary = f"Grade C was assigned primarily due to severe deductions in: {neg_text}."
        else:
            summary = "Grade C was assigned due to severe sub-standard visual defects."

    # Review warning text
    warning = None
    if human_review_required:
        warning = f"Low confidence ({confidence*100:.1f}% < {confidence_threshold*100:.0f}%) — Human review recommended."

    # Attribute qualitative labels
    def score_to_label(val: float) -> str:
        if val >= 80.0:
            return "Good / High"
        elif val >= 60.0:
            return "Moderate / Fair"
        else:
            return "Low / Deficient"

    attribute_labels = {
        "Colour quality": score_to_label(col),
        "Visible damage": "Low" if dmg >= 80 else ("Moderate" if dmg >= 60 else "High"),
        "Shape": score_to_label(shp),
        "Size": "Standard" if siz >= 75 else ("Acceptable" if siz >= 60 else "Irregular"),
        "Surface defects": "Low" if defects >= 80 else ("Moderate" if defects >= 60 else "Severe")
    }

    # Format simplified feature importance (if available)
    simplified_importance = {}
    if feature_importances:
        # Group into 5 user-friendly attributes
        simplified_importance = {
            "Damage": round(feature_importances.get("damage", 0.35) * 100, 1),
            "Colour": round(feature_importances.get("colour", 0.28) * 100, 1),
            "Surface Defects": round(feature_importances.get("surface_defects", 0.20) * 100, 1),
            "Shape": round(feature_importances.get("shape", 0.10) * 100, 1),
            "Size": round(feature_importances.get("size", 0.07) * 100, 1)
        }

    return {
        "summary": summary,
        "warning": warning,
        "human_review_required": human_review_required,
        "attribute_labels": attribute_labels,
        "attribute_scores": {
            "colour": col,
            "damage": dmg,
            "shape": shp,
            "size": siz,
            "surface_defects": defects
        },
        "feature_importance": simplified_importance
    }
