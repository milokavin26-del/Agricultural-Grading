"""
Prediction and Inference Pipeline Module for Produce Quality Grading (Tomato MVP)
---------------------------------------------------------------------------------
Orchestrates end-to-end inference:
Input (Image path / bytes / array)
  ↓
Image Validation (Lighting, blur, multiple objects)
  ↓
Preprocessing & Segmentation
  ↓
Measurable Feature Extraction
  ↓
Baseline Rule Model + ML Random Forest Model
  ↓
Confidence Calculation & Human Review Trigger
  ↓
Explainability Generation
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Union

from src.preprocessing import load_image, preprocess_and_segment, ValidationError
from src.feature_extraction import extract_measurable_features
from src.baseline import calculate_baseline_grade
from src.explainability import generate_prediction_explanation, DEFAULT_CONFIDENCE_THRESHOLD

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "quality_grading_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

class ProducePredictor:
    """
    Singleton-style predictor that caches trained Random Forest model and metadata.
    """
    def __init__(self, model_path: str = MODEL_PATH, meta_path: str = META_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.metadata = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
                
    def reload(self):
        self._load_artifacts()
        
    def predict(
        self,
        image_input: Union[str, bytes, np.ndarray],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Executes complete quality grading prediction pipeline on input image.
        """
        if self.model is None:
            self._load_artifacts()
            if self.model is None:
                raise RuntimeError("Model file not found. Please run src/train.py first.")

        # 1. Load image
        img = load_image(image_input)
        
        # 2. Image Validation & Preprocessing (with graceful handling of edge/failure cases)
        try:
            preprocessed = preprocess_and_segment(img)
        except ValidationError as e:
            # Field edge case triggered (poor lighting, blur, multiple objects)
            return {
                "status": "FLAGGED_FOR_REVIEW",
                "grade": "REVIEW_REQUIRED",
                "quality_score": 0.0,
                "confidence": 0.0,
                "attributes": {
                    "colour": 0.0,
                    "damage": 0.0,
                    "shape": 0.0,
                    "size": 0.0,
                    "surface_defects": 0.0
                },
                "explanation": f"Grading halted: {e.message}",
                "warning": e.message,
                "failure_code": e.failure_code,
                "human_review_required": True,
                "baseline_grade": "REVIEW_REQUIRED",
                "baseline_quality_score": 0.0,
                "feature_importance": {}
            }

        # 3. Measurable Feature Extraction
        features_data = extract_measurable_features(preprocessed)
        raw_feats = features_data["raw_features"]
        attrs = features_data["attributes"]

        # 4. Baseline Rule-Based Grading
        baseline_res = calculate_baseline_grade(attrs)

        # 5. ML Random Forest Prediction
        feature_cols = self.metadata.get("feature_cols", [
            "attr_colour", "attr_damage", "attr_shape", "attr_size", "attr_surface_defects",
            "mean_r", "mean_g", "mean_b", "rg_ratio", "mean_hue", "std_hue", "mean_sat", "mean_val",
            "area_ratio", "aspect_ratio", "circularity", "solidity", "damage_ratio", "grad_std"
        ])
        
        # Assemble feature vector matching trained model schema
        feat_vector_dict = {
            "attr_colour": attrs["colour"],
            "attr_damage": attrs["damage"],
            "attr_shape": attrs["shape"],
            "attr_size": attrs["size"],
            "attr_surface_defects": attrs["surface_defects"],
            **raw_feats
        }
        
        X_input = np.array([[feat_vector_dict.get(c, 0.0) for c in feature_cols]])
        
        # Inference
        pred_class = str(self.model.predict(X_input)[0])
        class_probs = self.model.predict_proba(X_input)[0]
        classes = list(self.model.classes_)
        confidence = float(np.max(class_probs))
        
        # Probabilities dict
        prob_dict = {classes[i]: round(float(class_probs[i]), 4) for i in range(len(classes))}

        # Continuous Quality Score (0 - 100) aligned with predicted grade
        # Uses probability weighting + baseline score refinement
        if pred_class == "A":
            grade_base = 80.0
            quality_score = min(99.0, grade_base + (attrs["colour"] * 0.10 + attrs["damage"] * 0.10))
        elif pred_class == "B":
            grade_base = 60.0
            quality_score = min(79.9, grade_base + (attrs["colour"] * 0.12 + attrs["damage"] * 0.12))
        else:
            quality_score = max(10.0, min(59.9, baseline_res["quality_score"]))
            
        quality_score = round(quality_score, 1)

        # 6. Explainability
        attr_importances = self.metadata.get("attribute_importances", {
            "damage": 0.35, "colour": 0.28, "surface_defects": 0.20, "shape": 0.10, "size": 0.07
        })
        
        explain_res = generate_prediction_explanation(
            grade=pred_class,
            confidence=confidence,
            attributes=attrs,
            feature_importances=attr_importances,
            confidence_threshold=confidence_threshold
        )

        return {
            "status": "SUCCESS",
            "grade": pred_class,
            "quality_score": quality_score,
            "confidence": round(confidence, 3),
            "attributes": attrs,
            "attribute_labels": explain_res["attribute_labels"],
            "explanation": explain_res["summary"],
            "warning": explain_res["warning"],
            "human_review_required": explain_res["human_review_required"],
            "class_probabilities": prob_dict,
            "feature_importance": explain_res["feature_importance"],
            "baseline_grade": baseline_res["grade"],
            "baseline_quality_score": baseline_res["quality_score"],
            "baseline_explanation": baseline_res["explanation"],
            "disagreement_with_baseline": (pred_class != baseline_res["grade"]),
            "raw_features": raw_feats
        }

# Global instance
_predictor = None

def get_predictor() -> ProducePredictor:
    global _predictor
    if _predictor is None:
        _predictor = ProducePredictor()
    return _predictor

def predict_produce(image_input, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    return get_predictor().predict(image_input, confidence_threshold=confidence_threshold)
