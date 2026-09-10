"""
Unit Tests for Machine Learning Random Forest Model & Explainability
--------------------------------------------------------------------
Validates model loading, schema alignment, prediction outputs, confidence thresholds,
and human-interpretable explanations.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import get_predictor, predict_produce
from src.explainability import generate_prediction_explanation

class TestMLModelAndExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.test_img_a = os.path.join(BASE_DIR, "data", "images", "test", "TOM_A_104.jpg")
        cls.test_img_b = os.path.join(BASE_DIR, "data", "images", "test", "TOM_B_104.jpg")
        cls.test_img_c = os.path.join(BASE_DIR, "data", "images", "test", "TOM_C_104.jpg")

    def test_model_loaded(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.metadata)
        self.assertIn("classes", self.predictor.metadata)
        self.assertEqual(self.predictor.metadata["classes"], ["A", "B", "C"])

    def test_predict_schema(self):
        res = predict_produce(self.test_img_a)
        self.assertIn("status", res)
        self.assertIn("grade", res)
        self.assertIn("quality_score", res)
        self.assertIn("confidence", res)
        self.assertIn("attributes", res)
        self.assertIn("explanation", res)
        self.assertIn("human_review_required", res)
        self.assertIn("feature_importance", res)

        # 5 required attributes
        attrs = res["attributes"]
        for k in ["colour", "damage", "shape", "size", "surface_defects"]:
            self.assertIn(k, attrs)
            self.assertTrue(0.0 <= attrs[k] <= 100.0)

    def test_explanation_generation_grade_a(self):
        exp = generate_prediction_explanation(
            grade="A",
            confidence=0.92,
            attributes={"colour": 90, "damage": 95, "shape": 90, "size": 85, "surface_defects": 92}
        )
        self.assertFalse(exp["human_review_required"])
        self.assertIsNone(exp["warning"])
        self.assertIn("Grade A", exp["summary"])

    def test_explanation_low_confidence_trigger(self):
        exp = generate_prediction_explanation(
            grade="B",
            confidence=0.62,
            attributes={"colour": 65, "damage": 65, "shape": 65, "size": 65, "surface_defects": 65},
            confidence_threshold=0.75
        )
        self.assertTrue(exp["human_review_required"])
        self.assertIsNotNone(exp["warning"])
        self.assertIn("Low confidence", exp["warning"])

if __name__ == "__main__":
    unittest.main()
