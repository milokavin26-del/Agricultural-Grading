"""
API Unit and Integration Tests (using standard unittest)
--------------------------------------------------------
Tests /health, /predict (Grade A, Grade B, Grade C, and edge cases), and /review.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import app

class TestProduceGradingAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_predict_grade_a(self):
        img_path = os.path.join(BASE_DIR, "data", "images", "test", "TOM_A_104.jpg")
        with open(img_path, "rb") as f:
            response = self.client.post("/predict", files={"file": ("test_a.jpg", f, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["grade"], "A")
        self.assertGreaterEqual(data["quality_score"], 80.0)
        self.assertGreaterEqual(data["confidence"], 0.75)
        self.assertFalse(data["human_review_required"])
        self.assertIn("explanation", data)
        self.assertIn("attributes", data)

    def test_predict_grade_b(self):
        img_path = os.path.join(BASE_DIR, "data", "images", "test", "TOM_B_104.jpg")
        with open(img_path, "rb") as f:
            response = self.client.post("/predict", files={"file": ("test_b.jpg", f, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["grade"], "B")
        self.assertTrue(60.0 <= data["quality_score"] < 80.0)

    def test_predict_grade_c(self):
        img_path = os.path.join(BASE_DIR, "data", "images", "test", "TOM_C_104.jpg")
        with open(img_path, "rb") as f:
            response = self.client.post("/predict", files={"file": ("test_c.jpg", f, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["grade"], "C")
        self.assertLess(data["quality_score"], 60.0)

    def test_predict_poor_lighting_edge_case(self):
        img_path = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_poor_lighting_01.jpg")
        with open(img_path, "rb") as f:
            response = self.client.post("/predict", files={"file": ("edge_light.jpg", f, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "FLAGGED_FOR_REVIEW")
        self.assertTrue(data["human_review_required"])
        self.assertIn("poor lighting", data["warning"].lower())

    def test_predict_multiple_objects_edge_case(self):
        img_path = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_multiple_objects_01.jpg")
        with open(img_path, "rb") as f:
            response = self.client.post("/predict", files={"file": ("edge_multi.jpg", f, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "FLAGGED_FOR_REVIEW")
        self.assertTrue(data["human_review_required"])
        self.assertIn("multiple objects", data["warning"].lower())

    def test_predict_blurry_image_edge_case(self):
        img_path = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_blurry_01.jpg")
        with open(img_path, "rb") as f:
            response = self.client.post("/predict", files={"file": ("edge_blur.jpg", f, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "FLAGGED_FOR_REVIEW")
        self.assertTrue(data["human_review_required"])
        self.assertIn("blurry", data["warning"].lower())

    def test_submit_review(self):
        payload = {
            "image_id": "TOM_B_104",
            "system_grade": "B",
            "expert_override_grade": "B",
            "officer_id": "OFFICER_PATEL",
            "comments": "Confirmed commercially acceptable Grade B produce."
        }
        response = self.client.post("/review", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")

if __name__ == "__main__":
    unittest.main()
