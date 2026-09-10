"""
Unit Tests for Preprocessing and Image Validation
-------------------------------------------------
Validates image loading, resolution handling, foreground segmentation,
and edge case failure detection (blur, poor lighting, multiple objects).
"""

import os
import sys
import unittest
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    load_image,
    validate_image,
    preprocess_and_segment,
    ValidationError
)

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.grade_a_path = os.path.join(BASE_DIR, "data", "images", "test", "TOM_A_104.jpg")
        self.poor_light_path = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_poor_lighting_01.jpg")
        self.blurry_path = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_blurry_01.jpg")
        self.multi_obj_path = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_multiple_objects_01.jpg")

    def test_load_valid_image(self):
        img = load_image(self.grade_a_path)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)

    def test_validate_valid_image(self):
        img = load_image(self.grade_a_path)
        val_info = validate_image(img)
        self.assertTrue(val_info["valid"])
        self.assertGreater(val_info["mean_luminance"], 30.0)
        self.assertGreater(val_info["laplacian_var"], 5.0)

    def test_fail_poor_lighting(self):
        img = load_image(self.poor_light_path)
        with self.assertRaises(ValidationError) as ctx:
            validate_image(img)
        self.assertEqual(ctx.exception.failure_code, "POOR_LIGHTING")
        self.assertIn("poor lighting", ctx.exception.message.lower())

    def test_fail_blurry_image(self):
        img = load_image(self.blurry_path)
        with self.assertRaises(ValidationError) as ctx:
            validate_image(img)
        self.assertEqual(ctx.exception.failure_code, "BLURRY_IMAGE")
        self.assertIn("blurry", ctx.exception.message.lower())

    def test_fail_multiple_objects(self):
        img = load_image(self.multi_obj_path)
        with self.assertRaises(ValidationError) as ctx:
            validate_image(img)
        self.assertEqual(ctx.exception.failure_code, "MULTIPLE_OBJECTS")
        self.assertIn("multiple objects", ctx.exception.message.lower())

    def test_preprocess_and_segment(self):
        img = load_image(self.grade_a_path)
        res = preprocess_and_segment(img, target_size=(640, 640))
        self.assertEqual(res["image"].shape, (640, 640, 3))
        self.assertEqual(res["mask"].shape, (640, 640))
        self.assertGreater(np.sum(res["mask"] > 0), 10000)
        self.assertIsNotNone(res["contour"])

if __name__ == "__main__":
    unittest.main()
