"""
Unit Tests for Baseline Rule-Based Produce Grader
-------------------------------------------------
Validates weighted scoring formula, grade boundaries, and deterministic rule behavior.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.baseline import calculate_baseline_grade, BASELINE_WEIGHTS

class TestBaselineGrader(unittest.TestCase):
    def test_weights_sum_to_one(self):
        total_weight = sum(BASELINE_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_high_quality_grade_a(self):
        attrs = {
            "colour": 92.0,
            "damage": 98.0,
            "shape": 90.0,
            "size": 85.0,
            "surface_defects": 94.0
        }
        res = calculate_baseline_grade(attrs)
        self.assertEqual(res["grade"], "A")
        self.assertGreaterEqual(res["quality_score"], 80.0)
        self.assertIn("Grade A", res["explanation"])

    def test_medium_quality_grade_b(self):
        attrs = {
            "colour": 70.0,
            "damage": 68.0,
            "shape": 75.0,
            "size": 72.0,
            "surface_defects": 65.0
        }
        res = calculate_baseline_grade(attrs)
        self.assertEqual(res["grade"], "B")
        self.assertTrue(60.0 <= res["quality_score"] < 80.0)
        self.assertIn("Grade B", res["explanation"])

    def test_low_quality_grade_c(self):
        attrs = {
            "colour": 35.0,
            "damage": 25.0,
            "shape": 50.0,
            "size": 60.0,
            "surface_defects": 30.0
        }
        res = calculate_baseline_grade(attrs)
        self.assertEqual(res["grade"], "C")
        self.assertLess(res["quality_score"], 60.0)
        self.assertIn("Grade C", res["explanation"])

    def test_boundary_values(self):
        # Exactly 80.0 should yield Grade A
        attrs_80 = {
            "colour": 80.0, "damage": 80.0, "shape": 80.0, "size": 80.0, "surface_defects": 80.0
        }
        res_80 = calculate_baseline_grade(attrs_80)
        self.assertEqual(res_80["grade"], "A")
        self.assertEqual(res_80["quality_score"], 80.0)

        # Exactly 60.0 should yield Grade B
        attrs_60 = {
            "colour": 60.0, "damage": 60.0, "shape": 60.0, "size": 60.0, "surface_defects": 60.0
        }
        res_60 = calculate_baseline_grade(attrs_60)
        self.assertEqual(res_60["grade"], "B")
        self.assertEqual(res_60["quality_score"], 60.0)

if __name__ == "__main__":
    unittest.main()
