# Produce Quality Grading: 50% MVP Evaluation Report

## 1. Executive Summary
This evaluation rigorously benchmarks the deterministic **Baseline Rule Grader** against the supervised **Random Forest ML Model** on a held-out test split of 54 structured tomato produce samples (18 Grade A, 18 Grade B, 18 Grade C) staged under varied microclimate conditions (5 lighting styles, 4 background trays).

---

## 2. Quantitative Performance Comparison

| Metric | Baseline Rule Grader | ML Model (Random Forest) | MVP Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **92.59%** | **100.00%** | $\ge 80.0\%$ | **MET** |
| **Weighted Precision** | 0.930 | 1.000 | $\ge 0.750$ | **MET** |
| **Weighted Recall** | 0.926 | 1.000 | $\ge 0.750$ | **MET** |
| **Weighted F1-Score** | **0.926** | **1.000** | $\ge 0.750$ | **MET** |
| **Expert Agreement Rate** | **92.59%** | **100.00%** | $\ge 80.0\%$ | **MET** |
| **Disagreement Rate** | **7.41%** | **0.00%** | < Baseline | **MET** |
| **Average Prediction Confidence** | N/A (Linear Rule) | **97.99%** | $\ge 75.0\%$ | **MET** |
| **Low-Confidence Flags (<75%)** | 0 | **2** | Informative | **FLAGGED** |

---

## 3. Disagreement and Dispute Reduction Analysis
* **Core Problem Addressed:** High human grader variance and grading disputes between farmers and extension officers.
* **Baseline Disagreement:** The baseline linear scoring formula disagreed with reference labels in **7.41%** of test cases due to rigid boundary cutoff penalties on minor natural shape and coloration variations.
* **ML Model Agreement:** The Random Forest achieved **100.00%** expert agreement, reducing grading discrepancies significantly on the test split.
* **Organizational Dispute Reality Check:** *Actual organisational dispute reduction cannot yet be measured in the field. Prototype-level disagreement against expert reference labels is used as a proxy.*

---

## 4. Feature Importance Breakdown
From the Random Forest ensemble (100 estimators, max depth 6), the top decision-driving visual attributes are:

1. **Damage (Rot / Necrotic Lesions):** **13.6%**
2. **Colour (Ripeness / Red-Green Ratio):** **53.8%**
3. **Surface Defects (Scars / Roughness):** **6.7%**
4. **Shape (Circularity / Symmetry):** **25.9%**
5. **Size (Area Standardization):** **0.1%**

---

## 5. Failure and Edge Cases Summary
| Edge Case | Failure Mode | Trigger Condition | System Action | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Case 1: Poor Lighting** | Underexposure / Heavy Shadow | Mean luminance < 30.0 lux | Halts grading; flags for human review | **PASSED** |
| **Case 2: Multiple Objects** | Cluttered staging tray | > 1 significant contour detected | Rejects image; requests single produce | **PASSED** |
| **Case 3: Blurry Image** | Defocus / Motion blur | Laplacian focus variance < 5.0 | Rejects image; prompts re-capture | **PASSED** |

---

## 6. Confusion Matrices
```
Baseline Rule Grader:
           Pred A   Pred B   Pred C
Actual A:    18      0       0       
Actual B:    2       16      0       
Actual C:    0       2       16      

Random Forest ML Model:
           Pred A   Pred B   Pred C
Actual A:    18      0       0       
Actual B:    0       18      0       
Actual C:    0       0       18      
```
