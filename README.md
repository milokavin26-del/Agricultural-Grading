# AgriGrade: Field-Ready Explainable Agricultural Produce Quality Grading Prototype (Tomato MVP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Phase: 50% MVP](https://img.shields.io/badge/Phase-50%25%20MVP-orange.svg)]()
[![Hardware: CPU Only](https://img.shields.io/badge/Hardware-Standard%20CPU-brightgreen.svg)]()

> **Project Target:** A lightweight, field-deployable prototype for agricultural extension teams advising farmers across varied rural microclimates, designed specifically to standardize produce quality grading, eliminate human grader disputes, and maintain explainable human-in-the-loop oversight.

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Produce Type Selection](#3-produce-type-selection)
4. [Ethical Dataset Creation & Annotation](#4-ethical-dataset-creation--annotation)
5. [Measurable Visual Produce Attributes](#5-measurable-visual-produce-attributes)
6. [Baseline Rule-Based Grading Algorithm](#6-baseline-rule-based-grading-algorithm)
7. [Machine Learning Random Forest Model](#7-machine-learning-random-forest-model)
8. [Data Preprocessing & Validation Pipeline](#8-data-preprocessing--validation-pipeline)
9. [Quantitative Model Evaluation & Comparison](#9-quantitative-model-evaluation--comparison)
10. [Dispute Reduction & Error Analysis](#10-dispute-reduction--error-analysis)
11. [Edge / Failure Case Demonstrations](#11-edge--failure-case-demonstrations)
12. [User Interface & Dashboard](#12-user-interface--dashboard)
13. [REST API Integration Stub](#13-rest-api-integration-stub)
14. [Cost, Maintenance & Limitations](#14-cost-maintenance--limitations)
15. [Quickstart & Execution Guide](#15-quickstart--execution-guide)
16. [Future Phase 2 Roadmap](#16-future-phase-2-roadmap)

---

## 1. Project Overview & Problem Statement

### The Problem
In agricultural supply chains, fresh produce quality grading at farmgate aggregation centres is largely manual, subjective, and prone to significant friction:
* **Inconsistency Across Microclimates:** Produce maturity and appearance vary with microclimate conditions (sunlight, elevation, rainfall), leading to unfair quality penalties.
* **Grading Disputes:** Smallholder farmers frequently dispute downgrades from commercial buyers who lack transparent, objective standards.
* **High Infrastructure Barrier:** Commercial optical sorters require multi-thousand dollar equipment, high-bandwidth cloud infrastructure, and dedicated ML engineers that small agricultural organizations cannot afford.

### The Solution
AgriGrade provides a field-ready, explainable quality grading prototype that:
1. Accepts a standard produce image taken under field conditions.
2. Extracts **5 quantitative, physically measurable visual attributes** (Colour, Damage, Shape, Size, and Surface Defects).
3. Evaluates quality using both a deterministic **Baseline Rule Grader** and an optimized **Random Forest ML Model**.
4. Generates a **plain-language explanation** detailing the primary positive drivers or negative defects.
5. Provides posterior **prediction confidence** and automatically flags low-confidence (<75%) or ambiguous cases for human agronomist review.
6. Runs 100% locally on standard commodity laptop hardware with zero cloud dependencies.

---

## 2. End-to-End System Architecture

```
User / Extension Worker (Smartphone / Laptop)
                 │
                 ▼
      [ User Interface / REST API ]
                 │
                 ▼
      [ Image Validation Gate ]
        ├─ Luminance Check (< 30 lux -> Flag Poor Lighting)
        ├─ Focus Check (Laplacian < 5.0 -> Flag Blurry)
        └─ Produce Count Check (> 1 object -> Flag Multi-Item)
                 │
                 ▼
      [ Preprocessing & Segmentation ]
        Standardize (640x640) → HSV & Otsu Masking → ROI Extraction
                 │
                 ▼
      [ Quantitative Feature Extraction ]
        1. Colour Quality (RGB, Lycopene R/G Ratio, Hue Std Dev)
        2. Visible Damage (Blossom-End Rot, Necrosis, Cracks)
        3. Shape Symmetry (Circularity, Aspect Ratio, Solidity)
        4. Fruit Size (Area Ratio, Equivalent Diameter)
        5. Surface Defects (Gradient Variance, Corky Scabs)
                 │
        ┌────────┴──────────────────────────┐
        ▼                                   ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│   Baseline Rule Model   │       │     ML Model Engine     │
│ Weighted Formula (0-100)│       │  Random Forest (N=100)  │
└─────────────────────────┘       └─────────────────────────┘
        │                                   │
        ▼                                   ▼
   Deterministic Grade                 Predicted Grade & Confidence
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
            [ Explainability Generator ]
        Positive Drivers / Root-Cause Downgrade Factors
                          │
                          ▼
            [ Confidence Threshold Gate ]
        Confidence >= 75%  ──►  Auto-Accept Suggestion
        Confidence <  75%  ──►  Mandate Extension Officer Review
                          │
                          ▼
            [ Audit & Retraining Log ]
             (data/human_reviews.csv)
```

---

## 3. Produce Type Selection
* **Produce Selected:** **Tomato** (*Solanum lycopersicum* / *Lycopersicon esculentum*).
* **Rationale:** Tomatoes are an economically critical horticultural staple exhibiting well-defined, measurable physical quality attributes:
  * Color transitions from chlorophyll green to lycopene deep crimson.
  * Symmetrical roundness essential for standard crate packaging.
  * Pronounced, distinct pathological defects (blossom-end rot, growth cracking, surface russeting).
* **Extensibility:** The codebase separates crop feature extraction from model inference. New produce profiles (e.g., Mango, Apple) can be configured by adding a new attribute scoring module without altering the core pipeline.

---

## 4. Ethical Dataset Creation & Annotation

### Dataset Specifications
* **Specimen Count:** 369 images total (120 Grade A, 120 Grade B, 120 Grade C, 9 Edge/Failure cases).
* **Partitions:** 70% Training (252), 15% Validation (54), 15% Held-Out Testing (54), plus 9 dedicated failure cases.
* **Controlled Staging Environments:**
  * *Lighting Profiles (5):* Diffuse daylight, direct sunlight, greenhouse filtered, shade overcast, indoor fluorescent.
  * *Staging Surfaces (4):* Neutral gray plastic tray, wooden packing bench, white laboratory inspection mat, burlap sack fabric.

### Ethical & Privacy Safeguards
1. **Zero Personally Identifiable Information (PII):** 100% non-identifiable produce specimens. Zero human faces, skin, clothing, hands, or biometric markers.
2. **Zero Copyright Encroachment:** Staged procedural generation pipeline creates fully owned, reproducible training data without unauthorized web scraping.
3. **Transparent Ground-Truth:** Clearly documented as project-created reference standards matching international OECD/UNECE grading guidelines.

---

## 5. Measurable Visual Produce Attributes

The system avoids black-box opacity by measuring five concrete agricultural parameters:

| Attribute | Measurement Method | Agricultural Significance | Score Range |
| :--- | :--- | :--- | :---: |
| **Colour Quality** | Mean RGB, Red-to-Green ratio ($R/G$), and HSV Hue variance. | Measures lycopene concentration, ripeness stage, and color uniformity. | $0 - 100$ |
| **Visible Damage** | Segmented dark necrotic lesion area and morphological black-hat crack detection. | Blossom-end rot, fungal lesions, and deep growth cracks that cause biological decay. | $0 - 100$ |
| **Shape Symmetry** | Circularity ($4\pi A / P^2$), Aspect Ratio ($W/H$), and Solidity ($A / A_{\text{hull}}$). | Packaging efficiency in standard transport crates and cosmetic market appeal. | $0 - 100$ |
| **Fruit Size** | Segmented pixel area, fruit-to-frame ratio, and equivalent diameter. | Size tiering for retail vs. processing market classification. | $0 - 100$ |
| **Surface Defects** | Sobel high-frequency gradient variance on body mask and localized scar areas. | Epidermal corking, russeting, and micro-abrasions that degrade shelf-life. | $0 - 100$ |

---

## 6. Baseline Rule-Based Grading Algorithm

A transparent, deterministic baseline was constructed prior to machine learning:

$$\text{Quality Score} = (\text{Colour} \times 0.25) + (\text{Damage} \times 0.30) + (\text{Shape} \times 0.15) + (\text{Size} \times 0.15) + (\text{Surface Defects} \times 0.15)$$

### Weight Selection Rationale
* **Damage (0.30, 30%):** Highest weight because rotting or deep cracks lead to total market rejection.
* **Colour (0.25, 25%):** Second highest because color reflects ripeness and flavor profile.
* **Defects (0.15, 15%):** Influences shelf-life and separates Grade A from Grade B.
* **Shape (0.15, 15%):** Governs packing crate density.
* **Size (0.15, 15%):** Retail standardization requirement.

### Grade Tiers
* **Grade A (Premium Quality):** Quality Score $\ge 80.0$
* **Grade B (Commercial Table):** $60.0 \le \text{Quality Score} < 80.0$
* **Grade C (Substandard / Processing):** $\text{Quality Score} < 60.0$

---

## 7. Machine Learning Random Forest Model
* **Model Class:** `RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=4, class_weight='balanced')`
* **Inputs:** 19 extracted visual features (5 calibrated attribute scores + 14 raw computer vision measurements).
* **Output:** Predicted Grade (`A`, `B`, `C`), class posterior probabilities, and continuous quality score.
* **Explainability Integration:** Gini impurity feature importances mapped to high-level visual attributes:
  * Damage: ~35%
  * Colour: ~28%
  * Surface Defects: ~20%
  * Shape: ~10%
  * Size: ~7%

---

## 8. Data Preprocessing & Validation Pipeline

Before inference, every image passes through an automated validation gate (`src/preprocessing.py`):
1. **Lighting Test:** Mean luminance must satisfy $30.0 \le \bar{Y} \le 250.0$.
2. **Focus Test:** Laplacian focus variance must satisfy $\sigma_{\text{Laplacian}}^2 \ge 5.0$.
3. **Item Isolation:** Foreground produce must occupy $\ge 4\%$ of the frame. Secondary contours must be $< 25\%$ of primary produce area.
4. **Standardization:** Image resized to $640 \times 640$ pixels; HSV Otsu segmentation isolates fruit body mask.

---

## 9. Quantitative Model Evaluation & Comparison

Evaluated on the held-out test partition ($N = 54$ samples; 18 Grade A, 18 Grade B, 18 Grade C) across varied microclimate lighting conditions:

| Evaluation Metric | Baseline Rule Grader | ML Model (Random Forest) | MVP Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **92.59%** | **100.00%** | $\ge 80.0\%$ | **MET** |
| **Weighted F1-Score** | 0.926 | **1.000** | $\ge 0.750$ | **MET** |
| **Weighted Precision** | 0.935 | **1.000** | $\ge 0.750$ | **MET** |
| **Weighted Recall** | 0.926 | **1.000** | $\ge 0.750$ | **MET** |
| **Expert Agreement Rate** | **92.59%** | **100.00%** | $\ge 80.0\%$ | **MET** |
| **Disagreement Rate** | **7.41%** | **0.00%** | Lower than Baseline | **MET** |
| **Average Prediction Confidence** | N/A (Rule) | **97.99%** | $\ge 75.0\%$ | **MET** |
| **Low-Confidence Flags (<75%)** | 0 | **2** | Low Count | **MET** |

---

## 10. Dispute Reduction & Error Analysis
* **Why Baseline Failed on 7.41% of Test Samples:** The rigid linear arithmetic cutoff ($<60.0$) over-penalized minor natural corky scarring on Grade B specimens that agronomists still accept commercially.
* **Why Random Forest Succeeded:** The ensemble learned non-linear boundaries that accommodate minor cosmetic blemishes when rot/damage is strictly absent.
* **Dispute Reduction Reality Check:** *Actual organisational dispute reduction cannot yet be measured in the field. Prototype-level disagreement against expert reference labels is used as a proxy.*
* **Error Analysis Records:** Documented in [reports/error_analysis.csv](reports/error_analysis.csv) with root-cause categorization and recommended remediation actions.

---

## 11. Edge / Failure Case Demonstrations

The system explicitly handles three critical field failure modes:

| Test Case | Scenario | Trigger Condition | System Action | UI / API Output |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1** | **Poor Lighting** | Mean luminance $< 30.0$ lux | Halts grading before inference | `"Low image quality / poor lighting. Human review recommended."` |
| **Case 2** | **Multiple Objects** | $>1$ produce contour detected | Rejects staging arrangement | `"Multiple objects detected. Please provide one produce item."` |
| **Case 3** | **Blurry Image** | Laplacian variance $< 5.0$ | Halts grading | `"Image quality insufficient for reliable grading (blurry image)."` |

---

## 12. User Interface & Dashboard

AgriGrade provides two complementary user interfaces:
1. **Interactive Streamlit Web Dashboard (`dashboard/app.py`):**
   * **Field Grader Tab:** Image uploader, quick-load demonstration presets (Grade A, B, C, and Edge Cases), live attribute progress bars, explanation callout, [Accept Grade] button, and [Send for Human Review] popup form.
   * **Metric Dashboard Tab:** Executive KPIs, comparative evaluation matrix, confusion matrix heatmaps, and feature importance charts.
   * **Error Analysis Tab:** Interactive table of `error_analysis.csv` and live human review audit logs.
2. **Standalone Zero-Dependency Web Page (`dashboard/index.html`):**
   * Modern, responsive single-file interface operating either in standalone client-side demo mode or connected to the local FastAPI backend.

---

## 13. REST API Integration Stub

A lightweight FastAPI service (`api/app.py`) enables headless integration with mobile field apps:

### `GET /health`
```json
{
  "status": "healthy"
}
```

### `POST /predict`
* **Input:** Form-data with file upload (`file: produce_image.jpg`).
* **Output:**
```json
{
  "status": "SUCCESS",
  "grade": "A",
  "quality_score": 99.0,
  "confidence": 1.0,
  "attributes": {
    "colour": 94.9,
    "damage": 98.0,
    "shape": 80.1,
    "size": 90.8,
    "surface_defects": 96.0
  },
  "explanation": "Grade A was assigned because the produce has low visible damage and good color consistency and ripeness.",
  "human_review_required": false,
  "warning": null,
  "baseline_grade": "A",
  "baseline_quality_score": 92.8
}
```

### `POST /review`
Records an agricultural extension officer's grade override or dispute resolution comment to `data/human_reviews.csv`.

---

## 14. Cost, Maintenance & Limitations
* **Hardware Requirement:** Low-cost entry-level laptop or Android smartphone ($250–$400 one-time). Zero cloud GPU required.
* **Maintenance:** Periodic seasonal retraining on new harvest varieties (~1 hour of CPU compute per season).
* **Limitations:** Currently calibrated for single round tomato varieties; evaluates visible front-facing produce surface; requires controlled staging surface.

---

## 15. Quickstart & Execution Guide

### 1. Installation
```powershell
# Clone or navigate to project directory
cd "c:\Users\KAVIN\Desktop\COE project"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated 16-Step Demonstration (CLI)
```powershell
python app.py --demo
```

### 3. Launch Interactive Streamlit Web UI & Dashboard
```powershell
streamlit run dashboard/app.py
```
*Opens automatically in browser at `http://localhost:8501`.*

### 4. Start REST API Server
```powershell
python api/app.py
```
*API available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.*

### 5. Run Unit & Integration Test Suite
```powershell
python -m unittest discover tests/
```

---

## 16. Future Phase 2 Roadmap
The remaining 50% of the project will focus on:
1. **Multi-Produce Expansion:** Adding calibrated profiles for Mango, Apple, Bell Pepper, and Citrus fruits.
2. **Multi-Angle 3D Capture:** Integrating rotating turntable or multi-camera rigs for $360^\circ$ surface defect detection.
3. **Mobile Edge APK:** Packaging feature extraction and Random Forest into an offline Android APK using ONNX runtime.
4. **Deep Learning Object Auto-Cropping:** Incorporating MobileNet/YOLOv8-nano for robust foreground cropping in cluttered field boxes.
5. **Real-World Pilot:** Partnering with rural extension offices to measure real-world dispute reduction against historical grading dispute records.
