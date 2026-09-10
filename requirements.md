# Requirements Specification: Field-Ready Explainable Agricultural Produce Quality Grading Prototype (Tomato MVP)

**Document Version:** 1.0.0 (50% Completion Phase)  
**Target Organization:** Small Agricultural Extension Services, Farmer Advisory Collectives, and Cooperative Societies  
**Target Produce Type:** Tomato (*Solanum lycopersicum* / *Lycopersicon esculentum*)

---

## 1. Problem Statement & Operational Context
Agricultural produce grading across varied rural microclimates suffers from significant inconsistency, subjective human bias, and friction between farmers and commercial buyers or cooperative aggregation hubs. Small agricultural organizations lack access to costly high-end optical sorters or cloud infrastructure requiring high bandwidth. This project delivers an explainable, lightweight, field-deployable visual grading prototype that standardizes produce grading, provides quantitative attribute metrics, explains every decision, flags low-confidence predictions, and preserves human oversight.

---

## 2. Functional Requirements (FR)

### FR1: Produce Image Ingestion
* **FR1.1:** The system shall accept single produce images via a web user interface (drag-and-drop / file browser) and a RESTful API (`POST /predict`).
* **FR1.2:** Supported formats shall include standard non-proprietary image formats: JPEG, PNG, and WebP.
* **FR1.3:** The system shall provide quick-load demonstration presets representing Grade A, Grade B, Grade C, and field failure/edge cases.

### FR2: Field Quality Image Validation & Preprocessing
* **FR2.1:** The system shall validate image illumination. If mean luminance is below 30.0 lux (underexposure) or above 250.0 lux (harsh glare), grading shall halt with a descriptive warning recommending human review.
* **FR2.2:** The system shall evaluate image focus using the Laplacian variance. If variance is below 5.0, the system shall halt grading with the message: *"Image quality insufficient for reliable grading (blurry image)."*
* **FR2.3:** The system shall detect produce items on the staging surface. If multiple significant produce items (>25% secondary area) are present, the system shall halt with: *"Multiple objects detected. Please provide one produce item."*
* **FR2.4:** The system shall standardize accepted images to $640 \times 640$ pixels and segment the foreground produce from the background.

### FR3: Quantitative Measurable Feature Extraction
* **FR3.1:** The system must not rely solely on black-box classifications. It shall extract 5 physical produce attributes:
  1. **Colour Quality:** Mean RGB, red-to-green lycopene ratio, and HSV hue standard deviation (uniformity).
  2. **Visible Damage:** Necrotic lesions, blossom-end rot, and surface growth cracks.
  3. **Shape Symmetry:** Circularity ($4\pi \cdot \text{Area} / \text{Perimeter}^2$) and bounding box aspect ratio.
  4. **Fruit Size:** Segmented pixel area, equivalent diameter, and fruit-to-frame area ratio.
  5. **Surface Defects:** High-frequency gradient variance (Sobel filter), corky scarring, and skin roughness.
* **FR3.2:** Each attribute shall be normalized to an interpretable 0 to 100 continuous score.

### FR4: Produce Quality Grading
* **FR4.1:** The system shall implement a transparent baseline rule grader using the scientifically weighted formula:
  $$\text{Quality Score} = (\text{Colour} \times 0.25) + (\text{Damage} \times 0.30) + (\text{Shape} \times 0.15) + (\text{Size} \times 0.15) + (\text{Surface Defects} \times 0.15)$$
* **FR4.2:** The system shall categorize produce into three commercial tiers:
  * **Grade A (Premium/Export):** Quality Score 80.0 – 100.0
  * **Grade B (Commercial Table):** Quality Score 60.0 – 79.9
  * **Grade C (Processing/Substandard):** Quality Score 0.0 – 59.9
* **FR4.3:** The system shall execute a trained Random Forest supervised classifier over the measurable visual features to generate the primary ML grade.

### FR5: Prediction Confidence Reporting
* **FR5.1:** The system shall output an explicit confidence score ($0.0$ to $1.0$ / $0\%$ to $100\%$) derived from the model's posterior probability distribution.
* **FR5.2:** A configurable confidence threshold (default: $75\%$) shall govern prediction acceptance.
* **FR5.3:** If confidence falls below the threshold, the system shall trigger the notification: *"Low confidence — Human review recommended."*

### FR6: Explainability & Decision Breakdown
* **FR6.1:** Every prediction shall include a plain-language explanation understandable by non-technical extension officers and farmers.
* **FR6.2:** The explanation shall highlight the positive quality drivers for Grade A assignments and explicitly identify the primary defect deductions causing downgrades to Grade B or C.
* **FR6.3:** The system shall present global feature importance percentages for the ML model (Damage: ~35%, Colour: ~28%, Defects: ~20%, Shape: ~10%, Size: ~7%).

### FR7: Human Review & Dispute Resolution
* **FR7.1:** The user interface shall provide an actionable **[Send for Human Review]** button on all prediction results.
* **FR7.2:** Extension workers shall have the ability to record an expert grade override along with qualitative field notes via `POST /review`.
* **FR7.3:** Review submissions shall be persistently logged to `data/human_reviews.csv` for audit trails and retraining data curation.

### FR8: Performance & Metric Dashboarding
* **FR8.1:** The system shall display dataset statistics, split sizes, and class balance.
* **FR8.2:** The system shall compare Baseline vs. ML accuracy, weighted F1-score, expert agreement rate, and disagreement rate side-by-side.
* **FR8.3:** The system shall display confusion matrix visualizations and error analysis breakdowns.

### FR9: Application Programming Interface (API)
* **FR9.1:** The system shall expose a lightweight REST API (`/predict`, `/health`, `/review`, `/metrics`) supporting headless integration with mobile field apps.

---

## 3. Non-Functional Requirements (NFR)

| ID | Requirement Area | Specification |
| :--- | :--- | :--- |
| **NFR1** | **User Interface Simplicity** | Intuitive single-screen workflow designed for non-technical field workers, featuring clear color-coded badges, visual progress bars, and zero complex mathematical jargon. |
| **NFR2** | **Low Computational Footprint** | System must run locally on standard commodity laptop hardware (Intel i3/i5 or AMD Ryzen, 4GB–8GB RAM, CPU-only inference, zero discrete GPU required). |
| **NFR3** | **Maintainability & Modularity** | Clean modular architecture separating data processing, feature engineering, baseline scoring, ML training, API, and UI into distinct Python files. |
| **NFR4** | **Explainability & Transparency** | Deterministic baseline rule formula available alongside ML model; feature importance and physical visual measurements visible for every prediction. |
| **NFR5** | **Privacy & Ethical Safety** | 100% non-identifiable produce dataset containing zero human faces, zero biometric markers, zero private data, and zero proprietary copyright material. |
| **NFR6** | **Latency & Response Time** | End-to-end inference (validation + feature extraction + inference + explanation) completed in under 150 milliseconds per image on CPU. |
| **NFR7** | **Human-in-the-Loop Safeguards** | System functions strictly as an advisory decision-support tool. It must never claim to replace agricultural experts and must mandate review on ambiguous cases. |
| **NFR8** | **Robust Error Handling** | Graceful error recovery: bad image formats, out-of-focus captures, and server disconnects return structured error messages rather than unhandled tracebacks. |
