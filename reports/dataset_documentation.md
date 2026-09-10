# Ethical Dataset Documentation: Controlled Produce Quality Dataset (Tomato MVP)

## 1. Dataset Overview
This dataset was purposefully created to establish a rigorous, reproducible, and ethical benchmark for visual agricultural produce quality grading. The dataset focuses on **Tomato** (*Solanum lycopersicum* / *Lycopersicon esculentum*), a globally vital horticultural crop with high economic sensitivity to post-harvest sorting and grading disputes.

* **Total Images Generated:** 369 images
  * Core Training Split: 252 images (70%)
  * Core Validation Split: 54 images (15%)
  * Core Held-Out Test Split: 54 images (15%)
  * Dedicated Edge & Failure Test Suite: 9 images
* **Class Balance:** Exactly 120 images for Grade A, 120 images for Grade B, 120 images for Grade C, ensuring zero class imbalance bias during training.
* **Resolution:** Standardized $640 \times 640$ pixels, 24-bit RGB.

---

## 2. Image Creation Methodology & Controlled Staging
The dataset was produced using a deterministic, physics-informed procedural image generation engine (`src/generate_dataset.py`) simulating real-world agricultural staging conditions.

Each produce sample incorporates:
1. **Biological Morphology:** Elliptical cross-sections with variable aspect ratio ($0.75 - 1.30$), circularity ($0.70 - 0.96$), apical depressions, and green sepals/calyx at the pedicel attachment point.
2. **3D Diffuse and Specular Shading:** Physically calibrated Lambertian reflectance models with directional illumination vectors and ground-plane contact drop shadows.
3. **Ripening & Pigmentation Dynamics:**
   * *Grade A:* Homogeneous deep red / lycopene-rich pigmentation with uniform hue distribution.
   * *Grade B:* Heterogeneous color maturation (turning/breaker stage, yellowish-orange shoulder patches, and slight green shoulder near calyx).
   * *Grade C:* Immature green patches, mottled discoloration, and unmarketable blotchy ripening.
4. **Pathological and Mechanical Defect Injection:**
   * *Blossom-End Rot:* Sunken, dark necrotic calcium-deficiency lesions positioned on the fruit's apical end.
   * *Growth Cracking:* Radial skin ruptures resulting from rapid moisture fluctuations during fruit development.
   * *Surface Scabbing & Russeting:* Corky epidermal scar tissue and micro-abrasions simulated with localized pigmentation drops and texture variance.

---

## 3. Microclimate & Environmental Conditions
To ensure the machine learning prototype learns robust produce-intrinsic features rather than background or lighting artifacts, every sample was rendered under combinations of realistic environmental conditions:

### Lighting Variations
1. `diffuse_daylight`: Neutral daylight illumination under standard field shade tents ($5500\text{K}$).
2. `direct_sunlight`: Warm high-intensity sunlight ($3200\text{K}$ golden spectrum) with pronounced specular highlights.
3. `greenhouse_filtered`: Soft, high-humidity diffuse lighting with subtle green tint typical of agricultural polyhouse shade netting.
4. `shade_overcast`: Cool overcast sky ($6500\text{K}$) with reduced color saturation.
5. `indoor_fluorescent`: Cool-white artificial illumination ($4000\text{K}$) representative of indoor cooperative packing sheds.

### Staging Background Variations
1. `neutral_gray_tray`: Standard neutral plastic inspection tray (RGB ~ $190, 192, 195$) with bevel border.
2. `wooden_grading_bench`: Traditional rural packing bench with natural horizontal wood grain and warm timber hues.
3. `white_inspection_mat`: High-contrast laboratory grading surface with subtle 60px measurement grid lines.
4. `burlap_fabric`: Textured jute sack material representing field harvesting bags with woven crosshatch texture.

---

## 4. Expert Grading & Reference Annotation Protocol
Reference grades were assigned according to international agricultural standards (OECD / UNECE Tomato Standards FFV-37):

* **Grade A (Premium Quality):** Expert Score $80.0 - 100.0$. Superior quality, uniform deep crimson color, highly symmetrical circular shape, firm skin, zero or negligible surface blemishes ($< 1\%$ total area).
* **Grade B (Commercial Table Quality):** Expert Score $60.0 - 79.9$. Good commercial marketability. Slight defects in shape or minor turning coloration permitted; minor corky scars or light russeting up to $10\%$ of surface area permitted.
* **Grade C (Substandard / Processing Quality):** Expert Score $0.0 - 59.9$. Produce exhibiting severe defects: blossom-end rot, growth cracks, severe scabbing, or mottled unripeness. Unmarketable as fresh table produce; relegated to industrial paste processing or livestock feed.

Annotations were logged into `data/annotations.csv` across three simulated agronomist IDs (`EXT_AGRONOMIST_01`, `EXT_AGRONOMIST_02`, `EXT_AGRONOMIST_03`) and supervised by `CHIEF_QUALITY_OFFICER`.

---

## 5. Failure & Edge Case Suite
To ensure field safety and prevent erroneous grading under adverse conditions, 9 dedicated failure test samples were created:
1. `EDGE_LIGHT_01 - 03` (**Poor Lighting**): Extreme underexposure ($< 30.0$ lux mean luminance) where camera sensors capture insufficient photon data for color/damage discrimination.
2. `EDGE_MULTI_01 - 03` (**Multiple Objects**): Two to three produce items simultaneously resting in frame, violating the single-item inspection protocol.
3. `EDGE_BLUR_01 - 03` (**Blurry Image**): Defocus and severe motion blur ($< 5.0$ Laplacian focus variance) rendering surface textures unmeasurable.

---

## 6. Ethical & Privacy Safeguards
1. **Zero Personally Identifiable Information (PII):** The dataset contains strictly agricultural objects resting on inanimate grading surfaces. No human faces, skin, clothing, hands, or identifiers are present.
2. **Zero Biometric or Surveillance Risks:** No geolocation tags, device serial numbers, or biometric metadata are embedded.
3. **Intellectual Property Safety:** The dataset was generated completely in-house without scraping copyrighted stock libraries or unauthorized farm records.
4. **Transparent Documentation:** All reference labels are explicitly identified as project-created reference standards, avoiding misleading claims of organic field sampling.

---

## 7. Metadata Schema (`data/metadata.csv`)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `image_id` | String | Unique specimen identifier (e.g., `TOM_A_001`, `EDGE_LIGHT_01`) |
| `image_path` | String | Relative filepath to image artifact (`images/train/...`) |
| `produce_type` | String | Agricultural crop name (`Tomato`) |
| `grade` | String | Quality classification (`A`, `B`, `C`, or `EDGE_CASE`) |
| `expert_score` | Float | Continuous ground-truth score ($0.0 - 100.0$) |
| `colour_score` | Float | Calibrated visual colour quality ($0.0 - 100.0$) |
| `damage_score` | Float | Calibrated pathological damage rating ($0.0 - 100.0$) |
| `shape_score` | Float | Calibrated geometric symmetry rating ($0.0 - 100.0$) |
| `size_score` | Float | Calibrated commercial size rating ($0.0 - 100.0$) |
| `surface_defect_score` | Float | Calibrated skin cleanliness rating ($0.0 - 100.0$) |
| `lighting_condition` | String | Environmental illumination simulation profile |
| `background_condition`| String | Staging bench surface type |
| `split` | String | Experimental partition (`train`, `val`, `test`, `edge_cases`) |
