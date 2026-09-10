"""
Ethical Synthetic Dataset Generator for Agricultural Produce Quality Grading (Tomato MVP)
-----------------------------------------------------------------------------------------
Generates an ethical, staged, non-identifiable produce image dataset under controlled
microclimate variations (lighting, backgrounds, defect types, and edge cases).
Zero human faces, zero PII, zero copyright-infringing content.
"""

import os
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

# Set deterministic seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
VAL_DIR = os.path.join(IMAGES_DIR, "val")
TEST_DIR = os.path.join(IMAGES_DIR, "test")
EDGE_DIR = os.path.join(IMAGES_DIR, "edge_cases")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, EDGE_DIR]:
    os.makedirs(d, exist_ok=True)

LIGHTING_CONDITIONS = [
    "diffuse_daylight",
    "direct_sunlight",
    "greenhouse_filtered",
    "shade_overcast",
    "indoor_fluorescent"
]

BACKGROUND_CONDITIONS = [
    "neutral_gray_tray",
    "wooden_grading_bench",
    "white_inspection_mat",
    "burlap_fabric"
]

def generate_background(bg_type, width=640, height=640):
    """Generates a realistic staging background for agricultural grading."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if bg_type == "neutral_gray_tray":
        base = np.full((height, width, 3), [190, 192, 195], dtype=np.uint8)
        noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        # Subtle rim bevel
        cv2.rectangle(img, (15, 15), (width - 15, height - 15), (170, 172, 175), 2)
        
    elif bg_type == "wooden_grading_bench":
        # Warm wood tone with grain
        base = np.full((height, width, 3), [150, 180, 210], dtype=np.uint8) # BGR
        y = np.linspace(0, 10 * math.pi, height)
        grain = (np.sin(y) * 12).astype(np.int16)
        for i in range(height):
            base[i, :, 0] = np.clip(base[i, :, 0] + grain[i], 0, 255)
            base[i, :, 1] = np.clip(base[i, :, 1] + grain[i] // 2, 0, 255)
        noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        
    elif bg_type == "white_inspection_mat":
        base = np.full((height, width, 3), [235, 238, 240], dtype=np.uint8)
        # Grid markings typical of laboratory produce inspection
        for x in range(40, width, 60):
            cv2.line(base, (x, 0), (x, height), (220, 223, 225), 1)
        for y in range(40, height, 60):
            cv2.line(base, (0, y), (width, y), (220, 223, 225), 1)
        noise = np.random.normal(0, 2, (height, width, 3)).astype(np.int16)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        
    elif bg_type == "burlap_fabric":
        base = np.full((height, width, 3), [120, 160, 190], dtype=np.uint8) # Jute/burlap
        # Crosshatch texture
        grid_x = (np.indices((height, width))[1] % 6 == 0).astype(np.int16) * 15
        grid_y = (np.indices((height, width))[0] % 6 == 0).astype(np.int16) * 15
        base = np.clip(base.astype(np.int16) + grid_x[:, :, None] - grid_y[:, :, None], 0, 255).astype(np.uint8)
        noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        
    return img

def render_single_produce(img, grade, center_x=320, center_y=320, radius_scale=1.0, is_edge_case=None):
    """
    Renders a physically accurate tomato with grading attributes onto the image.
    Returns: (img, ground_truth_attributes_dict)
    """
    h, w = img.shape[:2]
    
    # 1. Dimensions & Shape
    base_r = int(140 * radius_scale)
    # Grade shape characteristics
    if grade == "A":
        # Highly circular, uniform symmetry
        aspect_ratio = random.uniform(0.96, 1.04)
        rx = int(base_r * math.sqrt(aspect_ratio))
        ry = int(base_r / math.sqrt(aspect_ratio))
        shape_score = round(random.uniform(85, 98), 1)
        size_score = round(random.uniform(82, 96), 1)
    elif grade == "B":
        # Moderate asymmetry / slight flattening or elongation
        aspect_ratio = random.choice([random.uniform(0.85, 0.94), random.uniform(1.06, 1.15)])
        rx = int(base_r * math.sqrt(aspect_ratio))
        ry = int(base_r / math.sqrt(aspect_ratio))
        shape_score = round(random.uniform(64, 79), 1)
        size_score = round(random.uniform(62, 85), 1)
    else: # Grade C
        # Distinct irregular deformity or stunted/creased shape
        aspect_ratio = random.choice([random.uniform(0.72, 0.84), random.uniform(1.18, 1.35)])
        rx = int(base_r * math.sqrt(aspect_ratio))
        ry = int(base_r / math.sqrt(aspect_ratio))
        shape_score = round(random.uniform(35, 59), 1)
        size_score = round(random.uniform(40, 68), 1)
        
    # Drop shadow
    shadow_overlay = img.copy()
    cv2.ellipse(shadow_overlay, (center_x + 10, center_y + ry - 15), (int(rx * 1.08), int(ry * 0.35)), 
                0, 0, 360, (30, 30, 30), -1)
    shadow_overlay = cv2.GaussianBlur(shadow_overlay, (51, 51), 0)
    cv2.addWeighted(shadow_overlay, 0.45, img, 0.55, 0, img)

    # 2. Base Color & Ripening
    # BGR Color definitions
    # Tomato body rendering using radial    # 3D Normal approximation for sphere
    X_grid, Y_grid = np.meshgrid(np.arange(w), np.arange(h))
    dist_sq = ((X_grid - center_x) / max(rx, 1)) ** 2 + ((Y_grid - center_y) / max(ry, 1)) ** 2
    mask = dist_sq <= 1.0
    
    Z = np.zeros_like(dist_sq, dtype=np.float32)
    Z[mask] = np.sqrt(np.maximum(0.0, 1.0 - dist_sq[mask]))
    
    # Light source vector (from top-left and slightly towards viewer)
    lx, ly, lz = -0.4, -0.4, 0.8
    norm = math.sqrt(lx*lx + ly*ly + lz*lz)
    lx, ly, lz = lx/norm, ly/norm, lz/norm
    
    # Diffuse reflection: dot product of normal and light
    nx = np.zeros_like(Z)
    ny = np.zeros_like(Z)
    nx[mask] = (X_grid[mask] - center_x) / rx
    ny[mask] = (Y_grid[mask] - center_y) / ry
    nz = Z
    
    diffuse = np.maximum(0.15, nx * lx + ny * ly + nz * lz)
    
    # Specular highlight
    # Reflection vector
    rx_l = 2 * diffuse * nx - lx
    ry_l = 2 * diffuse * ny - ly
    rz_l = 2 * diffuse * nz - lz
    specular = np.maximum(0.0, rz_l) ** 16 * 0.45

    # Determine Base Palette based on Grade
    if grade == "A":
        # Deep ripe uniform red
        base_b = random.uniform(15, 30)
        base_g = random.uniform(25, 45)
        base_r_val = random.uniform(195, 235)
        colour_score = round(random.uniform(86, 99), 1)
        damage_score = round(random.uniform(88, 99), 1)
        surface_defect_score = round(random.uniform(88, 98), 1)
        
    elif grade == "B":
        # Turning/breaker stage or uneven yellowish-orange patches
        base_b = random.uniform(20, 50)
        base_g = random.uniform(70, 110)
        base_r_val = random.uniform(180, 220)
        colour_score = round(random.uniform(62, 79), 1)
        damage_score = round(random.uniform(62, 79), 1)
        surface_defect_score = round(random.uniform(60, 78), 1)
        
    else: # Grade C
        # Mottled, overripe or unripely blotched
        base_b = random.uniform(35, 75)
        base_g = random.uniform(60, 130)
        base_r_val = random.uniform(140, 190)
        colour_score = round(random.uniform(30, 58), 1)
        damage_score = round(random.uniform(20, 56), 1)
        surface_defect_score = round(random.uniform(22, 58), 1)

    tomato_rgb = np.zeros((h, w, 3), dtype=np.float32)
    tomato_rgb[:, :, 0] = base_b  # B
    tomato_rgb[:, :, 1] = base_g  # G
    tomato_rgb[:, :, 2] = base_r_val  # R
    
    # Apply shading
    shaded_tomato = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        col = tomato_rgb[:, :, c] * (diffuse * 0.85 + 0.15) + specular * 255.0
        shaded_tomato[:, :, c] = np.clip(col, 0, 255).astype(np.uint8)
        
    # Grade B / C color unevenness (greenish / yellow shoulder near calyx)
    if grade in ["B", "C"]:
        shoulder_mask = (dist_sq <= 0.65) & (Y_grid < center_y - ry * 0.15)
        if np.any(shoulder_mask):
            green_intensity = 0.35 if grade == "B" else 0.55
            shaded_tomato[shoulder_mask, 0] = np.clip(shaded_tomato[shoulder_mask, 0] * 0.8, 0, 255)
            shaded_tomato[shoulder_mask, 1] = np.clip(shaded_tomato[shoulder_mask, 1] * (1.0 + green_intensity), 0, 255)
            shaded_tomato[shoulder_mask, 2] = np.clip(shaded_tomato[shoulder_mask, 2] * (1.0 - green_intensity * 0.35), 0, 255)

    # 3. Apply Defects and Damage based on Grade
    # Defect layer
    defect_layer = shaded_tomato.copy()
    
    if grade == "B":
        # 1-3 minor defects (small russet scars or minor blemishes)
        num_defects = random.randint(1, 3)
        for _ in range(num_defects):
            def_x = center_x + int(random.uniform(-0.55, 0.55) * rx)
            def_y = center_y + int(random.uniform(-0.45, 0.55) * ry)
            def_rad = random.randint(5, 14)
            # Brownish/corky scar
            cv2.circle(defect_layer, (def_x, def_y), def_rad, (35, 75, 115), -1)
            
    elif grade == "C":
        # Severe defects: blossom-end rot (large black sunken lesion) or severe cracking
        defect_type = random.choice(["blossom_end_rot", "cracks", "severe_scabs", "rot_mottling"])
        if defect_type == "blossom_end_rot":
            rot_y = center_y + int(ry * 0.45)
            rot_x = center_x + int(random.uniform(-0.2, 0.2) * rx)
            cv2.ellipse(defect_layer, (rot_x, rot_y), (int(rx * 0.45), int(ry * 0.25)), 0, 0, 360, (15, 20, 25), -1)
        elif defect_type == "cracks":
            # Radial growth cracks near stem or body
            for angle in [30, 90, 150, 210, 320]:
                rad = math.radians(angle + random.uniform(-15, 15))
                x1 = int(center_x + math.cos(rad) * rx * 0.2)
                y1 = int(center_y + math.sin(rad) * ry * 0.2)
                x2 = int(center_x + math.cos(rad) * rx * 0.75)
                y2 = int(center_y + math.sin(rad) * ry * 0.75)
                cv2.line(defect_layer, (x1, y1), (x2, y2), (20, 30, 45), thickness=random.randint(4, 7))
        else:
            # Multiple dark necrotic lesions
            for _ in range(random.randint(4, 8)):
                def_x = center_x + int(random.uniform(-0.6, 0.6) * rx)
                def_y = center_y + int(random.uniform(-0.6, 0.6) * ry)
                cv2.ellipse(defect_layer, (def_x, def_y), (random.randint(12, 28), random.randint(10, 20)),
                            random.randint(0, 180), 0, 360, (25, 40, 55), -1)

    # Smooth the defects slightly into skin texture
    defect_layer = cv2.GaussianBlur(defect_layer, (5, 5), 0)
    
    # 4. Draw Calyx / Stem (top green crown)
    calyx_center_x = center_x
    calyx_center_y = center_y - ry + 8
    
    for angle in [0, 60, 120, 180, 240, 300]:
        sepal_len = random.randint(22, 38)
        rad = math.radians(angle + random.uniform(-10, 10))
        tip_x = int(calyx_center_x + math.sin(rad) * sepal_len)
        tip_y = int(calyx_center_y + math.cos(rad) * sepal_len * 0.6)
        pts = np.array([
            [calyx_center_x - 4, calyx_center_y],
            [tip_x, tip_y],
            [calyx_center_x + 4, calyx_center_y]
        ], np.int32)
        cv2.fillPoly(defect_layer, [pts], (25, 120, 45))
    cv2.circle(defect_layer, (calyx_center_x, calyx_center_y), 7, (20, 100, 35), -1)
    
    # Merge produce onto background using mask
    # Soft anti-aliased edge
    edge_blur = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    for c in range(3):
        img[:, :, c] = (defect_layer[:, :, c] * edge_blur + img[:, :, c] * (1.0 - edge_blur)).astype(np.uint8)
        
    # Calculate composite expert score:
    # Baseline weighting formula: Colour * 0.25 + Damage * 0.30 + Shape * 0.15 + Size * 0.15 + Surface Defect * 0.15
    expert_score = round(
        colour_score * 0.25 +
        damage_score * 0.30 +
        shape_score * 0.15 +
        size_score * 0.15 +
        surface_defect_score * 0.15,
        1
    )
    
    # Clamp to strict grade bounds
    if grade == "A":
        expert_score = max(80.0, min(100.0, expert_score))
    elif grade == "B":
        expert_score = max(60.0, min(79.9, expert_score))
    else:
        expert_score = max(10.0, min(59.9, expert_score))
        
    attrs = {
        "grade": grade,
        "expert_score": expert_score,
        "colour_score": colour_score,
        "damage_score": damage_score,
        "shape_score": shape_score,
        "size_score": size_score,
        "surface_defect_score": surface_defect_score
    }
    return img, attrs

def apply_lighting_filter(img, condition):
    """Applies realistic microclimatic lighting condition to the final image."""
    h, w = img.shape[:2]
    
    if condition == "diffuse_daylight":
        # Natural neutral daylight
        return img
        
    elif condition == "direct_sunlight":
        # Warm golden tint with increased contrast
        lut_warm = np.clip(img.astype(np.float32) * [0.92, 1.02, 1.12], 0, 255).astype(np.uint8)
        return cv2.convertScaleAbs(lut_warm, alpha=1.1, beta=10)
        
    elif condition == "greenhouse_filtered":
        # Diffused soft lighting, slight high humidity haze
        blur = cv2.GaussianBlur(img, (9, 9), 0)
        hazy = cv2.addWeighted(img, 0.85, blur, 0.15, 5)
        # Slight green cast from greenhouse shade netting
        hazy = np.clip(hazy.astype(np.float32) * [0.95, 1.05, 0.98], 0, 255).astype(np.uint8)
        return hazy
        
    elif condition == "shade_overcast":
        # Cool overcast / lower saturation
        cool = np.clip(img.astype(np.float32) * [1.08, 1.0, 0.94], 0, 255).astype(np.uint8)
        return cv2.convertScaleAbs(cool, alpha=0.92, beta=-5)
        
    elif condition == "indoor_fluorescent":
        # Slight green/white indoor illumination
        fl = np.clip(img.astype(np.float32) * [1.02, 1.08, 0.96], 0, 255).astype(np.uint8)
        return cv2.convertScaleAbs(fl, alpha=0.98, beta=8)
        
    return img

def generate_edge_cases():
    """Generates the required failure/edge cases."""
    edge_records = []
    
    # Case 1: Poor Lighting (Very dark underexposed image)
    for i in range(3):
        bg = generate_background("neutral_gray_tray")
        rendered, _ = render_single_produce(bg, grade="A", center_x=320, center_y=320)
        # Underexpose heavily
        dark_img = cv2.convertScaleAbs(rendered, alpha=0.12, beta=-15)
        filename = f"edge_poor_lighting_{i+1:02d}.jpg"
        filepath = os.path.join(EDGE_DIR, filename)
        cv2.imwrite(filepath, dark_img)
        edge_records.append({
            "image_id": f"EDGE_LIGHT_{i+1:02d}",
            "image_path": os.path.relpath(filepath, OUTPUT_DIR).replace("\\", "/"),
            "produce_type": "Tomato",
            "grade": "EDGE_CASE",
            "expert_score": 0.0,
            "colour_score": 0.0,
            "damage_score": 0.0,
            "shape_score": 0.0,
            "size_score": 0.0,
            "surface_defect_score": 0.0,
            "lighting_condition": "extreme_darkness",
            "background_condition": "neutral_gray_tray",
            "edge_case_type": "poor_lighting",
            "expected_behavior": "Low image quality / poor lighting. Human review recommended."
        })

    # Case 2: Multiple Produce Objects (2-3 tomatoes in frame)
    for i in range(3):
        bg = generate_background("wooden_grading_bench")
        # Render first tomato on left
        rendered, _ = render_single_produce(bg, grade="A", center_x=180, center_y=320, radius_scale=0.70)
        # Render second tomato on right
        rendered, _ = render_single_produce(rendered, grade="B", center_x=460, center_y=320, radius_scale=0.75)
        if i == 2:
            # Third tomato at top
            rendered, _ = render_single_produce(rendered, grade="A", center_x=320, center_y=160, radius_scale=0.60)
        filename = f"edge_multiple_objects_{i+1:02d}.jpg"
        filepath = os.path.join(EDGE_DIR, filename)
        cv2.imwrite(filepath, rendered)
        edge_records.append({
            "image_id": f"EDGE_MULTI_{i+1:02d}",
            "image_path": os.path.relpath(filepath, OUTPUT_DIR).replace("\\", "/"),
            "produce_type": "Tomato",
            "grade": "EDGE_CASE",
            "expert_score": 0.0,
            "colour_score": 0.0,
            "damage_score": 0.0,
            "shape_score": 0.0,
            "size_score": 0.0,
            "surface_defect_score": 0.0,
            "lighting_condition": "diffuse_daylight",
            "background_condition": "wooden_grading_bench",
            "edge_case_type": "multiple_objects",
            "expected_behavior": "Multiple objects detected. Please provide one produce item."
        })

    # Case 3: Blurry Image (Heavy motion or defocus blur)
    for i in range(3):
        bg = generate_background("white_inspection_mat")
        rendered, _ = render_single_produce(bg, grade="B", center_x=320, center_y=320)
        # Apply heavy blur
        blurred = cv2.GaussianBlur(rendered, (41, 41), sigmaX=15)
        filename = f"edge_blurry_{i+1:02d}.jpg"
        filepath = os.path.join(EDGE_DIR, filename)
        cv2.imwrite(filepath, blurred)
        edge_records.append({
            "image_id": f"EDGE_BLUR_{i+1:02d}",
            "image_path": os.path.relpath(filepath, OUTPUT_DIR).replace("\\", "/"),
            "produce_type": "Tomato",
            "grade": "EDGE_CASE",
            "expert_score": 0.0,
            "colour_score": 0.0,
            "damage_score": 0.0,
            "shape_score": 0.0,
            "size_score": 0.0,
            "surface_defect_score": 0.0,
            "lighting_condition": "diffuse_daylight",
            "background_condition": "white_inspection_mat",
            "edge_case_type": "blurry_image",
            "expected_behavior": "Image quality insufficient for reliable grading (blurry image)."
        })
        
    return edge_records

def generate_complete_dataset(total_samples_per_class=120):
    """
    Generates balanced ethical dataset:
    120 Grade A, 120 Grade B, 120 Grade C = 360 images total.
    Split: 70% Train (252), 15% Val (54), 15% Test (54) + 9 Edge cases.
    """
    print(f"Starting Ethical Produce Dataset Generation: {total_samples_per_class * 3} core images...")
    
    metadata_list = []
    annotations_list = []
    
    classes = ["A", "B", "C"]
    
    for grade in classes:
        print(f"Generating Grade {grade} produce samples...")
        for i in range(total_samples_per_class):
            image_idx = i + 1
            image_id = f"TOM_{grade}_{image_idx:03d}"
            
            # Select background & lighting
            bg_type = random.choice(BACKGROUND_CONDITIONS)
            light_type = random.choice(LIGHTING_CONDITIONS)
            
            # Generate background
            canvas = generate_background(bg_type)
            
            # Random slight position jitter simulating handheld field placement
            cx = 320 + random.randint(-25, 25)
            cy = 320 + random.randint(-25, 25)
            scale = random.uniform(0.90, 1.10)
            
            # Render produce
            img, attrs = render_single_produce(canvas, grade=grade, center_x=cx, center_y=cy, radius_scale=scale)
            
            # Apply microclimate lighting
            final_img = apply_lighting_filter(img, light_type)
            
            # Assign split: 70% train (0-83), 15% val (84-101), 15% test (102-119)
            if i < int(total_samples_per_class * 0.70):
                split_dir = TRAIN_DIR
                split_name = "train"
            elif i < int(total_samples_per_class * 0.85):
                split_dir = VAL_DIR
                split_name = "val"
            else:
                split_dir = TEST_DIR
                split_name = "test"
                
            filename = f"{image_id}.jpg"
            full_path = os.path.join(split_dir, filename)
            rel_path = os.path.relpath(full_path, OUTPUT_DIR).replace("\\", "/")
            
            cv2.imwrite(full_path, final_img)
            
            # Record metadata
            metadata_list.append({
                "image_id": image_id,
                "image_path": rel_path,
                "produce_type": "Tomato",
                "grade": grade,
                "expert_score": attrs["expert_score"],
                "colour_score": attrs["colour_score"],
                "damage_score": attrs["damage_score"],
                "shape_score": attrs["shape_score"],
                "size_score": attrs["size_score"],
                "surface_defect_score": attrs["surface_defect_score"],
                "lighting_condition": light_type,
                "background_condition": bg_type,
                "split": split_name
            })
            
            # Record expert annotations
            grader_id = f"EXT_AGRONOMIST_{(i % 3) + 1:02d}"
            notes_map = {
                "A": "Excellent uniform crimson color, minimal surface blemishes, high circularity.",
                "B": "Moderate color variation or small russeting marks; commercially acceptable table grade.",
                "C": "Severe defects, blossom-end rot, growth cracking or unmarketable mottled discoloration."
            }
            annotations_list.append({
                "image_id": image_id,
                "expert_grade": grade,
                "expert_score": attrs["expert_score"],
                "grader_id": grader_id,
                "notes": notes_map[grade]
            })
            
    # Generate edge cases
    print("Generating edge / failure cases...")
    edge_records = generate_edge_cases()
    for er in edge_records:
        metadata_list.append({
            "image_id": er["image_id"],
            "image_path": er["image_path"],
            "produce_type": er["produce_type"],
            "grade": er["grade"],
            "expert_score": er["expert_score"],
            "colour_score": er["colour_score"],
            "damage_score": er["damage_score"],
            "shape_score": er["shape_score"],
            "size_score": er["size_score"],
            "surface_defect_score": er["surface_defect_score"],
            "lighting_condition": er["lighting_condition"],
            "background_condition": er["background_condition"],
            "split": "edge_cases"
        })
        annotations_list.append({
            "image_id": er["image_id"],
            "expert_grade": "FLAG_REVIEW",
            "expert_score": 0.0,
            "grader_id": "CHIEF_QUALITY_OFFICER",
            "notes": f"Edge test case: {er['edge_case_type']}. System should reject or flag for human review."
        })

    # Save to CSV
    df_meta = pd.DataFrame(metadata_list)
    meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"Metadata saved to {meta_path} ({len(df_meta)} rows)")

    df_annot = pd.DataFrame(annotations_list)
    annot_path = os.path.join(OUTPUT_DIR, "annotations.csv")
    df_annot.to_csv(annot_path, index=False)
    print(f"Annotations saved to {annot_path} ({len(df_annot)} rows)")

if __name__ == "__main__":
    generate_complete_dataset()
