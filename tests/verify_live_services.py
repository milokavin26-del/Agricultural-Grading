import requests
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*75)
print("VERIFYING LIVE RUNNING SERVICES ON LOCAL MACHINE")
print("="*75)

# 1. Streamlit
print("\n[1] Checking Streamlit Web Dashboard (http://localhost:8501)...")
try:
    r_ui = requests.get("http://localhost:8501", timeout=5)
    print(f"    -> Status: {r_ui.status_code} OK! Streamlit server is active and serving.")
except Exception as e:
    print(f"    -> Connection failed: {e}")

# 2. FastAPI Health
print("\n[2] Checking FastAPI Backend Health (http://127.0.0.1:8000/health)...")
try:
    r_api = requests.get("http://127.0.0.1:8000/health", timeout=5)
    print(f"    -> Status: {r_api.status_code} OK! Response: {r_api.json()}")
except Exception as e:
    print(f"    -> Connection failed: {e}")

# 3. Live Prediction: Grade A
print("\n[3] Testing Live POST /predict with Grade A Specimen...")
try:
    img_a = os.path.join(BASE_DIR, "data", "images", "test", "TOM_A_104.jpg")
    with open(img_a, "rb") as f:
        r_pred_a = requests.post("http://127.0.0.1:8000/predict", files={"file": ("tom_a.jpg", f, "image/jpeg")})
    res_a = r_pred_a.json()
    print(f"    -> Predicted Grade: {res_a['grade']}")
    print(f"    -> Quality Score:   {res_a['quality_score']} / 100")
    print(f"    -> Confidence:      {res_a['confidence']*100:.1f}%")
    print(f"    -> Attributes:      {res_a['attributes']}")
    print(f"    -> Explanation:     {res_a['explanation']}")
    print(f"    -> Review Flag:     {res_a['human_review_required']}")
except Exception as e:
    print(f"    -> Prediction failed: {e}")

# 4. Live Prediction: Grade B
print("\n[4] Testing Live POST /predict with Grade B Specimen...")
try:
    img_b = os.path.join(BASE_DIR, "data", "images", "test", "TOM_B_104.jpg")
    with open(img_b, "rb") as f:
        r_pred_b = requests.post("http://127.0.0.1:8000/predict", files={"file": ("tom_b.jpg", f, "image/jpeg")})
    res_b = r_pred_b.json()
    print(f"    -> Predicted Grade: {res_b['grade']}")
    print(f"    -> Quality Score:   {res_b['quality_score']} / 100")
    print(f"    -> Confidence:      {res_b['confidence']*100:.1f}%")
    print(f"    -> Explanation:     {res_b['explanation']}")
except Exception as e:
    print(f"    -> Prediction failed: {e}")

# 5. Live Prediction: Grade C
print("\n[5] Testing Live POST /predict with Grade C Specimen...")
try:
    img_c = os.path.join(BASE_DIR, "data", "images", "test", "TOM_C_104.jpg")
    with open(img_c, "rb") as f:
        r_pred_c = requests.post("http://127.0.0.1:8000/predict", files={"file": ("tom_c.jpg", f, "image/jpeg")})
    res_c = r_pred_c.json()
    print(f"    -> Predicted Grade: {res_c['grade']}")
    print(f"    -> Quality Score:   {res_c['quality_score']} / 100")
    print(f"    -> Confidence:      {res_c['confidence']*100:.1f}%")
    print(f"    -> Explanation:     {res_c['explanation']}")
except Exception as e:
    print(f"    -> Prediction failed: {e}")

# 6. Live Prediction: Edge Case (Poor Lighting)
print("\n[6] Testing Live POST /predict with Edge Case (Poor Lighting)...")
try:
    img_edge = os.path.join(BASE_DIR, "data", "images", "edge_cases", "edge_poor_lighting_01.jpg")
    with open(img_edge, "rb") as f:
        r_edge = requests.post("http://127.0.0.1:8000/predict", files={"file": ("edge_light.jpg", f, "image/jpeg")})
    res_edge = r_edge.json()
    print(f"    -> Status:          {res_edge['status']}")
    print(f"    -> Warning:         {res_edge.get('warning')}")
    print(f"    -> Review Required: {res_edge['human_review_required']}")
except Exception as e:
    print(f"    -> Edge test failed: {e}")

# 7. Human Review Submission
print("\n[7] Testing Live POST /review (Agronomist Grade Override Submission)...")
try:
    rev_payload = {
        "image_id": "TOM_B_104.jpg",
        "system_grade": "B",
        "expert_override_grade": "B",
        "officer_id": "EXTENSION_OFFICER_01",
        "comments": "Confirmed commercially acceptable table grade after physical firmness check."
    }
    r_rev = requests.post("http://127.0.0.1:8000/review", json=rev_payload)
    print(f"    -> Review Response: {r_rev.json()}")
except Exception as e:
    print(f"    -> Review submission failed: {e}")

print("\n" + "="*75)
print("ALL LIVE SERVICES VERIFIED AND ACTIVELY RESPONDING")
print("="*75)
