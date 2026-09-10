import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_produce

tests = [
    ("Grade A", "data/images/test/TOM_A_104.jpg"),
    ("Grade B", "data/images/test/TOM_B_104.jpg"),
    ("Grade C", "data/images/test/TOM_C_104.jpg"),
    ("Edge 1 Poor Light", "data/images/edge_cases/edge_poor_lighting_01.jpg"),
    ("Edge 2 Multi Objects", "data/images/edge_cases/edge_multiple_objects_01.jpg"),
    ("Edge 3 Blur", "data/images/edge_cases/edge_blurry_01.jpg")
]

for title, path in tests:
    print(f"=== Testing {title} ({path}) ===")
    res = predict_produce(os.path.join(BASE_DIR, path))
    print(f"Status: {res['status']}")
    print(f"Grade: {res['grade']}, Quality: {res['quality_score']}, Confidence: {res['confidence']}")
    print(f"Human Review Required: {res['human_review_required']}")
    print(f"Warning: {res.get('warning')}")
    print(f"Explanation: {res['explanation']}")
    print(f"Attributes: {res['attributes']}\n")
