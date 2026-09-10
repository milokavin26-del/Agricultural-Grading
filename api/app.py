"""
FastAPI Backend Integration API for Produce Quality Grading (Tomato MVP)
-----------------------------------------------------------------------
Provides lightweight, field-ready REST API endpoints:
- GET  /health   : Health check
- POST /predict  : Single produce image grading and explainability
- POST /review   : Human review submission and dispute logging
- GET  /metrics  : Global dataset and model evaluation metrics
"""

import os
import sys
import io
import csv
from datetime import datetime
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_produce, DEFAULT_CONFIDENCE_THRESHOLD, get_predictor

app = FastAPI(
    title="Agricultural Produce Quality Grading API",
    description="Field-ready explainable produce quality grading API for agricultural extension teams.",
    version="1.0.0"
)

# Enable CORS for local web interface access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REVIEWS_CSV = os.path.join(BASE_DIR, "data", "human_reviews.csv")

class ReviewRequest(BaseModel):
    image_id: Optional[str] = "UNKNOWN"
    system_grade: str
    expert_override_grade: str
    officer_id: str
    comments: Optional[str] = ""

@app.get("/health")
def health_check():
    """Health check endpoint confirming API status."""
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD)
):
    """
    Accepts an agricultural produce image file and returns:
    - Predicted quality grade (A/B/C)
    - Quality score (0-100)
    - Confidence score (0-1.0)
    - Measurable attributes (colour, damage, shape, size, surface_defects)
    - Human-readable explanation
    - Human review warning if confidence is low or validation fails
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        result = predict_produce(image_bytes, confidence_threshold=confidence_threshold)
        
        # Format matching project specification
        response_payload = {
            "status": result["status"],
            "grade": result["grade"],
            "quality_score": result["quality_score"],
            "confidence": result["confidence"],
            "attributes": result["attributes"],
            "attribute_labels": result.get("attribute_labels", {}),
            "explanation": result["explanation"],
            "human_review_required": result["human_review_required"],
            "warning": result.get("warning"),
            "baseline_grade": result.get("baseline_grade"),
            "baseline_quality_score": result.get("baseline_quality_score"),
            "feature_importance": result.get("feature_importance", {})
        }
        return response_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/review")
def submit_human_review(review: ReviewRequest):
    """
    Records an agricultural extension officer's review or grade override.
    """
    file_exists = os.path.exists(REVIEWS_CSV)
    
    with open(REVIEWS_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image_id", "system_grade", "expert_override_grade", "officer_id", "comments"])
        writer.writerow([
            datetime.now().isoformat(),
            review.image_id,
            review.system_grade,
            review.expert_override_grade,
            review.officer_id,
            review.comments
        ])
        
    return {
        "status": "SUCCESS",
        "message": "Human review recorded successfully.",
        "recorded_review": review.model_dump()
    }

@app.get("/metrics")
def get_metrics():
    """
    Returns latest model metrics, agreement rates, and dataset distribution.
    """
    predictor = get_predictor()
    if predictor.metadata:
        return predictor.metadata
    return {"error": "Model metadata not available. Run train.py first."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
