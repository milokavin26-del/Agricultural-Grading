"""
Training and Validation Module for Produce Quality Grading (Tomato MVP)
-----------------------------------------------------------------------
Extracts features from train/val/test images, trains the Random Forest classifier
(along with comparative Logistic Regression & SVM benchmarks), evaluates performance,
and persists models and feature metadata.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import load_image, preprocess_and_segment, ValidationError
from src.feature_extraction import extract_measurable_features
from src.baseline import calculate_baseline_grade

def build_dataset_features(df_subset: pd.DataFrame, dataset_name: str = "train"):
    """
    Extracts features for all images in the dataframe subset.
    """
    print(f"Extracting features for {dataset_name} set ({len(df_subset)} samples)...")
    records = []
    skipped = 0
    
    for _, row in df_subset.iterrows():
        img_id = row["image_id"]
        rel_path = row["image_path"]
        grade = row["grade"]
        full_path = os.path.join(DATA_DIR, rel_path)
        
        try:
            img = load_image(full_path)
            preprocessed = preprocess_and_segment(img)
            feats = extract_measurable_features(preprocessed)
            
            flat_rec = {
                "image_id": img_id,
                "grade": grade,
                "expert_score": row["expert_score"],
                # Calibrated scores
                "attr_colour": feats["attributes"]["colour"],
                "attr_damage": feats["attributes"]["damage"],
                "attr_shape": feats["attributes"]["shape"],
                "attr_size": feats["attributes"]["size"],
                "attr_surface_defects": feats["attributes"]["surface_defects"],
                # Raw computer vision measurements
                **feats["raw_features"]
            }
            records.append(flat_rec)
        except ValidationError as e:
            print(f"  [Validation Warning] Image {img_id} skipped: {e.message}")
            skipped += 1
        except Exception as e:
            print(f"  [Error] Failed to process {img_id}: {str(e)}")
            skipped += 1
            
    print(f"Completed {dataset_name}: {len(records)} extracted, {skipped} skipped.")
    return pd.DataFrame(records)

def train_and_evaluate():
    meta_path = os.path.join(DATA_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run generate_dataset.py first.")
        
    df_meta = pd.read_csv(meta_path)
    
    # Filter core classes (A, B, C)
    df_core = df_meta[df_meta["grade"].isin(["A", "B", "C"])].copy()
    
    train_df = df_core[df_core["split"] == "train"]
    val_df = df_core[df_core["split"] == "val"]
    test_df = df_core[df_core["split"] == "test"]
    
    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 1. Feature extraction
    df_train_feats = build_dataset_features(train_df, "train")
    df_val_feats = build_dataset_features(val_df, "val")
    df_test_feats = build_dataset_features(test_df, "test")
    
    # Feature columns for model training
    feature_cols = [
        "attr_colour", "attr_damage", "attr_shape", "attr_size", "attr_surface_defects",
        "mean_r", "mean_g", "mean_b", "rg_ratio", "mean_hue", "std_hue", "mean_sat", "mean_val",
        "area_ratio", "aspect_ratio", "circularity", "solidity", "damage_ratio", "grad_std"
    ]
    
    X_train = df_train_feats[feature_cols].values
    y_train = df_train_feats["grade"].values
    
    X_val = df_val_feats[feature_cols].values
    y_val = df_val_feats["grade"].values
    
    X_test = df_test_feats[feature_cols].values
    y_test = df_test_feats["grade"].values
    
    # Standard scaler for models sensitive to scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # =========================================================================
    # 2. TRAIN MODELS
    # =========================================================================
    print("\n--- Training Random Forest Model (Primary) ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    
    print("--- Training Logistic Regression (Benchmark) ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    print("--- Training Support Vector Machine (Benchmark) ---")
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # =========================================================================
    # 3. BASELINE EVALUATION ON TEST SET
    # =========================================================================
    print("\n--- Evaluating Baseline Grader on Held-Out Test Set ---")
    baseline_preds = []
    for _, row in df_test_feats.iterrows():
        b_res = calculate_baseline_grade({
            "colour": row["attr_colour"],
            "damage": row["attr_damage"],
            "shape": row["attr_shape"],
            "size": row["attr_size"],
            "surface_defects": row["attr_surface_defects"]
        })
        baseline_preds.append(b_res["grade"])
    baseline_preds = np.array(baseline_preds)
    
    base_acc = accuracy_score(y_test, baseline_preds)
    base_prec, base_rec, base_f1, _ = precision_recall_fscore_support(y_test, baseline_preds, average="weighted")
    base_agreement = float(np.mean(baseline_preds == y_test) * 100.0)
    base_disagreement = 100.0 - base_agreement
    
    # =========================================================================
    # 4. ML MODEL EVALUATION ON TEST SET
    # =========================================================================
    print("--- Evaluating Random Forest on Held-Out Test Set ---")
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    rf_confidences = np.max(rf_probs, axis=1)
    
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average="weighted")
    rf_agreement = float(np.mean(rf_preds == y_test) * 100.0)
    rf_disagreement = 100.0 - rf_agreement
    
    avg_confidence = float(np.mean(rf_confidences))
    low_conf_count = int(np.sum(rf_confidences < 0.75))
    
    # Feature importances
    importances = rf_model.feature_importances_
    feat_imp_dict = {feature_cols[i]: round(float(importances[i]), 4) for i in range(len(feature_cols))}
    
    # Map raw features to user-friendly attribute importances
    attr_importance_map = {
        "damage": float(np.sum([feat_imp_dict.get(k, 0) for k in ["attr_damage", "damage_ratio"]])),
        "colour": float(np.sum([feat_imp_dict.get(k, 0) for k in ["attr_colour", "mean_r", "mean_g", "mean_b", "rg_ratio", "mean_hue", "std_hue", "mean_sat", "mean_val"]])),
        "surface_defects": float(np.sum([feat_imp_dict.get(k, 0) for k in ["attr_surface_defects", "grad_std"]])),
        "shape": float(np.sum([feat_imp_dict.get(k, 0) for k in ["attr_shape", "aspect_ratio", "circularity", "solidity"]])),
        "size": float(np.sum([feat_imp_dict.get(k, 0) for k in ["attr_size", "area_ratio"]]))
    }
    # Normalize to 1.0
    imp_sum = sum(attr_importance_map.values())
    attr_importance_map = {k: round(v / imp_sum, 4) for k, v in attr_importance_map.items()}

    # Print comparative results
    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'Baseline Grader':<15} | {'ML Random Forest':<15} | {'Target':<10}")
    print("="*70)
    print(f"{'Accuracy':<25} | {base_acc*100:>13.2f}% | {rf_acc*100:>13.2f}% | >= 80%")
    print(f"{'Weighted F1':<25} | {base_f1:>14.3f} | {rf_f1:>14.3f} | >= 0.75")
    print(f"{'Expert Agreement':<25} | {base_agreement:>13.2f}% | {rf_agreement:>13.2f}% | >= 80%")
    print(f"{'Disagreement Rate':<25} | {base_disagreement:>13.2f}% | {rf_disagreement:>13.2f}% | Lower than baseline")
    print(f"{'Average Confidence':<25} | {'N/A':>14} | {avg_confidence*100:>13.2f}% | >= 75%")
    print(f"{'Low-Conf Count (<75%)':<25} | {'N/A':>14} | {low_conf_count:>14d} | Low count")
    print("="*70)
    
    # Confusion matrices
    labels = ["A", "B", "C"]
    cm_base = confusion_matrix(y_test, baseline_preds, labels=labels).tolist()
    cm_rf = confusion_matrix(y_test, rf_preds, labels=labels).tolist()
    
    # Save artifacts
    model_save_path = os.path.join(MODELS_DIR, "quality_grading_model.pkl")
    joblib.dump(rf_model, model_save_path)
    print(f"\nTrained Random Forest model saved to: {model_save_path}")
    
    scaler_save_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_save_path)

    metadata_dict = {
        "model_type": "RandomForestClassifier",
        "classes": labels,
        "feature_cols": feature_cols,
        "feature_importances": feat_imp_dict,
        "attribute_importances": attr_importance_map,
        "metrics": {
            "baseline": {
                "accuracy": round(base_acc, 4),
                "precision": round(base_prec, 4),
                "recall": round(base_rec, 4),
                "f1": round(base_f1, 4),
                "expert_agreement_pct": round(base_agreement, 2),
                "disagreement_rate_pct": round(base_disagreement, 2),
                "confusion_matrix": cm_base
            },
            "random_forest": {
                "accuracy": round(rf_acc, 4),
                "precision": round(rf_prec, 4),
                "recall": round(rf_rec, 4),
                "f1": round(rf_f1, 4),
                "expert_agreement_pct": round(rf_agreement, 2),
                "disagreement_rate_pct": round(rf_disagreement, 2),
                "average_confidence": round(avg_confidence, 4),
                "low_confidence_count": low_conf_count,
                "confusion_matrix": cm_rf
            }
        }
    }
    
    meta_json_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_json_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"Model metadata & metrics saved to: {meta_json_path}")
    
    # Save test predictions for error analysis
    df_test_feats["baseline_pred"] = baseline_preds
    df_test_feats["ml_pred"] = rf_preds
    df_test_feats["ml_confidence"] = rf_confidences
    df_test_feats.to_csv(os.path.join(REPORTS_DIR, "test_evaluation_results.csv"), index=False)
    
    return metadata_dict

if __name__ == "__main__":
    train_and_evaluate()
