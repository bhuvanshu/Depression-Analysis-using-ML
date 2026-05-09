"""
Depression Risk Prediction API
-------------------------------
Flask server that loads the production pipeline (pipeline.joblib) and exposes
a REST endpoint for depression risk prediction.

The pipeline encapsulates ALL preprocessing (one-hot encoding for gender,
sleep duration, and degree) internally. Raw frontend inputs are passed
directly — no manual feature engineering needed.

Usage:
    python serve_model.py          # starts on port 5000
    python serve_model.py --port 8080

Endpoints:
    POST /predict   - Submit student data, get depression prediction
    GET  /health    - Server health check
    GET  /features  - Return expected input fields with metadata
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse

# -- Paths --
root = Path(__file__).resolve().parent
sys.path.append(str(root))

DEPLOY_DIR = root / "outputs" / "gradient_boosting"
PIPELINE_PATH = DEPLOY_DIR / "pipeline.joblib"
LEGACY_MODEL_PATH = DEPLOY_DIR / "model.joblib"
FEATURES_PATH = DEPLOY_DIR / "feature_names.joblib"
METADATA_PATH = DEPLOY_DIR / "model_metadata.json"

# -- Imports from centralized modules --
from src.risk_classification import get_risk_level
from src.config import RISK_JUSTIFICATION

# -- Load artifacts --
try:
    # Prefer the unified pipeline (preprocessor + model)
    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        USE_PIPELINE = True
        print(f"[OK] Loaded production pipeline from {PIPELINE_PATH}")
    elif LEGACY_MODEL_PATH.exists():
        pipeline = joblib.load(LEGACY_MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH) if FEATURES_PATH.exists() else []
        USE_PIPELINE = False
        print(f"[WARN] Using legacy model.joblib (manual preprocessing required)")
    else:
        print("[ERROR] No model artifact found!")
        sys.exit(1)

    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}

    # Load percentile-based risk thresholds (Q1/Q3)
    risk_cfg = metadata.get("risk_thresholds", {})
    RISK_Q1 = risk_cfg.get("q1", 0.25)  # fallback if metadata missing
    RISK_Q3 = risk_cfg.get("q3", 0.75)

    n_features = metadata.get("n_raw_features", len(metadata.get("raw_feature_schema", [])))
    print(f"   Accepts raw input: {metadata.get('accepts_raw_input', not USE_PIPELINE)}")
    print(f"   Raw features: {n_features}")
    print(f"   Risk thresholds: Q1 = {RISK_Q1:.4f}, Q3 = {RISK_Q3:.4f} (percentile-based)")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# -- Flask app --
app = Flask(__name__)
CORS(app)


def build_feature_dataframe(data: dict) -> pd.DataFrame:
    """
    Converts user-friendly JSON input into a single-row DataFrame
    matching the pipeline's expected raw feature schema.

    The pipeline's ColumnTransformer handles all encoding internally.
    This function only normalizes field names and applies defaults.

    Expected input fields (from frontend form):
        age                : int (18-28)
        academic_pressure  : int (0-5)
        cgpa               : float (0-10)
        study_satisfaction : int (0-5)
        work_study_hours   : int (0-12)
        financial_stress   : int (0-5)
        suicidal_thoughts  : bool / 0,1
        family_history     : bool / 0,1
        gender             : "Male" | "Female" | "Other"
        sleep_duration     : "Less than 5 hours" | "5-6 hours" | "7-8 hours" | "More than 8 hours"
        degree             : "School" | "Undergraduate" | "Postgraduate" | "Doctorate"
    """
    row = {
        "age": float(data.get("age", 20)),
        "academic_pressure": float(data.get("academic_pressure", 0)),
        "cgpa": float(data.get("cgpa", 5.0)),
        "study_satisfaction": float(data.get("study_satisfaction", 0)),
        "work_study_hours": float(data.get("work_study_hours", 0)),
        "financial_stress": float(data.get("financial_stress", 0)),
        "suicidal_thoughts": int(bool(data.get("suicidal_thoughts", 0))),
        "family_history": int(bool(data.get("family_history", 0))),
        "gender": str(data.get("gender", "Male")).strip().title(),
        "sleep_duration": str(data.get("sleep_duration", "7-8 hours")).strip(),
        "degree": str(data.get("degree", "Undergraduate")).strip(),
    }

    # Normalize degree aliases
    degree_val = row["degree"].lower()
    degree_normalize = {
        "school": "School", "class 12": "School",
        "undergraduate": "Undergraduate", "undergrad": "Undergraduate", "ug": "Undergraduate",
        "postgraduate": "Postgraduate", "postgrad": "Postgraduate", "pg": "Postgraduate",
        "phd": "Doctorate", "doctorate": "Doctorate",
    }
    row["degree"] = degree_normalize.get(degree_val, "Undergraduate")

    return pd.DataFrame([row])


def build_feature_vector_legacy(data: dict) -> pd.DataFrame:
    """
    Legacy: Manual one-hot encoding for use with raw model.joblib.
    Kept for backward compatibility only.
    """
    row = {
        "age": float(data.get("age", 20)),
        "academic pressure": float(data.get("academic_pressure", 0)),
        "cgpa": float(data.get("cgpa", 5.0)),
        "study satisfaction": float(data.get("study_satisfaction", 0)),
        "work/study hours": float(data.get("work_study_hours", 0)),
        "financial stress": float(data.get("financial_stress", 0)),
        "have you ever had suicidal thoughts ?": int(bool(data.get("suicidal_thoughts", 0))),
        "family history of mental illness": int(bool(data.get("family_history", 0))),
    }

    # Gender one-hot
    gender = str(data.get("gender", "Male")).strip().lower()
    row["gender_male"] = 1 if gender == "male" else 0
    row["gender_female"] = 1 if gender == "female" else 0
    row["gender_other"] = 1 if gender == "other" else 0

    # Sleep duration one-hot
    sleep = str(data.get("sleep_duration", "7-8 hours")).strip().lower()
    row["sleep_5-6 hours"] = 1 if sleep == "5-6 hours" else 0
    row["sleep_7-8 hours"] = 1 if sleep == "7-8 hours" else 0
    row["sleep_less than 5 hours"] = 1 if sleep in ("less than 5 hours", "<5 hours") else 0
    row["sleep_more than 8 hours"] = 1 if sleep in ("more than 8 hours", ">8 hours") else 0
    row["sleep_others"] = 1 if sleep in ("others", "other") else 0

    # Degree one-hot
    degree = str(data.get("degree", "Undergraduate")).strip().lower()
    row["degree_school"] = 1 if degree in ("school", "class 12") else 0
    row["degree_undergrad"] = 1 if degree in ("undergraduate", "undergrad", "ug") else 0
    row["degree_postgrad"] = 1 if degree in ("postgraduate", "postgrad", "pg") else 0
    row["degree_phd"] = 1 if degree in ("phd", "doctorate") else 0

    df = pd.DataFrame([row])
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    return df[feature_names]


# ==================================
#  API ENDPOINTS
# ==================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with user-friendly field names, returns prediction + risk level.

    Example POST body:
    {
        "age": 21,
        "academic_pressure": 4,
        "cgpa": 6.5,
        "study_satisfaction": 2,
        "work_study_hours": 8,
        "financial_stress": 3,
        "suicidal_thoughts": false,
        "family_history": false,
        "gender": "Male",
        "sleep_duration": "5-6 hours",
        "degree": "Undergraduate"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON body provided"}), 400

        # Build feature input based on which artifact is loaded
        if USE_PIPELINE:
            X = build_feature_dataframe(data)
        else:
            X = build_feature_vector_legacy(data)

        # Predict
        prediction = int(pipeline.predict(X)[0])
        probabilities = pipeline.predict_proba(X)[0].tolist()
        depression_prob = probabilities[1]

        # Risk assessment (using centralized get_risk_level)
        risk = get_risk_level(depression_prob, RISK_Q1, RISK_Q3)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "prediction_label": "Depressed" if prediction == 1 else "Not Depressed",
            "probability": {
                "not_depressed": round(probabilities[0], 4),
                "depressed": round(probabilities[1], 4)
            },
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "risk_percentile": risk["percentile"],
            "recommended_action": risk["action"],
            "input_features_used": X.to_dict(orient="records")[0],
            "pipeline_mode": "unified" if USE_PIPELINE else "legacy"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Server health check with model info and risk framework."""
    return jsonify({
        "status": "up",
        "model_type": metadata.get("model_type", "GradientBoostingClassifier"),
        "pipeline_mode": "unified (pipeline.joblib)" if USE_PIPELINE else "legacy (model.joblib)",
        "accepts_raw_input": USE_PIPELINE,
        "n_features": metadata.get("n_raw_features", len(metadata.get("feature_names", []))),
        "model_metrics": metadata.get("metrics", {}),
        "risk_framework": {
            "method": "percentile-based",
            "q1": RISK_Q1,
            "q3": RISK_Q3,
            "justification": RISK_JUSTIFICATION
        }
    })


@app.route("/features", methods=["GET"])
def features():
    """Returns the expected input fields and their metadata for frontend form generation."""
    return jsonify({
        "status": "success",
        "fields": metadata.get("form_field_mapping", {}),
        "raw_feature_schema": metadata.get("raw_feature_schema", [])
    })


# ==================================
#  ENTRY POINT
# ==================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depression Prediction API Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    mode = "UNIFIED PIPELINE" if USE_PIPELINE else "LEGACY MODEL"
    print(f"\n[STARTING] Depression Prediction API ({mode})")
    print(f"   http://{args.host}:{args.port}")
    print(f"   POST /predict  - Submit prediction request")
    print(f"   GET  /health   - Health check")
    print(f"   GET  /features - Input field metadata\n")

    app.run(host=args.host, port=args.port, debug=True)
