"""
Depression Risk Prediction API
-------------------------------
Flask server that loads the production pipeline (pipeline.joblib) and exposes
a REST endpoint for depression risk prediction.

Usage:
    gunicorn serve_model:app
    python serve_model.py --port 5000

Endpoints:
    POST /predict   - Submit student data, get depression prediction
    GET  /health    - Server health check
    GET  /features  - Return expected input fields with metadata
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import argparse
from pathlib import Path
import sys

# -- Internal imports --
from inference.predictor import DepressionPredictor
from inference.schema import get_features_schema
from inference.risk import RISK_JUSTIFICATION

# -- Paths --
root = Path(__file__).resolve().parent

# -- Initialize Predictor --
predictor = DepressionPredictor(root_dir=root)

# -- Flask app --
app = Flask(__name__)
CORS(app)


# ==================================
#  API ENDPOINTS
# ==================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with user-friendly field names, returns prediction + risk level.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON body provided"}), 400

        # Delegate everything to predictor
        result = predictor.predict(data)
        result["status"] = "success"

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Server health check with model info and risk framework."""
    return jsonify({
        "status": "up",
        "model_type": predictor.metadata.get("model_type", "GradientBoostingClassifier"),
        "pipeline_mode": "unified (pipeline.joblib)" if predictor.use_pipeline else "legacy (model.joblib)",
        "accepts_raw_input": predictor.use_pipeline,
        "n_features": predictor.n_features,
        "model_metrics": predictor.metadata.get("metrics", {}),
        "risk_framework": {
            "method": "percentile-based",
            "q1": predictor.risk_q1,
            "q3": predictor.risk_q3,
            "justification": RISK_JUSTIFICATION
        }
    })


@app.route("/features", methods=["GET"])
def features():
    """Returns the expected input fields and their metadata for frontend form generation."""
    schema_response = get_features_schema(predictor.metadata)
    return jsonify(schema_response)


# ==================================
#  ENTRY POINT
# ==================================

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Depression Prediction API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    # Get port from environment variable (Render sets this automatically) or fallback to 5000
    port = int(os.environ.get("PORT", 5000))

    mode = "UNIFIED PIPELINE" if predictor.use_pipeline else "LEGACY MODEL"
    print(f"\n[STARTING] Depression Prediction API ({mode})")
    print(f"   http://{args.host}:{port}")
    print(f"   POST /predict  - Submit prediction request")
    print(f"   GET  /health   - Health check")
    print(f"   GET  /features - Input field metadata\n")

    app.run(host=args.host, port=port, debug=False)
