from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add current dir to path
root = Path(__file__).resolve().parent
sys.path.append(str(root))

app = Flask(__name__)
CORS(app)

# Load model and features
MODEL_PATH = root / "outputs" / "gradient_boosting" / "model.joblib"
FEATURES_PATH = root / "outputs" / "gradient_boosting" / "feature_names.joblib"

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"Model and features loaded. Expecting {len(feature_names)} features.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare data in the same order as training
        input_data = []
        for feat in feature_names:
            if feat in data:
                input_data.append(data[feat])
            else:
                # Handle missing features with defaults or error
                # For this demo, we'll assume the frontend sends everything
                # or default to 0
                input_data.append(0)
        
        df_input = pd.DataFrame([input_data], columns=feature_names)
        
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0].tolist()
        
        return jsonify({
            "prediction": int(prediction),
            "prediction_label": "Depressed" if prediction == 1 else "Healthy",
            "probability": probability,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "up"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
