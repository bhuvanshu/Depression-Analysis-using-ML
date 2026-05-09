"""
Deployment Inference Test
─────────────────────────
Simulates real frontend/backend inference using raw user inputs.
Verifies that pipeline.joblib accepts raw values, preprocesses
automatically, and produces predictions without any external code.

Usage:
    python -m src.test_inference
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.risk_classification import get_risk_level


def run_inference_test():
    """Runs end-to-end inference tests with raw frontend-style inputs."""

    deploy_dir = root / "outputs" / "gradient_boosting"
    pipeline_path = deploy_dir / "pipeline.joblib"
    metadata_path = deploy_dir / "model_metadata.json"

    print("=" * 60)
    print("  DEPLOYMENT INFERENCE TEST")
    print("=" * 60)

    # ── Load pipeline ──
    print("\n[1] Loading pipeline.joblib...")
    if not pipeline_path.exists():
        sys.exit(f"ERROR: {pipeline_path} not found. Run build_pipeline.py first.")

    pipeline = joblib.load(pipeline_path)
    print(f"    Pipeline type: {type(pipeline)}")
    print(f"    Steps: {[name for name, _ in pipeline.steps]}")

    # ── Load metadata ──
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    risk_cfg = metadata.get("risk_thresholds", {})
    q1 = risk_cfg.get("q1", 0.25)
    q3 = risk_cfg.get("q3", 0.75)
    print(f"    Risk thresholds: Q1={q1:.4f}, Q3={q3:.4f}")

    # ── Test Cases: Raw frontend-style inputs ──
    test_cases = [
        {
            "name": "Test 1: High-risk male undergraduate",
            "input": {
                "age": 21,
                "academic_pressure": 4,
                "cgpa": 5.5,
                "study_satisfaction": 1,
                "work_study_hours": 10,
                "financial_stress": 4,
                "suicidal_thoughts": 1,
                "family_history": 1,
                "gender": "Male",
                "sleep_duration": "Less than 5 hours",
                "degree": "Undergraduate",
            }
        },
        {
            "name": "Test 2: Low-risk female postgraduate",
            "input": {
                "age": 24,
                "academic_pressure": 1,
                "cgpa": 8.5,
                "study_satisfaction": 4,
                "work_study_hours": 4,
                "financial_stress": 1,
                "suicidal_thoughts": 0,
                "family_history": 0,
                "gender": "Female",
                "sleep_duration": "7-8 hours",
                "degree": "Postgraduate",
            }
        },
        {
            "name": "Test 3: Moderate-risk (mixed signals)",
            "input": {
                "age": 20,
                "academic_pressure": 3,
                "cgpa": 6.0,
                "study_satisfaction": 2,
                "work_study_hours": 6,
                "financial_stress": 3,
                "suicidal_thoughts": 0,
                "family_history": 1,
                "gender": "Other",
                "sleep_duration": "5-6 hours",
                "degree": "School",
            }
        },
        {
            "name": "Test 4: Minimal input with defaults (edge case)",
            "input": {
                "age": 18,
                "academic_pressure": 0,
                "cgpa": 10.0,
                "study_satisfaction": 5,
                "work_study_hours": 0,
                "financial_stress": 0,
                "suicidal_thoughts": 0,
                "family_history": 0,
                "gender": "Female",
                "sleep_duration": "More than 8 hours",
                "degree": "Doctorate",
            }
        },
        {
            "name": "Test 5: Example from user prompt",
            "input": {
                "age": 22,
                "academic_pressure": 4,
                "cgpa": 6.0,
                "study_satisfaction": 2,
                "work_study_hours": 6,
                "financial_stress": 3,
                "suicidal_thoughts": 0,
                "family_history": 0,
                "gender": "Male",
                "sleep_duration": "5-6 hours",
                "degree": "Undergraduate",
            }
        },
    ]

    print(f"\n[2] Running {len(test_cases)} inference tests...\n")
    all_passed = True

    for tc in test_cases:
        print(f"    --- {tc['name']} ---")
        raw_input = tc["input"]

        try:
            # Convert to DataFrame (simulates what FastAPI/Flask would do)
            df_input = pd.DataFrame([raw_input])

            # Predict
            prediction = int(pipeline.predict(df_input)[0])
            probabilities = pipeline.predict_proba(df_input)[0]
            depression_prob = float(probabilities[1])

            # Risk level
            risk = get_risk_level(depression_prob, q1, q3)

            print(f"    Input: {json.dumps(raw_input)}")
            print(f"    Prediction: {prediction} ({'Depressed' if prediction == 1 else 'Not Depressed'})")
            print(f"    Probability: {depression_prob:.4f}")
            print(f"    Risk: {risk['level']} ({risk['percentile']})")
            print(f"    Status: PASSED\n")

        except Exception as e:
            print(f"    Status: FAILED - {e}\n")
            all_passed = False

    # ── Summary ──
    print("=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED - Pipeline is deployment-safe!")
    else:
        print("  SOME TESTS FAILED - Review errors above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_inference_test()
    sys.exit(0 if success else 1)
