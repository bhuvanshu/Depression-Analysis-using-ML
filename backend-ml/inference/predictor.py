"""
Prediction orchestration and pipeline loading.
"""
import joblib
import json
import pandas as pd
from pathlib import Path
import sys

from .risk import get_risk_level

class DepressionPredictor:
    def __init__(self, root_dir: Path):
        self.deploy_dir = root_dir / "outputs" / "gradient_boosting"
        self.pipeline_path = self.deploy_dir / "pipeline.joblib"
        self.legacy_model_path = self.deploy_dir / "model.joblib"
        self.features_path = self.deploy_dir / "feature_names.joblib"
        self.metadata_path = self.deploy_dir / "model_metadata.json"
        
        self.pipeline = None
        self.feature_names = []
        self.metadata = {}
        self.use_pipeline = True
        self.risk_q1 = 0.25
        self.risk_q3 = 0.75
        self.n_features = 0
        
        self._load_artifacts()

    def _load_artifacts(self):
        if self.pipeline_path.exists():
            self.pipeline = joblib.load(self.pipeline_path)
            self.use_pipeline = True
            print(f"[OK] Loaded production pipeline from {self.pipeline_path}")
        elif self.legacy_model_path.exists():
            self.pipeline = joblib.load(self.legacy_model_path)
            self.feature_names = joblib.load(self.features_path) if self.features_path.exists() else []
            self.use_pipeline = False
            print(f"[WARN] Using legacy model.joblib (manual preprocessing required)")
        else:
            print("[ERROR] No model artifact found!")
            sys.exit(1)

        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text())

        risk_cfg = self.metadata.get("risk_thresholds", {})
        self.risk_q1 = risk_cfg.get("q1", 0.25)
        self.risk_q3 = risk_cfg.get("q3", 0.75)
        self.n_features = self.metadata.get("n_raw_features", len(self.metadata.get("raw_feature_schema", [])))

    def build_feature_dataframe(self, data: dict) -> pd.DataFrame:
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

        degree_val = row["degree"].lower()
        degree_normalize = {
            "school": "School", "class 12": "School",
            "undergraduate": "Undergraduate", "undergrad": "Undergraduate", "ug": "Undergraduate",
            "postgraduate": "Postgraduate", "postgrad": "Postgraduate", "pg": "Postgraduate",
            "phd": "Doctorate", "doctorate": "Doctorate",
        }
        row["degree"] = degree_normalize.get(degree_val, "Undergraduate")

        return pd.DataFrame([row])

    def build_feature_vector_legacy(self, data: dict) -> pd.DataFrame:
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

        gender = str(data.get("gender", "Male")).strip().lower()
        row["gender_male"] = 1 if gender == "male" else 0
        row["gender_female"] = 1 if gender == "female" else 0
        row["gender_other"] = 1 if gender == "other" else 0

        sleep = str(data.get("sleep_duration", "7-8 hours")).strip().lower()
        row["sleep_5-6 hours"] = 1 if sleep == "5-6 hours" else 0
        row["sleep_7-8 hours"] = 1 if sleep == "7-8 hours" else 0
        row["sleep_less than 5 hours"] = 1 if sleep in ("less than 5 hours", "<5 hours") else 0
        row["sleep_more than 8 hours"] = 1 if sleep in ("more than 8 hours", ">8 hours") else 0
        row["sleep_others"] = 1 if sleep in ("others", "other") else 0

        degree = str(data.get("degree", "Undergraduate")).strip().lower()
        row["degree_school"] = 1 if degree in ("school", "class 12") else 0
        row["degree_undergrad"] = 1 if degree in ("undergraduate", "undergrad", "ug") else 0
        row["degree_postgrad"] = 1 if degree in ("postgraduate", "postgrad", "pg") else 0
        row["degree_phd"] = 1 if degree in ("phd", "doctorate") else 0

        df = pd.DataFrame([row])
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        return df[self.feature_names]

    def predict(self, data: dict) -> dict:
        if self.use_pipeline:
            X = self.build_feature_dataframe(data)
        else:
            X = self.build_feature_vector_legacy(data)

        prediction_val = int(self.pipeline.predict(X)[0])
        probabilities = self.pipeline.predict_proba(X)[0].tolist()
        depression_prob = probabilities[1]

        risk = get_risk_level(depression_prob, self.risk_q1, self.risk_q3)

        return {
            "prediction": prediction_val,
            "prediction_label": "Depressed" if prediction_val == 1 else "Not Depressed",
            "probability": {
                "not_depressed": round(probabilities[0], 4),
                "depressed": round(probabilities[1], 4)
            },
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "risk_percentile": risk["percentile"],
            "recommended_action": risk["action"],
            "input_features_used": X.to_dict(orient="records")[0],
            "pipeline_mode": "unified" if self.use_pipeline else "legacy"
        }
