"""
Production Pipeline Builder
────────────────────────────
Builds a unified sklearn Pipeline that encapsulates ALL preprocessing
(one-hot encoding for gender/sleep/degree, passthrough for numerics)
and the GradientBoostingClassifier into ONE deployable artifact.

The resulting `pipeline.joblib` accepts raw frontend-style input and
produces predictions without any external preprocessing code.

Usage:
    python -m src.build_pipeline

Preserves:
    - Same model type (GradientBoostingClassifier)
    - Same hyperparameters (n_estimators=200, learning_rate=0.05, random_state=42)
    - Same training data and split (random_state=42, stratify, test_size=0.2)
    - Same evaluation methodology
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

# ── Paths ──
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from training.config import RISK_JUSTIFICATION, RISK_ACTIONS
from training.risk_classification import build_risk_thresholds_dict


# ═══════════════════════════════════════════
#  RAW INPUT SCHEMA (what the frontend sends)
# ═══════════════════════════════════════════

# These are the 11 raw fields the frontend form collects.
# The pipeline must accept exactly these columns.
RAW_FEATURE_SCHEMA = [
    "age",                  # int 18-28
    "academic_pressure",    # int 0-5
    "cgpa",                 # float 0-10
    "study_satisfaction",   # int 0-5
    "work_study_hours",     # int 0-12
    "financial_stress",     # float 0-5
    "suicidal_thoughts",    # bool/int 0,1
    "family_history",       # bool/int 0,1
    "gender",               # str: Male, Female, Other
    "sleep_duration",       # str: Less than 5 hours, 5-6 hours, 7-8 hours, More than 8 hours, Others
    "degree",               # str: School, Undergraduate, Postgraduate, PhD/Doctorate
]

# Column groups for ColumnTransformer
NUMERIC_COLS = [
    "age", "academic_pressure", "cgpa", "study_satisfaction",
    "work_study_hours", "financial_stress", "suicidal_thoughts", "family_history"
]

CATEGORICAL_COLS = {
    "gender": ["Female", "Male", "Other"],
    "sleep_duration": ["5-6 hours", "7-8 hours", "Less than 5 hours",
                       "More than 8 hours", "Others"],
    "degree": ["School", "Undergraduate", "Postgraduate", "Doctorate"],
}


def build_raw_dataframe_from_csv(csv_path: Path, target: str = "depression"):
    """
    Reconstructs raw-style DataFrame from the original Kaggle CSV.

    Instead of using the pre-encoded df_ml.csv, we go back to the raw data
    and apply the same cleaning + degree grouping, but keep the categorical
    columns as strings (for the Pipeline's OneHotEncoder to handle).
    """
    raw_path = csv_path.parent / "student_depression_dataset kaggle.csv"
    if not raw_path.exists():
        # Fallback: reconstruct from df_ml.csv (reverse one-hot encoding)
        return _reconstruct_from_ml_csv(csv_path, target)

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip().str.lower()

    # Deduplicate
    df = df.drop_duplicates()

    # Clean string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Numeric coercion for mostly-numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            temp = pd.to_numeric(df[col], errors="coerce")
            if temp.notna().mean() > 0.85:
                df[col] = temp

    # Missing value handling
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Age filter (same as cleaning.py)
    df = df[(df["age"] >= 18) & (df["age"] <= 28)].copy()

    # Binarize suicidal thoughts
    def binarize(series):
        s = series.astype(str).str.strip().str.lower()
        return s.map({
            "true": 1, "false": 0, "yes": 1, "no": 0,
            "y": 1, "n": 0, "1": 1, "0": 0
        })

    suicide_cols = [c for c in df.columns if "suicid" in c]
    for c in suicide_cols:
        df[c] = binarize(df[c]).fillna(0).astype(int)

    # Binarize target
    df[target] = df[target].fillna(0).astype(int)

    # Binarize family history
    fam_cols = [c for c in df.columns if "family history" in c]
    for c in fam_cols:
        df[c] = binarize(df[c]).fillna(0).astype(int)

    # Degree grouping (same as cleaning.py)
    if "degree" in df.columns:
        df["degree_clean"] = (
            df["degree"].astype(str).str.strip()
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.strip().str.lower()
        )
        degree_map = {
            "class 12": "School",
            "bsc": "Undergraduate", "ba": "Undergraduate", "bca": "Undergraduate",
            "be": "Undergraduate", "b.ed": "Undergraduate", "llb": "Undergraduate",
            "b.tech": "Undergraduate", "b.com": "Undergraduate", "b.arch": "Undergraduate",
            "bba": "Undergraduate", "bhm": "Undergraduate", "b.pharm": "Undergraduate",
            "mbbs": "Undergraduate",
            "m.tech": "Postgraduate", "msc": "Postgraduate", "mca": "Postgraduate",
            "mba": "Postgraduate", "ma": "Postgraduate", "m.ed": "Postgraduate",
            "m.com": "Postgraduate", "m.pharm": "Postgraduate", "llm": "Postgraduate",
            "md": "Postgraduate", "mhm": "Postgraduate", "me": "Postgraduate",
            "phd": "Doctorate",
            "others": "Other",
        }
        df["degree"] = df["degree_clean"].map(degree_map).fillna("Other")
        # Treat "Other" as "Undergraduate" for consistency
        df["degree"] = df["degree"].replace("Other", "Undergraduate")
        df.drop(columns=["degree_clean"], inplace=True)

    # Sleep duration: keep as categorical string, clean
    if "sleep duration" in df.columns:
        df["sleep_duration"] = (
            df["sleep duration"].astype(str).str.strip()
            .str.replace("'", "", regex=False).str.lower()
        )
        # Capitalize properly for OneHotEncoder categories
        sleep_capitalize = {
            "less than 5 hours": "Less than 5 hours",
            "5-6 hours": "5-6 hours",
            "7-8 hours": "7-8 hours",
            "more than 8 hours": "More than 8 hours",
            "others": "Others",
        }
        df["sleep_duration"] = df["sleep_duration"].map(sleep_capitalize).fillna("Others")
        df.drop(columns=["sleep duration"], inplace=True)

    # Gender: keep as categorical string, clean
    if "gender" in df.columns:
        gender_capitalize = {"male": "Male", "female": "Female", "other": "Other"}
        df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(gender_capitalize).fillna("Other")

    # Rename columns to match frontend schema
    rename_map = {
        "academic pressure": "academic_pressure",
        "study satisfaction": "study_satisfaction",
        "work/study hours": "work_study_hours",
        "financial stress": "financial_stress",
        "have you ever had suicidal thoughts ?": "suicidal_thoughts",
        "family history of mental illness": "family_history",
    }
    df.rename(columns=rename_map, inplace=True)

    # Select only the columns we need
    keep_cols = RAW_FEATURE_SCHEMA + [target]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after processing: {missing}")

    df = df[keep_cols].copy()
    df = df.drop_duplicates()

    return df, target


def _reconstruct_from_ml_csv(csv_path: Path, target: str):
    """Fallback: reverse-engineer raw columns from the pre-encoded df_ml.csv."""
    df = pd.read_csv(csv_path)

    # Reverse one-hot for gender
    gender_map = {
        "gender_male": "Male", "gender_female": "Female", "gender_other": "Other"
    }
    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    df["gender"] = df[gender_cols].idxmax(axis=1).map(gender_map)

    # Reverse one-hot for sleep
    sleep_cols = [c for c in df.columns if c.startswith("sleep_")]
    sleep_map = {
        "sleep_5-6 hours": "5-6 hours",
        "sleep_7-8 hours": "7-8 hours",
        "sleep_less than 5 hours": "Less than 5 hours",
        "sleep_more than 8 hours": "More than 8 hours",
        "sleep_others": "Others",
    }
    df["sleep_duration"] = df[sleep_cols].idxmax(axis=1).map(sleep_map)

    # Reverse one-hot for degree
    degree_cols = [c for c in df.columns if c.startswith("degree_")]
    degree_map = {
        "degree_school": "School",
        "degree_undergrad": "Undergraduate",
        "degree_postgrad": "Postgraduate",
        "degree_phd": "Doctorate",
    }
    df["degree"] = df[degree_cols].idxmax(axis=1).map(degree_map)

    # Drop encoded columns
    df.drop(columns=gender_cols + sleep_cols + degree_cols, inplace=True)

    # Rename to frontend schema
    rename_map = {
        "academic pressure": "academic_pressure",
        "study satisfaction": "study_satisfaction",
        "work/study hours": "work_study_hours",
        "financial stress": "financial_stress",
        "have you ever had suicidal thoughts ?": "suicidal_thoughts",
        "family history of mental illness": "family_history",
    }
    df.rename(columns=rename_map, inplace=True)

    keep_cols = RAW_FEATURE_SCHEMA + [target]
    df = df[keep_cols].copy()
    df = df.drop_duplicates()

    return df, target


def build_production_pipeline():
    """
    Builds and saves the unified production pipeline.

    Returns the pipeline, metrics dict, and risk thresholds.
    """
    data_path = root / "data" / "df_ml.csv"
    if not data_path.exists():
        sys.exit(f"Dataset not found: {data_path}")

    print("=" * 60)
    print("  BUILDING PRODUCTION-SAFE PIPELINE")
    print("=" * 60)

    # ── Step 1: Load raw-format data ──
    print("\n[1/6] Loading and preparing raw-format training data...")
    df, target = build_raw_dataframe_from_csv(data_path)
    print(f"      Dataset shape: {df.shape}")
    print(f"      Target column: {target}")
    print(f"      Features: {[c for c in df.columns if c != target]}")

    # ── Step 2: Build ColumnTransformer ──
    print("\n[2/6] Building ColumnTransformer with OneHotEncoder...")

    preprocessor = ColumnTransformer(
        transformers=[
            ("gender_ohe", OneHotEncoder(
                categories=[CATEGORICAL_COLS["gender"]],
                sparse_output=False,
                handle_unknown="infrequent_if_exist"
            ), ["gender"]),
            ("sleep_ohe", OneHotEncoder(
                categories=[CATEGORICAL_COLS["sleep_duration"]],
                sparse_output=False,
                handle_unknown="infrequent_if_exist"
            ), ["sleep_duration"]),
            ("degree_ohe", OneHotEncoder(
                categories=[CATEGORICAL_COLS["degree"]],
                sparse_output=False,
                handle_unknown="infrequent_if_exist"
            ), ["degree"]),
        ],
        remainder="passthrough",  # numeric columns pass through unchanged
        verbose_feature_names_out=False,
    )

    # ── Step 3: Build Pipeline ──
    print("\n[3/6] Assembling Pipeline (preprocessor + GradientBoostingClassifier)...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
        ))
    ])

    # ── Step 4: Train with same split ──
    print("\n[4/6] Training pipeline with same split (random_state=42, stratify)...")
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    print(f"      Training complete. Samples: train={len(X_train)}, test={len(X_test)}")

    # ── Step 5: Evaluate ──
    print("\n[5/6] Evaluating pipeline...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
    }

    print(f"\n      {'Metric':<12} {'Value':>8}")
    print(f"      {'-'*12} {'-'*8}")
    for k, v in metrics.items():
        print(f"      {k:<12} {v:>8.4f}")

    print(f"\n      Classification Report:\n{classification_report(y_test, y_pred)}")

    # ── Step 6: Compute risk thresholds ──
    print("[6/6] Computing risk thresholds (percentile-based)...")
    all_probs = pipeline.predict_proba(X)[:, 1]
    q1 = float(np.quantile(all_probs, 0.25))
    q3 = float(np.quantile(all_probs, 0.75))
    print(f"      Q1 = {q1:.4f}, Q3 = {q3:.4f}")

    risk_thresholds = build_risk_thresholds_dict(q1, q3)

    # ── Save pipeline ──
    deploy_dir = root / "outputs" / "gradient_boosting"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = deploy_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"\n[OK] Pipeline saved to: {pipeline_path}")

    # Save updated risk thresholds
    with open(deploy_dir / "risk_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(risk_thresholds, f, indent=2, ensure_ascii=False)

    # Save updated metadata
    metadata = {
        "model_type": "GradientBoostingClassifier",
        "pipeline_type": "sklearn.pipeline.Pipeline (preprocessor + classifier)",
        "accepts_raw_input": True,
        "n_raw_features": len(RAW_FEATURE_SCHEMA),
        "raw_feature_schema": RAW_FEATURE_SCHEMA,
        "preprocessing": {
            "gender": "OneHotEncoder → [Female, Male, Other]",
            "sleep_duration": "OneHotEncoder → [5-6 hours, 7-8 hours, Less than 5 hours, More than 8 hours, Others]",
            "degree": "OneHotEncoder → [School, Undergraduate, Postgraduate, Doctorate]",
            "numeric_cols": "Passthrough (no scaling needed for GradientBoosting)",
        },
        "target": target,
        "hyperparameters": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": 42,
        },
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "risk_thresholds": risk_thresholds,
        "form_field_mapping": {
            "age": {"type": "number", "label": "Age", "min": 18, "max": 28},
            "academic_pressure": {"type": "select", "label": "Academic Pressure", "options": [0,1,2,3,4,5]},
            "cgpa": {"type": "number", "label": "CGPA", "min": 0, "max": 10, "step": 0.1},
            "study_satisfaction": {"type": "select", "label": "Study Satisfaction", "options": [0,1,2,3,4,5]},
            "work_study_hours": {"type": "number", "label": "Work/Study Hours per Day", "min": 0, "max": 12},
            "financial_stress": {"type": "select", "label": "Financial Stress", "options": [0,1,2,3,4,5]},
            "suicidal_thoughts": {"type": "boolean", "label": "Have you ever had suicidal thoughts?"},
            "family_history": {"type": "boolean", "label": "Family history of mental illness?"},
            "gender": {"type": "select", "label": "Gender", "options": ["Male", "Female", "Other"]},
            "sleep_duration": {"type": "select", "label": "Sleep Duration", "options": [
                "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"
            ]},
            "degree": {"type": "select", "label": "Degree / Education Level", "options": [
                "School", "Undergraduate", "Postgraduate", "Doctorate"
            ]},
        },
        "deployment_notes": {
            "artifact": "pipeline.joblib",
            "input_format": "DataFrame with 11 raw columns matching form_field_mapping keys",
            "no_external_preprocessing_needed": True,
            "legacy_model.joblib": "Still present for backward compatibility; requires manual preprocessing"
        }
    }
    with open(deploy_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metadata saved to: {deploy_dir / 'model_metadata.json'}")

    # ── Verify: encoded feature names after transform ──
    try:
        encoded_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        print(f"\n[INFO] Encoded feature names ({len(encoded_names)}):")
        for i, name in enumerate(encoded_names):
            print(f"       [{i:2d}] {name}")
    except Exception:
        pass

    return pipeline, metrics, risk_thresholds


if __name__ == "__main__":
    build_production_pipeline()
    print("\n" + "=" * 60)
    print("  PIPELINE BUILD COMPLETE")
    print("=" * 60)
