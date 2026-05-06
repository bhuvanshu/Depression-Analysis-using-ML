import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.utils import finalize_plot, ensure_outdir, save_pretty_table, save_text_report
from src.config import RISK_JUSTIFICATION, RISK_ACTIONS


def compute_risk_thresholds(model, df, target="Depression"):
    """Computes percentile-based risk thresholds (Q1/Q3) from prediction distribution.

    This is the single source of truth for threshold computation.
    Returns (q1, q3, probabilities).
    """
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    probabilities = model.predict_proba(X)[:, 1]
    q1 = float(np.quantile(probabilities, 0.25))
    q3 = float(np.quantile(probabilities, 0.75))
    return q1, q3, probabilities


def get_risk_level(prob: float, q1: float, q3: float) -> dict:
    """Maps a single probability to a risk level using percentile-based thresholds.

    Reusable by both batch risk classification and the Flask prediction API.
    """
    if prob > q3:
        return {"level": "High", "color": "#e74c3c",
                "percentile": "Top 25%",
                "action": RISK_ACTIONS["High"]}
    elif prob >= q1:
        return {"level": "Moderate", "color": "#f39c12",
                "percentile": "Middle 50%",
                "action": RISK_ACTIONS["Moderate"]}
    else:
        return {"level": "Low", "color": "#2ecc71",
                "percentile": "Bottom 25%",
                "action": RISK_ACTIONS["Low"]}


def build_risk_thresholds_dict(q1, q3):
    """Builds the standard risk thresholds dict for JSON serialization."""
    return {
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "method": "percentile-based",
        "justification": RISK_JUSTIFICATION,
        "risk_levels": {
            "Low":      {"range": f"probability < Q1 ({q1:.4f})",  "percentile": "Bottom 25%",
                         "action": RISK_ACTIONS["Low"]},
            "Moderate": {"range": f"Q1 ({q1:.4f}) \u2264 probability \u2264 Q3 ({q3:.4f})", "percentile": "Middle 50%",
                         "action": RISK_ACTIONS["Moderate"]},
            "High":     {"range": f"probability > Q3 ({q3:.4f})", "percentile": "Top 25%",
                         "action": RISK_ACTIONS["High"]}
        }
    }


def generate_risk_framework(model, df, target="Depression", outdir=Path("outputs")):
    """Calculates risk levels based on percentile-based prediction quartiles (Q1/Q3)."""
    ensure_outdir(outdir)

    q1, q3, probabilities = compute_risk_thresholds(model, df, target)

    risk_labels = np.where(probabilities > q3, "High",
                   np.where(probabilities >= q1, "Moderate", "Low"))

    assessment_df = df.copy()
    assessment_df["Risk_Score"] = probabilities
    assessment_df["Risk_Level"] = risk_labels

    assessment_df.to_csv(outdir / "risk_assessment_output.csv", index=False)

    # Save thresholds for reference
    thresholds = build_risk_thresholds_dict(q1, q3)
    with open(outdir / "risk_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    return assessment_df, q1, q3


def save_risk_visuals(risk_df, q1, q3, outdir):
    """Generates distribution plots and table summaries."""
    ensure_outdir(outdir)
    counts = risk_df["Risk_Level"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)

    # Risk Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax = counts.plot(kind="bar", color=colors, edgecolor="black", alpha=0.8, zorder=3)

    for i, v in enumerate(counts.values):
        ax.text(i, v + (counts.max() * 0.015), str(v), ha='center', fontweight='bold')

    plt.xticks(rotation=0)
    plt.xlabel("Risk Category")
    plt.ylabel("Student Count")
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    finalize_plot(outdir / "risk_distribution.png", "Mental Health Risk Distribution")

    # Probability Density Plot with Percentile Thresholds
    plt.figure(figsize=(10, 6))
    sns.histplot(risk_df["Risk_Score"], bins=30, kde=True, color="#4682B4", alpha=0.6)
    plt.axvline(q1, color='#f39c12', linestyle='--', label=f'Q1 = {q1:.4f} (25th percentile)')
    plt.axvline(q3, color='#e74c3c', linestyle='--', label=f'Q3 = {q3:.4f} (75th percentile)')
    plt.legend()
    plt.xlabel("Predicted Risk Probability")
    finalize_plot(outdir / "risk_score_distribution.png",
                  "Risk Score Density & Percentile-Based Thresholds")

    # Summary Statistics Table
    total = len(risk_df)
    summary_df = pd.DataFrame([
        {"Risk Category": lvl, "Count": counts[lvl],
         "Percentage (%)": f"{(counts[lvl]/total*100):.2f}%"}
        for lvl in ["Low", "Moderate", "High"]
    ])

    summary_df.to_csv(outdir / "risk_summary.csv", index=False)
    save_pretty_table(summary_df, outdir / "risk_summary_table.png", "Risk Distribution Summary")


def save_risk_actions(q1, q3, outdir):
    """Maps probability ranges to screening-oriented actions (non-clinical wording)."""
    actions = pd.DataFrame({
        "Risk Level": ["Low", "Moderate", "High"],
        "Percentile": ["Bottom 25%", "Middle 50%", "Top 25%"],
        "Probability Range": [f"0.00 \u2013 {q1:.4f}", f"{q1:.4f} \u2013 {q3:.4f}", f"> {q3:.4f}"],
        "Recommended Action": [
            RISK_ACTIONS["Low"],
            RISK_ACTIONS["Moderate"],
            RISK_ACTIONS["High"]
        ]
    })

    actions.to_csv(outdir / "risk_action_table.csv", index=False)
    save_pretty_table(actions, outdir / "risk_action_table.png",
                      "Risk Stratification Framework (Percentile-Based)")

    # Save justification text
    justification_text = (
        f"Risk Stratification Framework\n"
        f"{'=' * 40}\n\n"
        f"Method: Percentile-based (Q1/Q3) thresholds\n"
        f"Q1 (25th percentile): {q1:.4f}\n"
        f"Q3 (75th percentile): {q3:.4f}\n\n"
        f"Low Risk     (Bottom 25%):  probability < {q1:.4f}\n"
        f"  \u2192 {RISK_ACTIONS['Low']}\n\n"
        f"Moderate Risk (Middle 50%): {q1:.4f} \u2264 probability \u2264 {q3:.4f}\n"
        f"  \u2192 {RISK_ACTIONS['Moderate']}\n\n"
        f"High Risk    (Top 25%):     probability > {q3:.4f}\n"
        f"  \u2192 {RISK_ACTIONS['High']}\n\n"
        f"Justification:\n"
        f"{RISK_JUSTIFICATION}\n"
    )
    save_text_report(outdir / "risk_framework_justification.txt", justification_text)


if __name__ == "__main__":
    import joblib

    data_path = root / "data" / "df_ml.csv"
    model_path = root / "outputs" / "gradient_boosting" / "model.joblib"

    if not data_path.exists():
        sys.exit(f"Data error: {data_path} not found.")

    # Load pre-trained model (avoid retraining — model_trainer.py is the trainer)
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✅ Loaded pre-trained model from {model_path}")
    else:
        # Fallback: train if no saved model exists yet
        from sklearn.ensemble import GradientBoostingClassifier
        print("⚠️  No saved model found — training a fresh model for risk analysis...")
        df_temp = pd.read_csv(data_path)
        target = "Depression"
        if target not in df_temp.columns:
            matches = [c for c in df_temp.columns if c.lower() == "depression"]
            target = matches[0] if matches else sys.exit("Target column 'depression' not found.")
        X = df_temp.drop(columns=[target]).select_dtypes(include=[np.number])
        y = df_temp[target]
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        model.fit(X, y)

    df = pd.read_csv(data_path)

    # Resolve target column (case-insensitive)
    target = "Depression"
    if target not in df.columns:
        matches = [c for c in df.columns if c.lower() == "depression"]
        if matches:
            target = matches[0]
        else:
            sys.exit("Target column 'depression' not found in dataset.")

    output_path = root / "outputs" / "risk_classification"

    print("Generating percentile-based risk framework...")
    results, q1, q3 = generate_risk_framework(model, df, target, output_path)
    print(f"Thresholds: Q1 = {q1:.4f}, Q3 = {q3:.4f}")

    save_risk_visuals(results, q1, q3, output_path)
    save_risk_actions(q1, q3, output_path)

    print(f"\n✅ Risk classification complete. Results saved to: {output_path}")
