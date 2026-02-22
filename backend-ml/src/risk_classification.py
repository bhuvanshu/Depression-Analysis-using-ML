import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.utils import finalize_plot, ensure_outdir, save_pretty_table

def generate_risk_framework(model, df, target="Depression", outdir=Path("outputs")):
    """Calculates risk levels based on model prediction quartiles."""
    ensure_outdir(outdir)
    
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    probabilities = model.predict_proba(X)[:, 1]
    
    q1, q3 = np.quantile(probabilities, [0.25, 0.75])
    
    risk_labels = np.where(probabilities >= q3, "High",
                   np.where(probabilities >= q1, "Moderate", "Low"))
    
    assessment_df = df.copy()
    assessment_df["Risk_Score"] = probabilities
    assessment_df["Risk_Level"] = risk_labels
    
    assessment_df.to_csv(outdir / "risk_assessment_output.csv", index=False)
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
    
    # Probability Density Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(risk_df["Risk_Score"], bins=30, kde=True, color="#4682B4", alpha=0.6)
    plt.axvline(q1, color='#f39c12', linestyle='--', label=f'Q1 ({q1:.2f})')
    plt.axvline(q3, color='#e74c3c', linestyle='--', label=f'Q3 ({q3:.2f})')
    plt.legend()
    plt.xlabel("Predicted Risk Probability")
    finalize_plot(outdir / "risk_score_distribution.png", "Risk Score Density & Thresholds")

    # Summary Statistics Table
    total = len(risk_df)
    summary_df = pd.DataFrame([
        {"Risk Category": lvl, "Count": counts[lvl], "Percentage (%)": f"{(counts[lvl]/total*100):.2f}%"}
        for lvl in ["Low", "Moderate", "High"]
    ])
    
    summary_df.to_csv(outdir / "risk_summary.csv", index=False)
    save_pretty_table(summary_df, outdir / "risk_summary_table.png", "Risk Distribution Summary")

def save_risk_actions(q1, q3, outdir):
    """Maps probability ranges to specific clinical interventions."""
    actions = pd.DataFrame({
        "Risk Level": ["Low", "Moderate", "High"],
        "Percentile": ["Bottom 25%", "Middle 50%", "Top 25%"],
        "Probability Range": [f"0.00 – {q1:.2f}", f"{q1:.2f} – {q3:.2f}", f"≥ {q3:.2f}"],
        "Recommended Action": [
            "General wellness resources and awareness programs",
            "Counseling and stress management workshops",
            "Urgent psychological intervention and referral"
        ]
    })
    
    actions.to_csv(outdir / "risk_action_table.csv", index=False)
    save_pretty_table(actions, outdir / "risk_action_table.png", "Clinical Intervention Strategy")

if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    
    data_path = root / "data" / "df_ml.csv"
    if not data_path.exists():
        sys.exit(f"Data error: {data_path} not found.")
        
    df = pd.read_csv(data_path)
    target = "Depression"
    
    # Train reference model
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]
    
    print("Training risk analysis model...")
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    output_path = root / "outputs" / "risk_classification"
    
    print("Generating framework and visuals...")
    results, q1, q3 = generate_risk_framework(model, df, target, output_path)
    save_risk_visuals(results, q1, q3, output_path)
    save_risk_actions(q1, q3, output_path)
    
    print(f"Analysis complete. Results saved to: {output_path}")
