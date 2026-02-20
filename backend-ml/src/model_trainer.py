import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Setup path so we can import 'src' modules reliably
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.append(str(root))

from src.config import COLUMN_RENAMES
from src.utils import finalize_plot, save_text_report, safe_filename, ensure_outdir, save_pretty_table

def handle_plots(y_test, y_pred, y_prob, model, feature_names, name, outdir):
    """Generates all visual artifacts (CM, ROC, Feature Importance)."""
    # 1. Confusion Matrix - helps see False Positives vs False Negatives
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    finalize_plot(outdir / f"{safe_filename(name)}_cm.png", f"Confusion Matrix - {name}")

    # 2. ROC Curve - shows the model's trade-off between sensitivity and specificity
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.legend()
    finalize_plot(outdir / f"{safe_filename(name)}_roc.png", f"ROC Curve - {name}")

    # 3. Feature Importance - which questions/factors matter most?
    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        imps = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_[0])
        rename_map = {k.lower(): v for k, v in COLUMN_RENAMES.items()}
        mapped_names = [rename_map.get(f.lower(), f.title()) for f in feature_names]
        
        df_imp = pd.DataFrame({"Feature": mapped_names, "Importance": imps}).sort_values(by="Importance", ascending=False)
        df_imp.to_csv(outdir / f"{safe_filename(name)}_features.csv", index=False)
        
        plt.figure(figsize=(10, 6))
        top10 = df_imp.head(10)
        plt.barh(top10["Feature"], top10["Importance"], color=plt.cm.viridis(np.linspace(0.7, 0.2, 10)), edgecolor='black')
        plt.gca().invert_yaxis()
        finalize_plot(outdir / f"{safe_filename(name)}_fimp.png", f"Top 10 Features ({name})")

def run_pipeline(model, X_train, X_test, y_train, y_test, name, outdir, feature_names):
    """Trains the model and returns performance metrics."""
    ensure_outdir(outdir)
    model.fit(X_train, y_train)
    y_pred, y_prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Model": name, "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred), "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred), "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    # Save a detailed text report for the research logs
    report = f"{name}\n" + "="*len(name) + f"\nMetrics:\n{metrics}\n\nReport:\n{classification_report(y_test, y_pred)}"
    save_text_report(outdir / f"{safe_filename(name)}_report.txt", report)
    
    handle_plots(y_test, y_pred, y_prob, model, feature_names, name, outdir)
    return metrics

if __name__ == "__main__":
    data_path = root / "data" / "df_ml.csv"
    if not data_path.exists(): sys.exit("Dataset not found!")
    df = pd.read_csv(data_path)
    
    # Isolate targets and features
    target = "depression"
    if target not in df.columns: target = [c for c in df.columns if c.lower() == target][0]
    X, y = df.drop(columns=[target]).select_dtypes(include=[np.number]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Quick Scaling for LR
    sc = StandardScaler()
    XT_s, Xe_s = sc.fit_transform(X_train), sc.transform(X_test)
    
    # Execute Model Loop
    results = [
        run_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"), XT_s, Xe_s, y_train, y_test, 
                     "Logistic Regression", root / "outputs" / "logistic_regression", X.columns),
        run_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"), 
                     X_train, X_test, y_train, y_test, "Random Forest", root / "outputs" / "random_forest", X.columns)
    ]
    
    # Save comparison results
    comp_df = pd.DataFrame(results).round(4)
    out_path = root / "outputs"
    ensure_outdir(out_path)
    comp_df.to_csv(out_path / "model_comparison.csv", index=False)
    save_pretty_table(comp_df, out_path / "model_comparison.png", "Model Performance Comparison")
    
    print("\n--- Model Comparison ---\n", comp_df)
    print(f"\nResults exported to: {out_path}")
