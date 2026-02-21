import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add the project root to sys.path to allow 'from src.xxx' imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.config import COLUMN_RENAMES
from src.utils import finalize_plot, save_text_report, safe_filename, ensure_outdir, save_pretty_table


def plot_feature_importance(model, feature_names, name, outdir):
    """Saves feature importance plot and CSV with readable column aliases."""
    # Support both tree-based (feature_importances_) and linear (coef_) models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    # Map raw column names to readable aliases from config
    rename_map = {k.lower(): v for k, v in COLUMN_RENAMES.items()}
    mapped_names = [rename_map.get(f.lower(), f.title()) for f in feature_names]

    df_imp = pd.DataFrame({"Feature": mapped_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    df_imp.to_csv(outdir / f"{safe_filename(name)}_features.csv", index=False)

    plt.figure(figsize=(10, 6))
    top10 = df_imp.head(10)
    bars = plt.barh(top10["Feature"], top10["Importance"],
                    color=plt.cm.viridis(np.linspace(0.7, 0.2, len(top10))), edgecolor='black')
    plt.gca().invert_yaxis()
    for bar in bars:
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.4f}', va='center')
    plt.xlim(0, top10["Importance"].max() * 1.15)
    finalize_plot(outdir / f"{safe_filename(name)}_feature_importance.png", f"Feature Importance ({name})")


def run_model_pipeline(model, X_train, X_test, y_train, y_test, name, outdir, feature_names=None):
    """Complete pipeline including training, metrics, CM, ROC, and Feature Importance."""
    ensure_outdir(outdir)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    }

    # Save Text Report
    report = f"{name} Classifier\n" + "-" * len(name) + "\n"
    for k, v in metrics.items():
        if k != "Model":
            report += f"{k:10}: {v:.4f}\n"
    report += f"\nClassification Report:\n{classification_report(y_test, y_pred)}"
    save_text_report(outdir / f"{safe_filename(name)}_report.txt", report)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    finalize_plot(outdir / f"{safe_filename(name)}_cm.png", f"Confusion Matrix - {name}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['ROC-AUC']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    finalize_plot(outdir / f"{safe_filename(name)}_roc.png", f"ROC Curve - {name}")

    # Feature Importance
    if feature_names is not None:
        plot_feature_importance(model, feature_names, name, outdir)

    return metrics, model


def plot_model_comparison_bar(results_df, outdir):
    """Generates grouped bar chart comparing F1-score and ROC-AUC across all models."""
    models = results_df["Model"]
    f1_scores = results_df["F1-score"]
    roc_scores = results_df["ROC-AUC"]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_scores, width, label="F1-score", color="#1f77b4", edgecolor="black", zorder=3)
    bars2 = ax.bar(x + width/2, roc_scores, width, label="ROC-AUC", color="#ff7f0e", edgecolor="black", zorder=3)

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)

    # Dynamic y-axis: start slightly below the minimum value
    all_vals = list(f1_scores) + list(roc_scores)
    y_min = max(0, min(all_vals) - 0.03)
    y_max = max(all_vals) + 0.025
    ax.set_ylim(y_min, y_max)

    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    finalize_plot(outdir / "model_comparison_bar.png", "Model Comparison using F1-score and ROC-AUC")


def train_all_models(df, target="Depression", outdir=Path("outputs")):
    """Trains all models and returns results + best model."""
    ensure_outdir(outdir)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []

    # 1. Logistic Regression (needs scaling)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    lr_m, _ = run_model_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"),
                                 X_tr_s, X_te_s, y_train, y_test,
                                 "Logistic Regression", outdir / "logistic_regression", X.columns)
    results.append(lr_m)

    # 2. Random Forest (200 trees for stability)
    rf_m, _ = run_model_pipeline(RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
                                 X_train, X_test, y_train, y_test,
                                 "Random Forest", outdir / "random_forest", X.columns)
    results.append(rf_m)

    # 3. Gradient Boosting (low lr + more trees = better generalization)
    gb_m, gb_model = run_model_pipeline(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
                                        X_train, X_test, y_train, y_test,
                                        "Gradient Boosting", outdir / "gradient_boosting", X.columns)
    results.append(gb_m)

    # 4. Ablation: GB without Suicidal Thoughts
    suicidal_col = [c for c in X.columns if "suicidal" in c.lower()]
    if suicidal_col:
        X_abl = X.drop(columns=suicidal_col)
        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_abl, y, test_size=0.2, random_state=42, stratify=y)
        gb_a_m, _ = run_model_pipeline(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
                                       X_tr_a, X_te_a, y_tr_a, y_te_a,
                                       "GB Ablation", outdir / "gb_ablation", X_abl.columns)
        results.append(gb_a_m)

    return results, gb_model


if __name__ == "__main__":
    data_path = root / "data" / "df_ml.csv"
    if not data_path.exists():
        sys.exit("Dataset not found!")

    df = pd.read_csv(data_path)

    # Resolve target column name (case-insensitive)
    target = "Depression"
    if target not in df.columns:
        target = [c for c in df.columns if c.lower() == "depression"][0]

    outdir = root / "outputs"
    results, best_model = train_all_models(df, target=target, outdir=outdir)

    # Save comparison table (CSV + pretty PNG)
    comp_df = pd.DataFrame(results).round(4)
    comp_df.to_csv(outdir / "model_comparison.csv", index=False)
    save_pretty_table(comp_df, outdir / "model_comparison.png", "Model Performance Comparison")

    # Generate grouped bar chart (F1-score vs ROC-AUC)
    plot_model_comparison_bar(comp_df, outdir)

    print("\n--- Model Comparison ---\n", comp_df)
    print(f"\nResults exported to: {outdir}")
