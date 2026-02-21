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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import finalize_plot, save_text_report
from src.config import TARGET_COL, COLUMN_RENAMES, LABEL_MAPS
      

# -------------------------------------------------------------
# Helper: select numeric features consistently
# -------------------------------------------------------------
def get_numeric_features(df: pd.DataFrame, target: str):
    """
    Returns numeric feature dataframe X and the feature names.
    Ensures correlation, PCA, and biplot all use the exact same X.
    """
    X = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).copy()
    feature_names = X.columns.tolist()
    return X, feature_names


# -------------------------------------------------------------
# 2) PCA Scree Plot
# -------------------------------------------------------------
def plot_pca_scree(evr: np.ndarray, outpath: Path, title="PCA Explained Variance Trend"):
    pc_labels = [f"PC{i+1}" for i in range(len(evr))]
    evr_pct = evr * 100
    cum_evr_pct = np.cumsum(evr_pct)

    plt.figure(figsize=(9, 5))
    plt.bar(pc_labels, evr_pct, alpha=0.9, label="Individual")
    plt.plot(pc_labels, cum_evr_pct, marker="o", label="Cumulative", linewidth=1.5)

    plt.ylabel("Explained Variance (%)", fontsize=12)
    plt.xlabel("Principal Component", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)

    finalize_plot(outpath, title)


# -------------------------------------------------------------
# 3) PCA PC1 vs PC2 Scatter
# -------------------------------------------------------------
def plot_pc1_pc2_scatter(X_pca: np.ndarray, target_series: pd.Series, evr: np.ndarray, outpath: Path,
                         title="PCA Projection colored by Depression"):
    plt.figure(figsize=(10, 8))

    # Convert target to readable labels if binary
    mapping = LABEL_MAPS.get(TARGET_COL, {0: "0 (No)", 1: "1 (Yes)"})
    if set(target_series.dropna().unique()) <= set(mapping.keys()):
        target_labels = target_series.map(mapping)
    else:
        target_labels = target_series.astype(str)

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=target_labels,      
        alpha=0.35,
        edgecolor="black",
        linewidth=0.3
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(linestyle="--", alpha=0.4)

    plt.xlabel(f"PC1 ({evr[0]:.2%} variance)", fontsize=20, fontweight="bold")
    plt.ylabel(f"PC2 ({evr[1]:.2%} variance)", fontsize=20, fontweight="bold")
    plt.legend(title="", loc="upper right", fontsize=16, prop={'weight':'bold'})
    plt.title(title, fontsize=22, fontweight="bold")

    finalize_plot(outpath, title)


# -------------------------------------------------------------
# 4) PCA Loading Plot (Biplot)
# -------------------------------------------------------------
def plot_pca_biplot(X_pca: np.ndarray, pca: PCA, feature_names: list, outpath: Path,
                    evr: np.ndarray, title="PCA Loading Plot (Biplot)"):

    plt.figure(figsize=(14, 11))

    # Background points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.15, s=15, edgecolors="none", color='steelblue')

    # Scaling factor for arrows
    x_max = np.max(np.abs(X_pca[:, 0]))
    y_max = np.max(np.abs(X_pca[:, 1]))
    scale_factor = min(x_max, y_max) * 0.75

    # 1. Collect loading data
    loadings = []
    for i, feat in enumerate(feature_names):
        xi = pca.components_[0, i] * scale_factor
        yi = pca.components_[1, i] * scale_factor
        angle = np.degrees(np.arctan2(yi, xi))
        
        # Use formatted column name from config if available
        display_name = COLUMN_RENAMES.get(feat, feat)
        
        loadings.append({
            "name": display_name,
            "x": xi,
            "y": yi,
            "angle": angle,
            "mag": np.sqrt(xi**2 + yi**2)
        })

    # 2. Sort by angle to detect neighbors
    loadings.sort(key=lambda x: x["angle"])

    # 3. Detect angular collisions and assign radial boost
    # If two labels are within 18 degrees, boost the outer one further
    radial_boosts = [1.15] * len(loadings)
    for i in range(len(loadings)):
        for j in range(i + 1, len(loadings)):
            diff = abs(loadings[i]["angle"] - loadings[j]["angle"])
            if diff < 18:
                radial_boosts[j] *= 1.25 # Push neighbor further out

    # 4. Plot Arrows and Labels
    for i, l in enumerate(loadings):
        # Draw Arrow
        plt.arrow(0, 0, l["x"], l["y"], color="red", alpha=0.8,
                  head_width=0.08, linewidth=1.5, length_includes_head=True)

        # Determine alignment based on quadrant
        ha = "left" if l["x"] > 0 else "right"
        va = "bottom" if l["y"] > 0 else "top"
        
        # Fine-tune center labels
        if abs(l["x"]) < 0.1: ha = "center"
        if abs(l["y"]) < 0.1: va = "center"

        # Multi-line wrap for long feature names
        import textwrap
        display_name = textwrap.fill(l["name"], width=12)

        plt.text(
            l["x"] * radial_boosts[i],
            l["y"] * radial_boosts[i],
            display_name,
            color="black",
            ha=ha,
            va=va,
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="yellow",
                alpha=0.8,
                edgecolor="black",
                boxstyle="round,pad=0.3"
            )
        )

    plt.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    plt.axvline(0, color="black", linewidth=1.0, alpha=0.5)
    plt.grid(alpha=0.2, linestyle='--')

    plt.xlabel(f"PC1 ({evr[0]:.2%} variance)", fontsize=20, fontweight="bold", labelpad=12)
    plt.ylabel(f"PC2 ({evr[1]:.2%} variance)", fontsize=20, fontweight="bold", labelpad=12)
    plt.title(title, fontsize=26, fontweight="bold", y=1.03)

    plt.tight_layout()
    finalize_plot(outpath, title)


# -------------------------------------------------------------
# 5) Save PCA interpretation report
# -------------------------------------------------------------
def save_pca_interpretation(loadings: pd.DataFrame, evr: np.ndarray, outpath: Path, top_n=5):
    report = ["=== PCA COMPONENT INTERPRETATION ===\n"]

    for i in range(len(evr)):
        pc_name = f"PC{i+1}"
        report.append(f"\n{pc_name} (Explains {evr[i]:.2%} of variance)")
        report.append("-" * 60)
        report.append("Top contributing features:")

        comp = loadings[pc_name]
        top_contrib = comp.abs().sort_values(ascending=False).head(top_n)

        for feat in top_contrib.index:
            sign = "+" if comp[feat] > 0 else "-"
            report.append(f"  {sign} {feat}: {comp[feat]:.4f}")

    save_text_report(outpath, "\n".join(report))


# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------
def run_pca_pipeline(df: pd.DataFrame, outdir: Path, target="Depression"):
    """
    Fixed PCA pipeline:
    - Uses exact same numeric feature set in correlation + PCA
    - No label mismatch in heatmap
    - Saves PCA plots + loadings + interpretation
    """

    pca_dir = outdir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 1) Prepare consistent X
    # ---------------------------
    X, feature_names = get_numeric_features(df, target)

    if X.shape[1] < 2:
        raise ValueError("Not enough numeric features for PCA (need at least 2).")

    # ---------------------------
    # 3) Standardize
    # ---------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------
    # 4) Fit PCA
    # ---------------------------
    n_components = min(8, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    # ---------------------------
    # 5) Save EVR + Loadings
    # ---------------------------
    save_text_report(pca_dir / "pca_explained.txt", str(evr))

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(evr))],
        index=feature_names
    )
    loadings.to_csv(pca_dir / "pca_components.csv")

    # ---------------------------
    # 6) Scree Plot
    # ---------------------------
    plot_pca_scree(evr, pca_dir / "pca_scree.png")

    # ---------------------------
    # 7) PC1 vs PC2 Plot
    # ---------------------------
    plot_pc1_pc2_scatter(
        X_pca,
        df[target],
        evr,
        pca_dir / "pca_pc1_vs_pc2.png"
    )

    # ---------------------------
    # 8) Biplot / Loadings Plot
    # ---------------------------
    plot_pca_biplot(
        X_pca,
        pca,
        feature_names,
        pca_dir / "pca_loading_plot.png",
        evr
    )

    # ---------------------------
    # 9) Save Transformed Data
    # ---------------------------
    transformed_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(len(evr))]
    )
    transformed_df[target] = df[target].values
    transformed_df.to_csv(pca_dir / "pca_transformed.csv", index=False)

    # ---------------------------
    # 10) Save PCA Interpretation
    # ---------------------------
    save_pca_interpretation(loadings, evr, pca_dir / "pca_interpretation.txt", top_n=5)

    print(f"[DONE] PCA outputs saved to: {pca_dir}")


def main():
    # Configuration
    data_path = root / "data" / "df_paper.csv"
    out_dir = root / "outputs"
    
    if not data_path.exists():
        print(f"❌ Data file not found at {data_path}. Please run cleaning.py first.")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    # Capitalize columns for common formatting (consistent with eda.py)
    df.columns = [c.title() for c in df.columns]
    
    target = TARGET_COL
    if target not in df.columns:
        # Fallback if target naming is inconsistent
        target = "Depression"
        
    print(f"✅ Data Loaded. Shape: {df.shape}")
    print(f"Running PCA Pipeline for target: {target}")
    
    run_pca_pipeline(df, out_dir, target=target)


if __name__ == "__main__":
    main()
