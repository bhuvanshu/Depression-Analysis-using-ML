import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import textwrap
from matplotlib.ticker import MaxNLocator

src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from utils import finalize_plot, safe_filename, save_text_report
from config import TARGET_COL, STYLE_SETTINGS, LABEL_MAPS, COLUMN_RENAMES, DEGREE_GROUP_ORDINAL

# ===============================
# CONFIGURATION
# ===============================
DATA_PATH = Path("backend-ml/data/df_paper.csv")
EDA_OUTDIR = Path("backend-ml/outputs/eda")
CORR_OUTDIR = Path("backend-ml/outputs/correlation")

EDA_OUTDIR.mkdir(parents=True, exist_ok=True)
CORR_OUTDIR.mkdir(parents=True, exist_ok=True)

# ===============================
# PLOTTING FUNCTIONS
# ===============================

def plot_missing(df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", yticklabels=False, xticklabels=True)
    plt.title("Missing Values Map", fontsize=13, fontweight="bold")
    plt.xlabel("Columns")
    plt.ylabel("Rows (Records)")
    plt.xticks(rotation=45, ha="right")
    missing_count = int(df.isna().sum().sum())
    n_rows = df.shape[0]
    if missing_count == 0:
        plt.suptitle(f"All features contained 0 missing entries after cleaning (N = {n_rows:,})", color="green", fontsize=10)
    else:
        plt.suptitle(f"Total Missing Values: {missing_count}", color="red", fontsize=10)
    finalize_plot(outpath)

def plot_correlations(df: pd.DataFrame, outpath: Path):
    num = df.select_dtypes(include=[np.number])
    num = num.dropna(axis=1, how='all')
    
    if num.shape[1] < 2: return
    
    corr = num.corr(method="spearman")
    
    new_labels = []
    for c in corr.columns:
        label = COLUMN_RENAMES.get(c, c)
        label = label.replace("Sleep_", "")
        wrapped = textwrap.fill(label, width=12)
        new_labels.append(wrapped)
        
    corr.columns = new_labels
    corr.index = new_labels

    plt.figure(figsize=(18, 16))
    annot_labels = corr.map(lambda x: f"{x:.2f}".replace("-0.00", "0.00"))
    
    sns.heatmap(corr, annot=annot_labels, fmt="", cmap=STYLE_SETTINGS["heatmap_cmap"], 
                xticklabels=True, yticklabels=True, vmin=-1, vmax=1, center=0, 
                annot_kws={"size": 16, "weight": "bold"},
                cbar_kws={"shrink": 0.8},
                linewidths=0.5)
    
    plt.xticks(rotation=45, ha="right", fontsize=18, fontweight='bold')
    plt.yticks(rotation=0, fontsize=18, fontweight='bold')
    plt.title("Spearman Correlation Matrix", fontsize=24, fontweight='bold', pad=30)
    
    plt.tight_layout()
    finalize_plot(outpath)

def plot_outliers_iqr(df: pd.DataFrame, outdir: Path, top_n=20):
    num = df.select_dtypes(include=[np.number])
    outlier_counts = {}
    for col in num.columns:
        col_vals = num[col].dropna()
        if col_vals.empty: continue
        q1, q3 = col_vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = ((col_vals < q1 - 1.5 * iqr) | (col_vals > q3 + 1.5 * iqr)).sum()
        outlier_counts[col] = int(count)
    sorted_counts = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_counts: return
    cols, counts = zip(*sorted_counts)
    plt.figure(figsize=(10, max(6, 0.5 * len(cols))))
    ax = sns.barplot(x=list(counts), y=list(cols), hue=list(cols), palette=STYLE_SETTINGS["palette"], legend=False)
    max_count = max(counts)
    for i, v in enumerate(counts):
        ax.text(v + max(1, max_count * 0.01), i, str(v), va="center")
    finalize_plot(outdir / "outlier_counts.png", "Outlier counts per numeric column (IQR method)")

def plot_outlier_boxplots(df: pd.DataFrame, outdir: Path, max_cols=12):
    num = df.select_dtypes(include=[np.number])
    cols = list(num.columns)[:max_cols]
    for c in cols:
        vals = num[c].dropna()
        if vals.empty: continue
        plt.figure(figsize=(10, 7))
        sns.boxplot(y=vals, width=0.5, boxprops=dict(facecolor="#c8f7c8", edgecolor="k"),
                    flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="k", markersize=6))
        plt.grid(axis="y", linestyle="-", color="#e6e6e6")
        ymin, ymax = vals.min(), vals.max()
        yrange = ymax - ymin if ymax > ymin else 1.0
        plt.ylim(ymin - 0.08 * yrange, ymax + 0.12 * yrange)
        if c in LABEL_MAPS:
            label_text = f"{c} Scale:\n" + "\n".join([f"{k}: {v}" for k, v in LABEL_MAPS[c].items()])
            plt.gca().text(0.98, 0.02, label_text, transform=plt.gca().transAxes, fontsize=9,
                           va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        finalize_plot(outdir / f"box_{safe_filename(c)}.png", f"Boxplot for {c} (Outlier Detection)")

def plot_attribute_distributions(df: pd.DataFrame, outdir: Path, max_categories=10):
    cat = df.select_dtypes(include=[object, "category"]).copy()
    num = df.select_dtypes(include=[np.number])
    for c in num.columns:
        if num[c].nunique(dropna=True) <= 10: cat[c] = df[c]
    if "CGPA" in num.columns:
        cat["CGPA"] = pd.cut(num["CGPA"], bins=[0, 2, 4, 6, 8, 10], 
                             labels=["0-2 (Poor)", "2-4 (Below Average)", "4-6 (Average)", "6-8 (Good)", "8-10 (Excellent)"])

    for c in cat.columns:
        vals = cat[c].dropna()
        if vals.empty: continue
        counts = vals.value_counts().sort_values(ascending=False)
        if counts.size > max_categories:
            counts = counts.nlargest(max_categories)
            other = vals.size - counts.sum()
            if other > 0: counts["Other"] = other
        
        if c in LABEL_MAPS: counts.index = counts.index.map(LABEL_MAPS[c])
        
        # Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel(c if c != "Depression" else "Depression Label", fontsize=12)
        ax.set_ylabel("Number of Records", fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        total_count = counts.sum()
        for i, v in enumerate(counts.values):
            pct = (v / total_count) * 100
            ax.text(i, v + max(counts.values) * 0.01, f"{v} ({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
        finalize_plot(outdir / f"bar_{safe_filename(c)}.png", f"Distribution: {c}")
        
        # Pie Chart
        plt.figure(figsize=(12, 7))
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        wedges, texts, autotexts = plt.pie(counts.values, labels=counts.index, autopct="%.1f%%", 
                                            startangle=90, colors=colors, textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('black'); autotext.set_fontweight('bold')
        legend_labels = [f"{label} - {count}" for label, count in zip(counts.index, counts.values)]
        plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
        finalize_plot(outdir / f"pie_{safe_filename(c)}.png", f"Distribution: {c}")

def plot_depression_analytics(df: pd.DataFrame, outdir: Path):
    if "Depression" not in df.columns: return
    
    # 1. Depression by Age
    if "Age" in df.columns:
        sel = df[df["Depression"] == 1]
        counts = sel["Age"].value_counts().sort_index()
        plt.figure(figsize=(12, 4))
        plt.plot(counts.index, counts.values, marker="o", linestyle="-", color="#2b7bba")
        plt.scatter(counts.index, counts.values, color="#2b7bba")
        plt.xlabel("Age")
        plt.ylabel("Number of People with Depression")
        plt.grid(True, linestyle="-", color="#e6e6e6")
        finalize_plot(outdir / "depression_by_age.png", "Number of People with Depression by Age Group")


def plot_impact_analysis(df: pd.DataFrame, outdir: Path):
    features = ["Academic Pressure", "Financial Stress", "Study Satisfaction", "Work/Study Hours"]
    plot_df = df.copy()
    plot_df["Depression_Label"] = plot_df["Depression"].apply(lambda x: "1 (Yes)" if x == 1 else ("0 (No)" if x == 0 else str(x)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(features):
        if col in df.columns:
            sns.boxplot(data=plot_df, x="Depression_Label", y=col, hue="Depression_Label", legend=False, ax=axes[i], palette="Set2")
            axes[i].set_title("") 
            axes[i].set_xlabel("Depression Class", fontsize=12)
            axes[i].set_ylabel(col, fontsize=12)
            axes[i].grid(axis="y", linestyle="--", alpha=0.7)
            axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.suptitle("Key Indicators-Group wise Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outdir / "impact_analysis_grid.png", dpi=150)
    plt.close()

def plot_degree_depression(df: pd.DataFrame, outdir: Path):
    """Depression rate and count by Degree_Group."""
    deg_col = "Degree_Group"
    dep_col = "Depression"
    if deg_col not in df.columns or dep_col not in df.columns:
        return

    order = ["School", "Undergraduate", "Postgraduate", "Doctorate", "Other"]
    order = [o for o in order if o in df[deg_col].unique()]

    grouped = df.groupby(deg_col)[dep_col].agg(["mean", "sum", "count"]).reindex(order)
    grouped.columns = ["Rate", "Depressed", "Total"]
    grouped["Not Depressed"] = grouped["Total"] - grouped["Depressed"]
    overall_rate = df[dep_col].mean()

    # Plot 1: Depression rate by degree group
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if r > overall_rate else "#2ecc71" for r in grouped["Rate"]]
    bars = ax.bar(grouped.index, grouped["Rate"], color=colors, edgecolor="black", alpha=0.85)
    for bar, total, rate in zip(bars, grouped["Total"], grouped["Rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.1%}\n(n={int(total)})", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(y=overall_rate, color='navy', linestyle='--', alpha=0.6,
               label=f'Overall Rate: {overall_rate:.1%}')
    ax.set_ylabel("Depression Rate", fontsize=12)
    ax.set_xlabel("Education Level (Degree Group)", fontsize=12)
    ax.set_ylim(0, min(1.0, grouped["Rate"].max() * 1.3))
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    finalize_plot(outdir / "degree_group_vs_depression.png",
                  "Depression Rate by Education Level")

    # Plot 2: Stacked bar (depression counts)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grouped.index, grouped["Depressed"], label="Depressed (1)",
           color="#e74c3c", edgecolor="black")
    ax.bar(grouped.index, grouped["Not Depressed"], bottom=grouped["Depressed"],
           label="Not Depressed (0)", color="#2ecc71", edgecolor="black")
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(i, row["Total"] + grouped["Total"].max() * 0.01,
                f"n={int(row['Total'])}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_xlabel("Education Level (Degree Group)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    finalize_plot(outdir / "degree_group_depression_stacked.png",
                  "Depression Distribution by Education Level")

    # Save interpretation
    highest = grouped["Rate"].idxmax()
    lowest = grouped["Rate"].idxmin()
    interp = (
        f"EDA Interpretation — Degree_Group:\n"
        f"Depression rates vary across education levels.\n"
        f"Highest rate: {highest} ({grouped.loc[highest, 'Rate']:.1%}), "
        f"Lowest rate: {lowest} ({grouped.loc[lowest, 'Rate']:.1%}).\n"
        f"Overall depression rate: {overall_rate:.1%}.\n"
    )
    save_text_report(outdir / "degree_group_interpretation.txt", interp)
    print(interp)

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    if not DATA_PATH.exists():
        print(f"❌ Data file not found at {DATA_PATH}. Please run cleaning.py first.")
        return

    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Capitalize columns for common formatting
    df.columns = [c.title() for c in df.columns]
    
    print(f"✅ Data Loaded. Shape: {df.shape}")

    # Basic Statistics
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())

    # Professional Plotting Orchestration
    print("\nGenerating Professional EDA Visuals...")
    
    # 1. Missing Values
    plot_missing(df, EDA_OUTDIR / "missing_values_map.png")
    
    # 2. Outliers
    plot_outliers_iqr(df, EDA_OUTDIR)
    plot_outlier_boxplots(df, EDA_OUTDIR)
    
    # 3. Distributions (Categorical & Binned Numerical)
    plot_attribute_distributions(df, EDA_OUTDIR)
    
    # 4. Depression Analytics (Target Specific)
    plot_depression_analytics(df, EDA_OUTDIR)
    
    # 5. Impact Analysis (Key Indicators vs Target)
    plot_impact_analysis(df, EDA_OUTDIR)

    # 6. Degree_Group vs Depression Analysis
    print("\nGenerating Degree_Group Analysis...")
    plot_degree_depression(df, EDA_OUTDIR)

    # 7. Encode Degree_Group ordinally for correlation (AFTER categorical plots)
    if "Degree_Group" in df.columns:
        df["Degree_Group"] = df["Degree_Group"].map(DEGREE_GROUP_ORDINAL).fillna(2)
        print("✅ Encoded Degree_Group ordinally for correlation analysis")

    # 8. Correlation Analysis (now includes Degree_Group as numeric)
    print("\nGenerating Correlation Analysis...")
    plot_correlations(df, CORR_OUTDIR / "correlation_heatmap.png")

    target = TARGET_COL.title()
    if target in df.columns:
        print(f"\n--- Top Correlations with {target} ---")
        correlations = df.select_dtypes(include=[np.number]).corr(method="spearman")[target].sort_values(ascending=False).dropna()
        print(correlations)

        # Save correlation interpretation for Degree_Group
        dg_label = COLUMN_RENAMES.get("Degree_Group", "Degree_Group")
        dg_corr = correlations.get("Degree_Group", None)
        if dg_corr is not None:
            strength = "negligible" if abs(dg_corr) < 0.1 else "weak" if abs(dg_corr) < 0.3 else "moderate"
            direction = "positive" if dg_corr > 0 else "negative"
            corr_interp = (
                f"Correlation Interpretation — Degree_Group:\n"
                f"Spearman correlation with Depression: {dg_corr:.4f} ({strength}, {direction}).\n"
                f"This suggests {'minimal' if strength == 'negligible' else 'limited' if strength == 'weak' else 'some'} "
                f"linear association between education level and depression.\n"
            )
            save_text_report(CORR_OUTDIR / "degree_group_correlation_interpretation.txt", corr_interp)
            print(corr_interp)

    print(f"\n✅ Professional EDA & Correlation Analysis complete.")
    print(f"Plots saved to:\n  - {EDA_OUTDIR}\n  - {CORR_OUTDIR}")

if __name__ == "__main__":
    main()