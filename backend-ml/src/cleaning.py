import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Set up paths (relative to this script's location, not the working directory)
root = Path(__file__).resolve().parents[1]  # backend-ml/
INPUT_FILE = root / "data" / "student_depression_dataset kaggle.csv"
OUTDIR = root / "data"
OUTDIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "depression"

# Load and clean column names
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()
print("✅ Raw Shape:", df.shape)

# NOTE: Degree column is kept for df_paper (Degree_Group mapping).
# It will NOT be included in df_ml.

# Data cleaning: duplicates and whitespace
df = df.drop_duplicates()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()

# Numeric conversion for columns that are mostly numbers
for col in df.columns:
    if df[col].dtype == "object":
        temp = pd.to_numeric(df[col], errors="coerce")
        if temp.notna().mean() > 0.85:
            df[col] = temp

# Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Subset data: Filter age (18-28)
if "age" not in df.columns:
    raise ValueError("❌ 'age' column not found!")
df = df[(df["age"] >= 18) & (df["age"] <= 28)].copy()
print("✅ After Age Filter (18–28):", df.shape)

# Binarization logic
def binarize_values(series):
    s = series.astype(str).str.strip().str.lower()
    return s.map({
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "1": 1, "0": 0
    })

# Binarize Suicidal Thoughts, Target, and Family History
suicide_cols = [c for c in df.columns if "suicid" in c]
for c in suicide_cols:
    df[c] = binarize_values(df[c]).fillna(0).astype(int)

df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

fam_hist_cols = [c for c in df.columns if "family history" in c]
for c in fam_hist_cols:
    df[c] = binarize_values(df[c]).fillna(0).astype(int)
    print(f"✅ Binarized: {c}")

# Encoding: Sleep Duration
if "sleep duration" not in df.columns:
    raise ValueError("❌ 'sleep duration' column not found!")

if df["sleep duration"].dtype == "object":
    # Clean quotes and map to ordinal values
    df["sleep duration"] = df["sleep duration"].astype(str).str.strip().str.replace("'", "").str.lower()
    sleep_map = {
        "less than 5 hours": 1,
        "5-6 hours": 2,
        "7-8 hours": 3,
        "more than 8 hours": 4,
        "others": 2
    }
    df["sleep_duration_ordinal"] = df["sleep duration"].map(sleep_map)
    valid_median = df["sleep_duration_ordinal"].median()
    df["sleep_duration_ordinal"] = df["sleep_duration_ordinal"].fillna(valid_median if pd.notna(valid_median) else 2)
else:
    df["sleep_duration_ordinal"] = pd.to_numeric(df["sleep duration"], errors="coerce").fillna(df["sleep duration"].median())

# Encoding: Gender
if "gender" in df.columns:
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    
    # Binary encoding for paper (0=female, 1=male)
    df["gender_binary"] = df["gender"].map({"male": 1, "female": 0}).fillna(1).astype(int)
    
    # Robust categorical type for ML one-hot encoding (includes 'other')
    df["gender_cat"] = pd.Categorical(df["gender"], categories=["male", "female", "other"])
    print("✅ Created gender encoding (Binary + Categorical)")

# ── Degree_Group: Categorical grouping for df_paper ──
if "degree" in df.columns:
    # Clean: remove quotes, trim whitespace, lowercase
    df["degree_clean"] = (
        df["degree"]
        .astype(str)
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .str.lower()
    )

    # Categorical grouping: all Bachelor-level → Undergraduate, Master-level → Postgraduate
    degree_map = {
        # School
        "class 12": "School",
        # Undergraduate (all Bachelor-level & entry professional degrees)
        "bsc": "Undergraduate",
        "ba": "Undergraduate",
        "bca": "Undergraduate",
        "be": "Undergraduate",
        "b.ed": "Undergraduate",
        "llb": "Undergraduate",
        "b.tech": "Undergraduate",
        "b.com": "Undergraduate",
        "b.arch": "Undergraduate",
        "bba": "Undergraduate",
        "bhm": "Undergraduate",
        "b.pharm": "Undergraduate",
        "mbbs": "Undergraduate",
        # Postgraduate (all Master-level & postgrad professional degrees)
        "m.tech": "Postgraduate",
        "msc": "Postgraduate",
        "mca": "Postgraduate",
        "mba": "Postgraduate",
        "ma": "Postgraduate",
        "m.ed": "Postgraduate",
        "m.com": "Postgraduate",
        "m.pharm": "Postgraduate",
        "llm": "Postgraduate",
        "md": "Postgraduate",
        "mhm": "Postgraduate",
        "me": "Postgraduate",
        # Doctorate
        "phd": "Doctorate",
        # Catch-all
        "others": "Other",
    }

    df["degree_group"] = df["degree_clean"].map(degree_map).fillna("Other")
    unmapped = df.loc[df["degree_group"] == "Other", "degree_clean"].unique()
    if len(unmapped) > 0:
        print(f"⚠️  Unmapped Degree values grouped as 'Other': {list(unmapped)}")
    print("✅ Created Degree_Group column")
    print(df["degree_group"].value_counts())

    # Drop the temp cleaning column
    df.drop(columns=["degree_clean"], inplace=True)
else:
    warnings.warn("'degree' column not found — skipping Degree_Group creation.")

# Export: Paper Dataset (Ordinal/Binary + Degree_Group)
paper_cols = [
    "age", "gender_binary", "academic pressure", "cgpa", 
    "study satisfaction", "work/study hours", "financial stress", 
    "sleep_duration_ordinal"
] + suicide_cols + fam_hist_cols

# Add Degree_Group if it was created
if "degree_group" in df.columns:
    paper_cols.append("degree_group")

paper_cols.append(TARGET_COL)

paper_cols = [c for c in paper_cols if c in df.columns]
df_paper = df[paper_cols].copy()
df_paper = df_paper.drop_duplicates()
df_paper.to_csv(OUTDIR / "df_paper.csv", index=False)
print(f"✅ Saved df_paper: {OUTDIR / 'df_paper.csv'} | Shape: {df_paper.shape}")

# Export: ML Dataset (One-Hot) — includes one-hot encoded Degree
df_ml = df_paper.drop(columns=["gender_binary", "sleep_duration_ordinal", "degree_group"], errors="ignore")
gender_onehot = pd.get_dummies(df["gender_cat"], prefix="gender", drop_first=False).astype(int)
sleep_onehot = pd.get_dummies(df["sleep duration"], prefix="sleep", drop_first=False).astype(int)

# One-hot encode Degree_Group into 4 clean columns
if "degree_group" in df.columns:
    degree_onehot = pd.DataFrame({
        "degree_school": (df["degree_group"] == "School").astype(int),
        "degree_undergrad": (df["degree_group"] == "Undergraduate").astype(int),
        "degree_postgrad": (df["degree_group"] == "Postgraduate").astype(int),
        "degree_phd": (df["degree_group"] == "Doctorate").astype(int),
    })
    df_ml = pd.concat([df_ml, gender_onehot, sleep_onehot, degree_onehot], axis=1)
else:
    df_ml = pd.concat([df_ml, gender_onehot, sleep_onehot], axis=1)

# Final dedup after all transformations
df_ml = df_ml.drop_duplicates()
df_ml.to_csv(OUTDIR / "df_ml.csv", index=False)
print(f"✅ Saved df_ml: {OUTDIR / 'df_ml.csv'} | Shape: {df_ml.shape}")

print("\n✅ Processing Complete.")