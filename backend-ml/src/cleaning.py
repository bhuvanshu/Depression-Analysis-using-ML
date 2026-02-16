import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# 1) SETTINGS
# ===============================
INPUT_FILE = "backend-ml/data/student_depression_dataset kaggle.csv"
OUTDIR = Path("backend-ml/data")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "depression"

# ===============================
# 2) LOAD DATASET
# ===============================
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()

print("✅ Raw Shape:", df.shape)

# ===============================
# 3) DROP DEGREE COLUMN
# ===============================
if "degree" in df.columns:
    df.drop(columns=["degree"], inplace=True)
    print("✅ Dropped column: degree")

# ===============================
# 4) BASIC CLEANING
# ===============================
df = df.drop_duplicates()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()

for col in df.columns:
    if df[col].dtype == "object":
        temp = pd.to_numeric(df[col], errors="coerce")
        if temp.notna().mean() > 0.85:
            df[col] = temp

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ===============================
# 5) FILTER AGE 18–28
# ===============================
if "age" not in df.columns:
    raise ValueError("❌ 'age' column not found!")

df = df[(df["age"] >= 18) & (df["age"] <= 28)].copy()
print("✅ After Age Filter (18–28):", df.shape)

# ===============================
# 6) BINARY ENCODING (Suicidal + Target ONLY)
# ===============================
def binarize_tf(series):
    s = series.astype(str).str.strip().str.lower()
    return s.map({
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "1": 1, "0": 0
    })


# Suicidal thoughts
suicide_cols = [c for c in df.columns if "suicid" in c]
for c in suicide_cols:
    df[c] = binarize_tf(df[c]).fillna(0).astype(int)

# Target
if TARGET_COL not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COL}' not found!")

if df[TARGET_COL].dtype == "object":
    df[TARGET_COL] = binarize_tf(df[TARGET_COL])

df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

# ===============================
# 7) DEFINE ENCODING SCHEMES
# ===============================

# --- GENDER SCHEME ---
# df_paper: Binary (Male=1, Female=0, Other=NaN -> Impute)
# df_ml: One-Hot (always gender_male, gender_female, gender_other)
gender_categories = ["male", "female", "other"]

def encode_gender_paper(series):
    s = series.astype(str).str.strip().str.lower()
    mapping = {"male": 1, "female": 0}
    # Any unknown category becomes NaN, then we'll fill with median (usually 0 or 1)
    return s.map(mapping)

# --- SLEEP SCHEME ---
# df_paper: Ordinal (1 to 4)
# df_ml: One-Hot (fixed categories)
sleep_categories = ["less than 5 hours", "5-6 hours", "7-8 hours", "more than 8 hours", "others"]
sleep_map = {
    "less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "more than 8 hours": 4,
    "others": 2  # default/median-ish
}

def encode_sleep_paper(series):
    s = series.astype(str).str.strip().str.lower()
    return s.map(sleep_map).fillna(2).astype(int)

# ===============================
# 8) GENERATE DATASET 1: df_paper (Research/PCA)
# ===============================
print("\n🛠 Generating df_paper (Research)...")

df_paper = df.copy()

# Gender -> Binary
df_paper["gender"] = encode_gender_paper(df_paper["gender"])
df_paper["gender"] = df_paper["gender"].fillna(df_paper["gender"].median()).astype(int)

# Sleep -> Ordinal
df_paper["sleep_duration"] = encode_sleep_paper(df_paper["sleep duration"])

# Select columns
paper_features = [
    "age", "academic pressure", "cgpa", "study satisfaction", 
    "work/study hours", "financial stress", 
    "sleep_duration", "gender", "family history of mental illness"
]
paper_cols = [c for c in paper_features if c in df_paper.columns] + suicide_cols + [TARGET_COL]

df_paper = df_paper[paper_cols].copy()
df_paper.to_csv(OUTDIR / "df_paper.csv", index=False)
print("✅ Saved df_paper.csv (Shape:", df_paper.shape, ")")

# ===============================
# 9) GENERATE DATASET 2: df_ml (Prediction/Dashboard)
# ===============================
print("\n🛠 Generating df_ml (ML/Dashboard)...")

df_ml = df.copy()

# A) Gender -> One-Hot (Guaranteed Columns)
# Using Categorical ensures that even if 'other' isn't in this slice, the column is created
df_ml["gender"] = pd.Categorical(df_ml["gender"].astype(str).str.strip().str.lower(), categories=gender_categories)
gender_dummies = pd.get_dummies(df_ml["gender"], prefix="gender").astype(int)

# B) Sleep -> One-Hot (Guaranteed Columns)
df_ml["sleep"] = pd.Categorical(df_ml["sleep duration"].astype(str).str.strip().str.lower(), categories=sleep_categories)
sleep_dummies = pd.get_dummies(df_ml["sleep"], prefix="sleep").astype(int)

# C) Combine
# drop the raw text columns and the newly created categoricals
df_ml = df_ml.drop(columns=["gender", "sleep", "sleep duration"], errors="ignore")
df_ml = pd.concat([df_ml, gender_dummies, sleep_dummies], axis=1)

# Select ML columns (all relevant numeric + dummies)
ml_exclude = ["id", "city", "profession", "degree"]
ml_cols = [c for c in df_ml.columns if c not in ml_exclude]
df_ml = df_ml[ml_cols].copy()

df_ml.to_csv(OUTDIR / "df_ml.csv", index=False)
print("✅ Saved df_ml.csv (Shape:", df_ml.shape, ")")
print("✅ ML Columns:", [c for c in df_ml.columns if "gender" in c or "sleep" in c])

print("\n✅ DONE ✅")
print("1) df_paper: Gender=Binary, Sleep=Ordinal (Optimized for PCA)")
print("2) df_ml:    Gender=OneHot, Sleep=OneHot (Optimized for Dashboard/ML)")