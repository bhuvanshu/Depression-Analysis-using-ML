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
# 7) SLEEP ORDINAL (FOR PAPER / PCA)
# ===============================
if "sleep duration" not in df.columns:
    raise ValueError("❌ 'sleep duration' column not found!")

if df["sleep duration"].dtype == "object":
    sleep_map = {
        "less than 5 hours": 1,
        "5-6 hours": 2,
        "7-8 hours": 3,
        "more than 8 hours": 4
    }

    df["sleep_duration_ordinal"] = (
        df["sleep duration"]
        .astype(str).str.strip().str.lower()
        .map(sleep_map)
    )

    df["sleep_duration_ordinal"] = df["sleep_duration_ordinal"].fillna(
        df["sleep_duration_ordinal"].median()
    )
else:
    df["sleep_duration_ordinal"] = pd.to_numeric(
        df["sleep duration"], errors="coerce"
    ).fillna(df["sleep duration"].median())

# ===============================
# 8) ONE‑HOT ENCODE GENDER 
# ===============================
if "gender" in df.columns:
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()

    gender_onehot = pd.get_dummies(
        df["gender"],
        prefix="gender",
        drop_first=False
    ).astype(int)

    df = pd.concat([df.drop(columns=["gender"]), gender_onehot], axis=1)

    print("✅ Gender One‑Hot Columns:", gender_onehot.columns.tolist())

# ===============================
# 9) PAPER DATASET
# ===============================
paper_cols = [
    "age",
    "academic pressure",
    "cgpa",
    "study satisfaction",
    "work/study hours",
    "financial stress",
    "sleep_duration_ordinal",
] + suicide_cols + [TARGET_COL]

# Add gender columns automatically
gender_cols = [c for c in df.columns if c.startswith("gender_")]
paper_cols += gender_cols

paper_cols = [c for c in paper_cols if c in df.columns]

df_paper = df[paper_cols].copy()
df_paper.to_csv(OUTDIR / "df_paper.csv", index=False)

print("\n✅ Saved df_paper:", OUTDIR / "df_paper.csv")

# ===============================
# 10) ML DATASET (Sleep One‑Hot)
# ===============================
df_ml = df_paper.copy()

df_ml = df_ml.drop(columns=["sleep_duration_ordinal"], errors="ignore")

sleep_onehot = pd.get_dummies(
    df["sleep duration"],
    prefix="sleep",
    drop_first=False
).astype(int)

df_ml = pd.concat([df_ml, sleep_onehot], axis=1)

df_ml.to_csv(OUTDIR / "df_ml.csv", index=False)

print("\n✅ Saved df_ml:", OUTDIR / "df_ml.csv")
print("✅ df_ml shape:", df_ml.shape)

print("\n✅ DONE ✅")
print("1) df_paper.csv = PCA/correlation ready")
print("2) df_ml.csv    = ML ready")
