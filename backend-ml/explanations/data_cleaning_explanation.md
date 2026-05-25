# Data Cleaning & Preprocessing Explanation (`cleaning.py`)

The cleaning script takes the raw Kaggle dataset and transforms it into two analysis-ready datasets. The goal is to handle messy real-world data — inconsistent formatting, missing values, mixed data types — and produce clean, standardized outputs that the rest of the pipeline can rely on.

## How It Works

### 1. Loading and Standardization

The raw CSV file is loaded and all column names are immediately converted to lowercase with whitespace trimmed. This prevents subtle bugs caused by inconsistent capitalization (e.g., `"Age"` vs `"age"` vs `" Age "`). Duplicate rows are also removed at this stage.

### 2. Missing Value Handling

Rather than dropping incomplete records (which would reduce the dataset size), the script fills in gaps using simple imputation:

- **Numeric columns** → filled with the column's median (robust to outliers).
- **Categorical columns** → filled with the most common value (mode).

### 3. Data Type Correction

Some columns that should be numeric are stored as text in the raw data. The script automatically detects these by attempting numeric conversion — if more than 85% of a column's values successfully convert, the entire column is treated as numeric. This avoids manual column-by-column inspection.

### 4. Age Filtering

The dataset is restricted to students aged 18 to 28. This ensures the analysis focuses on the target population (college-age students) and removes outliers that may represent data entry errors or populations outside the study's scope.

### 5. Binary Encoding

Several columns contain yes/no or true/false values in various formats. The script standardizes these into clean 0/1 integers:

- **Depression** (target variable): `0` = Not Depressed, `1` = Depressed.
- **Suicidal Thoughts**: Automatically detected by column name and converted.
- **Family History of Mental Illness**: Same treatment.

### 6. Sleep Duration Encoding

Sleep duration is originally a text field with values like `"Less than 5 hours"` or `"7-8 hours"`. The script creates two representations:

- **Ordinal** (1–4): A numeric ranking used in correlation analysis and PCA, where the order of categories matters.
- **One-hot encoded**: A separate binary column for each sleep category, used by machine learning models that cannot assume ordinal relationships.

### 7. Gender Encoding

Gender is handled with two strategies to support different use cases:

- **Binary** (0/1): A simple Male/Female encoding used in the paper dataset for statistical analysis.
- **Categorical (one-hot)**: Preserves all categories including "Other", creating separate `gender_male`, `gender_female`, and `gender_other` columns. This is used for machine learning to avoid forcing a binary assumption.

### 8. Degree Group Creation

The raw `degree` column contains many specific values (BSc, BA, BCA, BE, M.Tech, PhD, Class 12, etc.). These are grouped into broader education levels to make the feature more meaningful for analysis:

| Original Values | Degree Group |
|---|---|
| Class 12 | School |
| BSc, BA, BCA, BE, B.Ed, LLB, B.Tech, B.Com, B.Arch, BBA, BHM, B.Pharm, MBBS | Undergraduate |
| M.Tech, MSc, MCA, MBA, MA, M.Ed, M.Com, M.Pharm, LLM, MD, MHM, ME | Postgraduate |
| PhD | Doctorate |

Before mapping, the values are cleaned: quotes are removed, whitespace is trimmed, and everything is converted to lowercase. Any values that don't match the mapping are labeled as `"Other"` and flagged with a warning.

The original `degree` column is kept unchanged — only a new `degree_group` column is added.

## Output Datasets

The script produces two separate CSV files, each designed for a different purpose:

### `df_paper.csv` — For Statistical Analysis
Contains ordinal/binary encodings and the Degree Group column. Used by the EDA, correlation, and PCA scripts. Features include: Age, Gender (binary), Academic Pressure, CGPA, Study Satisfaction, Work/Study Hours, Financial Stress, Sleep Duration (ordinal), Suicidal Thoughts, Family History, Degree Group, and Depression.

### `df_ml.csv` — For Machine Learning
Contains one-hot encoded versions of Gender, Sleep Duration, and Degree Group. Used by the model training pipeline. This format ensures the ML models don't incorrectly assume ordinal relationships in categorical features.

The `df_ml.csv` is generated from `df_paper.csv` by replacing binary/ordinal columns with their one-hot equivalents — so the same underlying data drives both outputs.