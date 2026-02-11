## 📊 Data Cleaning & Preprocessing

A structured preprocessing pipeline was implemented to ensure data quality and reproducibility.

The target variable used in this project is:

**`depression`**

- `0` → Not Depressed  
- `1` → Depressed  

### 1. Dataset Standardization
- Column names converted to lowercase
- Whitespace removed
- Duplicate records removed

### 2. Missing Value Handling
- Numeric columns → Median imputation
- Categorical columns → Mode imputation

### 3. Data Type Correction
- Automatically converted numeric-like object columns into numeric format

### 4. Age Filtering
- Dataset restricted to students aged 18–28

### 5. Encoding Strategy

#### Depression (Target)
- Binary encoded (0/1)

#### Suicidal Indicators
- Automatically detected and converted to binary format

#### Gender (Future-Ready Design)
- One-hot encoded instead of binary mapping
- Ensures scalability for additional gender categories in future dashboard deployment

#### Sleep Duration
Two representations created:
- Ordinal (1–4) → For PCA & statistical analysis
- One-hot encoding → For machine learning

---

## 📁 Generated Datasets

Two processed datasets are produced:

- `df_paper.csv` → Used for correlation and PCA
- `df_ml.csv` → Used for machine learning training