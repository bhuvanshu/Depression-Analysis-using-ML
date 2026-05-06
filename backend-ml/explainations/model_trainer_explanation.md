# Model Training Explanation (`model_trainer.py`)

The model trainer is the core ML script. It trains multiple classifiers, compares their performance, and saves the best model with deployment artifacts.

## How It Works

### 1. Data Loading
Reads `df_ml.csv` (one-hot encoded dataset). Detects the Depression target column automatically and separates features from the target.

### 2. Train/Test Split
80/20 split with stratification on the target variable and a fixed random seed (42) for reproducibility.

### 3. Models Trained

**Logistic Regression** — A linear baseline. Features are standardized first since linear models are scale-sensitive. Uses balanced class weights.

**Random Forest** — An ensemble of 200 decision trees. Does not require scaling. Provides a strong non-linear baseline.

**Gradient Boosting** — The primary model. Uses 200 trees with a conservative learning rate of 0.05 for better generalization. This model gets deployed.

**GB Ablation** — Gradient Boosting retrained without Suicidal Thoughts, showing how much performance depends on that single sensitive feature.

### 4. Evaluation Per Model
Each model generates: metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC), a text report, confusion matrix heatmap, ROC curve, and feature importance chart with readable labels from `COLUMN_RENAMES`.

### 5. Comparison
A styled PNG table and grouped bar chart compare F1-score and ROC-AUC across all models.

### 6. Deployment Artifacts
The Gradient Boosting model is saved with:
- `model.joblib` — serialized model
- `feature_names.joblib` — ordered feature names
- `risk_thresholds.json` — percentile-based thresholds from `risk_classification` module
- `model_metadata.json` — model info, metrics, and frontend form field mappings

### Design: Centralized Risk Thresholds
Threshold computation is delegated to `risk_classification.py` via `compute_risk_thresholds()` and `build_risk_thresholds_dict()`. This ensures one source of truth — thresholds from training match those used in classification and the API.

## How to Run
```bash
python backend-ml/src/model_trainer.py
```

## Outputs
| Location | Contents |
|---|---|
| `outputs/logistic_regression/` | LR metrics, CM, ROC, feature importance |
| `outputs/random_forest/` | RF metrics, CM, ROC, feature importance |
| `outputs/gradient_boosting/` | GB metrics + deployment artifacts |
| `outputs/gb_ablation/` | Ablation results (GB without Suicidal Thoughts) |
| `outputs/model_comparison.csv` | Metrics table |
| `outputs/model_comparison.png` | Styled comparison image |
| `outputs/model_comparison_bar.png` | F1 vs ROC-AUC bar chart |
