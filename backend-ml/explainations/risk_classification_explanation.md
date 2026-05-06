# Risk Classification Explanation (`risk_classification.py`)

The risk classification module assigns every student a risk level (Low, Moderate, or High) based on the trained model's predicted probability of depression. It is designed as the single source of truth for all risk-related logic in the project.

## Why This Module Exists

Risk classification touches three parts of the system: batch analysis (running over the entire dataset), model training (saving thresholds during deployment), and the prediction API (classifying individual predictions in real time). Without a centralized module, the same threshold computation and risk level logic would be duplicated across all three — leading to inconsistencies if any copy is updated independently. This module prevents that by exposing reusable functions that the other scripts import.

## Core Functions

### `compute_risk_thresholds(model, df, target)`
Takes the trained model and the full dataset, generates depression probabilities for every student, and computes the 25th percentile (Q1) and 75th percentile (Q3) of that distribution. These two values become the boundaries between Low, Moderate, and High risk. Returns `(q1, q3, probabilities)`.

### `get_risk_level(prob, q1, q3)`
Maps a single probability value to a risk category. This is the function used by the Flask API when processing individual predictions:
- **Low** (probability < Q1): Bottom 25% of the distribution. Action: general awareness level.
- **Moderate** (Q1 ≤ probability ≤ Q3): Middle 50%. Action: monitoring and supportive interventions.
- **High** (probability > Q3): Top 25%. Action: priority attention and further evaluation.

Returns a dictionary with the level name, display color, percentile label, and recommended action.

### `build_risk_thresholds_dict(q1, q3)`
Constructs a standardized JSON-serializable dictionary with the threshold values, method description, justification text, and detailed risk level definitions. Used by both `model_trainer.py` and this module itself when saving `risk_thresholds.json`.

### `generate_risk_framework(model, df, target, outdir)`
The batch pipeline function. Computes thresholds, labels every student as Low/Moderate/High, adds Risk_Score and Risk_Level columns to the dataset, and saves the result as `risk_assessment_output.csv` along with `risk_thresholds.json`.

## Visualizations

### Risk Distribution Bar Chart
A color-coded bar chart (green/yellow/red) showing how many students fall into each risk category.

### Risk Score Density Plot
A histogram with KDE overlay showing the full distribution of predicted probabilities. Vertical dashed lines mark the Q1 and Q3 thresholds, making it visually clear where the boundaries fall.

### Risk Summary Table
A styled PNG table with counts and percentages for each category.

### Risk Action Table
A formatted table mapping each risk level to its percentile range, probability range, and recommended action. Saved as both CSV and PNG.

### Justification Report
A text file documenting the method (percentile-based Q1/Q3), the computed threshold values, the risk level definitions, and the rationale for using this approach.

## How It Runs

When run standalone, the script loads the pre-trained model from `outputs/gradient_boosting/model.joblib` rather than training a new one. This ensures the risk analysis uses the exact same model that will serve predictions in production. If no saved model exists yet (e.g., first-time setup), it falls back to training a fresh model and prints a warning.

```bash
python backend-ml/src/risk_classification.py
```

## Where Constants Come From

The action descriptions and justification text are imported from `config.py` (`RISK_ACTIONS` and `RISK_JUSTIFICATION`). This means if the wording needs to change, it only needs to be updated in one place and will propagate to every report, table, API response, and saved JSON file automatically.

## Outputs

All results are saved to `backend-ml/outputs/risk_classification/`:

| File | Description |
|---|---|
| `risk_assessment_output.csv` | Full dataset with Risk_Score and Risk_Level columns |
| `risk_thresholds.json` | Q1/Q3 values and risk level definitions |
| `risk_distribution.png` | Bar chart of risk category counts |
| `risk_score_distribution.png` | Probability density with threshold markers |
| `risk_summary.csv` | Category counts and percentages |
| `risk_summary_table.png` | Styled summary table image |
| `risk_action_table.csv` | Action mappings in CSV format |
| `risk_action_table.png` | Styled action table image |
| `risk_framework_justification.txt` | Full methodology and rationale documentation |
