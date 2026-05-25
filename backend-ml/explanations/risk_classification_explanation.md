# Risk Classification Explanation (`risk_analysis.py`)

The risk classification module implements a **hybrid dual-axis interpretation system** that assigns every student two independent assessments based on the trained model's predicted probability. It is the single source of truth for all risk-related logic in the project.

> **Important language distinction**: The ML model predicts **probability of similarity to depressive-class patterns**, NOT clinical severity. All labels and descriptions in this system use pattern-similarity language deliberately.

## Why a Hybrid System?

The original system used only percentile-based Q1/Q3 thresholds to classify students. This created a conceptual problem: a student with a probability of 0.92 (very high pattern similarity) could be classified as "Moderate Risk" simply because 75% of other students also scored high. The percentile system answered "how does this student compare to others?" but failed to answer "how strong is this student's pattern similarity?"

The hybrid system solves this by producing **two independent assessments**:

| Axis | Question it Answers | Method |
|------|-------------------|--------|
| **Severity Interpretation** | "How strongly do this student's responses match depressive-class patterns?" | Fixed absolute probability thresholds |
| **Institutional Priority** | "How should the institution prioritize this student relative to peers?" | Percentile-based Q1/Q3 ranking |

## Core Functions

### `get_severity_interpretation(prob)`
Maps a probability to a severity level using **fixed thresholds** that do not depend on Q1/Q3 or the distribution of other students:

- **High Risk Tendency** (probability > 0.85): High probability of similarity to depressive-class patterns
- **Elevated Tendency** (0.60 – 0.85): Moderate-to-high similarity to depressive-class patterns
- **Mild Tendency** (0.35 – 0.60): Some similarity to depressive-class patterns detected
- **Minimal Tendency** (< 0.35): Low similarity to depressive-class patterns

### `get_institutional_priority(prob, q1, q3)`
Maps a probability to an institutional priority tier using **percentile thresholds**:

- **High Priority** (probability > Q3): Top 25% of the distribution. Action: priority attention and further evaluation.
- **Moderate Priority** (Q1 ≤ probability ≤ Q3): Middle 50%. Action: monitoring and supportive interventions.
- **Low Priority** (probability < Q1): Bottom 25%. Action: general awareness level.

### `get_hybrid_risk(prob, q1, q3)`
Orchestrator function that returns both axes in a single call. This is the function used by the Flask prediction API.

### `compute_risk_thresholds(model, df, target)`
Takes the trained model and the full dataset, generates depression probabilities for every student, and computes the 25th percentile (Q1) and 75th percentile (Q3). These values define the institutional priority boundaries.

### `build_risk_thresholds_dict(q1, q3)`
Constructs a JSON-serializable dictionary documenting both frameworks — severity thresholds (fixed) and institutional priority thresholds (Q1/Q3). Used by `model_trainer.py` and when saving `risk_thresholds.json`.

### `generate_risk_framework(model, df, target, outdir)`
Batch pipeline function. Computes thresholds, labels every student, saves results. Uses institutional priority labels (Low/Moderate/High) for batch classification since this is the comparative ranking axis.

## Visualizations

### Risk Distribution Bar Chart
Color-coded bar chart (green/yellow/red) showing institutional priority distribution.

### Risk Score Density Plot
Histogram with KDE overlay showing the full probability distribution. Vertical dashed lines mark Q1 and Q3 thresholds.

### Institutional Priority Action Table
Formatted table mapping each priority tier to its percentile range, probability range, and recommended action.

### Severity Interpretation Table
Formatted table mapping each severity level to its fixed probability range and pattern-similarity meaning.

### Justification Report
Text file documenting both axes: fixed severity thresholds and percentile-based institutional priority, with the rationale for the hybrid approach.

## Where Constants Come From

All constants are imported from `config.py`:
- `SEVERITY_THRESHOLDS` and `SEVERITY_LABELS` — fixed probability cutoffs and their meanings
- `INSTITUTIONAL_ACTIONS` — recommended actions per priority tier
- `RISK_FRAMEWORK_JUSTIFICATION` — documentation of the hybrid approach

Backward-compatible aliases `RISK_ACTIONS` and `RISK_JUSTIFICATION` are also available for batch scripts.

## Outputs

All results are saved to `backend-ml/outputs/risk_classification/`:

| File | Description |
|---|---|
| `risk_assessment_output.csv` | Full dataset with Risk_Score and Risk_Level columns |
| `risk_thresholds.json` | Both severity and institutional framework definitions |
| `risk_distribution.png` | Bar chart of institutional priority counts |
| `risk_score_distribution.png` | Probability density with Q1/Q3 markers |
| `risk_summary.csv` | Priority tier counts and percentages |
| `risk_summary_table.png` | Styled summary table image |
| `risk_action_table.csv` | Institutional priority action mappings |
| `risk_action_table.png` | Styled institutional priority table image |
| `severity_interpretation_table.csv` | Severity level definitions |
| `severity_interpretation_table.png` | Styled severity table image |
| `risk_framework_justification.txt` | Full hybrid framework documentation |
