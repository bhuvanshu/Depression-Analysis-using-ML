# Prediction API Explanation (`serve_model.py`)

The serve_model script is a Flask web server that exposes the trained Gradient Boosting model as a REST API. It allows external applications — such as a frontend student form — to submit student data and receive depression risk predictions in real time.

## How It Works

### Startup

When the server starts, it loads artifacts saved during model training:

- **`pipeline.joblib`**: The production pipeline containing preprocessing + trained classifier.
- **`model_metadata.json`**: Model information, performance metrics, risk thresholds, and frontend form field definitions.

The risk thresholds (Q1 and Q3) are extracted from the metadata and stored in memory for the institutional priority axis. If any artifact fails to load, the server exits immediately with a clear error message.

### Feature Vector Construction

The API accepts user-friendly field names (like `age`, `gender`, `sleep_duration`) from a form. The production pipeline handles all preprocessing internally (one-hot encoding, feature ordering). For legacy model.joblib deployments, a manual `build_feature_vector_legacy()` function handles the translation.

### Hybrid Risk Interpretation

When classifying a prediction, the API calls `get_hybrid_risk()` imported from `inference/risk.py`. This returns **two independent assessments**:

1. **Severity Interpretation** — based on fixed probability thresholds, measures how strongly the student's responses match depressive-class patterns (absolute assessment, independent of other students)
2. **Institutional Priority** — based on percentile Q1/Q3 thresholds, ranks the student relative to the institutional population for resource allocation (comparative ranking)

> **Important**: The ML model predicts probability of similarity to depressive-class patterns, NOT clinical severity. All API response labels reflect this distinction.

## API Endpoints

### `POST /predict`
Accepts a JSON body with student data and returns:

**Primary fields** (frontend should use these):
- `severity_interpretation` — object with `level`, `score`, `meaning`, `color`
- `institutional_priority` — object with `tier`, `percentile_group`, `action`, `color`

**Backward-compatible fields** (Java backend consumes these internally):
- `risk_level` — maps to institutional priority tier (High/Moderate/Low)
- `risk_percentile` — maps to institutional priority percentile group
- `recommended_action` — maps to institutional priority action
- `risk_color` — color code for institutional priority

**Always present:**
- `prediction` — binary (0 or 1) and `prediction_label` ("Depressed"/"Not Depressed")
- `probability` — scores for both classes
- `input_features_used` — exact feature vector used (debugging)
- `pipeline_mode` — "unified" or "legacy"

### `GET /health`
Returns server status, model type, number of features, performance metrics, and the complete interpretation framework configuration documenting both severity and institutional axes.

### `GET /features`
Returns the expected input fields with metadata (types, labels, options, min/max values).

## Design Decisions

**Hybrid interpretation.** The API returns both severity (absolute) and institutional priority (comparative) for every prediction. This eliminates the conceptual confusion where a 0.92 probability could be labeled "Moderate" simply because other students also scored high.

**Backward compatibility.** The `risk_level`, `recommended_action`, and `risk_percentile` fields are still present and map to the institutional priority axis. The Java backend continues to work without changes.

**No local risk logic.** All risk interpretation functions are imported from `inference/risk.py`, which in turn imports constants from `training/config.py`. Risk logic is defined in exactly one place.

**Threshold loading.** The Q1/Q3 thresholds are loaded from `model_metadata.json` at startup rather than recomputed. This ensures the API uses the exact same thresholds computed during training.

## How to Run

```bash
python serve_model.py              # default: port 5000
python serve_model.py --port 8080  # custom port
gunicorn serve_model:app           # production (Render)
```

The server binds to `0.0.0.0` by default, making it accessible from other machines on the network.
