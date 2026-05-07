# Prediction API Explanation (`serve_model.py`)

The serve_model script is a Flask web server that exposes the trained Gradient Boosting model as a REST API. It allows external applications — such as a frontend student form — to submit student data and receive depression risk predictions in real time.

## How It Works

### Startup

When the server starts, it loads three artifacts saved during model training:

- **`model.joblib`**: The trained Gradient Boosting classifier.
- **`feature_names.joblib`**: The exact ordered list of features the model expects.
- **`model_metadata.json`**: Model information, performance metrics, risk thresholds, and frontend form field definitions.

The risk thresholds (Q1 and Q3) are extracted from the metadata and stored in memory. If any artifact fails to load, the server exits immediately with a clear error message.

### Feature Vector Construction

The API accepts user-friendly field names (like `age`, `gender`, `sleep_duration`) from a form, but the model expects a 20-column one-hot encoded vector. The `build_feature_vector()` function handles this translation:

- **Scalar features** (age, CGPA, academic pressure, etc.) are passed through directly.
- **Gender** is converted from a single string ("Male", "Female", "Other") into three binary columns (`gender_male`, `gender_female`, `gender_other`).
- **Sleep Duration** is converted from a string ("5-6 hours", "Less than 5 hours", etc.) into five binary columns.
- **Degree** is converted from a string ("School", "Undergraduate", "Postgraduate", "PhD") into four binary columns.

Any model features that aren't present in the input are filled with 0, and the columns are reordered to match the exact training order.

### Risk Classification

When classifying a prediction, the API calls `get_risk_level()` imported from `risk_classification.py`. This is the same function used by the batch risk analysis, ensuring the API and offline analysis always produce identical risk labels for the same probability.

The justification text displayed in the health endpoint is imported from `config.py` (`RISK_JUSTIFICATION`), maintaining consistency with all other outputs.

## API Endpoints

### `POST /predict`
Accepts a JSON body with student data and returns:
- The binary prediction (0 or 1) and its label ("Not Depressed" / "Depressed")
- Probability scores for both classes
- Risk level (Low / Moderate / High) with color code and percentile bracket
- A recommended action based on the risk level
- The exact feature vector used for prediction (useful for debugging)

### `GET /health`
Returns server status, model type, number of features, performance metrics, and the risk framework configuration (method, thresholds, justification).

### `GET /features`
Returns the expected input fields with metadata (types, labels, options, min/max values). This endpoint can be used by a frontend to dynamically generate a form that matches the model's requirements.

## Design Decisions

**No local risk logic.** Previous versions of this file contained a local copy of the risk level classification function and hardcoded action strings. These have been removed. The server now imports `get_risk_level` from `risk_classification.py` and `RISK_JUSTIFICATION` from `config.py`, so all risk-related behavior is defined in exactly one place.

**Threshold loading.** The Q1/Q3 thresholds are loaded from `model_metadata.json` at startup rather than recomputed. This ensures the API uses the exact same thresholds that were computed during training.

## How to Run

```bash
python serve_model.py              # default: port 5000
python serve_model.py --port 8080  # custom port
```

The server binds to `0.0.0.0` by default, making it accessible from other machines on the network.
