"""
Risk assessment and recommendation logic.
"""

RISK_JUSTIFICATION = (
    "Percentile-based thresholds are used to provide relative risk "
    "stratification without relying on arbitrary probability cutoffs, "
    "ensuring consistency across varying data distributions."
)

RISK_ACTIONS = {
    "Low":      "Indicates general awareness level",
    "Moderate": "Suggests monitoring and supportive interventions",
    "High":     "Requires priority attention and further evaluation"
}

def get_risk_level(prob: float, q1: float, q3: float) -> dict:
    """Maps a single probability to a risk level using percentile-based thresholds."""
    if prob > q3:
        return {"level": "High", "color": "#e74c3c",
                "percentile": "Top 25%", "action": RISK_ACTIONS["High"]}
    elif prob >= q1:
        return {"level": "Moderate", "color": "#f39c12",
                "percentile": "Middle 50%", "action": RISK_ACTIONS["Moderate"]}
    else:
        return {"level": "Low", "color": "#2ecc71",
                "percentile": "Bottom 25%", "action": RISK_ACTIONS["Low"]}
