"""
Hybrid Risk Interpretation System
==================================
Provides dual-axis risk assessment:
  1. Severity Interpretation — absolute probability thresholds (pattern-similarity)
  2. Institutional Priority  — percentile-based Q1/Q3 ranking (comparative)

The ML model predicts probability of similarity to depressive-class patterns.
This module interprets that probability along both axes.
"""

try:
    from training.config import (
        SEVERITY_THRESHOLDS, SEVERITY_LABELS,
        INSTITUTIONAL_ACTIONS, RISK_FRAMEWORK_JUSTIFICATION,
    )
except ImportError:
    # Fallback if training module is not on the path (e.g., standalone deployment)
    SEVERITY_THRESHOLDS = {"high_risk": 0.85, "elevated": 0.60, "mild": 0.35}

    SEVERITY_LABELS = {
        "High Risk Tendency": {
            "range": "> 0.85",
            "meaning": "High probability of similarity to depressive-class patterns",
            "color": "#dc2626",
        },
        "Elevated Tendency": {
            "range": "0.60 – 0.85",
            "meaning": "Moderate-to-high similarity to depressive-class patterns",
            "color": "#ea580c",
        },
        "Mild Tendency": {
            "range": "0.35 – 0.60",
            "meaning": "Some similarity to depressive-class patterns detected",
            "color": "#f59e0b",
        },
        "Minimal Tendency": {
            "range": "< 0.35",
            "meaning": "Low similarity to depressive-class patterns",
            "color": "#22c55e",
        },
    }

    INSTITUTIONAL_ACTIONS = {
        "High":     "Requires priority attention and further evaluation",
        "Moderate": "Suggests monitoring and supportive interventions",
        "Low":      "Indicates general awareness level",
    }

    RISK_FRAMEWORK_JUSTIFICATION = (
        "This system uses a hybrid dual-axis interpretation: "
        "(1) Severity Interpretation uses fixed probability thresholds to assess "
        "how strongly a student's responses match depressive-class patterns, independent of other students. "
        "(2) Institutional Priority uses percentile-based Q1/Q3 thresholds to rank students "
        "relative to the institutional population for resource allocation and prioritization."
    )


# ── AXIS 1: Severity Interpretation (absolute, no Q1/Q3) ──

def get_severity_interpretation(prob: float) -> dict:
    """Maps a probability to a severity level using fixed thresholds.

    This represents how strongly the student's responses match depressive-class
    patterns, independent of how other students scored.
    """
    t = SEVERITY_THRESHOLDS

    if prob > t["high_risk"]:
        level = "High Risk Tendency"
    elif prob > t["elevated"]:
        level = "Elevated Tendency"
    elif prob > t["mild"]:
        level = "Mild Tendency"
    else:
        level = "Minimal Tendency"

    info = SEVERITY_LABELS[level]
    return {
        "level": level,
        "score": round(prob, 4),
        "meaning": info["meaning"],
        "color": info["color"],
    }


# ── AXIS 2: Institutional Priority (percentile-based Q1/Q3) ──

def get_institutional_priority(prob: float, q1: float, q3: float) -> dict:
    """Maps a probability to an institutional priority tier using percentile thresholds.

    This represents how the student ranks relative to the institutional population
    for resource allocation and intervention prioritization.
    """
    if prob > q3:
        tier, percentile = "High", "Top 25%"
    elif prob >= q1:
        tier, percentile = "Moderate", "Middle 50%"
    else:
        tier, percentile = "Low", "Bottom 25%"

    return {
        "tier": tier,
        "percentile_group": percentile,
        "action": INSTITUTIONAL_ACTIONS[tier],
        "color": {"High": "#e74c3c", "Moderate": "#f39c12", "Low": "#2ecc71"}[tier],
    }


# ── Hybrid Orchestrator ──

def get_hybrid_risk(prob: float, q1: float, q3: float) -> dict:
    """Returns both severity interpretation and institutional priority for a probability.

    This is the primary function called by the prediction API.
    """
    return {
        "severity": get_severity_interpretation(prob),
        "priority": get_institutional_priority(prob, q1, q3),
    }


# ── Backward-compatible alias (used by training/risk_analysis.py) ──

def get_risk_level(prob: float, q1: float, q3: float) -> dict:
    """Legacy function — returns institutional priority in the old format.

    Kept for backward compatibility with batch training scripts.
    """
    priority = get_institutional_priority(prob, q1, q3)
    return {
        "level": priority["tier"],
        "color": priority["color"],
        "percentile": priority["percentile_group"],
        "action": priority["action"],
    }
