import numpy as np

# Column Name Mappings
COLUMN_RENAMES = {
    "Have You Ever Had Suicidal Thoughts ?": "Suicidal Thoughts",
    "Academic Pressure": "Academic Pressure",
    "Study Satisfaction": "Study Satisfaction",
    "Work/Study Hours": "Work/Study Hours",
    "Financial Stress": "Financial Stress",
    "Family History Of Mental Illness": "Family History",
    "Gender_Binary": "Gender",
    "Sleep_Duration_Ordinal": "Sleep Duration",
    "Cgpa": "CGPA",
    "Gender_Female": "Female",
    "Gender_Male": "Male",
    "Degree_Group": "Degree Group",
    "Degree_School": "School",
    "Degree_Undergrad": "Undergraduate",
    "Degree_Postgrad": "Postgraduate",
    "Degree_Phd": "Doctorate"
}

# Feature Lists
CORE_FEATURES = [
    "Gender", "Age", "Academic Pressure", "CGPA", 
    "Study Satisfaction", "Sleep Duration",
    "Suicidal Thoughts", "Work/Study Hours", "Financial Stress", 
    "Family History", "Degree Group"
]

TARGET_COL = "Depression"

# Plotting Settings
STYLE_SETTINGS = {
    "palette": "viridis",
    "heatmap_cmap": "coolwarm",
    "figure_dpi": 150,
    "font_size_title": 16,
    "font_size_label": 14,
    "pie_colors": ["#90EE90", "#FF6B6B"]
}

# Extensive Scale Mappings
LABEL_MAPS = {
    "Gender": {0: "Female", 1: "Male"},
    "Academic Pressure": {
        0: "Very Low", 1: "Low", 2: "Moderate", 
        3: "High", 4: "Very High", 5: "Extreme"
    },
    "Sleep Duration": {
        1: "Less Than 5 Hours", 2: "5-6 Hours", 
        3: "7-8 Hours", 4: "More Than 8 Hours"
    },
    "Financial Stress": {
        0: "No Stress", 1: "Low", 2: "Moderate",
        3: "High", 4: "Very High", 5: "Extreme"
    },
    "Study Satisfaction": {
        0: "Very Dissatisfied", 1: "Dissatisfied", 2: "Neutral",
        3: "Satisfied", 4: "Very Satisfied", 5: "Extremely Satisfied"
    },
    "Suicidal Thoughts": {0: "No", 1: "Yes"},
    "Family History": {0: "No", 1: "Yes"},
    "Depression": {0: "0 (No)", 1: "1 (Yes)"},
    "Work/Study Hours": {
        0: "0-2h", 1: "2-4h", 2: "4-6h", 3: "6-8h", 4: "8-10h", 5: "10-12h"
    },
    "Degree Group": {
        1: "School", 2: "Undergraduate", 3: "Postgraduate", 4: "Doctorate"
    }
}

# Ordinal encoding for Degree_Group (used in correlation & PCA)
DEGREE_GROUP_ORDINAL = {
    "School": 1,
    "Undergraduate": 2,
    "Postgraduate": 3,
    "Doctorate": 4,
    "Other": 2  # treat same as Undergraduate
}

# ── Hybrid Risk Interpretation Framework (single source of truth) ──

# AXIS 1: Severity Interpretation — absolute probability thresholds
# These are independent of Q1/Q3 and represent pattern-similarity strength.
SEVERITY_THRESHOLDS = {
    "high_risk": 0.85,
    "elevated": 0.60,
    "mild": 0.35,
}

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

# AXIS 2: Institutional Priority — percentile-based Q1/Q3 ranking
# These use the same Q1/Q3 logic as before but with explicit naming.
INSTITUTIONAL_ACTIONS = {
    "High":     "Requires priority attention and further evaluation",
    "Moderate": "Suggests monitoring and supportive interventions",
    "Low":      "Indicates general awareness level",
}

# Framework documentation
RISK_FRAMEWORK_JUSTIFICATION = (
    "This system uses a hybrid dual-axis interpretation: "
    "(1) Severity Interpretation uses fixed probability thresholds to assess "
    "how strongly a student's responses match depressive-class patterns, independent of other students. "
    "(2) Institutional Priority uses percentile-based Q1/Q3 thresholds to rank students "
    "relative to the institutional population for resource allocation and prioritization."
)

# Backward-compatible aliases (used by training/risk_analysis.py batch scripts)
RISK_JUSTIFICATION = RISK_FRAMEWORK_JUSTIFICATION
RISK_ACTIONS = INSTITUTIONAL_ACTIONS
