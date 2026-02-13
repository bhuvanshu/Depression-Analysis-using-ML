import numpy as np

# Column Name Mappings
COLUMN_RENAMES = {
    "Have You Ever Had Suicidal Thoughts ?": "Suicidal",
    "Academic Pressure": "Acad. Pressure",
    "Study Satisfaction": "Satisfaction",
    "Work/Study Hours": "Hours/Day",
    "Financial Stress": "Fin. Stress",
    "Family History Of Mental Illness": "Family Hist.",
    "Gender_Female": "Female",
    "Gender_Male": "Male"
}

# Feature Lists
CORE_FEATURES = [
    "Gender", "Age", "Academic Pressure", "CGPA", 
    "Study Satisfaction", "Sleep Duration", "Dietary Habits", 
    "Suicidal Thoughts", "Work/Study Hours", "Financial Stress", 
    "Family History"
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
    }
}
