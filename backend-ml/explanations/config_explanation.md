# Config Explanation (`config.py`)

The `config.py` file acts as the single source of truth for all shared settings, mappings, and constants used across the entire backend pipeline. Every analysis script — from EDA to model training to the prediction API — imports its configuration from this one place, ensuring consistency and making future changes easy.

## What It Contains

### 1. Column Renaming (`COLUMN_RENAMES`)

A dictionary that translates raw column names from the dataset into cleaner, more readable labels. This is used by the plotting and reporting functions so that chart axes and table headers display professional names rather than raw database column names.

- **Example**: The column `"Have You Ever Had Suicidal Thoughts ?"` is displayed as `"Suicidal Thoughts"` in all visualizations.
- The degree-related columns (`Degree_School`, `Degree_Undergrad`, etc.) are also mapped here so feature importance plots and heatmaps show meaningful labels.

### 2. Feature Definitions

- **`CORE_FEATURES`**: The primary list of independent variables the project analyzes. This includes demographic, academic, and mental health indicators like Gender, CGPA, Sleep Duration, Financial Stress, and Degree Group.
- **`TARGET_COL`**: Identifies `"Depression"` as the dependent variable (the outcome we are predicting) for all scripts in the pipeline.

### 3. Global Styling (`STYLE_SETTINGS`)

Centralizes the visual look-and-feel of all generated charts and plots, so every output has a consistent, polished appearance:

- **`palette`**: Default color scheme (viridis) for bar charts and scatter plots.
- **`heatmap_cmap`**: The colormap used for correlation matrices (`coolwarm` provides intuitive red-blue contrast).
- **`figure_dpi`**: Sets the export resolution to 150 DPI for crisp, publication-quality images.
- **`pie_colors`**: Custom green and red colors used for binary distribution pie charts (e.g., Depressed vs Not Depressed).

### 4. Label Mapping (`LABEL_MAPS`)

Many features in the dataset are stored as numbers (0, 1, 2, ...) for analysis, but need to be displayed as meaningful text for humans. This dictionary provides those translations.

- **Example**: For Academic Pressure, the value `0` becomes `"Very Low"`, `3` becomes `"High"`, and `5` becomes `"Extreme"`.
- These mappings are automatically applied by the EDA and PCA scripts whenever they generate charts or tables, so no manual relabeling is needed.

### 5. Degree Group Encoding (`DEGREE_GROUP_ORDINAL`)

For analyses that require numeric data (like correlation matrices and PCA), the categorical Degree Group labels need to be converted into ordered numbers. This dictionary defines that ordering:

- School → 1, Undergraduate → 2, Postgraduate → 3, Doctorate → 4.
- The `"Other"` category is treated the same as Undergraduate (value 2), since it typically represents edge cases with similar characteristics.

### 6. Risk Framework Constants

These constants define the language used in the risk classification system. They are stored here so that every module — the batch risk classifier, the model trainer, and the prediction API — uses exactly the same wording:

- **`RISK_JUSTIFICATION`**: A clear explanation of why percentile-based thresholds are used. This text appears in generated reports, the API health endpoint, and saved JSON metadata.
- **`RISK_ACTIONS`**: The recommended actions for each risk level (Low, Moderate, High). These are deliberately worded as screening-oriented suggestions rather than clinical diagnoses, since the system is designed for awareness, not medical advice.

## Why This Matters

By keeping all of these settings in one file, the project avoids a common problem in data science pipelines: scattered, inconsistent configurations. If a label needs to change, a color needs to be updated, or a risk action description needs to be reworded, it only needs to be edited in one place, and the change automatically propagates to every script, plot, and API response.
