# Config Explanation (`config.py`)

The `config.py` file serves as the central source of truth for all configurations, metadata, and aesthetic settings used in the backend analysis.

## Key Components

### 1. Column Renaming (`COLUMN_RENAMES`)
Stores mapping for simplifying long or cryptic column names into concise labels. This is primarily used by the plotting engine to ensure that axis labels on heatmaps and charts are readable and don't overlap.
- **Example**: `"Have You Ever Had Suicidal Thoughts ?"` becomes `"Suicidal"`.

### 2. Feature Definitions
- **`CORE_FEATURES`**: A list of the primary independent variables being analyzed.
- **`TARGET_COL`**: Identifies the dependent variable (`Depression`) for all analysis scripts.

### 3. Global Styling (`STYLE_SETTINGS`)
Centralizes the look and feel of the project's visualizations.
- **`palette`**: Default color scheme for bar and pie charts.
- **`heatmap_cmap`**: The colormap used for correlation matrices (e.g., `coolwarm`).
- **`figure_dpi`**: Sets the resolution for output images to ensure high-quality exports.

### 4. Label Mapping (`LABEL_MAPS`)
Translations for numeric categories. Since much of the data is encoded as integers (0, 1, 2...), this dictionary allows the plotting functions to automatically replace those numbers with meaningful text (e.g., `0: "Female", 1: "Male"`) for the end-user.
