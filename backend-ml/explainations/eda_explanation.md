# EDA & Correlation Explanation (`eda.py`)

The `eda.py` script is a consolidated tool for Exploratory Data Analysis (EDA) and relationship discovery. It handles data loading, statistical summary generation, and the creation of professional visualizations for both descriptive statistics and correlation analysis.

## Key Features

### 1. Unified Pipeline
- **Orchestration**: Combines data loading, cleaning (column normalization), statistical reporting, and plotting into a single execution flow.
- **Organization**: Saves results into two distinct directories:
    - `/outputs/eda/`: For feature distributions and outlier analysis.
    - `/outputs/correlation/`: For relationship matrices and target-specific correlations.

### 2. Advanced Visualizations
- **Missing Values Map**: A heatmap that provides an immediate visual audit of data completeness.
- **Outlier Detection**:
    - **IQR Histogram**: Counts outliers per column using the Interquartile Range method.
    - **Label-Aware Boxplots**: Detailed boxplots that include reference keys for categorical scales (from `LABEL_MAPS`).
- **Distribution Analysis**: Generates both **Bar Charts** (for counts/percentages) and **Pie Charts** (for relative proportions) for categorical and binned numeric features.
- **Depression Analytics**:
    - **Age Trend**: A line plot showing depression counts across different age groups.
    - **Proportion Pie**: A clear visualization of the overall depressed vs. non-depressed population.
- **Impact Analysis Grid**: A 2x2 grid of boxplots comparing key performance indicators (e.g., Academic Pressure, Financial Stress) against the depression target class.

### 3. Correlation Discovery
- **Spearman Matrix**: Generates a high-resolution Spearman Correlation heatmap with text-wrapped labels and clean annotations.
- **Target Analysis**: Automatically calculates and ranks features based on their correlation with the `Depression` status, helping identify the strongest predictors.

## Usage
Run the script from the project root:
```bash
python backend-ml/src/eda.py
```


