# EDA & Correlation Explanation (`eda.py`)

The EDA script is the project's visual storytelling engine. It takes the cleaned paper dataset and generates a comprehensive set of professional charts, tables, and statistical summaries that help us understand the data before any machine learning is applied. It also handles correlation analysis to uncover relationships between features and the depression outcome.

## What It Does

### 1. Data Loading and Preparation

The script reads `df_paper.csv` and capitalizes column names for consistent, professional-looking chart labels. It then runs through a sequence of analysis steps, saving all outputs to two organized directories:

- `/outputs/eda/` — Feature distributions, outlier analysis, and depression analytics.
- `/outputs/correlation/` — Correlation matrices and target-specific relationship summaries.

### 2. Missing Values Audit

A heatmap is generated to provide a quick visual check of data completeness. After the cleaning pipeline, all features should have zero missing entries. The plot confirms this and displays the total record count, giving confidence that the data is ready for analysis.

### 3. Outlier Detection

Two complementary outlier views are produced:

- **IQR Bar Chart**: Counts how many outliers each numeric column has using the Interquartile Range method. This gives a quick overview of which features have extreme values.
- **Boxplots with Scale Keys**: Individual boxplots for each feature, with annotated reference keys from `LABEL_MAPS`. For example, the Academic Pressure boxplot includes a legend showing that `0 = Very Low`, `3 = High`, `5 = Extreme`, making the chart self-explanatory.

### 4. Feature Distributions

For every categorical or low-cardinality numeric feature, the script generates:

- **Bar Charts**: Showing exact counts and percentages for each category.
- **Pie Charts**: Showing the relative proportion of each category with a color-coded legend.

This covers all the key variables — Gender, Academic Pressure, Sleep Duration, Financial Stress, Study Satisfaction, Suicidal Thoughts, Family History, Depression status, and Degree Group.

### 5. Depression Analytics

Targeted analysis of the depression outcome:

- **Age Trend Plot**: A line chart showing how many people with depression exist in each age group (18–28). This reveals whether depression prevalence changes with age within the student population.

### 6. Impact Analysis

A 2×2 grid of boxplots comparing four key indicators (Academic Pressure, Financial Stress, Study Satisfaction, Work/Study Hours) across depressed and non-depressed groups. This side-by-side comparison makes it easy to spot which factors differ most between the two groups.

### 7. Degree Group vs Depression

Dedicated analysis of how education level relates to depression:

- **Rate Chart**: A bar chart showing the depression rate (percentage) for each Degree Group (School, Undergraduate, Postgraduate, Doctorate). Bars are colored red if they exceed the overall depression rate, and green if they're below it. A dashed line shows the overall baseline rate for reference.
- **Stacked Bar Chart**: Shows the absolute count of depressed vs not-depressed students in each Degree Group, with total sample sizes annotated.
- **Interpretation Report**: An automatically generated text file summarizing which education level has the highest and lowest depression rates compared to the overall average.

### 8. Correlation Analysis

After all categorical plots are complete, the Degree Group column is converted to ordinal numbers (School=1, Undergraduate=2, etc.) so it can participate in numeric correlation analysis.

- **Spearman Correlation Heatmap**: A large, high-resolution matrix showing the strength and direction of relationships between all numeric features. Spearman correlation is used instead of Pearson because it handles ordinal data (like Academic Pressure scales) more appropriately. Labels are text-wrapped and formatted using `COLUMN_RENAMES` from the config.
- **Top Correlations Report**: Features are ranked by their correlation with Depression, and a separate interpretation is generated for the Degree Group's correlation — noting whether it's negligible, weak, or moderate, and what that means in plain language.

## How to Run

```bash
python backend-ml/src/eda.py
```

All outputs are saved to `backend-ml/outputs/eda/` and `backend-ml/outputs/correlation/`.