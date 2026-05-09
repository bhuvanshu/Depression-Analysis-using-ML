# Principal Component Analysis Explanation (`pca.py`)

The PCA script reduces the dataset's dimensionality to uncover the underlying structure of the data. In simple terms, it answers the question: "If we had to summarize all these features into just a few combined factors, what would those factors look like, and which original features matter most?"

## What It Does

### 1. Data Preparation

The script loads the paper dataset (`df_paper.csv`) and prepares it for PCA:

- Column names are capitalized for consistent formatting (matching the EDA script).
- The **Degree Group** column, which is categorical (School, Undergraduate, Postgraduate, Doctorate), is converted into ordered numbers (1, 2, 3, 4) using the `DEGREE_GROUP_ORDINAL` mapping from `config.py`. This is necessary because PCA only works with numeric data.
- All features are then standardized using `StandardScaler` so that no single feature dominates the analysis simply because it has larger values.

### 2. Running PCA

PCA is fitted on all numeric features (excluding the Depression target). The number of components is set to 8 or the total number of features, whichever is smaller. This captures enough variance to be meaningful without overfitting to noise.

### 3. Visualizations

Three key plots are generated:

- **Scree Plot**: Shows how much variance each principal component explains, both individually and cumulatively. This helps determine how many components are needed to capture the majority of the data's variability. For example, if the first three components capture 60% of variance, the remaining components add diminishing insight.

- **PC1 vs PC2 Scatter Plot**: Projects every student onto the first two principal components and colors them by depression status. If the two groups (depressed / not depressed) separate into distinct clusters, it suggests the features collectively distinguish the groups well. Overlapping clusters indicate the relationship is more nuanced.

- **Biplot (Loading Plot)**: This is the most informative visualization. It overlays the original feature vectors (as red arrows) on top of the data projection. The direction and length of each arrow shows how much that feature contributes to each component. Features pointing in the same direction are correlated; features pointing in opposite directions are inversely related. Labels use `COLUMN_RENAMES` from the config for readability, and an automatic anti-collision system pushes overlapping labels apart.

### 4. Reporting

Several output files are generated for further analysis:

- **`pca_explained.txt`**: The raw explained variance ratios for each component.
- **`pca_components.csv`**: The full loading matrix — the exact contribution of every feature to every principal component. This is the quantitative backbone of the biplot.
- **`pca_interpretation.txt`**: An auto-generated report that lists the top 5 contributing features for each component, with their loading values and signs (+/-). This makes it easy to describe what each component "represents" in plain language.
- **`pca_transformed.csv`**: The entire dataset converted into principal component space, with the Depression target column preserved. This can be used for further analysis or as an alternative feature set for modeling.

### 5. Degree Group Interpretation

After PCA runs, the script checks how the Degree Group feature contributes to the principal components. It reads the loading values for PC1 and PC2 and generates a plain-language interpretation:

- If both loadings are small (< 0.1), it notes that Degree Group contributes minimally to the primary axes of variance.
- If the loadings are moderate (0.1–0.3), it notes partial contribution.
- If the loadings are notable (> 0.3), it highlights Degree Group as a meaningful factor in the PCA decomposition.

This interpretation is saved as `degree_group_pca_interpretation.txt`.

## How to Run

```bash
python backend-ml/src/pca.py
```

## Outputs

All results are saved to `backend-ml/outputs/pca/`:

| File | Description |
|---|---|
| `pca_scree.png` | Explained variance trend (individual + cumulative) |
| `pca_pc1_vs_pc2.png` | Data projection colored by depression status |
| `pca_loading_plot.png` | Feature influence biplot with labeled arrows |
| `pca_interpretation.txt` | Top features per component with loading values |
| `pca_components.csv` | Full loading matrix for all components |
| `pca_transformed.csv` | Dataset transformed into principal component space |
| `pca_explained.txt` | Raw variance ratios |
| `degree_group_pca_interpretation.txt` | Plain-language interpretation of Degree Group's PCA role |
