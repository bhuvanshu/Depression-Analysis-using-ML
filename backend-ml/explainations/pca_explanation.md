# Principal Component Analysis Explanation (`pca.py`)

The `pca.py` script implements a comprehensive Principal Component Analysis (PCA) pipeline to reduce data dimensionality and identify the primary factors contributing to depression variance in the dataset.

## Key Features

### 1. Robust Data Processing
- **Standardization**: Automatically scales features using `StandardScaler` to ensure all variables contribute equally to the variance analysis.
- **Dynamic Configuration**: Integrates with `config.py` to use centralized target column names and label mappings.

### 2. Multi-Dimensional Visualizations
- **Scree Plot**: Visualizes the explained variance ratio for each principal component, helping determine the optimal number of dimensions to retain.
- **PC1 vs PC2 Scatter Plot**: Projections of the data onto the first two principal components, colored by depression status to identify class separation.
- **Biplot (Loading Plot)**: A high-fidelity visualization that shows both the data points and the feature vectors (loadings). This reveals which features (e.g., Academic Pressure, Financial Stress) are most influential for each component.

### 3. Comprehensive Reporting
- **Variance Report**: Saves the exact variance ratio explained by each component to `pca_explained.txt`.
- **Component Loadings**: Exports the raw loading matrix to `pca_components.csv` for further quantitative analysis.
- **Interpretation Guide**: Generates a human-readable `pca_interpretation.txt` which automatically identifies the top 5 contributing features for each principal component.

### 4. Dynamic Labeling
The script uses `COLUMN_RENAMES` and `LABEL_MAPS` from `config.py` to ensure all plots use professional, human-readable labels instead of raw database column names.

## Usage
Run the script from the project root:
```bash
python backend-ml/src/pca.py
```

## Outputs
All results are saved to `backend-ml/outputs/pca/`, including:
- `pca_scree.png`: Explained variance trend.
- `pca_pc1_vs_pc2.png`: Data projection.
- `pca_loading_plot.png`: Feature influence (Biplot).
- `pca_interpretation.txt`: Automated feature analysis.
- `pca_transformed.csv`: The dataset converted into principal component space.
