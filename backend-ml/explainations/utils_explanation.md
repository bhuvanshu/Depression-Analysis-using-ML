# Utils Explanation (`utils.py`)

The `utils.py` file contains helper functions that provide common functionality needed by multiple scripts. It abstracts away repetitive tasks like file handling and plot formatting.

## Utilities

### 1. File & Directory Management
- **`ensure_outdir`**: A safety function that creates folders if they don't exist, preventing "File Not Found" errors during saving.
- **`safe_filename`**: Sanitizes strings by removing slashes and special characters, converting them to lowercase and replacing spaces with underscores. This ensures generated images have consistent, OS-compatible filenames.

### 2. Plotting Polish
- **`finalize_plot`**: This is called at the end of every plotting function. It standardized the saving process, applying `tight_layout` (to prevent cut-off labels) and setting a high DPI for crisp images.

### 3. Academic Reporting
- **`save_pretty_table`**: A specialized function that converts a pandas DataFrame into a styled image. 
    - It uses **Zebra Striping** (alternating row colors) for readability.
    - It applies **Steel Blue** headers.
    - It is designed to export tables directly as PNGs, which are easier to embed in research papers or presentations than raw text.
