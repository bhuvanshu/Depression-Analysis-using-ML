# Utils Explanation (`utils.py`)

The `utils.py` file contains shared helper functions that handle repetitive tasks across the entire pipeline. Rather than writing the same file-saving and plot-formatting code in every script, each module imports these utilities from one place.

## What It Provides

### 1. Directory Management (`ensure_outdir`)

A safety function that creates output folders automatically if they don't already exist. This prevents "File Not Found" errors when a script tries to save a chart or CSV to a directory that hasn't been created yet. Every analysis script calls this before writing any output.

### 2. Filename Sanitization (`safe_filename`)

Converts any string into a safe, OS-compatible filename by removing special characters (slashes, backslashes, punctuation), replacing spaces with underscores, and converting to lowercase. This is important because feature names like `"Have you ever had suicidal thoughts ?"` or `"Work/Study Hours"` would cause file system errors if used directly as filenames.

### 3. Plot Finalization (`finalize_plot`)

Called at the end of every plotting function across the project. It applies a consistent finishing process to every chart:

- Adds a bold title if one is provided.
- Applies `tight_layout()` to prevent labels and legends from being cut off.
- Saves the figure at the configured DPI (150 by default) for crisp output.
- Closes the plot to free memory — important when generating dozens of charts in sequence.

This ensures every chart in the project, regardless of which script generated it, has the same polished appearance.

### 4. Text Report Saving (`save_text_report`)

Writes a string to a text file, automatically creating parent directories if needed. Used by the PCA, EDA, and risk classification scripts to save interpretation reports and statistical summaries alongside their visual outputs.

### 5. Publication-Quality Tables (`save_pretty_table`)

Converts a pandas DataFrame into a styled PNG image that looks professional enough for a research paper or presentation. The styling includes:

- **Steel Blue headers** with white, bold text for a clean, modern look.
- **Zebra striping** (alternating row colors) for readability.
- **Bold first column** to highlight category labels or model names.
- **Dynamic sizing** that adjusts the table height based on the number of rows.
- **High DPI export** (250 DPI) for sharp rendering at any size.

This is used for model comparison tables, risk summary tables, and other structured outputs that need to be visually presentable without external formatting.
