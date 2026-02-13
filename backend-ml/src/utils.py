import re
from pathlib import Path
import matplotlib.pyplot as plt

def ensure_outdir(outdir: Path):
    """Ensures the output directory exists."""
    outdir.mkdir(parents=True, exist_ok=True)

def save_text_report(path: Path, text: str):
    """Saves a string to a text file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def safe_filename(name: str) -> str:
    """Sanitizes a string to be used as a filename."""
    s = str(name)
    s = re.sub(r"[\\/]", "_", s)
    s = "".join(c if (c.isalnum() or c in (" ","-","_",".")) else "_" for c in s)
    s = s.replace(" ", "_").lower()
    return s

def finalize_plot(path: Path, title: str = None, dpi: int = 150):
    """Common logic for finishing and saving a matplotlib plot."""
    if title:
        plt.title(title, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def save_pretty_table(df, path: Path, title: str):
    """Saves a dataframe as a high-quality, colorful PNG table for paper publication."""
    import numpy as np
    
    # Dynamic height based on number of rows
    n_rows = len(df)
    fig_height = max(3.5, 0.7 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height)) 
    ax.axis('off')
    
    # Create Table
    tbl = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center'
    )
    
    # Styling
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1.0, 2.5)
    
    # Header Styling (Steel Blue)
    for j, col in enumerate(df.columns):
        cell = tbl[0, j]
        cell.set_facecolor('#4682B4') 
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
        cell.get_text().set_fontsize(16)
        cell.set_edgecolor('white')
        
    # Row Styling (Zebra Striping)
    for i in range(1, len(df) + 1):
        facecolor = '#f2f2f2' if i % 2 == 0 else 'white'
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            cell.set_facecolor(facecolor)
            cell.set_edgecolor('#d9d9d9')
            # Bold the first column (Categorical labels/Model names)
            if j == 0:
                cell.get_text().set_weight('bold')

    # Ensure columns fit text
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    
    plt.title(title, fontsize=22, fontweight="bold", y=0.98, pad=25)
    
    plt.savefig(path, dpi=250, bbox_inches="tight") # higher DPI
    plt.close()
