# """
# comparison.py
# -------------
# Generates a PDF report of all clustering model comparison charts and tables.
# Output: processed/comparison/model_comparison_report.pdf

# Run from the project root:
#     python comparison.py
# """
# from __future__ import annotations
# import geopandas as gpd 
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# import os
# import io
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from matplotlib.backends.backend_pdf import PdfPages

# # ── Paths ─────────────────────────────────────────────────────────────────────
# PROCESSED_DIR = Path("processed")
# OUTPUT_DIR = Path("/Users/keerthana/Desktop/ALL/year 4/Group FYP/two/ecs7019u-group4/processed/comparison")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_PDF = OUTPUT_DIR / "model_comparison_report.pdf"

# CLUSTER_COLOURS = ["#E63946", "#2A9D8F", "#E9C46A", "#4361EE"]
# MODEL_COLOURS   = {"KMeans": "#4361EE", "GMM": "#E63946", "Hybrid GMM": "#2A9D8F"}

# # ── Helpers ───────────────────────────────────────────────────────────────────

