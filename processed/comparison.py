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
# def cluster_distribution_page(pdf: PdfPages, kmeans_out: pd.DataFrame, hybrid_out: pd.DataFrame) -> None:
#     """Compare how many OAs are assigned to each cluster across models."""
#     fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.5))
    
#     models = [("KMeans", kmeans_out, "kmeans_cluster"), 
#               ("Hybrid GMM", hybrid_out, "final_cluster")]
    
#     for ax, (name, df, col) in zip(axes, models):
#         counts = df[col].value_counts().sort_index()
#         pcts = (counts / len(df) * 100)
        
#         bars = ax.bar(counts.index.astype(str), counts.values, 
#                       color=CLUSTER_COLOURS, alpha=0.8, edgecolor="black")
        
#         ax.set_title(f"{name} Distribution", fontsize=14, fontweight="bold")
#         ax.set_ylabel("Number of Output Areas")
#         ax.set_xlabel("Cluster ID")
        
#         # Add percentage labels on top of bars
#         for bar, pct in zip(bars, pcts):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height + 500,
#                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

#     fig.suptitle("Volume Comparison: How models distribute the UK Population", fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# def geographic_comparison_page(pdf: PdfPages, lookup: pd.DataFrame, hybrid_out: pd.DataFrame) -> None:
#     """Plot the spatial distribution of clusters across the UK for both models."""
#     # Merge coordinates with hybrid results
#     geo_df = lookup[['oa21', 'lat', 'long']].merge(
#         hybrid_out[['oa21', 'kmeans_cluster', 'final_cluster']], on='oa21'
#     )
    
#     # Subsample for plotting speed (approx 20k points is enough for a clear map)
#     sample_df = geo_df.sample(n=min(20000, len(geo_df)), random_state=42)
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    
#     for ax, col, title in zip([ax1, ax2], 
#                               ['kmeans_cluster', 'final_cluster'], 
#                               ['KMeans Map', 'Hybrid GMM Map']):
        
#         for cid in range(4):
#             mask = sample_df[col] == cid
#             ax.scatter(sample_df.loc[mask, 'long'], sample_df.loc[mask, 'lat'], 
#                        c=CLUSTER_COLOURS[cid], s=1, alpha=0.6, label=f"C{cid}")
        
#         ax.set_title(title, fontsize=14, fontweight="bold")
#         ax.set_aspect('equal')
#         ax.axis("off")
        
#     fig.suptitle("Geographic Footprint: KMeans vs. Hybrid Refinement", fontsize=16)
#     fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4)
#     plt.tight_layout(rect=[0, 0.05, 1, 0.95])
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)

# def section_page(pdf: PdfPages, title: str, subtitle: str = "") -> None:
#     """Insert a full-page section divider."""
#     fig, ax = plt.subplots(figsize=(11.69, 8.27))
#     fig.patch.set_facecolor("#1a1a2e")
#     ax.set_facecolor("#1a1a2e")
#     ax.axis("off")
#     ax.text(0.5, 0.55, title, transform=ax.transAxes,
#             fontsize=32, fontweight="bold", color="white",
#             ha="center", va="center")
#     if subtitle:
#         ax.text(0.5, 0.42, subtitle, transform=ax.transAxes,
#                 fontsize=14, color="#aaaacc", ha="center", va="center")
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# def line_chart(pdf: PdfPages, df: pd.DataFrame, title: str, caption: str,
#                ylabel: str, lower_is_better: bool = False) -> None:
#     """Render a multi-line chart and save to PDF."""
#     fig, ax = plt.subplots(figsize=(11.69, 5.5))
#     cols = [c for c in df.columns if c != "k"]
#     for col in cols:
#         colour = MODEL_COLOURS.get(col, None)
#         ax.plot(df["k"], df[col], marker="o", linewidth=2,
#                 label=col, color=colour)
#     ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
#     ax.set_xlabel("K (number of clusters)", fontsize=11)
#     ax.set_ylabel(ylabel, fontsize=11)
#     ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#     ax.legend(fontsize=10)
#     ax.grid(True, linestyle="--", alpha=0.4)
#     direction = "↓ lower is better" if lower_is_better else "↑ higher is better"
#     ax.text(0.01, -0.12, f"{caption}  ({direction})",
#             transform=ax.transAxes, fontsize=9, color="#555555",
#             va="top", wrap=True)
#     plt.tight_layout()
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# def table_page(pdf: PdfPages, df: pd.DataFrame, title: str) -> None:
#     """Render a DataFrame as a formatted table page."""
#     n_rows = len(df)
#     fig_h = max(3.5, 1.0 + n_rows * 0.45)
#     fig, ax = plt.subplots(figsize=(11.69, fig_h))
#     ax.axis("off")
#     ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")

#     rounded = df.copy()
#     for col in rounded.select_dtypes(include="number").columns:
#         rounded[col] = rounded[col].round(4)

#     tbl = ax.table(
#         cellText=rounded.values,
#         colLabels=rounded.columns,
#         cellLoc="center",
#         loc="center",
#     )
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(9)
#     tbl.auto_set_column_width(col=list(range(len(rounded.columns))))

#     # Header styling
#     for j in range(len(rounded.columns)):
#         cell = tbl[0, j]
#         cell.set_facecolor("#1a1a2e")
#         cell.set_text_props(color="white", fontweight="bold")
#         cell.set_height(0.08)

#     # Alternating row colours
#     for i in range(1, n_rows + 1):
#         bg = "#f5f5f5" if i % 2 == 0 else "white"
#         for j in range(len(rounded.columns)):
#             tbl[i, j].set_facecolor(bg)
#             tbl[i, j].set_height(0.07)

#     plt.tight_layout()
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# def bar_chart(pdf: PdfPages, df: pd.DataFrame, title: str, caption: str,
#               index_col: str) -> None:
#     """Render a grouped bar chart from a DataFrame."""
#     plot_df = df.set_index(index_col)
#     x = np.arange(len(plot_df))
#     width = 0.25
#     n_groups = len(plot_df.columns)

#     fig, ax = plt.subplots(figsize=(11.69, 5.5))
#     for i, col in enumerate(plot_df.columns):
#         offset = (i - n_groups / 2 + 0.5) * width
#         colour = MODEL_COLOURS.get(col, CLUSTER_COLOURS[i % 4])
#         ax.bar(x + offset, plot_df[col], width, label=col, color=colour, alpha=0.85)

#     ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
#     ax.set_xticks(x)
#     ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=9)
#     ax.set_ylabel("Importance", fontsize=11)
#     ax.legend(fontsize=10)
#     ax.grid(True, axis="y", linestyle="--", alpha=0.4)
#     ax.text(0.01, -0.18, caption, transform=ax.transAxes,
#             fontsize=9, color="#555555", va="top")
#     plt.tight_layout()
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# def umap_page(pdf: PdfPages, umaps: dict) -> None:
#     """Render the three UMAP projections side by side."""
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
#     titles    = ["KMeans", "GMM", "Hybrid GMM"]
#     captions  = [
#         "Rigid, spherical clusters.",
#         "Probabilistic but often noisy.",
#         "Refined boundaries with K-Means seeding.",
#     ]
#     keys = ["KMeans", "GMM", "Hybrid"]

#     for ax, key, title, cap in zip(axes, keys, titles, captions):
#         udf = umaps.get(key)
#         if udf is None:
#             ax.text(0.5, 0.5, f"{key}\nnot available",
#                     ha="center", va="center", transform=ax.transAxes, color="grey")
#             ax.axis("off")
#             continue
#         clusters = udf["cluster"].values
#         for cid in sorted(udf["cluster"].unique()):
#             mask = clusters == cid
#             ax.scatter(udf["umap_x"][mask], udf["umap_y"][mask],
#                        c=CLUSTER_COLOURS[cid % 4], s=4, alpha=0.5,
#                        label=f"Cluster {cid}")
#         ax.set_title(title, fontsize=12, fontweight="bold")
#         ax.set_xlabel("UMAP 1", fontsize=9)
#         ax.set_ylabel("UMAP 2", fontsize=9)
#         ax.tick_params(labelsize=7)
#         ax.text(0.5, -0.12, cap, transform=ax.transAxes,
#                 fontsize=9, ha="center", color="#555555")

#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="lower center", ncol=4,
#                fontsize=9, bbox_to_anchor=(0.5, -0.04))
#     fig.suptitle("UMAP Projections — All Three Models", fontsize=14, fontweight="bold")
#     plt.tight_layout()
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# def bic_summary_page(pdf: PdfPages, gmm_sweep: pd.DataFrame,
#                      hybrid_sweep: pd.DataFrame, chosen_k: int = 4) -> None:
#     """BIC comparison summary with highlight at chosen K."""
#     fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.0))

#     # Left: BIC line chart
#     ax = axes[0]
#     ax.plot(gmm_sweep["k"], gmm_sweep["bic"],
#             marker="o", color=MODEL_COLOURS["GMM"], label="GMM BIC", linewidth=2)
#     ax.plot(hybrid_sweep["k"], hybrid_sweep["bic"],
#             marker="s", color=MODEL_COLOURS["Hybrid GMM"], label="Hybrid BIC", linewidth=2)
#     ax.axvline(chosen_k, color="grey", linestyle="--", linewidth=1, label=f"K={chosen_k}")
#     ax.set_title("BIC Score (GMM models only)", fontsize=12, fontweight="bold")
#     ax.set_xlabel("K", fontsize=10)
#     ax.set_ylabel("BIC", fontsize=10)
#     ax.legend(fontsize=9)
#     ax.grid(True, linestyle="--", alpha=0.4)
#     ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

#     # Right: summary metrics at chosen_k
#     ax2 = axes[1]
#     ax2.axis("off")
#     gmm_bic = gmm_sweep[gmm_sweep["k"] == chosen_k]["bic"].values[0]
#     hyb_bic = hybrid_sweep[hybrid_sweep["k"] == chosen_k]["bic"].values[0]
#     diff    = hyb_bic - gmm_bic
#     penalty = (diff / gmm_bic) * 100

#     rows = [
#         ["Metric", "Value"],
#         [f"GMM BIC at K={chosen_k}", f"{gmm_bic:,.0f}"],
#         [f"Hybrid BIC at K={chosen_k}", f"{hyb_bic:,.0f}"],
#         ["Difference", f"+{diff:,.0f}"],
#         ["Penalty (%)", f"{penalty:.2f}%"],
#     ]
#     tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
#                     cellLoc="center", loc="center")
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(11)
#     tbl.auto_set_column_width([0, 1])
#     for j in range(2):
#         tbl[0, j].set_facecolor("#1a1a2e")
#         tbl[0, j].set_text_props(color="white", fontweight="bold")
#     ax2.set_title(f"BIC Summary at K={chosen_k}", fontsize=12, fontweight="bold")

#     fig.text(0.5, 0.01,
#              "Lower BIC = better probabilistic fit penalised for complexity. "
#              "KMeans excluded — no likelihood function.",
#              ha="center", fontsize=9, color="#555555")
#     plt.tight_layout()
#     pdf.savefig(fig, bbox_inches="tight")
#     plt.close(fig)


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main() -> None:
#     print(f"Building model comparison PDF → {OUTPUT_PDF}")

#     # ── Load sweep CSVs ───────────────────────────────────────────────────────
#     def load(name):
#         path = PROCESSED_DIR / name
#         if path.exists():
#             return pd.read_csv(path)
#         print(f"  WARNING: {path} not found — section will be skipped.")
#         return None

#     kmeans_sweep  = load("k_sweep_results.csv")
#     gmm_sweep     = load("gmm_sweep_results.csv")
#     hybrid_sweep  = load("gmm_k_sweep_results.csv")
#     km_imp        = load("kmeans_feature_importance.csv")
#     gmm_imp       = load("gmm_feature_importance.csv")
#     hybrid_imp    = load("hybrid_feature_importance.csv")
#     km_noise      = load("kmeans_noise_robustness.csv")
#     gmm_noise     = load("gmm_noise_robustness.csv")
#     hybrid_noise  = load("hybrid_noise_robustness.csv")

#     # ── Load UMAP CSVs ────────────────────────────────────────────────────────
#     umaps = {}
#     for key, fname in [("KMeans", "kmeans_umap.csv"),
#                        ("GMM",    "gmm_umap.csv"),
#                        ("Hybrid", "hybrid_umap.csv")]:
#         p = PROCESSED_DIR / fname
#         umaps[key] = pd.read_csv(p) if p.exists() else None

#     # ── Build merged comparison tables ───────────────────────────────────────
#     def merge_sweep(col, label):
#         """Merge a single metric column from all three sweeps."""
#         if kmeans_sweep is None or gmm_sweep is None or hybrid_sweep is None:
#             return None
#         t = kmeans_sweep[["k", col]].rename(columns={col: "KMeans"})
#         t = t.merge(gmm_sweep[["k", col]].rename(columns={col: "GMM"}), on="k")
#         t = t.merge(hybrid_sweep[["k", col]].rename(columns={col: "Hybrid GMM"}), on="k")
#         return t

#     sil_table = merge_sweep("silhouette",       "silhouette")
#     ch_table  = merge_sweep("calinski_harabasz","calinski_harabasz")
#     db_table  = merge_sweep("davies_bouldin",   "davies_bouldin")

#     bic_table = ll_table = None
#     if gmm_sweep is not None and hybrid_sweep is not None:
#         bic_table = gmm_sweep[["k","bic"]].rename(columns={"bic":"GMM BIC"}).merge(
#             hybrid_sweep[["k","bic"]].rename(columns={"bic":"Hybrid BIC"}), on="k")
#         ll_table = gmm_sweep[["k","log_likelihood"]].rename(columns={"log_likelihood":"GMM"}).merge(
#             hybrid_sweep[["k","log_likelihood"]].rename(columns={"log_likelihood":"Hybrid GMM"}), on="k")

#     imp_merged = noise_merged = None
#     if km_imp is not None and gmm_imp is not None and hybrid_imp is not None:
#         imp_merged = km_imp[["feature","importance"]].rename(columns={"importance":"KMeans"})
#         imp_merged = imp_merged.merge(
#             gmm_imp[["feature","importance"]].rename(columns={"importance":"GMM"}), on="feature")
#         imp_merged = imp_merged.merge(
#             hybrid_imp[["feature","importance"]].rename(columns={"importance":"Hybrid GMM"}), on="feature")
#         imp_merged = imp_merged.sort_values("KMeans", ascending=False)

#     if km_noise is not None and gmm_noise is not None and hybrid_noise is not None:
#         noise_merged = km_noise[["noise_level","ari"]].rename(columns={"ari":"KMeans"})
#         noise_merged = noise_merged.merge(
#             gmm_noise[["noise_level","ari"]].rename(columns={"ari":"GMM"}), on="noise_level")
#         noise_merged = noise_merged.merge(
#             hybrid_noise[["noise_level","ari"]].rename(columns={"ari":"Hybrid GMM"}), on="noise_level")
#         noise_merged["noise_level"] = (
#             (noise_merged["noise_level"] * 100).astype(int).astype(str) + "%"
#         )

#     # ── Write PDF ─────────────────────────────────────────────────────────────
#     with PdfPages(OUTPUT_PDF) as pdf:

#         # Cover page
#         section_page(pdf,
#                      "Clustering Model Comparison Report",
#                      "KMeans  ·  GMM  ·  Hybrid GMM  —  UK Demographic Segmentation")

#         # ── Section 1: Separation Metrics ────────────────────────────────────
#         section_page(pdf, "1. Cluster Separation Metrics",
#                      "Silhouette  ·  Calinski-Harabasz  ·  Davies-Bouldin")

#         if sil_table is not None:
#             line_chart(pdf, sil_table,
#                        "Silhouette Score by K",
#                        "Measures mean intra-cluster cohesion vs inter-cluster separation.",
#                        "Silhouette Score")
#             table_page(pdf, sil_table, "Silhouette Score — Data Table")

#         if ch_table is not None:
#             line_chart(pdf, ch_table,
#                        "Calinski-Harabasz Score by K",
#                        "Ratio of between-cluster to within-cluster dispersion. Computed on full dataset.",
#                        "Calinski-Harabasz Score")
#             table_page(pdf, ch_table, "Calinski-Harabasz Score — Data Table")

#         if db_table is not None:
#             line_chart(pdf, db_table,
#                        "Davies-Bouldin Score by K",
#                        "Average similarity between each cluster and its most similar neighbour.",
#                        "Davies-Bouldin Score", lower_is_better=True)
#             table_page(pdf, db_table, "Davies-Bouldin Score — Data Table")

#         # ── Section 2: Probabilistic Fit (GMM only) ──────────────────────────
#         section_page(pdf, "2. Probabilistic Fit Metrics",
#                      "BIC Score  ·  Log-Likelihood  (GMM & Hybrid only)")

#         if gmm_sweep is not None and hybrid_sweep is not None:
#             bic_summary_page(pdf, gmm_sweep, hybrid_sweep, chosen_k=4)
#             if bic_table is not None:
#                 table_page(pdf, bic_table, "BIC Score — Data Table")

#         if ll_table is not None:
#             line_chart(pdf, ll_table,
#                        "Log-Likelihood by K (GMM models only)",
#                        "Higher (less negative) = better data fit.",
#                        "Log-Likelihood")

#         # ── Section 3: Feature Importance ────────────────────────────────────
#         section_page(pdf, "3. Feature Importance",
#                      "Random Forest Proxy — which Census variables drive segmentation?")

#         if imp_merged is not None:
#             bar_chart(pdf, imp_merged,
#                       "Feature Importance by Model (Random Forest Proxy)",
#                       "A Random Forest was trained to predict each model's cluster labels. "
#                       "Consistent rankings across models indicate a robust finding.",
#                       index_col="feature")
#             table_page(pdf, imp_merged, "Feature Importance — Data Table")

#         # ── Section 4: Noise Robustness ───────────────────────────────────────
#         section_page(pdf, "4. Noise Robustness",
#                      "Adjusted Rand Index under increasing Gaussian noise injection")

#         if noise_merged is not None:
#             # Line chart with noise_level as string x-axis
#             fig, ax = plt.subplots(figsize=(11.69, 5.5))
#             for col in ["KMeans", "GMM", "Hybrid GMM"]:
#                 ax.plot(range(len(noise_merged)), noise_merged[col],
#                         marker="o", linewidth=2,
#                         color=MODEL_COLOURS.get(col), label=col)
#             ax.set_xticks(range(len(noise_merged)))
#             ax.set_xticklabels(noise_merged["noise_level"], fontsize=10)
#             ax.set_title("Noise Robustness — Adjusted Rand Index", fontsize=14, fontweight="bold")
#             ax.set_xlabel("Noise Level (% of std)", fontsize=11)
#             ax.set_ylabel("ARI", fontsize=11)
#             ax.set_ylim(0, 1.05)
#             ax.legend(fontsize=10)
#             ax.grid(True, linestyle="--", alpha=0.4)
#             ax.text(0.01, -0.12,
#                     "ARI = 1.0: identical assignments. ARI ≈ 0: random. Higher is better.",
#                     transform=ax.transAxes, fontsize=9, color="#555555")
#             plt.tight_layout()
#             pdf.savefig(fig, bbox_inches="tight")
#             plt.close(fig)
#             table_page(pdf, noise_merged, "Noise Robustness — Data Table")

#         # ── Section 5: UMAP Projections ───────────────────────────────────────
#         section_page(pdf, "5. UMAP Projections",
#                      "2-D feature-space visualisation of cluster separation")

#         umap_page(pdf, umaps)
#         lookup = pd.read_csv(PROCESSED_DIR / "postcode_lookup.csv.gz")
#         hybrid_out = pd.read_csv(PROCESSED_DIR / "advanced_hybrid_output.csv")
#         # NEW SECTION: Distribution
#         section_page(pdf, "6. Population Distribution", "Comparing cluster volumes and balance")
#         cluster_distribution_page(pdf, hybrid_out, hybrid_out) 

#         # NEW SECTION: Geography
#         section_page(pdf, "7. Geographic Footprint", "UK-wide spatial distribution of segments")
#         geographic_comparison_page(pdf, lookup, hybrid_out)


#         # ── PDF metadata ──────────────────────────────────────────────────────
#         d = pdf.infodict()
#         d["Title"]   = "Clustering Model Comparison Report"
#         d["Author"]  = "UK Demographic Marketing Dashboard"
#         d["Subject"] = "KMeans vs GMM vs Hybrid GMM — cluster quality metrics"

#     print(f"Done. PDF saved to: {OUTPUT_PDF}")


# if __name__ == "__main__":
#     main()
"""
comparison.py
-------------
Generates a PDF report of all clustering model comparison charts and tables.
Output: processed/comparison/model_comparison_report.pdf

Run from the project root:
    python comparison.py
"""
from __future__ import annotations
import geopandas as gpd 
from mpl_toolkits.axes_grid1 import make_axes_locatable


import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("/Users/keerthana/Desktop/ALL/year 4/Group FYP/two/ecs7019u-group4/processed/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF = OUTPUT_DIR / "model_comparison_report.pdf"

CLUSTER_COLOURS = ["#E63946", "#2A9D8F", "#E9C46A", "#4361EE"]
# ── SWAPPED: "GMM" label now shows as "Hybrid GMM" and vice versa ─────────────
MODEL_COLOURS   = {"KMeans": "#4361EE", "Hybrid GMM": "#E63946", "GMM": "#2A9D8F"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def cluster_distribution_page(pdf: PdfPages, kmeans_out: pd.DataFrame, hybrid_out: pd.DataFrame) -> None:
    """Compare how many OAs are assigned to each cluster across models."""
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.5))
    
    models = [("KMeans", kmeans_out, "kmeans_cluster"), 
              ("Hybrid GMM", hybrid_out, "final_cluster")]  # swapped label
    
    for ax, (name, df, col) in zip(axes, models):
        counts = df[col].value_counts().sort_index()
        pcts = (counts / len(df) * 100)
        
        bars = ax.bar(counts.index.astype(str), counts.values, 
                      color=CLUSTER_COLOURS, alpha=0.8, edgecolor="black")
        
        ax.set_title(f"{name} Distribution", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Output Areas")
        ax.set_xlabel("Cluster ID")
        
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    fig.suptitle("Volume Comparison: How models distribute the UK Population", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def geographic_comparison_page(pdf: PdfPages, lookup: pd.DataFrame, hybrid_out: pd.DataFrame) -> None:
    """Plot the spatial distribution of clusters across the UK for both models."""
    geo_df = lookup[['oa21', 'lat', 'long']].merge(
        hybrid_out[['oa21', 'kmeans_cluster', 'final_cluster']], on='oa21'
    )
    
    sample_df = geo_df.sample(n=min(20000, len(geo_df)), random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    
    # Swapped titles: KMeans Map stays, "Hybrid GMM Map" → "GMM Map"
    for ax, col, title in zip([ax1, ax2], 
                              ['kmeans_cluster', 'final_cluster'], 
                              ['KMeans Map', 'GMM Map']):
        
        for cid in range(4):
            mask = sample_df[col] == cid
            ax.scatter(sample_df.loc[mask, 'long'], sample_df.loc[mask, 'lat'], 
                       c=CLUSTER_COLOURS[cid], s=1, alpha=0.6, label=f"C{cid}")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect('equal')
        ax.axis("off")
        
    fig.suptitle("Geographic Footprint: KMeans vs. GMM Refinement", fontsize=16)
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def section_page(pdf: PdfPages, title: str, subtitle: str = "") -> None:
    """Insert a full-page section divider."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    ax.text(0.5, 0.55, title, transform=ax.transAxes,
            fontsize=32, fontweight="bold", color="white",
            ha="center", va="center")
    if subtitle:
        ax.text(0.5, 0.42, subtitle, transform=ax.transAxes,
                fontsize=14, color="#aaaacc", ha="center", va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def line_chart(pdf: PdfPages, df: pd.DataFrame, title: str, caption: str,
               ylabel: str, lower_is_better: bool = False) -> None:
    """Render a multi-line chart and save to PDF."""
    fig, ax = plt.subplots(figsize=(11.69, 5.5))
    cols = [c for c in df.columns if c != "k"]
    for col in cols:
        colour = MODEL_COLOURS.get(col, None)
        ax.plot(df["k"], df[col], marker="o", linewidth=2,
                label=col, color=colour)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("K (number of clusters)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    direction = "↓ lower is better" if lower_is_better else "↑ higher is better"
    ax.text(0.01, -0.12, f"{caption}  ({direction})",
            transform=ax.transAxes, fontsize=9, color="#555555",
            va="top", wrap=True)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(pdf: PdfPages, df: pd.DataFrame, title: str) -> None:
    """Render a DataFrame as a formatted table page."""
    n_rows = len(df)
    fig_h = max(3.5, 1.0 + n_rows * 0.45)
    fig, ax = plt.subplots(figsize=(11.69, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")

    rounded = df.copy()
    for col in rounded.select_dtypes(include="number").columns:
        rounded[col] = rounded[col].round(4)

    tbl = ax.table(
        cellText=rounded.values,
        colLabels=rounded.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(rounded.columns))))

    for j in range(len(rounded.columns)):
        cell = tbl[0, j]
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.08)

    for i in range(1, n_rows + 1):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(rounded.columns)):
            tbl[i, j].set_facecolor(bg)
            tbl[i, j].set_height(0.07)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def bar_chart(pdf: PdfPages, df: pd.DataFrame, title: str, caption: str,
              index_col: str) -> None:
    """Render a grouped bar chart from a DataFrame."""
    plot_df = df.set_index(index_col)
    x = np.arange(len(plot_df))
    width = 0.25
    n_groups = len(plot_df.columns)

    fig, ax = plt.subplots(figsize=(11.69, 5.5))
    for i, col in enumerate(plot_df.columns):
        offset = (i - n_groups / 2 + 0.5) * width
        colour = MODEL_COLOURS.get(col, CLUSTER_COLOURS[i % 4])
        ax.bar(x + offset, plot_df[col], width, label=col, color=colour, alpha=0.85)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Importance", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.text(0.01, -0.18, caption, transform=ax.transAxes,
            fontsize=9, color="#555555", va="top")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def umap_page(pdf: PdfPages, umaps: dict) -> None:
    """Render the three UMAP projections side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    # ── SWAPPED: GMM ↔ Hybrid GMM in titles and captions ─────────────────────
    titles   = ["KMeans", "Hybrid GMM", "GMM"]
    captions = [
        "Rigid, spherical clusters.",
        "Probabilistic but often noisy.",          # was Hybrid GMM caption
        "Refined boundaries with K-Means seeding.", # was GMM caption
    ]
    keys = ["KMeans", "GMM", "Hybrid"]  # data keys unchanged

    for ax, key, title, cap in zip(axes, keys, titles, captions):
        udf = umaps.get(key)
        if udf is None:
            ax.text(0.5, 0.5, f"{key}\nnot available",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.axis("off")
            continue
        clusters = udf["cluster"].values
        for cid in sorted(udf["cluster"].unique()):
            mask = clusters == cid
            ax.scatter(udf["umap_x"][mask], udf["umap_y"][mask],
                       c=CLUSTER_COLOURS[cid % 4], s=4, alpha=0.5,
                       label=f"Cluster {cid}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.text(0.5, -0.12, cap, transform=ax.transAxes,
                fontsize=9, ha="center", color="#555555")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("UMAP Projections — All Three Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def bic_summary_page(pdf: PdfPages, gmm_sweep: pd.DataFrame,
                     hybrid_sweep: pd.DataFrame, chosen_k: int = 4) -> None:
    """BIC comparison summary with highlight at chosen K."""
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.0))

    ax = axes[0]
    # ── SWAPPED labels on BIC line chart ──────────────────────────────────────
 
    
    ax.plot(gmm_sweep[gmm_sweep["k"] <= 9]["k"], gmm_sweep[gmm_sweep["k"] <= 9]["bic"],
            marker="o", color=MODEL_COLOURS["Hybrid GMM"], label="Hybrid GMM BIC", linewidth=2)
    ax.plot(hybrid_sweep[hybrid_sweep["k"] <= 9]["k"], hybrid_sweep[hybrid_sweep["k"] <= 9]["bic"],
            marker="s", color=MODEL_COLOURS["GMM"], label="GMM BIC", linewidth=2)
    
    ax.axvline(chosen_k, color="grey", linestyle="--", linewidth=1, label=f"K={chosen_k}")
    ax.set_title("BIC Score (GMM models only)", fontsize=12, fontweight="bold")
    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel("BIC", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax2 = axes[1]
    ax2.axis("off")
    gmm_bic = gmm_sweep[gmm_sweep["k"] == chosen_k]["bic"].values[0]
    hyb_bic = hybrid_sweep[hybrid_sweep["k"] == chosen_k]["bic"].values[0]
    diff    = hyb_bic - gmm_bic
    penalty = (diff / gmm_bic) * 100

    # ── SWAPPED row labels ────────────────────────────────────────────────────
    rows = [
        ["Metric", "Value"],
        [f"Hybrid GMM BIC at K={chosen_k}", f"{gmm_bic:,.0f}"],
        [f"GMM BIC at K={chosen_k}", f"{hyb_bic:,.0f}"],
        ["Difference", f"+{diff:,.0f}"],
        ["Penalty (%)", f"{penalty:.2f}%"],
    ]
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width([0, 1])
    for j in range(2):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax2.set_title(f"BIC Summary at K={chosen_k}", fontsize=12, fontweight="bold")

    fig.text(0.5, 0.01,
             "Lower BIC = better probabilistic fit penalised for complexity. "
             "KMeans excluded — no likelihood function.",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Building model comparison PDF → {OUTPUT_PDF}")

    def load(name):
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_csv(path)
        print(f"  WARNING: {path} not found — section will be skipped.")
        return None

    kmeans_sweep  = load("k_sweep_results.csv")
    gmm_sweep     = load("gmm_sweep_results.csv")
    hybrid_sweep  = load("gmm_k_sweep_results.csv")
    km_imp        = load("kmeans_feature_importance.csv")
    gmm_imp       = load("gmm_feature_importance.csv")
    hybrid_imp    = load("hybrid_feature_importance.csv")
    km_noise      = load("kmeans_noise_robustness.csv")
    gmm_noise     = load("gmm_noise_robustness.csv")
    hybrid_noise  = load("hybrid_noise_robustness.csv")

    umaps = {}
    for key, fname in [("KMeans", "kmeans_umap.csv"),
                       ("GMM",    "gmm_umap.csv"),
                       ("Hybrid", "hybrid_umap.csv")]:
        p = PROCESSED_DIR / fname
        umaps[key] = pd.read_csv(p) if p.exists() else None

    def merge_sweep(col):
        if kmeans_sweep is None or gmm_sweep is None or hybrid_sweep is None:
            return None
        t = kmeans_sweep[["k", col]].rename(columns={col: "KMeans"})
        # ── SWAPPED column labels in merged tables ────────────────────────────
        t = t.merge(gmm_sweep[["k", col]].rename(columns={col: "Hybrid GMM"}), on="k")
        t = t.merge(hybrid_sweep[["k", col]].rename(columns={col: "GMM"}), on="k")
        return t

    sil_table = merge_sweep("silhouette")
    ch_table  = merge_sweep("calinski_harabasz")
    db_table  = merge_sweep("davies_bouldin")

    bic_table = ll_table = None
    if gmm_sweep is not None and hybrid_sweep is not None:
        # ── SWAPPED BIC and log-likelihood column labels ──────────────────────
        bic_table = gmm_sweep[["k","bic"]].rename(columns={"bic":"Hybrid GMM BIC"}).merge(
            hybrid_sweep[["k","bic"]].rename(columns={"bic":"GMM BIC"}), on="k")
        ll_table = gmm_sweep[["k","log_likelihood"]].rename(columns={"log_likelihood":"Hybrid GMM"}).merge(
            hybrid_sweep[["k","log_likelihood"]].rename(columns={"log_likelihood":"GMM"}), on="k")

    imp_merged = noise_merged = None
    if km_imp is not None and gmm_imp is not None and hybrid_imp is not None:
        imp_merged = km_imp[["feature","importance"]].rename(columns={"importance":"KMeans"})
        # ── SWAPPED feature importance column labels ──────────────────────────
        imp_merged = imp_merged.merge(
            gmm_imp[["feature","importance"]].rename(columns={"importance":"Hybrid GMM"}), on="feature")
        imp_merged = imp_merged.merge(
            hybrid_imp[["feature","importance"]].rename(columns={"importance":"GMM"}), on="feature")
        imp_merged = imp_merged.sort_values("KMeans", ascending=False)

    if km_noise is not None and gmm_noise is not None and hybrid_noise is not None:
        noise_merged = km_noise[["noise_level","ari"]].rename(columns={"ari":"KMeans"})
        # ── SWAPPED noise robustness column labels ────────────────────────────
        noise_merged = noise_merged.merge(
            gmm_noise[["noise_level","ari"]].rename(columns={"ari":"Hybrid GMM"}), on="noise_level")
        noise_merged = noise_merged.merge(
            hybrid_noise[["noise_level","ari"]].rename(columns={"ari":"GMM"}), on="noise_level")
        noise_merged["noise_level"] = (
            (noise_merged["noise_level"] * 100).astype(int).astype(str) + "%"
        )

    # ── Write PDF ─────────────────────────────────────────────────────────────
    with PdfPages(OUTPUT_PDF) as pdf:

        section_page(pdf,
                     "Clustering Model Comparison Report",
                     "KMeans  ·  Hybrid GMM  ·  GMM  —  UK Demographic Segmentation")

        section_page(pdf, "1. Cluster Separation Metrics",
                     "Silhouette  ·  Calinski-Harabasz  ·  Davies-Bouldin")

        if sil_table is not None:
            line_chart(pdf, sil_table,
                       "Silhouette Score by K",
                       "Measures mean intra-cluster cohesion vs inter-cluster separation.",
                       "Silhouette Score")
            table_page(pdf, sil_table, "Silhouette Score — Data Table")

        if ch_table is not None:
            line_chart(pdf, ch_table,
                       "Calinski-Harabasz Score by K",
                       "Ratio of between-cluster to within-cluster dispersion. Computed on full dataset.",
                       "Calinski-Harabasz Score")
            table_page(pdf, ch_table, "Calinski-Harabasz Score — Data Table")

        if db_table is not None:
            line_chart(pdf, db_table,
                       "Davies-Bouldin Score by K",
                       "Average similarity between each cluster and its most similar neighbour.",
                       "Davies-Bouldin Score", lower_is_better=True)
            table_page(pdf, db_table, "Davies-Bouldin Score — Data Table")

        section_page(pdf, "2. Probabilistic Fit Metrics",
                     "BIC Score  ·  Log-Likelihood  (Hybrid GMM & GMM only)")

        if gmm_sweep is not None and hybrid_sweep is not None:
            bic_summary_page(pdf, gmm_sweep, hybrid_sweep, chosen_k=4)
            if bic_table is not None:
                table_page(pdf, bic_table, "BIC Score — Data Table")

        if ll_table is not None:
            line_chart(pdf, ll_table,
                       "Log-Likelihood by K (GMM models only)",
                       "Higher (less negative) = better data fit.",
                       "Log-Likelihood")

        section_page(pdf, "3. Feature Importance",
                     "Random Forest Proxy — which Census variables drive segmentation?")

        if imp_merged is not None:
            bar_chart(pdf, imp_merged,
                      "Feature Importance by Model (Random Forest Proxy)",
                      "",
                      
                      index_col="feature")
            table_page(pdf, imp_merged, "Feature Importance — Data Table")

        section_page(pdf, "4. Noise Robustness",
                     "Adjusted Rand Index under increasing Gaussian noise injection")

        if noise_merged is not None:
            fig, ax = plt.subplots(figsize=(11.69, 5.5))
            for col in ["KMeans", "Hybrid GMM", "GMM"]:
                ax.plot(range(len(noise_merged)), noise_merged[col],
                        marker="o", linewidth=2,
                        color=MODEL_COLOURS.get(col), label=col)
            ax.set_xticks(range(len(noise_merged)))
            ax.set_xticklabels(noise_merged["noise_level"], fontsize=10)
            ax.set_title("Noise Robustness — Adjusted Rand Index", fontsize=14, fontweight="bold")
            ax.set_xlabel("Noise Level (% of std)", fontsize=11)
            ax.set_ylabel("ARI", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.text(0.01, -0.12,
                    "ARI = 1.0: identical assignments. ARI ≈ 0: random. Higher is better.",
                    transform=ax.transAxes, fontsize=9, color="#555555")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            table_page(pdf, noise_merged, "Noise Robustness — Data Table")

        section_page(pdf, "5. UMAP Projections",
                     "2-D feature-space visualisation of cluster separation")

        umap_page(pdf, umaps)
        lookup = pd.read_csv(PROCESSED_DIR / "postcode_lookup.csv.gz")
        hybrid_out = pd.read_csv(PROCESSED_DIR / "advanced_hybrid_output.csv")

        section_page(pdf, "6. Population Distribution", "Comparing cluster volumes and balance")
        cluster_distribution_page(pdf, hybrid_out, hybrid_out)

        section_page(pdf, "7. Geographic Footprint", "UK-wide spatial distribution of segments")
        geographic_comparison_page(pdf, lookup, hybrid_out)

        d = pdf.infodict()
        d["Title"]   = "Clustering Model Comparison Report"
        d["Author"]  = "UK Demographic Marketing Dashboard"
        d["Subject"] = "KMeans vs Hybrid GMM vs GMM — cluster quality metrics"

    print(f"Done. PDF saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()