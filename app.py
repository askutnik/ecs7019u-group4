from __future__ import annotations
import pandas as pd
import streamlit as st
from src.recommendations import get_cluster_recommendation
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="UK Demographic Marketing Dashboard",
    layout="wide",
)

FEATURE_COLS = [
    "imd", "pct_econ_active", "pct_econ_inactive", "pct_retired",
    "pct_students", "pct_home_family", "pct_long_term_sick",
    "pct_unemployed", "pct_self_employed",
]

CLUSTER_COLOURS = {
    0: "#E63946",
    1: "#2A9D8F",
    2: "#E9C46A",
    3: "#4361EE",
}


@st.cache_data
def load_artifacts():
    usecols = ["pcds", "oa21", "lat", "long", "imd", "cluster"]
    dtype = {
        "pcds": "string", "oa21": "string", "cluster": "int16",
        "lat": "float32", "long": "float32", "imd": "float32",
    }
    lookup = pd.read_csv("processed/postcode_lookup.csv.gz", usecols=usecols, dtype=dtype)
    lookup["pcds"] = lookup["pcds"].astype(str).str.strip().str.upper()
    profiles_mean = pd.read_csv("processed/cluster_profiles_mean.csv")
    profiles_delta = pd.read_csv("processed/cluster_profiles_delta_vs_global.csv")
    sizes = pd.read_csv("processed/cluster_sizes.csv")
    for df in (profiles_mean, profiles_delta, sizes):
        df["cluster"] = df["cluster"].astype(int)
    umaps = {
        "KMeans": pd.read_csv("processed/kmeans_umap.csv"),
        "GMM": pd.read_csv("processed/gmm_umap.csv"),
        "Hybrid": pd.read_csv("processed/hybrid_umap.csv")
    }
    return lookup, profiles_mean, profiles_delta, sizes, umaps


#  Load hybrid output for confidence scores 
@st.cache_data
def load_hybrid_output():
    try:
        hybrid = pd.read_csv(
            "processed/advanced_hybrid_output.csv",
            usecols=["oa21", "confidence", "entropy", "is_ambiguous"],
        )
        return hybrid
    except FileNotFoundError:
        return None


def normalise_postcode(pc: str) -> str:
    return str(pc).strip().upper()


def top_deltas(delta_row: pd.Series, n: int = 3):
    deltas = delta_row.drop(labels=["cluster"])
    deltas = deltas.reindex(deltas.abs().sort_values(ascending=False).index)
    return deltas.head(n)


def cluster_label(cid: int) -> str:
    try:
        r = get_cluster_recommendation(cid)
        return f"Cluster {cid} — {r.get('cluster_name', '')}"
    except Exception:
        return f"Cluster {cid}"


#  Page header
st.title("UK Demographic Marketing Dashboard")
st.caption("Enter a UK postcode to view its demographic cluster, summary profile and transparent marketing recommendations.")

lookup, profiles_mean, profiles_delta, sizes, umaps = load_artifacts()

# Load hybrid confidence data
hybrid_output = load_hybrid_output()

CLUSTER_LABELS = {cid: cluster_label(cid) for cid in range(4)}

#  Sidebar (outside tabs so it's always visible)
with st.sidebar:
    st.header("Lookup")
    postcode = st.text_input("Postcode (e.g., E1 4PD)", value="")
    st.markdown("---")
    st.subheader("About")
    st.write(
        "Clusters are learned using K-means on public UK demographic indicators (Census + ONSPD). "
        "Recommendations are rule-based for interpretability."
    )

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗺️ Cluster Map", "📈 Model Comparison"])


# TAB 1 — Dashboard
with tab1:
    if not postcode:
        st.info("Enter a postcode in the sidebar to begin.")
        st.stop()

    pc = normalise_postcode(postcode)
    matches = lookup[lookup["pcds"] == pc]

    if matches.empty:
        st.error(
            f"Postcode **{pc}** not found in the lookup table. "
            "Check formatting (include a space if applicable, e.g. 'E1 4PD')."
        )
        st.stop()

    row = matches.iloc[0]
    cluster_id = int(row["cluster"])
    rec = get_cluster_recommendation(cluster_id)

    mean_row = profiles_mean[profiles_mean["cluster"] == cluster_id]
    delta_row = profiles_delta[profiles_delta["cluster"] == cluster_id]
    size_row = sizes[sizes["cluster"] == cluster_id]

    if mean_row.empty or delta_row.empty or size_row.empty:
        st.warning("Cluster profile tables are missing this cluster ID.")
        st.stop()

    mean_row = mean_row.iloc[0]
    delta_row = delta_row.iloc[0]
    oa_count = int(size_row.iloc[0]["oa_count"])

    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader(" Postcode result")
        st.markdown(
            f"**{pc}** falls within Output Area `{row['oa21']}`, "
            f"assigned to **Cluster {cluster_id} — {rec.get('cluster_name', '')}**. "
            f"The area has an IMD score of **{float(row['imd']):.1f}**."
        )
        st.caption(
            "The IMD measure is included as an area-level index; all other indicators "
            "are percentages derived from Census 2021 economic activity categories."
        )

        # Confidence Score display 
        # Pulls confidence and ambiguity from the hybrid model output.
        # Shows the business user how certain the model is about this postcode.
        if hybrid_output is not None:
            oa_code = str(row["oa21"])
            hybrid_row = hybrid_output[hybrid_output["oa21"] == oa_code]
            if not hybrid_row.empty:
                conf = float(hybrid_row.iloc[0]["confidence"])
                is_amb = bool(hybrid_row.iloc[0]["is_ambiguous"])
                ent = float(hybrid_row.iloc[0]["entropy"])

                conf_col1, conf_col2 = st.columns(2)
                conf_col1.metric(
                    label="Cluster Confidence",
                    value=f"{conf:.0%}",
                    help="Probability this postcode belongs to this cluster (from Hybrid GMM)."
                )
                conf_col2.metric(
                    label="Assignment Entropy",
                    value=f"{ent:.3f}",
                    help="Low entropy = clear assignment. High entropy = sits between clusters."
                )

                if is_amb:
                    st.warning(
                        "⚠️ **Boundary postcode:** This area sits near the edge of its cluster. "
                        "Marketing strategies from adjacent clusters may also apply. "
                        f"Model confidence: {conf:.0%}."
                    )
                else:
                    st.success(f" **Clear assignment** — confidence {conf:.0%}.")
        # ── [END ADDITION] ────────────────────────────────────────────────────

        st.markdown("---")

        st.subheader(" Marketing recommendation")
        st.markdown(f"#### {rec['cluster_name']}")
        st.markdown(f"**Who to target:** {rec['target_profile']}")

        st.markdown("**Recommended channels**")
        channels = rec["recommended_channels"]
        if isinstance(channels, list):
            for ch in channels:
                st.markdown(f"- {ch}")
        else:
            st.markdown(channels)

        st.markdown("**Messaging strategy**")
        strategy = rec["messaging_strategy"]
        if isinstance(strategy, list):
            for point in strategy:
                st.markdown(f"- {point}")
        else:
            paragraphs = [p.strip() for p in str(strategy).split("\n") if p.strip()]
            for para in paragraphs:
                st.markdown(para)

        with st.expander("💡 Rationale"):
            st.markdown(rec["rationale"])

    with colB:
        st.subheader("Cluster context")
        st.markdown(f"This cluster contains **{oa_count:,}** Output Areas across the UK.")
        st.markdown("---")

        st.markdown("**Top distinguishing features (vs UK average):**")
        top = top_deltas(delta_row, n=5)
        top_df = (
            top.rename("delta_vs_global")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        st.dataframe(top_df, use_container_width=True)

        st.subheader("Average profile (cluster means)")
        PCT_FEATURES = [
            "pct_econ_active", "pct_econ_inactive", "pct_retired",
            "pct_students", "pct_home_family", "pct_long_term_sick",
            "pct_unemployed", "pct_self_employed",
        ]
        profile_df = (
            mean_row[PCT_FEATURES]
            .rename("cluster_mean")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        st.bar_chart(profile_df.set_index("feature"))

        imd_percentile = (mean_row["imd"] / 32844) * 100
        st.metric("Deprivation percentile", f"{imd_percentile:.1f}%")
    st.markdown("**What defines this area:**")
    st.caption("The demographic factors that most strongly distinguish this cluster from the UK average.")

    # Logic fix: Use delta_row (specific to this cluster) instead of global importance
    # delta_row is already defined earlier in Tab 1 based on the postcode's cluster
    top_defining_features = top_deltas(delta_row, n=4) 

    for feature, delta_val in top_defining_features.items():
        # Show how much higher/lower this feature is than average
        label = feature.replace("pct_", "% ").replace("_", " ").title()
        # Normalize the delta for the progress bar (e.g., 0 to 1 scale)
        bar_val = min(abs(delta_val) / 50, 1.0) # Assuming max delta is around 50%
        st.progress(bar_val, text=f"{label}: {'+' if delta_val > 0 else ''}{delta_val:.1f}% vs UK Avg")
    
# TAB 2 — Cluster Map
with tab2:
    st.subheader("Geographic cluster visualisation")
    st.caption("All four clusters plotted simultaneously. Each colour represents a distinct demographic segment.")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

    with ctrl_col1:
        map_mode = st.radio(
            "Display mode",
            options=["Scatter", "Density"],
            horizontal=True,
            help="Scatter shows individual postcodes; Density shows concentration hotspots.",
        )

    with ctrl_col2:
        visible_clusters = st.multiselect(
            "Show clusters",
            options=list(range(4)),
            default=list(range(4)),
            format_func=lambda cid: CLUSTER_LABELS[cid],
        )

    with ctrl_col3:
        sample_size = st.slider(
            "Sample size per cluster",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Higher values are slower but more representative.",
        )

    if not visible_clusters:
        st.warning("Select at least one cluster to display.")
        st.stop()

    frames = []
    for cid in visible_clusters:
        subset = lookup[lookup["cluster"] == cid]
        sampled = subset.sample(n=min(sample_size, len(subset)), random_state=42).copy()
        sampled["cluster_label"] = CLUSTER_LABELS[cid]
        frames.append(sampled)

    map_df = pd.concat(frames, ignore_index=True)
    map_df["cluster_label"] = map_df["cluster_label"].astype(str)

    colour_map = {CLUSTER_LABELS[cid]: CLUSTER_COLOURS[cid] for cid in visible_clusters}

    if map_mode == "Scatter":
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="long",
            color="cluster_label",
            color_discrete_map=colour_map,
            hover_data={"lat": False, "long": False, "cluster_label": True, "imd": True},
            zoom=5,
            center={"lat": 53.5, "lon": -1.5},
            opacity=0.55,
            size_max=6,
        )
    else:
        fig = go.Figure()
        for cid in visible_clusters:
            subset = map_df[map_df["cluster"] == cid]
            fig.add_trace(go.Densitymapbox(
                lat=subset["lat"],
                lon=subset["long"],
                radius=8,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, CLUSTER_COLOURS[cid]]],
                showscale=False,
                name=CLUSTER_LABELS[cid],
                opacity=0.6,
            ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=5,
        mapbox_center={"lat": 53.5, "lon": -1.5},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        legend=dict(
            title="Demographic segment",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster sizes (Output Areas)**")
    size_cols = st.columns(4)
    for i, cid in enumerate(range(4)):
        size_val = sizes[sizes["cluster"] == cid]["oa_count"].values
        count_str = f"{int(size_val[0]):,}" if len(size_val) else "—"
        colour = CLUSTER_COLOURS[cid]
        size_cols[i].markdown(
            f"<div style='border-left:4px solid {colour}; padding-left:10px'>"
            f"<strong>{CLUSTER_LABELS[cid]}</strong><br/>{count_str} OAs"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # UMAP Visualisation 
    # Shows cluster structure in 2D feature space. Well-separated clouds
    # of points confirm the clusters are genuinely distinct in the data.
    # Inside Tab 1 code:
    st.subheader(" Geographic & Demographic Landscape")
    st.caption("This map visualizes how different postcodes relate to each other. Areas closer together have similar demographic profiles.")

    # Display ONLY the Hybrid UMAP
    hybrid_umap = umaps["Hybrid"]
    fig = px.scatter(
        hybrid_umap, x="umap_x", y="umap_y", color="cluster",
        color_discrete_map=CLUSTER_COLOURS,
        title="UK Demographic Clusters (Hybrid GMM Projection)"
    )
    st.plotly_chart(fig, use_container_width=True)


    # st.markdown("### UMAP Cluster Visualisation (Feature Space)")
    # st.caption(
    #     "UMAP projects 9 demographic features into 2D. "
    #     "Clearly separated point clouds confirm the clusters capture real structure "
    #     "in the data — not just artefacts of the algorithm. "
    #     "Select a model to inspect its cluster geometry."
    # )

    # umap_model = st.radio(
    #     "Select model",
    #     options=["KMeans", "GMM", "Hybrid GMM"],
    #     horizontal=True,
    # )

    # umap_file_map = {
    #     "KMeans": "processed/kmeans_umap.csv",
    #     "GMM": "processed/gmm_umap.csv",
    #     "Hybrid GMM": "processed/hybrid_umap.csv",
    # }

    # try:
    #     umap_df = pd.read_csv(umap_file_map[umap_model])
    #     umap_df["cluster"] = umap_df["cluster"].astype(str)

    #     colour_seq = [CLUSTER_COLOURS[i] for i in sorted(CLUSTER_COLOURS.keys())]

    #     umap_fig = px.scatter(
    #         umap_df,
    #         x="umap_x",
    #         y="umap_y",
    #         color="cluster",
    #         color_discrete_sequence=colour_seq,
    #         opacity=0.5,
    #         title=f"UMAP Projection — {umap_model}",
    #         labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
    #     )
    #     umap_fig.update_traces(marker=dict(size=3))
    #     umap_fig.update_layout(height=500, legend_title="Cluster")
    #     st.plotly_chart(umap_fig, use_container_width=True)

    #     #  Confidence overlay for Hybrid/GMM
    #     # If confidence column exists, offer a second view coloured by
    #     # confidence rather than cluster — shows boundary uncertainty visually.
    #     if "confidence" in umap_df.columns:
    #         show_conf = st.checkbox("Colour by confidence score instead of cluster")
    #         if show_conf:
    #             conf_fig = px.scatter(
    #                 umap_df,
    #                 x="umap_x",
    #                 y="umap_y",
    #                 color="confidence",
    #                 color_continuous_scale="RdYlGn",
    #                 opacity=0.5,
    #                 title=f"UMAP — Confidence Score Overlay ({umap_model})",
    #                 labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2",
    #                         "confidence": "Confidence"},
    #             )
    #             conf_fig.update_traces(marker=dict(size=3))
    #             conf_fig.update_layout(height=500)
    #             st.plotly_chart(conf_fig, use_container_width=True)
    #             st.caption(
    #                 "Green = high confidence (clearly in cluster). "
    #                 "Red = low confidence (sits on a boundary). "
    #                 "Red points in the interior of a cluster suggest sub-structure the model hasn't captured."
    #             )

    # except FileNotFoundError:
    #     st.info(f"Run training scripts to generate UMAP data. Install umap-learn if not already: `pip install umap-learn`")

# TAB 3 — Model Comparison

with tab3:
    st.subheader("Clustering Model Comparison")

    try:
        kmeans_sweep = pd.read_csv("processed/k_sweep_results.csv")
        gmm_sweep = pd.read_csv("processed/gmm_sweep_results.csv")
        hybrid_sweep = pd.read_csv("processed/gmm_k_sweep_results.csv")

        kmeans_sweep["model"] = "KMeans"
        gmm_sweep["model"] = "GMM"
        hybrid_sweep["model"] = "Hybrid GMM"

        # Silhouette Comparison
        st.markdown("### Model Performance Matrix: Silhouette Score")
        st.caption("Higher = better cluster separation. KMeans wins at K=4 (0.241).")

        sil_table = kmeans_sweep[["k", "silhouette"]].rename(columns={"silhouette": "KMeans"})
        sil_table = sil_table.merge(
            gmm_sweep[["k", "silhouette"]].rename(columns={"silhouette": "GMM"}), on="k"
        )
        sil_table = sil_table.merge(
            hybrid_sweep[["k", "silhouette"]].rename(columns={"silhouette": "Hybrid GMM"}), on="k"
        )
        st.line_chart(sil_table.set_index("k"))
        st.dataframe(sil_table, use_container_width=True)

        st.divider()

        # Calinski-Harabasz Comparison
        # Higher is better. Measures ratio of between-cluster to within-cluster
        # dispersion. Complements silhouette with a different geometric measure.
        st.markdown("### Calinski-Harabasz Score")
        st.caption("Higher = better defined clusters. Does not require a distance sample — computed on full dataset.")

        ch_table = kmeans_sweep[["k", "calinski_harabasz"]].rename(columns={"calinski_harabasz": "KMeans"})
        ch_table = ch_table.merge(
            gmm_sweep[["k", "calinski_harabasz"]].rename(columns={"calinski_harabasz": "GMM"}), on="k"
        )
        ch_table = ch_table.merge(
            hybrid_sweep[["k", "calinski_harabasz"]].rename(columns={"calinski_harabasz": "Hybrid GMM"}), on="k"
        )
        st.line_chart(ch_table.set_index("k"))
        st.dataframe(ch_table, use_container_width=True)

        st.divider()

        #  Davies-Bouldin Comparison
        # Lower is better. Measures average similarity between each cluster
        # and its most similar neighbour. Penalises overlapping clusters.
        st.markdown("### Davies-Bouldin Score")
        st.caption("Lower = better. Penalises clusters that overlap with their nearest neighbour.")

        db_table = kmeans_sweep[["k", "davies_bouldin"]].rename(columns={"davies_bouldin": "KMeans"})
        db_table = db_table.merge(
            gmm_sweep[["k", "davies_bouldin"]].rename(columns={"davies_bouldin": "GMM"}), on="k"
        )
        db_table = db_table.merge(
            hybrid_sweep[["k", "davies_bouldin"]].rename(columns={"davies_bouldin": "Hybrid GMM"}), on="k"
        )
        st.line_chart(db_table.set_index("k"))
        st.dataframe(db_table, use_container_width=True)

        st.divider()

        # BIC Comparison (GMM models only) 
        st.markdown("### BIC Score (GMM models only)")
        st.caption(
            "Lower = better probabilistic fit penalised for model complexity. "
            "KMeans excluded — it has no likelihood function. "
            "Hybrid BIC is marginally higher than pure GMM at K=4, reflecting "
            "the trade-off between cluster separability and probabilistic fit."
        )
        chosen_k = 4
        gmm_bic_at_k = gmm_sweep[gmm_sweep["k"] == chosen_k]["bic"].values[0]
        hybrid_bic_at_k = hybrid_sweep[hybrid_sweep["k"] == chosen_k]["bic"].values[0]
        diff = hybrid_bic_at_k - gmm_bic_at_k

        bic_col1, bic_col2, bic_col3 = st.columns(3)
        bic_col1.metric("GMM BIC at K=4", f"{gmm_bic_at_k:,.0f}")
        bic_col2.metric("Hybrid BIC at K=4", f"{hybrid_bic_at_k:,.0f}",
                        delta=f"+{diff:,.0f}", delta_color="inverse")
        bic_col3.metric("Penalty", f"{(diff/gmm_bic_at_k)*100:.2f}%",
                        help="Hybrid BIC overhead vs pure GMM — marginal cost for better separability.")

        bic_table = gmm_sweep[["k", "bic"]].rename(columns={"bic": "GMM BIC"})
        bic_table = bic_table.merge(
            hybrid_sweep[["k", "bic"]].rename(columns={"bic": "Hybrid BIC"}), on="k"
        )
        st.line_chart(bic_table.set_index("k"))
        st.dataframe(bic_table, use_container_width=True)

        st.divider()

        # ── Log-Likelihood Comparison ─────────────────────────────────────────
        st.markdown("### Log-Likelihood (GMM models only)")
        st.caption("Higher (less negative) = better data fit. Shows how well each model explains the observed data.")
        ll_table = gmm_sweep[["k", "log_likelihood"]].rename(columns={"log_likelihood": "GMM"})
        ll_table = ll_table.merge(
            hybrid_sweep[["k", "log_likelihood"]].rename(columns={"log_likelihood": "Hybrid GMM"}), on="k"
        )
        st.line_chart(ll_table.set_index("k"))

    except FileNotFoundError as e:
        st.info(f"Run training scripts to generate model comparison data. Missing: {e.filename}")

    st.divider()

    # ── [ADDITION] Feature Importance Comparison ──────────────────────────────
    # Loads the Random Forest proxy importance CSVs from all three models.
    # Shows which demographic features drive each model's segmentation.
    st.markdown("### Feature Importance (Random Forest Proxy)")
    st.caption(
        "A Random Forest was trained to predict each model's cluster labels. "
        "Its feature importances reveal which Census variables drive segmentation. "
        "Consistent rankings across models = robust finding."
    )
    try:
        km_imp = pd.read_csv("processed/kmeans_feature_importance.csv")
        gmm_imp = pd.read_csv("processed/gmm_feature_importance.csv")
        hybrid_imp = pd.read_csv("processed/hybrid_feature_importance.csv")

        imp_merged = km_imp[["feature", "importance"]].rename(columns={"importance": "KMeans"})
        imp_merged = imp_merged.merge(
            gmm_imp[["feature", "importance"]].rename(columns={"importance": "GMM"}), on="feature"
        )
        imp_merged = imp_merged.merge(
            hybrid_imp[["feature", "importance"]].rename(columns={"importance": "Hybrid GMM"}), on="feature"
        )
        imp_merged = imp_merged.sort_values("KMeans", ascending=False)

        st.bar_chart(imp_merged.set_index("feature"))
        st.dataframe(imp_merged, use_container_width=True)

    except FileNotFoundError:
        st.info("Run all three training scripts to generate feature importance data.")

    st.divider()

    #  Noise Robustness Comparison
    # Shows ARI at each noise level for all three models side by side.
    # A robust model maintains ARI > 0.8 even at 10% noise.
    st.markdown("### Noise Robustness (Adjusted Rand Index)")
    st.caption(
        "Gaussian noise was injected at increasing levels to simulate census measurement error. "
        "ARI = 1.0 means cluster assignments are identical to the noise-free model. "
        "ARI near 0 means assignments become random. Higher is better."
    )
    try:
        km_noise = pd.read_csv("processed/kmeans_noise_robustness.csv")
        gmm_noise = pd.read_csv("processed/gmm_noise_robustness.csv")
        hybrid_noise = pd.read_csv("processed/hybrid_noise_robustness.csv")

        noise_merged = km_noise[["noise_level", "ari"]].rename(columns={"ari": "KMeans"})
        noise_merged = noise_merged.merge(
            gmm_noise[["noise_level", "ari"]].rename(columns={"ari": "GMM"}), on="noise_level"
        )
        noise_merged = noise_merged.merge(
            hybrid_noise[["noise_level", "ari"]].rename(columns={"ari": "Hybrid GMM"}), on="noise_level"
        )
        noise_merged["noise_level"] = (noise_merged["noise_level"] * 100).astype(int).astype(str) + "%"

        st.line_chart(noise_merged.set_index("noise_level"))
        st.dataframe(noise_merged, use_container_width=True)

    except FileNotFoundError:
        st.info("Run all three training scripts to generate noise robustness data.")

    st.divider()
    

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**K-Means (Baseline)**")
        st.plotly_chart(px.scatter(umaps["KMeans"], x="umap_x", y="umap_y", color="cluster"), use_container_width=True)
        st.caption("Rigid, spherical clusters.")

    with col2:
        st.markdown("**Standalone GMM**")
        st.plotly_chart(px.scatter(umaps["GMM"], x="umap_x", y="umap_y", color="cluster"), use_container_width=True)
        st.caption("Probabilistic but often noisy.")

    with col3:
        st.markdown("**Hybrid Ensemble (Proposed)**")
        st.plotly_chart(px.scatter(umaps["Hybrid"], x="umap_x", y="umap_y", color="cluster"), use_container_width=True)
        st.caption("Refined boundaries with K-Means seeding.")
        